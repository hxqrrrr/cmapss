# models/parallel_tsmixer_rul.py
# 完整可用：包含 ParallelTSMixerNet、依赖分支与包装类 ParallelTSMixerRUL

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseRULModel



# -------------------------
# Utilities
# -------------------------
class StochasticDepth(nn.Module):
    """DropPath / Stochastic Depth per sample."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class CausalDWConv1d(nn.Module):
    """Depthwise 1D conv with causal padding. Input [B, C, T] -> [B, C, T]"""
    def __init__(self, channels: int, kernel_size: int = 7, dilation: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation,
            groups=channels, bias=True
        )

    def forward(self, x):  # [B, C, T]
        pad_left = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad_left, 0))  # causal
        return self.conv(x)


# -------------------------
# Branches (Time-only / Feature-only)
# -------------------------
class TemporalBranch(nn.Module):
    """Time-only mixing: LayerNorm(C) -> DW causal conv (T) -> pointwise GLU FFN.
       Expects [B, T, C] and returns [B, T, C].
    """
    def __init__(self, C: int, kernel_size: int = 7, dilation: int = 2, ffn_expand: int = 1):
        super().__init__()
        self.norm = nn.LayerNorm(C)
        self.dwconv = CausalDWConv1d(C, kernel_size=kernel_size, dilation=dilation)
        self.pw_in = nn.Conv1d(C, 2 * C * ffn_expand, kernel_size=1)
        self.pw_out = nn.Conv1d(C * ffn_expand, C, kernel_size=1)

    def forward(self, x):  # [B, T, C]
        x_n = self.norm(x)
        t = x_n.transpose(1, 2)              # [B, C, T]
        t = self.dwconv(t)                    # [B, C, T]
        t = self.pw_in(t)                     # [B, 2*C*e, T]
        t = F.glu(t, dim=1)                   # [B, C*e, T]
        t = self.pw_out(t)                    # [B, C, T]
        return t.transpose(1, 2)              # [B, T, C]


class ChannelBranch(nn.Module):
    """Feature-only mixing: per-time-step MLP over channels. [B, T, C] -> [B, T, C]"""
    def __init__(self, C: int, expand_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = max(4, expand_ratio * C)
        self.norm = nn.LayerNorm(C)
        self.fc1 = nn.Linear(C, hidden)
        self.fc2 = nn.Linear(hidden, C)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # [B, T, C]
        x_n = self.norm(x)
        c = self.fc2(F.gelu(self.fc1(x_n)))
        c = self.dropout(c)
        return c


class ParallelMixerBlock(nn.Module):
    """并行双支 + 门控残差融合 + 轻量 cross modulation（FiLM 风格）"""
    def __init__(
        self,
        T: int,
        C: int,
        ch_expand: int = 4,
        t_kernel: int = 7,
        t_dilation: int = 2,
        t_ffn_expand: int = 1,
        gate: bool = True,
        droppath: float = 0.0,
        branch_dropout: float = 0.0,
        enable_cross: bool = True,
    ):
        super().__init__()
        self.t_branch = TemporalBranch(C, kernel_size=t_kernel, dilation=t_dilation, ffn_expand=t_ffn_expand)
        self.c_branch = ChannelBranch(C, expand_ratio=ch_expand, dropout=branch_dropout)

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.use_gate = gate
        if gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * C, C), nn.GELU(), nn.Linear(C, 1)
            )
        self.drop_path = StochasticDepth(droppath)

        self.enable_cross = enable_cross
        if enable_cross:
            self.film_from_c = nn.Linear(C, 2 * C)  # modulate T-branch
            self.film_from_t = nn.Linear(C, 2 * C)  # modulate C-branch

    def forward(self, x):  # [B, T, C]
        y_t = self.t_branch(x)                 # [B, T, C]
        y_c = self.c_branch(x)                 # [B, T, C]

        if self.enable_cross:
            gt = y_t.mean(dim=1)               # [B, C]
            gc = y_c.mean(dim=1)               # [B, C]
            gamma_t, beta_t = self.film_from_c(gc).chunk(2, dim=-1)
            gamma_c, beta_c = self.film_from_t(gt).chunk(2, dim=-1)
            y_t = y_t * (1 + gamma_t.unsqueeze(1)) + beta_t.unsqueeze(1)
            y_c = y_c * (1 + gamma_c.unsqueeze(1)) + beta_c.unsqueeze(1)

        if self.use_gate:
            h = torch.cat([y_t.mean(dim=1), y_c.mean(dim=1)], dim=-1)  # [B, 2C]
            g = torch.sigmoid(self.gate_mlp(h)).view(-1, 1, 1)         # [B,1,1]
            res = g * self.alpha * y_t + (1.0 - g) * self.beta * y_c
        else:
            res = self.alpha * y_t + self.beta * y_c

        res = self.drop_path(res)
        return x + res


# -------------------------
# Heads
# -------------------------
class TokenPoolHead(nn.Module):
    """Weighted token pooling over time + linear regressor."""
    def __init__(self, C: int):
        super().__init__()
        self.score = nn.Linear(C, 1)
        self.norm = nn.LayerNorm(C)
        self.out = nn.Linear(C, 1)

    def forward(self, x):  # [B, T, C]
        s = self.score(x).squeeze(-1)               # [B, T]
        w = torch.softmax(s, dim=-1).unsqueeze(-1)  # [B, T, 1]
        pooled = (x * w).sum(dim=1)                 # [B, C]
        return self.out(self.norm(pooled)).squeeze(-1)


class AvgPoolHead(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.norm = nn.LayerNorm(C)
        self.mlp = nn.Sequential(
            nn.Linear(C, 2 * C), nn.GELU(), nn.Linear(2 * C, 1)
        )

    def forward(self, x):  # [B, T, C]
        pooled = x.mean(dim=1)
        return self.mlp(self.norm(pooled)).squeeze(-1)


# -------------------------
# Main Net
# -------------------------
class ParallelTSMixerNet(nn.Module):
    def __init__(
        self,
        input_size: int,   # C
        seq_len: int,      # T
        depth: int = 6,
        ch_expand: int = 4,
        t_kernel: int = 7,
        t_dilation: int = 2,
        t_ffn_expand: int = 1,
        droppath_base: float = 0.1,
        branch_dropout: float = 0.0,
        pooling: str = "token",  # "token" | "avg"
        input_dropout: float = 0.0,
        enable_cross: bool = True,
    ):
        super().__init__()
        C, T = input_size, seq_len
        self.in_drop = nn.Dropout(input_dropout)

        # depth-wise droppath schedule
        if depth > 1:
            dp_rates = torch.linspace(0, droppath_base, steps=depth).tolist()
        else:
            dp_rates = [0.0]

        blocks = []
        for i in range(depth):
            blocks.append(
                ParallelMixerBlock(
                    T=T,
                    C=C,
                    ch_expand=ch_expand,
                    t_kernel=t_kernel,
                    t_dilation=t_dilation,
                    t_ffn_expand=t_ffn_expand,
                    gate=True,
                    droppath=dp_rates[i],
                    branch_dropout=branch_dropout,
                    enable_cross=enable_cross,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(C)

        if pooling == "token":
            self.head = TokenPoolHead(C)
        elif pooling == "avg":
            self.head = AvgPoolHead(C)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    def forward(self, x):  # x: [B, T, C]
        x = self.in_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        y = self.head(x)  # [B]
        return y


# -------------------------
# Wrapper for BaseRULModel
# -------------------------
class ParallelTSMixerRUL(BaseRULModel):
    """
    RUL model wrapper for ParallelTSMixerNet, compatible with your BaseRULModel.
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        out_channels: int = 1,
        # 兼容多种命名
        depth: int = 6,
        ch_expand: int = None,          # 新脚本用的名字
        channel_expand: int = None,     # 兼容别名
        expand_ratio: int = 4,          # 老名字
        t_kernel: int = 7,
        t_dilation: int = 2,
        t_ffn_expand: int = 1,
        droppath_base: float = 0.1,
        branch_dropout: float = 0.0,
        pooling: str = "token",
        input_dropout: float = 0.0,
        enable_cross: bool = True,
    ):
        super().__init__(input_size, seq_len, out_channels)

        # 统一特征分支扩展倍数
        if ch_expand is not None:
            chx = ch_expand
        elif channel_expand is not None:
            chx = channel_expand
        else:
            chx = expand_ratio

        self.hparams = dict(
            depth=depth,
            ch_expand=chx,
            t_kernel=t_kernel,
            t_dilation=t_dilation,
            t_ffn_expand=t_ffn_expand,
            droppath_base=droppath_base,
            branch_dropout=branch_dropout,
            pooling=pooling,
            input_dropout=input_dropout,
            enable_cross=enable_cross,
        )

        self.model = self.build_model().to(self.device)

    def build_model(self) -> nn.Module:
        return ParallelTSMixerNet(
            input_size=self.input_size,
            seq_len=self.seq_len,
            depth=self.hparams["depth"],
            ch_expand=self.hparams["ch_expand"],
            t_kernel=self.hparams["t_kernel"],
            t_dilation=self.hparams["t_dilation"],
            t_ffn_expand=self.hparams["t_ffn_expand"],
            droppath_base=self.hparams["droppath_base"],
            branch_dropout=self.hparams["branch_dropout"],
            pooling=self.hparams["pooling"],
            input_dropout=self.hparams["input_dropout"],
            enable_cross=self.hparams["enable_cross"],
        )

    def compile(
        self,
        learning_rate: float = 8e-4,
        weight_decay: float = 2e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        scheduler: str = "cosine",
        warmup_steps: int = 0,
        total_steps: Optional[int] = None,
        **kwargs,
    ):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )
        sch = None
        if scheduler == "cosine":
            T_max = total_steps if total_steps is not None else 200
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler == "plateau":
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        elif scheduler == "none":
            sch = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")
        self.scheduler = sch


# -------------------------
# Minimal sanity test
# -------------------------
if __name__ == "__main__":
    B, T, C = 8, 120, 24
    x = torch.randn(B, T, C)
    net = ParallelTSMixerNet(input_size=C, seq_len=T, depth=4, ch_expand=4, pooling="token")
    with torch.no_grad():
        y = net(x)
    print("Net OK, output:", y.shape)
    mdl = ParallelTSMixerRUL(input_size=C, seq_len=T, depth=4, ch_expand=4)
    mdl.compile()
    print("Wrapper OK")
