# tsmixer_cnn_model.py
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from math import ceil
from models.base_model import BaseRULModel

# ----------------- 小工具：nn.Lambda（PyTorch 无内置） -----------------
class Lambda(nn.Module):
    def __init__(self, fn): 
        super().__init__(); self.fn = fn
    def forward(self, x): 
        return self.fn(x)

# ----------------- 前端 CNN：提取局部形态/短期依赖 -----------------
class CNNFrontEnd(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64,
                 num_layers: int = 3, kernel_size: int = 5,
                 use_dilation: bool = True, dropout: float = 0.1):
        super().__init__()
        layers = []
        c_prev = in_channels
        dilation = 1
        for _ in range(num_layers):
            padding = (kernel_size // 2) * dilation
            layers += [
                nn.Conv1d(c_prev, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            ]
            if use_dilation:
                dilation *= 2  # 1,2,4
            c_prev = out_channels
        self.net = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, T, C]
        x = x.transpose(1, 2)         # -> [B, C, T]
        z = self.net(x)               # -> [B, C_out, T]
        z = self.dropout(z)
        return z.transpose(1, 2)      # -> [B, T, C_out]

# ----------------- Patchify：Conv1d 在时间维做分块+升维 -----------------
class TimePatchify(nn.Module):
    def __init__(self, in_dim: int, d_model: int, patch: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv1d(in_dim, d_model, kernel_size=patch, stride=patch, padding=0)

    def forward(self, x):  # [B, T, C]
        B, T, C = x.shape
        T_eff = (T // self.patch) * self.patch
        if T_eff == 0:
            raise ValueError(f"seq_len {T} shorter than patch {self.patch}")
        x = x[:, :T_eff, :]                 # 保证长度是 patch 的整数倍
        z = self.proj(x.transpose(1, 2))    # [B, d_model, N_tokens]
        return z.transpose(1, 2)            # [B, N_tokens, d_model]

# ----------------- 前馈层 -----------------
def FeedForward(dim_in, dim_hidden, dropout=0.0):
    return nn.Sequential(
        nn.Linear(dim_in, dim_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim_in),
        nn.Dropout(dropout),
    )

# ----------------- 单个 Mixer Block（Token- & Channel-Mixing） -----------------
class MixerBlock(nn.Module):
    def __init__(self, num_tokens: int, dim: int,
                 token_mlp_dim: int, channel_mlp_dim: int, dropout=0.0):
        super().__init__()
        # Token-Mixing: 针对 token 维度（N_tokens）的 MLP
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Lambda(lambda x: x.transpose(1, 2)),     # [B, D, N]
            nn.LayerNorm(num_tokens),
            nn.Linear(num_tokens, token_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_mlp_dim, num_tokens),
            nn.Dropout(dropout),
            Lambda(lambda x: x.transpose(1, 2)),     # -> [B, N, D]
        )
        # Channel-Mixing: 针对通道/嵌入维（D）的 MLP
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_mlp_dim, dropout=dropout)
        )

    def forward(self, x):  # [B, N_tokens, D]
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x

# ----------------- TSMixer 主干 -----------------
class TSMixerBackbone(nn.Module):
    def __init__(self, num_tokens: int, d_model: int, depth: int = 6,
                 token_mlp_dim: int = 256, channel_mlp_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.Sequential(*[
            MixerBlock(num_tokens, d_model, token_mlp_dim, channel_mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # [B, N_tokens, D]
        x = self.blocks(x)
        return self.norm(x)

# ----------------- 回归头：时间池化 + 线性 -----------------
class RULHead(nn.Module):
    def __init__(self, d_model: int, pool: str = "mean"):
        super().__init__()
        self.pool = pool
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):  # [B, N_tokens, D]
        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "last":
            x = x[:, -1, :]
        else:
            # 加权靠近末端（更关注“临近失效”的 token）
            weights = torch.linspace(0.1, 1.0, steps=x.size(1), device=x.device).unsqueeze(0).unsqueeze(-1)
            x = (x * weights).sum(dim=1) / weights.sum(dim=1)
        return self.fc(x)  # [B, 1]

# ----------------- 完整模型：CNN → Patchify → TSMixer → Head -----------------
class CNNMixerRULModule(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, patch: int,
                 cnn_channels: int = 64, cnn_layers: int = 3, cnn_kernel: int = 5,
                 d_model: int = 128, depth: int = 6,
                 token_mlp_dim: int = 256, channel_mlp_dim: int = 128,
                 dropout: float = 0.1, pool: str = "mean"):
        super().__init__()
        self.seq_len = seq_len
        self.patch = patch
        self.cnn = CNNFrontEnd(in_channels, cnn_channels, cnn_layers, cnn_kernel,
                               use_dilation=True, dropout=dropout)
        self.patchify = TimePatchify(cnn_channels, d_model, patch)
        
        # 动态计算CNN输出后的实际序列长度和token数量
        # 使用一个dummy输入来获取CNN输出的实际长度
        with torch.no_grad():
            dummy_input = torch.randn(1, seq_len, in_channels)
            cnn_output = self.cnn(dummy_input)
            actual_seq_len = cnn_output.shape[1]
            T_eff = (actual_seq_len // patch) * patch
            num_tokens = T_eff // patch
            
        self.mixer = TSMixerBackbone(num_tokens, d_model, depth,
                                     token_mlp_dim, channel_mlp_dim, dropout)
        self.head = RULHead(d_model, pool)

    def forward(self, x):  # x: [B, T, C]
        z = self.cnn(x)        # [B, T, C_cnn]
        z = self.patchify(z)   # [B, N_tokens, d_model]
        z = self.mixer(z)      # [B, N_tokens, d_model]
        y = self.head(z)       # [B, 1]
        return y.squeeze(-1)   # [B]

# =======================  子类：对接你的 BaseRULModel  =======================
class CNNMixerRULModel(BaseRULModel):  # 继承你给的 BaseRULModel
    def __init__(self,
                 input_size: int,     # 传感器通道数（特征维 C）
                 seq_len: int,        # 窗口长度（时间步 T）
                 out_channels: int = 1,
                 # --- 结构超参 ---
                 patch: int = 5,
                 cnn_channels: int = 64, cnn_layers: int = 3, cnn_kernel: int = 5,
                 d_model: int = 128, depth: int = 6,
                 token_mlp_dim: int = 256, channel_mlp_dim: int = 128,
                 dropout: float = 0.1, pool: str = "mean"):
        super().__init__(input_size, seq_len, out_channels)
        self.hparams: Dict[str, Any] = dict(
            patch=patch,
            cnn_channels=cnn_channels, cnn_layers=cnn_layers, cnn_kernel=cnn_kernel,
            d_model=d_model, depth=depth,
            token_mlp_dim=token_mlp_dim, channel_mlp_dim=channel_mlp_dim,
            dropout=dropout, pool=pool
        )
        self.model = self.build_model().to(self.device)

    # ---- 必须实现：构建 nn.Module ----
    def build_model(self) -> nn.Module:
        return CNNMixerRULModule(
            in_channels=self.input_size,
            seq_len=self.seq_len,
            patch=self.hparams["patch"],
            cnn_channels=self.hparams["cnn_channels"],
            cnn_layers=self.hparams["cnn_layers"],
            cnn_kernel=self.hparams["cnn_kernel"],
            d_model=self.hparams["d_model"],
            depth=self.hparams["depth"],
            token_mlp_dim=self.hparams["token_mlp_dim"],
            channel_mlp_dim=self.hparams["channel_mlp_dim"],
            dropout=self.hparams["dropout"],
            pool=self.hparams["pool"],
        )

    # ---- 必须实现：优化器与调度器 ----
    def compile(self, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                scheduler: str = "cosine", T_max: int = 100, plateau_patience: int = 5,
                epochs: int = None, steps_per_epoch: int = None):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min",
                                                                        patience=plateau_patience, factor=0.5)
        elif scheduler == "onecycle" and epochs is not None and steps_per_epoch is not None:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=learning_rate, 
                epochs=epochs, steps_per_epoch=steps_per_epoch
            )
        else:
            self.scheduler = None

    # --------- 便捷工厂：常用配置 ---------
    @classmethod
    def config_fd001(cls, input_size: int):
        """FD001/FD003 推荐起点"""
        return dict(
            input_size=input_size, seq_len=50, patch=5,
            cnn_channels=64, cnn_layers=2, cnn_kernel=5,
            d_model=128, depth=4, token_mlp_dim=256, channel_mlp_dim=128,
            dropout=0.10, pool="mean"
        )

    @classmethod
    def config_fd004(cls, input_size: int):
        """FD002/FD004 推荐起点（多工况、序列更长）"""
        return dict(
            input_size=input_size, seq_len=80, patch=8,
            cnn_channels=64, cnn_layers=3, cnn_kernel=5,
            d_model=160, depth=6, token_mlp_dim=384, channel_mlp_dim=192,
            dropout=0.15, pool="weighted"  # 更关注后期 token
        )
