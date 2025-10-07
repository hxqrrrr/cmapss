# tsmixer_cnn_gated.py
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from models.base_model import BaseRULModel

# ----------------- 小工具：nn.Lambda（PyTorch 无内置） -----------------
class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__(); self.fn = fn
    def forward(self, x):
        return self.fn(x)

# ----------------- 前端 CNN（GN + 小核 + 无膨胀，保持长度不变） -----------------
class CNNFrontEnd(nn.Module):
    """
    轻量前端特征抽取：
    - 使用 GroupNorm 替代 BatchNorm，提升小 batch / 多工况下稳定性
    - kernel=3, dilation=1，padding=1，确保 T 不变，避免过度平滑/边界伪影
    """
    def __init__(self, in_channels: int, out_channels: int = 64,
                 num_layers: int = 2, kernel_size: int = 3,
                 dropout: float = 0.1, num_groups: int = 8, use_groupnorm: bool = True):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 建议为奇数以保持长度"
        layers = []
        c_prev = in_channels
        for _ in range(num_layers):
            layers += [
                nn.Conv1d(c_prev, out_channels, kernel_size=kernel_size,
                          padding=kernel_size // 2, dilation=1, bias=False),
                (nn.GroupNorm(num_groups, out_channels) if use_groupnorm else nn.BatchNorm1d(out_channels)),
                nn.GELU(),
            ]
            c_prev = out_channels
        self.net = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, T, C]
        z = self.net(x.transpose(1, 2))   # [B, C_out, T]
        z = self.dropout(z)
        return z.transpose(1, 2)          # [B, T, C_out]

# ----------------- 可学习旁路：门控残差（自动“少用或绕过”CNN） -----------------
class GatedBypass(nn.Module):
    """
    y = sigmoid(alpha) * x_cnn + (1 - sigmoid(alpha)) * Proj(x_raw)
    - 若 CNN 有帮助：alpha ↑，更多使用 CNN
    - 若 CNN 无帮助：alpha ↓，退化为接近纯 TSMixer 的输入
    """
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Conv1d(d_in, d_out, kernel_size=1, bias=False) if d_in != d_out else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 从0.5起步更稳

    def forward(self, x_cnn, x_raw):  # [B, T, C_out], [B, T, C_in]
        x_raw_proj = self.proj(x_raw.transpose(1, 2)).transpose(1, 2)
        a = torch.sigmoid(self.alpha)
        return a * x_cnn + (1.0 - a) * x_raw_proj

    def gate_value(self) -> float:
        with torch.no_grad():
            return torch.sigmoid(self.alpha).item()

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
        x = x[:, :T_eff, :]
        z = self.proj(x.transpose(1, 2))  # [B, d_model, N_tokens]
        return z.transpose(1, 2)          # [B, N_tokens, d_model]

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
                 token_mlp_dim: int, channel_mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        # Token-Mixing: 针对 token 维（N_tokens）的 MLP
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Lambda(lambda x: x.transpose(1, 2)),   # [B, D, N]
            nn.LayerNorm(num_tokens),
            nn.Linear(num_tokens, token_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_mlp_dim, num_tokens),
            nn.Dropout(dropout),
            Lambda(lambda x: x.transpose(1, 2)),   # -> [B, N, D]
        )
        # Channel-Mixing: 针对嵌入维（D）的 MLP
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
            # 末端加权池化：更关注临近失效的 token
            weights = torch.linspace(0.1, 1.0, steps=x.size(1), device=x.device).unsqueeze(0).unsqueeze(-1)
            x = (x * weights).sum(dim=1) / weights.sum(dim=1)
        return self.fc(x)  # [B, 1]

# ----------------- 完整模型：CNN → 门控旁路 → Patchify → TSMixer → Head -----------------
class CNNMixerGatedModule(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, patch: int,
                 cnn_channels: int = 64, cnn_layers: int = 2, cnn_kernel: int = 3,
                 d_model: int = 128, depth: int = 6,
                 token_mlp_dim: int = 256, channel_mlp_dim: int = 128,
                 dropout: float = 0.1, pool: str = "mean",
                 gn_groups: int = 8, use_groupnorm: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.patch = patch

        # 前端：GN + 小核，无膨胀，保持时长
        self.cnn = CNNFrontEnd(
            in_channels, out_channels=cnn_channels,
            num_layers=cnn_layers, kernel_size=cnn_kernel,
            dropout=dropout, num_groups=gn_groups, use_groupnorm=use_groupnorm
        )
        # 门控旁路：自动调节 CNN 贡献
        self.gate = GatedBypass(d_in=in_channels, d_out=cnn_channels)

        # Patchify（长度已保持不变，可直接用）
        self.patchify = TimePatchify(cnn_channels, d_model, patch)
        num_tokens = (seq_len // patch)

        # TSMixer 主干 + 回归头
        self.mixer = TSMixerBackbone(num_tokens, d_model, depth,
                                     token_mlp_dim, channel_mlp_dim, dropout)
        self.head = RULHead(d_model, pool)

    def forward(self, x):  # x: [B, T, C]
        z_cnn = self.cnn(x)          # [B, T, C_cnn]
        z = self.gate(z_cnn, x)      # 门控融合：α*CNN(x) + (1-α)*Proj(x)
        z = self.patchify(z)         # [B, N_tokens, d_model]
        z = self.mixer(z)            # [B, N_tokens, d_model]
        y = self.head(z)             # [B, 1]
        return y.squeeze(-1)         # [B]

    def gate_alpha(self) -> float:
        return self.gate.gate_value()

# =======================  子类：对接你的 BaseRULModel  =======================
class CNNMixerGatedRULModel(BaseRULModel):
    """
    与 BaseRULModel 训练循环/接口完全兼容：
    - build_model() 返回 nn.Module
    - compile() 配优化器/调度器
    - 提供快捷配置：config_fd001 / config_fd004
    - 新增 get_gate_alpha() 便于日志观察 α
    """
    def __init__(self,
                 input_size: int,     # 传感器通道数（特征维 C）
                 seq_len: int,        # 窗口长度（时间步 T）
                 out_channels: int = 1,
                 # --- 结构超参 ---
                 patch: int = 5,
                 cnn_channels: int = 64, cnn_layers: int = 2, cnn_kernel: int = 3,
                 d_model: int = 128, depth: int = 6,
                 token_mlp_dim: int = 256, channel_mlp_dim: int = 128,
                 dropout: float = 0.1, pool: str = "mean",
                 gn_groups: int = 8, use_groupnorm: bool = True):
        super().__init__(input_size, seq_len, out_channels)
        self.hparams: Dict[str, Any] = dict(
            patch=patch,
            cnn_channels=cnn_channels, cnn_layers=cnn_layers, cnn_kernel=cnn_kernel,
            d_model=d_model, depth=depth,
            token_mlp_dim=token_mlp_dim, channel_mlp_dim=channel_mlp_dim,
            dropout=dropout, pool=pool,
            gn_groups=gn_groups, use_groupnorm=use_groupnorm
        )
        self.model = self.build_model().to(self.device)

    # ---- 构建 nn.Module ----
    def build_model(self) -> nn.Module:
        return CNNMixerGatedModule(
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
            gn_groups=self.hparams["gn_groups"],
            use_groupnorm=self.hparams["use_groupnorm"],
        )

    # ---- 优化器与调度器 ----
    def compile(self, learning_rate: float = 5e-4, weight_decay: float = 1e-4,
                scheduler: str = "cosine", T_max: int = 100, plateau_patience: int = 5,
                epochs: int = None, steps_per_epoch: int = None):
        """
        注意：默认 lr 设置为 5e-4（相较无门控/BN版减半），通常更稳。
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=plateau_patience, factor=0.5
            )
        elif scheduler == "onecycle" and epochs is not None and steps_per_epoch is not None:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=learning_rate, 
                epochs=epochs, steps_per_epoch=steps_per_epoch
            )
        else:
            self.scheduler = None

    # ---- 快捷配置 ----
    @classmethod
    def config_fd001(cls, input_size: int):
        """FD001/FD003 推荐起点：保持 num_tokens≈10"""
        return dict(
            input_size=input_size, seq_len=50, patch=5,
            cnn_channels=64, cnn_layers=2, cnn_kernel=3,
            d_model=128, depth=4, token_mlp_dim=256, channel_mlp_dim=128,
            dropout=0.10, pool="mean",
            gn_groups=8, use_groupnorm=True
        )

    @classmethod
    def config_fd004(cls, input_size: int):
        """FD002/FD004 推荐起点（多工况、序列更长）"""
        return dict(
            input_size=input_size, seq_len=80, patch=8,
            cnn_channels=64, cnn_layers=3, cnn_kernel=3,
            d_model=160, depth=6, token_mlp_dim=384, channel_mlp_dim=192,
            dropout=0.15, pool="weighted",
            gn_groups=8, use_groupnorm=True
        )

    # ---- 便捷：获取门控系数 α（记录到日志里） ----
    def get_gate_alpha(self) -> float:
        if hasattr(self.model, "gate_alpha"):
            return self.model.gate_alpha()
        return float("nan")
    
    # ---- 重写日志记录方法，添加门控系数显示 ----
    def log_training_metrics(self, epoch, epochs, train_rmse, val_rmse_global, 
                           val_rmse_last, val_score_last, lr):
        """详细的训练指标日志记录 - 门控CNN-TSMixer模型（包含门控系数α）"""
        import logging
        logger = logging.getLogger(__name__)
        
        # 格式化指标显示
        def format_metric(val):
            if hasattr(val, 'isnan') and val.isnan():
                return "N/A"
            elif val == float('nan') or str(val) == 'nan':
                return "N/A"
            return f"{val:.2f}"
        
        # 获取当前门控系数
        gate_alpha = self.get_gate_alpha()
        gate_str = f"α={gate_alpha:.3f}" if not (gate_alpha != gate_alpha or gate_alpha == float('nan')) else "α=N/A"
        
        # 详细的日志格式，添加门控系数
        metrics_msg = (f"[Epoch {epoch:3d}/{epochs}] "
                      f"train_rmse={format_metric(train_rmse)} | "
                      f"val_rmse(global)={format_metric(val_rmse_global)} cycles | "
                      f"val_rmse(last)={format_metric(val_rmse_last)} cycles | "
                      f"val_score(last)={format_metric(val_score_last)} | "
                      f"{gate_str} | "
                      f"lr={lr:.2e}")
        
        logger.info(metrics_msg)
