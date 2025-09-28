# models/tsmixer_model.py
import torch
import torch.nn as nn
from typing import Literal
from models.base_model import BaseRULModel


# ---------------- Core Mixer ----------------
class TimeMixing(nn.Module):
    """沿时间维度的 MLP（对每个通道共享），含 LayerNorm + 残差 + Dropout"""
    def __init__(self, seq_len: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = seq_len * expansion
        self.norm = nn.LayerNorm(seq_len)
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, seq_len),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        x_t = x.transpose(1, 2).contiguous()     # (B, C, L)
        y = self.mlp(self.norm(x_t))
        y = x_t + y
        return y.transpose(1, 2).contiguous()    # (B, L, C)


class FeatureMixing(nn.Module):
    """沿特征维度的 MLP（对每个时间步共享），含 LayerNorm + 残差 + Dropout"""
    def __init__(self, num_features: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = num_features * expansion
        self.norm = nn.LayerNorm(num_features)
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.norm(x))        # (B, L, C)


class MixerBlock(nn.Module):
    """TSMixer Block = TimeMixing -> FeatureMixing"""
    def __init__(self, seq_len: int, num_features: int,
                 time_expansion: int = 4, feat_expansion: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.time = TimeMixing(seq_len, time_expansion, dropout)
        self.feat = FeatureMixing(num_features, feat_expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feat(self.time(x))


class TSMixerRegressor(nn.Module):
    """
    输入:  (B, L, C)
    输出:  (B,)   —— 先过多层 MixerBlock，再做全局时间平均 + 线性回归到 1 维
    """
    def __init__(self,
                 input_length: int,
                 num_features: int,
                 num_layers: int = 4,
                 time_expansion: int = 4,
                 feat_expansion: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            MixerBlock(input_length, num_features,
                       time_expansion=time_expansion,
                       feat_expansion=feat_expansion,
                       dropout=dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(num_features, 1)

        # 轻微初始化：更稳定
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)                 # (B, L, C)
        h = x.mean(dim=1)              # 时间池化 → (B, C)
        y = self.head(h).squeeze(-1)   # (B,)
        return y


# ---------------- 子类：继承 BaseRULModel ----------------
class TSMixerModel(BaseRULModel):
    """
    仅实现：
      - build_model: 返回具体 nn.Module
      - compile: 配置优化器与调度器
    其余训练/验证逻辑复用 BaseRULModel 的默认实现
    """
    def __init__(self,
                 input_size: int,
                 seq_len: int,
                 num_layers: int = 4,
                 time_expansion: int = 4,
                 feat_expansion: int = 4,
                 dropout: float = 0.1):
        super().__init__(input_size=input_size, seq_len=seq_len, out_channels=1)
        self.num_layers = num_layers
        self.time_expansion = time_expansion
        self.feat_expansion = feat_expansion
        self.dropout = dropout

        # 构建真实 nn.Module 并放到设备
        self.model = self.build_model().to(self.device)

    # --- 必须实现 ---
    def build_model(self) -> nn.Module:
        return TSMixerRegressor(
            input_length=self.seq_len,
            num_features=self.input_size,
            num_layers=self.num_layers,
            time_expansion=self.time_expansion,
            feat_expansion=self.feat_expansion,
            dropout=self.dropout,
        )

    # --- 必须实现 ---
    def compile(self,
                learning_rate: float = 1e-3,
                weight_decay: float = 1e-4,
                *,
                scheduler: Literal["onecycle", "plateau", "cosine", "none"] = "plateau",
                epochs: int = 30,
                steps_per_epoch: int = 100):
        """
        默认优化器: AdamW
        默认调度器: ReduceLROnPlateau（更适合 RMSE/Score 驱动的收敛）
        可选: onecycle / cosine / none
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=learning_rate,
                                           weight_decay=weight_decay)

        if scheduler == "onecycle":
            from torch.optim.lr_scheduler import OneCycleLR
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=learning_rate,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.2,
                div_factor=10,
                final_div_factor=10,
            )
        elif scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        elif scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3
            )
        else:
            self.scheduler = None
