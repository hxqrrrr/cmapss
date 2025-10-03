# models/tsmixer_sga.py
import torch
import torch.nn as nn
from typing import Literal
from models.base_model import BaseRULModel

# ---------------- Core Mixer ----------------
class TimeMixing(nn.Module):
    """沿时间维度的 MLP（对每个通道共享），含 LayerNorm + 残差 + Dropout"""
    def __init__(self, seq_len: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = max(4, int(seq_len * expansion))
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
        hidden = max(4, int(num_features * expansion))
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

# ---------------- SGA: 双维可扩展全局注意力 ----------------
class SGA2D(nn.Module):
    """
    双向 squeeze + 自适应融合 + 残差门控
      输入 X: [B, L, C]
      输出 Y: [B, L, C]，其中 Y = X + X ⊙ σ(γ_t·A_time + γ_f·A_feat + β)
    """
    def __init__(self, seq_len: int, num_features: int,
                 time_reduce_ratio: int = 4,
                 feat_reduce_ratio: int = 4,
                 dropout: float = 0.05):
        super().__init__()
        # 时间方向的门：对 [B,L] 做一个极小 MLP
        t_hidden = max(4, seq_len // time_reduce_ratio)
        self.t_norm = nn.LayerNorm(seq_len)
        self.t_mlp = nn.Sequential(
            nn.Linear(seq_len, t_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(t_hidden, seq_len)
        )

        # 特征方向的门：对 [B,C] 做极小 MLP
        f_hidden = max(4, num_features // feat_reduce_ratio)
        self.f_norm = nn.LayerNorm(num_features)
        self.f_mlp = nn.Sequential(
            nn.Linear(num_features, f_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(f_hidden, num_features)
        )

        # 自适应融合参数
        self.gamma_t = nn.Parameter(torch.tensor(1.0))
        self.gamma_f = nn.Parameter(torch.tensor(1.0))
        self.beta    = nn.Parameter(torch.tensor(0.0))

        # 初始化更稳一些
        for m in [self.t_mlp[-1], self.f_mlp[-1]]:
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    @torch.no_grad()
    def attn_stats(self, A: torch.Tensor):
        """
        简单统计：返回 (峰值均值, 覆盖率近似, 末端权重均值)，帮助你在日志里排查注意力是否塌缩
        A: [B, L, C] 的 sigmoid 后权重
        """
        B, L, C = A.shape
        peak = A.amax(dim=(1,2)).mean().item()
        cover = (A > 0.01).float().mean().item()
        end_w = A[:, -1, :].mean().item()
        return peak, cover, end_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X: [B, L, C]
        B, L, C = x.shape

        # 时间 squeeze：沿特征均值 -> [B, L]
        t = x.mean(dim=2)
        a_t = self.t_mlp(self.t_norm(t)).unsqueeze(-1)   # [B, L, 1]

        # 特征 squeeze：沿时间均值 -> [B, C]
        f = x.mean(dim=1)
        a_f = self.f_mlp(self.f_norm(f)).unsqueeze(1)    # [B, 1, C]

        # 融合并 Sigmoid 成门
        A = torch.sigmoid(self.gamma_t * a_t + self.gamma_f * a_f + self.beta)  # [B, L, C]

        # 残差门控：更稳
        y = x + x * A
        return y, A

# ---------------- 带 SGA 的回归器 ----------------
class TSMixerRegressor(nn.Module):
    """
    输入:  (B, L, C)
    输出:  (B,)
    结构:  多层 MixerBlock -> (可选) SGA2D -> 时间池化 -> 线性回归
    """
    def __init__(self,
                 input_length: int,
                 num_features: int,
                 num_layers: int = 4,
                 time_expansion: int = 4,
                 feat_expansion: int = 4,
                 dropout: float = 0.1,
                 *,
                 use_sga: bool = False,
                 sga_time_rr: int = 4,
                 sga_feat_rr: int = 4,
                 sga_dropout: float = 0.05,
                 pool: Literal["mean", "last", "weighted"] = "mean"):
        super().__init__()
        self.pool = pool
        self.use_sga = use_sga

        self.blocks = nn.ModuleList([
            MixerBlock(input_length, num_features,
                       time_expansion=time_expansion,
                       feat_expansion=feat_expansion,
                       dropout=dropout)
            for _ in range(num_layers)
        ])

        if use_sga:
            self.sga = SGA2D(input_length, num_features,
                             time_reduce_ratio=sga_time_rr,
                             feat_reduce_ratio=sga_feat_rr,
                             dropout=sga_dropout)
        else:
            self.sga = None

        self.head = nn.Linear(num_features, 1)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)                 # (B, L, C)

        if self.sga is not None:
            x, _ = self.sga(x)         # (B, L, C)

        # 时间池化
        if self.pool == "mean":
            h = x.mean(dim=1)                          # (B, C)
        elif self.pool == "last":
            h = x[:, -1, :]                             # (B, C)
        else:
            # 末端加权池化（线性递增权重）
            B, L, _ = x.shape
            w = torch.linspace(0.1, 1.0, steps=L, device=x.device).view(1, L, 1)
            h = (x * w).sum(dim=1) / w.sum(dim=1)      # (B, C)

        y = self.head(h).squeeze(-1)   # (B,)
        return y

# ---------------- 子类：继承 BaseRULModel ----------------
class TSMixerModel(BaseRULModel):
    """
    保持与原接口一致；新增 SGA 相关可选参数：
      - use_sga: 是否启用 SGA
      - sga_time_rr / sga_feat_rr: 时间/特征方向的压缩比（越大越“轻”）
      - sga_dropout: SGA 内部的 dropout
      - pool: "mean" | "last" | "weighted"
    """
    def __init__(self,
                 input_size: int,
                 seq_len: int,
                 num_layers: int = 4,
                 time_expansion: int = 4,
                 feat_expansion: int = 4,
                 dropout: float = 0.1,
                 *,
                 use_sga: bool = False,
                 sga_time_rr: int = 4,
                 sga_feat_rr: int = 4,
                 sga_dropout: float = 0.05,
                 pool: Literal["mean", "last", "weighted"] = "mean"):
        super().__init__(input_size=input_size, seq_len=seq_len, out_channels=1)
        self.cfg = dict(
            num_layers=num_layers,
            time_expansion=time_expansion,
            feat_expansion=feat_expansion,
            dropout=dropout,
            use_sga=use_sga,
            sga_time_rr=sga_time_rr,
            sga_feat_rr=sga_feat_rr,
            sga_dropout=sga_dropout,
            pool=pool
        )
        self.model = self.build_model().to(self.device)

    # --- 必须实现 ---
    def build_model(self) -> nn.Module:
        return TSMixerRegressor(
            input_length=self.seq_len,
            num_features=self.input_size,
            num_layers=self.cfg["num_layers"],
            time_expansion=self.cfg["time_expansion"],
            feat_expansion=self.cfg["feat_expansion"],
            dropout=self.cfg["dropout"],
            use_sga=self.cfg["use_sga"],
            sga_time_rr=self.cfg["sga_time_rr"],
            sga_feat_rr=self.cfg["sga_feat_rr"],
            sga_dropout=self.cfg["sga_dropout"],
            pool=self.cfg["pool"],
        )

    # --- 必须实现 ---
    def compile(self,
                learning_rate: float = 1e-3,
                weight_decay: float = 1e-4,
                *,
                scheduler: Literal["onecycle", "plateau", "cosine", "none"] = "plateau",
                epochs: int = 30,
                steps_per_epoch: int = 100):
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
                pct_start=0.3,
                div_factor=25,
                final_div_factor=100,
                anneal_strategy='cos'
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

    # 与之前格式一致的日志
    def log_training_metrics(self, epoch, epochs, train_rmse, val_rmse_global, 
                             val_rmse_last, val_score_last, lr):
        import logging
        logger = logging.getLogger(__name__)
        def fmt(v):
            try:
                return f"{float(v):.2f}"
            except Exception:
                return "N/A"
        msg = (f"[Epoch {epoch:3d}/{epochs}] "
               f"train_rmse={fmt(train_rmse)} | "
               f"val_rmse(global)={fmt(val_rmse_global)} cycles | "
               f"val_rmse(last)={fmt(val_rmse_last)} cycles | "
               f"val_score(last)={fmt(val_score_last)} | "
               f"lr={lr:.2e}")
        logger.info(msg)
