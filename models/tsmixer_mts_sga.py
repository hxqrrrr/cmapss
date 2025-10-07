# models/tsmixer_mts_sga.py
import math
from typing import List, Tuple, Literal, Dict, Any
import torch
import torch.nn as nn

from models.base_model import BaseRULModel

# ---------------------------- 小工具 ----------------------------
class Lambda(nn.Module):
    def __init__(self, fn): super().__init__(); self.fn = fn
    def forward(self, x): return self.fn(x)


# ------------------------- Feature-Mix（通道混合） -------------------------
class FeatureMix(nn.Module):
    """沿特征维的 MLP：LayerNorm(C) + Linear(C->C*exp) + GELU + Dropout + Linear + Dropout + 残差"""
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
    def forward(self, x):  # x: [B,L,C]
        return x + self.mlp(self.norm(x))

# --------------------- 标准 TimeMix（时间混合） ---------------------
class TimeMix(nn.Module):
    """
    标准 TSMixer 的 Time-Mix：在时间维做 MLP + 残差。
    实现：x -> (B,C,L)，LayerNorm(L) + Linear(L->L*exp) + GELU + Dropout + Linear + Dropout -> 残差
    """
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
    
    def forward(self, x):  # x: [B,L,C]
        xt = x.transpose(1,2).contiguous()   # [B,C,L]
        y = self.mlp(self.norm(xt))
        y = xt + y
        return y.transpose(1,2).contiguous() # [B,L,C]

# ------------------------- SGA：双轴注意力 -------------------------
class SGA(nn.Module):
    """
    双向 squeeze：
      - 时间轴：沿特征均值 → [B,L,1] -> 小 MLP -> A_time
      - 特征轴：沿时间均值 → [B,1,C] -> 小 MLP -> A_feat
    自适应融合：A = sigmoid( gamma*(A_time + A_feat) + beta )
    输出：Y = X * A
    """
    def __init__(self, seq_len: int, num_features: int,
                 hidden_time: int = 16, hidden_feat: int = 16,
                 dropout: float = 0.05):
        super().__init__()
        # 时间轴注意
        self.t_mlp = nn.Sequential(
            nn.LayerNorm(1),
            nn.Linear(1, hidden_time),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_time, 1)
        )
        # 特征轴注意
        self.f_mlp = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, hidden_feat),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feat, num_features)
        )
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):  # x: [B,L,C]
        # 时间轴权重：对每个 t，聚合通道均值 → [B,L,1]
        t_agg = x.mean(dim=2, keepdim=True)               # [B,L,1]
        A_t = self.t_mlp(t_agg)                           # [B,L,1]

        # 特征轴权重：对每个通道，聚合时间均值 → [B,1,C]
        f_agg = x.mean(dim=1, keepdim=True)               # [B,1,C]
        A_f = self.f_mlp(f_agg.squeeze(1)).unsqueeze(1)   # [B,1,C]

        A = torch.sigmoid(self.gamma * (A_t + A_f) + self.beta)  # [B,L,C] (广播)
        return x * A

# ------------------------- TSMixer + SGA Block -------------------------
class TSMixerSGABlock(nn.Module):
    """
    一个 Block = 标准 TimeMix -> FeatureMix -> SGA
    三者各带残差，组合后再做一次轻 Norm 稳定。
    """
    def __init__(self, seq_len: int, num_features: int,
                 time_expansion: int = 4,
                 feat_expansion: int = 4,
                 dropout: float = 0.1,
                 sga_time_hidden: int = 16,
                 sga_feat_hidden: int = 16,
                 sga_dropout: float = 0.05):
        super().__init__()
        self.time_mix = TimeMix(seq_len, time_expansion, dropout)
        self.feat_mix = FeatureMix(num_features, feat_expansion, dropout)
        self.sga = SGA(seq_len, num_features, sga_time_hidden, sga_feat_hidden, sga_dropout)
        self.tail_norm = nn.LayerNorm(num_features)

    def forward(self, x):  # [B,L,C]
        x = self.time_mix(x)
        x = self.feat_mix(x)
        x = self.sga(x)
        return self.tail_norm(x)

# ------------------------- 整体网络 -------------------------
class TSMixerMTS_SGA_Backbone(nn.Module):
    """
    输入:  (B, L, C)
    输出:  (B,)  —— 均值池化 + 线性
    """
    def __init__(self,
                 input_length: int,
                 num_features: int,
                 num_layers: int,
                 time_expansion: int = 4,
                 feat_expansion: int = 4,
                 dropout: float = 0.1,
                 sga_time_hidden: int = 16,
                 sga_feat_hidden: int = 16,
                 sga_dropout: float = 0.05):
        super().__init__()
        self.blocks = nn.ModuleList([
            TSMixerSGABlock(
                seq_len=input_length,
                num_features=num_features,
                time_expansion=time_expansion,
                feat_expansion=feat_expansion,
                dropout=dropout,
                sga_time_hidden=sga_time_hidden,
                sga_feat_hidden=sga_feat_hidden,
                sga_dropout=sga_dropout
            ) for _ in range(num_layers)
        ])
        self.head = nn.Linear(num_features, 1)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):  # [B,L,C]
        for blk in self.blocks:
            x = blk(x)
        h = x.mean(dim=1)              # 时间池化
        y = self.head(h).squeeze(-1)   # [B]
        return y

# ------------------------- RUL Model Wrapper -------------------------
class TSMixerMTS_SGA_RULModel(BaseRULModel):
    """
    标准 TSMixer + SGA 双轴注意力
    命令行示例：
    --model tsmixer_mts_sga
    --fault FD002
    --tsmixer_layers 6
    --time_expansion 4
    --feat_expansion 5
    --dropout 0.12
    --mts_sga_time_hidden 16
    --mts_sga_feat_hidden 16
    --mts_sga_dropout 0.05
    """
    def __init__(self,
                 input_size: int,
                 seq_len: int,
                 out_channels: int = 1,
                 tsmixer_layers: int = 6,
                 time_expansion: int = 4,
                 feat_expansion: int = 4,
                 dropout: float = 0.12,
                 mts_scales: str = "3-1,5-2,7-3",  # 保留参数兼容性，但不使用
                 mts_gate_hidden: int = 16,        # 保留参数兼容性，但不使用
                 sga_time_hidden: int = 16,
                 sga_feat_hidden: int = 16,
                 sga_dropout: float = 0.05):
        super().__init__(input_size=input_size, seq_len=seq_len, out_channels=out_channels)

        self.hparams: Dict[str, Any] = dict(
            tsmixer_layers=tsmixer_layers,
            time_expansion=time_expansion,
            feat_expansion=feat_expansion,
            dropout=dropout,
            sga_time_hidden=sga_time_hidden,
            sga_feat_hidden=sga_feat_hidden,
            sga_dropout=sga_dropout
        )
        self.model = self.build_model().to(self.device)
    
    def build_model(self) -> nn.Module:
        """构建模型（BaseRULModel 必须实现的抽象方法）"""
        return TSMixerMTS_SGA_Backbone(
            input_length=self.seq_len,
            num_features=self.input_size,
            num_layers=self.hparams["tsmixer_layers"],
            time_expansion=self.hparams["time_expansion"],
            feat_expansion=self.hparams["feat_expansion"],
            dropout=self.hparams["dropout"],
            sga_time_hidden=self.hparams["sga_time_hidden"],
            sga_feat_hidden=self.hparams["sga_feat_hidden"],
            sga_dropout=self.hparams["sga_dropout"]
        )

    def compile(self,
                learning_rate: float = 8e-4,
                weight_decay: float = 2e-4,
                *,
                scheduler: Literal["onecycle","plateau","cosine","none"] = "cosine",
                epochs: int = 100,
                steps_per_epoch: int = 100,
                warmup_epochs: int = 0):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=learning_rate,
                                           weight_decay=weight_decay)
        if scheduler == "onecycle":
            from torch.optim.lr_scheduler import OneCycleLR
            self.scheduler = OneCycleLR(self.optimizer,
                                        max_lr=learning_rate,
                                        epochs=epochs,
                                        steps_per_epoch=steps_per_epoch,
                                        pct_start=0.2,
                                        anneal_strategy='cos')
        elif scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, epochs - warmup_epochs)
            )
        elif scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )
        else:
            self.scheduler = None

    # 训练日志（与现有风格对齐）
    def log_training_metrics(self, epoch, epochs, train_rmse, val_rmse_global,
                             val_rmse_last, val_score_last, lr):
        import logging
        logger = logging.getLogger(__name__)
        def fmt(x):
            try: return f"{float(x):.2f}"
            except: return "N/A"
        msg = (f"[Epoch {epoch:3d}/{epochs}] "
               f"train_rmse={fmt(train_rmse)} | "
               f"val_rmse(global)={fmt(val_rmse_global)} cycles | "
               f"val_rmse(last)={fmt(val_rmse_last)} cycles | "
               f"val_score(last)={fmt(val_score_last)} | "
               f"lr={lr:.2e}")
        logger.info(msg)

    # 一键 FD002 推荐（示例）
    @classmethod
    def config_fd002(cls, input_size: int):
        return dict(
            input_size=input_size, seq_len=30,         # 你的 FD002 数据窗口
            tsmixer_layers=6,
            time_expansion=4, feat_expansion=5,
            dropout=0.12,
            sga_time_hidden=16, sga_feat_hidden=16,
            sga_dropout=0.05
        )
