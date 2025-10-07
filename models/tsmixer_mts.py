# models/tsmixer_mts.py
# -----------------------------------------------------------------------------
# MTS-TSMixer: 多尺度 TimeMix（替换标准 TimeMix），适配 BaseRULModel 训练框架
# 设计目标：
#   - 在 TimeMix 内做“多分支不同时间尺度”的并行变换（短/中/长/超长）
#   - 每个分支：Depthwise 1D Conv（可设不同 dilation/kernel）→ LayerNorm → MLP(沿时间维)
#   - 分支融合：基于片段统计的门控（softmax），可随工况/退化节律自适应
#   - 之后接标准 FeatureMixing，不改整体深度与头部
# 适用：FD002/FD004（多工况/长依赖），其余子集也可用
# -----------------------------------------------------------------------------

import math
from typing import List, Literal, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseRULModel


# --------------------- 基础模块：Depthwise 1D Conv ---------------------
class DepthwiseConv1d(nn.Module):
    """深度可分离卷积中的 depthwise 部分（每个通道独立卷积），用于时间预过滤"""
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, bias: bool = True):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.dw = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            dilation=dilation,
            padding=padding,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> (B, C, L) -> conv -> (B, L, C)
        x = x.transpose(1, 2)
        x = self.dw(x)
        return x.transpose(1, 2)


# --------------------- 时间维 MLP（单尺度） ---------------------
class TimeMLP(nn.Module):
    """
    在时间维上的 MLP：对每个通道共享（等价在 (B,C,L) 的 L 维做线性）
    实现方式：转置到 (B,C,L)，对 L 做 LN+Linear，保持广播友好
    """
    def __init__(self, seq_len: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = seq_len * expansion
        self.norm = nn.LayerNorm(seq_len)
        self.ff = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, seq_len),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        xt = x.transpose(1, 2).contiguous()   # (B, C, L)
        y = self.ff(self.norm(xt))            # (B, C, L)
        return y.transpose(1, 2).contiguous() # (B, L, C)


# --------------------- 多尺度 TimeMix 分支 ---------------------
class TimeBranch(nn.Module):
    """
    单个尺度分支：DepthwiseConv1d(不同感受野) → TimeMLP → 残差
    - conv 只做“轻过滤+对齐相位/节律”
    - TimeMLP 做真正的 Token-mixing（沿时间维）
    """
    def __init__(
        self,
        seq_len: int,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        expansion: int,
        dropout: float
    ):
        super().__init__()
        self.prefilter = DepthwiseConv1d(channels, kernel_size=kernel_size, dilation=dilation, bias=True)
        self.time_mlp = TimeMLP(seq_len=seq_len, expansion=expansion, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.prefilter(x)
        y = self.time_mlp(y)
        return y


# --------------------- 多尺度 TimeMix（并行分支 + 门控融合） ---------------------
class MultiScaleTimeMix(nn.Module):
    """
    多尺度并行：
      outputs = [branch_i(x) for i in branches]
      gates   = Gate(x_stats)  # B x n_scales, softmax
      y = sum_i gates[:, i] * outputs[i]
    其中 Gate 由片段统计（均值/标准差/趋势）生成，能够随工况自适应地选尺度。
    """
    def __init__(
        self,
        seq_len: int,
        channels: int,
        scales: List[Dict[str, int]],
        *,
        expansion: int = 4,
        dropout: float = 0.1,
        gate_hidden: int = 16
    ):
        super().__init__()
        # 构建多尺度分支
        self.branches = nn.ModuleList([
            TimeBranch(
                seq_len=seq_len,
                channels=channels,
                kernel_size=cfg.get("kernel", 3),
                dilation=cfg.get("dilation", 1),
                expansion=cfg.get("expansion", expansion),
                dropout=dropout
            ) for cfg in scales
        ])
        self.n_scales = len(self.branches)

        # 片段统计特征：均值、标准差、简单趋势（末均值-首均值） -> 3 维
        gate_in_dim = 3
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.n_scales)
        )

        # 残差/Dropout
        self.residual = True
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _segment_stats(x: torch.Tensor) -> torch.Tensor:
        """
        从输入片段 x 计算门控所需统计：
          - 全局均值（沿 L,C）
          - 全局方差的 sqrt（标准差）
          - 简单趋势：后1/4窗口均值 - 前1/4窗口均值（沿 L,C）
        返回 (B, 3)
        """
        # x: (B, L, C)
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))

        L = x.size(1)
        q = max(1, L // 4)
        head = x[:, :q, :].mean(dim=(1, 2))
        tail = x[:, -q:, :].mean(dim=(1, 2))
        trend = tail - head
        stats = torch.stack([mean, std, trend], dim=-1)  # (B,3)
        return stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 并行分支
        outs = [b(x) for b in self.branches]   # list of (B, L, C)
        Y = torch.stack(outs, dim=1)           # (B, n_scales, L, C)

        # 门控（随片段统计变化）
        stats = self._segment_stats(x)         # (B,3)
        gates = F.softmax(self.gate_mlp(stats), dim=-1)  # (B, n_scales)

        # 融合
        gates = gates.unsqueeze(-1).unsqueeze(-1)        # (B, n_scales, 1, 1)
        y = (Y * gates).sum(dim=1)                       # (B, L, C)

        y = self.drop(y)
        return x + y if self.residual else y


# --------------------- 特征维 MLP（与标准 TSMixer 一致） ---------------------
class FeatureMixing(nn.Module):
    """沿特征维的 MLP（对每个时间步共享）"""
    def __init__(self, num_features: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = num_features * expansion
        self.norm = nn.LayerNorm(num_features)
        self.ff = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


# --------------------- MTS-TSMixer Block ---------------------
class MTSMixerBlock(nn.Module):
    """
    一个 Block = MultiScaleTimeMix（并行时间分支+门控） → FeatureMixing
    """
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        *,
        time_scales: List[Dict[str, int]],
        time_expansion: int = 4,
        feat_expansion: int = 4,
        dropout: float = 0.1,
        gate_hidden: int = 16
    ):
        super().__init__()
        self.time = MultiScaleTimeMix(
            seq_len=seq_len,
            channels=num_features,
            scales=time_scales,
            expansion=time_expansion,
            dropout=dropout,
            gate_hidden=gate_hidden
        )
        self.feat = FeatureMixing(num_features=num_features, expansion=feat_expansion, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feat(self.time(x))


# --------------------- 整体网络：MTS-TSMixer ---------------------
class MTS_TSMixerRegressor(nn.Module):
    """
    输入: (B, L, C)
    Block 堆叠后：时间平均 → 线性回归到 1 维 → (B,)
    """
    def __init__(
        self,
        input_length: int,
        num_features: int,
        num_layers: int = 4,
        *,
        time_scales: Optional[List[Dict[str, int]]] = None,
        time_expansion: int = 4,
        feat_expansion: int = 4,
        dropout: float = 0.1,
        gate_hidden: int = 16
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        if time_scales is None:
            # 默认 4 个尺度：短/中/长/超长（核大小 & 膨胀率可按需调整）
            time_scales = [
                {"kernel": 3, "dilation": 1, "expansion": time_expansion},   # 短
                {"kernel": 3, "dilation": 2, "expansion": time_expansion},   # 中
                {"kernel": 5, "dilation": 3, "expansion": time_expansion},   # 长
                {"kernel": 7, "dilation": 4, "expansion": time_expansion},   # 超长
            ]
        for _ in range(num_layers):
            self.blocks.append(
                MTSMixerBlock(
                    seq_len=input_length,
                    num_features=num_features,
                    time_scales=time_scales,
                    time_expansion=time_expansion,
                    feat_expansion=feat_expansion,
                    dropout=dropout,
                    gate_hidden=gate_hidden
                )
            )
        self.head = nn.Linear(num_features, 1)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        for blk in self.blocks:
            x = blk(x)
        h = x.mean(dim=1)                 # 时间池化
        y = self.head(h).squeeze(-1)      # (B,)
        return y


# --------------------- 适配 BaseRULModel 的封装 ---------------------
class MTSTSMixerRULModel(BaseRULModel):
    """
    与已有训练脚本对齐：
      - build_model: 返回 nn.Module
      - compile: AdamW + 可选调度器
      - log_training_metrics: 与现有格式一致
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        *,
        num_layers: int = 4,
        time_expansion: int = 4,
        feat_expansion: int = 4,
        dropout: float = 0.1,
        gate_hidden: int = 16,
        time_scales: Optional[List[Dict[str, int]]] = None
    ):
        super().__init__(input_size=input_size, seq_len=seq_len, out_channels=1)
        self.hparams: Dict[str, Any] = dict(
            num_layers=num_layers,
            time_expansion=time_expansion,
            feat_expansion=feat_expansion,
            dropout=dropout,
            gate_hidden=gate_hidden,
            time_scales=time_scales
        )
        self.model = self.build_model().to(self.device)

    def build_model(self) -> nn.Module:
        return MTS_TSMixerRegressor(
            input_length=self.seq_len,
            num_features=self.input_size,
            num_layers=self.hparams["num_layers"],
            time_scales=self.hparams["time_scales"],
            time_expansion=self.hparams["time_expansion"],
            feat_expansion=self.hparams["feat_expansion"],
            dropout=self.hparams["dropout"],
            gate_hidden=self.hparams["gate_hidden"]
        )

    # 与现有训练脚本保持一致
    def compile(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        *,
        scheduler: Literal["onecycle", "plateau", "cosine", "none"] = "plateau",
        epochs: int = 30,
        steps_per_epoch: int = 100
    ):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
                anneal_strategy="cos"
            )
        elif scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3
            )
        else:
            self.scheduler = None

    def log_training_metrics(
        self,
        epoch, epochs,
        train_rmse, val_rmse_global,
        val_rmse_last, val_score_last, lr
    ):
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

    # --------------------- 便捷配置（示例） ---------------------
    @classmethod
    def config_fd002(cls, input_size: int):
        """
        FD002 多工况建议：保持总层数不变，采用 4 尺度分支，控制参数量 ≈ 原 TSMixer 的 ~1.3x
        可在命令行暴露 time_scales 的 dilation/kernel
        """
        return dict(
            input_size=input_size,
            seq_len=30,            # 按你数据加载器设置
            num_layers=5,          # 与你常用深度对齐
            time_expansion=4,
            feat_expansion=4,
            dropout=0.12,
            gate_hidden=16,
            time_scales=[
                {"kernel": 3, "dilation": 1, "expansion": 4},  # 短
                {"kernel": 3, "dilation": 2, "expansion": 4},  # 中
                {"kernel": 5, "dilation": 3, "expansion": 4},  # 长
                {"kernel": 7, "dilation": 4, "expansion": 4},  # 超长
            ]
        )

    @classmethod
    def config_fd004(cls, input_size: int):
        """
        FD004 更复杂，可略加强长尺度（更大 kernel/dilation），并加大 dropout 防过拟合
        """
        return dict(
            input_size=input_size,
            seq_len=30,
            num_layers=6,
            time_expansion=5,
            feat_expansion=4,
            dropout=0.15,
            gate_hidden=16,
            time_scales=[
                {"kernel": 3, "dilation": 1, "expansion": 5},
                {"kernel": 5, "dilation": 2, "expansion": 5},
                {"kernel": 5, "dilation": 3, "expansion": 5},
                {"kernel": 7, "dilation": 5, "expansion": 5},
            ]
        )
