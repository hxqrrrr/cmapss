# models/tsmixer_eca_model.py
import math
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseRULModel


# -------------------------
# 前端：ECA 通道注意力
# -------------------------
class ECAGate(nn.Module):
    """
    Efficient Channel Attention (ECA) for sequence data.
    输入:  (B, L, C)
    输出:  (B, L, C)  —— 仅做通道重标定，不改变形状
    机制: 先沿时间维做全局平均，得 (B, C)，再做 1D 卷积 (跨通道的局部交互) → Sigmoid → 通道权重
    """
    def __init__(self, channels: int, k_size: int = 5):
        super().__init__()
        if k_size % 2 == 0:
            k_size += 1  # ECA 需要奇数卷积核
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        # 全局时间池化 → (B, C)
        s = x.mean(dim=1, keepdim=False)  # (B, C)
        # 按 ECA 做跨通道局部交互：把通道当作“长度”，在一个“伪”1D上卷积
        # reshape 为 (B, 1, C)
        s = s.unsqueeze(1)
        a = self.conv1d(s)               # (B, 1, C)
        a = torch.sigmoid(a).squeeze(1)  # (B, C)
        # 施加到原序列（广播到 L）
        return x * a.unsqueeze(1)        # (B, L, C)


# -------------------------
# 前端：TCN 残差块（保持长度）
# -------------------------
class TCNBlock(nn.Module):
    """
    标准 TCN 残差块（非因果，等长，适合“双向”信息）
    输入/输出: (B, L, C)
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 应为奇数以便等长填充"

        pad = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                               padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                               padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

        # 残差映射（此处通道不变，保留接口）
        self.res_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) → (B, C, L)
        x_c = x.transpose(1, 2)
        y = self.conv1(x_c)
        y = self.bn1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.drop(y)
        # 残差
        y = y + self.res_proj(x_c)
        y = self.act(y)
        return y.transpose(1, 2)  # 回到 (B, L, C)


class TCNStack(nn.Module):
    """
    多层 TCNBlock 串联，支持不同 dilation
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilations: Optional[List[int]] = None, dropout: float = 0.1):
        super().__init__()
        if dilations is None or len(dilations) == 0:
            dilations = [1, 2]
        self.blocks = nn.ModuleList([
            TCNBlock(channels, kernel_size=kernel_size, dilation=d, dropout=dropout)
            for d in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class BiTCN(nn.Module):
    """
    双向 TCN：共享一套 TCNStack，分别处理正序与反序，再融合（平均/相加）
    输入/输出: (B, L, C)
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilations: Optional[List[int]] = None,
                 dropout: float = 0.1, fuse: Literal["mean", "sum", "cat"] = "mean"):
        super().__init__()
        self.tcn = TCNStack(channels, kernel_size=kernel_size, dilations=dilations, dropout=dropout)
        self.fuse = fuse
        if fuse == "cat":
            # 如果选择拼接，再投回原通道数
            self.proj = nn.Linear(channels * 2, channels)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward 分支
        y_f = self.tcn(x)  # (B, L, C)

        # backward 分支（时间反转）
        y_b = self.tcn(torch.flip(x, dims=[1]))  # (B, L, C)
        y_b = torch.flip(y_b, dims=[1])

        if self.fuse == "mean":
            y = 0.5 * (y_f + y_b)
        elif self.fuse == "sum":
            y = y_f + y_b
        else:  # "cat"
            y = torch.cat([y_f, y_b], dim=-1)
            y = self.proj(y)

        return y  # (B, L, C)


# -------------------------
# 主干：TSMixer（与原实现一致）
# -------------------------
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


# -------------------------
# 整体回归器：ECA → BiTCN → TSMixer → Head
# -------------------------
class ECABiTCN_TSMixerRegressor(nn.Module):
    """
    输入:  (B, L, C)
    输出:  (B,)   —— 前端(ECA/BiTCN) + Mixer 堆栈 + 全局时间池化 + 线性头
    """
    def __init__(self,
                 input_length: int,
                 num_features: int,
                 num_layers: int = 4,
                 time_expansion: int = 4,
                 feat_expansion: int = 4,
                 mixer_dropout: float = 0.1,
                 *,
                 use_eca: bool = True,
                 eca_kernel: int = 5,
                 use_bitcn: bool = True,
                 tcn_kernel: int = 3,
                 tcn_dilations: Optional[List[int]] = None,
                 tcn_dropout: float = 0.1,
                 tcn_fuse: Literal["mean", "sum", "cat"] = "mean"):
        super().__init__()

        # 前端 ECA
        self.eca = ECAGate(num_features, k_size=eca_kernel) if use_eca else nn.Identity()

        # 前端 BiTCN
        if use_bitcn:
            if tcn_dilations is None:
                # 1~2 个残差块即可：如 [1, 2] 或 [1, 2, 4]
                tcn_dilations = [1, 2]
            self.bitcn = BiTCN(
                channels=num_features,
                kernel_size=tcn_kernel,
                dilations=tcn_dilations,
                dropout=tcn_dropout,
                fuse=tcn_fuse
            )
        else:
            self.bitcn = nn.Identity()

        # TSMixer 主干
        self.blocks = nn.ModuleList([
            MixerBlock(
                input_length, num_features,
                time_expansion=time_expansion,
                feat_expansion=feat_expansion,
                dropout=mixer_dropout
            )
            for _ in range(num_layers)
        ])

        # 轻量回归头
        self.head = nn.Linear(num_features, 1)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前端整形：ECA → BiTCN
        x = self.eca(x)         # (B, L, C)
        x = self.bitcn(x)       # (B, L, C)

        # Mixer 主干
        for blk in self.blocks:
            x = blk(x)          # (B, L, C)

        # 全局时间平均 + 线性头
        h = x.mean(dim=1)       # (B, C)
        y = self.head(h).squeeze(-1)
        return y                # (B,)


# -------------------------
# 顶层封装：继承 BaseRULModel
# -------------------------
class ECATSMixerModel(BaseRULModel):
    """
    方案 A：ECA-前置适配 + 双向TCN 前端 → TSMixer 主干 → 轻量回归头
    提供开关:
      - use_eca:   True/False
      - use_bitcn: True/False
    """
    def __init__(self,
                 input_size: int,
                 seq_len: int,
                 *,
                 num_layers: int = 4,
                 time_expansion: int = 4,
                 feat_expansion: int = 4,
                 mixer_dropout: float = 0.1,
                 # ECA
                 use_eca: bool = True,
                 eca_kernel: int = 5,
                 # BiTCN
                 use_bitcn: bool = True,
                 tcn_kernel: int = 3,
                 tcn_dilations: Optional[List[int]] = None,
                 tcn_dropout: float = 0.1,
                 tcn_fuse: Literal["mean", "sum", "cat"] = "mean"):
        super().__init__(input_size=input_size, seq_len=seq_len, out_channels=1)

        # 保存结构配置
        self.num_layers = num_layers
        self.time_expansion = time_expansion
        self.feat_expansion = feat_expansion
        self.mixer_dropout = mixer_dropout

        self.use_eca = use_eca
        self.eca_kernel = eca_kernel

        self.use_bitcn = use_bitcn
        self.tcn_kernel = tcn_kernel
        self.tcn_dilations = tcn_dilations
        self.tcn_dropout = tcn_dropout
        self.tcn_fuse = tcn_fuse

        # 构建 nn.Module 并放到设备
        self.model = self.build_model().to(self.device)

    # --- 必须实现 ---
    def build_model(self) -> nn.Module:
        return ECABiTCN_TSMixerRegressor(
            input_length=self.seq_len,
            num_features=self.input_size,
            num_layers=self.num_layers,
            time_expansion=self.time_expansion,
            feat_expansion=self.feat_expansion,
            mixer_dropout=self.mixer_dropout,
            use_eca=self.use_eca,
            eca_kernel=self.eca_kernel,
            use_bitcn=self.use_bitcn,
            tcn_kernel=self.tcn_kernel,
            tcn_dilations=self.tcn_dilations,
            tcn_dropout=self.tcn_dropout,
            tcn_fuse=self.tcn_fuse
        )

    # --- 必须实现 ---
    def compile(self,
                learning_rate: float = 1e-3,
                weight_decay: float = 1e-4,
                *,
                scheduler: Literal["onecycle", "plateau", "cosine", "none"] = "plateau",
                epochs: int = 30,
                steps_per_epoch: int = 100,
                warmup_epochs: int = 0):
        """
        优化器: AdamW
        调度器:
        - onecycle  : per-batch step
        - cosine    : 支持 Linear warmup + Cosine (per-batch step)
        - plateau   : ReduceLROnPlateau (per-epoch step with metric)
        - none      : 不使用调度器
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=learning_rate,
                                        weight_decay=weight_decay)

        # 缺省：不在 batch 内 step
        self._sched_per_batch = False
        sch = scheduler.lower() if scheduler else "none"

        if sch == "onecycle":
            from torch.optim.lr_scheduler import OneCycleLR
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=learning_rate,
                epochs=epochs,
                steps_per_epoch=max(1, steps_per_epoch),
                pct_start=0.3,
                div_factor=25,
                final_div_factor=100,
                anneal_strategy='cos'
            )
            self._sched_per_batch = True  # OneCycle 必须 per-batch step

        elif sch == "cosine":
            # 统一用 per-batch step
            total_steps = max(1, epochs * max(1, steps_per_epoch))
            if warmup_epochs and warmup_epochs > 0:
                from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
                warmup_steps = max(1, warmup_epochs * max(1, steps_per_epoch))
                cosine_steps = max(1, total_steps - warmup_steps)
                warmup = LinearLR(self.optimizer, start_factor=1e-3, end_factor=1.0,
                                total_iters=warmup_steps)
                cosine = CosineAnnealingLR(self.optimizer, T_max=cosine_steps)
                self.scheduler = SequentialLR(self.optimizer, [warmup, cosine],
                                            milestones=[warmup_steps])
            else:
                from torch.optim.lr_scheduler import CosineAnnealingLR
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
            self._sched_per_batch = True  # cosine 也改为 per-batch step

        elif sch == "plateau":
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)
            self._sched_per_batch = False  # epoch 末 step(metric)

        else:
            self.scheduler = None
            self._sched_per_batch = False


# -------------------------
# 兼容别名：不改旧训练脚本也能用
# -------------------------
# 老脚本如果还在 `from models.tsmixer_model import TSMixerModel`，可把导入路径改到本文件；
# 或者直接在项目里将旧文件替换为本文件。
TSMixerModel = ECATSMixerModel
