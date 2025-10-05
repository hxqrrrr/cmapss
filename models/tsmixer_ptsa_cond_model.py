# models/tsmixer_ptsa_cond_model.py
import torch
import torch.nn as nn
from typing import Optional, Any, Dict

from .base_model import BaseRULModel
from .tsmixer_ptsa import ModelTSMixerPTSA     # 你已有
from .dual_gate_film import DualAxisCondGate

# ---------- 可选：简易标准化头，适配骨干输入 ----------
class InputProj(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        if in_ch == out_ch:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(in_ch, out_ch)
    def forward(self, x):  # [B,T,Cin] -> [B,T,Cout]
        return self.proj(x)

class _BackboneWrapper(nn.Module):
    """把现有 ModelTSMixerPTSA 的 .model 拿出来当普通模块用"""
    def __init__(self, core: ModelTSMixerPTSA):
        super().__init__()
        # 现有类继承 BaseRULModel，一般把真正的 nn.Module 放在 core.model
        self.core = core
    def forward(self, x):
        # core.model 应该接受 [B,T,C] 并输出 [B,1]（或 [B]）
        return self.core.model(x)

class TSMixerPTSACore(nn.Module):
    """
    CondGate(工况门控) → TSMixer-PTSA骨干 → (可选)CondGate → 线性头
    """
    def __init__(
        self,
        input_size: int, seq_len: int, out_channels: int,
        channels: int, depth: int,
        time_expansion: int, feat_expansion: int, dropout: float,
        use_ptsa: bool, ptsa_every_k: int, ptsa_heads: int,
        ptsa_local_window: int, ptsa_topk: int, ptsa_levels: int,
        ptsa_parent_neigh: int, ptsa_dropout: float,
        distill_type: str, distill_stride: int, reduce_channels: float, drop_path: float,
        cond_dim: int, eca_kernel: int, time_kernel: int,
        use_post_gate: bool
    ):
        super().__init__()
        # 1) 前置工况门控（通道数使用骨干主干 channels）
        self.pre_gate = DualAxisCondGate(channels=channels, cond_dim=cond_dim,
                                         k_eca=eca_kernel, k_time=time_kernel)
        # 2) 将输入投到骨干通道数（如需要）
        self.in_proj = InputProj(in_ch=input_size, out_ch=channels)

        # 3) 复用已有骨干：直接实例化一个 ModelTSMixerPTSA，然后只用它的 .model 模块
        core = ModelTSMixerPTSA(
            input_size=channels,       # 已投影
            seq_len=seq_len, out_channels=out_channels,
            channels=channels, depth=depth,
            time_expansion=time_expansion, feat_expansion=feat_expansion, dropout=dropout,
            use_ptsa=use_ptsa, ptsa_every_k=ptsa_every_k, ptsa_heads=ptsa_heads,
            ptsa_local_window=ptsa_local_window, ptsa_topk=ptsa_topk, ptsa_levels=ptsa_levels,
            ptsa_parent_neigh=ptsa_parent_neigh, ptsa_dropout=ptsa_dropout,
            distill_type=distill_type, distill_stride=distill_stride,
            reduce_channels=reduce_channels, drop_path=drop_path
        )
        self.backbone = _BackboneWrapper(core)

        # 4) 可选后置门控（对骨干输出的token特征再调制；如果骨干已是标量输出则跳过）
        self.use_post_gate = use_post_gate
        if use_post_gate:
            self.post_gate = DualAxisCondGate(channels=channels, cond_dim=cond_dim,
                                              k_eca=eca_kernel, k_time=time_kernel)

        # 5) 预测头：骨干通常已经输出标量；若输出为序列/通道，这里兜底
        self.head = nn.Identity()

    def _extract_cond(self, x):
        """
        假定 FD002/004 使用 features='all'：前三维为工况设置量。
        非 all 时安全退化为零向量。
        """
        if x.size(-1) >= 3:
            cond = x[..., :3].mean(dim=1)  # [B,3]
        else:
            cond = x.new_zeros(x.size(0), 3)
        return cond

    def forward(self, x):  # x:[B,T,C_in]
        cond = self._extract_cond(x)
        x = self.in_proj(x)              # [B,T,channels]
        x = self.pre_gate(x, cond)       # 工况感知加权
        y = self.backbone(x)             # 复用骨干推理（期望 [B,1] 或 [B]）

        # 若骨干返回形状不是标量，这里可做一次 post_gate + 池化
        if self.use_post_gate and y.dim() == 3:
            y = self.post_gate(y, cond)
            y = y.mean(dim=1)            # [B,C] → 线性头
            y = self.head(y)             # 这里为 Identity；需要可加 Linear(C,1)

        return y

class ModelTSMixerPTSACOND(BaseRULModel):
    """
    面向 train.py 的封装：遵守 BaseRULModel 接口
    """
    def __init__(self, input_size: int, seq_len: int, out_channels: int = 1,
                 channels: int = 128, depth: int = 6,
                 time_expansion: int = 4, feat_expansion: int = 8, dropout: float = 0.04,
                 use_ptsa: bool = True, ptsa_every_k: int = 2, ptsa_heads: int = 6,
                 ptsa_local_window: int = 12, ptsa_topk: int = 10, ptsa_levels: int = 1,
                 ptsa_parent_neigh: int = 1, ptsa_dropout: float = 0.0,
                 distill_type: str = "conv", distill_stride: int = 2, reduce_channels: float = 1.0,
                 drop_path: float = 0.0,
                 # 条件门控
                 cond_dim: int = 3, eca_kernel: int = 5, time_kernel: int = 11, use_post_gate: bool = False):
        super().__init__(input_size, seq_len, out_channels)
        self.cfg = dict(
            input_size=input_size, seq_len=seq_len, out_channels=out_channels,
            channels=channels, depth=depth,
            time_expansion=time_expansion, feat_expansion=feat_expansion, dropout=dropout,
            use_ptsa=use_ptsa, ptsa_every_k=ptsa_every_k, ptsa_heads=ptsa_heads,
            ptsa_local_window=ptsa_local_window, ptsa_topk=ptsa_topk, ptsa_levels=ptsa_levels,
            ptsa_parent_neigh=ptsa_parent_neigh, ptsa_dropout=ptsa_dropout,
            distill_type=distill_type, distill_stride=distill_stride, reduce_channels=reduce_channels,
            drop_path=drop_path,
            cond_dim=cond_dim, eca_kernel=eca_kernel, time_kernel=time_kernel, use_post_gate=use_post_gate
        )
        self.model = self.build_model().to(self.device)

    def build_model(self) -> nn.Module:
        return TSMixerPTSACore(**self.cfg)

    # ====== 与你现有 compile 习惯保持一致：支持 onecycle / cosine(+warmup) / plateau ======
    def compile(self, learning_rate: float, weight_decay: float, **kwargs):
        import math
        opt = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optimizer = opt

        scheduler_name = kwargs.get("scheduler", "onecycle")
        epochs = int(kwargs.get("epochs", 50))
        steps_per_epoch = int(kwargs.get("steps_per_epoch", 100))
        warmup_epochs = int(kwargs.get("warmup_epochs", 0))

        if scheduler_name == "onecycle":
            total_steps = max(1, epochs * steps_per_epoch)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=learning_rate, total_steps=total_steps,
                pct_start=max(1, warmup_epochs) / epochs if epochs > 0 else 0.1,
                anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4
            )
        elif scheduler_name == "cosine":
            if warmup_epochs > 0:
                warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0,
                                                         total_iters=max(1, warmup_epochs * steps_per_epoch))
                cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, (epochs - warmup_epochs) * steps_per_epoch))
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, cos],
                                                                       milestones=[warmup_epochs * steps_per_epoch])
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * steps_per_epoch))
        elif scheduler_name == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
        else:
            self.scheduler = None

        # 把 hint 也挂上（AMP + 累积）
        self.accum_steps = int(kwargs.get("accum_steps", getattr(self, "accum_steps", 1)))
        self.use_amp = bool(kwargs.get("use_amp", True))
