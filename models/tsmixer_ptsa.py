# models/tsmixer_ptsa.py
import math
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseRULModel


# ---------------------------
# 基础层：Time/Feature Mixing
# ---------------------------
class TimeMixing(nn.Module):
    """
    Token/Time Mixing：对时间维(L)做 MLP-Mixer 风格混合
    输入: (B, L, C)；内部转为 (B, C, L) 做 1x1+DWConv 再回到 (B, L, C)
    """
    def __init__(self, channels: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.LayerNorm(channels)
        # 在时间维做线性变换：1x1 + 深度可分离 1D conv
        self.pw1 = nn.Conv1d(channels, hidden, kernel_size=1, groups=1, bias=True)
        self.dw  = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=True)
        self.pw2 = nn.Conv1d(hidden, channels, kernel_size=1, groups=1, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        x0 = self.norm(x)
        x1 = x0.transpose(1, 2)  # (B, C, L)
        x1 = self.pw1(x1)
        x1 = F.gelu(self.dw(x1))
        x1 = self.pw2(x1)
        x1 = x1.transpose(1, 2)  # (B, L, C)
        x1 = self.dropout(x1)
        return x + x1


class FeatureMixing(nn.Module):
    """
    Channel/Feature Mixing：对通道维(C)做两层 MLP
    """
    def __init__(self, channels: int, expansion: int = 8, dropout: float = 0.0):
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        x0 = self.norm(x)
        x1 = self.fc2(F.gelu(self.fc1(x0)))
        x1 = self.dropout(x1)
        return x + x1


# ---------------------------
# 稀疏注意力：PTSA (A/C/P)
# ---------------------------
class PTSA(nn.Module):
    """
    Pyramid Temporal Sparse Attention
    - A: 同尺度邻域窗口
    - C: 子尺度链接（来自更细一级）
    - P: 父尺度链接（来自更粗一级）
    稀疏 TopK 遮蔽：在候选集合上做相似度初筛+TopK
    """
    def __init__(
        self,
        channels: int,
        heads: int = 6,
        local_window: int = 12,
        topk: int = 16,
        parent_neigh: int = 1,
        dropout: float = 0.0,
        attn_temp: float = 1.0,
    ):
        super().__init__()

        # 自动对齐 heads（不再直接 assert）
        if channels % heads != 0:
            divisors = [h for h in range(1, channels + 1) if channels % h == 0]
            heads = min(divisors, key=lambda h: abs(h - heads))

        self.channels = channels
        self.heads = heads
        self.dim = channels // heads
        self.local_window = local_window
        self.topk = topk
        self.parent_neigh = parent_neigh
        self.attn_temp = attn_temp

        self.norm = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _build_pyramid(x: torch.Tensor, levels: int, downsample: str = "pool") -> List[torch.Tensor]:
        """
        构建金字塔缓存（仅用于 K/V）
        输入: x (B, L, C)；输出: [x^0, x^1, ..., x^{levels}]，长度逐级 /2
        """
        pyr = [x]
        cur = x
        for _ in range(levels):
            if downsample == "pool":
                cur = F.max_pool1d(cur.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
            else:
                C = cur.size(-1)
                cur_t = cur.transpose(1, 2)  # (B, C, L)
                eye = torch.eye(C, device=cur.device).unsqueeze(-1)  # (C,C,1)
                cur_t = F.conv1d(cur_t, weight=eye, stride=2)
                cur = cur_t.transpose(1, 2)
            pyr.append(cur)
        return pyr

    def forward(
        self,
        x: torch.Tensor,
        pyr_k: List[torch.Tensor],
        pyr_v: List[torch.Tensor],
        scale_idx: int
    ) -> torch.Tensor:
        """
        x: (B, L, C) - 当前尺度 s=scale_idx 的特征
        pyr_k/v: 按尺度从细到粗的列表
        """
        B, L, C = x.shape
        H, D = self.heads, self.dim
        s = scale_idx
        x0 = self.norm(x)

        q  = self.to_q(x0).view(B, L, H, D).transpose(1, 2)  # (B, H, L, D)
        k0 = self.to_k(pyr_k[s]).view(B, -1, H, D).transpose(1, 2)  # (B, H, Ls, D)
        v0 = self.to_v(pyr_v[s]).view(B, -1, H, D).transpose(1, 2)

        # --- A: 同尺度局部窗口（固定宽度的相对偏移广播，避免 stack 尺寸不齐）---
        radius  = max(1, self.local_window // 2)
        time_idx = torch.arange(L, device=x.device)                   # (L,)
        offsets  = torch.arange(-radius, radius + 1, device=x.device) # (W,)
        band_idx = (time_idx.unsqueeze(1) + offsets.unsqueeze(0)).clamp_(0, L - 1)  # (L, W)

        # --- P: 父尺度 ---
        kP = vP = parent_idx = None
        if s + 1 < len(pyr_k):
            kP = self.to_k(pyr_k[s + 1]).view(B, -1, H, D).transpose(1, 2)
            vP = self.to_v(pyr_v[s + 1]).view(B, -1, H, D).transpose(1, 2)
            parent = (time_idx // 2).clamp(max=kP.size(2) - 1)
            pset = [parent]
            for p in range(1, self.parent_neigh + 1):
                pset.append((parent - p).clamp(min=0))
                pset.append((parent + p).clamp(max=kP.size(2) - 1))
            parent_idx = torch.stack(pset, dim=-1)  # (L, 1+2p)

        # --- C: 子尺度 ---
        kC = vC = child_idx = None
        if s - 1 >= 0:
            kC = self.to_k(pyr_k[s - 1]).view(B, -1, H, D).transpose(1, 2)
            vC = self.to_v(pyr_v[s - 1]).view(B, -1, H, D).transpose(1, 2)
            child0 = (time_idx * 2).clamp(max=kC.size(2) - 1)
            child1 = (time_idx * 2 + 1).clamp(max=kC.size(2) - 1)
            child_idx = torch.stack([child0, child1], dim=-1)  # (L, 2)

        # 收集候选 keys/values：A 必有；P/C 视存在追加
        def gather_by_index(kv: torch.Tensor, idx: torch.Tensor, Lq: int) -> torch.Tensor:
            """
            kv: (B,H,Lx,D); idx: (Lq, K) or (Lq,)  -> return (B,H,Lq,K,D)
            """
            B_, H_, Lx, D_ = kv.shape
            if idx.dim() == 1:
                idx = idx.unsqueeze(-1)
            K = idx.size(-1)
            idx_exp = idx.view(1, 1, Lq, K).expand(B_, H_, Lq, K)
            kv_exp  = kv.unsqueeze(2).expand(B_, H_, Lq, Lx, D_)
            gathered = torch.gather(kv_exp, dim=3, index=idx_exp.unsqueeze(-1).expand(B_, H_, Lq, K, D_))
            return gathered

        kA = gather_by_index(k0, band_idx, L)
        vA = gather_by_index(v0, band_idx, L)

        k_list = [kA]; v_list = [vA]
        if kP is not None:
            kP_g = gather_by_index(kP, parent_idx, L)
            vP_g = gather_by_index(vP, parent_idx, L)
            k_list.append(kP_g); v_list.append(vP_g)
        if kC is not None:
            kC_g = gather_by_index(kC, child_idx, L)
            vC_g = gather_by_index(vC, child_idx, L)
            k_list.append(kC_g); v_list.append(vC_g)

        K = torch.cat(k_list, dim=3)  # (B,H,L,Kcand,D)
        V = torch.cat(v_list, dim=3)  # (B,H,L,Kcand,D)

        # 相似度 + 稀疏 TopK
        scores = torch.einsum("bhld,bhlkd->bhlk", q, K) / math.sqrt(D)
        if self.attn_temp != 1.0:
            scores = scores / self.attn_temp

        Kc = scores.size(-1)
        tk = max(1, min(self.topk, Kc, L))
        topv, topi = torch.topk(scores, k=tk, dim=-1)  # (B,H,L,tk)
        V_sel = torch.gather(V, dim=3, index=topi.unsqueeze(-1).expand(*V.shape[:3], tk, D))  # (B,H,L,tk,D)

        attn = F.softmax(topv, dim=-1)
        out = torch.einsum("bhlk,bhlkd->bhld", attn, V_sel)  # (B,H,L,D)
        out = out.transpose(1, 2).contiguous().view(B, L, C)  # (B,L,C)
        out = self.proj(out)
        out = self.dropout(out)
        return out


# ---------------------------
# 蒸馏下采样（Conv/Pool）
# ---------------------------
class DistillDownsample(nn.Module):
    """
    对时间维 L 做 stride=2 下采样；可选通道投影
    """
    def __init__(self, channels: int, stride: int = 2, method: str = "conv",
                 out_channels: Optional[int] = None):
        super().__init__()
        out_channels = out_channels or channels
        self.method = method
        if method == "conv":
            self.dw = nn.Conv1d(channels, channels, kernel_size=3, stride=stride, padding=1,
                                groups=channels, bias=False)
            self.pw = nn.Conv1d(channels, out_channels, kernel_size=1, bias=True)
        elif method == "maxpool":
            self.pool = nn.MaxPool1d(kernel_size=stride, stride=stride)
            self.pw = nn.Conv1d(channels, out_channels, kernel_size=1, bias=True)
        else:
            raise ValueError("distill method must be 'conv' or 'maxpool'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        x1 = x.transpose(1, 2)  # (B, C, L)
        if self.method == "conv":
            x1 = self.pw(self.dw(x1))
        else:
            x1 = self.pw(self.pool(x1))
        x1 = x1.transpose(1, 2)  # (B, L/2, C')
        return x1


# ---------------------------
# Mixer Block + 插入点包装
# ---------------------------
class MixerBlock(nn.Module):
    def __init__(self, channels: int, time_expansion: int, feat_expansion: int, dropout: float):
        super().__init__()
        self.time_mixing = TimeMixing(channels, time_expansion, dropout)
        self.feature_mixing = FeatureMixing(channels, feat_expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time_mixing(x)
        x = self.feature_mixing(x)
        return x


# ---------------------------
# 主干网络：TSMixer + 内插 PTSA + 蒸馏
# ---------------------------
class TSMixerPTSA(nn.Module):
    """
    仅包含特征提取与回归头；不负责优化器与训练循环（由 BaseRULModel 管理）
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        channels: int = 128,
        depth: int = 6,
        time_expansion: int = 4,
        feat_expansion: int = 8,
        dropout: float = 0.05,
        # PTSA
        use_ptsa: bool = True,
        ptsa_every_k: int = 2,
        ptsa_heads: int = 6,
        ptsa_local_window: int = 12,
        ptsa_topk: int = 16,
        ptsa_levels: int = 2,
        ptsa_parent_neigh: int = 1,
        ptsa_dropout: float = 0.0,
        distill_type: str = "conv",
        distill_stride: int = 2,
        reduce_channels: float = 1.0,
        drop_path: float = 0.0,
        out_channels: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.channels = channels
        self.depth = depth
        self.use_ptsa = use_ptsa
        self.ptsa_every_k = max(1, int(ptsa_every_k))
        self.ptsa_levels = max(0, int(ptsa_levels))
        self.distill_type = distill_type
        self.distill_stride = distill_stride
        self.reduce_channels = reduce_channels
        self.ptsa_heads = ptsa_heads
        self.ptsa_local_window = ptsa_local_window
        self.ptsa_topk = ptsa_topk
        self.ptsa_parent_neigh = ptsa_parent_neigh
        self.ptsa_dropout = ptsa_dropout

        # 输入投影到 channels
        self.in_proj = nn.Linear(input_size, channels)

        # 构建层堆叠
        blocks = []
        ptsa_blocks = []
        distills = []

        self.ptsa_after_tm: List[int] = []  # 记录在哪些 TimeMixing 之后插 PTSA（1-based）
        tm_count = 0

        cur_channels = channels
        cur_len = seq_len

        for _ in range(depth):
            # 每层：TimeMixing -> (PTSA? + Distill + FM) : FM
            blocks.append(TimeMixing(cur_channels, time_expansion, dropout))
            tm_count += 1

            insert_ptsa = (self.use_ptsa and (tm_count % self.ptsa_every_k == 0))
            if insert_ptsa:
                self.ptsa_after_tm.append(tm_count)

                ptsa = PTSA(
                    channels=cur_channels,
                    heads=self.ptsa_heads,
                    local_window=self.ptsa_local_window,
                    topk=self.ptsa_topk,
                    parent_neigh=self.ptsa_parent_neigh,
                    dropout=self.ptsa_dropout,
                )
                ptsa_blocks.append(ptsa)

                next_channels = int(round(cur_channels * self.reduce_channels))
                next_channels = max(4, next_channels)
                distill = DistillDownsample(cur_channels, stride=self.distill_stride,
                                            method=self.distill_type, out_channels=next_channels)
                distills.append(distill)

                blocks.append(FeatureMixing(next_channels, feat_expansion, dropout))

                cur_channels = next_channels
                cur_len = max(1, math.ceil(cur_len / self.distill_stride))
            else:
                blocks.append(FeatureMixing(cur_channels, feat_expansion, dropout))

        self.blocks = nn.ModuleList(blocks)
        self.ptsa_blocks = nn.ModuleList(ptsa_blocks)
        self.distills = nn.ModuleList(distills)
        self.tm_count_total = tm_count

        # 回归头
        self.head_norm = nn.LayerNorm(cur_channels)
        self.out = nn.Linear(cur_channels, out_channels)

    @torch.no_grad()
    def _build_pyramid_cache(self, x: torch.Tensor, levels: int, method: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        为 K/V 构建金字塔缓存（此处直接使用 x 本身；若需 Q/K/V 分开可扩展）
        返回: pyr_k, pyr_v  (List of tensors at scales)
        """
        pyr = [x]
        cur = x
        for _ in range(levels):
            if method == "maxpool":
                cur = F.max_pool1d(cur.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
            else:
                C = cur.size(-1)
                cur_t = cur.transpose(1, 2)  # (B,C,L)
                eye = torch.eye(C, device=cur.device).unsqueeze(-1)
                cur_t = F.conv1d(cur_t, weight=eye, stride=2)
                cur = cur_t.transpose(1, 2)
            pyr.append(cur)
        return pyr, pyr  # 简化：K 与 V 用同一组特征构建

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C_in)
        """
        B, L, Cin = x.shape
        x = self.in_proj(x)  # (B,L,C)

        # 初始金字塔：以当前 x 为最细尺度（scale=0），共 self.ptsa_levels 层
        levels = self.ptsa_levels if self.use_ptsa else 0
        pyr_k, pyr_v = self._build_pyramid_cache(x, levels, method="maxpool")

        p_idx = 0          # 指向第几个 PTSA/Distill
        scale_idx = 0      # 当前缓存的尺度索引（以现 x 为 scale=0）
        tm_seen = 0        # 已经过的 TimeMixing 个数（1-based）

        i = 0
        while i < len(self.blocks):
            # 1) TimeMixing
            x = self.blocks[i](x); i += 1
            tm_seen += 1

            insert_here = (self.use_ptsa and (tm_seen in self.ptsa_after_tm))
            if insert_here:
                # 在当前尺度做 PTSA
                ptsa = self.ptsa_blocks[p_idx]
                attn_out = ptsa(x, pyr_k, pyr_v, scale_idx=scale_idx)  # 此处 scale_idx 应为 0（见下重置逻辑）
                x = x + attn_out

                # 下采样蒸馏（改变 L 与 C）
                distill = self.distills[p_idx]
                x = distill(x)
                p_idx += 1

                # 以新的 x 作为最细尺度，重建完整层数的金字塔，并把 scale_idx 归零
                if self.use_ptsa and self.ptsa_levels > 0:
                    pyr_k, pyr_v = self._build_pyramid_cache(x, self.ptsa_levels, method="maxpool")
                else:
                    pyr_k, pyr_v = [x], [x]
                scale_idx = 0  # ✅ 关键修复：重建后从 0 开始

                # 2) 紧接着的 FeatureMixing（构图时已加入且通道已对齐）
                x = self.blocks[i](x); i += 1
            else:
                # 普通路径：2) FeatureMixing
                x = self.blocks[i](x); i += 1

        # 头部：通道 LN + 时间平均
        x = self.head_norm(x)          # (B,L,C)
        x_t = x.mean(dim=1)            # (B,C)
        out = self.out(x_t).squeeze(-1)
        return out


# ---------------------------
# 与 BaseRULModel 对接的外壳
# ---------------------------
class ModelTSMixerPTSA(BaseRULModel):
    """
    继承你的 BaseRULModel：复用 train/eval/early-stopping/AMP/accum 等
    """
    def __init__(self, input_size: int, seq_len: int, out_channels: int = 1, **kwargs):
        super().__init__(input_size, seq_len, out_channels)
        self.hparams = dict(
            channels = kwargs.get("channels", 128),
            depth = kwargs.get("depth", 6),
            time_expansion = kwargs.get("time_expansion", 4),
            feat_expansion = kwargs.get("feat_expansion", 8),
            dropout = kwargs.get("dropout", 0.05),
            use_ptsa = kwargs.get("use_ptsa", True),
            ptsa_every_k = kwargs.get("ptsa_every_k", 2),
            ptsa_heads = kwargs.get("ptsa_heads", 6),
            ptsa_local_window = kwargs.get("ptsa_local_window", 12),
            ptsa_topk = kwargs.get("ptsa_topk", 16),
            ptsa_levels = kwargs.get("ptsa_levels", 2),
            ptsa_parent_neigh = kwargs.get("ptsa_parent_neigh", 1),
            ptsa_dropout = kwargs.get("ptsa_dropout", 0.0),
            distill_type = kwargs.get("distill_type", "conv"),
            distill_stride = kwargs.get("distill_stride", 2),
            reduce_channels = kwargs.get("reduce_channels", 1.0),
            drop_path = kwargs.get("drop_path", 0.0),
        )
        # 训练增强配置（AMP/accum）
        self.accum_steps = int(kwargs.get("accum_steps", 1))
        self.use_amp = bool(kwargs.get("use_amp", True))

        self.model = self.build_model().to(self.device)

    def build_model(self) -> nn.Module:
        m = TSMixerPTSA(
            input_size=self.input_size,
            seq_len=self.seq_len,
            out_channels=self.out_channels,
            **self.hparams
        )
        return m

    def compile(self, learning_rate: float, weight_decay: float, **kwargs):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        sched = kwargs.get("scheduler", "cosine")
        if sched == "cosine":
            T_max = kwargs.get("T_max", kwargs.get("epochs", 180))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif sched == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        else:
            self.scheduler = None
