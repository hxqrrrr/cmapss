# models/dual_gate_film.py
#轻量实现 ECA（通道）+ 时间门控（轻PTSA）+ FiLM 条件化
import torch
import torch.nn as nn
import torch.nn.functional as F

class ECAGate(nn.Module):
    def __init__(self, channels: int, k_eca: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_eca, padding=k_eca // 2, bias=False)

    def forward(self, x):  # x: [B,T,C]
        a = x.mean(dim=1, keepdim=False)     # [B,C]
        a = self.conv(a.unsqueeze(1)).squeeze(1)  # [B,C]
        return torch.sigmoid(a).unsqueeze(1)      # [B,1,C]

class LiteTemporalGate(nn.Module):
    """深度可分离卷积近似时间注意：对通道取均值后做局部相关建模，再 softmax 到 [B,T,1]"""
    def __init__(self, k_time: int = 11, dilation: int = 1):
        super().__init__()
        pad = (k_time // 2) * dilation
        self.dw = nn.Conv1d(1, 1, kernel_size=k_time, padding=pad, dilation=dilation, bias=False)
        self.pw = nn.Conv1d(1, 1, kernel_size=1, bias=True)

    def forward(self, x):  # x: [B,T,C]
        m = x.mean(dim=2, keepdim=True).transpose(1, 2)  # [B,1,T]
        a = self.pw(self.dw(m)).transpose(1, 2)          # [B,T,1]
        return torch.softmax(a, dim=1)                   # [B,T,1]

class FiLMConditioner(nn.Module):
    """cond_vec->[gamma,beta]（per-channel）"""
    def __init__(self, cond_dim: int, channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * channels)
        )

    def forward(self, cond_vec):  # [B,cond_dim]
        gb = self.net(cond_vec)                   # [B,2C]
        gamma, beta = torch.chunk(gb, 2, dim=-1)  # [B,C],[B,C]
        return gamma, beta

class DualAxisCondGate(nn.Module):
    """
    Y = X ⊙ σ( γ ⊙ A_time ⊕ A_feat + β )
    A_feat:[B,1,C] → broadcast，[B,T,C]
    A_time:[B,T,1] → broadcast，[B,T,C]
    γ,β:[B,C]      → broadcast，[B,1,C]
    """
    def __init__(self, channels: int, cond_dim: int, k_eca: int = 5, k_time: int = 11):
        super().__init__()
        self.eca = ECAGate(channels, k_eca)
        self.tgate = LiteTemporalGate(k_time)
        self.film = FiLMConditioner(cond_dim, channels)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, cond_vec):  # x:[B,T,C]
        x = x.to(torch.float32)  # 避免混精度下逐点数值不稳
        A_feat = self.eca(x)                   # [B,1,C]
        A_time = self.tgate(x)                 # [B,T,1]
        gamma, beta = self.film(cond_vec)      # [B,C],[B,C]
        gate = torch.sigmoid(A_time * gamma.unsqueeze(1) + A_feat + beta.unsqueeze(1))
        return (x * gate).to(dtype=x.dtype)
