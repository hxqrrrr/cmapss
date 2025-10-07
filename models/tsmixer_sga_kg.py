# models/tsmixer_sga_kg.py
import math
import torch
import torch.nn as nn
from typing import Optional, Literal, Dict, Any
from models.base_model import BaseRULModel


# ----------------- 基础：时间/特征两路 MLP Mixer -----------------
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


# ----------------- 轻头盔：知识引导的 SGA（Scalable Global Attention + Prior） -----------------
class KnowledgeGuidedSGA(nn.Module):
    """
    输入:  X \in R^{B×L×C}
    输出:  Y \in R^{B×L×C}  （逐元素加权后的特征）

    步骤（与用户给定方案一致）：
      1) 时间 squeeze：对 X 沿特征维做均值 -> [B,L,1]，经小 MLP 得 A_time
      2) 特征 squeeze：对 X 沿时间维做均值 -> [B,1,C]，经小 MLP 得 A_feat
      3) 先验门控：   A_feat_tilde = λ * A_feat + (1-λ) * g(P)
         - P \in R^{C×C}：传感器/设定量的物理或统计依赖矩阵
         - g(P) 此处实现为：对 P 做行归一化后取行和向量（degree-like），再 Sigmoid 约束到 [0,1]
      4) 融合与加权：  A = σ( γ*(A_time ⊕ A_feat_tilde) + β )，Y = X ⊙ A
         - ⊕ 表示广播相加；A_time 形状 [B,L,1]，A_feat_tilde 形状 [B,1,C]

    设计要点：
      - 不引入大算子，参数量极小；可选 dropout 稍作正则。
      - P 若未提供，默认单位阵（等价于不强加先验）。
      - λ (lambda_prior) 推荐从 0.5 起扫；γ、β可学习以自适应融合强度与偏置。
    """
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        rr_time: int = 4,          # 时间轴小 MLP 的压缩倍数（越大越轻）
        rr_feat: int = 4,          # 特征轴小 MLP 的压缩倍数
        lambda_prior: float = 0.5,  # 先验门控权重 λ
        dropout: float = 0.05,
        use_bias: bool = True,
        eps: float = 1e-6
    ):
        super().__init__()
        self.L = seq_len
        self.C = num_features
        self.lambda_prior = nn.Parameter(torch.tensor(lambda_prior, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.beta  = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.eps = eps

        # 时间轴 squeeze-MLP: [B,L,1] <- mean_C([B,L,C])
        hid_t = max(1, seq_len // rr_time)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, hid_t, bias=use_bias),
            nn.GELU(),
            nn.Linear(hid_t, seq_len, bias=use_bias),
        )

        # 特征轴 squeeze-MLP: [B,1,C] <- mean_L([B,L,C])
        hid_f = max(1, num_features // rr_feat)
        self.feat_mlp = nn.Sequential(
            nn.Linear(num_features, hid_f, bias=use_bias),
            nn.GELU(),
            nn.Linear(hid_f, num_features, bias=use_bias),
        )

        self.dropout = nn.Dropout(dropout)

        # 注册一个默认的单位先验；可通过 set_prior(P) 覆盖
        P_default = torch.eye(num_features, dtype=torch.float32)
        self.register_buffer("P", P_default, persistent=False)

    @torch.no_grad()
    def set_prior(self, P: torch.Tensor):
        """
        设置/更新先验矩阵 P（C×C）。建议非负、对角含自连接。
        可以在外部训练/验证开始前调用；设备迁移由 forward 自动处理。
        """
        assert P.dim() == 2 and P.size(0) == self.C and P.size(1) == self.C, \
            f"P must be [C,C]={self.C}"
        self.P = P.detach().float()

    def _g_prior(self) -> torch.Tensor:
        """
        g(P): 行归一化 -> 按行求和得到度向量，再 Sigmoid 约束到 [0,1]
        返回形状 [1,1,C]，便于与 A_feat 做广播融合。
        """
        # 防止除零
        row_sum = self.P.sum(dim=1, keepdim=True).clamp_min(self.eps)
        P_norm = self.P / row_sum
        deg = P_norm.sum(dim=0)               # [C]
        prior_vec = torch.sigmoid(deg).view(1, 1, self.C)
        return prior_vec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,C]
        B, L, C = x.shape

        # 1) 时间轴 squeeze
        x_time = x.mean(dim=2)                        # [B,L]
        A_time = self.time_mlp(x_time).unsqueeze(-1)  # [B,L,1]

        # 2) 特征轴 squeeze
        x_feat = x.mean(dim=1)                        # [B,C]
        A_feat = self.feat_mlp(x_feat).unsqueeze(1)   # [B,1,C]

        # 3) 先验门控
        prior_vec = self._g_prior().to(x.device)      # [1,1,C]
        lam = torch.clamp(self.lambda_prior, 0.0, 1.0)
        A_feat_tilde = lam * A_feat + (1.0 - lam) * prior_vec  # [B,1,C]

        # 4) 融合与加权
        A = torch.sigmoid(self.gamma * (A_time + A_feat_tilde) + self.beta)  # [B,L,C] via broadcast
        A = self.dropout(A)
        return x * A


class MixerBlockWithSGA(nn.Module):
    """标准 TSMixer Block + 知识引导 SGA 轻头盔"""
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        time_expansion: int = 4,
        feat_expansion: int = 4,
        dropout: float = 0.1,
        # SGA 相关
        rr_time: int = 4,
        rr_feat: int = 4,
        lambda_prior: float = 0.5,
        sga_dropout: float = 0.05,
    ):
        super().__init__()
        self.time = TimeMixing(seq_len, time_expansion, dropout)
        self.feat = FeatureMixing(num_features, feat_expansion, dropout)
        self.sga  = KnowledgeGuidedSGA(
            seq_len, num_features,
            rr_time=rr_time, rr_feat=rr_feat,
            lambda_prior=lambda_prior, dropout=sga_dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time(x)
        x = self.feat(x)
        x = self.sga(x)
        return x


# ----------------- 主干：堆叠 Mixer+SGA，线性回归头 -----------------
class TSMixerSGAKGNet(nn.Module):
    """
    输入:  (B, L, C)
    输出:  (B,)  —— 经过 k 层 [TimeMix -> FeatureMix -> SGA(KG)]，再时间池化 + 线性
    """
    def __init__(
        self,
        input_length: int,
        num_features: int,
        num_layers: int = 5,
        time_expansion: int = 4,
        feat_expansion: int = 4,
        dropout: float = 0.1,
        # SGA
        rr_time: int = 4,
        rr_feat: int = 4,
        lambda_prior: float = 0.5,
        sga_dropout: float = 0.05,
        pool: Literal["mean", "weighted", "last"] = "weighted"
    ):
        super().__init__()
        self.L = input_length
        self.C = num_features
        self.pool = pool

        self.blocks = nn.ModuleList([
            MixerBlockWithSGA(
                seq_len=input_length, num_features=num_features,
                time_expansion=time_expansion, feat_expansion=feat_expansion,
                dropout=dropout,
                rr_time=rr_time, rr_feat=rr_feat,
                lambda_prior=lambda_prior, sga_dropout=sga_dropout
            )
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(num_features, 1)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    @torch.no_grad()
    def set_prior(self, P: torch.Tensor):
        """将先验矩阵 P 下发到每个 SGA 轻头盔"""
        for blk in self.blocks:
            blk.sga.set_prior(P)

    def _temporal_pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,C]
        if self.pool == "mean":
            return x.mean(dim=1)
        if self.pool == "last":
            return x[:, -1, :]
        # weighted: 末端加权，强调临近失效
        weights = torch.linspace(0.1, 1.0, steps=x.size(1), device=x.device).view(1, -1, 1)
        return (x * weights).sum(dim=1) / weights.sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,C]
        for blk in self.blocks:
            x = blk(x)
        h = self._temporal_pool(x)      # [B,C]
        y = self.head(h).squeeze(-1)    # [B]
        return y


# ----------------- RUL 封装：与 BaseRULModel 对接 -----------------
class TSMixerSGAKGRULModel(BaseRULModel):
    """
    与仓库现有训练/验证循环兼容：
      - build_model(): 返回 nn.Module
      - compile(): 配优化器/调度器
      - 可选 set_prior(P)：在训练前/阶段性更新知识先验
    关键超参：
      * rr_time / rr_feat: SGA 两条轴的小 MLP 压缩比
      * lambda_prior: 先验门控 λ，建议从 0.5 起扫
      * pool: 'weighted' 建议用于 FD002/FD004
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        out_channels: int = 1,
        *,
        # Mixer
        num_layers: int = 5,
        time_expansion: int = 6,
        feat_expansion: int = 5,
        dropout: float = 0.12,
        # SGA
        rr_time: int = 4,
        rr_feat: int = 4,
        lambda_prior: float = 0.5,
        sga_dropout: float = 0.08,
        pool: Literal["mean", "weighted", "last"] = "weighted",
    ):
        super().__init__(input_size=input_size, seq_len=seq_len, out_channels=out_channels)
        self.hparams: Dict[str, Any] = dict(
            num_layers=num_layers,
            time_expansion=time_expansion,
            feat_expansion=feat_expansion,
            dropout=dropout,
            rr_time=rr_time,
            rr_feat=rr_feat,
            lambda_prior=lambda_prior,
            sga_dropout=sga_dropout,
            pool=pool
        )
        self.model = self.build_model().to(self.device)

    # --- 构建 nn.Module ---
    def build_model(self) -> nn.Module:
        hp = self.hparams
        return TSMixerSGAKGNet(
            input_length=self.seq_len,
            num_features=self.input_size,
            num_layers=hp["num_layers"],
            time_expansion=hp["time_expansion"],
            feat_expansion=hp["feat_expansion"],
            dropout=hp["dropout"],
            rr_time=hp["rr_time"],
            rr_feat=hp["rr_feat"],
            lambda_prior=hp["lambda_prior"],
            sga_dropout=hp["sga_dropout"],
            pool=hp["pool"],
        )

    # --- 优化器与调度器 ---
    def compile(
        self,
        learning_rate: float = 8e-4,
        weight_decay: float = 2e-4,
        *,
        scheduler: Literal["cosine", "plateau", "onecycle", "none"] = "cosine",
        epochs: int = 60,
        steps_per_epoch: Optional[int] = None,
        plateau_patience: int = 6
    ):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=plateau_patience, factor=0.5
            )
        elif scheduler == "onecycle" and steps_per_epoch is not None:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=steps_per_epoch
            )
        else:
            self.scheduler = None

    # --- 外部接口：设置知识先验矩阵 P ---
    @torch.no_grad()
    def set_prior(self, P: torch.Tensor):
        """
        P: [C,C] 传感器/设定量依赖矩阵（非负）。可来自统计（互信息/相关/共现）或专家知识。
        训练前或每个 epoch 开始时调用：
            model.set_prior(P)
        """
        if hasattr(self.model, "set_prior"):
            self.model.set_prior(P.to(self.device))

    # --- 便捷配置 ---
    @classmethod
    def config_fd002(cls, input_size: int):
        """FD002 多工况推荐起点："""
        return dict(
            input_size=input_size, seq_len=80,
            num_layers=6, time_expansion=6, feat_expansion=5, dropout=0.12,
            rr_time=4, rr_feat=4, lambda_prior=0.5, sga_dropout=0.08,
            pool="weighted"
        )

    @classmethod
    def config_fd004(cls, input_size: int):
        """FD004 多工况/多故障推荐起点："""
        return dict(
            input_size=input_size, seq_len=100,
            num_layers=6, time_expansion=6, feat_expansion=6, dropout=0.15,
            rr_time=4, rr_feat=4, lambda_prior=0.5, sga_dropout=0.10,
            pool="weighted"
        )

    # --- 训练日志（与既有格式一致） ---
    def log_training_metrics(self, epoch, epochs, train_rmse, val_rmse_global,
                             val_rmse_last, val_score_last, lr):
        import logging
        logger = logging.getLogger(__name__)
        def fmt(x):
            try:
                return f"{float(x):.2f}"
            except Exception:
                return "N/A"
        msg = (f"[Epoch {epoch:3d}/{epochs}] "
               f"train_rmse={fmt(train_rmse)} | "
               f"val_rmse(global)={fmt(val_rmse_global)} cycles | "
               f"val_rmse(last)={fmt(val_rmse_last)} cycles | "
               f"val_score(last)={fmt(val_score_last)} | "
               f"lr={lr:.2e}")
        logger.info(msg)
