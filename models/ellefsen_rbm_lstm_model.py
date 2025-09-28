# models/ellefsen_rbm_lstm_model.py
# -*- coding: utf-8 -*-
"""
Ellefsen 等 (Reliability Engineering & System Safety 2019) 半监督框架的 PyTorch 版本：
- L1: 高斯-伯努利 RBM（仅用于无监督预训练，提供权重初始化）
- L2-L3: 两层 LSTM（vanilla LSTM）
- L4: 全连接映射 (FNN)
- L5: 输出层 (time-distributed -> 这里用池化后做标量回归)

实现为 BaseRULModel 子类：
- 只需实现 build_model() 与 compile()；训练/评估复用基类默认实现
参考与配置取自论文 §3.1-3.3、§4（RBM 预训练、两层 LSTM、FNN + 输出、早停与调参思想）。"""

from typing import Optional, Literal, Dict, Any
import torch
import torch.nn as nn

from models.base_model import BaseRULModel  # 确保路径正确

# --------------------------- RBM (Gaussian-Bernoulli, CD-1) ---------------------------

class GaussianBernoulliRBM(nn.Module):
    """
    简化版高斯-伯努利 RBM:
    - 可用于对第一层特征做无监督预训练（对齐论文先用 RBM 抽取退化相关特征，然后再监督微调）:contentReference[oaicite:1]{index=1}
    说明：
    - 可见层 v 连续（高斯），隐层 h 伯努利
    - 这里采用标准化后的输入（z-score/minmax 后等价于单位方差近似），sigma 固定为1 简化实现
    - 对下游 Linear 的初始化：W^T -> Linear.weight, h_bias -> Linear.bias
    """
    def __init__(self, n_visible: int, n_hidden: int):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # 参数：W ∈ R^{V×H}, v_bias ∈ R^V, h_bias ∈ R^H
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    @staticmethod
    def _sigmoid(x):  # 稍快
        return torch.sigmoid(x)

    def sample_h_given_v(self, v):
        # p(h=1|v) = sigmoid(v W + h_bias)    (sigma=1 假设)
        probs = self._sigmoid(v @ self.W + self.h_bias)
        return probs, torch.bernoulli(probs)

    def sample_v_given_h(self, h):
        # v|h ~ N( v_bias + h W^T, I )
        mean = self.v_bias + h @ self.W.t()
        # 采样高斯；训练 RBM 时可用噪声，权重初始化时也可用 mean 代替采样值
        v_sample = mean + torch.randn_like(mean) * 0.1  # 减小噪声避免数值不稳定
        return mean, v_sample

    def contrastive_divergence(self, v0, k: int = 1):
        # CD-k：k 通常取 1 即可
        v = v0
        for _ in range(k):
            h_prob, h = self.sample_h_given_v(v)
            v_mean, v = self.sample_v_given_h(h)

        # 更新梯度所需期望
        h0_prob, _ = self.sample_h_given_v(v0)   # 正相
        hk_prob, _ = self.sample_h_given_v(v)    # 负相

        # 返回统计量，便于外部优化器更新
        # 注意：这里返回 v_mean 而不是采样值，避免随机性影响梯度计算
        return {
            "v0": v0, "vk": v_mean,  # 使用重构均值而不是采样值
            "h0_prob": h0_prob, "hk_prob": hk_prob
        }

    def forward(self, v):
        # 返回隐层概率（可作为无监督特征）
        h_prob, _ = self.sample_h_given_v(v)
        return h_prob

    @torch.no_grad()
    def init_linear_from_rbm(self, linear: nn.Linear):
        """
        用 RBM 权重初始化下游 Linear（对齐论文：预训练后丢弃重构分支，仅用权重做初始化）:contentReference[oaicite:2]{index=2}
        """
        assert linear.in_features == self.n_visible and linear.out_features == self.n_hidden
        linear.weight.copy_(self.W.t())
        if linear.bias is not None:
            linear.bias.copy_(self.h_bias)


# --------------------------- 主干网络：RBM 初始化 + LSTM 堆叠 + 全连接 ---------------------------

class RBMLSTMRegressor(nn.Module):
    """
    结构对齐论文 Fig.2：RBM(作初始化) -> LSTM -> LSTM -> FNN -> 输出层（标量 RUL）
    训练时输入：(B, L, C)，先经 Linear 投影到 d_model，再堆叠 LSTM，最后池化 + 全连接得到回归输出
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        rbm_hidden: int = 64,          # 论文表5常见 32/64/128 备选
        lstm_hidden1: int = 64,         # FD001/FD003 常见 64，FD002/FD004 常见 128/32 组合 :contentReference[oaicite:3]{index=3}
        lstm_hidden2: int = 64,
        ff_hidden: int = 8,             # 表5中 L4 常为 8 单元
        dropout_lstm: float = 0.5,      # 论文使用 dropout（非循环连接）作为正则 :contentReference[oaicite:4]{index=4}
        pool: Literal["last", "mean"] = "last"
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pool = pool

        # L1: 线性层（会被 RBM 预训练权重初始化）
        self.in_proj = nn.Linear(input_size, rbm_hidden)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)

        # L2-L3: 两层 LSTM（vanilla）
        # 使用 batch_first=True 接收 (B,L,feat)
        self.lstm1 = nn.LSTM(input_size=rbm_hidden, hidden_size=lstm_hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout_lstm)
        self.lstm2 = nn.LSTM(input_size=lstm_hidden1, hidden_size=lstm_hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout_lstm)

        # L4: FNN (逐时刻特征->隐藏)
        self.ff = nn.Linear(lstm_hidden2, ff_hidden)
        self.act_ff = nn.Sigmoid()  # 论文观察 sigmoid I/O 更佳的设置之一 :contentReference[oaicite:5]{index=5}

        # L5: 输出层（time-distributed -> 池化后到标量）
        self.out = nn.Linear(ff_hidden, 1)

        # 归一化
        self.norm = nn.LayerNorm(rbm_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, C)
        return: (B,)  —— 单步标量 RUL 回归（窗口末步或池化表征）
        """
        B, L, C = x.shape
        # L1
        h = self.in_proj(x)            # (B,L,H1)
        h = self.norm(h)

        # L2-L3
        h, _ = self.lstm1(h)           # (B,L,Hl1)
        h = self.drop1(h)
        h, _ = self.lstm2(h)           # (B,L,Hl2)
        h = self.drop2(h)

        # 池化（“last”=取最后时刻；或 mean）
        if self.pool == "last":
            h_last = h[:, -1, :]       # (B,Hl2)
        else:
            h_last = h.mean(dim=1)     # (B,Hl2)

        # L4-L5
        z = self.act_ff(self.ff(h_last))  # (B,Hf)
        y = self.out(z).squeeze(-1)       # (B,)
        return y


# --------------------------- 包装为 BaseRULModel 子类 ---------------------------

class EllefsenRBMLSTMModel(BaseRULModel):
    """
    基于 Ellefsen 等的半监督结构的实现：
    - 通过 .pretrain_rbm() 可选地先做 RBM 无监督预训练，然后初始化第一层 Linear
    - build_model(): 构建 RBM+LSTM+FNN 主干（RBM 对训练时仅用于初始化，不在 forward 图中）
    - compile(): 设定优化器与可选调度器
    - 训练/评估：直接使用 BaseRULModel 的默认实现（含早停、last-window 评估等）
    """

    def __init__(self,
                 input_size: int,
                 seq_len: int,
                 out_channels: int = 1,
                 rbm_hidden: int = 64,
                 lstm_hidden1: int = 64,
                 lstm_hidden2: int = 64,
                 ff_hidden: int = 8,
                 dropout_lstm: float = 0.5,
                 pool: Literal["last", "mean"] = "last"):
        super().__init__(input_size, seq_len, out_channels)
        self.hparams = dict(rbm_hidden=rbm_hidden,
                            lstm_hidden1=lstm_hidden1,
                            lstm_hidden2=lstm_hidden2,
                            ff_hidden=ff_hidden,
                            dropout_lstm=dropout_lstm,
                            pool=pool)
        self.model = self.build_model().to(self.device)

        # RBM（仅在需要预训练时使用）
        self._rbm: Optional[GaussianBernoulliRBM] = GaussianBernoulliRBM(
            n_visible=input_size, n_hidden=rbm_hidden
        )

    # --------- BaseRULModel 抽象方法实现 ---------

    def build_model(self) -> nn.Module:
        return RBMLSTMRegressor(
            input_size=self.input_size,
            seq_len=self.seq_len,
            rbm_hidden=self.hparams["rbm_hidden"],
            lstm_hidden1=self.hparams["lstm_hidden1"],
            lstm_hidden2=self.hparams["lstm_hidden2"],
            ff_hidden=self.hparams["ff_hidden"],
            dropout_lstm=self.hparams["dropout_lstm"],
            pool=self.hparams["pool"]
        )

    def compile(self,
                learning_rate: float = 3e-4,
                weight_decay: float = 1e-4,
                scheduler: Optional[str] = "plateau",
                epochs: Optional[int] = None,
                steps_per_epoch: Optional[int] = None):
        """
        优化器/调度器：
        - AdamW：论文采用 Adam 类自适应方法效果较好 :contentReference[oaicite:6]{index=6}
        - 调度器支持：onecycle / plateau / cosine
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if scheduler == "onecycle" and epochs is not None and steps_per_epoch is not None:
            from torch.optim.lr_scheduler import OneCycleLR
            self.scheduler = OneCycleLR(self.optimizer, max_lr=learning_rate,
                                        epochs=epochs, steps_per_epoch=steps_per_epoch,
                                        pct_start=0.2, div_factor=10, final_div_factor=10)
        elif scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, epochs or 10))
        elif scheduler == "plateau":
            # 基类会在每个 epoch 结束时用 val 指标调用 step()
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                        factor=0.5, patience=3)
        else:
            self.scheduler = None

    # --------- 可选：RBM 预训练（半监督） ---------

    def pretrain_rbm(self,
                     unlabeled_loader,
                     epochs: int = 1,
                     lr_rbm: float = 1e-2,
                     cd_k: int = 1,
                     device: Optional[torch.device] = None):
        """
        用无标签数据（输入窗口 X，忽略 y）对 RBM 做对比散度预训练，然后把权重拷到第一层 Linear。
        备注：
        - 论文中 RBM 仅作初始权重的无监督学习（预训练1~若干epoch），随后在监督阶段整体微调 :contentReference[oaicite:7]{index=7}
        """
        if device is None:
            device = self.device
        rbm = self._rbm.to(device)
        opt = torch.optim.SGD(rbm.parameters(), lr=lr_rbm, momentum=0.0)

        rbm.train()
        for ep in range(1, epochs + 1):
            total_loss = 0.0
            n = 0
            for batch_idx, (xb, _yb) in enumerate(unlabeled_loader):
                # xb: (B,L,C) —— 使用窗口末步而不是均值，保留更多时序信息
                xb = xb.to(device, non_blocking=True)
                v0 = xb[:, -1, :]  # 取窗口最后一个时间步 (B,C)
                stats = rbm.contrastive_divergence(v0, k=cd_k)

                # 近似 free-energy 损失的替代：重构误差 (MSE)
                vk = stats["vk"]
                loss = torch.mean((v0 - vk) ** 2)

                opt.zero_grad()
                loss.backward()
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(rbm.parameters(), max_norm=1.0)
                opt.step()

                total_loss += float(loss.item()) * v0.size(0)
                n += v0.size(0)
                
                # 每100个batch打印一次进度
                if batch_idx % 100 == 0:
                    print(f"[RBM Pretrain] epoch {ep}/{epochs} | batch {batch_idx} | loss={loss.item():.6f}")

            avg = total_loss / max(1, n)
            print(f"[RBM Pretrain] epoch {ep}/{epochs} | avg_recon-MSE={avg:.6f}")

        # 用 RBM 权重初始化第一层 Linear
        if isinstance(self.model, RBMLSTMRegressor):
            rbm.init_linear_from_rbm(self.model.in_proj)
            print("[RBM] Initialized first Linear layer from pretrained RBM weights.")

    # （其余训练/评估逻辑，完全复用 BaseRULModel）
