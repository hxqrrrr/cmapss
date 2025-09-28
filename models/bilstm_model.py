# models/bilstm_model.py
import torch
import torch.nn as nn
from typing import Optional
from .base_model import BaseRULModel


class _BiLSTMHead(nn.Module):
    """
    输入:  (B, L, C)
    结构:  BiLSTM(lstm_hidden) x num_layers  ->  池化/取末步  ->  MLP -> 标量
    """
    def __init__(self,
                 input_size: int,
                 seq_len: int,
                 lstm_hidden: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True,
                 mlp_hidden: int = 64,
                 pool: str = "last"):   # "last" | "mean" | "max"
        super().__init__()
        self.seq_len = seq_len
        self.pool = pool
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        feat_dim = lstm_hidden * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        B, L, C = x.shape
        assert L == self.seq_len, f"seq_len mismatch: got {L}, expect {self.seq_len}"
        y, _ = self.lstm(x)  # (B, L, H*)
        if self.pool == "mean":
            h = y.mean(dim=1)
        elif self.pool == "max":
            h, _ = y.max(dim=1)
        else:
            h = y[:, -1, :]  # last step
        out = self.head(h)   # (B, 1)
        return out.squeeze(-1)  # (B,)
        

class BiLSTMModel(BaseRULModel):
    """
    基于 BiLSTM 的 RUL 回归模型（继承 BaseRULModel）
    只需实现 build_model() 与 compile()，其余训练/评估直接用基类默认实现
    """
    def __init__(self,
                 input_size: int,
                 seq_len: int,
                 lstm_hidden: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = True,
                 mlp_hidden: int = 64,
                 pool: str = "last"):
        super().__init__(input_size=input_size, seq_len=seq_len, out_channels=1)
        self.hparams = dict(
            lstm_hidden=lstm_hidden,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            mlp_hidden=mlp_hidden,
            pool=pool,
        )
        self.model = self.build_model().to(self.device)

    # ==== 子类必须实现 ====
    def build_model(self) -> nn.Module:
        return _BiLSTMHead(
            input_size=self.input_size,
            seq_len=self.seq_len,
            lstm_hidden=self.hparams["lstm_hidden"],
            num_layers=self.hparams["num_layers"],
            dropout=self.hparams["dropout"],
            bidirectional=self.hparams["bidirectional"],
            mlp_hidden=self.hparams["mlp_hidden"],
            pool=self.hparams["pool"],
        )

    def compile(self,
                learning_rate: float = 3e-4,
                weight_decay: float = 1e-4,
                scheduler: Optional[str] = "plateau",
                epochs: Optional[int] = None,
                steps_per_epoch: Optional[int] = None):
        """
        - 优化器: AdamW
        - 调度器:
            * "plateau": ReduceLROnPlateau(监控验证RMSE，基类已在每个epoch调用 .step(metric))
            * "onecycle": 需给定 epochs 和 steps_per_epoch
            * "cosine": CosineAnnealingLR
        """
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=learning_rate,
                                           weight_decay=weight_decay)

        if scheduler == "onecycle":
            assert epochs is not None and steps_per_epoch is not None, \
                "OneCycleLR 需要 epochs 与 steps_per_epoch"
            from torch.optim.lr_scheduler import OneCycleLR
            self.scheduler = OneCycleLR(self.optimizer,
                                        max_lr=learning_rate,
                                        epochs=epochs,
                                        steps_per_epoch=steps_per_epoch,
                                        pct_start=0.2, div_factor=10, final_div_factor=10)
        elif scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            # 余弦退火到接近 0
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, epochs or 30))
        else:
            # 默认更稳：监控验证指标的 Plateau
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
    
    def log_training_metrics(self, epoch, epochs, train_rmse, val_rmse_global, 
                           val_rmse_last, val_score_last, lr):
        """详细的训练指标日志记录 - BiLSTM模型"""
        import logging
        logger = logging.getLogger(__name__)
        
        # 格式化指标显示
        def format_metric(val):
            if torch.isnan(torch.tensor(val)) or val == float('nan'):
                return "N/A"
            return f"{val:.2f}"
        
        # 详细的日志格式
        metrics_msg = (f"[Epoch {epoch:3d}/{epochs}] "
                      f"train_rmse={format_metric(train_rmse)} | "
                      f"val_rmse(global)={format_metric(val_rmse_global)} cycles | "
                      f"val_rmse(last)={format_metric(val_rmse_last)} cycles | "
                      f"val_score(last)={format_metric(val_score_last)} | "
                      f"lr={lr:.2e}")
        
        logger.info(metrics_msg)