import torch
import torch.nn as nn
import logging
from models.base_model import BaseRULModel

# ========== 模型内部模块 ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, L, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerRegressor(nn.Module):
    def __init__(self, input_size: int, seq_len: int,
                 d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 256,
                 dropout: float = 0.1, pool: str = "cls"):
        super().__init__()
        self.seq_len = seq_len
        self.pool = pool

        self.in_proj = nn.Linear(input_size, d_model)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len+1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        nn.init.xavier_uniform_(self.cls_token)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        assert L == self.seq_len, f"Expected {self.seq_len}, got {L}"

        x = self.in_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.norm(x)

        if self.pool == "cls":
            h = x[:, 0]
        elif self.pool == "last":
            h = x[:, -1]
        else:
            h = x[:, 1:].mean(dim=1)

        return self.head(h).squeeze(-1)


# ========== 子类 ==========
class TransformerModel(BaseRULModel):
    def __init__(self, input_size: int, seq_len: int,
                 d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 256,
                 dropout: float = 0.1, pool: str = "cls"):
        super().__init__(input_size, seq_len, out_channels=1)
        self.model = self.build_model(
            input_size, seq_len, d_model, nhead,
            num_layers, dim_feedforward, dropout, pool
        ).to(self.device)

    def build_model(self, input_size, seq_len, d_model, nhead,
                    num_layers, dim_feedforward, dropout, pool) -> nn.Module:
        return TransformerRegressor(
            input_size, seq_len, d_model, nhead,
            num_layers, dim_feedforward, dropout, pool
        )

    def compile(self, learning_rate: float, weight_decay: float, **kwargs):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=learning_rate,
                                           weight_decay=weight_decay)
        sched = kwargs.get("scheduler", None)
        if sched == "onecycle":
            from torch.optim.lr_scheduler import OneCycleLR
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=learning_rate,
                epochs=kwargs["epochs"],
                steps_per_epoch=kwargs["steps_per_epoch"],
                pct_start=0.2, div_factor=10, final_div_factor=10
            )
        elif sched == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=3)
        elif sched == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=kwargs.get("epochs", 30))
        else:
            self.scheduler = None

    def log_training_metrics(self, epoch, epochs, train_rmse, val_rmse_global, 
                           val_rmse_last, val_score_last, lr):
        """Log training metrics using the logging system"""
        logger = logging.getLogger(__name__)
        
        # Format the metrics message
        metrics_msg = (f"[Epoch {epoch}/{epochs}] "
                      f"Train RMSE: {train_rmse:.3f} | "
                      f"Val RMSE(global): {val_rmse_global:.3f} | "
                      f"Val RMSE(last): {val_rmse_last:.3f} | "
                      f"Score(last): {val_score_last:.1f} | "
                      f"LR: {lr:.2e}")
        
        logger.info(metrics_msg)
