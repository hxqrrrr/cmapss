import math
import torch
import torch.nn as nn
from typing import Dict, Any
from models.base_model import BaseRULModel

# ----------------- 小工具：nn.Lambda -----------------
class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__(); self.fn = fn
    def forward(self, x):
        return self.fn(x)

# ----------------- 正弦位置编码（时间步） -----------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x):  # x: [B,T,D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.device)

# ----------------- TokenPool-A（学习型查询注意力池化） -----------------
class TokenPoolA(nn.Module):
    """
    将 [B,T,C_in] 压缩为 [B,N_tokens,D]:
      - N_tokens = seq_len // patch
      - 学习查询 Q，对时间维做多头注意力池化
      - 加温度 τ 防注意力塌缩，可配合 dropout
    """
    def __init__(self, in_dim: int, d_model: int, n_tokens: int, n_heads: int = 4,
                 add_positional_encoding: bool = True, attn_dropout: float = 0.0, proj_out: bool = False,
                 temperature: float = 1.5):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_tokens = n_tokens
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.temperature = temperature

        self.q = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)  # [N,D]
        self.key = nn.Linear(in_dim, d_model, bias=False)
        self.val = nn.Linear(in_dim, d_model, bias=False)
        self.pe = SinusoidalPositionalEncoding(d_model) if add_positional_encoding else nn.Identity()
        self.dropout = nn.Dropout(attn_dropout)
        self.out = nn.Linear(d_model, d_model) if proj_out else nn.Identity()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, return_attn: bool = False):  # x: [B,T,C_in]
        B, T, _ = x.shape
        k = self.key(x)               # [B,T,D]
        v = self.val(x)               # [B,T,D]
        kv = self.pe(k)               # 位置编码加在 K

        # 多头重排
        q = self.q.unsqueeze(0).expand(B, -1, -1)                    # [B,N,D]
        q = q.view(B, self.n_tokens, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,N,Dh]
        k = kv.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)             # [B,H,T,Dh]
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)              # [B,H,T,Dh]

        # 注意力（含温度）
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5 * self.temperature)  # [B,H,N,T]
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 聚合
        z = torch.matmul(attn, v)  # [B,H,N,Dh]
        z = z.transpose(1, 2).contiguous().view(B, self.n_tokens, self.d_model)  # [B,N,D]
        z = self.norm(self.out(z))
        return (z, attn.mean(dim=1)) if return_attn else z  # 返回 [B,N,D]；可选 [B,N,T] 注意力

# ----------------- 前馈层 -----------------
def FeedForward(dim_in, dim_hidden, dropout=0.0):
    return nn.Sequential(
        nn.Linear(dim_in, dim_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim_in),
        nn.Dropout(dropout),
    )

# ----------------- Mixer Block（顺序：token-mix → channel-mix） -----------------
class MixerBlock(nn.Module):
    def __init__(self, num_tokens: int, dim: int,
                 token_mlp_dim: int, channel_mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Lambda(lambda x: x.transpose(1, 2)),   # [B,D,N]
            nn.LayerNorm(num_tokens),
            nn.Linear(num_tokens, token_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_mlp_dim, num_tokens),
            nn.Dropout(dropout),
            Lambda(lambda x: x.transpose(1, 2)),   # -> [B,N,D]
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_mlp_dim, dropout=dropout)
        )

    def forward(self, x):  # [B,N,D]
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x

# ----------------- TSMixer 主干 -----------------
class TSMixerBackbone(nn.Module):
    def __init__(self, num_tokens: int, d_model: int, depth: int = 6,
                 token_mlp_dim: int = 256, channel_mlp_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.Sequential(*[
            MixerBlock(num_tokens, d_model, token_mlp_dim, channel_mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # [B,N,D]
        x = self.blocks(x)
        return self.norm(x)

# ----------------- 回归头 -----------------
class RULHead(nn.Module):
    def __init__(self, d_model: int, pool: str = "mean"):
        super().__init__()
        self.pool = pool
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):  # [B,N,D]
        if self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "last":
            x = x[:, -1, :]
        else:
            weights = torch.linspace(0.1, 1.0, steps=x.size(1), device=x.device).unsqueeze(0).unsqueeze(-1)
            x = (x * weights).sum(dim=1) / weights.sum(dim=1)
        return self.fc(x)  # [B,1]

# ----------------- 纯 TokenPool → TSMixer → Head -----------------
class TokenPoolMixerModule(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, patch: int,
                 d_model: int = 128, depth: int = 6,
                 token_mlp_dim: int = 256, channel_mlp_dim: int = 128,
                 dropout: float = 0.1, pool: str = "mean",
                 tokenpool_heads: int = 4, tokenpool_dropout: float = 0.0,
                 tokenpool_temperature: float = 1.5):
        super().__init__()
        self.seq_len = seq_len
        self.patch = patch
        self.num_tokens = max(1, seq_len // patch)

        # 直接用原始输入做 TokenPool（无 CNN、无门控）
        self.tokenpool = TokenPoolA(
            in_dim=in_channels, d_model=d_model, n_tokens=self.num_tokens,
            n_heads=tokenpool_heads, add_positional_encoding=True,
            attn_dropout=tokenpool_dropout, proj_out=False,
            temperature=tokenpool_temperature
        )

        self.mixer = TSMixerBackbone(self.num_tokens, d_model, depth,
                                     token_mlp_dim, channel_mlp_dim, dropout)
        self.head = RULHead(d_model, pool=pool)

    def forward(self, x, return_attn=False):  # x: [B,T,C_in]
        if return_attn:
            z, attn = self.tokenpool(x, return_attn=True)  # [B,N_tokens,D], [B,N,T]
            z = self.mixer(z)             # [B,N_tokens,D]
            y = self.head(z)              # [B,1]
            return y.squeeze(-1), attn    # [B], [B,N,T]
        else:
            z = self.tokenpool(x)         # [B,N_tokens,D]
            z = self.mixer(z)             # [B,N_tokens,D]
            y = self.head(z)              # [B,1]
            return y.squeeze(-1)          # [B]

# =======================  子类：对接 BaseRULModel  =======================
class TokenPoolRULModel(BaseRULModel):
    def __init__(self,
                 input_size: int,     # 特征维（C_in）
                 seq_len: int,        # 窗口长度（T）
                 out_channels: int = 1,
                 patch: int = 5,
                 d_model: int = 128, depth: int = 6,
                 token_mlp_dim: int = 256, channel_mlp_dim: int = 128,
                 dropout: float = 0.1, pool: str = "mean",
                 tokenpool_heads: int = 4, tokenpool_dropout: float = 0.0,
                 tokenpool_temperature: float = 1.5):
        super().__init__(input_size, seq_len, out_channels)
        self.hparams: Dict[str, Any] = dict(
            patch=patch,
            d_model=d_model, depth=depth,
            token_mlp_dim=token_mlp_dim, channel_mlp_dim=channel_mlp_dim,
            dropout=dropout, pool=pool,
            tokenpool_heads=tokenpool_heads,
            tokenpool_dropout=tokenpool_dropout,
            tokenpool_temperature=tokenpool_temperature,
        )
        self.model = self.build_model().to(self.device)

    def build_model(self) -> nn.Module:
        return TokenPoolMixerModule(
            in_channels=self.input_size,
            seq_len=self.seq_len,
            patch=self.hparams["patch"],
            d_model=self.hparams["d_model"],
            depth=self.hparams["depth"],
            token_mlp_dim=self.hparams["token_mlp_dim"],
            channel_mlp_dim=self.hparams["channel_mlp_dim"],
            dropout=self.hparams["dropout"],
            pool=self.hparams["pool"],
            tokenpool_heads=self.hparams["tokenpool_heads"],
            tokenpool_dropout=self.hparams["tokenpool_dropout"],
            tokenpool_temperature=self.hparams["tokenpool_temperature"],
        )

    def compile(self, learning_rate: float = 8e-4, weight_decay: float = 1e-4,
                scheduler: str = "cosine", T_max: int = 100, plateau_patience: int = 5,
                epochs: int = None, steps_per_epoch: int = None):
        # 纯 TokenPool 往往比含 CNN 更稳，lr 可略低
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=plateau_patience, factor=0.5, verbose=False
            )
        elif scheduler == "onecycle" and epochs is not None and steps_per_epoch is not None:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=learning_rate,
                epochs=epochs, steps_per_epoch=steps_per_epoch
            )
        else:
            self.scheduler = None

    # 便捷配置：保持 num_tokens≈10
    @classmethod
    def config_fd001(cls, input_size: int):
        return dict(
            input_size=input_size, seq_len=50, patch=5,   # 50/5=10 tokens
            d_model=128, depth=4, token_mlp_dim=256, channel_mlp_dim=128,
            dropout=0.10, pool="mean",
            tokenpool_heads=4, tokenpool_dropout=0.10, tokenpool_temperature=1.5
        )

    @classmethod
    def config_fd004(cls, input_size: int):
        return dict(
            input_size=input_size, seq_len=80, patch=8,   # 80/8=10 tokens
            d_model=160, depth=6, token_mlp_dim=384, channel_mlp_dim=192,
            dropout=0.12, pool="weighted",
            tokenpool_heads=6, tokenpool_dropout=0.12, tokenpool_temperature=1.5
        )

    def train_model(self, train_loader, val_loader=None,
                   criterion=None, epochs: int = 20,
                   test_dataset_for_last=None, early_stopping_patience=7):
        """
        重写训练方法，添加注意力监控
        """
        import torch.nn as nn
        import torch
        import logging
        
        logger = logging.getLogger(__name__)
        
        if criterion is None:
            criterion = nn.MSELoss()

        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        self.model.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for data, target in train_loader:
                data, target = self.to_device(data, target)
                target = target.view(-1)

                self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(data).view(-1)
                loss = criterion(pred, target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                if self.scheduler is not None:
                    if hasattr(self.scheduler, "step") and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()

                running_loss += loss.item()

            # 训练损失
            train_rmse = (running_loss / max(1, len(train_loader))) ** 0.5
            
            # 验证全局RMSE
            val_rmse_global = float('nan')
            if val_loader is not None:
                val_results = self.evaluate(val_loader)
                val_rmse_global = val_results.get('rmse', float('nan'))
            
            # 最后窗口RMSE和Score + 注意力监控
            val_rmse_last, val_score_last = float('nan'), float('nan')
            attn_metrics = {}
            if test_dataset_for_last is not None:
                val_rmse_last, val_score_last = self.evaluate_last_window(test_dataset_for_last)
                
                # 添加注意力监控
                attn_metrics = self._monitor_attention(test_dataset_for_last)
            
            # 确保模型回到训练模式
            self.model.train()
            
            # 获取学习率
            lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
            
            # 扩展日志格式，包含注意力监控
            self.log_training_metrics_with_attention(epoch, epochs, train_rmse, val_rmse_global, 
                                                    val_rmse_last, val_score_last, lr, attn_metrics)
            
            # ReduceLROnPlateau调度器需要在epoch结束后更新
            if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = val_rmse_last if not torch.isnan(torch.tensor(val_rmse_last)) else val_rmse_global
                if not torch.isnan(torch.tensor(metric)):
                    self.scheduler.step(metric)
            
            # 早停机制
            if early_stopping_patience > 0:
                current_val_loss = val_rmse_last if not torch.isnan(torch.tensor(val_rmse_last)) else val_rmse_global
                if not torch.isnan(torch.tensor(current_val_loss)):
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        patience_counter = 0
                        best_epoch = epoch
                        
                        # 保存最佳指标
                        self.best_metrics = {
                            'train_rmse': train_rmse,
                            'val_rmse_global': val_rmse_global,
                            'val_rmse_last': val_rmse_last,
                            'val_score_last': val_score_last,
                            'epoch': best_epoch,
                            **attn_metrics  # 包含注意力指标
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            logger.info(f"早停触发：验证损失连续{early_stopping_patience}轮未改善")
                            break
            else:
                # 不使用早停时，记录最终指标
                self.best_metrics = {
                    'train_rmse': train_rmse,
                    'val_rmse_global': val_rmse_global,
                    'val_rmse_last': val_rmse_last,
                    'val_score_last': val_score_last,
                    'epoch': epoch,
                    **attn_metrics
                }
        
        return self.best_metrics
    
    def _monitor_attention(self, test_dataset, max_samples=100):
        """
        监控注意力权重的关键指标
        """
        import torch
        
        self.model.eval()
        
        all_peak_rates = []
        all_coverage_rates = []
        all_end_weights = []
        
        sample_count = 0
        
        with torch.no_grad():
            for data, _ in test_dataset.get_dataloader(batch_size=32, shuffle=False):
                if sample_count >= max_samples:
                    break
                
                data = data.to(self.device)
                
                # 获取注意力权重
                try:
                    _, attn = self.model(data, return_attn=True)  # [B,N,T]
                    B, N, T = attn.shape
                    
                    # 1. 注意力峰值率：越接近1越"塌缩"
                    peak_rates = attn.max(dim=-1).values.mean(dim=1)  # [B]
                    all_peak_rates.extend(peak_rates.cpu().tolist())
                    
                    # 2. 时间覆盖度：使用τ = 2/T作为阈值
                    tau = 2.0 / T
                    coverage_per_query = (attn.mean(dim=1) > tau).float().mean(dim=-1)  # [B]
                    all_coverage_rates.extend(coverage_per_query.cpu().tolist())
                    
                    # 3. 末端权重：最后5个时间步的权重
                    k = min(5, T)
                    end_weights = attn[..., -k:].sum(dim=-1).mean(dim=1)  # [B]
                    all_end_weights.extend(end_weights.cpu().tolist())
                    
                    sample_count += B
                    
                except Exception as e:
                    # 如果模型不支持return_attn，跳过监控
                    break
        
        if all_peak_rates:
            return {
                'attn_peak_rate': sum(all_peak_rates) / len(all_peak_rates),
                'attn_coverage': sum(all_coverage_rates) / len(all_coverage_rates),
                'attn_end_weight': sum(all_end_weights) / len(all_end_weights),
                'attn_samples': len(all_peak_rates)
            }
        else:
            return {}
    
    def log_training_metrics_with_attention(self, epoch, total_epochs, train_rmse, val_rmse_global, 
                                           val_rmse_last, val_score_last, lr, attn_metrics):
        """
        扩展的日志记录，包含注意力监控指标
        """
        import logging
        import torch
        
        logger = logging.getLogger(__name__)
        
        # 基础训练日志
        log_parts = [
            f"[Epoch {epoch:3d}/{total_epochs}]",
            f"train_rmse={train_rmse:.2f}"
        ]
        
        if not torch.isnan(torch.tensor(val_rmse_global)):
            log_parts.append(f"val_rmse(global)={val_rmse_global:.2f} cycles")
        
        if not torch.isnan(torch.tensor(val_rmse_last)):
            log_parts.append(f"val_rmse(last)={val_rmse_last:.2f} cycles")
        
        if not torch.isnan(torch.tensor(val_score_last)):
            log_parts.append(f"val_score(last)={val_score_last:.2f}")
        
        log_parts.append(f"lr={lr:.2e}")
        
        # 添加注意力监控指标
        if attn_metrics:
            if 'attn_peak_rate' in attn_metrics:
                log_parts.append(f"attn_peak={attn_metrics['attn_peak_rate']:.3f}")
            if 'attn_coverage' in attn_metrics:
                log_parts.append(f"attn_cov={attn_metrics['attn_coverage']:.3f}")
            if 'attn_end_weight' in attn_metrics:
                log_parts.append(f"attn_end={attn_metrics['attn_end_weight']:.3f}")
        
        logger.info(" | ".join(log_parts))
