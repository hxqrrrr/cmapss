# models/tokenpool_tsmixer_sga.py
import math
import torch
import torch.nn as nn
from typing import Dict, Any, Literal, Optional, Tuple
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
      - N_tokens = ceil(T / patch)（由外部决定）
      - 学习查询 Q，对时间维做多头注意力池化
      - 温度 τ + dropout 抗注意力塌缩
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

# ----------------- Mixer Block（顺序：token-mix → feature-mix） -----------------
class MixerBlock(nn.Module):
    def __init__(self, num_tokens: int, dim: int,
                 token_mlp_dim: int, channel_mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        # token-mix: 在 token 维（N）上做 MLP
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
        # feature-mix: 在特征维（D）上做 MLP
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_mlp_dim, dropout=dropout)
        )

    def forward(self, x):  # [B,N,D]
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x

# ----------------- SGA（Scalable Global Attention，双轴 squeeze + 融合） -----------------
class SGA2D(nn.Module):
    """
    输入: X [B,N,D]
    - 时间 squeeze:  对 D 取均值得到 [B,N,1]，经小 MLP -> A_time
    - 特征 squeeze:  对 N 取均值得到 [B,1,D]，经小 MLP -> A_feat
    - 融合: A = σ(γ·A_time ⊕ A_feat + β)，对 X 逐元素加权
    """
    def __init__(self, num_tokens: int, d_model: int,
                 time_hidden: int = 24, feat_hidden: int = 24,
                 dropout: float = 0.05, fuse: Literal["add","hadamard"]="add"):
        super().__init__()
        self.fuse = fuse
        self.time_mlp = nn.Sequential(
            nn.LayerNorm(1),
            nn.Linear(1, time_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(time_hidden, 1)
        )
        self.feat_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, feat_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feat_hidden, d_model)
        )
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(0.0))
        self.act = nn.Sigmoid()
        self.num_tokens = num_tokens
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: [B,N,D]
        # 时间轴 squeeze → [B,N,1] → MLP
        a_t = x.mean(dim=-1, keepdim=True)                 # [B,N,1]
        a_t = self.time_mlp(a_t)                           # [B,N,1]
        # 特征轴 squeeze → [B,1,D] → MLP
        a_f = x.mean(dim=1, keepdim=True)                 # [B,1,D]
        a_f = self.feat_mlp(a_f)                          # [B,1,D]
        # 融合
        if self.fuse == "hadamard":
            # 广播逐元素乘 + 偏置
            A = self.act(self.gamma * (a_t * a_f) + self.beta)
        else:
            # 广播相加 + 偏置
            A = self.act(self.gamma * (a_t + a_f) + self.beta)
        return x * A  # [B,N,D]

# ----------------- TSMixer 主干 -----------------
class TSMixerBackbone(nn.Module):
    def __init__(self, num_tokens: int, d_model: int, depth: int = 6,
                 token_mlp_dim: int = 256, channel_mlp_dim: int = 128, dropout: float = 0.1,
                 use_sga: bool = False, sga_time_hidden: int = 24, sga_feat_hidden: int = 24,
                 sga_dropout: float = 0.05, sga_fuse: str = "add", sga_every_k: int = 0):
        super().__init__()
        self.blocks = nn.ModuleList([
            MixerBlock(num_tokens, d_model, token_mlp_dim, channel_mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.use_sga = use_sga
        self.sga_every_k = sga_every_k
        if use_sga:
            if sga_every_k and sga_every_k > 0:
                # 在堆栈内隔 k 层插 SGA
                self.sgas = nn.ModuleList([
                    SGA2D(num_tokens, d_model, sga_time_hidden, sga_feat_hidden, sga_dropout, sga_fuse)
                    for _ in range(math.ceil(depth / sga_every_k))
                ])
            else:
                # 堆栈末尾统一 SGA
                self.sga_tail = SGA2D(num_tokens, d_model, sga_time_hidden, sga_feat_hidden, sga_dropout, sga_fuse)

    def forward(self, x):  # [B,N,D]
        if self.use_sga and (self.sga_every_k and self.sga_every_k > 0):
            sga_idx = 0
            for i, blk in enumerate(self.blocks, start=1):
                x = blk(x)
                if i % self.sga_every_k == 0 and sga_idx < len(self.sgas):
                    x = self.sgas[sga_idx](x)
                    sga_idx += 1
            x = self.norm(x)
            return x
        else:
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            if self.use_sga:
                x = self.sga_tail(x)
            return x

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
            w = torch.linspace(0.1, 1.0, steps=x.size(1), device=x.device).unsqueeze(0).unsqueeze(-1)
            x = (x * w).sum(dim=1) / w.sum(dim=1)
        return self.fc(x)  # [B,1]

# ----------------- 组合模块：TokenPool → TSMixer（可选SGA）→ Head -----------------
class TokenPoolTSMixerSGAModule(nn.Module):
    def __init__(self, in_channels: int, seq_len: int, patch: int,
                 d_model: int = 128, depth: int = 6,
                 token_mlp_dim: int = 256, channel_mlp_dim: int = 128,
                 dropout: float = 0.1, pool: str = "mean",
                 tokenpool_heads: int = 4, tokenpool_dropout: float = 0.0,
                 tokenpool_temperature: float = 1.5,
                 use_sga: bool = False, sga_time_hidden: int = 24, sga_feat_hidden: int = 24,
                 sga_dropout: float = 0.05, sga_fuse: str = "add", sga_every_k: int = 0):
        super().__init__()
        self.seq_len = seq_len
        self.patch = max(1, patch)
        self.num_tokens = max(1, math.ceil(seq_len / self.patch))

        # TokenPool: 直接对原始输入做时间聚合到 N_tokens
        self.tokenpool = TokenPoolA(
            in_dim=in_channels, d_model=d_model, n_tokens=self.num_tokens,
            n_heads=tokenpool_heads, add_positional_encoding=True,
            attn_dropout=tokenpool_dropout, proj_out=False,
            temperature=tokenpool_temperature
        )

        self.mixer = TSMixerBackbone(
            num_tokens=self.num_tokens, d_model=d_model, depth=depth,
            token_mlp_dim=token_mlp_dim, channel_mlp_dim=channel_mlp_dim,
            dropout=dropout,
            use_sga=use_sga, sga_time_hidden=sga_time_hidden, sga_feat_hidden=sga_feat_hidden,
            sga_dropout=sga_dropout, sga_fuse=sga_fuse, sga_every_k=sga_every_k
        )
        self.head = RULHead(d_model, pool=pool)

    def forward(self, x, return_attn: bool = False):
        # x: [B,T,C_in]
        if return_attn:
            z, attn = self.tokenpool(x, return_attn=True)  # [B,N,D], [B,N,T]
            z = self.mixer(z)                               # [B,N,D]
            y = self.head(z)                                # [B,1]
            return y.squeeze(-1), attn
        else:
            z = self.tokenpool(x)                           # [B,N,D]
            z = self.mixer(z)                               # [B,N,D]
            y = self.head(z)                                # [B,1]
            return y.squeeze(-1)

# =======================  子类：对接 BaseRULModel  =======================
class TokenPoolTSMixerSGAModel(BaseRULModel):
    """
    对接训练框架的 RUL 模型：
    - build_model: 返回 nn.Module
    - compile: 配置优化器/调度器
    - train_model: 复用并扩展，支持注意力监控
    """
    def __init__(self,
                 input_size: int,     # 特征维（C_in）
                 seq_len: int,        # 窗口长度（T）
                 out_channels: int = 1,
                 patch: int = 5,
                 d_model: int = 128, depth: int = 6,
                 token_mlp_dim: int = 256, channel_mlp_dim: int = 128,
                 dropout: float = 0.1, pool: str = "mean",
                 tokenpool_heads: int = 4, tokenpool_dropout: float = 0.0,
                 tokenpool_temperature: float = 1.5,
                 use_sga: bool = False, sga_time_hidden: int = 24, sga_feat_hidden: int = 24,
                 sga_dropout: float = 0.05, sga_fuse: str = "add", sga_every_k: int = 0):
        super().__init__(input_size, seq_len, out_channels)
        self.hparams: Dict[str, Any] = dict(
            patch=patch,
            d_model=d_model, depth=depth,
            token_mlp_dim=token_mlp_dim, channel_mlp_dim=channel_mlp_dim,
            dropout=dropout, pool=pool,
            tokenpool_heads=tokenpool_heads,
            tokenpool_dropout=tokenpool_dropout,
            tokenpool_temperature=tokenpool_temperature,
            use_sga=use_sga,
            sga_time_hidden=sga_time_hidden,
            sga_feat_hidden=sga_feat_hidden,
            sga_dropout=sga_dropout,
            sga_fuse=sga_fuse,
            sga_every_k=sga_every_k,
        )
        self.model = self.build_model().to(self.device)

    # --- BaseRULModel 接口 ---
    def build_model(self) -> nn.Module:
        return TokenPoolTSMixerSGAModule(
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
            use_sga=self.hparams["use_sga"],
            sga_time_hidden=self.hparams["sga_time_hidden"],
            sga_feat_hidden=self.hparams["sga_feat_hidden"],
            sga_dropout=self.hparams["sga_dropout"],
            sga_fuse=self.hparams["sga_fuse"],
            sga_every_k=self.hparams["sga_every_k"],
        )

    def compile(self, learning_rate: float = 8e-4, weight_decay: float = 1e-4,
                scheduler: Literal["cosine","plateau","onecycle","none"] = "cosine",
                T_max: int = 100, plateau_patience: int = 5,
                epochs: Optional[int] = None, steps_per_epoch: Optional[int] = None):
        # TokenPool + Mixer 一般较稳，可用略低 lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=plateau_patience, factor=0.5
            )
        elif scheduler == "onecycle" and epochs is not None and steps_per_epoch is not None:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=steps_per_epoch,
                pct_start=0.3, div_factor=25, final_div_factor=100, anneal_strategy='cos'
            )
        else:
            self.scheduler = None

    # ---------- 训练：复用你的日志风格 + 注意力监控 ----------
    def train_model(self, train_loader, val_loader=None,
                   criterion=None, epochs: int = 20,
                   test_dataset_for_last=None, early_stopping_patience=7):
        import torch.nn as nn
        import torch
        import logging
        logger = logging.getLogger(__name__)

        if criterion is None:
            criterion = nn.MSELoss()

        best_val = float('inf')
        patience = 0
        best_epoch = 0
        self.model.train()

        for epoch in range(1, epochs + 1):
            running = 0.0
            for data, target in train_loader:
                data, target = self.to_device(data, target)
                target = target.view(-1)

                self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(data).view(-1)
                loss = criterion(pred, target)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # 非 plateau 调度器步进
                if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()

                running += loss.item()

            train_rmse = (running / max(1, len(train_loader))) ** 0.5

            # 验证
            val_rmse_global = float('nan')
            if val_loader is not None:
                val_results = self.evaluate(val_loader)
                val_rmse_global = val_results.get('rmse', float('nan'))

            # last-window + Score + 注意力监控
            val_rmse_last, val_score_last = float('nan'), float('nan')
            attn_metrics = {}
            if test_dataset_for_last is not None:
                val_rmse_last, val_score_last = self.evaluate_last_window(test_dataset_for_last)
                attn_metrics = self._monitor_attention(test_dataset_for_last)

            self.model.train()  # 回到训练态
            lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0

            # 日志
            self._log_metrics(epoch, epochs, train_rmse, val_rmse_global, val_rmse_last, val_score_last, lr, attn_metrics)

            # plateau 调度器步进
            if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = val_rmse_last if not torch.isnan(torch.tensor(val_rmse_last)) else val_rmse_global
                if not torch.isnan(torch.tensor(metric)):
                    self.scheduler.step(metric)

            # 早停
            if early_stopping_patience > 0:
                current = val_rmse_last if not torch.isnan(torch.tensor(val_rmse_last)) else val_rmse_global
                if not torch.isnan(torch.tensor(current)):
                    if current < best_val:
                        best_val = current
                        patience = 0
                        best_epoch = epoch
                        self.best_metrics = {
                            'train_rmse': train_rmse,
                            'val_rmse_global': val_rmse_global,
                            'val_rmse_last': val_rmse_last,
                            'val_score_last': val_score_last,
                            'epoch': best_epoch,
                            **attn_metrics
                        }
                    else:
                        patience += 1
                        if patience >= early_stopping_patience:
                            logger.info(f"早停触发：验证损失连续{early_stopping_patience}轮未改善")
                            break
            else:
                self.best_metrics = {
                    'train_rmse': train_rmse,
                    'val_rmse_global': val_rmse_global,
                    'val_rmse_last': val_rmse_last,
                    'val_score_last': val_score_last,
                    'epoch': epoch,
                    **attn_metrics
                }

        return getattr(self, "best_metrics", {})

    # ---------- 注意力监控 ----------
    def _monitor_attention(self, test_dataset, max_samples=100):
        self.model.eval()
        all_peak, all_cov, all_end = [], [], []
        samp = 0
        with torch.no_grad():
            for data, _ in test_dataset.get_dataloader(batch_size=32, shuffle=False):
                if samp >= max_samples: break
                data = data.to(self.device)
                try:
                    _, attn = self.model(data, return_attn=True)  # [B,N,T]
                    B, N, T = attn.shape
                    peak = attn.max(dim=-1).values.mean(dim=1)              # [B]
                    tau = 2.0 / T
                    cov = (attn.mean(dim=1) > tau).float().mean(dim=-1)     # [B]
                    k = min(5, T)
                    endw = attn[..., -k:].sum(dim=-1).mean(dim=1)           # [B]
                    all_peak += peak.cpu().tolist()
                    all_cov  += cov.cpu().tolist()
                    all_end  += endw.cpu().tolist()
                    samp += B
                except Exception:
                    break
        if all_peak:
            return {
                'attn_peak_rate': sum(all_peak)/len(all_peak),
                'attn_coverage':  sum(all_cov)/len(all_cov),
                'attn_end_weight':sum(all_end)/len(all_end),
                'attn_samples':    len(all_peak)
            }
        return {}

    def _log_metrics(self, epoch, total_epochs, train_rmse, val_rmse_global, 
                     val_rmse_last, val_score_last, lr, attn_metrics):
        import logging, torch
        logger = logging.getLogger(__name__)
        parts = [f"[Epoch {epoch:3d}/{total_epochs}]",
                 f"train_rmse={train_rmse:.2f}"]
        if not torch.isnan(torch.tensor(val_rmse_global)):
            parts.append(f"val_rmse(global)={val_rmse_global:.2f} cycles")
        if not torch.isnan(torch.tensor(val_rmse_last)):
            parts.append(f"val_rmse(last)={val_rmse_last:.2f} cycles")
        if not torch.isnan(torch.tensor(val_score_last)):
            parts.append(f"val_score(last)={val_score_last:.2f}")
        parts.append(f"lr={lr:.2e}")
        if attn_metrics:
            if 'attn_peak_rate' in attn_metrics: parts.append(f"attn_peak={attn_metrics['attn_peak_rate']:.3f}")
            if 'attn_coverage' in attn_metrics:  parts.append(f"attn_cov={attn_metrics['attn_coverage']:.3f}")
            if 'attn_end_weight' in attn_metrics:parts.append(f"attn_end={attn_metrics['attn_end_weight']:.3f}")
        logger.info(" | ".join(parts))

    # 便捷配置：保持 num_tokens≈10 的建议
    @classmethod
    def config_fd001(cls, input_size: int):
        return dict(
            input_size=input_size, seq_len=50, patch=5,   # 50/5=10 tokens
            d_model=128, depth=4, token_mlp_dim=256, channel_mlp_dim=128,
            dropout=0.10, pool="mean",
            tokenpool_heads=4, tokenpool_dropout=0.10, tokenpool_temperature=1.5,
            use_sga=True, sga_time_hidden=24, sga_feat_hidden=24, sga_dropout=0.06, sga_fuse="add", sga_every_k=0
        )

    @classmethod
    def config_fd002(cls, input_size: int):
        return dict(
            input_size=input_size, seq_len=60, patch=6,   # 60/6=10 tokens
            d_model=144, depth=5, token_mlp_dim=288, channel_mlp_dim=144,
            dropout=0.12, pool="weighted",
            tokenpool_heads=6, tokenpool_dropout=0.10, tokenpool_temperature=1.6,
            use_sga=True, sga_time_hidden=24, sga_feat_hidden=24, sga_dropout=0.06, sga_fuse="add", sga_every_k=0
        )

    @classmethod
    def config_fd004(cls, input_size: int):
        return dict(
            input_size=input_size, seq_len=80, patch=8,   # 80/8=10 tokens
            d_model=160, depth=6, token_mlp_dim=384, channel_mlp_dim=192,
            dropout=0.12, pool="weighted",
            tokenpool_heads=8, tokenpool_dropout=0.12, tokenpool_temperature=1.8,
            use_sga=True, sga_time_hidden=32, sga_feat_hidden=32, sga_dropout=0.08, sga_fuse="add", sga_every_k=0
        )
