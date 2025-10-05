# models/base_model.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np


class BaseRULModel(ABC):
    """
    抽象基类：统一RUL预测模型的接口
    所有子类必须实现这些方法，保证功能一致
    """

    def __init__(self, input_size: int, seq_len: int, out_channels: int = 1):
        self.input_size = input_size
        self.seq_len = seq_len
        self.out_channels = out_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.best_metrics: Dict[str, float] = {}  # 存储最佳训练指标

    # --------- 必须由子类实现的方法 ---------
    @abstractmethod
    def build_model(self) -> nn.Module:
        pass

    @abstractmethod
    def compile(self, learning_rate: float, weight_decay: float, **kwargs):
        pass

    # ======================== 强化版训练：AMP + 梯度累积 ========================
    def train_model(self, train_loader, val_loader=None,
                    criterion: Optional[Any] = None, epochs: int = 20,
                    test_dataset_for_last=None, early_stopping_patience=7):
        """
        默认训练流程（已增强：AMP + 梯度累积 + 稳健的 scheduler.step 时机）
        """
        import math
        if criterion is None:
            criterion = nn.MSELoss(reduction="mean")

        # 配置：AMP & 梯度累积（从子类/外部 hint 读取，未设置则使用默认）
        accum_steps = max(1, int(getattr(self, "accum_steps", 1)))
        use_amp = bool(getattr(self, "use_amp", True) and torch.cuda.is_available())
        amp_dtype = (
            torch.bfloat16
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        # 新 API
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # 调度器类型判定
        reduce_on_plateau = isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )
        per_step_classes = (
            torch.optim.lr_scheduler.OneCycleLR,
            torch.optim.lr_scheduler.SequentialLR,
        )
        step_sched_each_step = isinstance(self.scheduler, per_step_classes)

        # 早停
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        self.model.train()
        if self.optimizer is None:
            raise RuntimeError("请先在 model.compile(...) 中设置 optimizer")

        for epoch in range(1, epochs + 1):
            running_mse, seen = 0.0, 0
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            for it, (data, target) in enumerate(train_loader, start=1):
                data, target = self.to_device(data, target)
                target = target.view(-1)

                # 新 API
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    pred = self.model(data).view(-1)
                    loss = criterion(pred, target) / accum_steps

                with torch.no_grad():
                    running_mse += torch.sum((pred.detach() - target.detach()) ** 2).item()
                    seen += target.numel()

                scaler.scale(loss).backward()

                if (it % accum_steps == 0) or (it == len(train_loader)):
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                if step_sched_each_step and hasattr(self.scheduler, "step"):
                    self.scheduler.step()

            train_rmse = math.sqrt(running_mse / max(1, seen))

            # 验证
            val_rmse_global = float('nan')
            if val_loader is not None:
                val_results = self.evaluate(val_loader)
                val_rmse_global = val_results.get('rmse', float('nan'))

            val_rmse_last, val_score_last = float('nan'), float('nan')
            if test_dataset_for_last is not None:
                val_rmse_last, val_score_last = self.evaluate_last_window(test_dataset_for_last)

            self.model.train()

            lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
            self.log_training_metrics(
                epoch, epochs, train_rmse,
                val_rmse_global, val_rmse_last, val_score_last, lr
            )

            # epoch 调度
            if self.scheduler is not None:
                if reduce_on_plateau:
                    metric = val_rmse_last if not torch.isnan(torch.tensor(val_rmse_last)) else val_rmse_global
                    if not torch.isnan(torch.tensor(metric)):
                        self.scheduler.step(metric)
                else:
                    if (not step_sched_each_step) and hasattr(self.scheduler, "step"):
                        self.scheduler.step()

            # 早停
            if early_stopping_patience > 0:
                current_val = val_rmse_last if not torch.isnan(torch.tensor(val_rmse_last)) else val_rmse_global
                if not torch.isnan(torch.tensor(current_val)):
                    if current_val < best_val_loss:
                        best_val_loss = current_val
                        patience_counter = 0
                        best_epoch = epoch
                        self.best_metrics = {
                            'train_rmse': train_rmse,
                            'val_rmse_global': val_rmse_global,
                            'val_rmse_last': val_rmse_last,
                            'val_score_last': val_score_last,
                            'best_epoch': epoch,
                            'learning_rate': lr
                        }
                        print(f"📈 New best validation RMSE: {best_val_loss:.4f}")
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f" Early stopping triggered after {epoch} epochs (patience: {early_stopping_patience})")
                            print(f" Best validation RMSE: {best_val_loss:.4f}")
                            break

        if early_stopping_patience > 0 and best_val_loss != float('inf'):
            return {
                "best_val_rmse": best_val_loss,
                "early_stopped": True,
                "best_epoch": best_epoch
            }
        else:
            final_results = {}
            if val_loader is not None:
                final_val_results = self.evaluate(val_loader)
                final_results["final_val_rmse"] = final_val_results.get('rmse', float('nan'))
                self.best_metrics = {
                    'train_rmse': train_rmse,
                    'val_rmse_global': final_val_results.get('rmse', float('nan')),
                    'val_rmse_last': val_rmse_last,
                    'val_score_last': val_score_last,
                    'final_epoch': epochs,
                    'learning_rate': lr
                }
            final_results["early_stopped"] = False
            return final_results

    # ======================== 评估：推理也用 AMP 提速 ========================
    def evaluate(self, test_loader) -> Dict[str, float]:
        self.model.eval()
        mse, n = 0.0, 0
        use_amp = bool(getattr(self, "use_amp", True) and torch.cuda.is_available())
        amp_dtype = (
            torch.bfloat16
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                for data, target in test_loader:
                    data, target = self.to_device(data, target)
                    target = target.view(-1)
                    pred = self.model(data).view(-1)
                    mse += torch.sum((pred - target) ** 2).item()
                    n += target.numel()
        rmse = (mse / max(1, n)) ** 0.5
        return {"rmse": rmse}

    # --------- 通用工具函数 ---------
    def to_device(self, data, target):
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        return data, target

    def log_training_metrics(self, epoch: int, total_epochs: int,
                             train_rmse: float, val_rmse_global: float,
                             val_rmse_last: float, val_score_last: float, lr: float):
        import logging
        logger = logging.getLogger(__name__)

        def format_metric(val):
            if np.isnan(val) or val == float('nan'):
                return "N/A"
            return f"{val:.2f}"

        msg = (f"[Epoch {epoch:3d}/{total_epochs}] "
               f"train_rmse={format_metric(train_rmse)} | "
               f"val_rmse(global)={format_metric(val_rmse_global)} cycles | "
               f"val_rmse(last)={format_metric(val_rmse_last)} cycles | "
               f"val_score(last)={format_metric(val_score_last)} | "
               f"lr={lr:.2e}")
        print(msg)
        logger.info(msg)

    # --------- 评估工具 ---------
    @staticmethod
    def rmse_score(pred: torch.Tensor, target: torch.Tensor) -> float:
        d = (pred - target).detach().cpu().numpy()
        return float(np.sqrt((d ** 2).mean()))

    @staticmethod
    def phm08_score(pred: torch.Tensor, target: torch.Tensor, clip_max: float = 125.0) -> float:
        pred = torch.clamp(pred, max=clip_max).detach().cpu().numpy()
        target = torch.clamp(target, max=clip_max).detach().cpu().numpy()
        d = pred - target
        score = np.sum(np.where(d > 0, np.exp(d / 10.0) - 1.0, np.exp(-d / 13.0) - 1.0))
        return float(score)

    @torch.no_grad()
    def evaluate_last_window(self, test_dataset, clip_max: float = 125.0) -> Tuple[float, float]:
        self.model.eval()
        last_preds, last_targets = [], []
        use_amp = bool(getattr(self, "use_amp", True) and torch.cuda.is_available())
        amp_dtype = (
            torch.bfloat16
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            for uid in test_dataset.units:
                unit_indices = [i for i, (u, _) in enumerate(test_dataset.sample_index) if u == uid]
                if unit_indices:
                    x, y = test_dataset[unit_indices[-1]]
                    x = x.unsqueeze(0).to(self.device)   # (1,L,C)
                    pred = self.model(x)                 # (1,)
                    last_preds.append(float(pred.item()))
                    last_targets.append(float(y.item()))
        if last_preds:
            pred_t = torch.tensor(last_preds)
            target_t = torch.tensor(last_targets)
            rmse = self.rmse_score(pred_t, target_t)
            score = self.phm08_score(pred_t, target_t, clip_max=clip_max)
            return rmse, score
        else:
            return float("nan"), float("nan")

    # --------- 常用损失函数（可选） ---------
    @staticmethod
    def weighted_mse_phm_like(pred, y, norm_scale=125.0, normalized=True):
        if normalized:
            pos_k, neg_k = 10.0 / norm_scale, 13.0 / norm_scale
        else:
            pos_k, neg_k = 10.0, 13.0
        err = pred - y
        w = torch.where(err > 0, torch.exp(err / pos_k), torch.exp(-err / neg_k))
        return (w * err.pow(2)).mean()
