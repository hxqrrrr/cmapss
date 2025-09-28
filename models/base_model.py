# base_model.py
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
        """定义具体模型结构 (nn.Module)"""
        pass

    @abstractmethod
    def compile(self, learning_rate: float, weight_decay: float, **kwargs):
        """设定优化器 & 调度器"""
        pass

    def train_model(self, train_loader, val_loader=None,
               criterion: Optional[Any] = None, epochs: int = 20,
               test_dataset_for_last=None, early_stopping_patience=7):
        """默认训练流程，子类可重写
        
        Returns:
            dict: 包含最佳验证性能的字典，如果启用早停则返回最佳结果，否则返回最终结果
        """
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
            
            # 最后窗口RMSE和Score
            val_rmse_last, val_score_last = float('nan'), float('nan')
            if test_dataset_for_last is not None:
                val_rmse_last, val_score_last = self.evaluate_last_window(test_dataset_for_last)
            
            # 确保模型回到训练模式（重要：LSTM等RNN模型需要这个）
            # 在验证后重新设置训练模式，对所有模型都是安全的
            self.model.train()
            
            # 获取学习率
            lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
            
            # 统一的日志格式 - 每个模型都必须记录这些指标
            self.log_training_metrics(epoch, epochs, train_rmse, val_rmse_global, 
                                    val_rmse_last, val_score_last, lr)
            
            # ReduceLROnPlateau调度器需要在epoch结束后更新
            if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = val_rmse_last if not torch.isnan(torch.tensor(val_rmse_last)) else val_rmse_global
                if not torch.isnan(torch.tensor(metric)):
                    self.scheduler.step(metric)
            
            # 早停机制（仅在early_stopping_patience > 0时启用）
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
        
        # 返回最佳验证结果
        if early_stopping_patience > 0 and best_val_loss != float('inf'):
            return {
                "best_val_rmse": best_val_loss, 
                "early_stopped": True,
                "best_epoch": best_epoch
            }
        else:
            # 如果没有早停或没有验证数据，返回最终验证结果
            final_results = {}
            if val_loader is not None:
                final_val_results = self.evaluate(val_loader)
                final_results["final_val_rmse"] = final_val_results.get('rmse', float('nan'))
                
                # 保存最终指标
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

    def evaluate(self, test_loader) -> Dict[str, float]:
        """默认测试集评估"""
        self.model.eval()
        mse, n = 0.0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = self.to_device(data, target)
                target = target.view(-1)
                pred = self.model(data).view(-1)
                mse += torch.sum((pred - target) ** 2).item()
                n += target.numel()
        rmse = (mse / max(1, n)) ** 0.5
        return {"rmse": rmse}


    # --------- 通用工具函数 (子类可复用) ---------
    def to_device(self, data, target):
        """统一的数据迁移接口"""
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        return data, target

    def log_training_metrics(self, epoch: int, total_epochs: int, 
                           train_rmse: float, val_rmse_global: float,
                           val_rmse_last: float, val_score_last: float, lr: float):
        """
        统一的训练指标日志记录格式 - 每个模型都必须记录这些指标
        Args:
            epoch: 当前轮次
            total_epochs: 总轮次
            train_rmse: 训练集RMSE
            val_rmse_global: 验证集全局RMSE (cycles)
            val_rmse_last: 验证集最后窗口RMSE (cycles) 
            val_score_last: 验证集最后窗口PHM08 Score
            lr: 当前学习率
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # 格式化数值显示
        def format_metric(val):
            if np.isnan(val) or val == float('nan'):
                return "N/A"
            return f"{val:.2f}"
        
        # 详细的日志格式
        metrics_msg = (f"[Epoch {epoch:3d}/{total_epochs}] "
                      f"train_rmse={format_metric(train_rmse)} | "
                      f"val_rmse(global)={format_metric(val_rmse_global)} cycles | "
                      f"val_rmse(last)={format_metric(val_rmse_last)} cycles | "
                      f"val_score(last)={format_metric(val_score_last)} | "
                      f"lr={lr:.2e}")
        
        # 同时输出到控制台和日志文件
        print(metrics_msg)
        logger.info(metrics_msg)

    # 移除模型保存和加载功能

    # --------- 评估相关工具函数 ---------
    @staticmethod
    def rmse_score(pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算 RMSE"""
        d = (pred - target).detach().cpu().numpy()
        return float(np.sqrt((d ** 2).mean()))

    @staticmethod
    def phm08_score(pred: torch.Tensor, target: torch.Tensor, clip_max: float = 125.0) -> float:
        """PHM08竞赛风格的非对称评分函数"""
        pred = torch.clamp(pred, max=clip_max).detach().cpu().numpy()
        target = torch.clamp(target, max=clip_max).detach().cpu().numpy()
        d = pred - target
        score = np.sum(np.where(d > 0, np.exp(d / 10.0) - 1.0, np.exp(-d / 13.0) - 1.0))
        return float(score)

    @torch.no_grad()
    def evaluate_last_window(self, test_dataset, clip_max: float = 125.0) -> Tuple[float, float]:
        """
        每台测试发动机取最后一个窗口，返回 (RMSE, Score)
        """
        self.model.eval()
        last_preds, last_targets = [], []

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
        """
        对 MSE 按 PHM08 不对称代价加权：
        err>0 (晚报):  weight = exp(err / (10/norm_scale))
        err<=0 (早报): weight = exp(-err / (13/norm_scale))
        """
        if normalized:
            pos_k, neg_k = 10.0 / norm_scale, 13.0 / norm_scale
        else:
            pos_k, neg_k = 10.0, 13.0
        err = pred - y
        w = torch.where(err > 0, torch.exp(err / pos_k), torch.exp(-err / neg_k))
        return (w * err.pow(2)).mean()
