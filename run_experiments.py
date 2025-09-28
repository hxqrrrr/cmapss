#!/usr/bin/env python3
"""
实验运行脚本：批量运行不同配置的实验
"""
import subprocess
import sys
import time


def run_command(cmd):
    """运行命令并打印结果"""
    print(f"\n{'='*60}")
    print(f"运行命令: {' '.join(cmd)}")
    print('='*60)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    end_time = time.time()
    
    print(f"\n命令执行完成，耗时: {end_time - start_time:.1f}秒")
    print(f"返回码: {result.returncode}")
    
    return result.returncode == 0


def main():
    """运行一系列实验"""
    
    experiments = [
        
        # # TSMixer实验
        # {
        #     "name": "TSMixer + FD001 (OneCycle + 早停)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "256",
        #            "--epochs", "80",  # 增加最大epochs，让早停决定何时停止
        #            "--learning_rate", "0.001",  # 降低学习率
        #            "--tsmixer_layers", "3",  # 减少复杂度
        #            "--time_expansion", "3",
        #            "--feat_expansion", "3",
        #            "--dropout", "0.15",  # 增加正则化
        #            "--scheduler", "onecycle",
        #            "--early_stopping", "10"  # 10轮不改善就停止
        #            ]
        # },
        
        # {
        #     "name": "TSMixer + FD001 (Cosine调度器)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "128",
        #            "--epochs", "80",
        #            "--learning_rate", "0.002",  # 适中的初始学习率
        #            "--tsmixer_layers", "6",
        #            "--time_expansion", "6",
        #            "--feat_expansion", "4",
        #            "--dropout", "0.15",
        #            "--scheduler", "cosine"  # 使用Cosine调度器
        #            ]
        # },
        
        # {
        #     "name": "TSMixer + FD001 (Plateau + 早停3)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "256",
        #            "--epochs", "100",  # 更多epochs配合早停
        #            "--learning_rate", "0.002",
        #            "--tsmixer_layers", "4",
        #            "--time_expansion", "4",
        #            "--feat_expansion", "4",
        #            "--dropout", "0.1",
        #            "--scheduler", "plateau",
        #            "--early_stopping", "3"  # 更严格的早停
        #            ]
        # },
        
        # {
        #     "name": "TSMixer + FD001 (无早停对比)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "256",
        #            "--epochs", "30",  # 较少epochs
        #            "--learning_rate", "0.001",
        #            "--tsmixer_layers", "3",
        #            "--time_expansion", "3",
        #            "--feat_expansion", "3",
        #            "--dropout", "0.15",
        #            "--scheduler", "cosine",
        #            "--early_stopping", "0"  # 禁用早停
        #            ]
        # },
        
        # {
        #     "name": "TSMixer + FD002 (复杂数据)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD002", 
        #            "--batch_size", "128",
        #            "--epochs", "100",
        #            "--learning_rate", "0.001",
        #            "--tsmixer_layers", "5",
        #            "--time_expansion", "8",
        #            "--feat_expansion", "6",
        #            "--dropout", "0.2",
        #            "--scheduler", "onecycle"
        #            ]
        # },
        
        # RBM-LSTM实验 - 寻找最优配置
        # {
        #     "name": "RBM-LSTM-1: 基础优化配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "rbmlstm", 
        #            "--fault", "FD001", 
        #            "--batch_size", "128",
        #            "--epochs", "60",
        #            "--learning_rate", "0.001",
        #            "--rbm_hidden", "128",
        #            "--lstm_hidden1", "128",
        #            "--lstm_hidden2", "64",
        #            "--ff_hidden", "32",
        #            "--dropout_lstm", "0.3",
        #            "--rbm_pool", "last",
        #            "--enable_rbm_pretrain",
        #            "--rbm_epochs", "5",
        #            "--rbm_lr", "0.005",
        #            "--rbm_cd_k", "1",
        #            "--scheduler", "onecycle",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "RBM-LSTM-2: 更大模型",
        #     "cmd": ["python", "train.py", 
        #            "--model", "rbmlstm", 
        #            "--fault", "FD001", 
        #            "--batch_size", "96",         # 减小batch_size适应大模型
        #            "--epochs", "60",
        #            "--learning_rate", "0.0008",  # 大模型用稍小学习率
        #            "--rbm_hidden", "256",        # 更大的RBM
        #            "--lstm_hidden1", "256",      # 更大的LSTM
        #            "--lstm_hidden2", "128",
        #            "--ff_hidden", "64",          # 更大的前馈层
        #            "--dropout_lstm", "0.4",      # 大模型需要更多正则化
        #            "--rbm_pool", "last",
        #            "--enable_rbm_pretrain",
        #            "--rbm_epochs", "6",          # 更多预训练
        #            "--rbm_lr", "0.003",          # 更小的RBM学习率
        #            "--rbm_cd_k", "1",
        #            "--scheduler", "onecycle",
        #            "--early_stopping", "15"
        #            ]
        # },
        
        # {
        #     "name": "RBM-LSTM-3: 深层网络",
        #     "cmd": ["python", "train.py", 
        #            "--model", "rbmlstm", 
        #            "--fault", "FD001", 
        #            "--batch_size", "128",
        #            "--epochs", "60",
        #            "--learning_rate", "0.0012",  # 稍高学习率
        #            "--rbm_hidden", "128",
        #            "--lstm_hidden1", "128",
        #            "--lstm_hidden2", "128",      # 两层LSTM相同大小
        #            "--ff_hidden", "64",          # 更大前馈层
        #            "--dropout_lstm", "0.25",     # 稍低dropout
        #            "--rbm_pool", "mean",         # 尝试均值池化
        #            "--enable_rbm_pretrain",
        #            "--rbm_epochs", "4",
        #            "--rbm_lr", "0.008",          # 稍高RBM学习率
        #            "--rbm_cd_k", "1",
        #            "--scheduler", "onecycle",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "RBM-LSTM-4: 高学习率配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "rbmlstm", 
        #            "--fault", "FD001", 
        #            "--batch_size", "160",        # 更大batch
        #            "--epochs", "50",
        #            "--learning_rate", "0.0015",  # 更高学习率
        #            "--rbm_hidden", "128",
        #            "--lstm_hidden1", "128",
        #            "--lstm_hidden2", "64",
        #            "--ff_hidden", "32",
        #            "--dropout_lstm", "0.2",      # 更低dropout
        #            "--rbm_pool", "last",
        #            "--enable_rbm_pretrain",
        #            "--rbm_epochs", "3",          # 较少预训练轮数
        #            "--rbm_lr", "0.01",           # 较高RBM学习率
        #            "--rbm_cd_k", "1",
        #            "--scheduler", "onecycle",
        #            "--early_stopping", "10"
        #            ]
        # },
        
        # {
        #     "name": "RBM-LSTM-5: 无预训练基线",
        #     "cmd": ["python", "train.py", 
        #            "--model", "rbmlstm", 
        #            "--fault", "FD001", 
        #            "--batch_size", "128",
        #            "--epochs", "60",
        #            "--learning_rate", "0.001",
        #            "--rbm_hidden", "128",
        #            "--lstm_hidden1", "128",
        #            "--lstm_hidden2", "64",
        #            "--ff_hidden", "32",
        #            "--dropout_lstm", "0.3",
        #            "--rbm_pool", "last",
        #            # 不启用RBM预训练
        #            "--scheduler", "onecycle",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "RBM-LSTM-6: Cosine调度器",
        #     "cmd": ["python", "train.py", 
        #            "--model", "rbmlstm", 
        #            "--fault", "FD001", 
        #            "--batch_size", "128",
        #            "--epochs", "80",             # 更多轮数配合cosine
        #            "--learning_rate", "0.002",   # 更高初始学习率
        #            "--rbm_hidden", "128",
        #            "--lstm_hidden1", "128",
        #            "--lstm_hidden2", "64",
        #            "--ff_hidden", "32",
        #            "--dropout_lstm", "0.35",
        #            "--rbm_pool", "last",
        #            "--enable_rbm_pretrain",
        #            "--rbm_epochs", "5",
        #            "--rbm_lr", "0.005",
        #            "--rbm_cd_k", "1",
        #            "--scheduler", "cosine",      # 使用cosine调度器
        #            "--early_stopping", "15"
        #            ]
        #         },
        
        # # TSMixer 优化实验 - 基于日志分析
        # {
        #     "name": "TSMixer-1: 最优基线配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "256",
        #            "--epochs", "60",
        #            "--learning_rate", "0.002",      # 验证有效的高学习率
        #            "--tsmixer_layers", "4",         # 最佳层数
        #            "--time_expansion", "4",
        #            "--feat_expansion", "4",
        #            "--dropout", "0.1",              # 较低dropout
        #            "--scheduler", "plateau",        # 最佳调度器
        #            "--early_stopping", "5"          # 稍微增加耐心值
        #            ]
        # },
        
        # {
        #     "name": "TSMixer-2: 强正则化配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "128",           # 减小batch size
        #            "--epochs", "60",
        #            "--learning_rate", "0.0015",     # 稍低学习率
        #            "--weight_decay", "0.0005",      # 增强权重衰减
        #            "--tsmixer_layers", "4",
        #            "--time_expansion", "3",         # 减少复杂度
        #            "--feat_expansion", "3",
        #            "--dropout", "0.2",              # 更强dropout
        #            "--scheduler", "plateau",
        #            "--early_stopping", "8"
        #            ]
        # },
        
        # {
        #     "name": "TSMixer-3: 深层网络配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "192",
        #            "--epochs", "80",
        #            "--learning_rate", "0.001",      # 深层网络用较低学习率
        #            "--weight_decay", "0.0002",
        #            "--tsmixer_layers", "6",         # 更深网络
        #            "--time_expansion", "5",
        #            "--feat_expansion", "4",
        #            "--dropout", "0.15",
        #            "--scheduler", "onecycle",       # 深层网络适合onecycle
        #            "--early_stopping", "10"
        #            ]
        # },
        
        # {
        #     "name": "TSMixer-4: 保守训练配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "256",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0008",     # 保守学习率
        #            "--weight_decay", "0.0003",
        #            "--tsmixer_layers", "3",         # 较浅网络
        #            "--time_expansion", "4",
        #            "--feat_expansion", "4",
        #            "--dropout", "0.25",             # 强正则化
        #            "--scheduler", "cosine",         # 长期训练用cosine
        #            "--early_stopping", "15"
        #            ]
        # },
        
        # {
        #     "name": "TSMixer-5: 高效快速配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "384",           # 大batch快速训练
        #            "--epochs", "40",                # 较少轮数
        #            "--learning_rate", "0.003",      # 更高学习率快速收敛
        #            "--weight_decay", "0.0001",
        #            "--tsmixer_layers", "4",
        #            "--time_expansion", "4",
        #            "--feat_expansion", "4",
        #            "--dropout", "0.1",
        #            "--scheduler", "plateau",
        #            "--early_stopping", "3"          # 激进早停
        #            ]
        # },
        
        # {
        #     "name": "TSMixer-6: FD002复杂数据",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD002", 
        #            "--batch_size", "128",
        #            "--epochs", "100",
        #            "--learning_rate", "0.001",      # FD002用较保守参数
        #            "--weight_decay", "0.0005",
        #            "--tsmixer_layers", "5",         # 复杂数据需要更多层
        #            "--time_expansion", "6",
        #            "--feat_expansion", "5",
        #            "--dropout", "0.2",              # 更强正则化
        #            "--scheduler", "onecycle",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # ============================================================================
        # TSMixer 优化实验 - 基于日志分析的新配置
        # ============================================================================
        
        # {
        #     "name": "TSMixer + FD001 (深层优化配置)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "256",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0012",
        #            "--weight_decay", "0.0002",
        #            "--tsmixer_layers", "8",         # 更深的网络
        #            "--time_expansion", "6",         # 更大的时间扩展
        #            "--feat_expansion", "4",
        #            "--dropout", "0.12",             # 精调的dropout
        #            "--scheduler", "onecycle",       # 最佳调度器
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "TSMixer + FD001 (快速高效配置)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "512",           # 大批量
        #            "--epochs", "60",
        #            "--learning_rate", "0.004",      # 高学习率
        #            "--weight_decay", "0.0001",
        #            "--tsmixer_layers", "5",
        #            "--time_expansion", "4",
        #            "--feat_expansion", "5",         # 平衡的扩展
        #            "--dropout", "0.08",             # 低dropout
        #            "--scheduler", "plateau",
        #            "--early_stopping", "5"
        #            ]
        # },
        
        # {
        #     "name": "TSMixer + FD001 (极深网络测试)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "192",
        #            "--epochs", "120",
        #            "--learning_rate", "0.001",
        #            "--weight_decay", "0.0003",
        #            "--tsmixer_layers", "10",        # 极深网络
        #            "--time_expansion", "5",
        #            "--feat_expansion", "4",
        #            "--dropout", "0.15",             # 适中正则化
        #            "--scheduler", "onecycle",
        #            "--early_stopping", "15"
        #            ]
        # },
        
        # ============================================================================
        # 前三名最优配置 - 基于13个实验的性能分析
        # ============================================================================
        
        {
            "name": "🥇 TSMixer冠军配置 (11.46 RMSE)",
            "cmd": ["python", "train.py", 
                   "--model", "tsmixer", 
                   "--fault", "FD001", 
                   "--batch_size", "384",           # 大批量训练
                   "--epochs", "50",
                   "--learning_rate", "0.005",      # 极高学习率
                   "--weight_decay", "0.00005",     # 极低权重衰减
                   "--tsmixer_layers", "4",         # 轻量网络
                   "--time_expansion", "3",         # 适中时间扩展
                   "--feat_expansion", "4",         # 适中特征扩展
                   "--dropout", "0.05",             # 极低dropout
                   "--scheduler", "plateau",        # 平台调度器
                   "--early_stopping", "10"          # 激进早停
                   ]
        },
        
        {
            "name": "🥈 TSMixer亚军配置 (11.69 RMSE)",
            "cmd": ["python", "train.py", 
                   "--model", "tsmixer", 
                   "--fault", "FD001", 
                   "--batch_size", "512",           # 超大批量
                   "--epochs", "60",
                   "--learning_rate", "0.004",      # 高学习率
                   "--weight_decay", "0.0001",      # 低权重衰减
                   "--tsmixer_layers", "5",         # 中等深度
                   "--time_expansion", "4",         # 平衡时间扩展
                   "--feat_expansion", "5",         # 较大特征扩展
                   "--dropout", "0.08",             # 低dropout
                   "--scheduler", "plateau",        # 平台调度器
                   "--early_stopping", "10"          # 快速早停
                   ]
        },
        
        {
            "name": "🥉 TSMixer季军配置 (11.91 RMSE)",
            "cmd": ["python", "train.py", 
                   "--model", "tsmixer", 
                   "--fault", "FD001", 
                   "--batch_size", "256",           # 中等批量
                   "--epochs", "100",               # 更多训练轮数
                   "--learning_rate", "0.0012",     # 精调学习率
                   "--weight_decay", "0.0002",      # 适中权重衰减
                   "--tsmixer_layers", "8",         # 深层网络
                   "--time_expansion", "6",         # 大时间扩展
                   "--feat_expansion", "4",         # 适中特征扩展
                   "--dropout", "0.12",             # 适中dropout
                   "--scheduler", "onecycle",       # OneCycle调度器
                   "--early_stopping", "12"         # 耐心早停
                   ]
        },
        
        
        
        # {
        #     "name": "RBM-LSTM + FD001 (无预训练基线)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "rbmlstm", 
        #            "--fault", "FD001", 
        #            "--batch_size", "256",
        #            "--epochs", "60",
        #            "--learning_rate", "0.0003",  # 论文建议的AdamW学习率
        #            "--rbm_hidden", "64",
        #            "--lstm_hidden1", "64",
        #            "--lstm_hidden2", "64",
        #            "--ff_hidden", "8",
        #            "--dropout_lstm", "0.5",
        #            "--rbm_pool", "last",
        #            "--scheduler", "plateau",
        #            "--early_stopping", "7"
        #            ]
        # },
        
        # {
        #     "name": "RBM-LSTM + FD001 (RBM预训练)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "rbmlstm", 
        #            "--fault", "FD001", 
        #            "--batch_size", "256",
        #            "--epochs", "30",            # 减少epochs便于测试
        #            "--learning_rate", "0.0003",
        #            "--rbm_hidden", "64",
        #            "--lstm_hidden1", "64",
        #            "--lstm_hidden2", "64",
        #            "--ff_hidden", "8",
        #            "--dropout_lstm", "0.5",
        #            "--rbm_pool", "last",
        #            "--enable_rbm_pretrain",  # 启用RBM预训练
        #            "--rbm_epochs", "3",      # RBM预训练轮数
        #            "--rbm_lr", "0.01",       # RBM预训练学习率
        #            "--rbm_cd_k", "1",        # 对比散度步数
        #            "--scheduler", "plateau",
        #            "--early_stopping", "7"
        #            ]
        # },
        
        # {
        #     "name": "RBM-LSTM + FD001 (大模型+预训练)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "rbmlstm", 
        #            "--fault", "FD001", 
        #            "--batch_size", "128",
        #            "--epochs", "80",
        #            "--learning_rate", "0.0002",  # 大模型用较小学习率
        #            "--rbm_hidden", "128",         # 更大的RBM隐藏层
        #            "--lstm_hidden1", "128",       # 更大的LSTM层
        #            "--lstm_hidden2", "64",
        #            "--ff_hidden", "16",           # 更大的前馈层
        #            "--dropout_lstm", "0.6",       # 更强的正则化
        #            "--rbm_pool", "mean",          # 尝试平均池化
        #            "--enable_rbm_pretrain",
        #            "--rbm_epochs", "5",           # 更多预训练轮数
        #            "--rbm_lr", "0.008",
        #            "--rbm_cd_k", "1",
        #            "--scheduler", "onecycle",
        #            "--early_stopping", "10"
        #            ]
        # },
        
        # {
        #     "name": "RBM-LSTM + FD002 (复杂数据+预训练)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "rbmlstm", 
        #            "--fault", "FD002", 
        #            "--batch_size", "128",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0002",
        #            "--rbm_hidden", "96",
        #            "--lstm_hidden1", "96",
        #            "--lstm_hidden2", "64",
        #            "--ff_hidden", "12",
        #            "--dropout_lstm", "0.6",       # FD002更复杂，需要更强正则化
        #            "--rbm_pool", "last",
        #            "--enable_rbm_pretrain",
        #            "--rbm_epochs", "4",
        #            "--rbm_lr", "0.01",
        #            "--rbm_cd_k", "1",
        #            "--scheduler", "cosine",
        #            "--early_stopping", "8"
        #            ]
        # },

        # BiLSTM实验（注释掉以便专注测试RBM-LSTM）
        # {
        #     "name": "BiLSTM + FD001 (基础配置)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "bilstm", 
        #            "--fault", "FD001", 
        #            "--batch_size", "256",
        #            "--epochs", "60",
        #            "--learning_rate", "0.001",
        #            "--lstm_hidden", "64",
        #            "--lstm_layers", "2",
        #            "--bidirectional",  # 双向LSTM
        #            "--mlp_hidden", "64",
        #            "--lstm_pool", "last",
        #            "--dropout", "0.1",
        #            "--scheduler", "plateau",
        #            "--early_stopping", "7"
        #            ]
        # },
        
        # {
        #     "name": "BiLSTM + FD001 (深层网络)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "bilstm", 
        #            "--fault", "FD001", 
        #            "--batch_size", "128",
        #            "--epochs", "80",
        #            "--learning_rate", "0.0008",
        #            "--lstm_hidden", "128",  # 更大的隐藏层
        #            "--lstm_layers", "3",    # 更多层数
        #            "--bidirectional",
        #            "--mlp_hidden", "128",
        #            "--lstm_pool", "mean",   # 平均池化
        #            "--dropout", "0.15",
        #            "--scheduler", "onecycle",
        #            "--early_stopping", "10"
        #            ]
        # },
        
        # {
        #     "name": "BiLSTM + FD002 (复杂数据)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "bilstm", 
        #            "--fault", "FD002", 
        #            "--batch_size", "128",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0005",
        #            "--lstm_hidden", "96",
        #            "--lstm_layers", "2",
        #            "--bidirectional",
        #            "--mlp_hidden", "96",
        #            "--lstm_pool", "last",
        #            "--dropout", "0.2",
        #            "--scheduler", "cosine",
        #            "--early_stopping", "8"
        #            ]
        # },

        
        # {
        #     "name": "Transformer + FD001 (小模型)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "transformer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "256",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0003",
        #            "--d_model", "128",
        #            "--nhead", "8",
        #            "--num_layers", "6",
        #            ]
        # },
        
        # {
        #     "name": "Transformer + FD001 (中等模型)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "transformer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "256",
        #            "--epochs", "30",
        #            "--learning_rate", "0.0003",
        #            "--d_model", "128",
        #            "--nhead", "4",
        #            "--num_layers", "3",
        #            ]
        # },
        
       
        # {
        #     "name": "Transformer + FD002",
        #     "cmd": ["python", "train.py", 
        #            "--model", "transformer", 
        #            "--fault", "FD002", 
        #            "--batch_size", "256",
        #            "--epochs", "40",  # FD002更复杂，需要更多轮次
        #            "--learning_rate", "0.0002",  # 稍微降低学习率
        #            "--d_model", "128",
        #            "--nhead", "4",
        #            "--num_layers", "3",
        #            ]
        # }
    ]
    
    print("开始批量实验...")
    results = {}
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n\n🚀 实验 {i}/{len(experiments)}: {exp['name']}")
        
        success = run_command(exp["cmd"])
        results[exp["name"]] = "成功" if success else "失败"
        
        if not success:
            print(f"❌ 实验失败: {exp['name']}")
            choice = input("继续下一个实验？(y/n): ").lower()
            if choice != 'y':
                break
        else:
            print(f"✅ 实验成功: {exp['name']}")
    
    # 打印总结
    print(f"\n\n{'='*60}")
    print("实验总结:")
    print('='*60)
    for name, status in results.items():
        status_icon = "✅" if status == "成功" else "❌"
        print(f"{status_icon} {name}: {status}")


if __name__ == "__main__":
    main()
