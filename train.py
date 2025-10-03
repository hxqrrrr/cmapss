#!/usr/bin/env python3
"""
统一训练脚本：支持多种模型架构（TSMixer、Transformer）
基于 BaseRULModel 的接口，无需额外 Trainer 类
"""
import argparse
import os
import torch
import logging
import sys
from datetime import datetime
from dataset import CMAPSSDataset, _xu_window_for
from models.tsmixer_model import TSMixerModel
from models.tsmixer_sga import TSMixerModel as TSMixerSGAModel
from models.transformer_model import TransformerModel
from models.bilstm_model import BiLSTMModel
from models.ellefsen_rbm_lstm_model import EllefsenRBMLSTMModel
from models.cnn_tsmixer_model import CNNMixerRULModel
from models.tsmixer_cnn_gated import CNNMixerGatedRULModel
from models.tsmixer_gated_tokenpool import TokenPoolRULModel


def parse_args():
    parser = argparse.ArgumentParser(description='CMAPSS RUL Prediction Training')

    # 模型选择
    parser.add_argument("--model", type=str, default="transformer",
                        choices=["tsmixer", "tsmixer_sga", "transformer", "bilstm", "rbmlstm", "cnn_tsmixer", "cnn_tsmixer_gated", "tokenpool"], help="选择模型架构")

    # 数据集配置
    parser.add_argument("--fault", type=str, default="FD001",
                        choices=["FD001", "FD002", "FD003", "FD004"], help="故障模式")
    parser.add_argument("--data_dir", type=str, default="CMAPSSData",
                        help="数据目录")

    # 训练配置
    parser.add_argument("--batch_size", type=int, default=256, help="批量大小")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")

    # 模型超参数
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout比例")

    # Transformer特定参数
    parser.add_argument("--d_model", type=int, default=128, help="Transformer模型维度")
    parser.add_argument("--nhead", type=int, default=4, help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=3, help="Transformer层数")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="前馈网络维度")
    parser.add_argument("--pool", type=str, default="cls", choices=["cls", "mean", "last"], help="池化方式")

    # TSMixer特定参数
    parser.add_argument("--tsmixer_layers", type=int, default=4, help="TSMixer层数")
    parser.add_argument("--time_expansion", type=int, default=4, help="时间混合层扩展倍数")
    parser.add_argument("--feat_expansion", type=int, default=4, help="特征混合层扩展倍数")
    
    # TSMixer-SGA特定参数
    parser.add_argument("--use_sga", action="store_true", help="启用SGA注意力机制")
    parser.add_argument("--sga_time_rr", type=int, default=4, help="SGA时间方向压缩比")
    parser.add_argument("--sga_feat_rr", type=int, default=4, help="SGA特征方向压缩比")
    parser.add_argument("--sga_dropout", type=float, default=0.05, help="SGA内部dropout")
    parser.add_argument("--sga_pool", type=str, default="mean", 
                        choices=["mean", "last", "weighted"], help="TSMixer-SGA池化方式")
    
    # BiLSTM特定参数
    parser.add_argument("--lstm_hidden", type=int, default=64, help="LSTM隐藏层维度")
    parser.add_argument("--lstm_layers", type=int, default=2, help="LSTM层数")
    parser.add_argument("--bidirectional", action="store_true", default=True, help="是否使用双向LSTM")
    parser.add_argument("--mlp_hidden", type=int, default=64, help="MLP隐藏层维度")
    parser.add_argument("--lstm_pool", type=str, default="last", choices=["last", "mean", "max"], help="LSTM输出池化方式")

    # RBM-LSTM特定参数
    parser.add_argument("--rbm_hidden", type=int, default=64, help="RBM隐藏单元数")
    parser.add_argument("--lstm_hidden1", type=int, default=64, help="第一层LSTM隐藏单元数")
    parser.add_argument("--lstm_hidden2", type=int, default=64, help="第二层LSTM隐藏单元数")
    parser.add_argument("--ff_hidden", type=int, default=8, help="前馈网络隐藏单元数")
    parser.add_argument("--dropout_lstm", type=float, default=0.5, help="LSTM层Dropout比例")
    parser.add_argument("--rbm_pool", type=str, default="last", choices=["last", "mean"], help="RBM-LSTM池化方式")
    
    # RBM预训练参数
    parser.add_argument("--enable_rbm_pretrain", action="store_true", help="启用RBM预训练")
    parser.add_argument("--rbm_epochs", type=int, default=1, help="RBM预训练轮数")
    parser.add_argument("--rbm_lr", type=float, default=1e-2, help="RBM预训练学习率")
    parser.add_argument("--rbm_cd_k", type=int, default=1, help="RBM对比散度步数")

    # CNN-TSMixer特定参数
    parser.add_argument("--patch", type=int, default=5, help="CNN-TSMixer时间补丁大小")
    parser.add_argument("--cnn_channels", type=int, default=64, help="CNN前端输出通道数")
    parser.add_argument("--cnn_layers", type=int, default=3, help="CNN前端层数")
    parser.add_argument("--cnn_kernel", type=int, default=5, help="CNN卷积核大小")
    parser.add_argument("--depth", type=int, default=6, help="TSMixer层数")
    parser.add_argument("--token_mlp_dim", type=int, default=256, help="Token混合MLP维度")
    parser.add_argument("--channel_mlp_dim", type=int, default=128, help="Channel混合MLP维度")
    parser.add_argument("--cnn_pool", type=str, default="mean", 
                        choices=["mean", "last", "weighted"], help="CNN-TSMixer池化方式")

    # 门控CNN-TSMixer特定参数
    parser.add_argument("--gn_groups", type=int, default=8, help="GroupNorm分组数")
    parser.add_argument("--use_groupnorm", action="store_true", default=True, help="是否使用GroupNorm")
    parser.add_argument("--no_groupnorm", action="store_true", help="禁用GroupNorm，使用BatchNorm")
    
    # TokenPool特定参数
    parser.add_argument("--tokenpool_heads", type=int, default=4, help="TokenPool注意力头数")
    parser.add_argument("--tokenpool_dropout", type=float, default=0.0, help="TokenPool注意力dropout")
    parser.add_argument("--tokenpool_temperature", type=float, default=1.5, help="TokenPool注意力温度")

    # 学习率调度器
    parser.add_argument("--scheduler", type=str, default="onecycle",
                        choices=["onecycle", "plateau", "cosine"], help="学习率调度器")

    # 早停参数
    parser.add_argument("--early_stopping", type=int, default=7, help="早停耐心值，0表示不使用早停")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志保存目录")
    parser.add_argument("--no_log", action="store_true", help="不保存日志文件")

    return parser.parse_args()


def setup_logging(args):
    if args.no_log:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout)])
        return None

    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{args.model}_{args.fault}_{timestamp}.log"
    log_path = os.path.join(args.log_dir, log_filename)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_path, encoding='utf-8'),
                                  logging.StreamHandler(sys.stdout)])
    return log_path


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np, random
    np.random.seed(seed)
    random.seed(seed)


def create_datasets(args):
    print(f"=== 创建 {args.fault} 数据集 ===")
    
    # 为多工况数据集（FD002/FD004）使用工况设置+传感器，其他使用14传感器
    if args.fault in ["FD002", "FD004"]:
        print(f"检测到多工况数据集 {args.fault}，使用工况设置+传感器特征")
        features_mode = "all"  # 包含3个工况设置 + 21个传感器 = 24特征
    else:
        print(f"检测到单工况数据集 {args.fault}，使用14传感器特征")
        features_mode = "14sensors"  # 14个传感器
    
    train_set = CMAPSSDataset(
        data_dir=args.data_dir, fault_mode=args.fault, split="train", 
        preset="custom", features=features_mode,
        window_size=_xu_window_for(args.fault), stride=1,
        norm="minmax", label_mode="piecewise_125"
    )
    test_set = CMAPSSDataset(
        data_dir=args.data_dir, fault_mode=args.fault, split="test",
        preset="custom", features=features_mode,
        window_size=_xu_window_for(args.fault), stride=1,
        norm="minmax", label_mode="piecewise_125",
        scaler=train_set.get_scaler()
    )
    print(f"训练样本数: {len(train_set)} | 测试样本数: {len(test_set)}")
    print(f"特征维度: {train_set.n_features}, 窗口大小: {train_set.window_size}")
    return train_set, test_set


def create_model(args, input_size, seq_len):
    if args.model == "transformer":
        model = TransformerModel(
            input_size=input_size, seq_len=seq_len,
            d_model=args.d_model, nhead=args.nhead,
            num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
            dropout=args.dropout, pool=args.pool
        )
    elif args.model == "tsmixer":
        model = TSMixerModel(
            input_size=input_size, seq_len=seq_len,
            num_layers=args.tsmixer_layers,
            time_expansion=args.time_expansion,
            feat_expansion=args.feat_expansion,
            dropout=args.dropout
        )
    elif args.model == "tsmixer_sga":
        model = TSMixerSGAModel(
            input_size=input_size, seq_len=seq_len,
            num_layers=args.tsmixer_layers,
            time_expansion=args.time_expansion,
            feat_expansion=args.feat_expansion,
            dropout=args.dropout,
            use_sga=args.use_sga,
            sga_time_rr=args.sga_time_rr,
            sga_feat_rr=args.sga_feat_rr,
            sga_dropout=args.sga_dropout,
            pool=args.sga_pool
        )
    elif args.model == "bilstm":
        model = BiLSTMModel(
            input_size=input_size, seq_len=seq_len,
            lstm_hidden=args.lstm_hidden,
            num_layers=args.lstm_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            mlp_hidden=args.mlp_hidden,
            pool=args.lstm_pool
        )
    elif args.model == "rbmlstm":
        model = EllefsenRBMLSTMModel(
            input_size=input_size, seq_len=seq_len,
            rbm_hidden=args.rbm_hidden,
            lstm_hidden1=args.lstm_hidden1,
            lstm_hidden2=args.lstm_hidden2,
            ff_hidden=args.ff_hidden,
            dropout_lstm=args.dropout_lstm,
            pool=args.rbm_pool
        )
    elif args.model == "cnn_tsmixer":
        model = CNNMixerRULModel(
            input_size=input_size, seq_len=seq_len,
            patch=args.patch,
            cnn_channels=args.cnn_channels,
            cnn_layers=args.cnn_layers,
            cnn_kernel=args.cnn_kernel,
            d_model=args.d_model,
            depth=args.depth,
            token_mlp_dim=args.token_mlp_dim,
            channel_mlp_dim=args.channel_mlp_dim,
            dropout=args.dropout,
            pool=args.cnn_pool
        )
    elif args.model == "cnn_tsmixer_gated":
        # 处理GroupNorm参数
        use_groupnorm = args.use_groupnorm and not args.no_groupnorm
        model = CNNMixerGatedRULModel(
            input_size=input_size, seq_len=seq_len,
            patch=args.patch,
            cnn_channels=args.cnn_channels,
            cnn_layers=args.cnn_layers,
            cnn_kernel=args.cnn_kernel,
            d_model=args.d_model,
            depth=args.depth,
            token_mlp_dim=args.token_mlp_dim,
            channel_mlp_dim=args.channel_mlp_dim,
            dropout=args.dropout,
            pool=args.cnn_pool,
            gn_groups=args.gn_groups,
            use_groupnorm=use_groupnorm
        )
    elif args.model == "tokenpool":
        model = TokenPoolRULModel(
            input_size=input_size, seq_len=seq_len,
            patch=args.patch,
            d_model=args.d_model,
            depth=args.depth,
            token_mlp_dim=args.token_mlp_dim,
            channel_mlp_dim=args.channel_mlp_dim,
            dropout=args.dropout,
            pool=args.cnn_pool,  # TokenPool使用pool参数
            tokenpool_heads=args.tokenpool_heads,
            tokenpool_dropout=args.tokenpool_dropout,
            tokenpool_temperature=args.tokenpool_temperature
        )
    else:
        raise ValueError(f"未支持的模型类型: {args.model}")
    return model


def main():
    args = parse_args()
    log_path = setup_logging(args)
    logger = logging.getLogger(__name__)

    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == "auto" else torch.device(args.device)
    
    # 记录实验配置
    logger.info("="*80)
    logger.info("实验配置信息")
    logger.info("="*80)
    logger.info(f"模型类型: {args.model}")
    logger.info(f"数据集: {args.fault}")
    logger.info(f"使用设备: {device}")
    logger.info(f"随机种子: {args.seed}")
    
    # 基础训练参数
    logger.info(f"批量大小: {args.batch_size}")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"权重衰减: {args.weight_decay}")
    logger.info(f"学习率调度器: {args.scheduler}")
    logger.info(f"早停耐心值: {args.early_stopping}")
    
    # 模型特定参数
    if args.model == "rbmlstm":
        logger.info("--- RBM-LSTM 特定参数 ---")
        logger.info(f"RBM隐藏单元数: {args.rbm_hidden}")
        logger.info(f"LSTM第一层隐藏单元数: {args.lstm_hidden1}")
        logger.info(f"LSTM第二层隐藏单元数: {args.lstm_hidden2}")
        logger.info(f"前馈网络隐藏单元数: {args.ff_hidden}")
        logger.info(f"LSTM Dropout: {args.dropout_lstm}")
        logger.info(f"池化方式: {args.rbm_pool}")
        logger.info(f"启用RBM预训练: {args.enable_rbm_pretrain}")
        if args.enable_rbm_pretrain:
            logger.info(f"RBM预训练轮数: {args.rbm_epochs}")
            logger.info(f"RBM学习率: {args.rbm_lr}")
            logger.info(f"RBM对比散度步数: {args.rbm_cd_k}")
    elif args.model == "transformer":
        logger.info("--- Transformer 特定参数 ---")
        logger.info(f"模型维度: {args.d_model}")
        logger.info(f"注意力头数: {args.nhead}")
        logger.info(f"Transformer层数: {args.num_layers}")
        logger.info(f"前馈网络维度: {args.dim_feedforward}")
        logger.info(f"池化方式: {args.pool}")
        logger.info(f"Dropout: {args.dropout}")
    elif args.model == "bilstm":
        logger.info("--- BiLSTM 特定参数 ---")
        logger.info(f"LSTM隐藏层维度: {args.lstm_hidden}")
        logger.info(f"LSTM层数: {args.lstm_layers}")
        logger.info(f"双向LSTM: {args.bidirectional}")
        logger.info(f"MLP隐藏层维度: {args.mlp_hidden}")
        logger.info(f"池化方式: {args.lstm_pool}")
        logger.info(f"Dropout: {args.dropout}")
    elif args.model == "tsmixer":
        logger.info("--- TSMixer 特定参数 ---")
        logger.info(f"TSMixer层数: {args.tsmixer_layers}")
        logger.info(f"时间混合层扩展倍数: {args.time_expansion}")
        logger.info(f"特征混合层扩展倍数: {args.feat_expansion}")
        logger.info(f"Dropout: {args.dropout}")
    elif args.model == "tsmixer_sga":
        logger.info("--- TSMixer-SGA 特定参数 ---")
        logger.info(f"TSMixer层数: {args.tsmixer_layers}")
        logger.info(f"时间混合层扩展倍数: {args.time_expansion}")
        logger.info(f"特征混合层扩展倍数: {args.feat_expansion}")
        logger.info(f"Dropout: {args.dropout}")
        logger.info(f"启用SGA: {args.use_sga}")
        if args.use_sga:
            logger.info(f"SGA时间压缩比: {args.sga_time_rr}")
            logger.info(f"SGA特征压缩比: {args.sga_feat_rr}")
            logger.info(f"SGA Dropout: {args.sga_dropout}")
        logger.info(f"池化方式: {args.sga_pool}")
    elif args.model == "cnn_tsmixer":
        logger.info("--- CNN-TSMixer 特定参数 ---")
        logger.info(f"时间补丁大小: {args.patch}")
        logger.info(f"CNN前端通道数: {args.cnn_channels}")
        logger.info(f"CNN前端层数: {args.cnn_layers}")
        logger.info(f"CNN卷积核大小: {args.cnn_kernel}")
        logger.info(f"模型维度: {args.d_model}")
        logger.info(f"TSMixer层数: {args.depth}")
        logger.info(f"Token混合MLP维度: {args.token_mlp_dim}")
        logger.info(f"Channel混合MLP维度: {args.channel_mlp_dim}")
        logger.info(f"池化方式: {args.cnn_pool}")
        logger.info(f"Dropout: {args.dropout}")
    elif args.model == "cnn_tsmixer_gated":
        use_groupnorm = args.use_groupnorm and not args.no_groupnorm
        logger.info("--- 门控CNN-TSMixer 特定参数 ---")
        logger.info(f"时间补丁大小: {args.patch}")
        logger.info(f"CNN前端通道数: {args.cnn_channels}")
        logger.info(f"CNN前端层数: {args.cnn_layers}")
        logger.info(f"CNN卷积核大小: {args.cnn_kernel}")
        logger.info(f"模型维度: {args.d_model}")
        logger.info(f"TSMixer层数: {args.depth}")
        logger.info(f"Token混合MLP维度: {args.token_mlp_dim}")
        logger.info(f"Channel混合MLP维度: {args.channel_mlp_dim}")
        logger.info(f"池化方式: {args.cnn_pool}")
        logger.info(f"归一化方式: {'GroupNorm' if use_groupnorm else 'BatchNorm'}")
        if use_groupnorm:
            logger.info(f"GroupNorm分组数: {args.gn_groups}")
        logger.info(f"门控机制: 启用自适应CNN-原始输入融合")
        logger.info(f"Dropout: {args.dropout}")
    elif args.model == "tokenpool":
        logger.info("--- TokenPool 特定参数 ---")
        logger.info(f"时间补丁大小: {args.patch}")
        logger.info(f"模型维度: {args.d_model}")
        logger.info(f"TSMixer层数: {args.depth}")
        logger.info(f"Token混合MLP维度: {args.token_mlp_dim}")
        logger.info(f"Channel混合MLP维度: {args.channel_mlp_dim}")
        logger.info(f"池化方式: {args.cnn_pool}")
        logger.info(f"TokenPool注意力头数: {args.tokenpool_heads}")
        logger.info(f"TokenPool注意力dropout: {args.tokenpool_dropout}")
        logger.info(f"TokenPool注意力温度: {args.tokenpool_temperature}")
        logger.info(f"Dropout: {args.dropout}")
    
    logger.info("="*80)

    # 数据集
    train_set, test_set = create_datasets(args)
    # Windows系统使用num_workers=0避免多进程问题
    num_workers = 0 if os.name == 'nt' else 2
    train_loader = train_set.get_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = test_set.get_dataloader(batch_size=args.batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 模型
    model = create_model(args, train_set.n_features, train_set.window_size)

    # 编译
    compile_kwargs = {"learning_rate": args.learning_rate, "weight_decay": args.weight_decay}
    if args.scheduler == "onecycle":
        compile_kwargs.update({
            "scheduler": "onecycle",
            "epochs": args.epochs,
            "steps_per_epoch": len(train_loader)
        })
    elif args.scheduler in ["plateau", "cosine"]:
        compile_kwargs["scheduler"] = args.scheduler

    model.compile(**compile_kwargs)

    # RBM预训练（仅对rbmlstm模型且启用时执行）
    if args.model == "rbmlstm" and args.enable_rbm_pretrain:
        logger.info("="*60)
        logger.info("开始RBM无监督预训练")
        logger.info("="*60)
        logger.info(f"预训练轮数: {args.rbm_epochs}")
        logger.info(f"RBM学习率: {args.rbm_lr}")
        logger.info(f"对比散度步数: {args.rbm_cd_k}")
        logger.info(f"RBM架构: {train_set.n_features} -> {args.rbm_hidden}")
        
        rbm_start = datetime.now()
        model.pretrain_rbm(
            unlabeled_loader=train_loader,
            epochs=args.rbm_epochs,
            lr_rbm=args.rbm_lr,
            cd_k=args.rbm_cd_k,
            device=device
        )
        rbm_end = datetime.now()
        
        logger.info(f"RBM预训练完成，耗时: {rbm_end - rbm_start}")
        logger.info("="*60)

    # 训练
    logger.info("开始训练...")
    if args.early_stopping > 0:
        logger.info(f"启用早停机制，耐心值: {args.early_stopping}")
    else:
        logger.info("未启用早停机制")
    
    start = datetime.now()
    training_results = model.train_model(train_loader, val_loader=val_loader, epochs=args.epochs, 
                                        test_dataset_for_last=test_set, early_stopping_patience=args.early_stopping)
    end = datetime.now()

    # 评估 - 使用训练返回的最佳结果
    logger.info("评估模型...")
    if training_results.get("early_stopped", False):
        # 早停情况：显示最佳验证结果
        best_rmse = training_results.get("best_val_rmse", float('nan'))
        logger.info(f"最优验证结果 (早停): RMSE = {best_rmse:.4f}")
        results = {"rmse": best_rmse, "source": "best_validation"}
        
        # 记录训练过程中的最佳指标
        if hasattr(model, 'best_metrics') and model.best_metrics:
            logger.info("最优训练指标详情:")
            for key, value in model.best_metrics.items():
                if 'rmse' in key or 'score' in key:
                    logger.info(f"  {key}: {value:.4f}")
    else:
        # 正常完成：重新评估最终模型
        results = model.evaluate(val_loader)
        results["source"] = "final_evaluation"

    # 跳过模型保存

    # 记录最终结果
    logger.info("="*80)
    logger.info("实验结果总结")
    logger.info("="*80)
    
    # 根据结果来源显示不同的信息
    if results.get("source") == "best_validation":
        logger.info(f"最优结果 (早停): RMSE = {results['rmse']:.6f} cycles")
    else:
        logger.info(f"最终结果: RMSE = {results['rmse']:.6f} cycles")
    
    # 尝试获取并记录详细的验证指标
    if hasattr(model, 'best_metrics') and model.best_metrics:
        logger.info("详细验证指标:")
        metrics_to_log = ['val_rmse_global', 'val_rmse_last', 'val_score_last', 'train_rmse']
        for metric in metrics_to_log:
            if metric in model.best_metrics:
                value = model.best_metrics[metric]
                if 'rmse' in metric:
                    logger.info(f"  {metric}: {value:.4f} cycles")
                elif 'score' in metric:
                    logger.info(f"  {metric}: {value:.2f}")
                else:
                    logger.info(f"  {metric}: {value:.4f}")
    
    # 记录训练信息
    logger.info(f"训练耗时: {end-start}")
    if training_results.get("early_stopped", False):
        best_epoch = training_results.get("best_epoch", "未知")
        logger.info(f"早停触发: 在第 {best_epoch} 轮达到最佳结果")
    
    # 记录实验标识信息
    logger.info(f"实验时间: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"模型配置: {args.model} + {args.fault}")
    
    # 模型特定的架构摘要
    if args.model == "rbmlstm":
        config_summary = f"RBM{args.rbm_hidden}-LSTM{args.lstm_hidden1}x{args.lstm_hidden2}-FF{args.ff_hidden}"
        if args.enable_rbm_pretrain:
            config_summary += f"-PreTrain{args.rbm_epochs}e"
        logger.info(f"架构摘要: {config_summary}")
    elif args.model == "tsmixer":
        config_summary = f"TSMixer-L{args.tsmixer_layers}-T{args.time_expansion}x{args.feat_expansion}-D{args.dropout}"
        logger.info(f"架构摘要: {config_summary}")
    elif args.model == "tsmixer_sga":
        sga_info = f"-SGA{args.sga_time_rr}x{args.sga_feat_rr}" if args.use_sga else "-NoSGA"
        config_summary = f"TSMixer{sga_info}-L{args.tsmixer_layers}-T{args.time_expansion}x{args.feat_expansion}-P{args.sga_pool}-D{args.dropout}"
        logger.info(f"架构摘要: {config_summary}")
    elif args.model == "transformer":
        config_summary = f"Transformer-D{args.d_model}-H{args.nhead}-L{args.num_layers}-FF{args.dim_feedforward}"
        logger.info(f"架构摘要: {config_summary}")
    elif args.model == "bilstm":
        direction = "Bi" if args.bidirectional else "Uni"
        config_summary = f"{direction}LSTM-H{args.lstm_hidden}-L{args.lstm_layers}-MLP{args.mlp_hidden}"
        logger.info(f"架构摘要: {config_summary}")
    elif args.model == "cnn_tsmixer":
        config_summary = f"CNN{args.cnn_channels}x{args.cnn_layers}K{args.cnn_kernel}-TSMixer{args.d_model}x{args.depth}P{args.patch}-{args.cnn_pool}"
        logger.info(f"架构摘要: {config_summary}")
    elif args.model == "cnn_tsmixer_gated":
        use_groupnorm = args.use_groupnorm and not args.no_groupnorm
        norm_type = "GN" if use_groupnorm else "BN"
        gn_suffix = f"G{args.gn_groups}" if use_groupnorm else ""
        config_summary = f"GatedCNN{args.cnn_channels}x{args.cnn_layers}K{args.cnn_kernel}{norm_type}{gn_suffix}-TSMixer{args.d_model}x{args.depth}P{args.patch}-{args.cnn_pool}"
        logger.info(f"架构摘要: {config_summary}")
    elif args.model == "tokenpool":
        config_summary = f"TokenPool{args.tokenpool_heads}H-T{args.tokenpool_temperature}-TSMixer{args.d_model}x{args.depth}P{args.patch}-{args.cnn_pool}"
        logger.info(f"架构摘要: {config_summary}")
    
    logger.info("="*80)
    
    if log_path:
        logger.info(f"完整日志文件: {log_path}")


if __name__ == "__main__":
    main()
