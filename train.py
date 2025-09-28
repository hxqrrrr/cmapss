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
from dataset import CMAPSSDataset
from models.tsmixer_model import TSMixerModel
from models.transformer_model import TransformerModel


def parse_args():
    parser = argparse.ArgumentParser(description='CMAPSS RUL Prediction Training')

    # 模型选择
    parser.add_argument("--model", type=str, default="transformer",
                        choices=["tsmixer", "transformer"], help="选择模型架构")

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

    # 学习率调度器
    parser.add_argument("--scheduler", type=str, default="onecycle",
                        choices=["onecycle", "plateau", "cosine"], help="学习率调度器")

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
    train_set = CMAPSSDataset(
        data_dir=args.data_dir, fault_mode=args.fault, split="train", preset="xu2023"
    )
    test_set = CMAPSSDataset(
        data_dir=args.data_dir, fault_mode=args.fault, split="test",
        preset="xu2023", scaler=train_set.get_scaler()
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
    else:
        raise ValueError(f"未支持的模型类型: {args.model}")
    return model


def main():
    args = parse_args()
    log_path = setup_logging(args)
    logger = logging.getLogger(__name__)

    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == "auto" else torch.device(args.device)
    logger.info(f"使用设备: {device}")

    # 数据集
    train_set, test_set = create_datasets(args)
    train_loader = train_set.get_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = test_set.get_dataloader(batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

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

    # 训练
    logger.info("开始训练...")
    start = datetime.now()
    model.train_model(train_loader, val_loader=val_loader, epochs=args.epochs, test_dataset_for_last=test_set)
    end = datetime.now()

    # 评估
    logger.info("评估模型...")
    results = model.evaluate(val_loader)

    # 保存模型
    model_path = f"{args.model}_{args.fault}.pth"
    if hasattr(model, "model"):
        torch.save(model.model.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存到: {model_path}")

    logger.info(f"最终结果: {results}")
    logger.info(f"训练耗时: {end-start}")
    if log_path:
        logger.info(f"日志文件: {log_path}")


if __name__ == "__main__":
    main()
