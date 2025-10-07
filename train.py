#!/usr/bin/env python3
"""
统一训练脚本：支持多种模型架构（TSMixer、Transformer、ECATSMixer 等）
基于 BaseRULModel 的接口，无需额外 Trainer 类
"""
import argparse
import os
import torch
import logging
import sys
from datetime import datetime
import multiprocessing as mp  # NEW
import torch.nn as nn

from dataset import CMAPSSDataset, _xu_window_for
from models.tsmixer_model import TSMixerModel
from models.tsmixer_sga import TSMixerModel as TSMixerSGAModel
from models.transformer_model import TransformerModel
from models.bilstm_model import BiLSTMModel
from models.ellefsen_rbm_lstm_model import EllefsenRBMLSTMModel
from models.cnn_tsmixer_model import CNNMixerRULModel
from models.tsmixer_cnn_gated import CNNMixerGatedRULModel
from models.tsmixer_gated_tokenpool import TokenPoolRULModel
from models.parallel_tsmixer_rul import ParallelTSMixerRUL
from models.tsmixer_eca_model import ECATSMixerModel
from models.tsmixer_ptsa import ModelTSMixerPTSA
from models.tsmixer_sga_kg import TSMixerSGAKGRULModel
from models.tsmixer_mts import MTSTSMixerRULModel
from models.tsmixer_mts_sga import TSMixerMTS_SGA_RULModel
from models.tokenpool_tsmixer_sga import TokenPoolTSMixerSGAModel 

# ======= 性能相关：允许 TF32 + cuDNN benchmark（固定长度时会提速） =======
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # 对固定窗口/尺寸的卷积和 matmul 加速
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='CMAPSS RUL Prediction Training')

    # 模型选择（合并所有模型类型）
    parser.add_argument(
        "--model", type=str, default="transformer",
        choices=[
            "tsmixer", "tsmixer_eca", "tsmixer_sga", "tsmixer_sga_kg", "tsmixer_mts", "tsmixer_mts_sga", 
            "transformer", "bilstm", "rbmlstm",
            "cnn_tsmixer", "cnn_tsmixer_gated", "tokenpool", "tokenpool_sga", "ptsmixer", "tsmixer_ptsa", "tsmixer_ptsa_cond" 
        ],
        help="选择模型架构"
    )

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

    # NEW: 训练性能相关
    parser.add_argument("--num_workers", type=int, default=-1, help="DataLoader worker 数；-1=自动")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader 预取批次数（每个 worker）")
    parser.add_argument("--persistent_workers", action="store_true", default=True, help="DataLoader 持久化 worker")
    parser.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数（显存不够时等效放大 batch）")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="cosine 下的线性 warmup 轮数（compile 需支持）")

    # 通用 Dropout
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

    # === 新增：ECATSMixer 特定参数（前端 ECA + BiTCN）
    parser.add_argument("--use_eca", action="store_true", default=True, help="是否启用 ECA 通道门控")
    parser.add_argument("--eca_kernel", type=int, default=5, help="ECA 卷积核大小（奇数）")
    parser.add_argument("--use_bitcn", action="store_true", default=True, help="是否启用 BiTCN 前端")
    parser.add_argument("--tcn_kernel", type=int, default=3, help="TCN 卷积核大小（奇数）")
    parser.add_argument("--tcn_dilations", type=str, default="1,2", help="TCN 膨胀系数列表，逗号分隔，如 1,2,4")
    parser.add_argument("--tcn_dropout", type=float, default=0.1, help="TCN 前端 dropout")
    parser.add_argument("--tcn_fuse", type=str, default="mean", choices=["mean", "sum", "cat"], help="BiTCN 正反分支融合方式")

    
    # TSMixer-SGA特定参数
    parser.add_argument("--use_sga", action="store_true", help="启用SGA注意力机制")
    parser.add_argument("--sga_time_rr", type=int, default=4, help="SGA时间方向压缩比")
    parser.add_argument("--sga_feat_rr", type=int, default=4, help="SGA特征方向压缩比")
    parser.add_argument("--sga_dropout", type=float, default=0.05, help="SGA内部dropout")
    parser.add_argument("--sga_pool", type=str, default="mean", 
                        choices=["mean", "last", "weighted"], help="TSMixer-SGA池化方式")
    parser.add_argument("--lambda_prior", type=float, default=0.5, help="知识引导SGA的先验门控权重λ")
    parser.add_argument("--kg_pool", type=str, default="weighted",
                        choices=["mean", "last", "weighted"], help="TSMixer-SGA-KG池化方式")
    
    # MTS-TSMixer特定参数（多尺度时间混合）
    parser.add_argument("--mts_gate_hidden", type=int, default=16, help="MTS门控MLP隐藏层维度")
    parser.add_argument("--mts_scales", type=str, default="3-1,3-2,5-3,7-4", 
                        help="MTS时间尺度配置，格式：kernel-dilation,...（例如：3-1,3-2,5-3,7-4）")
    
    # MTS-TSMixer-SGA特定参数（多尺度+双轴注意力）
    parser.add_argument("--mts_sga_time_hidden", type=int, default=16, help="MTS-SGA时间轴注意力隐藏层")
    parser.add_argument("--mts_sga_feat_hidden", type=int, default=16, help="MTS-SGA特征轴注意力隐藏层")
    parser.add_argument("--mts_sga_dropout", type=float, default=0.05, help="MTS-SGA注意力dropout")
    
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
    
    # TokenPool-SGA特定参数（复用上面的tokenpool参数 + 新增SGA参数）
    parser.add_argument("--tokenpool_sga_time_hidden", type=int, default=24, help="TokenPool-SGA时间轴注意力隐藏层")
    parser.add_argument("--tokenpool_sga_feat_hidden", type=int, default=24, help="TokenPool-SGA特征轴注意力隐藏层")
    parser.add_argument("--tokenpool_sga_dropout", type=float, default=0.05, help="TokenPool-SGA注意力dropout")
    parser.add_argument("--tokenpool_sga_fuse", type=str, default="add", choices=["add", "hadamard"], 
                        help="TokenPool-SGA注意力融合方式")
    parser.add_argument("--tokenpool_sga_every_k", type=int, default=0, 
                        help="TokenPool-SGA每k层插入一次SGA（0=仅尾部）")

    # pTSMixer（并行时间/特征双支）特定参数
    parser.add_argument("--pt_depth", type=int, default=6, help="pTSMixer Block 层数")
    parser.add_argument("--pt_ch_expand", type=int, default=4, help="特征分支 MLP 扩展倍数")
    parser.add_argument("--pt_t_kernel", type=int, default=7, help="时间分支深度可分离卷积核大小")
    parser.add_argument("--pt_t_dilation", type=int, default=2, help="时间分支卷积膨胀系数")
    parser.add_argument("--pt_t_ffn_expand", type=int, default=1, help="时间分支 pointwise FFN 扩展倍数")
    parser.add_argument("--pt_droppath", type=float, default=0.10, help="分层 DropPath 最大比例")
    parser.add_argument("--pt_branch_dropout", type=float, default=0.00, help="分支内的 dropout")
    parser.add_argument("--pt_pooling", type=str, default="token", choices=["token", "avg"], help="pTSMixer 头部池化方式")
    parser.add_argument("--pt_input_dropout", type=float, default=0.00, help="输入层 dropout")

    # === NEW: TSMixer+PTSA 特定参数 ===
    parser.add_argument("--hidden_channels", type=int, default=128, help="TSMixer-PTSA 主干通道数")
    parser.add_argument("--ptsa_every_k", type=int, default=2, help="每隔 k 个 MixerBlock 插入一个 PTSA 块")
    parser.add_argument("--ptsa_heads", type=int, default=6, help="PTSA 注意力头数")
    parser.add_argument("--ptsa_local_window", type=int, default=12, help="同尺度局部窗口大小（A邻域）")
    parser.add_argument("--ptsa_topk", type=int, default=16, help="稀疏 TopK 保留的键数量")
    parser.add_argument("--ptsa_levels", type=int, default=2, help="金字塔层数（含当前尺度向下的层数）")
    parser.add_argument("--ptsa_parent_neigh", type=int, default=1, help="父尺度邻域半径（P邻域）")
    parser.add_argument("--ptsa_dropout", type=float, default=0.0, help="PTSA 内部 dropout")
    parser.add_argument("--distill_type", type=str, default="conv", choices=["conv","maxpool"], help="PTSA 后的下采样类型")
    parser.add_argument("--distill_stride", type=int, default=2, help="时间维下采样步幅（通常=2）")
    parser.add_argument("--reduce_channels", type=float, default=1.0, help="蒸馏后通道缩放比例（如 0.75）")
    parser.add_argument("--drop_path", type=float, default=0.0, help="可选 Stochastic Depth 比例")


    # === TSMixer+PTSA+Cond 条件门控参数 ===
    parser.add_argument("--cond_dim", type=int, default=3, help="条件变量维度（如工况设置数）")
    parser.add_argument("--film_hidden", type=int, default=32, help="FiLM隐藏层维度")
    parser.add_argument("--time_kernel", type=int, default=11, help="时间门控卷积核大小")
    parser.add_argument("--use_post_gate", action="store_true", default=False, help="是否使用后置门控")




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
    parser.add_argument("--val_batch_mul", type=float, default=2.0,
                    help="验证集 batch 相对训练 batch 的放大倍数（提高验证吞吐）")
   

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


def _parse_tcn_dilations(dils: str):
    try:
        return [int(x.strip()) for x in dils.split(",") if x.strip() != ""]
    except Exception:
        logging.getLogger(__name__).warning(f"无法解析 tcn_dilations='{dils}'，改用默认 [1,2]")
        return [1, 2]


def _parse_mts_scales(scales_str: str, time_expansion: int = 4):
    """
    解析 MTS 时间尺度配置字符串
    格式：kernel-dilation,...
    例如：3-1,3-2,5-3,7-4
    返回：[{"kernel": 3, "dilation": 1, "expansion": time_expansion}, ...]
    """
    try:
        scales = []
        for s in scales_str.split(","):
            s = s.strip()
            if s:
                parts = s.split("-")
                if len(parts) == 2:
                    k, d = int(parts[0]), int(parts[1])
                    scales.append({"kernel": k, "dilation": d, "expansion": time_expansion})
        if scales:
            return scales
    except Exception as e:
        logging.getLogger(__name__).warning(f"无法解析 mts_scales='{scales_str}'，使用默认配置: {e}")
    
    # 默认配置：4个尺度（短/中/长/超长）
    return [
        {"kernel": 3, "dilation": 1, "expansion": time_expansion},
        {"kernel": 3, "dilation": 2, "expansion": time_expansion},
        {"kernel": 5, "dilation": 3, "expansion": time_expansion},
        {"kernel": 7, "dilation": 4, "expansion": time_expansion},
    ]


def create_datasets(args):
    print(f"=== 创建 {args.fault} 数据集 ===")

    # 为多工况数据集（FD002/FD004）使用工况设置+传感器，其他使用14传感器
    if args.fault in ["FD002", "FD004"]:
        print(f"检测到多工况数据集 {args.fault}，使用工况设置+传感器特征")
        features_mode = "all"  # 3个工况设置 + 21个传感器 = 24特征
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


def _auto_workers(args):
    if args.num_workers is not None and args.num_workers >= 0:
        return args.num_workers
    # 自动：一半 CPU，范围 [2, 8]
    try:
        return min(8, max(2, mp.cpu_count() // 2))
    except Exception:
        # Windows 上多进程易出问题，兜底 0
        return 0


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

    # === ECA + BiTCN + TSMixer ===
    elif args.model == "tsmixer_eca":
        model = ECATSMixerModel(
            input_size=input_size, seq_len=seq_len,
            num_layers=args.tsmixer_layers,
            time_expansion=args.time_expansion,
            feat_expansion=args.feat_expansion,
            mixer_dropout=args.dropout,  # 与原 dropout 对齐，用于 Mixer 内部
            # 前端
            use_eca=args.use_eca,
            eca_kernel=args.eca_kernel,
            use_bitcn=args.use_bitcn,
            tcn_kernel=args.tcn_kernel,
            tcn_dilations=_parse_tcn_dilations(args.tcn_dilations),
            tcn_dropout=args.tcn_dropout,
            tcn_fuse=args.tcn_fuse
        )
    
    # === TSMixer + SGA 注意力 ===
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
    
    # === TSMixer + SGA + 知识引导 (Knowledge-Guided) ===
    elif args.model == "tsmixer_sga_kg":
        model = TSMixerSGAKGRULModel(
            input_size=input_size, seq_len=seq_len,
            num_layers=args.tsmixer_layers,
            time_expansion=args.time_expansion,
            feat_expansion=args.feat_expansion,
            dropout=args.dropout,
            rr_time=args.sga_time_rr,
            rr_feat=args.sga_feat_rr,
            lambda_prior=args.lambda_prior,
            sga_dropout=args.sga_dropout,
            pool=args.kg_pool
        )
    
    # === MTS-TSMixer (多尺度时间混合) ===
    elif args.model == "tsmixer_mts":
        time_scales = _parse_mts_scales(args.mts_scales, args.time_expansion)
        model = MTSTSMixerRULModel(
            input_size=input_size,
            seq_len=seq_len,
            num_layers=args.tsmixer_layers,
            time_expansion=args.time_expansion,
            feat_expansion=args.feat_expansion,
            dropout=args.dropout,
            gate_hidden=args.mts_gate_hidden,
            time_scales=time_scales
        )
    
    # === MTS-TSMixer-SGA (多尺度时间混合 + 双轴注意力) ===
    elif args.model == "tsmixer_mts_sga":
        model = TSMixerMTS_SGA_RULModel(
            input_size=input_size,
            seq_len=seq_len,
            tsmixer_layers=args.tsmixer_layers,
            time_expansion=args.time_expansion,
            feat_expansion=args.feat_expansion,
            dropout=args.dropout,
            mts_scales=args.mts_scales,
            mts_gate_hidden=args.mts_gate_hidden,
            sga_time_hidden=args.mts_sga_time_hidden,
            sga_feat_hidden=args.mts_sga_feat_hidden,
            sga_dropout=args.mts_sga_dropout
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
            pool=args.cnn_pool,
            tokenpool_heads=args.tokenpool_heads,
            tokenpool_dropout=args.tokenpool_dropout,
            tokenpool_temperature=args.tokenpool_temperature
        )
    
    # === TokenPool + TSMixer + SGA ===
    elif args.model == "tokenpool_sga":
        model = TokenPoolTSMixerSGAModel(
            input_size=input_size,
            seq_len=seq_len,
            patch=args.patch,
            d_model=args.d_model,
            depth=args.depth,
            token_mlp_dim=args.token_mlp_dim,
            channel_mlp_dim=args.channel_mlp_dim,
            dropout=args.dropout,
            pool=args.cnn_pool,
            tokenpool_heads=args.tokenpool_heads,
            tokenpool_dropout=args.tokenpool_dropout,
            tokenpool_temperature=args.tokenpool_temperature,
            use_sga=True,
            sga_time_hidden=args.tokenpool_sga_time_hidden,
            sga_feat_hidden=args.tokenpool_sga_feat_hidden,
            sga_dropout=args.tokenpool_sga_dropout,
            sga_fuse=args.tokenpool_sga_fuse,
            sga_every_k=args.tokenpool_sga_every_k
        )

    elif args.model == "ptsmixer":
        model = ParallelTSMixerRUL(
            input_size=input_size,
            seq_len=seq_len,
            depth=args.pt_depth,
            ch_expand=args.pt_ch_expand,
            t_kernel=args.pt_t_kernel,
            t_dilation=args.pt_t_dilation,
            t_ffn_expand=args.pt_t_ffn_expand,
            droppath_base=args.pt_droppath,
            branch_dropout=args.pt_branch_dropout,
            pooling=args.pt_pooling,
            input_dropout=args.pt_input_dropout,
        )

    elif args.model == "tsmixer_ptsa":
        model = ModelTSMixerPTSA(
            input_size=input_size, seq_len=seq_len, out_channels=1,
            channels=args.hidden_channels,
            depth=args.tsmixer_layers,                   # 继续复用你的 --tsmixer_layers
            time_expansion=args.time_expansion,
            feat_expansion=args.feat_expansion,
            dropout=args.dropout,
            # PTSA & Distill
            use_ptsa=True,
            ptsa_every_k=args.ptsa_every_k,
            ptsa_heads=args.ptsa_heads,
            ptsa_local_window=args.ptsa_local_window,
            ptsa_topk=args.ptsa_topk,
            ptsa_levels=args.ptsa_levels,
            ptsa_parent_neigh=args.ptsa_parent_neigh,
            ptsa_dropout=args.ptsa_dropout,
            distill_type=args.distill_type,
            distill_stride=args.distill_stride,
            reduce_channels=args.reduce_channels,
            drop_path=args.drop_path,
            # 训练增强 hint（BaseRULModel 会读取）
            accum_steps=args.grad_accum,
            use_amp=True,
        )

    elif args.model == "tsmixer_ptsa_cond":
        from models.tsmixer_ptsa_cond_model import ModelTSMixerPTSACOND
        model = ModelTSMixerPTSACOND(
            input_size=input_size, seq_len=seq_len, out_channels=1,
            channels=args.hidden_channels, depth=args.tsmixer_layers,
            time_expansion=args.time_expansion, feat_expansion=args.feat_expansion,
            dropout=args.dropout,
            use_ptsa=True, ptsa_every_k=args.ptsa_every_k, ptsa_heads=args.ptsa_heads,
            ptsa_local_window=args.ptsa_local_window, ptsa_topk=args.ptsa_topk,
            ptsa_levels=args.ptsa_levels, ptsa_parent_neigh=args.ptsa_parent_neigh,
            ptsa_dropout=args.ptsa_dropout,
            distill_type=args.distill_type, distill_stride=args.distill_stride,
            reduce_channels=args.reduce_channels, drop_path=args.drop_path,
            cond_dim=args.cond_dim, eca_kernel=args.eca_kernel,  # ← 直接用现有 eca_kernel
            time_kernel=args.time_kernel, use_post_gate=args.use_post_gate
        )



    else:
        raise ValueError(f"未支持的模型类型: {args.model}")
    return model


def _safe_compile_model(model, compile_kwargs, steps_per_epoch: int):
    """
    尝试以完整参数编译；若模型的 compile 不支持某些 key（如 warmup_epochs/epochs），
    自动剔除后重试，避免 TypeError 直接中断。
    """
    logger = logging.getLogger(__name__)
    try:
        # 对 onecycle 这类按 step 设计的调度器，确保提供 steps_per_epoch（若上层没给）
        if compile_kwargs.get("scheduler", "") == "onecycle" and "steps_per_epoch" not in compile_kwargs:
            compile_kwargs["steps_per_epoch"] = steps_per_epoch
        model.compile(**compile_kwargs)
        return
    except TypeError as e:
        msg = str(e)
        removed = []
        for k in ["warmup_epochs", "epochs", "steps_per_epoch"]:
            if k in compile_kwargs and k in msg:
                removed.append(k)
        for k in removed:
            compile_kwargs.pop(k, None)
        model.compile(**compile_kwargs)
        if removed:
            logger.info(f"{model.__class__.__name__}.compile 不支持参数 {removed}，已自动忽略并继续。")


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
    logger.info(f"warmup_epochs: {args.warmup_epochs}")
    logger.info(f"grad_accum: {args.grad_accum}")
    logger.info(f"val_batch_mul: {args.val_batch_mul}")

    # 模型特定参数...
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

    elif args.model == "tsmixer_eca":
        logger.info("--- ECATSMixer 特定参数（ECA+BiTCN→TSMixer） ---")
        logger.info(f"TSMixer层数: {args.tsmixer_layers}")
        logger.info(f"时间混合层扩展倍数: {args.time_expansion}")
        logger.info(f"特征混合层扩展倍数: {args.feat_expansion}")
        logger.info(f"Mixer Dropout: {args.dropout}")
        logger.info(f"use_eca: {args.use_eca}, eca_kernel: {args.eca_kernel}")
        logger.info(f"use_bitcn: {args.use_bitcn}, tcn_kernel: {args.tcn_kernel}")
        logger.info(f"tcn_dilations: {args.tcn_dilations}, tcn_dropout: {args.tcn_dropout}, tcn_fuse: {args.tcn_fuse}")

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

    elif args.model == "ptsmixer":
        logger.info("--- pTSMixer（并行双支） 特定参数 ---")
        logger.info(f"深度: {args.pt_depth}")
        logger.info(f"特征分支扩展倍数: {args.pt_ch_expand}")
        logger.info(f"时间核大小: {args.pt_t_kernel}")
        logger.info(f"时间膨胀系数: {args.pt_t_dilation}")
        logger.info(f"时间支 FFN 扩展: {args.pt_t_ffn_expand}")
        logger.info(f"DropPath最大比率: {args.pt_droppath}")
        logger.info(f"分支内dropout: {args.pt_branch_dropout}")
        logger.info(f"池化方式: {args.pt_pooling}")
        logger.info(f"输入dropout: {args.pt_input_dropout}")

    elif args.model == "tsmixer_ptsa":
        logger.info("--- TSMixer + PTSA（金字塔稀疏注意力） 特定参数 ---")
        logger.info(f"Hidden Channels: {args.hidden_channels}")
        logger.info(f"TSMixer层数: {args.tsmixer_layers}")
        logger.info(f"时间混合扩展: {args.time_expansion} | 特征混合扩展: {args.feat_expansion}")
        logger.info(f"Dropout: {args.dropout}")
        logger.info(f"PTSA: every_k={args.ptsa_every_k}, heads={args.ptsa_heads}, "
                    f"W(local)={args.ptsa_local_window}, TopK={args.ptsa_topk}, "
                    f"levels={args.ptsa_levels}, parent_neigh={args.ptsa_parent_neigh}, "
                    f"attn_dropout={args.ptsa_dropout}")
        logger.info(f"Distill: type={args.distill_type}, stride={args.distill_stride}, "
                    f"reduce_channels={args.reduce_channels}, drop_path={args.drop_path}")
        
    elif args.model == "tsmixer_ptsa_cond":
        logger.info("--- TSMixer + PTSA + 条件双门控（ECA×Time×FiLM） ---")
        logger.info(f"Hidden Channels: {args.hidden_channels} | Layers: {args.tsmixer_layers}")
        logger.info(f"PTSA: k={args.ptsa_every_k}, heads={args.ptsa_heads}, W={args.ptsa_local_window},"
                    f" TopK={args.ptsa_topk}, Lv={args.ptsa_levels}, P={args.ptsa_parent_neigh}")
        logger.info(f"CondGate: cond_dim={args.cond_dim}, ECAk={args.eca_kernel}, Timek={args.time_kernel}, post={args.use_post_gate}")



    logger.info("="*80)

    # ================== 数据集 & DataLoader 优化 ==================
    train_set, test_set = create_datasets(args)

    # 自动计算/或使用用户指定 worker 数
    num_workers = _auto_workers(args)
    # 兼容 Windows：如果系统是 Windows，强制降级（避免多进程问题）
    if os.name == 'nt':
        num_workers = 0
        args.persistent_workers = False

    # 高级 DataLoader 形参，若不支持自动回退
    try:
        train_loader = train_set.get_dataloader(
            batch_size=args.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers
        )
        val_bs = max(1, int(args.batch_size * args.val_batch_mul))
        # val_loader = test_set.get_dataloader(
        #     batch_size=max(1, args.batch_size * 2), shuffle=False,
        #     num_workers=num_workers, pin_memory=True,
        #     prefetch_factor=args.prefetch_factor,
        #     persistent_workers=args.persistent_workers
        # )
        val_loader = test_set.get_dataloader(
            batch_size=val_bs, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers
        )
    except TypeError:
        
        train_loader = train_set.get_dataloader(
            batch_size=args.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        val_bs = max(1, int(args.batch_size * args.val_batch_mul))
        # val_loader = test_set.get_dataloader(
        #     batch_size=max(1, args.batch_size * 2), shuffle=False,
        #     num_workers=num_workers, pin_memory=True
        # )
        val_loader = test_set.get_dataloader(
            batch_size=val_bs, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )


    # ================== 模型 ==================
    model = create_model(args, train_set.n_features, train_set.window_size)

    # === 性能 hint（只有在 BaseRULModel 里读取才会生效，不影响兼容） ===
    model.accum_steps = max(1, int(args.grad_accum))  # 梯度累积（若 train_model 支持则生效）
    model.use_amp = True                              # AMP hint（若 train_model 支持则生效）

    # ================== 编译（学习率调度） ==================
    compile_kwargs = {"learning_rate": args.learning_rate, "weight_decay": args.weight_decay}

    if args.model == "ptsmixer":
        # pTSMixer 内置: cosine / plateau / none
        if args.scheduler == "onecycle":
            logging.getLogger(__name__).info("pTSMixer 暂不支持 onecycle，已自动切换到 cosine 调度器。")
            compile_kwargs["scheduler"] = "cosine"
        else:
            compile_kwargs["scheduler"] = args.scheduler
        compile_kwargs["epochs"] = args.epochs
        compile_kwargs["warmup_epochs"] = args.warmup_epochs
    else:
        if args.scheduler == "onecycle":
            compile_kwargs.update({
                "scheduler": "onecycle",
                "epochs": args.epochs,
                "steps_per_epoch": len(train_loader)
            })
        elif args.scheduler in ["plateau", "cosine"]:
            compile_kwargs["scheduler"] = args.scheduler
            compile_kwargs["epochs"] = args.epochs
            compile_kwargs["warmup_epochs"] = args.warmup_epochs
            # 关键：cosine + warmup 走“按 step”的调度
            if args.scheduler == "cosine" and args.warmup_epochs > 0:
                compile_kwargs["steps_per_epoch"] = len(train_loader)


    # NOTE: 关键变化——安全编译，自动忽略模型不支持的 compile 关键字
    _safe_compile_model(model, compile_kwargs, steps_per_epoch=len(train_loader))

    # ================== RBM 预训练（可选） ==================
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

    # ================== 训练 ==================
    logger.info("开始训练...")
    if args.early_stopping > 0:
        logger.info(f"启用早停机制，耐心值: {args.early_stopping}")
    else:
        logger.info("未启用早停机制")

    start = datetime.now()
    training_results = model.train_model(
        train_loader, val_loader=val_loader, epochs=args.epochs,
        test_dataset_for_last=test_set, early_stopping_patience=args.early_stopping
    )
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
                    logger.info(f"  {key}: {value:.4f}" if 'rmse' in key else f"  {key}: {value:.2f}")
    else:
        # 正常完成：重新评估最终模型
        results = model.evaluate(val_loader)
        results["source"] = "final_evaluation"

    # 记录最终结果
    logger.info("="*80)
    logger.info("实验结果总结")
    logger.info("="*80)

    if results.get("source") == "best_validation":
        logger.info(f"最优结果 (早停): RMSE = {results['rmse']:.6f} cycles")
    else:
        logger.info(f"最终结果: RMSE = {results['rmse']:.6f} cycles")

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

    logger.info(f"训练耗时: {end-start}")
    if training_results.get("early_stopped", False):
        best_epoch = training_results.get("best_epoch", "未知")
        logger.info(f"早停触发: 在第 {best_epoch} 轮达到最佳结果")

    logger.info(f"实验时间: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"模型配置: {args.model} + {args.fault}")

    # 架构摘要
    if args.model == "rbmlstm":
        config_summary = f"RBM{args.rbm_hidden}-LSTM{args.lstm_hidden1}x{args.lstm_hidden2}-FF{args.ff_hidden}"
        if args.enable_rbm_pretrain:
            config_summary += f"-PreTrain{args.rbm_epochs}e"
        logger.info(f"架构摘要: {config_summary}")

    elif args.model == "tsmixer":
        config_summary = f"TSMixer-L{args.tsmixer_layers}-T{args.time_expansion}x{args.feat_expansion}-D{args.dropout}"
        logger.info(f"架构摘要: {config_summary}")

    elif args.model == "tsmixer_eca":
        config_summary = (f"ECATSMixer-L{args.tsmixer_layers}"
                          f"-T{args.time_expansion}x{args.feat_expansion}"
                          f"-ECA{k if (k:=args.eca_kernel) else 'off'}"
                          f"-BiTCN{'on' if args.use_bitcn else 'off'}"
                          f"-dils{args.tcn_dilations}-fuse{args.tcn_fuse}")
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

    elif args.model == "ptsmixer":
        config_summary = (f"pTSMixer-Depth{args.pt_depth}"
                          f"-Tker{args.pt_t_kernel}Dil{args.pt_t_dilation}"
                          f"-Cexp{args.pt_ch_expand}-DP{args.pt_droppath}-{args.pt_pooling}")
        logger.info(f"架构摘要: {config_summary}")

    elif args.model == "tsmixer_ptsa":
        config_summary = (f"TSMixPTSA-L{args.tsmixer_layers}"
                        f"-C{args.hidden_channels}"
                        f"-T{args.time_expansion}x{args.feat_expansion}"
                        f"-PTSA(k={args.ptsa_every_k},H{args.ptsa_heads},W{args.ptsa_local_window},K{args.ptsa_topk},Lv{args.ptsa_levels},P{args.ptsa_parent_neigh})"
                        f"-DS{args.distill_type}{args.distill_stride}x-rC{args.reduce_channels}")
        logger.info(f"架构摘要: {config_summary}")



    logger.info("="*80)

    if log_path:
        logger.info(f"完整日志文件: {log_path}")


if __name__ == "__main__":
    main()
