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
        # {
        # "name": "FD002 | TSMixer-PTSA + CondGate | L6 C128 k=2 W12 TopK10 Lv1",
        # "cmd": ["python","train.py",
        #     "--model","tsmixer_ptsa_cond","--fault","FD002",
        #     "--batch_size","640","--epochs","180",
        #     "--learning_rate","0.00035","--weight_decay","0.00008",
        #     "--scheduler","cosine","--warmup_epochs","8","--early_stopping","32",
        #     "--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8",
        #     "--hidden_channels","128","--dropout","0.04",
        #     "--ptsa_every_k","2","--ptsa_heads","6","--ptsa_local_window","12","--ptsa_topk","10",
        #     "--ptsa_levels","1","--ptsa_parent_neigh","1","--ptsa_dropout","0.0",
        #     "--distill_type","conv","--distill_stride","2","--reduce_channels","0.95","--drop_path","0.05",
        #     "--cond_dim","3","--eca_kernel","5","--time_kernel","11",
        #     "--val_batch_mul","8"
        # ]
        # },

        # {
        # "name": "🎯 | PTSA(FD002) HiAcc | L6 C128 k=2 H6 W12 TopK12 Lv1 rC1.0",
        # "cmd": ["python","train.py",
        #     "--model","tsmixer_ptsa","--fault","FD002",
        #     "--batch_size","640","--epochs","180",
        #     "--learning_rate","0.00033","--weight_decay","0.00008",
        #     "--scheduler","cosine","--warmup_epochs","8","--early_stopping","32",
        #     "--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8",
        #     "--hidden_channels","128","--dropout","0.04",
        #     "--ptsa_every_k","2","--ptsa_heads","6",
        #     "--ptsa_local_window","12","--ptsa_topk","12",
        #     "--ptsa_levels","1","--ptsa_parent_neigh","1","--ptsa_dropout","0.0",
        #     "--distill_type","conv","--distill_stride","2","--reduce_channels","1.0",
        #     "--drop_path","0.05",
        #     "--val_batch_mul","8"
        # ]
        # },
        # {
        # "name": "🔬 | PTSA(FD002) TopK Sweep | k=2 W12 Lv1 rC0.95",
        # "cmd": ["bash","-lc",
        #     "for K in 8 10 12; do python train.py --model tsmixer_ptsa --fault FD002 \
        #     --batch_size 640 --epochs 180 --learning_rate 0.00035 --weight_decay 0.00008 \
        #     --scheduler cosine --warmup_epochs 8 --early_stopping 32 \
        #     --tsmixer_layers 6 --time_expansion 4 --feat_expansion 8 \
        #     --hidden_channels 128 --dropout 0.04 \
        #     --ptsa_every_k 2 --ptsa_heads 6 --ptsa_local_window 12 --ptsa_topk $K \
        #     --ptsa_levels 1 --ptsa_parent_neigh 1 --ptsa_dropout 0.0 \
        #     --distill_type conv --distill_stride 2 --reduce_channels 0.95 \
        #     --drop_path 0.05 --val_batch_mul 8; done"
        # ]
        # }





        # {
        # "name": "🧩 | FD002-配方A（更稳稀疏）| H120 L6 T4x8 W12 TopK10 Lv1 rC0.95",
        # "cmd": ["python","train.py",
        # "--model","tsmixer_ptsa","--fault","FD002",
        # "--batch_size","640","--epochs","180",
        # "--learning_rate","0.00033","--weight_decay","0.00008",
        # "--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8",
        # "--hidden_channels","120","--dropout","0.04",
        # "--scheduler","cosine","--warmup_epochs","8","--early_stopping","32",
        # "--ptsa_every_k","2","--ptsa_heads","6","--ptsa_local_window","12","--ptsa_topk","10",
        # "--ptsa_levels","1","--ptsa_parent_neigh","1",
        # "--distill_type","conv","--distill_stride","2","--reduce_channels","0.95"
        # ]
        # },
        # {
        # "name": "🧩 | FD002-配方B（更宽视域+5头）| H120 L6 T4x8 W14 TopK12 Lv1 rC0.95",
        # "cmd": ["python","train.py",
        # "--model","tsmixer_ptsa","--fault","FD002",
        # "--batch_size","640","--epochs","180",
        # "--learning_rate","0.00032","--weight_decay","0.00008",
        # "--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8",
        # "--hidden_channels","120","--dropout","0.05",
        # "--scheduler","cosine","--warmup_epochs","8","--early_stopping","32",
        # "--ptsa_every_k","2","--ptsa_heads","5","--ptsa_local_window","14","--ptsa_topk","12",
        # "--ptsa_levels","1","--ptsa_parent_neigh","1",
        # "--distill_type","conv","--distill_stride","2","--reduce_channels","0.95"
        # ]
        # },
        # {
        # "name": "🧩 | FD002-对拍：无父邻域P | H120 L6 T4x8 W12 TopK10 Lv1 P0",
        # "cmd": ["python","train.py",
        # "--model","tsmixer_ptsa","--fault","FD002",
        # "--batch_size","640","--epochs","180",
        # "--learning_rate","0.00033","--weight_decay","0.00008",
        # "--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8",
        # "--hidden_channels","120","--dropout","0.04",
        # "--scheduler","cosine","--warmup_epochs","8","--early_stopping","32",
        # "--ptsa_every_k","2","--ptsa_heads","6","--ptsa_local_window","12","--ptsa_topk","10",
        # "--ptsa_levels","1","--ptsa_parent_neigh","0",
        # "--distill_type","conv","--distill_stride","2","--reduce_channels","0.95"
        # ]
        # },
        # {
        # "name": "🧩 | FD002-大有效批（accum=2）| H120 L6 T4x8 W12 TopK12 Lv1",
        # "cmd": ["python","train.py",
        # "--model","tsmixer_ptsa","--fault","FD002",
        # "--batch_size","640","--epochs","180","--grad_accum","2",
        # "--learning_rate","0.00030","--weight_decay","0.00008",
        # "--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8",
        # "--hidden_channels","120","--dropout","0.04",
        # "--scheduler","cosine","--warmup_epochs","8","--early_stopping","32",
        # "--ptsa_every_k","2","--ptsa_heads","6","--ptsa_local_window","12","--ptsa_topk","12",
        # "--ptsa_levels","1","--ptsa_parent_neigh","1",
        # "--distill_type","conv","--distill_stride","2","--reduce_channels","1.0"
        # ]
        # }


        # {
        # "name": "🏆 | Baseline†Best · Dropout 0.04",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00032","--weight_decay","0.00008","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.04","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "⏱ | Longer Warmup · LR微调(-) · Dropout 0.04",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00030","--weight_decay","0.00008","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.04","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","10","--early_stopping","32"]
        # },
        # {
        # "name": "🧱 | Dilations 1,2,4 · 感受野增强",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2,4","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00032","--weight_decay","0.00008","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.04","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # }






    #         {
    #     "name": "🧩 | FE×8 · TSL=6 · 保守强化：dils=1,2 · wd=1e-4 · warmup=10",
    #     "cmd": ["python","train.py",
    #     "--model","tsmixer_eca",
    #     "--fault","FD002",
    #     "--use_eca","--eca_kernel","5",
    #     "--use_bitcn","--tcn_kernel","3",
    #     "--tcn_dilations","1,2","--tcn_fuse","mean",
    #     "--batch_size","576",
    #     "--epochs","180",
    #     "--learning_rate","0.00031",
    #     "--weight_decay","0.00010",
    #     "--tsmixer_layers","6",
    #     "--time_expansion","4",
    #     "--feat_expansion","8",
    #     "--dropout","0.05",
    #     "--tcn_dropout","0.10",
    #     "--scheduler","cosine",
    #     "--warmup_epochs","10",
    #     "--early_stopping","32"
    #     ]
    # },
    # {
    #     "name": "🛰️ | FE×8 · TSL=6 · 轻扩感受野：dils=1,2,3 · lr=3.3e-4",
    #     "cmd": ["python","train.py",
    #     "--model","tsmixer_eca",
    #     "--fault","FD002",
    #     "--use_eca","--eca_kernel","5",
    #     "--use_bitcn","--tcn_kernel","3",
    #     "--tcn_dilations","1,2,3","--tcn_fuse","mean",
    #     "--batch_size","576",
    #     "--epochs","180",
    #     "--learning_rate","0.00033",
    #     "--weight_decay","0.00012",
    #     "--tsmixer_layers","6",
    #     "--time_expansion","4",
    #     "--feat_expansion","8",
    #     "--dropout","0.05",
    #     "--tcn_dropout","0.10",
    #     "--scheduler","cosine",
    #     "--warmup_epochs","8",
    #     "--early_stopping","32"
    #     ]
    # },
    # {
    #     "name": "🔧 | FE×8 · TSL=6 · 前段加密：dils=1,2,2 · warmup=12 · lr=3.0e-4",
    #     "cmd": ["python","train.py",
    #     "--model","tsmixer_eca",
    #     "--fault","FD002",
    #     "--use_eca","--eca_kernel","5",
    #     "--use_bitcn","--tcn_kernel","3",
    #     "--tcn_dilations","1,2,2","--tcn_fuse","mean",
    #     "--batch_size","576",
    #     "--epochs","180",
    #     "--learning_rate","0.00030",
    #     "--weight_decay","0.00012",
    #     "--tsmixer_layers","6",
    #     "--time_expansion","4",
    #     "--feat_expansion","8",
    #     "--dropout","0.04",
    #     "--tcn_dropout","0.10",
    #     "--scheduler","cosine",
    #     "--warmup_epochs","12",
    #     "--early_stopping","32"
    #     ]
    # }
        # {
        # "name": "🔧A1 | FE×8 · TSL=6 · Dropout 0.04 · wd 8e-5 · dils 1,2 · mean",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00032","--weight_decay","0.00008","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.04","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🔧A2 | FE×8 · TSL=6 · Dropout 0.03 · wd 8e-5 · dils 1,2 · mean",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00032","--weight_decay","0.00008","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.03","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🔧B1 | FE×8 · TSL=6 · Dropout 0.04 · wd 6e-5 · dils 1,2 · mean",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00032","--weight_decay","0.00006","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.04","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🔧B2 | FE×8 · TSL=6 · Dropout 0.04 · wd 1e-4 · dils 1,2 · mean",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00032","--weight_decay","0.00010","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.04","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🔧C1 | FE×8 · TSL=6 · Dropout 0.04 · lr 3.2e-4 · wd 8e-5 · dils 1,2 · mean",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00032","--weight_decay","0.00008","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.04","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🔧C2 | FE×8 · TSL=6 · Dropout 0.04 · lr 3.8e-4 · wd 8e-5 · dils 1,2 · mean",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00038","--weight_decay","0.00008","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.04","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🔧D1 | FE×8 · TSL=6 · Dropout 0.04 · wd 8e-5 · dils 1,3 · mean",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,3","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00032","--weight_decay","0.00008","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.04","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🔧E1 | FE×8 · TSL=6 · Dropout 0.04 · wd 8e-5 · dils 1,2 · fuse=concat",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","concat","--batch_size","576","--epochs","180","--learning_rate","0.00032","--weight_decay","0.00008","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.04","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # }
        # {
        # "name": "🔧 A1 | FE×8 · TSL=6 · Dropout 0.04",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00033","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.04","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🔧 A2 | FE×8 · TSL=6 · Dropout 0.06",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00033","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.06","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "⚙️ A3 | FE×8 · TSL=6 · LR 3e-4",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00030","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "⚙️ A4 | FE×8 · TSL=6 · LR 3.8e-4",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00038","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "📦 A5 | Batch 768 · FE×8 · TSL=6",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","768","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "📦 A6 | Batch 768 · FE×8 · LR 3.2e-4",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","768","--epochs","180","--learning_rate","0.00032","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🔎 A7 | ECA k=3 探索 · 其余同最优邻域",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","3","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🧱 A8 | FE×10 · 更宽特征混合",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","10","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },

        # {
        # "name": "🌀 B1 | TCN dilations {1,3}",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,3","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🌀 B2 | TCN dilations {1,2,3}",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2,3","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🧩 B3 | TCN kernel=5 · dil {1,2}",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","5","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🧩 B4 | TCN kernel=5 · LR 3.2e-4",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","5","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00032","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🧪 B5 | TCN dilations {1,2,4} · LR 3.2e-4",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2,4","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00032","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🧪 B6 | 更小 WD=1.2e-4→8e-5 · 保 {1,2}",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00008","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },

        # {
        # "name": "🏗️ C1 | 层数↑ 8 · Dropout 0.04 · LR 3.2e-4",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00032","--weight_decay","0.00012","--tsmixer_layers","8","--time_expansion","4","--feat_expansion","8","--dropout","0.04","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🏗️ C2 | 层数↓ 4 · 轻量快速对照",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","4","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🫧 C3 | 更小 Dropout 0.03",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.03","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🫧 C4 | 更大 Dropout 0.07 · WD 1.4e-4 · LR 3.3e-4",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00033","--weight_decay","0.00014","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.07","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },

        # {
        # "name": "🧪 D1 | Ablation 去 ECA（仅 BiTCN+TSMixer）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🧪 D2 | Ablation 去 BiTCN（仅 ECA+TSMixer）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # }   
        # {
        # "name": "🏆 A1 | FE×8 + TSL=6（基线最优线）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.10","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🥇 A2 | FE×8 + TSL=6 + 低MixerDropout(0.05)",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "💪 A3 | FE×8 + TSL=6（加大batch=704）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","704","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.10","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🪄 A4 | FE×8 + TSL=6 + ECA=7（细粒度通道注意）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","7","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "⚡ B1 | 简洁线：TSL=4, FE×4, do=0.05, bs=768（强优化范式）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","768","--epochs","160","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","4","--time_expansion","4","--feat_expansion","4","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "⚖️ B2 | 上条的wd扫描：wd=1e-4（常见最稳点之一）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","768","--epochs","160","--learning_rate","0.00035","--weight_decay","0.0001","--tsmixer_layers","4","--time_expansion","4","--feat_expansion","4","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🧪 B3 | 上条的wd扫描：wd=1.5e-4（轻微加正则）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","768","--epochs","160","--learning_rate","0.00035","--weight_decay","0.00015","--tsmixer_layers","4","--time_expansion","4","--feat_expansion","4","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🆙 B4 | 在B1上仅改TSL=6（看结构+优化能否叠加）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","768","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","4","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🧭 C1 | 控制实验：FE×8 + do=0.05，但 dils=1,2,4",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2,4","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.05","--tcn_dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },

        # {
        # "name": "🥇 ECATSM-FD002-LongRun++: L4 TE4 FE4 dils=1,2 mean (200e)",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","768","--epochs","200","--learning_rate","0.0004","--weight_decay","0.0001","--tsmixer_layers","4","--time_expansion","4","--feat_expansion","4","--dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","36"]
        # },
        # {
        # "name": "🥈 ECATSM-FD002-DilWide-124: L4 TE4 FE4 dils=1,2,4 mean",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2,4","--tcn_fuse","mean","--batch_size","704","--epochs","180","--learning_rate","0.0004","--weight_decay","0.0001","--tsmixer_layers","4","--time_expansion","4","--feat_expansion","4","--dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🥉 ECATSM-FD002-DilWide-1248: 更大感受野",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2,4,8","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00035","--weight_decay","0.0001","--tsmixer_layers","4","--time_expansion","4","--feat_expansion","4","--dropout","0.10","--scheduler","cosine","--warmup_epochs","10","--early_stopping","30"]
        # },
        # {
        # "name": "🏃 ECATSM-FD002-DeepMixer-L6: L6 TE4 FE4 dils=1,2 mean",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","640","--epochs","180","--learning_rate","0.00035","--weight_decay","0.0001","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","4","--dropout","0.10","--scheduler","cosine","--warmup_epochs","10","--early_stopping","32"]
        # },
        # {
        # "name": "🧠 ECATSM-FD002-DeepWide-L6FE8: L6 TE4 FE8 dils=1,2 mean",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","576","--epochs","160","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","6","--time_expansion","4","--feat_expansion","8","--dropout","0.10","--scheduler","cosine","--warmup_epochs","10","--early_stopping","28"]
        # },
        # {
        # "name": "🌊 ECATSM-FD002-LowDrop: dropout=0.05（更少正则）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","768","--epochs","180","--learning_rate","0.00035","--weight_decay","0.0001","--tsmixer_layers","4","--time_expansion","4","--feat_expansion","4","--dropout","0.05","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🧊 ECATSM-FD002-HighWD: wd=3e-4（略增权重衰减）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","768","--epochs","160","--learning_rate","0.00035","--weight_decay","0.0003","--tsmixer_layers","4","--time_expansion","4","--feat_expansion","4","--dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","28"]
        # },
        # {
        # "name": "📡 ECATSM-FD002-ECABigger: eca_kernel=7（更平滑通道注意）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","7","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2","--tcn_fuse","mean","--batch_size","768","--epochs","160","--learning_rate","0.0004","--weight_decay","0.0001","--tsmixer_layers","4","--time_expansion","4","--feat_expansion","4","--dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","28"]
        # },
        # {
        # "name": "➕ ECATSM-FD002-FuseSum: BiTCN 融合=sum",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2,4","--tcn_fuse","sum","--batch_size","704","--epochs","180","--learning_rate","0.00035","--weight_decay","0.0001","--tsmixer_layers","4","--time_expansion","4","--feat_expansion","4","--dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🧩 ECATSM-FD002-TCNk5: TCN kernel=5（更宽卷积核）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","5","--tcn_dilations","1,2,4","--tcn_fuse","mean","--batch_size","640","--epochs","160","--learning_rate","0.00035","--weight_decay","0.0001","--tsmixer_layers","4","--time_expansion","4","--feat_expansion","4","--dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","28"]
        # },
        # {
        # "name": "🪶 ECATSM-FD002-TCNdrop06: tcn_dropout=0.06（轻度正则）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2,4","--tcn_fuse","mean","--batch_size","704","--epochs","180","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","4","--time_expansion","4","--feat_expansion","4","--dropout","0.10","--tcn_dropout","0.06","--scheduler","cosine","--warmup_epochs","8","--early_stopping","32"]
        # },
        # {
        # "name": "🧱 ECATSM-FD002-TE6: 时间混合更强（TE=6）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2,4","--tcn_fuse","mean","--batch_size","704","--epochs","160","--learning_rate","0.00035","--weight_decay","0.0001","--tsmixer_layers","4","--time_expansion","6","--feat_expansion","4","--dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","28"]
        # },
        # {
        # "name": "🗂 ECATSM-FD002-FE8: 特征混合更强（FE=8）",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2,4","--tcn_fuse","mean","--batch_size","640","--epochs","160","--learning_rate","0.00035","--weight_decay","0.00012","--tsmixer_layers","4","--time_expansion","4","--feat_expansion","8","--dropout","0.10","--scheduler","cosine","--warmup_epochs","8","--early_stopping","28"]
        # },
        # {
        # "name": "⚙️ ECATSM-FD002-GradAccum2: 显存友好长跑",
        # "cmd": ["python","train.py","--model","tsmixer_eca","--fault","FD002","--use_eca","--eca_kernel","5","--use_bitcn","--tcn_kernel","3","--tcn_dilations","1,2,4","--tcn_fuse","mean","--batch_size","384","--grad_accum","2","--epochs","200","--learning_rate","0.00035","--weight_decay","0.0001","--tsmixer_layers","4","--time_expansion","4","--feat_expansion","4","--dropout","0.10","--scheduler","cosine","--warmup_epochs","10","--early_stopping","36"]
        # }








        # {
        #     "name": "💎 pTSM-FD002-Micro-A1: dp=0.25 br=0.10 wd=3e-4 lr=5e-4",
        #     "cmd": ["python", "train.py",
        #     "--model", "ptsmixer", "--fault", "FD002",
        #     "--batch_size", "192", "--epochs", "80",
        #     "--learning_rate", "0.0005", "--weight_decay", "0.0003",
        #     "--pt_depth", "8", "--pt_ch_expand", "4",
        #     "--pt_t_kernel", "9", "--pt_t_dilation", "2",
        #     "--pt_t_ffn_expand", "1",
        #     "--pt_droppath", "0.25", "--pt_branch_dropout", "0.10",
        #     "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
        #     "--scheduler", "cosine", "--early_stopping", "18"
        #     ]
        # },
        # {
        #     "name": "💎 pTSM-FD002-Micro-A2: dp=0.25 br=0.10 wd=5e-4 lr=5e-4",
        #     "cmd": ["python", "train.py",
        #     "--model", "ptsmixer", "--fault", "FD002",
        #     "--batch_size", "192", "--epochs", "80",
        #     "--learning_rate", "0.0005", "--weight_decay", "0.0005",
        #     "--pt_depth", "8", "--pt_ch_expand", "4",
        #     "--pt_t_kernel", "9", "--pt_t_dilation", "2",
        #     "--pt_t_ffn_expand", "1",
        #     "--pt_droppath", "0.25", "--pt_branch_dropout", "0.10",
        #     "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
        #     "--scheduler", "cosine", "--early_stopping", "18"
        #     ]
        # },
        # {
        #     "name": "💎 pTSM-FD002-Micro-A3: dp=0.25 br=0.10 wd=3e-4 lr=6e-4",
        #     "cmd": ["python", "train.py",
        #     "--model", "ptsmixer", "--fault", "FD002",
        #     "--batch_size", "192", "--epochs", "80",
        #     "--learning_rate", "0.0006", "--weight_decay", "0.0003",
        #     "--pt_depth", "8", "--pt_ch_expand", "4",
        #     "--pt_t_kernel", "9", "--pt_t_dilation", "2",
        #     "--pt_t_ffn_expand", "1",
        #     "--pt_droppath", "0.25", "--pt_branch_dropout", "0.10",
        #     "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
        #     "--scheduler", "cosine", "--early_stopping", "18"
        #     ]
        # },
        # {
        #     "name": "💎 pTSM-FD002-Micro-A4: dp=0.25 br=0.10 wd=5e-4 lr=6e-4",
        #     "cmd": ["python", "train.py",
        #     "--model", "ptsmixer", "--fault", "FD002",
        #     "--batch_size", "192", "--epochs", "80",
        #     "--learning_rate", "0.0006", "--weight_decay", "0.0005",
        #     "--pt_depth", "8", "--pt_ch_expand", "4",
        #     "--pt_t_kernel", "9", "--pt_t_dilation", "2",
        #     "--pt_t_ffn_expand", "1",
        #     "--pt_droppath", "0.25", "--pt_branch_dropout", "0.10",
        #     "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
        #     "--scheduler", "cosine", "--early_stopping", "18"
        #     ]
        # },
        # {
        #     "name": "💎 pTSM-FD002-Micro-A5: dp=0.25 br=0.10 wd=3e-4 lr=7e-4",
        #     "cmd": ["python", "train.py",
        #     "--model", "ptsmixer", "--fault", "FD002",
        #     "--batch_size", "192", "--epochs", "80",
        #     "--learning_rate", "0.0007", "--weight_decay", "0.0003",
        #     "--pt_depth", "8", "--pt_ch_expand", "4",
        #     "--pt_t_kernel", "9", "--pt_t_dilation", "2",
        #     "--pt_t_ffn_expand", "1",
        #     "--pt_droppath", "0.25", "--pt_branch_dropout", "0.10",
        #     "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
        #     "--scheduler", "cosine", "--early_stopping", "18"
        #     ]
        # },
        # {
        #     "name": "💎 pTSM-FD002-Micro-A6: dp=0.24 br=0.10 wd=5e-4 lr=7e-4",
        #     "cmd": ["python", "train.py",
        #     "--model", "ptsmixer", "--fault", "FD002",
        #     "--batch_size", "192", "--epochs", "80",
        #     "--learning_rate", "0.0007", "--weight_decay", "0.0005",
        #     "--pt_depth", "8", "--pt_ch_expand", "4",
        #     "--pt_t_kernel", "9", "--pt_t_dilation", "2",
        #     "--pt_t_ffn_expand", "1",
        #     "--pt_droppath", "0.24", "--pt_branch_dropout", "0.10",
        #     "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
        #     "--scheduler", "cosine", "--early_stopping", "18"
        #     ]
        # },
        # {
        #     "name": "💎 pTSM-FD002-Micro-B1: dp=0.28 br=0.12 wd=2e-4 lr=6e-4",
        #     "cmd": ["python", "train.py",
        #     "--model", "ptsmixer", "--fault", "FD002",
        #     "--batch_size", "192", "--epochs", "80",
        #     "--learning_rate", "0.0006", "--weight_decay", "0.0002",
        #     "--pt_depth", "8", "--pt_ch_expand", "4",
        #     "--pt_t_kernel", "9", "--pt_t_dilation", "2",
        #     "--pt_t_ffn_expand", "1",
        #     "--pt_droppath", "0.28", "--pt_branch_dropout", "0.12",
        #     "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
        #     "--scheduler", "cosine", "--early_stopping", "18"
        #     ]
        # },
        # {
        #     "name": "💎 pTSM-FD002-Micro-B2: dp=0.30 br=0.12 wd=1e-4 lr=6e-4",
        #     "cmd": ["python", "train.py",
        #     "--model", "ptsmixer", "--fault", "FD002",
        #     "--batch_size", "192", "--epochs", "80",
        #     "--learning_rate", "0.0006", "--weight_decay", "0.0001",
        #     "--pt_depth", "8", "--pt_ch_expand", "4",
        #     "--pt_t_kernel", "9", "--pt_t_dilation", "2",
        #     "--pt_t_ffn_expand", "1",
        #     "--pt_droppath", "0.30", "--pt_branch_dropout", "0.12",
        #     "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
        #     "--scheduler", "cosine", "--early_stopping", "18"
        #     ]
        # },
        # {
        #     "name": "💎 pTSM-FD002-Micro-B3: dp=0.30 br=0.12 wd=2e-4 lr=5e-4",
        #     "cmd": ["python", "train.py",
        #     "--model", "ptsmixer", "--fault", "FD002",
        #     "--batch_size", "192", "--epochs", "80",
        #     "--learning_rate", "0.0005", "--weight_decay", "0.0002",
        #     "--pt_depth", "8", "--pt_ch_expand", "4",
        #     "--pt_t_kernel", "9", "--pt_t_dilation", "2",
        #     "--pt_t_ffn_expand", "1",
        #     "--pt_droppath", "0.30", "--pt_branch_dropout", "0.12",
        #     "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
        #     "--scheduler", "cosine", "--early_stopping", "18"
        #     ]
        # },
        # {
        #     "name": "💎 pTSM-FD002-Micro-B4: dp=0.30 br=0.12 wd=2e-4 lr=7e-4",
        #     "cmd": ["python", "train.py",
        #     "--model", "ptsmixer", "--fault", "FD002",
        #     "--batch_size", "192", "--epochs", "80",
        #     "--learning_rate", "0.0007", "--weight_decay", "0.0002",
        #     "--pt_depth", "8", "--pt_ch_expand", "4",
        #     "--pt_t_kernel", "9", "--pt_t_dilation", "2",
        #     "--pt_t_ffn_expand", "1",
        #     "--pt_droppath", "0.30", "--pt_branch_dropout", "0.12",
        #     "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
        #     "--scheduler", "cosine", "--early_stopping", "18"
        #     ]
        # },
        # {
        #     "name": "💎 pTSM-FD002-Micro-B5: dp=0.32 br=0.12 wd=2e-4 lr=6e-4",
        #     "cmd": ["python", "train.py",
        #     "--model", "ptsmixer", "--fault", "FD002",
        #     "--batch_size", "192", "--epochs", "80",
        #     "--learning_rate", "0.0006", "--weight_decay", "0.0002",
        #     "--pt_depth", "8", "--pt_ch_expand", "4",
        #     "--pt_t_kernel", "9", "--pt_t_dilation", "2",
        #     "--pt_t_ffn_expand", "1",
        #     "--pt_droppath", "0.32", "--pt_branch_dropout", "0.12",
        #     "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
        #     "--scheduler", "cosine", "--early_stopping", "18"
        #     ]
        # },
        # {
        #     "name": "💎 pTSM-FD002-Micro-B6: dp=0.28 br=0.10 wd=2e-4 lr=6e-4",
        #     "cmd": ["python", "train.py",
        #     "--model", "ptsmixer", "--fault", "FD002",
        #     "--batch_size", "192", "--epochs", "80",
        #     "--learning_rate", "0.0006", "--weight_decay", "0.0002",
        #     "--pt_depth", "8", "--pt_ch_expand", "4",
        #     "--pt_t_kernel", "9", "--pt_t_dilation", "2",
        #     "--pt_t_ffn_expand", "1",
        #     "--pt_droppath", "0.28", "--pt_branch_dropout", "0.10",
        #     "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
        #     "--scheduler", "cosine", "--early_stopping", "18"
        #     ]
        # },
        

    # {
    # "name": "💎 pTSM-FD002-Narrow-1: best+wd↑",
    # "cmd": ["python", "train.py",
    #     "--model", "ptsmixer", "--fault", "FD002",
    #     "--batch_size", "192", "--epochs", "80",
    #     "--learning_rate", "0.0006", "--weight_decay", "0.0006",
    #     "--pt_depth", "8", "--pt_ch_expand", "4",
    #     "--pt_t_kernel", "9", "--pt_t_dilation", "2",
    #     "--pt_t_ffn_expand", "1",
    #     "--pt_droppath", "0.25", "--pt_branch_dropout", "0.12",
    #     "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
    #     "--scheduler", "cosine", "--early_stopping", "15"
    # ]
    # },
    # {
    #     "name": "💎 pTSM-FD002-Narrow-2: dp↑0.28",
    #     "cmd": ["python", "train.py",
    #         "--model", "ptsmixer", "--fault", "FD002",
    #         "--batch_size", "192", "--epochs", "80",
    #         "--learning_rate", "0.0006", "--weight_decay", "0.0005",
    #         "--pt_depth", "8", "--pt_ch_expand", "4",
    #         "--pt_t_kernel", "9", "--pt_t_dilation", "2",
    #         "--pt_t_ffn_expand", "1",
    #         "--pt_droppath", "0.28", "--pt_branch_dropout", "0.12",
    #         "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
    #         "--scheduler", "cosine", "--early_stopping", "15"
    #     ]
    # },
    # {
    #     "name": "💎 pTSM-FD002-Narrow-3: lr↑7e-4",
    #     "cmd": ["python", "train.py",
    #         "--model", "ptsmixer", "--fault", "FD002",
    #         "--batch_size", "192", "--epochs", "80",
    #         "--learning_rate", "0.0007", "--weight_decay", "0.0005",
    #         "--pt_depth", "8", "--pt_ch_expand", "4",
    #         "--pt_t_kernel", "9", "--pt_t_dilation", "2",
    #         "--pt_t_ffn_expand", "1",
    #         "--pt_droppath", "0.25", "--pt_branch_dropout", "0.10",
    #         "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
    #         "--scheduler", "cosine", "--early_stopping", "15"
    #     ]
    # },
    # {
    #     "name": "💎 pTSM-FD002-Narrow-4: ker=11 对照",
    #     "cmd": ["python", "train.py",
    #         "--model", "ptsmixer", "--fault", "FD002",
    #         "--batch_size", "192", "--epochs", "80",
    #         "--learning_rate", "0.0006", "--weight_decay", "0.0005",
    #         "--pt_depth", "8", "--pt_ch_expand", "4",
    #         "--pt_t_kernel", "11", "--pt_t_dilation", "2",
    #         "--pt_t_ffn_expand", "1",
    #         "--pt_droppath", "0.25", "--pt_branch_dropout", "0.12",
    #         "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
    #         "--scheduler", "cosine", "--early_stopping", "15"
    #     ]
    # },
    # {
    #     "name": "💎 pTSM-FD002-Narrow-5: wd=2e-4 + dp=0.30",
    #     "cmd": ["python", "train.py",
    #         "--model", "ptsmixer", "--fault", "FD002",
    #         "--batch_size", "192", "--epochs", "80",
    #         "--learning_rate", "0.0006", "--weight_decay", "0.0002",
    #         "--pt_depth", "8", "--pt_ch_expand", "4",
    #         "--pt_t_kernel", "9", "--pt_t_dilation", "2",
    #         "--pt_t_ffn_expand", "1",
    #         "--pt_droppath", "0.30", "--pt_branch_dropout", "0.12",
    #         "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
    #         "--scheduler", "cosine", "--early_stopping", "18"
    #     ]
    # },
    # {
    #     "name": "💎 pTSM-FD002-Narrow-6: depth=7 稳健性",
    #     "cmd": ["python", "train.py",
    #         "--model", "ptsmixer", "--fault", "FD002",
    #         "--batch_size", "192", "--epochs", "80",
    #         "--learning_rate", "0.0006", "--weight_decay", "0.0005",
    #         "--pt_depth", "7", "--pt_ch_expand", "4",
    #         "--pt_t_kernel", "9", "--pt_t_dilation", "2",
    #         "--pt_t_ffn_expand", "1",
    #         "--pt_droppath", "0.24", "--pt_branch_dropout", "0.12",
    #         "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
    #         "--scheduler", "cosine", "--early_stopping", "15"
    #     ]
    # },
    # {
    #     "name": "💎 pTSM-FD002-Narrow-7: branch-drop=0.15",
    #     "cmd": ["python", "train.py",
    #         "--model", "ptsmixer", "--fault", "FD002",
    #         "--batch_size", "192", "--epochs", "80",
    #         "--learning_rate", "0.0006", "--weight_decay", "0.0005",
    #         "--pt_depth", "8", "--pt_ch_expand", "4",
    #         "--pt_t_kernel", "9", "--pt_t_dilation", "2",
    #         "--pt_t_ffn_expand", "1",
    #         "--pt_droppath", "0.26", "--pt_branch_dropout", "0.15",
    #         "--pt_pooling", "avg", "--pt_input_dropout", "0.00",
    #         "--scheduler", "cosine", "--early_stopping", "15"
    #     ]
    # }
    #          # 1) 基线（你当前最优 15.48）
    # {
    #     "name": "pTSM-FD002-BASE-avg_dp0.20",
    #     "cmd": ["python", "train.py",
    #             "--model", "ptsmixer",
    #             "--fault", "FD002",
    #             "--batch_size", "192",
    #             "--epochs", "60",
    #             "--learning_rate", "8e-4",
    #             "--weight_decay", "2e-4",
    #             "--pt_depth", "6",
    #             "--pt_ch_expand", "4",
    #             "--pt_t_kernel", "11",
    #             "--pt_t_dilation", "2",
    #             "--pt_t_ffn_expand", "1",
    #             "--pt_droppath", "0.20",
    #             "--pt_branch_dropout", "0.10",
    #             "--pt_pooling", "avg",
    #             "--pt_input_dropout", "0.00",
    #             "--scheduler", "cosine",
    #             "--early_stopping", "12"]
    # },
    #  # 2) 仅增强权重衰减（压末端过拟合）
    # {
    #     "name": "pTSM-FD002-wd5e-4_lr8e-4",
    #     "cmd": ["python", "train.py",
    #             "--model", "ptsmixer",
    #             "--fault", "FD002",
    #             "--batch_size", "192",
    #             "--epochs", "60",
    #             "--learning_rate", "8e-4",
    #             "--weight_decay", "5e-4",
    #             "--pt_depth", "6",
    #             "--pt_ch_expand", "4",
    #             "--pt_t_kernel", "11",
    #             "--pt_t_dilation", "2",
    #             "--pt_t_ffn_expand", "1",
    #             "--pt_droppath", "0.20",
    #             "--pt_branch_dropout", "0.10",
    #             "--pt_pooling", "avg",
    #             "--pt_input_dropout", "0.00",
    #             "--scheduler", "cosine",
    #             "--early_stopping", "12"]
    # },
    # # 3) wd↑ + lr↓ + plateau（更稳的尾期微调）
    # {
    #     "name": "pTSM-FD002-wd5e-4_lr6e-4_plateau",
    #     "cmd": ["python", "train.py",
    #             "--model", "ptsmixer",
    #             "--fault", "FD002",
    #             "--batch_size", "192",
    #             "--epochs", "60",
    #             "--learning_rate", "6e-4",
    #             "--weight_decay", "5e-4",
    #             "--pt_depth", "6",
    #             "--pt_ch_expand", "4",
    #             "--pt_t_kernel", "11",
    #             "--pt_t_dilation", "2",
    #             "--pt_t_ffn_expand", "1",
    #             "--pt_droppath", "0.20",
    #             "--pt_branch_dropout", "0.10",
    #             "--pt_pooling", "avg",
    #             "--pt_input_dropout", "0.00",
    #             "--scheduler", "plateau",
    #             "--early_stopping", "12"]
    # },

    # # 4) 感受野对照：kernel 9 / dilation 2（更平滑）
    # {
    #     "name": "pTSM-FD002-ker9_dil2",
    #     "cmd": ["python", "train.py",
    #             "--model", "ptsmixer",
    #             "--fault", "FD002",
    #             "--batch_size", "192",
    #             "--epochs", "60",
    #             "--learning_rate", "8e-4",
    #             "--weight_decay", "2e-4",
    #             "--pt_depth", "6",
    #             "--pt_ch_expand", "4",
    #             "--pt_t_kernel", "9",
    #             "--pt_t_dilation", "2",
    #             "--pt_t_ffn_expand", "1",
    #             "--pt_droppath", "0.20",
    #             "--pt_branch_dropout", "0.10",
    #             "--pt_pooling", "avg",
    #             "--pt_input_dropout", "0.00",
    #             "--scheduler", "cosine",
    #             "--early_stopping", "12"]
    # },

    # # 5) 感受野对照：kernel 7 / dilation 3（更大有效视野、低参数）
    # {
    #     "name": "pTSM-FD002-ker7_dil3",
    #     "cmd": ["python", "train.py",
    #             "--model", "ptsmixer",
    #             "--fault", "FD002",
    #             "--batch_size", "192",
    #             "--epochs", "60",
    #             "--learning_rate", "8e-4",
    #             "--weight_decay", "2e-4",
    #             "--pt_depth", "6",
    #             "--pt_ch_expand", "4",
    #             "--pt_t_kernel", "7",
    #             "--pt_t_dilation", "3",
    #             "--pt_t_ffn_expand", "1",
    #             "--pt_droppath", "0.20",
    #             "--pt_branch_dropout", "0.10",
    #             "--pt_pooling", "avg",
    #             "--pt_input_dropout", "0.00",
    #             "--scheduler", "cosine",
    #             "--early_stopping", "12"]
    # },

    # # 6) 深度↑ + DropPath↑（表征更强 + 正则配套）
    # {
    #     "name": "pTSM-FD002-depth8_dp0.25",
    #     "cmd": ["python", "train.py",
    #             "--model", "ptsmixer",
    #             "--fault", "FD002",
    #             "--batch_size", "192",
    #             "--epochs", "80",
    #             "--learning_rate", "6e-4",
    #             "--weight_decay", "5e-4",
    #             "--pt_depth", "8",
    #             "--pt_ch_expand", "4",
    #             "--pt_t_kernel", "9",
    #             "--pt_t_dilation", "2",
    #             "--pt_t_ffn_expand", "1",
    #             "--pt_droppath", "0.25",
    #             "--pt_branch_dropout", "0.10",
    #             "--pt_pooling", "avg",
    #             "--pt_input_dropout", "0.00",
    #             "--scheduler", "cosine",
    #             "--early_stopping", "15"]
    # },

    # # 7) 通道扩展↑（ch_expand=6），保留正则
    # {
    #     "name": "pTSM-FD002-chx6_dp0.20",
    #     "cmd": ["python", "train.py",
    #             "--model", "ptsmixer",
    #             "--fault", "FD002",
    #             "--batch_size", "192",
    #             "--epochs", "60",
    #             "--learning_rate", "8e-4",
    #             "--weight_decay", "2e-4",
    #             "--pt_depth", "6",
    #             "--pt_ch_expand", "6",
    #             "--pt_t_kernel", "11",
    #             "--pt_t_dilation", "2",
    #             "--pt_t_ffn_expand", "1",
    #             "--pt_droppath", "0.20",
    #             "--pt_branch_dropout", "0.10",
    #             "--pt_pooling", "avg",
    #             "--pt_input_dropout", "0.00",
    #             "--scheduler", "cosine",
    #             "--early_stopping", "12"]
    # },

    # # 8) “稳健拉低”组合（wd↑ + lr↓ + ker9 + plateau）
    # {
    #     "name": "pTSM-FD002-steady-wd5e-4_lr6e-4_ker9_plateau",
    #     "cmd": ["python", "train.py",
    #             "--model", "ptsmixer",
    #             "--fault", "FD002",
    #             "--batch_size", "192",
    #             "--epochs", "60",
    #             "--learning_rate", "6e-4",
    #             "--weight_decay", "5e-4",
    #             "--pt_depth", "6",
    #             "--pt_ch_expand", "4",
    #             "--pt_t_kernel", "9",
    #             "--pt_t_dilation", "2",
    #             "--pt_t_ffn_expand", "1",
    #             "--pt_droppath", "0.20",
    #             "--pt_branch_dropout", "0.10",
    #             "--pt_pooling", "avg",
    #             "--pt_input_dropout", "0.00",
    #             "--scheduler", "plateau",
    #             "--early_stopping", "12"]
    # },
        
        # ============================================================================
        # TSMixer-SGA 新模型实验 - 带双维可扩展全局注意力的TSMixer
        # ============================================================================
        
        # {
        #     "name": "🟩 FD002-A: 提高token粒度（patch=4）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tokenpool",
        #            "--fault", "FD002",
        #            "--batch_size", "192",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0002",
        #            "--patch", "4",
        #            "--d_model", "160",
        #            "--depth", "5",
        #            "--token_mlp_dim", "320",
        #            "--channel_mlp_dim", "160",
        #            "--dropout", "0.12",
        #            "--cnn_pool", "weighted",
        #            "--tokenpool_heads", "8",
        #            "--tokenpool_dropout", "0.12",
        #            "--tokenpool_temperature", "1.8",
        #            "--scheduler", "cosine",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "🟦 FD002-B: 多头分工（heads=10, 温度2.0）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tokenpool",
        #            "--fault", "FD002",
        #            "--batch_size", "192",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0002",
        #            "--patch", "5",
        #            "--d_model", "200",
        #            "--depth", "5",
        #            "--token_mlp_dim", "400",
        #            "--channel_mlp_dim", "200",
        #            "--dropout", "0.12",
        #            "--cnn_pool", "weighted",
        #            "--tokenpool_heads", "10",
        #            "--tokenpool_dropout", "0.12",
        #            "--tokenpool_temperature", "2.0",
        #            "--scheduler", "cosine",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "🟨 FD002-C: 更细粒度上限（patch=3）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tokenpool",
        #            "--fault", "FD002",
        #            "--batch_size", "160",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0002",
        #            "--patch", "3",
        #            "--d_model", "160",
        #            "--depth", "5",
        #            "--token_mlp_dim", "384",
        #            "--channel_mlp_dim", "160",
        #            "--dropout", "0.14",
        #            "--cnn_pool", "weighted",
        #            "--tokenpool_heads", "8",
        #            "--tokenpool_dropout", "0.14",
        #            "--tokenpool_temperature", "1.9",
        #            "--scheduler", "cosine",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "🟪 FD002-D: 大容量稳态（d_model↑, depth↑）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tokenpool",
        #            "--fault", "FD002",
        #            "--batch_size", "160",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0007",
        #            "--weight_decay", "0.00025",
        #            "--patch", "5",
        #            "--d_model", "192",
        #            "--depth", "6",
        #            "--token_mlp_dim", "384",
        #            "--channel_mlp_dim", "192",
        #            "--dropout", "0.13",
        #            "--cnn_pool", "weighted",
        #            "--tokenpool_heads", "8",
        #            "--tokenpool_dropout", "0.12",
        #            "--tokenpool_temperature", "1.7",
        #            "--scheduler", "cosine",
        #            "--early_stopping", "14"
        #            ]
        # },
        
        # {
        #     "name": "🟥 FD002-E: 末端更关注（pool=last 对照）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tokenpool",
        #            "--fault", "FD002",
        #            "--batch_size", "192",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0002",
        #            "--patch", "4",
        #            "--d_model", "160",
        #            "--depth", "5",
        #            "--token_mlp_dim", "320",
        #            "--channel_mlp_dim", "160",
        #            "--dropout", "0.12",
        #            "--cnn_pool", "last",
        #            "--tokenpool_heads", "8",
        #            "--tokenpool_dropout", "0.12",
        #            "--tokenpool_temperature", "1.8",
        #            "--scheduler", "plateau",
        #            "--early_stopping", "12"
        #            ]
        # },
    
        
        # ============================================================================
        # MTS-TSMixer 多尺度时间混合实验 - 新模型测试
        # ============================================================================
        
        # {
        #     "name": "🔥 MTS-1: 基础4尺度配置（3-1,3-2,5-3,7-4）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_mts",
        #            "--fault", "FD002",
        #            "--batch_size", "2304",         # 优化：增大batch size
        #            "--num_workers", "32",          # 优化：增加workers
        #            "--prefetch_factor", "8",       # 优化：增加预取
        #            "--epochs", "100",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0002",
        #            "--tsmixer_layers", "6",
        #            "--time_expansion", "4",        # 每个尺度的扩展
        #            "--feat_expansion", "6",
        #            "--dropout", "0.12",
        #            "--mts_scales", "3-1,3-2,5-3,7-4",  # 4个时间尺度
        #            "--mts_gate_hidden", "16",      # 门控隐藏层
        #            "--scheduler", "cosine",
        #            "--warmup_epochs", "8",
        #            "--early_stopping", "15"
        #            ]
        # },
        
        # {
        #     "name": "🌊 MTS-2: 密集小尺度（3-1,3-2,3-3,5-2）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_mts",
        #            "--fault", "FD002",
        #            "--batch_size", "2304",         # 优化：增大batch size
        #            "--num_workers", "32",          # 优化：增加workers
        #            "--prefetch_factor", "8",       # 优化：增加预取
        #            "--epochs", "100",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0002",
        #            "--tsmixer_layers", "6",
        #            "--time_expansion", "5",
        #            "--feat_expansion", "6",
        #            "--dropout", "0.12",
        #            "--mts_scales", "3-1,3-2,3-3,5-2",  # 密集小卷积核
        #            "--mts_gate_hidden", "20",
        #            "--scheduler", "cosine",
        #            "--warmup_epochs", "8",
        #            "--early_stopping", "15"
        #            ]
        # },
        
       

        # ============================================================================
        # TSMixer-MTS-SGA 多尺度+双轴注意力实验 (FD002)
        # ============================================================================
        
        # {
        #     "name": "⚙️ MTS-SGA-1: 偏短期敏感（3-1,5-1,5-2,7-2）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_mts_sga",
        #            "--fault", "FD002",
        #            "--batch_size", "3072",
        #            "--num_workers", "32",
        #            "--prefetch_factor", "8",
        #            "--epochs", "80",
        #            "--learning_rate", "0.0011",
        #            "--weight_decay", "0.00018",
        #            "--tsmixer_layers", "5",
        #            "--time_expansion", "6",
        #            "--feat_expansion", "4",
        #            "--dropout", "0.10",
        #            "--mts_scales", "3-1,5-1,5-2,7-2",
        #            "--mts_gate_hidden", "16",
        #            "--mts_sga_time_hidden", "24",
        #            "--mts_sga_feat_hidden", "24",
        #            "--mts_sga_dropout", "0.08",
        #            "--scheduler", "plateau",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "⚡ MTS-SGA-2: 大感受野（5-2,7-3,9-4,11-5）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_mts_sga",
        #            "--fault", "FD002",
        #            "--batch_size", "2048",
        #            "--num_workers", "32",
        #            "--prefetch_factor", "8",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0007",
        #            "--weight_decay", "0.00025",
        #            "--tsmixer_layers", "6",
        #            "--time_expansion", "4",
        #            "--feat_expansion", "5",
        #            "--dropout", "0.14",
        #            "--mts_scales", "5-2,7-3,9-4,11-5",
        #            "--mts_gate_hidden", "24",
        #            "--mts_sga_time_hidden", "16",
        #            "--mts_sga_feat_hidden", "16",
        #            "--mts_sga_dropout", "0.06",
        #            "--scheduler", "cosine",
        #            "--warmup_epochs", "10",
        #            "--early_stopping", "15"
        #            ]
        # },
        
        # {
        #     "name": "🎯 MTS-SGA-3: 3尺度轻量版（3-1,5-2,7-3）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_mts_sga",
        #            "--fault", "FD002",
        #            "--batch_size", "3072",
        #            "--num_workers", "32",
        #            "--prefetch_factor", "8",
        #            "--epochs", "80",
        #            "--learning_rate", "0.0010",
        #            "--weight_decay", "0.00020",
        #            "--tsmixer_layers", "5",
        #            "--time_expansion", "5",
        #            "--feat_expansion", "4",
        #            "--dropout", "0.10",
        #            "--mts_scales", "3-1,5-2,7-3",
        #            "--mts_gate_hidden", "12",
        #            "--mts_sga_time_hidden", "16",
        #            "--mts_sga_feat_hidden", "16",
        #            "--mts_sga_dropout", "0.05",
        #            "--scheduler", "plateau",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "💪 MTS-SGA-4: 5尺度深层版（3-1,5-1,5-2,7-2,9-3）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_mts_sga",
        #            "--fault", "FD002",
        #            "--batch_size", "1792",
        #            "--num_workers", "32",
        #            "--prefetch_factor", "8",
        #            "--epochs", "120",
        #            "--learning_rate", "0.0006",
        #            "--weight_decay", "0.00030",
        #            "--tsmixer_layers", "8",
        #            "--time_expansion", "4",
        #            "--feat_expansion", "6",
        #            "--dropout", "0.15",
        #            "--mts_scales", "3-1,5-1,5-2,7-2,9-3",
        #            "--mts_gate_hidden", "32",
        #            "--mts_sga_time_hidden", "24",
        #            "--mts_sga_feat_hidden", "24",
        #            "--mts_sga_dropout", "0.08",
        #            "--scheduler", "cosine",
        #            "--warmup_epochs", "10",
        #            "--early_stopping", "18"
        #            ]
        # },

        # ============================================================================
        # TokenPool-TSMixer 纯注意力池化实验 (FD002)
        # ============================================================================
        
        {
            "name": "🟥 TokenPool-FD002: 末端池化+Plateau调度器（最优配置）",
            "cmd": ["python", "train.py",
                   "--model", "tokenpool",
                   "--fault", "FD002",
                   "--batch_size", "1920",
                   "--epochs", "100",
                   "--learning_rate", "0.0008",
                   "--weight_decay", "0.0002",
                   "--patch", "4",
                   "--d_model", "160",
                   "--depth", "5",
                   "--token_mlp_dim", "320",
                   "--channel_mlp_dim", "160",
                   "--dropout", "0.12",
                   "--cnn_pool", "last",
                   "--tokenpool_heads", "8",
                   "--tokenpool_dropout", "0.12",
                   "--tokenpool_temperature", "1.8",
                   "--num_workers", "32",
                   "--prefetch_factor", "8",
                   "--scheduler", "plateau",
                   "--early_stopping", "12"
                   ]
        },

        # ============================================================================
        # TokenPool-TSMixer-SGA 注意力池化+双轴注意力实验 (FD002)
        # ============================================================================
        
        # {
        #     "name": "🎨 TokenPool-SGA-1: 基础配置（patch=6, 5 tokens）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tokenpool_sga",
        #            "--fault", "FD002",
        #            "--batch_size", "1024",
        #            "--num_workers", "32",
        #            "--prefetch_factor", "8",
        #            "--epochs", "80",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0001",
        #            "--patch", "6",                  # 30/6=5 tokens
        #            "--d_model", "120",              # 120 % 6 = 0 ✓
        #            "--depth", "5",
        #            "--token_mlp_dim", "240",
        #            "--channel_mlp_dim", "120",
        #            "--dropout", "0.12",
        #            "--tokenpool_heads", "6",
        #            "--tokenpool_dropout", "0.10",
        #            "--tokenpool_temperature", "1.5",
        #            "--tokenpool_sga_time_hidden", "24",
        #            "--tokenpool_sga_feat_hidden", "24",
        #            "--tokenpool_sga_dropout", "0.06",
        #            "--tokenpool_sga_fuse", "add",
        #            "--tokenpool_sga_every_k", "0",
        #            "--scheduler", "cosine",
        #            "--warmup_epochs", "8",
        #            "--early_stopping", "15"
        #            ]
        # },
        
        # {
        #     "name": "🔮 TokenPool-SGA-2: 高温注意力（temp=1.8, 更平滑）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tokenpool_sga",
        #            "--fault", "FD002",
        #            "--batch_size", "1024",
        #            "--num_workers", "32",
        #            "--prefetch_factor", "8",
        #            "--epochs", "80",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0001",
        #            "--patch", "5",
        #            "--d_model", "144",
        #            "--depth", "5",
        #            "--token_mlp_dim", "288",
        #            "--channel_mlp_dim", "144",
        #            "--dropout", "0.12",
        #            "--tokenpool_heads", "6",
        #            "--tokenpool_dropout", "0.10",
        #            "--tokenpool_temperature", "1.8",
        #            "--tokenpool_sga_time_hidden", "24",
        #            "--tokenpool_sga_feat_hidden", "24",
        #            "--tokenpool_sga_dropout", "0.06",
        #            "--tokenpool_sga_fuse", "add",
        #            "--tokenpool_sga_every_k", "0",
        #            "--scheduler", "cosine",
        #            "--warmup_epochs", "8",
        #            "--early_stopping", "15"
        #            ]
        # },
        
        # {
        #     "name": "⚡ TokenPool-SGA-3: 分层插入SGA（every_k=2）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tokenpool_sga",
        #            "--fault", "FD002",
        #            "--batch_size", "896",
        #            "--num_workers", "32",
        #            "--prefetch_factor", "8",
        #            "--epochs", "90",
        #            "--learning_rate", "0.0007",
        #            "--weight_decay", "0.00012",
        #            "--patch", "6",
        #            "--d_model", "120",              # 120 % 6 = 0 ✓
        #            "--depth", "6",
        #            "--token_mlp_dim", "240",
        #            "--channel_mlp_dim", "120",
        #            "--dropout", "0.12",
        #            "--tokenpool_heads", "6",
        #            "--tokenpool_dropout", "0.10",
        #            "--tokenpool_temperature", "1.6",
        #            "--tokenpool_sga_time_hidden", "32",
        #            "--tokenpool_sga_feat_hidden", "32",
        #            "--tokenpool_sga_dropout", "0.08",
        #            "--tokenpool_sga_fuse", "add",
        #            "--tokenpool_sga_every_k", "2",   # 每2层插入SGA
        #            "--scheduler", "cosine",
        #            "--warmup_epochs", "10",
        #            "--early_stopping", "15"
        #            ]
        # },
        
        # {
        #     "name": "💎 TokenPool-SGA-4: Hadamard融合（门控乘法）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tokenpool_sga",
        #            "--fault", "FD002",
        #            "--batch_size", "1024",
        #            "--num_workers", "32",
        #            "--prefetch_factor", "8",
        #            "--epochs", "80",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0001",
        #            "--patch", "6",
        #            "--d_model", "120",              # 120 % 6 = 0 ✓
        #            "--depth", "5",
        #            "--token_mlp_dim", "240",
        #            "--channel_mlp_dim", "120",
        #            "--dropout", "0.12",
        #            "--tokenpool_heads", "6",
        #            "--tokenpool_dropout", "0.10",
        #            "--tokenpool_temperature", "1.5",
        #            "--tokenpool_sga_time_hidden", "24",
        #            "--tokenpool_sga_feat_hidden", "24",
        #            "--tokenpool_sga_dropout", "0.06",
        #            "--tokenpool_sga_fuse", "hadamard",  # 使用Hadamard融合
        #            "--tokenpool_sga_every_k", "0",
        #            "--scheduler", "cosine",
        #            "--warmup_epochs", "8",
        #            "--early_stopping", "15"
        #            ]
        # },

        # ============================================================================
        # TSMixer-SGA-KG 知识引导实验（已注释）
        # ============================================================================
        
        # {
        #     "name": "🧠 KG-1: FD002基础配置（λ=0.5, weighted池化）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_sga_kg",
        #            "--fault", "FD002",
        #            "--batch_size", "576",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0002",
        #            "--tsmixer_layers", "6",
        #            "--time_expansion", "6",
        #            "--feat_expansion", "5",
        #            "--dropout", "0.12",
        #            "--sga_time_rr", "4",
        #            "--sga_feat_rr", "4",
        #            "--lambda_prior", "0.5",        # 先验权重平衡
        #            "--sga_dropout", "0.08",
        #            "--kg_pool", "weighted",        # 末端加权
        #            "--scheduler", "cosine",
        #            "--warmup_epochs", "8",
        #            "--early_stopping", "15"
        #            ]
        # },
        
        # {
        #     "name": "🔬 KG-2: 强先验引导（λ=0.7, 更依赖物理知识）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_sga_kg",
        #            "--fault", "FD002",
        #            "--batch_size", "576",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0002",
        #            "--tsmixer_layers", "6",
        #            "--time_expansion", "6",
        #            "--feat_expansion", "5",
        #            "--dropout", "0.12",
        #            "--sga_time_rr", "4",
        #            "--sga_feat_rr", "4",
        #            "--lambda_prior", "0.7",        # 更强先验引导
        #            "--sga_dropout", "0.08",
        #            "--kg_pool", "weighted",
        #            "--scheduler", "cosine",
        #            "--warmup_epochs", "8",
        #            "--early_stopping", "15"
        #            ]
        # },
        
        # {
        #     "name": "📊 KG-3: 弱先验引导（λ=0.3, 更依赖数据驱动）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_sga_kg",
        #            "--fault", "FD002",
        #            "--batch_size", "576",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0002",
        #            "--tsmixer_layers", "6",
        #            "--time_expansion", "6",
        #            "--feat_expansion", "5",
        #            "--dropout", "0.12",
        #            "--sga_time_rr", "4",
        #            "--sga_feat_rr", "4",
        #            "--lambda_prior", "0.3",        # 更弱先验引导
        #            "--sga_dropout", "0.08",
        #            "--kg_pool", "weighted",
        #            "--scheduler", "cosine",
        #            "--warmup_epochs", "8",
        #            "--early_stopping", "15"
        #            ]
        # },
        
        # {
        #     "name": "🎯 KG-4: 末端聚焦（last池化）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_sga_kg",
        #            "--fault", "FD002",
        #            "--batch_size", "576",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0002",
        #            "--tsmixer_layers", "6",
        #            "--time_expansion", "6",
        #            "--feat_expansion", "5",
        #            "--dropout", "0.12",
        #            "--sga_time_rr", "4",
        #            "--sga_feat_rr", "4",
        #            "--lambda_prior", "0.5",
        #            "--sga_dropout", "0.08",
        #            "--kg_pool", "last",            # 仅关注最后时刻
        #            "--scheduler", "cosine",
        #            "--warmup_epochs", "8",
        #            "--early_stopping", "15"
        #            ]
        # },
        
        # {
        #     "name": "⚡ KG-5: 轻量高效（更大压缩比rr=6）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_sga_kg",
        #            "--fault", "FD002",
        #            "--batch_size", "768",          # 更大批量
        #            "--epochs", "80",
        #            "--learning_rate", "0.001",
        #            "--weight_decay", "0.00025",
        #            "--tsmixer_layers", "5",
        #            "--time_expansion", "5",
        #            "--feat_expansion", "4",
        #            "--dropout", "0.10",
        #            "--sga_time_rr", "6",           # 更轻量的SGA
        #            "--sga_feat_rr", "6",
        #            "--lambda_prior", "0.5",
        #            "--sga_dropout", "0.05",
        #            "--kg_pool", "weighted",
        #            "--scheduler", "plateau",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "💪 KG-6: 深层强化（layers=8, 强正则）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_sga_kg",
        #            "--fault", "FD002",
        #            "--batch_size", "384",          # 深层模型减小批量
        #            "--epochs", "120",
        #            "--learning_rate", "0.0006",
        #            "--weight_decay", "0.0003",
        #            "--tsmixer_layers", "8",        # 更深网络
        #            "--time_expansion", "6",
        #            "--feat_expansion", "5",
        #            "--dropout", "0.15",            # 强正则化
        #            "--sga_time_rr", "4",
        #            "--sga_feat_rr", "4",
        #            "--lambda_prior", "0.5",
        #            "--sga_dropout", "0.10",
        #            "--kg_pool", "weighted",
        #            "--scheduler", "cosine",
        #            "--warmup_epochs", "10",
        #            "--early_stopping", "18"
        #            ]
        # },

        # ============================================================================
        # TSMixer-SGA 旧模型实验（已注释）
        # ============================================================================
        
        # {
        #     "name": "🟩 TSMixer-SGA-1: FD002基础+SGA（稳健起步）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_sga",
        #            "--fault", "FD002",
        #            "--batch_size", "3072",  # 充分利用RTX 3090的24GB显存
        #            "--epochs", "70",
        #            "--learning_rate", "0.0009",
        #            "--weight_decay", "0.0002",
        #            "--tsmixer_layers", "5",
        #            "--time_expansion", "4",
        #            "--feat_expansion", "4",
        #            "--dropout", "0.12",
        #            "--use_sga",
        #            "--sga_time_rr", "4",
        #            "--sga_feat_rr", "4",
        #            "--sga_dropout", "0.05",
        #            "--sga_pool", "weighted",       # 多工况更稳
        #            "--scheduler", "plateau",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "🚀 TSMixer-SGA-2: 深层Mixer + 轻SGA（容量↑，门控更轻）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_sga",
        #            "--fault", "FD002",
        #            "--batch_size", "160",
        #            "--epochs", "90",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.00025",
        #            "--tsmixer_layers", "6",        # 深一些
        #            "--time_expansion", "5",
        #            "--feat_expansion", "4",
        #            "--dropout", "0.12",
        #            "--use_sga",
        #            "--sga_time_rr", "6",           # 更大压缩比=更轻
        #            "--sga_feat_rr", "6",
        #            "--sga_dropout", "0.05",
        #            "--sga_pool", "weighted",
        #            "--scheduler", "cosine",
        #            "--early_stopping", "14"
        #            ]
        # },
        
        # {
        #     "name": "⚡ TSMixer-SGA-3: 中层Mixer + 强SGA（更强剔噪）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_sga",
        #            "--fault", "FD002",
        #            "--batch_size", "176",
        #            "--epochs", "80",
        #            "--learning_rate", "0.00085",
        #            "--weight_decay", "0.0003",
        #            "--tsmixer_layers", "5",
        #            "--time_expansion", "6",
        #            "--feat_expansion", "5",
        #            "--dropout", "0.15",            # 正则更强
        #            "--use_sga",
        #            "--sga_time_rr", "4",           # 门控更"细"
        #            "--sga_feat_rr", "4",
        #            "--sga_dropout", "0.08",
        #            "--sga_pool", "weighted",
        #            "--scheduler", "plateau",
        #            "--early_stopping", "15"
        #            ]
        # },
        
        # {
        #     "name": "🎯 TSMixer-SGA-4: 池化策略对比（末端last）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_sga",
        #            "--fault", "FD002",
        #            "--batch_size", "192",
        #            "--epochs", "70",
        #            "--learning_rate", "0.0009",
        #            "--weight_decay", "0.0002",
        #            "--tsmixer_layers", "5",
        #            "--time_expansion", "4",
        #            "--feat_expansion", "4",
        #            "--dropout", "0.12",
        #            "--use_sga",
        #            "--sga_time_rr", "5",
        #            "--sga_feat_rr", "5",
        #            "--sga_dropout", "0.06",
        #            "--sga_pool", "last",           # 末期RUL更敏感
        #            "--scheduler", "plateau",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "🧪 TSMixer-SGA-5: OneCycle收敛加速 + mean池化",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_sga",
        #            "--fault", "FD002",
        #            "--batch_size", "180",
        #            "--epochs", "70",
        #            "--learning_rate", "0.0010",    # OneCycle 允许略高峰值
        #            "--weight_decay", "0.0002",
        #            "--tsmixer_layers", "6",
        #            "--time_expansion", "5",
        #            "--feat_expansion", "5",
        #            "--dropout", "0.12",
        #            "--use_sga",
        #            "--sga_time_rr", "6",
        #            "--sga_feat_rr", "6",
        #            "--sga_dropout", "0.05",
        #            "--sga_pool", "mean",           # 对比 weighted/last
        #            "--scheduler", "onecycle",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "🧰 TSMixer-SGA-6: 无SGA对照（确认增益）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tsmixer_sga",
        #            "--fault", "FD002",
        #            "--batch_size", "192",
        #            "--epochs", "70",
        #            "--learning_rate", "0.0009",
        #            "--weight_decay", "0.0002",
        #            "--tsmixer_layers", "5",
        #            "--time_expansion", "4",
        #            "--feat_expansion", "4",
        #            "--dropout", "0.12",
        #            # 不使用 --use_sga，验证SGA的效果
        #            "--sga_pool", "weighted",       # 保持其它设置一致
        #            "--scheduler", "plateau",
        #            "--early_stopping", "12"
        #            ]
        # },

        
        # ============================================================================
        # TokenPool 纯注意力池化实验 - 无CNN前端，直接学习时间特征
        # ============================================================================
        
        # {
        #     "name": "🟩 FD002-A: 提高token粒度（patch=4）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tokenpool",
        #            "--fault", "FD002",
        #            "--batch_size", "192",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0002",
        #            "--patch", "4",
        #            "--d_model", "160",
        #            "--depth", "5",
        #            "--token_mlp_dim", "320",
        #            "--channel_mlp_dim", "160",
        #            "--dropout", "0.12",
        #            "--cnn_pool", "weighted",
        #            "--tokenpool_heads", "8",
        #            "--tokenpool_dropout", "0.12",
        #            "--tokenpool_temperature", "1.8",
        #            "--scheduler", "cosine",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "🟦 FD002-B: 多头分工（heads=10, 温度2.0）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tokenpool",
        #            "--fault", "FD002",
        #            "--batch_size", "192",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0002",
        #            "--patch", "5",
        #            "--d_model", "200",
        #            "--depth", "5",
        #            "--token_mlp_dim", "400",
        #            "--channel_mlp_dim", "200",
        #            "--dropout", "0.12",
        #            "--cnn_pool", "weighted",
        #            "--tokenpool_heads", "10",
        #            "--tokenpool_dropout", "0.12",
        #            "--tokenpool_temperature", "2.0",
        #            "--scheduler", "cosine",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "🟨 FD002-C: 更细粒度上限（patch=3）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tokenpool",
        #            "--fault", "FD002",
        #            "--batch_size", "160",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0002",
        #            "--patch", "3",  
        #            "--d_model", "160",
        #            "--depth", "5",
        #            "--token_mlp_dim", "384",
        #            "--channel_mlp_dim", "160",
        #            "--dropout", "0.14",
        #            "--cnn_pool", "weighted",
        #            "--tokenpool_heads", "8",
        #            "--tokenpool_dropout", "0.14",
        #            "--tokenpool_temperature", "1.9",
        #            "--scheduler", "cosine",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "🟪 FD002-D: 大容量稳态（d_model↑, depth↑）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tokenpool",
        #            "--fault", "FD002",
        #            "--batch_size", "160",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0007",
        #            "--weight_decay", "0.00025",
        #            "--patch", "5",
        #            "--d_model", "192",
        #            "--depth", "6",
        #            "--token_mlp_dim", "384",
        #            "--channel_mlp_dim", "192",
        #            "--dropout", "0.13",
        #            "--cnn_pool", "weighted",
        #            "--tokenpool_heads", "8",
        #            "--tokenpool_dropout", "0.12",
        #            "--tokenpool_temperature", "1.7",
        #            "--scheduler", "cosine",
        #            "--early_stopping", "14"
        #            ]
        # },
        
        # {
        #     "name": "🟥 FD002-E: 末端更关注（pool=last 对照）",
        #     "cmd": ["python", "train.py",
        #            "--model", "tokenpool",
        #            "--fault", "FD002",
        #            "--batch_size", "192",
        #            "--epochs", "100",
        #            "--learning_rate", "0.0008",
        #            "--weight_decay", "0.0002",
        #            "--patch", "4",
        #            "--d_model", "160",
        #            "--depth", "5",
        #            "--token_mlp_dim", "320",
        #            "--channel_mlp_dim", "160",
        #            "--dropout", "0.12",
        #            "--cnn_pool", "last",
        #            "--tokenpool_heads", "8",
        #            "--tokenpool_dropout", "0.12",
        #            "--tokenpool_temperature", "1.8",
        #            "--scheduler", "plateau",
        #            "--early_stopping", "12"
        #            ]
        # },
    
        
        # {
        #     "name": "⚡ TokenPool-3: FD004极限挑战",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tokenpool", 
        #            "--fault", "FD004", 
        #            "--batch_size", "128",           # FD004最复杂，小批量
        #            "--epochs", "100",                # FD004需要充分训练
        #            "--learning_rate", "0.0006",     # FD004保守学习率
        #            "--weight_decay", "0.0003",      # 强权重衰减
        #            # TokenPool参数 - 为复杂数据集优化
        #            "--patch", "5",                  # 保持10个tokens
        #            "--d_model", "160",              # 更大模型容量
        #            "--depth", "6",                  # 深层TSMixer
        #            "--token_mlp_dim", "384",        # 大MLP
        #            "--channel_mlp_dim", "192",
        #            "--dropout", "0.15",             # 强dropout防过拟合
        #            "--cnn_pool", "weighted",        # 关注后期特征
        #            "--tokenpool_heads", "8",        # 更多注意力头处理复杂模式
        #            "--tokenpool_dropout", "0.15",   
        #            "--tokenpool_temperature", "2.0", # 更高温度防塌缩
        #            "--scheduler", "cosine",
        #            "--early_stopping", "15"
        #            ]
        # },
        
        # ============================================================================
        # CNN-TSMixer vs TSMixer 对比实验 - 目标超越 11.39 RMSE (TSMixer最佳)
        # ============================================================================
        
        # {
        #     "name": "🎯 CNN-TSMixer-1: 对标冠军配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "384",           # 对标TSMixer冠军
        #            "--epochs", "50",
        #            "--learning_rate", "0.005",      # 对标高学习率
        #            "--weight_decay", "0.00005",     # 对标低权重衰减
        #            # CNN-TSMixer特定参数
        #            "--patch", "5",                  # 适中patch size
        #            "--cnn_channels", "64",
        #            "--cnn_layers", "2",             # 轻量CNN前端
        #            "--cnn_kernel", "5",
        #            "--d_model", "128",
        #            "--depth", "4",                  # 对标TSMixer层数
        #            "--token_mlp_dim", "256",
        #            "--channel_mlp_dim", "128",
        #            "--dropout", "0.05",             # 对标极低dropout
        #            "--cnn_pool", "mean",
        #            "--scheduler", "plateau",
        #            "--early_stopping", "10"
        #            ]
        # },
        
        # {
        #     "name": "🚀 CNN-TSMixer-2: 增强CNN前端",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "256",           # 适中批量适应更复杂模型
        #            "--epochs", "60",
        #            "--learning_rate", "0.003",      # 稍低学习率适应复杂度
        #            "--weight_decay", "0.0001",
        #            # 强化CNN特征提取
        #            "--patch", "4",                  # 更细粒度patch
        #            "--cnn_channels", "80",          # 更多CNN通道
        #            "--cnn_layers", "3",             # 更深CNN
        #            "--cnn_kernel", "7",             # 更大卷积核
        #            "--d_model", "160",              # 更大模型维度
        #            "--depth", "4",
        #            "--token_mlp_dim", "320",
        #            "--channel_mlp_dim", "160",
        #            "--dropout", "0.08",             # 适当增加正则化
        #            "--cnn_pool", "weighted",        # 关注后期特征
        #            "--scheduler", "plateau",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "⚡ CNN-TSMixer-3: 高效轻量版",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "512",           # 大批量快速训练
        #            "--epochs", "40",
        #            "--learning_rate", "0.006",      # 更高学习率快速收敛
        #            "--weight_decay", "0.00003",     # 更低权重衰减
        #            # 轻量高效配置
        #            "--patch", "8",                  # 大patch减少计算
        #            "--cnn_channels", "48",          # 轻量CNN
        #            "--cnn_layers", "2",
        #            "--cnn_kernel", "3",             # 小卷积核
        #            "--d_model", "96",               # 小模型维度
        #            "--depth", "3",                  # 浅层TSMixer
        #            "--token_mlp_dim", "192",
        #            "--channel_mlp_dim", "96",
        #            "--dropout", "0.03",             # 极低dropout
        #            "--cnn_pool", "last",            # 关注最终状态
        #            "--scheduler", "plateau",
        #            "--early_stopping", "8"
        #            ]
        # },
        
        # {
        #     "name": "🔥 CNN-TSMixer-4: 深度混合架构",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "192",           # 中等批量适应深层网络
        #            "--epochs", "80",
        #            "--learning_rate", "0.002",      # 深层网络保守学习率
        #            "--weight_decay", "0.0002",
        #            # 深层混合架构
        #            "--patch", "5",
        #            "--cnn_channels", "96",          # 丰富CNN特征
        #            "--cnn_layers", "4",             # 深层CNN
        #            "--cnn_kernel", "5",
        #            "--d_model", "192",              # 大模型容量
        #            "--depth", "6",                  # 深层TSMixer
        #            "--token_mlp_dim", "384",
        #            "--channel_mlp_dim", "192",
        #            "--dropout", "0.10",             # 适中正则化
        #            "--cnn_pool", "weighted",
        #            "--scheduler", "onecycle",       # 深层网络用onecycle
        #            "--early_stopping", "15"
        #            ]
        # },
        
        # {
        #     "name": "🎨 CNN-TSMixer-5: 精细调优版",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "320",           # 精心选择的批量大小
        #            "--epochs", "70",
        #            "--learning_rate", "0.0035",     # 精调学习率
        #            "--weight_decay", "0.00008",     # 精调权重衰减
        #            # 精细调优参数
        #            "--patch", "6",                  # 平衡的patch size
        #            "--cnn_channels", "72",          # 精调通道数
        #            "--cnn_layers", "3",
        #            "--cnn_kernel", "6",             # 偶数卷积核
        #            "--d_model", "144",              # 144 = 72 * 2
        #            "--depth", "5",                  # 中等深度
        #            "--token_mlp_dim", "288",        # 144 * 2
        #            "--channel_mlp_dim", "144",
        #            "--dropout", "0.06",             # 精调dropout
        #            "--cnn_pool", "mean",
        #            "--scheduler", "plateau",
        #            "--early_stopping", "12"
        #            ]
        # },
        
        # {
        #     "name": "🏆 CNN-TSMixer-6: 极限挑战版",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "128",           # 小批量支持大模型
        #            "--epochs", "100",
        #            "--learning_rate", "0.001",      # 保守学习率长期训练
        #            "--weight_decay", "0.0003",
        #            # 极限配置
        #            "--patch", "3",                  # 最细粒度
        #            "--cnn_channels", "128",         # 最多CNN通道
        #            "--cnn_layers", "4",             # 深层CNN
        #            "--cnn_kernel", "7",             # 大卷积核
        #            "--d_model", "256",              # 最大模型维度
        #            "--depth", "8",                  # 最深TSMixer
        #            "--token_mlp_dim", "512",        # 最大MLP
        #            "--channel_mlp_dim", "256",
        #            "--dropout", "0.12",             # 强正则化防过拟合
        #            "--cnn_pool", "weighted",
        #            "--scheduler", "cosine",         # 长期训练用cosine
        #            "--early_stopping", "20"
        #            ]
        # },

        # ============================================================================
        # 门控CNN-TSMixer实验 - 自适应特征融合架构
        # ============================================================================
        
        # {
        #     "name": "🔥 门控CNN-TSMixer-1: 对标TSMixer冠军",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer_gated", 
        #            "--fault", "FD001", 
        #            "--batch_size", "384",           # 对标TSMixer冠军配置
        #            "--epochs", "50",
        #            "--learning_rate", "0.005",      # 高学习率
        #            "--weight_decay", "0.00005",     # 极低权重衰减
        #            # 门控CNN-TSMixer参数
        #            "--patch", "5",                  # 适中patch
        #            "--cnn_channels", "64",          # 标准CNN通道
        #            "--cnn_layers", "2",             # 轻量CNN前端
        #            "--cnn_kernel", "3",             # 小卷积核保持长度
        #            "--d_model", "128",
        #            "--depth", "4",                  # 对标TSMixer层数
        #            "--token_mlp_dim", "256",
        #            "--channel_mlp_dim", "128",
        #            "--dropout", "0.05",             # 极低dropout
        #            "--cnn_pool", "mean",
        #            "--use_groupnorm",               # 使用GroupNorm
        #            "--gn_groups", "8",              # 8组GroupNorm
        #            "--scheduler", "plateau",
        #            "--early_stopping", "10"
        #            ]
        # },
        
        

        # ============================================================================
        # 门控CNN-TSMixer最佳配置 - 全数据集测试 (基于FD001最优结果11.23 RMSE)
        # ============================================================================
        
        # {
        #     "name": "🥇 门控CNN-TSMixer最佳配置 + FD001",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer_gated", 
        #            "--fault", "FD001", 
        #            "--batch_size", "384",           # 最佳配置
        #            "--epochs", "50",
        #            "--learning_rate", "0.005",      # 高学习率
        #            "--weight_decay", "0.00005",     # 极低权重衰减
        #            # 门控CNN-TSMixer最佳参数
        #            "--patch", "5",                  # 最佳patch size
        #            "--cnn_channels", "64",          # 最佳CNN通道数
        #            "--cnn_layers", "2",             # 最佳CNN层数
        #            "--cnn_kernel", "3",             # 最佳卷积核大小
        #            "--d_model", "128",              # 最佳模型维度
        #            "--depth", "4",                  # 最佳TSMixer层数
        #            "--token_mlp_dim", "256",        # 最佳Token MLP维度
        #            "--channel_mlp_dim", "128",      # 最佳Channel MLP维度
        #            "--dropout", "0.05",             # 最佳dropout
        #            "--cnn_pool", "mean",            # 最佳池化方式
        #            "--use_groupnorm",               # 使用GroupNorm
        #            "--gn_groups", "8",              # 最佳GroupNorm分组
        #            "--scheduler", "plateau",        # 最佳调度器
        #            "--early_stopping", "10"         # 最佳早停策略
        #            ]
        # },
        
        # {
        #     "name": "🥇 门控CNN-TSMixer最佳配置 + FD002",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer_gated", 
        #            "--fault", "FD002", 
        #            "--batch_size", "256",           # FD002复杂数据适当减小批量
        #            "--epochs", "60",                # FD002需要更多训练轮数
        #            "--learning_rate", "0.003",      # FD002用稍低学习率
        #            "--weight_decay", "0.0001",      # FD002增加权重衰减
        #            # 门控CNN-TSMixer最佳参数
        #            "--patch", "5",                  
        #            "--cnn_channels", "64",          
        #            "--cnn_layers", "2",             
        #            "--cnn_kernel", "3",             
        #            "--d_model", "128",              
        #            "--depth", "4",                  
        #            "--token_mlp_dim", "256",        
        #            "--channel_mlp_dim", "128",      
        #            "--dropout", "0.08",             # FD002增加dropout防过拟合
        #            "--cnn_pool", "mean",            
        #            "--use_groupnorm",               
        #            "--gn_groups", "8",              
        #            "--scheduler", "plateau",        
        #            "--early_stopping", "12"         # FD002增加早停耐心值
        #            ]
        # },
        
        # {
        #     "name": "🥇 门控CNN-TSMixer最佳配置 + FD003",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer_gated", 
        #            "--fault", "FD003", 
        #            "--batch_size", "384",           # FD003与FD001类似，使用相同配置
        #            "--epochs", "50",
        #            "--learning_rate", "0.005",      
        #            "--weight_decay", "0.00005",     
        #            # 门控CNN-TSMixer最佳参数
        #            "--patch", "5",                  
        #            "--cnn_channels", "64",          
        #            "--cnn_layers", "2",             
        #            "--cnn_kernel", "3",             
        #            "--d_model", "128",              
        #            "--depth", "4",                  
        #            "--token_mlp_dim", "256",        
        #            "--channel_mlp_dim", "128",      
        #            "--dropout", "0.05",             
        #            "--cnn_pool", "mean",            
        #            "--use_groupnorm",               
        #            "--gn_groups", "8",              
        #            "--scheduler", "plateau",        
        #            "--early_stopping", "10"         
        #            ]
        # },
        
        # {
        #     "name": "🥇 门控CNN-TSMixer最佳配置 + FD004",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer_gated", 
        #            "--fault", "FD004", 
        #            "--batch_size", "192",           # FD004最复杂，进一步减小批量
        #            "--epochs", "80",                # FD004需要最多训练轮数
        #            "--learning_rate", "0.002",      # FD004用更保守学习率
        #            "--weight_decay", "0.0002",      # FD004增加更多权重衰减
        #            # 门控CNN-TSMixer最佳参数
        #            "--patch", "5",                  
        #            "--cnn_channels", "64",          
        #            "--cnn_layers", "2",             
        #            "--cnn_kernel", "3",             
        #            "--d_model", "128",              
        #            "--depth", "4",                  
        #            "--token_mlp_dim", "256",        
        #            "--channel_mlp_dim", "128",      
        #            "--dropout", "0.10",             # FD004最高dropout防过拟合
        #            "--cnn_pool", "mean",            
        #            "--use_groupnorm",               
        #            "--gn_groups", "8",              
        #            "--scheduler", "plateau",        
        #            "--early_stopping", "15"         # FD004最高早停耐心值
        #            ]
        # },

        # # ============================================================================
        # # FD002专项优化实验 - 针对多工况单故障模式的性能提升
        # # ============================================================================
        
        # {
        #     "name": "🎯 FD002优化-1: 强正则化配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer_gated", 
        #            "--fault", "FD002", 
        #            "--batch_size", "128",           # 减小批量提升泛化
        #            "--epochs", "80",                # 更多轮数
        #            "--learning_rate", "0.002",      # 更保守学习率
        #            "--weight_decay", "0.0003",      # 更强权重衰减
        #            # 强正则化参数
        #            "--patch", "5",                  
        #            "--cnn_channels", "64",          
        #            "--cnn_layers", "2",             
        #            "--cnn_kernel", "3",             
        #            "--d_model", "128",              
        #            "--depth", "4",                  
        #            "--token_mlp_dim", "256",        
        #            "--channel_mlp_dim", "128",      
        #            "--dropout", "0.15",             # 大幅增加dropout
        #            "--cnn_pool", "mean",            
        #            "--use_groupnorm",               
        #            "--gn_groups", "8",              
        #            "--scheduler", "plateau",        
        #            "--early_stopping", "15"         # 更大耐心值
        #            ]
        # },
        
        # {
        #     "name": "🚀 FD002优化-2: 增强CNN特征提取",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer_gated", 
        #            "--fault", "FD002", 
        #            "--batch_size", "192",           # 适中批量
        #            "--epochs", "70",                
        #            "--learning_rate", "0.0025",     # 适中学习率
        #            "--weight_decay", "0.0002",      
        #            # 增强CNN配置
        #            "--patch", "4",                  # 更细粒度patch
        #            "--cnn_channels", "96",          # 更多CNN通道
        #            "--cnn_layers", "3",             # 更深CNN层
        #            "--cnn_kernel", "3",             
        #            "--d_model", "160",              # 更大模型容量
        #            "--depth", "5",                  # 更深TSMixer
        #            "--token_mlp_dim", "320",        
        #            "--channel_mlp_dim", "160",      
        #            "--dropout", "0.12",             # 适中dropout
        #            "--cnn_pool", "weighted",        # 加权池化关注后期
        #            "--use_groupnorm",               
        #            "--gn_groups", "12",             # 更多GroupNorm组
        #            "--scheduler", "onecycle",       # OneCycle调度器
        #            "--early_stopping", "12"         
        #            ]
        # },
        
        # {
        #     "name": "⚡ FD002优化-3: 稳定训练配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer_gated", 
        #            "--fault", "FD002", 
        #            "--batch_size", "160",           # 中小批量平衡
        #            "--epochs", "100",               # 充足训练轮数
        #            "--learning_rate", "0.0015",     # 更稳定学习率
        #            "--weight_decay", "0.0001",      
        #            # 稳定训练参数
        #            "--patch", "6",                  # 稍大patch减少token数
        #            "--cnn_channels", "80",          # 适中CNN通道
        #            "--cnn_layers", "2",             
        #            "--cnn_kernel", "3",             
        #            "--d_model", "144",              # 稍大模型维度
        #            "--depth", "4",                  
        #            "--token_mlp_dim", "288",        
        #            "--channel_mlp_dim", "144",      
        #            "--dropout", "0.10",             
        #            "--cnn_pool", "mean",            
        #            "--use_groupnorm",               
        #            "--gn_groups", "10",             # 适中GroupNorm组数
        #            "--scheduler", "cosine",         # Cosine调度器长期稳定
        #            "--early_stopping", "20"         # 更大耐心值防早停
        #            ]
        # },

        # ============================================================================
        # FD004高级优化配置 - 基于token数量和时间分辨率的精细调优（多工况+多故障）
        # ============================================================================
        
        # {
        #     "name": "🎯 FD004-Stable-Conditioned: 稳定10token配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer_gated", 
        #            "--fault", "FD004", 
        #            "--batch_size", "192",           # FD004更复杂，减小批量
        #            "--epochs", "80",                # FD004需要更多训练轮数
        #            "--learning_rate", "0.001",      # FD004用更保守学习率
        #            "--weight_decay", "0.0002",      # FD004增加权重衰减
        #            # 稳定10token配置 (50/5=10 tokens, FD004窗口=50)
        #            "--patch", "5",                  # FD004窗口50，patch=5获得10个tokens
        #            "--cnn_channels", "64",
        #            "--cnn_layers", "2",
        #            "--cnn_kernel", "3",
        #            "--d_model", "160",              # 适中模型维度
        #            "--depth", "5",                  # 适中深度
        #            "--token_mlp_dim", "320",
        #            "--channel_mlp_dim", "160",
        #            "--dropout", "0.15",             # FD004需要更强正则化
        #            "--cnn_pool", "weighted",        # 加权池化关注后期
        #            "--use_groupnorm",
        #            "--gn_groups", "8",
        #            "--scheduler", "cosine",         # 长期稳定调度
        #            "--early_stopping", "15"         # FD004需要更大耐心值
        #            ]
        # },
        
    

        # ============================================================================
        # TSMixer + FD004 专项优化配置 - 针对最复杂数据集的参数调优
        # ============================================================================
        
        # {
        #     "name": "🎯 TSMixer + FD004: 强正则化稳定配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD004", 
        #            "--batch_size", "128",           # FD004复杂数据用小批量
        #            "--epochs", "100",               # FD004需要充分训练
        #            "--learning_rate", "0.001",      # FD004用保守学习率
        #            "--weight_decay", "0.0003",      # FD004需要强权重衰减
        #            "--tsmixer_layers", "5",         # 适中层数平衡容量和过拟合
        #            "--time_expansion", "6",         # 较大时间扩展处理复杂时序
        #            "--feat_expansion", "4",         # 适中特征扩展
        #            "--dropout", "0.20",             # FD004需要强正则化防过拟合
        #            "--scheduler", "cosine",         # 长期稳定训练
        #            "--early_stopping", "15"         # FD004需要更大耐心值
        #            ]
        # },
        
        # {
        #     "name": "🚀 TSMixer + FD004: 深层网络配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD004", 
        #            "--batch_size", "96",            # 更小批量适应深层网络
        #            "--epochs", "120",               # 深层网络需要更多训练
        #            "--learning_rate", "0.0008",     # 深层网络用更保守学习率
        #            "--weight_decay", "0.0004",      # 深层网络增加权重衰减
        #            "--tsmixer_layers", "8",         # 更深的网络
        #            "--time_expansion", "8",         # 更大时间扩展
        #            "--feat_expansion", "5",         # 更大特征扩展
        #            "--dropout", "0.25",             # 深层网络需要更强正则化
        #            "--scheduler", "onecycle",       # 深层网络适合OneCycle
        #            "--early_stopping", "20"         # 深层网络需要更多耐心
        #            ]
        # },
        
        # {
        #     "name": "⚡ TSMixer + FD004: 高效轻量配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD004", 
        #            "--batch_size", "192",           # 轻量模型可用较大批量
        #            "--epochs", "80",                # 轻量模型训练更快
        #            "--learning_rate", "0.0015",     # 轻量模型可用稍高学习率
        #            "--weight_decay", "0.0002",      # 适中权重衰减
        #            "--tsmixer_layers", "3",         # 较浅网络
        #            "--time_expansion", "4",         # 适中时间扩展
        #            "--feat_expansion", "3",         # 适中特征扩展
        #            "--dropout", "0.15",             # 适中正则化
        #            "--scheduler", "plateau",        # 快速响应调度器
        #            "--early_stopping", "12"         # 适中早停耐心
        #            ]
        # },
        
        # {
        #     "name": "🚀 FD004-Deeper-Token12: 深度12token配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer_gated", 
        #            "--fault", "FD004", 
        #            "--batch_size", "160",           # FD004深层模型用更小批量
        #            "--epochs", "90",                # FD004需要更多轮数
        #            "--learning_rate", "0.0008",     # FD004深层模型更保守学习率
        #            "--weight_decay", "0.0002",      # 增加权重衰减
        #            # 深度12token配置 (50//4=12 tokens, T_eff=48)
        #            "--patch", "4",                  # 更细粒度patch获得12个tokens
        #            "--cnn_channels", "80",          # 更多CNN通道
        #            "--cnn_layers", "3",             # 更深CNN层
        #            "--cnn_kernel", "3",
        #            "--d_model", "192",              # 更大模型容量
        #            "--depth", "6",                  # 更深TSMixer
        #            "--token_mlp_dim", "384",
        #            "--channel_mlp_dim", "192",
        #            "--dropout", "0.18",             # FD004深层模型需要更强正则化
        #            "--cnn_pool", "weighted",        # 末端敏感池化
        #            "--use_groupnorm",
        #            "--gn_groups", "10",             # 更多GroupNorm组
        #            "--scheduler", "cosine",
        #            "--early_stopping", "18"         # FD004深层模型需要更多耐心
        #            ]
        # },
        
        # {
        #     "name": "⚡ FD004-Light-Fast: 轻量16token高效配置",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer_gated", 
        #            "--fault", "FD004", 
        #            "--batch_size", "224",           # FD004轻量版适中批量
        #            "--epochs", "70",                # FD004需要更多轮数
        #            "--learning_rate", "0.0012",     # FD004稍保守学习率
        #            "--weight_decay", "0.0002",      # FD004增加正则化
        #            # 轻量16token配置 (50//3=16 tokens, T_eff=48)
        #            "--patch", "3",                  # 小patch获得16个tokens
        #            "--cnn_channels", "64",          # 轻量CNN通道
        #            "--cnn_layers", "2",
        #            "--cnn_kernel", "3",
        #            "--d_model", "144",              # 平衡的模型维度
        #            "--depth", "5",                  # 适中深度
        #            "--token_mlp_dim", "288",
        #            "--channel_mlp_dim", "144",
        #            "--dropout", "0.20",             # FD004轻量模型需要更强dropout
        #            "--cnn_pool", "weighted",
        #            "--use_groupnorm",
        #            "--gn_groups", "8",
        #            "--scheduler", "plateau",        # 快速响应的plateau调度
        #            "--early_stopping", "15"         # FD004需要更多耐心
        #            ]
        # },
        
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
        
        # {
        #     "name": "🥇 TSMixer冠军配置 (11.46 RMSE)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "384",           # 大批量训练
        #            "--epochs", "50",
        #            "--learning_rate", "0.005",      # 极高学习率
        #            "--weight_decay", "0.00005",     # 极低权重衰减
        #            "--tsmixer_layers", "4",         # 轻量网络
        #            "--time_expansion", "3",         # 适中时间扩展
        #            "--feat_expansion", "4",         # 适中特征扩展
        #            "--dropout", "0.05",             # 极低dropout
        #            "--scheduler", "plateau",        # 平台调度器
        #            "--early_stopping", "10"          # 激进早停
        #            ]
        # },
        
        # {
        #     "name": "🥈 TSMixer亚军配置 (11.69 RMSE)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "512",           # 超大批量
        #            "--epochs", "60",
        #            "--learning_rate", "0.004",      # 高学习率
        #            "--weight_decay", "0.0001",      # 低权重衰减
        #            "--tsmixer_layers", "5",         # 中等深度
        #            "--time_expansion", "4",         # 平衡时间扩展
        #            "--feat_expansion", "5",         # 较大特征扩展
        #            "--dropout", "0.08",             # 低dropout
        #            "--scheduler", "plateau",        # 平台调度器
        #            "--early_stopping", "10"          # 快速早停
        #            ]
        # },
        
        # {
        #     "name": "🥉 TSMixer季军配置 (11.91 RMSE)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "256",           # 中等批量
        #            "--epochs", "100",               # 更多训练轮数
        #            "--learning_rate", "0.0012",     # 精调学习率
        #            "--weight_decay", "0.0002",      # 适中权重衰减
        #            "--tsmixer_layers", "8",         # 深层网络
        #            "--time_expansion", "6",         # 大时间扩展
        #            "--feat_expansion", "4",         # 适中特征扩展
        #            "--dropout", "0.12",             # 适中dropout
        #            "--scheduler", "onecycle",       # OneCycle调度器
        #            "--early_stopping", "12"         # 耐心早停
        #            ]
        # },
        
        # ============================================================================
        # CNN-TSMixer 实验配置 - 新的混合架构
        # ============================================================================
        
        # {
        #     "name": "🚀 CNN-TSMixer + FD001 (基础配置)",
        #     "cmd": ["python", "train.py", 
        #            "--model", "cnn_tsmixer", 
        #            "--fault", "FD001", 
        #            "--batch_size", "128",
        #            "--epochs", "50",
        #            "--learning_rate", "0.001",
        #            "--weight_decay", "0.0001",
        #            # CNN-TSMixer特定参数
        #            "--patch", "5",
        #            "--cnn_channels", "64",
        #            "--cnn_layers", "2",
        #            "--cnn_kernel", "5",
        #            "--d_model", "128",
        #            "--depth", "4",
        #            "--token_mlp_dim", "256",
        #            "--channel_mlp_dim", "128",
        #            "--dropout", "0.10",
        #            "--cnn_pool", "mean",
        #            "--scheduler", "cosine",
        #            "--early_stopping", "10"
        #            ]
        # },
        
        
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
