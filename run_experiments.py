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

        
        # Transformer实验
        {
            "name": "Transformer + FD001 (小模型)",
            "cmd": ["python", "train.py", 
                   "--model", "transformer", 
                   "--fault", "FD001", 
                   "--batch_size", "256",
                   "--epochs", "100",
                   "--learning_rate", "0.0003",
                   "--d_model", "128",
                   "--nhead", "8",
                   "--num_layers", "6",
                   ]
        },
        
        {
            "name": "Transformer + FD001 (中等模型)",
            "cmd": ["python", "train.py", 
                   "--model", "transformer", 
                   "--fault", "FD001", 
                   "--batch_size", "256",
                   "--epochs", "30",
                   "--learning_rate", "0.0003",
                   "--d_model", "128",
                   "--nhead", "4",
                   "--num_layers", "3",
                   ]
        },
        
        # FD002实验
        {
            "name": "Transformer + FD002",
            "cmd": ["python", "train.py", 
                   "--model", "transformer", 
                   "--fault", "FD002", 
                   "--batch_size", "256",
                   "--epochs", "40",  # FD002更复杂，需要更多轮次
                   "--learning_rate", "0.0002",  # 稍微降低学习率
                   "--d_model", "128",
                   "--nhead", "4",
                   "--num_layers", "3",
                   ]
        }
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
