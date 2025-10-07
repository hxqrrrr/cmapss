# C-MAPSS 航空发动机剩余使用寿命预测项目

基于多种深度学习模型的航空发动机剩余使用寿命（RUL）预测项目，使用NASA C-MAPSS数据集进行训练和评估。

## 项目简介

本项目实现了多种先进的深度学习模型用于航空发动机的剩余使用寿命预测，主要包括：

### 🔥 核心模型系列
- **TSMixer模型** - 基于时间序列混合的轻量级架构
- **TSMixer-SGA** ⭐ 新增 - TSMixer + 双维可扩展全局注意力（SGA）
- **TSMixer-ECA** ⭐ 新增 - ECA通道注意力 + BiTCN时序前端 + TSMixer
- **TSMixer-PTSA** ⭐ 新增 - TSMixer + 金字塔稀疏时空注意力
- **TSMixer-PTSA-Cond** ⭐ 新增 - 条件门控增强版（ECA×Time×FiLM）

### 🚀 混合架构系列
- **CNN-TSMixer模型** - 卷积神经网络与TSMixer的混合架构
- **门控CNN-TSMixer模型** - 具有自适应特征融合的门控机制混合架构
- **pTSMixer** ⭐ 新增 - 并行时间/特征双支TSMixer
- **TokenPool** ⭐ 新增 - 纯注意力池化 + TSMixer

### 📊 经典基线模型
- **Transformer模型** - 基于注意力机制的序列建模
- **BiLSTM模型** - 双向长短期记忆网络
- **RBM-LSTM模型** - 基于受限玻尔兹曼机的半监督学习框架

## 项目结构

```
project/
│
├── dataset.py              # 数据集处理模块
├── models/                 # 模型模块
│   ├── base_model.py      # 基础模型类（统一接口）
│   │
│   ├── # === TSMixer系列 ===
│   ├── tsmixer_model.py   # 基础TSMixer模型
│   ├── tsmixer_sga.py     # TSMixer + SGA注意力 ⭐
│   ├── tsmixer_eca_model.py # ECA + BiTCN + TSMixer ⭐
│   ├── tsmixer_ptsa.py    # TSMixer + 金字塔稀疏注意力 ⭐
│   ├── tsmixer_ptsa_cond_model.py # 条件门控版 ⭐
│   │
│   ├── # === 混合架构系列 ===
│   ├── cnn_tsmixer_model.py # CNN-TSMixer混合模型
│   ├── tsmixer_cnn_gated.py # 门控CNN-TSMixer模型
│   ├── parallel_tsmixer_rul.py # pTSMixer并行双支 ⭐
│   ├── tsmixer_gated_tokenpool.py # TokenPool注意力 ⭐
│   │
│   ├── # === 经典模型 ===
│   ├── transformer_model.py # Transformer模型
│   ├── bilstm_model.py    # BiLSTM模型
│   └── ellefsen_rbm_lstm_model.py # RBM-LSTM模型
│
├── train.py               # 统一训练脚本（支持所有模型）
├── run_experiments.py     # 批量实验运行脚本
├── requirements.txt       # 依赖包列表
├── CMAPSSData/           # C-MAPSS数据集
├── logs/                 # 训练日志输出
└── README.md             # 项目说明文档
```

## 数据集

使用NASA的C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) 数据集，包含四个子数据集：

| 数据集 | 故障模式 | 操作条件 | 训练样本 | 测试样本 | 特征维度 |
|--------|---------|---------|---------|---------|---------|
| **FD001** | 单故障 | 单工况 | ~20,000 | ~13,000 | 14传感器 |
| **FD002** | 单故障 | 多工况(6种) | ~53,000 | ~33,000 | 3工况设置+21传感器 |
| **FD003** | 多故障 | 单工况 | ~24,000 | ~16,000 | 14传感器 |
| **FD004** | 多故障 | 多工况(6种) | ~61,000 | ~41,000 | 3工况设置+21传感器 |

## 环境配置

### 依赖安装

1. 安装依赖包：
```bash
pip install -r requirements.txt
```

2. 主要依赖：
- Python >= 3.8
- PyTorch >= 1.12.0
- numpy
- pandas
- scikit-learn

3. 确保数据集文件在 `CMAPSSData/` 目录下

### GPU支持（推荐）
- CUDA >= 11.0
- cuDNN >= 8.0
- 建议显存 >= 8GB

## 使用方法

### 快速开始

```bash
# 运行批量实验（推荐）
python run_experiments.py

# 单个模型训练示例
python train.py --model tsmixer_sga --fault FD001 --batch_size 256 --epochs 60
```

### 命令行参数

#### 基础参数
```bash
--model              # 模型类型（见下文支持的模型列表）
--fault              # 数据集选择：FD001/FD002/FD003/FD004
--batch_size         # 批量大小（影响GPU利用率）
--epochs             # 训练轮数
--learning_rate      # 学习率
--weight_decay       # 权重衰减
--scheduler          # 学习率调度器：onecycle/plateau/cosine
--early_stopping     # 早停耐心值（0表示不使用早停）
```

#### 性能优化参数
```bash
--num_workers        # 数据加载线程数（-1=自动，Windows建议0）
--prefetch_factor    # DataLoader预取批次数（默认4）
--grad_accum         # 梯度累积步数（显存不够时有效）
--val_batch_mul      # 验证集batch放大倍数（默认2.0）
```

## 模型架构详解

### 1. TSMixer-SGA ⭐ 新增

**特点**：双维可扩展全局注意力机制

```bash
python train.py --model tsmixer_sga \
    --use_sga \              # 启用SGA
    --sga_time_rr 4 \        # 时间方向压缩比
    --sga_feat_rr 4 \        # 特征方向压缩比
    --sga_pool weighted \    # 池化方式：mean/last/weighted
    --tsmixer_layers 5
```

**核心优势**：
- 🔹 双向squeeze注意力（时间+特征）
- 🔹 自适应融合参数（γ_t, γ_f, β）
- 🔹 残差门控机制，训练更稳定
- 🔹 支持mean/last/weighted多种池化

**适用场景**：需要全局特征建模的复杂工况（FD002/FD004）

---

### 2. TSMixer-ECA ⭐ 新增

**特点**：通道注意力 + 双向时间卷积网络

```bash
python train.py --model tsmixer_eca \
    --use_eca \              # 启用ECA通道注意力
    --eca_kernel 5 \         # ECA卷积核大小
    --use_bitcn \            # 启用BiTCN前端
    --tcn_kernel 3 \         # TCN卷积核大小
    --tcn_dilations "1,2,4"  # 膨胀系数
```

**核心优势**：
- 🔹 ECA通道注意力：轻量级、无降维
- 🔹 BiTCN前端：正向+反向时序特征融合
- 🔹 多尺度感受野：通过膨胀卷积捕获长距离依赖
- 🔹 灵活融合方式：mean/sum/cat

**适用场景**：需要多尺度时序特征的场景

---

### 3. TSMixer-PTSA ⭐ 新增

**特点**：金字塔稀疏时空注意力

```bash
python train.py --model tsmixer_ptsa \
    --ptsa_every_k 2 \       # 每2个MixerBlock插入1个PTSA
    --ptsa_heads 6 \         # 注意力头数
    --ptsa_local_window 12 \ # 局部窗口大小
    --ptsa_topk 16 \         # TopK稀疏保留数
    --ptsa_levels 2          # 金字塔层数
```

**核心优势**：
- 🔹 金字塔多尺度注意力（粗→细）
- 🔹 TopK稀疏机制：降低计算复杂度
- 🔹 局部+全局双重注意力
- 🔹 时序下采样蒸馏：逐层压缩

**适用场景**：长序列、需要多尺度时空建模

---

### 4. TSMixer-PTSA-Cond ⭐ 新增

**特点**：条件双门控增强版

```bash
python train.py --model tsmixer_ptsa_cond \
    --cond_dim 3 \           # 条件变量维度（工况设置数）
    --eca_kernel 5 \         # ECA通道门控核大小
    --time_kernel 11 \       # 时间门控核大小
    --use_post_gate          # 启用后置门控
```

**核心优势**：
- 🔹 ECA通道门控：条件自适应通道选择
- 🔹 时间卷积门控：条件自适应时序加权
- 🔹 FiLM机制：特征级条件调制
- 🔹 双重门控：pre-gate + post-gate

**适用场景**：多工况数据集（FD002/FD004）

---

### 5. pTSMixer ⭐ 新增

**特点**：并行时间/特征双支架构

```bash
python train.py --model ptsmixer \
    --pt_depth 6 \           # Block层数
    --pt_ch_expand 4 \       # 特征分支扩展倍数
    --pt_t_kernel 7 \        # 时间分支卷积核
    --pt_droppath 0.1        # DropPath比例
```

**核心优势**：
- 🔹 并行双支：时间支（深度可分离卷积）+ 特征支（MLP）
- 🔹 高效计算：避免串行瓶颈
- 🔹 DropPath正则化：提升泛化能力

**适用场景**：需要高效并行计算的场景

---

### 6. TokenPool ⭐ 新增

**特点**：纯注意力池化机制

```bash
python train.py --model tokenpool \
    --tokenpool_heads 8 \        # 多头注意力数
    --tokenpool_temperature 1.5 \ # 注意力温度
    --tokenpool_dropout 0.1      # 注意力dropout
```

**核心优势**：
- 🔹 多头注意力池化：自适应token重要性
- 🔹 温度控制：防止注意力塌缩
- 🔹 无CNN前端：直接学习时间特征

**适用场景**：需要细粒度时间注意力的场景

---

### 7. 基础TSMixer

**特点**：轻量高效的时序混合架构

```bash
python train.py --model tsmixer \
    --tsmixer_layers 4 \     # TSMixer层数
    --time_expansion 4 \     # 时间混合扩展
    --feat_expansion 4       # 特征混合扩展
```

**核心优势**：
- 🔹 参数量小、训练快
- 🔹 可解释性强
- 🔹 性能与效率平衡好

---

### 8. 门控CNN-TSMixer

**特点**：自适应特征融合

```bash
python train.py --model cnn_tsmixer_gated \
    --use_groupnorm \        # 使用GroupNorm
    --gn_groups 8 \          # GroupNorm分组数
    --cnn_channels 64
```

**核心优势**：
- 🔹 门控融合：自动学习CNN和原始特征权重
- 🔹 GroupNorm：小批量稳定性好
- 🔹 已在FD001-004上验证性能优异

---

### 9. Transformer/BiLSTM/RBM-LSTM

经典基线模型，详见原README说明。

## 性能基准

### 🏆 当前最佳结果

基于实际实验结果，不同模型在C-MAPSS数据集上的性能表现：

#### FD001数据集（单故障模式，单操作条件）
- **门控CNN-TSMixer**: **11.08 RMSE** 🥇 - 最佳性能
- **TSMixer**: **11.39 RMSE** - 轻量级架构
- **CNN-TSMixer**: 约12-13 RMSE
- **TSMixer-SGA**: 待测试 ⭐
- **TSMixer-ECA**: 待测试 ⭐

#### FD002数据集（单故障模式，多操作条件）
- **门控CNN-TSMixer**: **16.69 RMSE** 🥇 - 多工况最佳
- **TSMixer**: **17.68 RMSE**
- **CNN-TSMixer**: 约18-20 RMSE
- **TSMixer-PTSA-Cond**: 待测试 ⭐（预期适合多工况）

#### FD003数据集（多故障模式，单操作条件）
- **门控CNN-TSMixer**: **10.88 RMSE** 🥇 - 全数据集最佳
- **TSMixer**: 约12-15 RMSE

#### FD004数据集（多故障模式，多操作条件）
- **门控CNN-TSMixer**: **16.90 RMSE** 🥇 - 最复杂场景
- **TSMixer**: 约18-22 RMSE
- **TSMixer-PTSA**: 待测试 ⭐（预期适合长序列）

### 📊 性能特点总结
- **难度排序**: FD003(10.88) < FD001(11.08) < FD002(16.69) ≈ FD004(16.90)
- **多工况挑战更大**：FD002/FD004的RMSE显著高于FD001/FD003
- **新增模型优势**：
  - TSMixer-SGA：全局注意力建模
  - TSMixer-ECA：多尺度时序特征
  - TSMixer-PTSA：长序列、多尺度
  - TSMixer-PTSA-Cond：多工况自适应

## GPU利用率优化

### 影响GPU利用率的关键参数

1. **batch_size** ⭐ 最重要
   - 增大 → GPU利用率↑（显存允许情况下）
   - 推荐：8GB显存用256-384，12GB用512+

2. **num_workers**
   - 增大 → 减少GPU等待数据时间
   - Windows：建议0（多进程问题）
   - Linux：建议2-8

3. **模型规模参数**
   - `--tsmixer_layers 6-8`：增加深度
   - `--time_expansion 6`：增加容量
   - `--hidden_channels 128-256`：增加通道数

### 推荐配置

```bash
# GPU利用率优化配置（8-12GB显存）
python train.py --model tsmixer_sga \
    --batch_size 384 \       # 大批量
    --num_workers 4 \        # Linux环境
    --prefetch_factor 4 \    # 预取加速
    --grad_accum 1           # 显存够用时=1
```

## 主要功能

1. **数据处理模块** (`dataset.py`)
   - 多种数据切片策略（滑动窗口、固定窗口等）
   - 自动特征选择（单/多工况）
   - 数据预处理（MinMax归一化）
   - 高效批量数据生成器

2. **模型模块** (`models/`)
   - 统一的BaseRULModel接口设计
   - 支持12种模型架构
   - 自动混合精度训练（AMP）
   - 梯度累积支持

3. **训练模块** (`train.py`)
   - 完整的训练流程
   - 多种学习率调度器（OneCycle/Plateau/Cosine）
   - 早停策略
   - 详细的日志记录

## 扩展性

### 添加新模型

1. 继承`BaseRULModel`基类
2. 实现`build_model()`和`compile()`方法
3. 在`train.py`的`create_model()`中注册
4. 添加对应的命令行参数

示例：
```python
class YourNewModel(BaseRULModel):
    def build_model(self) -> nn.Module:
        # 定义模型架构
        pass
    
    def compile(self, learning_rate, weight_decay, **kwargs):
        # 设置优化器
        pass
```

### 添加新数据集

在`dataset.py`中添加新的数据源支持，保持接口一致即可。

### 超参数优化

支持在`run_experiments.py`中配置多组实验进行网格搜索。

## 实验运行建议

### 快速测试
```bash
# 小数据集快速验证
python train.py --model tsmixer --fault FD001 --epochs 10 --batch_size 128
```

### 完整训练
```bash
# 门控CNN-TSMixer（已验证最佳）
python train.py --model cnn_tsmixer_gated --fault FD001 \
    --batch_size 384 --epochs 50 --learning_rate 0.005 \
    --scheduler plateau --early_stopping 10

# 新增模型测试
python train.py --model tsmixer_sga --fault FD002 \
    --use_sga --sga_pool weighted \
    --batch_size 256 --epochs 70 --early_stopping 12
```

### 批量实验
```bash
# 运行预配置的实验序列
python run_experiments.py
```

## 常见问题

### Q1: Windows下训练很慢？
A: 设置`--num_workers 0`，Windows多进程支持不佳。

### Q2: 显存不足？
A: 减小`--batch_size`或使用`--grad_accum 2`梯度累积。

### Q3: 如何选择模型？
A: 
- 单工况简单场景：TSMixer、门控CNN-TSMixer
- 多工况复杂场景：TSMixer-PTSA-Cond、TSMixer-SGA
- 长序列场景：TSMixer-PTSA
- 需要高解释性：TSMixer、pTSMixer

### Q4: 如何提高GPU利用率？
A: 优先增大`batch_size`，然后调整`num_workers`。

## 更新日志

### 2024-10 版本
- ✅ 新增TSMixer-SGA模型（双维全局注意力）
- ✅ 新增TSMixer-ECA模型（ECA + BiTCN）
- ✅ 新增TSMixer-PTSA模型（金字塔稀疏注意力）
- ✅ 新增TSMixer-PTSA-Cond模型（条件门控）
- ✅ 新增pTSMixer模型（并行双支）
- ✅ 新增TokenPool模型（注意力池化）
- ✅ 优化DataLoader性能（prefetch、persistent_workers）
- ✅ 添加混合精度训练支持（AMP）
- ✅ 添加梯度累积功能
- ✅ 统一所有模型接口

### 历史版本
- 2024-09: 添加门控CNN-TSMixer、TokenPool
- 2024-08: 添加CNN-TSMixer、基础TSMixer
- 2024-07: 初始版本（Transformer、BiLSTM、RBM-LSTM）

## 许可证

MIT License

## 联系方式

如有问题或建议，欢迎提Issue或Pull Request。
