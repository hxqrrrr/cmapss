# C-MAPSS 航空发动机剩余使用寿命预测项目

基于多种深度学习模型的航空发动机剩余使用寿命（RUL）预测项目，使用NASA C-MAPSS数据集进行训练和评估。

## 项目简介

本项目实现了多种先进的深度学习模型用于航空发动机的剩余使用寿命预测，主要包括：
- **TSMixer模型** - 基于时间序列混合的轻量级架构
- **CNN-TSMixer模型** - 卷积神经网络与TSMixer的混合架构
- **门控CNN-TSMixer模型** - 具有自适应特征融合的门控机制混合架构
- **Transformer模型** - 基于注意力机制的序列建模
- **BiLSTM模型** - 双向长短期记忆网络
- **RBM-LSTM模型** - 基于受限玻尔兹曼机的半监督学习框架
- 支持多种数据预处理和切片策略

## 项目结构

```
project/
│
├── dataset.py              # 数据集处理模块
├── models/                 # 模型模块
│   ├── base_model.py      # 基础模型类
│   ├── tsmixer_model.py   # TSMixer模型实现
│   ├── cnn_tsmixer_model.py # CNN-TSMixer混合模型实现
│   ├── tsmixer_cnn_gated.py # 门控CNN-TSMixer模型实现
│   ├── transformer_model.py # Transformer模型实现
│   ├── bilstm_model.py    # BiLSTM模型实现
│   └── ellefsen_rbm_lstm_model.py # RBM-LSTM模型实现
├── train.py               # 训练脚本
├── run_experiments.py     # 实验运行脚本
├── requirements.txt       # 依赖包列表
├── CMAPSSData/           # C-MAPSS数据集
└── README.md             # 项目说明文档
```

## 数据集

使用NASA的C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) 数据集，包含四个子数据集：
- FD001: 单故障模式，单操作条件
- FD002: 单故障模式，多操作条件  
- FD003: 多故障模式，单操作条件
- FD004: 多故障模式，多操作条件

## 环境配置

1. 安装依赖包：
```bash
pip install -r requirements.txt
```

2. 安装torch

3. 确保数据集文件在 `CMAPSSData/` 目录下



##  使用方法

```bash
python run_experiments.py
```

## 模型架构

### TSMixer模型
- 基于时间序列混合的轻量级架构
- 支持多变量时间序列预测
- 具有良好的可解释性和计算效率

### CNN-TSMixer混合模型
- **前端CNN特征提取**: 使用多层卷积网络提取局部时序特征
- **后端TSMixer处理**: 将CNN特征转换为patch序列，利用TSMixer处理长期依赖
- **灵活池化策略**: 支持mean、last、weighted等多种池化方式
- **优势**: 结合CNN的局部特征提取能力和TSMixer的全局建模能力

### 门控CNN-TSMixer模型
- **自适应特征融合**: 通过门控机制自动学习CNN和TSMixer特征的最优组合
- **GroupNorm归一化**: 使用GroupNorm替代BatchNorm，提升小批量和多工况下的稳定性
- **轻量级设计**: 采用小卷积核(kernel=3)和适中深度，避免过度平滑
- **优势**: 具有更强的泛化能力和对复杂数据的适应性

### Transformer模型
- 基于注意力机制的序列建模
- 支持长序列依赖关系学习
- 适用于复杂的时间模式识别

### BiLSTM模型
- 双向长短期记忆网络架构
- 能够捕捉前向和后向的时序依赖关系
- 适合处理中等长度的时间序列

### RBM-LSTM模型
- 基于受限玻尔兹曼机的半监督学习框架
- 支持预训练和端到端训练两种模式
- 能够学习数据的潜在表示

## 主要功能

1. **数据处理模块** (`dataset.py`)
   - 多种数据切片策略（滑动窗口、固定窗口等）
   - 数据预处理（标准化、归一化）
   - 批量数据生成器

2. **模型模块** (`models/`)
   - 统一的模型接口设计
   - 支持模型保存和加载
   - 可扩展的模型架构

3. **训练模块** (`train.py`)
   - 完整的训练流程
   - 早停策略
   - 模型评估和保存

## 实验结果

模型在各数据集上的性能表现：
- 支持RMSE、MAE等多种评估指标
- 训练过程日志记录
- 模型权重自动保存

### 性能基准
基于实际实验结果，不同模型在C-MAPSS数据集上的性能表现：

#### FD001数据集（单故障模式，单操作条件）
- **门控CNN-TSMixer**: **11.08 RMSE** - 最佳性能，自适应特征融合优势明显
- **TSMixer**: **11.39 RMSE** - 轻量级架构，训练高效
- **CNN-TSMixer**: 约12-13 RMSE - 结合CNN局部特征提取能力
- **Transformer**: 约15-20 RMSE - 适合长序列建模
- **BiLSTM**: 约18-25 RMSE - 双向时序依赖捕捉

#### FD002数据集（单故障模式，多操作条件）
- **门控CNN-TSMixer**: **16.69 RMSE** - 在多工况下表现稳定
- **TSMixer**: **17.68 RMSE** - 复杂条件下仍保持良好性能
- **CNN-TSMixer**: 约18-20 RMSE - 通过CNN增强局部特征识别

#### FD003数据集（多故障模式，单操作条件）
- **门控CNN-TSMixer**: **10.88 RMSE** - 多故障模式下的卓越表现
- **TSMixer**: 约12-15 RMSE - 在多故障场景下的良好适应性
- **CNN-TSMixer**: 约13-16 RMSE - 多故障模式特征识别

#### FD004数据集（多故障模式，多操作条件）
- **门控CNN-TSMixer**: **16.90 RMSE** - 最复杂数据集上的优异表现
- **TSMixer**: 约18-22 RMSE - 在最复杂场景下的挑战

#### 性能特点总结
- **门控CNN-TSMixer** 在所有数据集上均表现最佳，特别在FD003上取得**10.88 RMSE**的卓越成绩
- **TSMixer** 提供了性能与效率的良好平衡，在各数据集上都有稳定表现
- **CNN-TSMixer** 在局部特征重要的场景下表现稳定
- **性能排序**: FD003(10.88) < FD001(11.08) < FD002(16.69) < FD004(16.90)
- 多操作条件比多故障模式对模型挑战更大（FD002、FD004 RMSE显著高于FD001、FD003）

## 扩展性

- **更换模型**: 只需修改模型模块，保持接口一致
- **新增数据集**: 在dataset.py中添加新的数据源支持
- **超参数优化**: 支持网格搜索和随机搜索
- **自定义损失函数**: 可在模型中自定义损失函数
