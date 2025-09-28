# C-MAPSS 航空发动机剩余使用寿命预测项目

基于多种深度学习模型的航空发动机剩余使用寿命（RUL）预测项目，使用NASA C-MAPSS数据集进行训练和评估。

## 项目简介

本项目实现了多种先进的深度学习模型用于航空发动机的剩余使用寿命预测，主要包括：
- **TSMixer模型** - 基于时间序列混合的轻量级架构
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

2. 确保数据集文件在 `CMAPSSData/` 目录下

## 使用方法

### 单个模型训练
```bash
python train.py --model tsmixer --dataset FD001 --epochs 100
```

### 批量实验
```bash
python run_experiments.py
```

## 模型架构

### TSMixer模型
- 基于时间序列混合的架构
- 支持多变量时间序列预测
- 具有良好的可解释性

### Transformer模型
- 基于注意力机制的序列建模
- 支持长序列依赖关系学习
- 适用于复杂的时间模式识别

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

## 扩展性

- **更换模型**: 只需修改模型模块，保持接口一致
- **新增数据集**: 在dataset.py中添加新的数据源支持
- **超参数优化**: 支持网格搜索和随机搜索
- **自定义损失函数**: 可在模型中自定义损失函数

## 贡献指南

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至 [your-email@example.com]

## 致谢

- NASA C-MAPSS数据集
- PyTorch深度学习框架
- 相关研究论文和开源项目
