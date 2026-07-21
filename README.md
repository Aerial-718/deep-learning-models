# Deep Learning Models

> 深度学习经典模型的从零实现、数学推导、实验验证与主动回忆练习。

这个仓库用于系统记录不同深度学习模型的学习过程。每个模型不仅包含可以直接训练的
PyTorch 实现，还尽可能提供 NumPy 手写前向/反向传播、数值梯度检查、官方模块对齐、
可复现实验和独立的回忆练习。

## 仓库目标

- 从数学公式出发理解模型，而不只是调用高级 API。
- 用 NumPy 显式实现关键前向传播和反向传播。
- 用 PyTorch 基础张量与 `nn.Parameter` 重建模型核心模块。
- 通过有限差分、官方实现对齐和端到端训练验证正确性。
- 使用 Notebook 记录推导、可视化、消融实验和学习结论。
- 将正式答案与主动回忆练习分离，便于反复复习。

## 已收录模型

| 项目 | 内容 | 状态 |
|---|---|---|
| [RNN_LSTM_GRU](./RNN_LSTM_GRU/) | Vanilla RNN、LSTM、GRU；NumPy BPTT；PyTorch 手写 Cell；字符语言模型 | ✅ 可运行 |

## 仓库结构

```text
deep-learning-models/
├── README.md
├── UPLOAD_FROM_LOCAL.md
└── RNN_LSTM_GRU/
    ├── vanilla_rnn/       # Vanilla RNN 正式实现
    ├── lstm/              # LSTM 正式实现
    ├── gru/               # GRU 正式实现
    ├── exercises/         # 不含答案的主动回忆版本
    ├── notebooks/         # 推导、核心实现与实验
    ├── tests/             # 数值梯度、官方对齐和集成测试
    ├── scripts/           # 数据下载、训练、评估与比较
    └── configs/           # 可复现实验配置
```

每个模型项目是相对独立的 Python 工程，拥有自己的环境说明、测试和运行入口。

## 快速开始

```bash
git clone https://github.com/YOUR_USERNAME/deep-learning-models.git
cd deep-learning-models/RNN_LSTM_GRU

conda env create -f environment.yml
conda activate rnn-lstm-gru
python -m pip install -e .
pytest -q
```

随后启动 Notebook：

```bash
jupyter lab
```

推荐按每个项目 README 中的“正式实现 → Notebook → 回忆练习 → 实验”顺序学习。

## 计划中的模型

- 多层感知机与反向传播基础
- CNN：LeNet、AlexNet、VGG、ResNet
- Autoencoder 与 VAE
- Seq2Seq 与 Attention
- Transformer
- GAN
- Diffusion Models

该列表是学习路线，不代表固定发布时间；每个模型只有在实现、测试和说明齐全后才会加入
“已收录模型”表格。

## 项目约定

- 默认使用 batch-first 张量布局。
- 随机实验必须记录 seed、配置和运行环境。
- 核心反向传播优先使用数值梯度检查。
- 手写 PyTorch 模块必须与官方实现进行前向和梯度对齐。
- 数据集、checkpoint、训练日志和生成结果不提交到 Git。
- Notebook 负责解释和实验，Python 模块是正式实现的唯一权威来源。

## 本地发布到 GitHub

本仓库包是在服务器上生成的，但不会从服务器连接 GitHub。请下载后按照
[UPLOAD_FROM_LOCAL.md](./UPLOAD_FROM_LOCAL.md) 在个人电脑上初始化 Git 并推送。

## License

当前发布包未预设开源许可证。首次发布前，请根据自己的分享和复用需求在 GitHub 仓库中
选择并添加许可证。
