# RNN, LSTM & GRU From Scratch

这个项目从零实现三类经典循环神经网络：Vanilla RNN、LSTM 和 GRU。项目同时提供
NumPy 显式 BPTT、PyTorch 手写 Cell、官方实现对齐、字符级语言模型、延迟记忆实验和
主动回忆练习。

它既是一套可运行实现，也是一条完整学习路线：

```text
数学公式 → NumPy 前向 → 显式 BPTT → 数值梯度
        → PyTorch 手写 Cell → 官方模块对齐
        → 字符语言模型与记忆任务 → 回忆复现
```

## 模型概览

| 模型 | 状态 | 门顺序 | NumPy BPTT | PyTorch Cell | 语言模型 |
|---|---|---|---|---|---|
| Vanilla RNN | `h` | 无门控 | ✅ | ✅ | ✅ |
| LSTM | `h, c` | `i, f, g, o` | ✅ | ✅ | ✅ |
| GRU | `h` | `r, z, n` | ✅ | ✅ | ✅ |

GRU 使用与 PyTorch 一致的 reset-after 候选公式，因此能够直接复制权重并与
`torch.nn.GRUCell` 对齐。

## 你会学到什么

- 循环状态如何沿时间维展开。
- 共享参数的梯度为什么需要沿时间累加。
- Vanilla RNN 梯度消失和梯度爆炸的来源。
- LSTM 中 hidden state 与 cell state 的双梯度路径。
- GRU 的 reset/update gate 如何控制候选状态和旧状态。
- 如何通过有限差分验证手写 BPTT。
- 如何只使用基础张量操作复现 PyTorch 官方 Cell。
- 如何构建、训练和采样字符级语言模型。

## 项目结构

```text
RNN_LSTM_GRU/
├── common/              # 数据、数值梯度、指标、训练和采样工具
├── vanilla_rnn/         # Vanilla RNN 正式实现
├── lstm/                # LSTM 正式实现
├── gru/                 # GRU 正式实现
├── exercises/           # 相同接口的空白回忆版本
├── notebooks/           # 数学、核心实现、可视化与对比实验
├── tests/               # 正式实现测试
├── configs/             # Debug、完整训练和 delayed-recall 配置
├── scripts/             # 数据下载、训练、评估与模型比较
├── data/                # 仅保存数据说明；原始语料不提交
├── artifacts/           # checkpoint、日志和生成结果；不提交
├── environment.yml
└── pyproject.toml
```

## 环境安装

```bash
conda env create -f environment.yml
conda activate rnn-lstm-gru
python -m pip install -e .
pytest -q
```

正式测试包含 NumPy 有限差分、PyTorch 官方模块对齐、错误输入、微型 batch 过拟合和
生成接口验证。

## 推荐学习顺序

### 1. 数学与梯度检查

打开 `notebooks/00_math_and_gradcheck.ipynb`，完成 tanh、sigmoid、
softmax-cross-entropy 和中心有限差分练习。

### 2. 阅读正式实现

依次阅读：

```text
vanilla_rnn/numpy_impl.py
lstm/numpy_impl.py
gru/numpy_impl.py
```

每个目录还包含 PyTorch Cell、字符模型、实验工具和三级提示。

### 3. 运行核心 Notebook

```text
notebooks/05_vanilla_rnn_core.ipynb
notebooks/06_lstm_core.ipynb
notebooks/07_gru_core.ipynb
```

Notebook 会动态展示正式源码，运行数值梯度与官方 Cell 对齐，并在末尾提供默认关闭的
回忆自测区。

### 4. 关闭答案并回忆实现

只修改 `exercises/`，不要覆盖正式模块：

```bash
pytest exercises/tests/test_vanilla.py -k v01 -q
pytest exercises/tests/test_lstm.py -k l02 -q
pytest exercises/tests/test_gru.py -k g05 -q
```

未完成的练习会抛出带 CORE 编号的 `NotImplementedError`。`exercises/tests/` 不会被默认
`pytest` 收集，因此练习进度不会破坏正式项目状态。

### 5. 完成实验

```text
notebooks/01_vanilla_gradient_flow.ipynb
notebooks/02_lstm_memory_path.ipynb
notebooks/03_gru_gate_analysis.ipynb
notebooks/04_model_comparison.ipynb
```

## 字符语言模型

下载并校验 Tiny Shakespeare：

```bash
python scripts/download_corpus.py
```

运行快速训练：

```bash
python scripts/train.py --config configs/char_lm_debug.yaml --model vanilla
python scripts/train.py --config configs/char_lm_debug.yaml --model lstm
python scripts/train.py --config configs/char_lm_debug.yaml --model gru
```

评估某次运行：

```bash
python scripts/evaluate.py --run-dir artifacts/<run-name>
```

正式实验使用 `configs/char_lm_full.yaml` 中的三个固定随机种子。

## 延迟记忆任务

第一个位置给出待记忆 token，中间插入干扰 token，最后一个 query 位置要求恢复首 token。
默认比较 delay 5、20、50：

```bash
python scripts/train.py --config configs/delayed_recall.yaml --model vanilla
python scripts/train.py --config configs/delayed_recall.yaml --model lstm
python scripts/train.py --config configs/delayed_recall.yaml --model gru
```

## 统一接口

- 输入序列：`[batch, time, input_size]`
- 隐状态：`[batch, hidden_size]`
- 语言模型 logits：`[batch, time, vocabulary_size]`
- NumPy 梯度检查：`float64`
- 正常训练：`float32`
- 权重布局：与 PyTorch 一致的 `[gates * hidden, input]`

手写 PyTorch 模块禁止调用 `nn.RNNCell`、`nn.LSTMCell`、`nn.GRUCell` 或对应序列模块；
官方模块只出现在测试中作为判题器。

## 当前范围

本项目聚焦单层、单向的 Vanilla RNN、LSTM 和 GRU。Attention、Seq2Seq、双向网络、
Transformer 不在本项目范围，它们适合在仓库中作为后续独立项目实现。

