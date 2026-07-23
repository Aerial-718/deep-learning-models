# 经典 CNN：MNIST 训练与推理

这是一个完整、可运行的 PyTorch 图像分类项目。模型采用 LeNet 风格的经典
CNN：卷积负责提取局部特征，平均池化逐步降低空间分辨率，最后由全连接层完成
0–9 手写数字分类。

项目包含 MNIST 自动下载、训练/验证划分、最佳模型保存、断点续训、独立测试集
评估、单图及批量推理、学习曲线和混淆矩阵。

## 1. 环境安装

建议使用 Python 3.10–3.13。下面的命令会在项目目录创建隔离环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果只使用 CPU，也可以先从 PyTorch 的 CPU 软件源安装 `torch` 和
`torchvision`，再安装其余依赖：

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install matplotlib Pillow pytest
```

## 2. 训练

使用默认配置训练 10 轮：

```bash
python -m cnn.train --data-dir data --output-dir runs/lenet
```

常用参数：

```bash
python -m cnn.train \
  --data-dir data \
  --output-dir runs/lenet \
  --epochs 10 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --seed 42 \
  --num-workers 0 \
  --device auto
```

`--device auto` 会依次选择 CUDA、Apple MPS 或 CPU。MNIST 首次运行会自动
下载；如果数据已准备好，可传入 `--no-download`。

输出目录包括：

- `best.pt`：验证准确率最高的检查点；
- `last.pt`：最近一轮检查点，可用于续训；
- `config.json`：本次训练配置；
- `history.csv` 和 `metrics.json`：逐轮指标及最终测试结果；
- `learning_curves.png`：训练/验证损失和准确率曲线；
- `confusion_matrix.png`：最佳模型在测试集上的混淆矩阵。

断点续训时，`--epochs` 表示训练完成后的总轮数：

```bash
python -m cnn.train \
  --data-dir data \
  --output-dir runs/lenet \
  --epochs 15 \
  --resume runs/lenet/last.pt
```

## 3. 独立评估

```bash
python -m cnn.evaluate \
  --checkpoint runs/lenet/best.pt \
  --data-dir data \
  --output runs/lenet/evaluation.json \
  --confusion-matrix runs/lenet/evaluation_confusion.png
```

## 4. 图片推理

MNIST 是“黑底白字”。对于同样风格的单张图片：

```bash
python -m cnn.predict \
  --checkpoint runs/lenet/best.pt \
  --image samples/digit.png
```

普通手写图片经常是“白底黑字”，此时增加 `--invert`：

```bash
python -m cnn.predict \
  --checkpoint runs/lenet/best.pt \
  --image samples/digit.png \
  --invert
```

批量识别一个目录：

```bash
python -m cnn.predict \
  --checkpoint runs/lenet/best.pt \
  --input-dir samples \
  --invert \
  --batch-size 256
```

批量模式会将每张图片的预测类别、置信度和十类概率写入
`samples/predictions.json`。可通过 `--output` 指定其他路径。

输入图片会被转为灰度并缩放到 28×28。模型适合已经裁剪、居中的单个数字，
不包含多数字检测、文本行切割或通用 OCR。

## 5. 运行测试

测试使用本地合成数据，不需要下载 MNIST：

```bash
python -m pytest
```

测试覆盖模型形状、参数更新、确定性数据划分、图片预处理、检查点保存/加载，
以及训练到推理的完整冒烟链路。

GitHub Actions 会在每次推送和 Pull Request 时，使用 Python 3.13 与 CPU 版
PyTorch 自动运行同一组离线测试。

## 项目结构

```text
.github/workflows/
  tests.yml       # GitHub Actions 自动测试
cnn/
  model.py       # LeNet 模型
  data.py        # MNIST 加载与图片预处理
  engine.py      # 训练和评估循环
  utils.py       # 设备、检查点、图表等工具
  train.py       # 训练 CLI
  evaluate.py    # 评估 CLI
  predict.py     # 推理 CLI
tests/           # 离线测试
pyproject.toml    # Python 项目元数据
```
