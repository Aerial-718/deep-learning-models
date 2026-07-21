# GRU

GRU 将长期状态与暴露状态合并为单个 `h`。本项目采用 PyTorch 兼容的 reset-after 公式。
根目录文件是正式答案，回忆练习位于 `exercises/gru/`。

## 学习顺序

1. 阅读正式 `numpy_impl.py`，重点观察输入投影与隐藏投影为何保持分开。
2. 从头运行 `notebooks/07_gru_core.ipynb`。
3. 关闭答案，在 `exercises/gru/` 完成 CORE-G01—G09。
4. 对齐官方 `nn.GRUCell`，再分析 reset/update gate。
5. 完成字符模型和统一比较实验。

## 阶段命令

```bash
pytest tests/gru -q
pytest exercises/tests/test_gru.py -k g01 -q
```

## 完成后的复盘

- update gate 接近 1 时，新状态更接近哪一项？
- reset gate 只影响候选分支，还是也影响 update 分支？
- GRU 参数更少是否必然意味着训练吞吐量按同比例提升？
