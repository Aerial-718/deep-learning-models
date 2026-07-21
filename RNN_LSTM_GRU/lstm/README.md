# LSTM

本子项目关注 LSTM 中两条状态路径：隐藏状态 `h` 负责暴露信息，cell state `c` 提供更直接
的长期梯度通道。根目录文件是正式答案，回忆练习位于 `exercises/lstm/`。

## 学习顺序

1. 完成 Vanilla RNN 后，阅读正式 `numpy_impl.py` 的四门与双路径 BPTT。
2. 从头运行 `notebooks/06_lstm_core.ipynb`。
3. 关闭答案，在 `exercises/lstm/` 完成 CORE-L01—L09。
4. 对比 forget bias 为 0、1、2 的梯度曲线。
5. 完成延迟记忆与字符模型实验。

## 阶段命令

```bash
pytest tests/lstm -q
pytest exercises/tests/test_lstm.py -k l01 -q
```

## 完成后的复盘

- 哪条路径允许梯度不经过 tanh 的导数？
- forget gate 接近 1 是否保证模型一定记得有用信息？
- 为什么 output gate 影响 `h_t`，却不直接决定 `c_t` 是否保留？
