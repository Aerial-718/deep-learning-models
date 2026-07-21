# Vanilla RNN

目标是理解最简单的循环计算图，并验证 BPTT 中“来自当前输出的梯度”和“来自未来时间步
的梯度”为何必须相加。根目录文件是正式答案，回忆练习位于 `exercises/vanilla_rnn/`。

## 学习顺序

1. 阅读 `numpy_impl.py` 的单步、序列和 BPTT。
2. 从头运行 `notebooks/05_vanilla_rnn_core.ipynb` 并回答复盘问题。
3. 关闭答案，在 `exercises/vanilla_rnn/` 完成 CORE-V01—V10。
4. 使用三级提示和独立测试定位错误。
5. 进入梯度流与字符语言模型实验。

不要把 NumPy 时间循环向量化掉：batch 维可以向量化，time 维应显式展开。

## 阶段命令

```bash
pytest tests/vanilla_rnn -q
pytest exercises/tests/test_vanilla.py -k v01 -q
```

## 完成后的复盘

- 如果所有 recurrent eigenvalue 的模都小于 1，长距离梯度通常会发生什么？
- 即使使用 tanh，为什么仍然可能梯度爆炸？
- global-norm clipping 修复的是优化稳定性，还是长期记忆能力？
