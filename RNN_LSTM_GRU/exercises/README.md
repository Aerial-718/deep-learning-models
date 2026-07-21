# Active-recall exercises

正式答案位于仓库根目录的 `vanilla_rnn/`、`lstm/`、`gru/`。本目录保存同接口的空白版本，
用于阅读答案后关闭参考资料，凭记忆重新实现。

推荐流程：

1. 阅读对应正式模块和 `notebooks/05`—`07`。
2. 关闭参考代码，只打开本目录的文件。
3. 按 CORE 编号填写一个函数。
4. 显式运行 `exercises/tests/` 中对应测试；这些测试不会被默认 `pytest` 收集。

示例：

```bash
pytest exercises/tests/test_vanilla.py -k v01 -q
pytest exercises/tests/test_lstm.py -k l02 -q
pytest exercises/tests/test_gru.py -k g05 -q
```

