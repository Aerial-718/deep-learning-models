# GRU 分级提示

正式答案位于 `gru/`，回忆填空位于 `exercises/gru/`。Notebook 阅读完成后，只修改
练习目录并使用下面的独立测试。

## CORE-G01：三门参数

目标：初始化顺序为 `r, z, n` 的合并参数。

提示 1——数学：输入侧和循环侧各产生三个 H 维分块。

提示 2——形状：两个权重第一维为 `3H`，两个 bias 为 `[3H]`。

提示 3——伪代码：检查维度；用传入 RNG 初始化两个权重；创建两个零 bias；保持固定
键名和 float64。

验证：`pytest exercises/tests/test_gru.py -k g01 -q`

复盘：为什么 gate 顺序必须写进接口约定？只看总形状能发现门顺序错误吗？

## CORE-G02：reset-after 单步前向

目标：实现与 PyTorch 相同的 GRU 方程。

提示 1——数学：先分别计算输入和隐藏的三块仿射结果；`r`、`z` 使用对应两块之和；
候选 `n` 将输入候选块与 `r * hidden_candidate_block` 相加后通过 tanh；最终状态在
候选和旧状态之间插值。

提示 2——形状：输入侧与隐藏侧合并结果均为 `[B,3H]`；切分后所有项为 `[B,H]`。

提示 3——伪代码：分别算 input projection 和 hidden projection；各自切成 r/z/n；
计算 r、z；将 r 乘到隐藏候选分块；得到 n；按 update gate 混合 n 和 h_prev；缓存所有
分支需要的量。

验证：`pytest exercises/tests/test_gru.py -k g02 -q`

复盘：`W_hn(r*h)` 与 `r*(W_hn*h+b_hn)` 为什么通常不相等？z 接近 1 时会发生什么？

## CORE-G03：序列展开

目标：用显式时间循环传播唯一的隐藏状态。

提示 1——数学：结构与 Vanilla RNN 的序列外壳相同，差异全部在 Cell 内。

提示 2——形状：输出 `[B,T,H]`，最终状态 `[B,H]`，缓存长度 T。

提示 3——伪代码：从 h0 开始逐时间调用单步函数；保存每步 h 与 cache；把新 h 作为
下一步旧状态。

验证：`pytest exercises/tests/test_gru.py -k g03 -q`

复盘：为什么不同 Cell 可以共享几乎相同的 sequence wrapper？状态类型何时会破坏共享？

## CORE-G05：GRU BPTT

目标：正确合并 update 直连、候选、reset 和仿射分支的梯度。

提示 1——数学：最终插值同时向旧状态、候选和 update gate 分流；候选梯度经过 tanh 后
又分别流向输入候选块、reset gate 和隐藏候选块。

提示 2——形状：三块 pre-activation 梯度分别拼回 `[B,3H]`；`dh_prev` 至少接收插值
直连与隐藏仿射两类贡献。

提示 3——伪代码：逆序合并 dh；先反传最终插值；再反传候选 tanh；从 reset 乘法拆成
dr 与 hidden-candidate 梯度；反传 r/z sigmoid；拼接输入侧和隐藏侧梯度；通过两个权重
回传并累加参数梯度。

验证：`pytest exercises/tests/test_gru.py -k g05 -q`

复盘：最容易遗漏的 h_prev 直连来自哪里？为何输入侧和隐藏侧候选梯度不能简单设为相同？

`CORE-G04` 是纸笔检查点：展开最终插值、候选 tanh、reset 乘法和两侧仿射，给每条指向
`h_prev` 的路径编号。确认路径完整后再进入 G05。

## CORE-G06 / G07：PyTorch GRU

目标：用基础张量操作复现官方 reset-after Cell 和单层序列。

提示 1——数学：保持输入投影与隐藏投影分开，直到三块切分完成。

提示 2——形状：缺省状态使用 `x.new_zeros(B,H)`；输出按 time 维 stack。

提示 3——伪代码：Cell 分别调用两个线性表达式；切成三块；按规定顺序构造 r/z/n/h；
Layer 校验输入与状态并从左到右调用 Cell。

验证：`pytest exercises/tests/test_gru.py -k 'g06 or g07' -q`

复盘：若先把两侧投影整体相加，为何候选分支会失去所需结构？如何用单元测试发现？

## CORE-G08 / G09：语言模型

目标：完成 embedding、手写 GRU、输出投影与自回归生成。

提示 1——数学：接口与 Vanilla RNN 相同，但状态更新方式不同。

提示 2——形状：前向 `[B,T] -> [B,T,V]`；生成每步只输入一个时间位置。

提示 3——伪代码：前向三段连接；生成先消费 prefix，再循环最新 token、状态与最后
logits；使用公共采样器并拼接 token。

验证：`pytest exercises/tests/test_gru.py -k 'g08 or g09' -q`

复盘：为何三个模型可以共享训练循环？将模型类型分支写进训练循环有什么坏处？
