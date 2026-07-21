# Vanilla RNN 分级提示

正式答案位于 `vanilla_rnn/`，回忆填空位于 `exercises/vanilla_rnn/`。阅读答案后关闭
正式文件，只修改练习目录；只有卡住时才依次展开三级提示。

## CORE-V01：参数初始化

目标：创建与 PyTorch 布局一致的四个 `float64` 参数。

提示 1——数学：输入投影和循环投影应保持相近的初始方差；偏置从零开始。

提示 2——形状：`weight_ih=[H,D]`、`weight_hh=[H,H]`、两个 bias 均为 `[H]`。

提示 3——伪代码：验证维度为正；按 hidden size 构造对称均匀区间；从传入的 RNG
采样两个权重；创建两个零向量；按规定键名返回。

验证：`pytest exercises/tests/test_vanilla.py -k v01 -q`

复盘：为什么必须使用传入的 RNG？为什么梯度检查使用 float64？

## CORE-V02：单步前向

目标：根据当前输入和上一隐藏状态产生新隐藏状态，并保存反向所需的最小缓存。

提示 1——数学：两个仿射变换与两个 bias 相加后通过 tanh。

提示 2——形状：`[B,D] @ [D,H]` 与 `[B,H] @ [H,H]` 的结果均为 `[B,H]`。

提示 3——伪代码：检查 batch 和 feature；计算 pre-activation；计算新状态；缓存输入、
前一状态、新状态及参数引用。

验证：`pytest exercises/tests/test_vanilla.py -k v02 -q`

复盘：反向时是保存 pre-activation 还是保存 tanh 输出更方便？bias 如何广播？

## CORE-V03：序列前向

目标：处理 `[B,T,D]` 输入并返回所有隐藏状态。

提示 1——数学：每一步的 `h_t` 是下一步的 `h_prev`。

提示 2——形状：预分配 `[B,T,H]`；缓存列表长度必须恰好为 T。

提示 3——伪代码：令当前状态等于 h0；从左到右遍历 time；调用单步函数；写入对应
时间位置；追加缓存；更新当前状态。

验证：`pytest exercises/tests/test_vanilla.py -k v03 -q`

复盘：为什么不能在 batch 维上循环？T=1 时序列函数应与哪个结果一致？

## CORE-V05：完整 BPTT

目标：计算 `dx`、`dh0` 和四个参数梯度。

提示 1——数学：`dh_total(t) = doutputs(t) + dh_from_future`；tanh 的局部梯度可由
已缓存的输出直接得到。

提示 2——形状：`dh_total` 与 `dh_next` 为 `[B,H]`；权重梯度要沿 batch 与 time
累加，bias 梯度沿 batch 求和。

提示 3——伪代码：初始化所有梯度为零；从最后一步逆序遍历；合并隐藏梯度；通过
tanh；累加两组权重和两组 bias；计算当前输入梯度与前一状态梯度。

验证：`pytest exercises/tests/test_vanilla.py -k v05 -q`

复盘：为什么 `bias_ih` 和 `bias_hh` 的梯度相同？遗漏 `dh_next` 会通过短序列测试吗？

`CORE-V04` 是写代码前的纸笔检查点：先单独推导一个时间步的局部反向传播，并在展开图
上标出 `dh_from_output` 与 `dh_from_future`。它没有对应函数，完成推导后再进入 V05。

## CORE-V06：全局梯度裁剪

目标：按所有梯度共同的 L2 norm 缩放，且不修改输入字典。

提示 1——数学：全局平方范数是所有数组元素平方和的总和。

提示 2——形状：缩放因子是标量，对字典中的每个数组完全相同。

提示 3——伪代码：计算原 norm；若不超过阈值则复制；否则乘
`max_norm/(norm+epsilon)`；同时返回裁剪前 norm。

验证：`pytest exercises/tests/test_vanilla.py -k v06 -q`

复盘：逐元素 clip 与 global-norm clip 的方向有什么区别？为什么返回裁剪前 norm？

## CORE-V07 / V08：PyTorch Cell 与序列层

目标：只用基础张量操作复现官方 tanh RNN。

提示 1——数学：Cell 公式与 NumPy 完全相同；序列层只负责建立时间依赖。

提示 2——形状：缺省状态为 `[B,H]`，必须继承输入的 device 与 dtype。

提示 3——伪代码：Cell 完成两个线性投影并激活；Layer 校验三维输入，构造或校验
初始状态，从左到右调用 Cell，最后 stack 时间输出。

验证：`pytest exercises/tests/test_vanilla.py -k 'v07 or v08' -q`

复盘：为什么这里不能原地覆盖 autograd 仍需要的张量？官方 RNN 的最终状态为何多一维？

## CORE-V09 / V10：语言模型与生成

目标：完成 embedding → recurrence → projection，并复用状态逐 token 生成。

提示 1——数学：时间 t 的 logits 预测 t+1；生成时只从最后时间位置采样。

提示 2——形状：token 输入 `[B,T]`，logits `[B,T,V]`，新采样 token `[B]`。

提示 3——伪代码：前向依次调用三层；生成先消费完整 prefix 得到状态，再循环输入最新
token、取最后 logits、采样并拼接；保持 no-grad。

验证：`pytest exercises/tests/test_vanilla.py -k 'v09 or v10' -q`

复盘：训练跨 batch 保留状态时为什么要 detach？temperature 趋近 0 时分布如何变化？
