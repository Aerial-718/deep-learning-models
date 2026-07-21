# LSTM 分级提示

正式答案位于 `lstm/`，回忆填空位于 `exercises/lstm/`。Notebook 阅读完成后，只修改
练习目录并使用下面的独立测试。

## CORE-L01：四门参数

目标：初始化 gate 顺序固定为 `i, f, g, o` 的参数。

提示 1——数学：一次仿射计算同时产生四个长度 H 的向量。

提示 2——形状：两个权重第一维为 `4H`，两个 bias 为 `[4H]`。

提示 3——伪代码：验证维度；创建两个四门权重和两个零 bias；不要在初始化时拆成十六个
独立数组。

验证：`pytest exercises/tests/test_lstm.py -k l01 -q`

复盘：合并四门计算有什么工程收益？forget bias 实验为何应在初始化之后显式设置？

## CORE-L02：单步前向

目标：实现四门、cell state 和 hidden state。

提示 1——数学：`i/f/o` 通过 sigmoid，候选 `g` 通过 tanh；新 cell 是保留项与写入项
之和，新 hidden 是经过 output gate 的 cell 视图。

提示 2——形状：总 gate `[B,4H]`，按最后一维切成四个 `[B,H]`。

提示 3——伪代码：完成两次仿射并相加；按固定顺序切分；分别激活；更新 c；由 c 计算 h；
缓存输入、旧状态、所有 gate 和新 cell。

验证：`pytest exercises/tests/test_lstm.py -k l02 -q`

复盘：为什么缓存 gate 输出比缓存 gate pre-activation 更方便？c 是否必须限制在 [-1,1]？

## CORE-L03：序列展开

目标：在每一步同时传播 h 和 c。

提示 1——数学：下一步状态是 `(h_t,c_t)`，两者都不能重置。

提示 2——形状：对外 outputs 只保存所有 h；最终状态返回 h 与 c。

提示 3——伪代码：拆出初始二元状态；从左到右调用单步；写入 h；更新两条状态；缓存。

验证：`pytest exercises/tests/test_lstm.py -k l03 -q`

复盘：语言模型为何通常只投影 h？若只返回最终 c，会丢失什么？

## CORE-L05：LSTM BPTT

目标：同时沿 hidden 与 cell 路径反传。

提示 1——数学：当前 `dc_total` 同时接收未来 cell 梯度，以及由 `h_t=o*tanh(c_t)`
传来的梯度；旧 cell、forget gate、input gate、candidate 都从它分支。

提示 2——形状：四个 gate 的 pre-activation 梯度重新拼接为 `[B,4H]`；所有参数梯度
沿 batch 和 time 累加。

提示 3——伪代码：逆序遍历；合并 dh；先经过 hidden 公式得到 do 与新增 dc；合并 dc；
分别得到 dc_prev、df、di、dg；通过各自激活导数；拼接；完成两个仿射层的反向。

验证：`pytest exercises/tests/test_lstm.py -k l05 -q`

复盘：若忘记 `dc_next`，短序列可能仍表现正常吗？哪几个梯度共同依赖 `c_prev`？

`CORE-L04` 是纸笔检查点：画出单步计算图，分别标记 `dh_next`、`dc_next` 以及当前输出
梯度的入口。确认每条分支都能回到旧状态后再进入 L05。

## CORE-L06 / L07：PyTorch LSTM

目标：用基础张量操作复现官方 Cell，并建立 batch-first 序列层。

提示 1——数学：公式与 NumPy 完全相同，autograd 只替代手写反向。

提示 2——形状：缺省 h/c 都由 `x.new_zeros(B,H)` 创建；官方最终状态通常表示为
`[layers,B,H]`，手写单层接口返回 `[B,H]`。

提示 3——伪代码：Cell 合并仿射、切分、激活、更新；Layer 验证或创建二元初态，按时间
循环 Cell，stack h 并返回最终二元状态。

验证：`pytest exercises/tests/test_lstm.py -k 'l06 or l07' -q`

复盘：为什么测试必须复制完全相同的权重？两组 bias 能否在数学上合并？

## CORE-L08 / L09：语言模型

目标：连接 embedding、手写 LSTM 和输出投影，并保持双状态生成。

提示 1——数学：模型输出只由 h 投影，但下一步需要 h 与 c。

提示 2——形状：logits `[B,T,V]`；生成时采样 token `[B]` 并恢复时间维。

提示 3——伪代码：前向三段连接；生成先消费 prefix，再循环输入最新 token；每次完整
保存新的 `(h,c)`，不要只保存 h。

验证：`pytest exercises/tests/test_lstm.py -k 'l08 or l09' -q`

复盘：stateful training 中 h 和 c 是否都要 detach？丢弃 c 会出现什么行为？
