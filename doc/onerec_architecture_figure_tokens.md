# OneRec 架构图中的 SEP、BOS、M 解释

> 说明：本文基于 OneRec 论文《OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment》及其 arXiv 源码整理。`BOS` 在论文正文中有明确公式说明；`SEP` 和图中的 `M` 主要依据图中位置和训练序列做解释。论文没有给出 `M` 的英文全称，因此这里不会强行编一个全称。

## 1. 这张图整体在表达什么

这张图分成上下两部分：

| 区域                         | 作用                                                         |
|------------------------------|--------------------------------------------------------------|
| `(a) The Architecture`        | OneRec 的 encoder-decoder 生成式推荐模型结构                 |
| `(b) Iterative Alignment`     | 用 reward model 选择 chosen/rejected session，再做 DPO 对齐   |

用户历史行为序列记为：

```text
H_u = 用户历史正反馈 / 有效观看 / 互动过的视频序列
```

模型输出是一个 session：

```text
S = 一次推荐请求返回的一组视频，通常可以理解成 5 到 10 个视频
```

每个视频不是直接用原始 `item_id` 表示，而是被 tokenizer 变成多层 semantic ID。例如图里的：

```text
<a_9> <b_7> <c_1>
```

可以理解成某个视频的三层语义 token：

```text
video_i -> (sid_layer_1, sid_layer_2, sid_layer_3)
```

其中 `a / b / c` 对应不同层级或不同 codebook 的 semantic token。一般可以粗略理解为：

```text
a 层更粗粒度
b 层中等粒度
c 层更细粒度
```

但具体语义粒度由 tokenizer 学到，不能简单等同于人工类目、作者或 tag。

## 2. SEP 是什么

`SEP` 通常表示 **separator token**，也就是分隔符。

在图里，`SEP` 出现在左侧 encoder 的用户行为序列中：

```text
SEP <a_6> <b_1> <c_5> SEP <a_2> <b_1> <c_7>
```

它的作用是把不同历史视频的 semantic ID 片段分开，让 encoder 知道：

```text
<a_6> <b_1> <c_5> 是一个历史视频
<a_2> <b_1> <c_7> 是另一个历史视频
```

如果没有 `SEP`，模型看到的是一长串 token：

```text
<a_6> <b_1> <c_5> <a_2> <b_1> <c_7>
```

它仍然可能通过位置学习到三三一组，但边界信息更弱。加 `SEP` 后，历史行为的 item 边界更明确。

从推荐工程角度看，`SEP` 的意义类似：

```text
用户历史 item 序列中的 item boundary marker
```

也就是“一个历史视频结束，下一个历史视频开始”。

## 3. BOS 是什么

`BOS` 表示 **beginning of sequence**，即序列开始符。

论文正文明确写到：训练时会在 codes 的开头添加 start token `s_[BOS]` 来构造 decoder input。

图里 decoder 输入是：

```text
BOS <a_9> <b_7> <c_1> ... BOS <a_4> <b_5> <c_4>
```

含义是：每生成一个目标视频的 semantic ID 前，都先放一个 `BOS`，告诉 decoder：

```text
现在开始生成一个新视频的 semantic ID。
```

用 teacher forcing 来看，decoder input 和 target 大致是错位一格：

```text
decoder input:  BOS, <a_9>, <b_7>, <c_1>
target output:  <a_9>, <b_7>, <c_1>, M
```

因此 `BOS` 不是一个真实视频 token，而是生成起点提示。

## 4. M 是什么

先说严谨结论：**论文正文没有给出图中 `M` 的英文全称**。所以不能武断说它一定叫 `mask`、`marker` 或 `merge`。

但根据图里的位置，`M` 的功能比较清楚：它出现在每个目标视频的 semantic ID 之后：

```text
<a_9> <b_7> <c_1> M
<a_4> <b_5> <c_4> M
```

结合 decoder input / target 的错位关系，可以把它理解成：

```text
一个视频 semantic ID 生成结束后的特殊边界 token
```

也就是功能上类似：

```text
end-of-item / item boundary / code completion marker
```

如果一个视频用三层 semantic ID 表示：

```text
video_i = <a_i> <b_i> <c_i>
```

那么训练目标可以理解成：

```text
BOS -> 预测 <a_i>
<a_i> -> 预测 <b_i>
<b_i> -> 预测 <c_i>
<c_i> -> 预测 M
```

这里的 `M` 让模型学到“这个 item 的 semantic ID 已经结束”。否则模型只知道预测完 `<c_i>`，但不知道当前位置应该结束当前 item，还是继续生成别的层级 token。

## 5. 不要把 M 和这些符号混淆

图里和论文里还有几个相似符号，容易混：

| 符号                         | 含义                                                         |
|------------------------------|--------------------------------------------------------------|
| `M`，图中灰色 token           | 目标 item semantic ID 之后的特殊边界 token，正文未给全称      |
| `mathcal{M}`                 | OneRec session-wise generation model                         |
| `M_t / M_{t+1}`              | 第 `t` 轮 / 下一轮迭代的 OneRec 模型                          |
| `m`                          | session 内视频数量                                           |
| `N_MoE`                      | MoE expert 总数                                               |

所以图中灰色 token `M` 不应该理解成 MoE expert 数量，也不是模型本身。

## 6. 为什么 encoder 用 SEP，decoder 用 BOS/M

因为两边处理的任务不同：

| 模块                 | 输入/输出对象                  | 特殊 token 作用                                         |
|----------------------|--------------------------------|--------------------------------------------------------|
| Encoder              | 用户历史行为序列 `H_u`         | `SEP` 分隔不同历史 item                                |
| Decoder input        | 目标 session 的前缀            | `BOS` 表示开始生成一个目标 item                        |
| Decoder target       | next-token prediction 标签     | `M` 表示当前目标 item 的 semantic ID 生成结束           |

左侧 encoder 是“读历史”，需要把历史视频切开；右侧 decoder 是“生成推荐 session”，需要知道每个 item 从哪里开始、在哪里结束。

## 7. 和 NTP loss 的关系

图右上角的 `L_NTP` 是 **next-token prediction loss**。

论文中的训练目标是对 target session 的 semantic IDs 做 cross-entropy next-token prediction。直观地说，就是让模型在每个位置预测下一个 token：

```text
已知 BOS，预测 <a_9>
已知 BOS <a_9>，预测 <b_7>
已知 BOS <a_9> <b_7>，预测 <c_1>
已知 BOS <a_9> <b_7> <c_1>，预测 M
```

如果 session 里有多个视频，就重复这个过程。

## 8. 一句话总结

`SEP` 是 encoder 侧用户历史行为的分隔符，用来标记不同历史视频边界；`BOS` 是 decoder 侧每个目标视频 semantic ID 的开始符；`M` 是图中放在每个目标视频 semantic ID 之后的特殊边界/结束 token，功能上表示当前 item 的 semantic ID 生成完成，但论文正文没有明确给出 `M` 的全称。

## 9. 参考资料

1. OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment  
   https://arxiv.org/abs/2502.18965
2. arXiv source 中 `Session-wise List Generation` 相关公式：论文明确写到训练时在 codes 开头添加 start token `s_[BOS]`，并使用 cross-entropy 做 next-token prediction。

