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

## 8. 用户上下文和场景信息在哪里体现

先区分两层：

```text
论文图里明确画出来的：用户历史行为上下文 H_u
工业实现里通常会加入的：用户画像、请求场景、地理位置、时间、设备等 context feature
```

在这张图里，用户上下文主要体现在左侧 encoder：

```text
User Behavior Sequences H_u
-> OneRec Encoder
-> hidden states H
```

然后 `H` 通过右侧 decoder 的 cross-attention 进入生成过程：

```text
decoder hidden state 作为 query
encoder output H 作为 key/value
```

所以 decoder 每生成一个 token，比如 `<a_9>`、`<b_7>`、`<c_1>`，都不是只看已经生成的 token，而是同时通过 cross-attention 读取用户历史上下文 `H`。这就是图中从 encoder 到 decoder 的那条 `key/value` 箭头。

可以写成概率形式：

```text
P(next token | already generated tokens, H)
```

其中 `H` 就是用户历史行为序列被 encoder 编码后的上下文表示。

如果加上工业里的场景信息，可以抽象成：

```text
H = Encoder(user_behavior_tokens, user_profile_tokens, scene_context_tokens)
```

常见做法是把这些信息变成额外 token 或 feature embedding，例如：

```text
[USER_AGE] [USER_GENDER] [CITY] [POI] [HOUR] [DEVICE] [SCENE]
SEP <a_6> <b_1> <c_5>
SEP <a_2> <b_1> <c_7>
```

或者把场景特征作为 dense feature embedding 加到每个行为 token 或 encoder 的全局 context token 上。

需要严谨一点：这张 OneRec 论文图本身没有专门画出 `city_id`、`device`、`tab`、`request scene` 这些字段，所以不能说图里某个固定 token 就是场景信息。能确定的是：

```text
用户历史行为 H_u -> encoder output H -> decoder cross-attention
```

工程上，场景信息通常就是并入 encoder 输入或 context processor，最后也体现在 `H` 里。

## 9. SID 和 embedding 在输入中如何结合

一句话：

```text
SID 是离散 token id；embedding 是这个 token id 查表后得到的连续向量。
```

也就是说，实际喂给 Transformer 的不是字符串 `<a_9>`，而是：

```text
<a_9> -> token_id -> embedding lookup -> dense vector
```

### 9.1 离线阶段：先把 item 变成 SID

视频先经过 item tokenizer：

```text
video 多模态/协同 embedding
-> RQ-Kmeans / RQ-VAE / balanced quantization
-> SID = <a_9> <b_7> <c_1>
```

这里的多模态/协同 embedding 主要用于 **生成 SID**。生成完后，训练生成模型时通常不直接把原始 item dense embedding 塞给 decoder 输出层，而是把 SID 当作离散 token 来建模。

### 9.2 在线/训练输入阶段：SID 通过 embedding table 变成向量

假设历史视频是：

```text
video_1 -> <a_6> <b_1> <c_5>
video_2 -> <a_2> <b_1> <c_7>
```

encoder 侧输入 token 可以是：

```text
SEP <a_6> <b_1> <c_5> SEP <a_2> <b_1> <c_7>
```

每个 token 都会查 embedding table：

```text
E[SEP], E[a_6], E[b_1], E[c_5], E[SEP], E[a_2], E[b_1], E[c_7]
```

然后再加上位置、层级、类型等 embedding：

```text
x_t =
  token_embedding(token_t)
  + position_embedding(pos_t)
  + level_embedding(level_t)
  + type_embedding(type_t)
```

其中：

| embedding 类型              | 作用                                                         |
|-----------------------------|--------------------------------------------------------------|
| `token_embedding`           | 表示当前 token 是 `<a_6>`、`<b_1>`、`SEP` 还是 `BOS`          |
| `position_embedding`        | 表示 token 在序列中的位置                                    |
| `level_embedding`           | 区分 token 属于第 1/2/3 层 SID，例如 `a/b/c` 层              |
| `type_embedding`            | 区分历史 item、目标 item、场景 token、用户画像 token 等       |
| `behavior_embedding`        | 可选，表示 watch、like、follow、long_view、dislike 等行为类型 |
| `time_embedding`            | 可选，表示行为发生时间、时间间隔、小时、星期等               |

实际工程里不一定全都相加，也可能是 concat 后过一层 projection：

```text
x_t = Projection(concat(token_emb, pos_emb, level_emb, behavior_emb, time_emb))
```

核心思想不变：**SID 先离散化，再 embedding lookup 成连续向量，最后进入 Transformer。**

### 9.3 场景 token 怎么融合

场景信息通常也会有自己的 embedding，但它一般不是 item SID：

```text
[CITY_110000] -> city_embedding
[HOUR_22]     -> hour_embedding
[DEVICE_IOS]  -> device_embedding
```

可以把它们作为 encoder 前缀 token：

```text
[CITY_110000] [HOUR_22] [DEVICE_IOS]
SEP <a_6> <b_1> <c_5>
SEP <a_2> <b_1> <c_7>
```

也可以把场景 embedding 加到每个行为 token 上：

```text
x_t = sid_token_embedding + behavior_embedding + scene_embedding + position_embedding
```

这两种做法的区别是：

| 做法                         | 特点                                                         |
|------------------------------|--------------------------------------------------------------|
| 场景作为前缀 token           | 更接近 LLM token 化方式，结构统一，便于 attention 读取        |
| 场景 embedding 加到每个 token | 更像传统推荐特征融合，能让每个行为 token 都携带请求场景       |
| concat 后 projection          | 适合特征很多、维度不一致、需要保留多路特征来源的场景         |

### 9.4 decoder 输出层怎么用 SID embedding

decoder 每一步输出 hidden state：

```text
h_t
```

然后通过一个 softmax 预测下一个 SID token：

```text
logits_t = h_t @ W_vocab
P(token_t | prefix, H) = softmax(logits_t)
```

这里的 `W_vocab` 对应 SID token 词表，比如：

```text
<a_0> ... <a_K>
<b_0> ... <b_K>
<c_0> ... <c_K>
BOS / SEP / M / 其他 special tokens
```

有些实现会做 input embedding 和 output embedding tying：

```text
W_vocab = token_embedding_table^T
```

也就是输入查表用的 embedding 和输出分类用的 embedding 共享参数。是否共享是实现选择，不是这张图里能确定的事实。

### 9.5 和传统推荐 embedding 的区别

传统推荐里常见的是：

```text
item_id -> item_embedding
user_id -> user_embedding
特征 embedding 拼接 -> MLP / DIN / SIM / Transformer
```

OneRec / 生成式推荐更像：

```text
item_id -> SID tokens -> token embeddings -> Transformer
Transformer hidden state -> softmax 生成下一个 SID token
SID tokens -> 查表还原 item_id
```

所以 SID 和 embedding 不是二选一，而是上下游关系：

```text
SID 负责把 item 离散化成可生成 token
embedding 负责把 SID token 变成模型可计算的向量
```

## 10. decoder 如何逐渐生成

decoder 的“逐渐生成”本质是 **autoregressive decoding**：一次只预测下一个 token，然后把这个 token 拼回当前前缀，再继续预测下一个 token。

### 10.1 训练阶段：teacher forcing

训练时目标 session 是已知的，比如：

```text
video_1 = <a_9> <b_7> <c_1> M
video_2 = <a_4> <b_5> <c_4> M
```

decoder input 会右移一位，用真实 token 前缀喂给模型：

```text
input : BOS   <a_9> <b_7> <c_1>  BOS   <a_4> <b_5> <c_4>
target: <a_9> <b_7> <c_1> M      <a_4> <b_5> <c_4> M
```

每个位置都做 cross-entropy：

```text
L_NTP = - log P(target_token | previous_target_tokens, H)
```

这里的 `H` 是 encoder 输出的用户上下文。

### 10.2 推理阶段：自回归生成

推理时没有真实 target，只能一步一步生成：

```text
step 1: 输入 BOS，预测第一个 token
        P(<a> | BOS, H)

step 2: 把采样/beam 选中的 <a_9> 拼回去，预测第二个 token
        P(<b> | BOS, <a_9>, H)

step 3: 继续预测第三个 token
        P(<c> | BOS, <a_9>, <b_7>, H)

step 4: 预测 M，表示当前 item 结束
        P(M | BOS, <a_9>, <b_7>, <c_1>, H)

step 5: 再放一个 BOS，开始生成下一个 item
        P(<a> | previous item tokens, BOS, H)
```

所以“一点点生成”不是模型一次吐出完整 session，而是每一步输出一个 token 的概率分布：

```text
softmax(logits_t) -> 选择 token_t -> 拼到 prefix -> 进入下一步
```

生成完整 session 可以写成链式分解：

```text
P(S | H)
= product_t P(token_t | token_<t, H)
```

如果一个 session 里有 `m` 个视频，每个视频有 `L` 层 semantic ID，再加一个结束 token `M`，那么总生成步数大致是：

```text
m * (L + 1)
```

图中每个视频是 3 个 semantic token 加一个 `M`，所以每个视频需要 4 个 target token。

### 10.3 causal self-attention 和 cross-attention 各自做什么

decoder 里有两种 attention：

| 模块                              | 作用                                                         |
|-----------------------------------|--------------------------------------------------------------|
| `Causal Self-Attention`            | 只看已经生成的左侧 token，保证不能偷看未来 token             |
| `Fully Visible Cross-Attention`    | 读取 encoder 输出 `H`，让每一步生成都条件化在用户上下文上     |

也就是说：

```text
causal self-attention 决定“已经生成了什么”
cross-attention 决定“基于这个用户/场景应该生成什么”
```

### 10.4 beam search 在这里做什么

真实 serving 时一般不会只贪心选概率最大的 token，而会用 beam search 保留多条候选前缀。

例如 beam size = 3：

```text
BOS
-> 保留 top3 个 <a> 前缀
-> 每个前缀继续扩展 top token
-> 保留累计 log probability 最高的 top3 条
-> 直到生成完整 session
```

这样可以得到多个候选 session，再结合 reward model、DPO/IPA 或业务规则选择更优结果。

## 11. 一句话总结

`SEP` 是 encoder 侧用户历史行为的分隔符，用来标记不同历史视频边界；`BOS` 是 decoder 侧每个目标视频 semantic ID 的开始符；`M` 是图中放在每个目标视频 semantic ID 之后的特殊边界/结束 token，功能上表示当前 item 的 semantic ID 生成完成，但论文正文没有明确给出 `M` 的全称。

用户上下文在图里不是一个单独模块，而是 `H_u` 经过 encoder 得到的 `H`；decoder 通过 cross-attention 把 `H` 作为 key/value 读取。场景信息如果加入，工程上通常会作为额外 context token 或 feature embedding 并入 encoder/context processor。SID 本身是离散 token id，实际输入时还要查 embedding table 变成连续向量，再加位置、层级、类型、行为、时间、场景等 embedding。decoder 的逐渐生成则是标准自回归：`BOS -> a -> b -> c -> M`，生成完一个 item 后再进入下一个 item。

## 12. 参考资料

1. OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment  
   https://arxiv.org/abs/2502.18965
2. arXiv source 中 `Session-wise List Generation` 相关公式：论文明确写到训练时在 codes 开头添加 start token `s_[BOS]`，并使用 cross-entropy 做 next-token prediction。
