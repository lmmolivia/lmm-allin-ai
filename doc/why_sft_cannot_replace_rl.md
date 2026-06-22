# 为什么不能把多目标 reward 直接塞进 SFT，从而不需要 RL

## 1. 先给结论

可以把点击、时长、转化、负反馈、多样性、生态目标等融合成一个 reward，然后用它筛选或加权 SFT 样本。这在工程上很常见，例如：

```text
只保留高 reward 样本做 SFT
或者对高 reward 样本加大 loss weight
```

但这仍然不能完全替代 RL。

核心原因是：

```text
SFT 学的是日志里“已经发生的行为”
RL 优化的是模型“自己可能采取的行为”的期望收益
```

SFT 可以模仿高质量历史样本，但它不能充分回答：

```text
如果模型换一个推荐列表，reward 会不会更高？
```

RL/偏好优化的价值就在这里：它让模型在候选输出空间里比较、探索和优化，而不只是模仿日志。

## 2. SFT 的训练目标是什么

SFT 通常是最大似然训练：

```text
maximize log P(y | x)
```

在推荐里：

```text
x = 用户历史 + 场景上下文
y = 日志里的目标 item / item list
```

如果把 reward 加进去，常见会变成加权 SFT：

```text
maximize w(x, y) * log P(y | x)
```

其中：

```text
w(x, y) = f(点击, 时长, 转化, 负反馈, 多样性, 生态目标)
```

这确实能让模型更关注高价值样本。

但是注意：它仍然是在提高日志中这个 `y` 的概率。

也就是说，SFT 的基本动作是：

```text
看到一个样本，就把这个样本里的答案概率拉高
```

reward 只是在决定“拉高多少”。

## 3. RL 的训练目标是什么

RL 优化的是模型策略下的期望 reward：

```text
maximize E_{y ~ pi_theta(.|x)} [ R(x, y) ]
```

它关心的是：

```text
模型自己生成 y 之后，这个 y 的 reward 高不高
```

因此 RL 可以对模型当前生成的多个候选结果做比较：

```text
同一个用户上下文 x
模型生成 y1, y2, y3, ..., yK
reward model / 用户反馈 给每个 y 打分
提高高 reward 输出概率
降低低 reward 输出概率
```

这和 SFT 的关键区别是：

```text
SFT：模仿历史 y
RL：优化当前策略生成的 y
```

## 4. 为什么“reward 加权 SFT”不够

### 4.1 SFT 只能学习日志里出现过的答案

假设同一个用户上下文下，日志里曝光了 item A：

```text
x -> A
```

A 的 reward 很高，于是 SFT 会提高 `P(A|x)`。

但如果模型其实可以推荐 B，而 B 的长期收益更高，只是历史系统没有曝光 B，那么 SFT 不会知道。

这就是推荐日志的 **exposure bias**：

```text
日志只记录旧系统展示过的东西
不记录旧系统没展示、但可能更好的东西
```

RL 或偏好优化可以通过当前模型采样、候选生成、reward 打分，探索日志之外的候选。

### 4.2 Reward 不能自动告诉 SFT “应该生成哪个替代答案”

reward 是一个分数：

```text
R(x, y) = 0.82
```

但 SFT 需要一个明确 label：

```text
y = 哪个 item / 哪个列表
```

如果日志里只有一个正样本，SFT 只能把这个正样本概率拉高。它不知道还有哪些可行输出，也不知道这些输出之间的相对优劣。

RL 的做法是：

```text
先让模型生成多个 y
再用 reward 比较这些 y
```

这就把“分数”变成了可优化信号。

### 4.3 SFT 容易继承旧系统偏差

推荐日志来自旧系统：

```text
召回 -> 粗排 -> 精排 -> 重排 -> 曝光
```

所以 SFT 学到的是：

```text
旧系统曝光分布中的高质量样本
```

不是全 item 空间里的最优策略。

即使你用多目标 reward 筛选高质量样本，SFT 仍然在旧系统可达的候选空间里学习。

这也是 OneRec Technical Report 里强调 RL / preference alignment 的原因：预训练和 SFT 只拟合历史曝光分布，容易被传统 pipeline 的上限限制。

### 4.4 SFT 缺少“反事实比较”

推荐里真正想知道的是：

```text
如果同一个用户看到列表 A 和列表 B，哪个更好？
```

但日志通常只包含用户实际看到的那个列表：

```text
用户看到了 A，反馈是 10 秒观看
```

你不知道：

```text
如果给他 B，会不会看 30 秒？
```

SFT 只能模仿 A。

RL / DPO / GRPO 这类方法会显式构造比较：

```text
A better than B
或者
reward(A) > reward(B)
```

这能更直接优化选择行为。

### 4.5 多样性、生态目标通常是 list-level / long-term 的

点击、时长、转化可以是单 item 指标，但多样性、供给生态、用户疲劳、长期留存常常是列表级或长期指标：

```text
R(user, [item1, item2, ..., itemK])
```

SFT 的 token-level cross entropy 很难天然表达：

- 列表内部重复惩罚。
- 类目分散。
- 作者/商家覆盖。
- 长短期兴趣平衡。
- 用户疲劳。
- 长期满意度。

可以把这些指标融成一个样本权重，但样本权重还是只能告诉模型：

```text
这个历史列表整体不错，多学一点
```

它不直接告诉模型：

```text
把第 3 个 item 换成另一个类目会更好
或者
这个列表缺少多样性，需要降低概率
```

RL 更适合把整个列表作为 action，然后用 list-level reward 训练。

### 4.6 加权 SFT 会丢掉低分样本里的信息

如果你只保留高 reward 样本，低 reward 样本会被过滤掉。

但低 reward 样本很有价值，因为它告诉模型：

```text
什么不要做
```

普通 SFT 对低 reward 样本的处理通常是：

```text
丢掉
或者
给很小权重
```

DPO/RL 则可以显式使用负例：

```text
winner > loser
高 reward action 提升概率
低 reward action 降低概率
```

这比单纯模仿正样本更直接。

## 5. 一个推荐例子

假设用户喜欢本地生活直播，同时最近看了短视频 A、B、C。

旧系统曝光了列表：

```text
L1 = [直播间1, 直播间2, 短视频3]
```

用户看了 60 秒，reward 很高。

加权 SFT 会做：

```text
提高 P(L1 | user_context)
```

但模型可能还能生成：

```text
L2 = [直播间1, 团购内容4, 直播间5]
L3 = [直播间6, 直播间7, 直播间8]
```

其中：

- L2 可能点击略低，但转化更高。
- L3 可能时长高，但多样性差、用户疲劳高。
- L1 是旧系统给出的，还不错，但未必最优。

SFT 只知道 L1。

RL 可以做：

```text
模型生成 L1/L2/L3
reward model 或线上反馈打分
综合点击、时长、转化、负反馈、多样性
提高综合 reward 更高的列表概率
```

这就是为什么把 reward 塞进 SFT 后，仍然可能需要 RL。

## 6. SFT 能不能部分替代 RL

可以，尤其在工程初期。

常见做法包括：

### 6.1 Reward-weighted SFT

```text
loss = - reward_weight * log P(y | x)
```

高 reward 样本权重大，低 reward 样本权重小。

### 6.2 Reject Sampling Fine-Tuning

先让模型或系统生成多个候选，保留高 reward 的候选做 SFT：

```text
generate candidates
score by reward
keep top samples
SFT on kept samples
```

OneRec Technical Report 里的 RSFT 就有类似思想：基于 play duration 过滤低质量曝光 session，再继续 SFT。

### 6.3 Distillation from ranker

用精排/重排模型选出高质量推荐结果，再让生成模型模仿：

```text
user_context -> ranker top list
```

这可以把多目标 ranker 的能力蒸馏给生成模型。

这些方法都很实用，而且比 RL 稳定。

但它们本质上仍是：

```text
把高 reward 样本变成监督 label
```

而不是直接优化当前 policy 的期望 reward。

## 7. 那什么时候可以不用 RL

如果满足下面条件，确实可以先不用 RL：

- 高质量 SFT 数据非常充分。
- 业务目标能很好地通过样本筛选表达。
- 不需要当前模型主动探索。
- 线上候选空间基本由旧系统覆盖。
- reward model 不稳定，贸然 RL 风险更大。
- 对模型生成合法性和稳定性要求高。

很多工业系统第一阶段都会这样做：

```text
预训练 + SFT/RSFT + ranker distillation
```

先把生成式推荐模型做可用，再考虑 RL。

## 8. 什么时候必须考虑 RL / 偏好优化

当你遇到这些问题时，SFT 就不太够了：

- 模型只是复刻旧系统，上限不高。
- 想优化列表级 reward，而不是单 item label。
- 想利用模型自己生成的候选。
- 有可靠 reward model 或真实用户反馈。
- 想显式降低低质量输出概率。
- 想做多目标 trade-off，而不是只模仿历史正样本。
- 想让模型突破传统召回/排序链路的候选限制。

这时就需要 DPO、GRPO、PPO、ECPO、GBPO 这类偏好优化或 RL 方法。

## 9. 和 OneRec 的关系

OneRec 的训练逻辑可以这样理解：

```text
Pretraining:
  学会从用户上下文生成 item semantic IDs

RSFT/SFT:
  用实时高质量曝光继续监督学习
  让模型稳定、合法、跟上用户兴趣变化

RL / Preference Alignment:
  用 reward / 真实反馈 / 偏好对
  优化模型自己生成结果的综合收益
```

也就是说，OneRec 没有把 SFT 和 RL 看成互斥关系，而是组合使用：

```text
SFT 负责稳定和基础能力
RL 负责目标对齐和突破历史分布
```

## 10. 面试表达模板

如果被问：

> 为什么不直接把点击、时长、转化、负反馈、多样性都融成 reward 做 SFT，而要 RL？

可以这样答：

> 可以把这些目标融成 reward 后用于样本筛选或加权 SFT，这在工程上很常见，比如 reward-weighted SFT 或 RSFT。但它不能完全替代 RL，因为 SFT 本质上还是在模仿日志中已经出现的 action，只是高 reward 样本权重大一些。推荐日志受旧系统曝光分布限制，SFT 很难学习没有被旧系统展示过但可能更优的 item/list，也缺少对模型当前生成结果的反事实比较。RL 优化的是当前 policy 下输出的期望 reward，可以让模型生成多个候选，用 reward 或真实反馈比较它们，再提升高收益候选、压低低收益候选。尤其多样性、负反馈、生态目标往往是 list-level 或 long-term reward，更适合通过 RL/偏好优化来做综合目标对齐。所以 SFT 可以吸收 reward 信息，但更多是稳定地模仿高质量样本；RL 才是在模型已会推荐之后，进一步优化最终业务目标。

