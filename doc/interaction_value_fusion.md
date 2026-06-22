# Interaction Value 融合公式解析

## 1. 先回答：为什么不用直接 weighted_sum

`interaction_value` 里使用的是**加权平均**，而不是直接返回 `weighted_sum`：

```python
return weighted_sum / total_weight
```

核心原因是：这里希望 `InteractionWeights` 表达的是各个互动行为的**相对重要性**，而不是用来改变最终分数尺度的**绝对放大系数**。

也就是说：

- `share` 比 `like` 更重要，可以给 `share` 更高权重。
- 但不希望因为所有权重加起来变大，导致整个 `interaction_value` 被整体放大。
- 最终 `interaction_value` 最好仍然像一个概率或概率均值一样，保持在比较稳定的 `[0, 1]` 范围。

如果直接返回 `weighted_sum`，一旦权重总和不是 1，分数尺度就会跟着变。

## 2. 当前默认值下有没有区别

当前默认权重是：

```python
class InteractionWeights:
    like: float = 0.35
    share: float = 0.25
    comment: float = 0.15
    follow: float = 0.15
    favorite: float = 0.10
```

这些权重加起来刚好是：

```text
0.35 + 0.25 + 0.15 + 0.15 + 0.10 = 1.00
```

所以在默认配置下：

```text
weighted_sum / total_weight = weighted_sum / 1 = weighted_sum
```

也就是说，当前默认值下直接用 `weighted_sum` 和用加权平均结果相同。

但写成加权平均更稳，因为以后如果调整权重，不需要强制保证它们加起来等于 1。

## 3. 加权平均解决了什么问题

假设有两个配置：

```text
配置 A:
like = 0.35
share = 0.25
comment = 0.15
follow = 0.15
favorite = 0.10
total = 1.00

配置 B:
like = 3.5
share = 2.5
comment = 1.5
follow = 1.5
favorite = 1.0
total = 10.00
```

这两个配置的相对比例完全一样，只是 B 整体放大了 10 倍。

如果使用加权平均，A 和 B 得到的 `interaction_value` 是一样的，因为整体缩放会被 `total_weight` 消掉。

如果直接使用 `weighted_sum`，B 的结果会变成 A 的 10 倍。这就会把“相对重要性调整”错误地变成“整体互动分放大”。

## 4. 为什么要保持 interaction_value 的尺度稳定

`interaction_value` 不是最终排序分，它会进入 `fuse_video_fine_rank_score` 的乘法融合公式：

```text
score = p_play^play
      * watch_value^watch
      * interaction_value^interaction
      * (1 - p_short_play)^short_play_penalty
      * (1 - p_negative)^negative_penalty
      * quality^quality
      * freshness^freshness
```

在这个公式里，各个子目标最好有相对稳定、可比较的尺度。比如：

- `p_play` 是概率，范围约为 `[0, 1]`。
- `watch_value` 是观看价值，通常也希望接近概率尺度。
- `interaction_value` 也应该像一个概率型分数。
- `short_play`、`negative` 也是概率。

如果 `interaction_value` 因为权重总和变大而超过 1，或者整体尺度漂移，就会影响乘法公式的含义，导致 `weights.interaction` 这个指数也需要重新调。

加权平均可以让 `interaction_value` 更稳定。

## 5. 为什么不是所有互动直接相加

点赞、分享、评论、关注、收藏的业务含义不同：

- `like` 成本低，信号比较便宜。
- `share` 往往表示更强认可。
- `comment` 表示参与，但有时也可能是争议。
- `follow` 对作者和长期关系更重要。
- `favorite` 或 `collect` 表示内容有复看或保存价值。

所以不应该简单等权相加，而是给不同行为不同权重。

本项目默认权重里，`like` 最高，是因为它最常见，覆盖更广；`share`、`follow` 虽然权重略低，但在真实系统中通常会被视为更强信号，具体大小需要结合数据调。

这里的默认值只是 demo 级配置，不代表线上最优参数。

## 6. 为什么 favorite 有 fallback_key="collect"

代码里这一行：

```python
task_probability(scores, "favorite", fallback_key="collect")
```

表示优先读取 `favorite`，如果没有，就读取 `collect`。

原因是不同系统对“收藏”这个行为的命名可能不同：

- 有的叫 `favorite`
- 有的叫 `collect`

加 fallback 可以让这个 demo 对两种命名都兼容。

## 7. 什么时候可以直接用 weighted_sum

直接使用 `weighted_sum` 也不是绝对错误，适合以下情况：

- 你明确要求权重具有绝对业务含义。
- 权重总和本来就是一个有意义的强度系数。
- 你希望某组权重整体变大时，互动模块整体影响也变大。
- 后续已经对 `interaction_value` 做了校准、截断或重新归一化。

但如果只是想表达“share 比 like 更重要”“follow 比 comment 更重要”这种相对偏好，用加权平均更合理。

## 8. 什么时候需要重新调参

下面这些情况需要重新评估 `InteractionWeights`：

- 平台互动行为分布变化，比如点赞率大幅上升。
- 新增或删除互动任务，比如加入 `dislike`、`download`、`profile_click`。
- 某些互动信号噪声很大，比如评论里负面评论比例高。
- 产品目标变化，比如更重视关注作者或收藏复看。
- 最终排序明显偏向标题党、高争议或强互动但低满意内容。

真实线上系统里，互动价值最好结合离线评估和 A/B 测试确定，而不是只凭经验固定。

## 9. 小结

这段实现的核心思想是：

```text
interaction_value = 加权互动概率的平均值
```

而不是：

```text
interaction_value = 加权互动概率的总和
```

这样设计的好处是：

- 权重表达相对重要性。
- 分数尺度更稳定。
- 默认保持概率型分数语义。
- 方便以后改权重，不要求权重和必须等于 1。
- 更适合进入后续乘法融合公式。

所以，当前默认权重下直接用 `weighted_sum` 没有差别；但从可维护性和调参安全性看，`weighted_sum / total_weight` 更稳。
