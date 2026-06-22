# Watch Value 融合公式解析

## 1. 先回答：这是自研还是工业界有依据

`watch_value` 这个函数不是某篇论文或某家公司公开方案的原样复刻，也不能说是一个行业统一标准公式。

更准确的说法是：它是一个**工业界常见思想的简化工程版本**。

它背后的思想在视频推荐里很常见：

- 不只优化 click/play，还要优化观看质量。
- 观看质量不能只看完播率，因为长视频天然更难完播。
- 观看质量也不能只看观看时长，因为长视频天然更容易获得更长 watch time。
- 所以需要把 `finish`、`long_play`、`duration` 这类信号组合起来，得到一个更稳定的观看价值分。

本项目里的实现是：

```python
def watch_value(
    scores: Mapping[str, float],
    duration_seconds: float,
    config: WatchValueConfig | None = None,
) -> float:
    """Blend finish-rate and long-play-rate into one effective watch value."""

    config = config or WatchValueConfig()
    finish = task_probability(scores, "finish")
    long_play = task_probability(scores, "long_play")
    return (
        config.finish_weight * finish
        + config.long_play_weight * long_play * duration_gain(duration_seconds, config)
    )
```

可以理解为：

```text
watch_value =
    finish_weight * p_finish
    + long_play_weight * p_long_play * duration_gain(duration)
```

其中：

- `p_finish`：预测用户是否会看完。
- `p_long_play`：预测用户是否会长播。
- `duration_gain(duration)`：根据视频时长给长播信号一个温和增益。

## 2. 为什么要融合 finish 和 long_play

短视频推荐里，单一观看指标容易有偏。

只看 `finish` 的问题：

- 对短视频有利。
- 一个 8 秒视频被看完很容易。
- 一个 120 秒视频即使用户看了 70 秒，也可能没有完播。
- 如果只优化完播率，系统可能偏向短、轻、容易看完的内容。

只看 `long_play` 或 watch time 的问题：

- 对长视频有利。
- 长视频天然更容易产生更长观看时长。
- 如果只优化观看时长，系统可能过度推荐长视频，哪怕用户真实兴趣并不强。

所以一个更合理的做法是：

- 用 `finish` 捕捉“用户是否完整消费了内容”。
- 用 `long_play` 捕捉“用户是否投入了足够多观看时间”。
- 用 `duration_gain` 处理时长差异，避免短视频和长视频互相不公平。

这就是 `watch_value` 的设计动机。

## 3. duration_gain 为什么要加

本项目里的 `duration_gain` 是：

```python
def duration_gain(duration_seconds: float, config: WatchValueConfig | None = None) -> float:
    config = config or WatchValueConfig()
    duration = max(0.0, min(duration_seconds, config.duration_cap_seconds))
    reference = max(1.0, config.duration_reference_seconds)
    return math.log1p(duration) / math.log1p(reference)
```

它有三个关键点：

1. 使用 `log1p(duration)`，让时长增长带来的收益逐渐变慢。
2. 使用 `duration_cap_seconds` 截断超长视频，避免极长视频仅凭时长占优。
3. 使用 `duration_reference_seconds` 作为参考时长，让公式更容易调参。

直觉上：

- 30 秒到 60 秒，确实可以认为长播价值更高。
- 120 秒到 600 秒，不应该线性放大 5 倍。
- 所以用对数函数做饱和增益，比直接乘时长更稳。

这是一种常见的工程化防偏做法：保留长视频能贡献更多观看时间的事实，但不让原始时长直接主导排序。

## 4. duration_cap_seconds 和 duration_reference_seconds 的含义

`duration_cap_seconds` 是时长上限，也就是公式里 `min(duration, cap)` 的 `cap`。

它的作用是：超过这个时长后，不再继续增加 `duration_gain`。比如 `duration_cap_seconds=120` 时，120 秒视频和 600 秒视频在这个增益项里都会按 120 秒处理。

这样做是为了防止超长视频只因为“更长”就获得更大的长播价值。它不是说 600 秒视频没有价值，而是说这条简单公式不应该让 600 秒相对 120 秒继续线性占优。

`duration_reference_seconds` 是参考时长，也就是分母里的 `reference`：

```text
gain = log(1 + min(duration, cap)) / log(1 + reference)
```

它的作用是定义“多长的视频增益为 1”。如果 `duration_reference_seconds=30`，那么：

```text
duration = 30 秒时，duration_gain = 1
duration < 30 秒时，duration_gain < 1
duration > 30 秒时，duration_gain > 1
```

所以它可以理解为一个基准视频长度。比基准短的视频，长播价值会被轻微压低；比基准长的视频，长播价值会被适度放大。

## 5. 一般设置为多少

没有行业统一固定值，需要按内容时长分布、产品目标和实验结果调。常见经验可以这样理解：

| 场景 | `duration_reference_seconds` | `duration_cap_seconds` |
| --- | ---: | ---: |
| 极短视频 feed，主体 5-20 秒 | 10-15 秒 | 30-60 秒 |
| 常见短视频 feed，主体 10-60 秒 | 20-30 秒 | 90-180 秒 |
| 中视频或混合视频，主体 1-5 分钟 | 60-120 秒 | 300-600 秒 |
| 长视频推荐 | 通常不建议只用这个简单公式 | 需要按业务另设 |

本项目默认值是：

```text
duration_reference_seconds = 30
duration_cap_seconds = 120
```

这个默认值适合用来表达“短视频到中短视频”的 demo 场景：

- 30 秒作为基准时长。
- 120 秒作为长播增益封顶点。
- 30 秒以下不会因为 `long_play` 获得太多时长加成。
- 30-120 秒会有温和加成。
- 超过 120 秒不继续靠时长获得更多加成。

举例看这个默认配置下的增益：

| 视频时长 | 近似 `duration_gain` | 含义 |
| --- | ---: | --- |
| 10 秒 | 0.70 | 比 30 秒基准低 |
| 30 秒 | 1.00 | 基准时长 |
| 60 秒 | 1.20 | 有一定长播加成 |
| 120 秒 | 1.40 | 达到默认封顶附近 |
| 600 秒 | 1.40 | 因为 cap=120，不继续增加 |

调参时可以先看视频时长分布：

- 如果大多数内容都在 10-20 秒，`reference=30` 可能偏大，会压低短视频长播价值。
- 如果希望鼓励 1-2 分钟内容，`cap=120` 是合理起点。
- 如果平台主要是 3-5 分钟内容，`cap=120` 可能太低，长内容差异会被过早抹平。
- 如果发现排序明显偏长视频，可以降低 `cap` 或降低 `long_play_weight`。
- 如果发现排序过度偏短、长视频很难出头，可以提高 `reference` 或 `cap`，但要盯住负反馈和跳出率。

更稳的生产做法不是拍一个固定值，而是按时长桶评估：

- 0-10 秒
- 10-30 秒
- 30-60 秒
- 60-120 秒
- 120 秒以上

观察每个桶的曝光、点击、完播、长播、负反馈和留存，再决定参数。

## 6. 工业界依据是什么

工业界视频推荐确实长期围绕 watch time、finish、long-play、interaction、negative feedback 等多目标做排序。

公开论文里常见的方向包括：

- 把 watch time 作为视频推荐的重要优化目标。
- 使用多任务学习同时预测点击、观看、互动、负反馈等目标。
- 研究视频时长带来的 duration bias。
- 对 watch time 做去偏或归一化，而不是直接使用原始观看时长。
- 在短视频推荐中，同时兼顾观看时长和点赞、关注、分享等辅助目标。

因此，本项目公式里的思想有工业界依据：

```text
观看价值 = 完播相关信号 + 长播相关信号 + 时长修正
```

但具体到：

```text
finish_weight = 0.45
long_play_weight = 0.55
duration_cap_seconds = 120
duration_reference_seconds = 30
```

这些数值不是行业固定答案，而是这个 demo 项目里为了表达思想设置的默认值。真实线上系统需要通过离线评估和 A/B 测试来调。

## 7. 为什么不是直接用 finish * long_play

可以，但不一定适合。

如果用乘法：

```text
watch_value = p_finish * p_long_play
```

它会很严格：只要其中一个概率低，整体就会很低。

这在某些场景里合理，但对不同视频长度可能不公平：

- 短视频 `p_finish` 高，但 `p_long_play` 可能不高。
- 长视频 `p_long_play` 高，但 `p_finish` 可能不高。

线性融合更温和：

```text
finish_weight * p_finish + long_play_weight * p_long_play * duration_gain
```

它允许短视频通过完播体现价值，也允许长视频通过长播体现价值。

## 8. 为什么不是直接用 watch_time

直接预测或优化 watch time 是工业界常见目标，但它有 duration bias。

比如：

- 视频 A：10 秒，用户看完 10 秒。
- 视频 B：120 秒，用户看了 30 秒。

如果只看原始观看时长，B 比 A 更好。但如果看用户满意度，A 未必比 B 差。

所以短视频推荐里经常需要处理：

- 原始观看时长。
- 完播率。
- 长播。
- 相对观看比例。
- 时长分桶或时长归一化。
- duration bias 去偏。

本项目里的 `duration_gain` 是一个非常简化的处理方式：它没有完整解决 duration bias，但表达了“不能让原始时长线性主导排序”的原则。

## 9. 和本项目 fine-rank 公式的关系

`watch_value` 不是最终排序分，它只是 fine-rank 公式中的一个子目标。

本项目最终 fine-rank 融合在 `fuse_video_fine_rank_score` 中：

```text
score = p_play^play
      * watch_value^watch
      * interaction_value^interaction
      * (1 - p_short_play)^short_play_penalty
      * (1 - p_negative)^negative_penalty
      * quality^quality
      * freshness^freshness
```

也就是说：

- `watch_value` 表示有效观看价值。
- `interaction_value` 表示点赞、分享、评论、关注、收藏等互动价值。
- `short_play` 和 `negative` 是负向信号。
- `quality` 和 `freshness` 是内容侧先验。

这种设计思路和工业界多目标排序是一致的：不把推荐质量压成一个单一行为，而是融合多个任务目标。

## 10. 这个实现适合什么用途

适合：

- 教学或 demo 项目。
- 解释短视频 fine-rank 多目标融合。
- 在没有真实模型和 A/B 系统时，先建立一个可读、可调的排序公式。
- 作为后续替换成真实模型分数的接口。

不适合直接当成线上最终公式：

- 默认权重没有经过数据验证。
- 没有做完整校准。
- 没有做 position bias、duration bias、selection bias 的系统去偏。
- 没有按不同视频长度、用户群体、场景做分桶调参。
- 没有考虑长期留存、创作者生态、内容安全等更复杂目标。

## 11. 如果要更接近生产系统，可以怎么改

可以逐步增强：

1. 对 `finish`、`long_play`、`short_play`、`negative` 做概率校准。
2. 按视频时长分桶评估 `watch_value` 是否偏向短视频或长视频。
3. 用离线指标比较不同 `finish_weight` 和 `long_play_weight`。
4. 用 A/B 测试观察 watch time、完播率、负反馈、留存的综合变化。
5. 把 `duration_gain` 从手写函数升级为数据学习出的校正项。
6. 引入更完整的去偏方法，比如位置偏差和时长偏差修正。

## 12. 小结

这段公式可以这样定位：

```text
它不是拍脑袋，但也是简化版。
```

它借鉴的是工业界视频推荐中很常见的多目标排序思想：

- 完播表示消费完整度。
- 长播表示观看投入。
- 时长需要被修正，不能粗暴线性放大。
- 多个目标最终通过 fine-rank 公式融合。

但本项目里的权重和具体函数是为了 demo 清晰度设计的，不代表任何平台的线上真实参数。真正上线时，这些参数必须通过数据、校准和实验来定。

## 13. 参考方向

可参考的公开研究方向：

- 短视频推荐中的 watch time 优化和多目标约束。
- 视频推荐中的 duration bias / watch-time bias。
- 多任务排序模型对 click、finish、long_play、interaction、negative feedback 的联合建模。
- watch time 去偏和归一化指标，如 watch time gain、quantile-based watch-time modeling。

公开资料示例：

- [DVR: Micro-Video Recommendation Optimizing Watch-Time-Gain under Duration Bias](https://arxiv.org/abs/2208.05190)
- [Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation](https://arxiv.org/abs/2206.06003)
- [Constrained Reinforcement Learning for Short Video Recommendation](https://arxiv.org/abs/2205.13248)
