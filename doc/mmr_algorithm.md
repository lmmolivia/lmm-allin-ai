# MMR 算法解析

## 1. MMR 解决什么问题

MMR 是 **Maximal Marginal Relevance** 的缩写，通常翻译为“最大边际相关性”。它是一种贪心重排序算法，用来在“相关性”和“多样性”之间做平衡。

在推荐、搜索、召回和 RAG 文档检索里，排序模型经常会把一批非常相似的候选排在最前面。比如短视频推荐中，用户刚看完篮球内容，模型可能会连续把同一个作者、同一个类目、同一批标签的视频排在顶部。这些内容单看都很相关，但放在一个列表里会显得重复，也会降低探索性。

MMR 的核心目标是：每次选择下一个结果时，同时考虑两件事：

- 这个候选和用户兴趣、搜索词或查询目标是否足够相关。
- 这个候选和已经选出的结果是否过于相似。

所以 MMR 一般不是召回算法，也不是主排序模型，而是放在召回和排序之后的后处理阶段，用来调整最终展示列表。

## 2. 核心思想

假设有一个候选集合 `R`，以及已经选出的结果集合 `S`。MMR 每一轮都会从还没被选中的候选里，挑出“边际价值”最大的 item：

```text
argmax item in R - S:
    relevance(item) - diversity_penalty(item, S)
```

这里的 `relevance(item)` 表示候选本身的相关性，通常来自排序分；`diversity_penalty(item, S)` 表示候选和已选结果的重复程度。

多样性惩罚通常取候选与已选结果中“最相似”的那一个：

```text
diversity_penalty(item, S) = max_similarity(item, selected_item)
```

这样做的直觉是：只要候选和任意一个已选结果非常相似，就应该被扣分。MMR 是贪心算法，它不会一次性求整个列表的全局最优，而是在当前已选结果的基础上，每一轮挑当前看来最合适的下一个。

## 3. 经典公式

经典 MMR 公式通常写成：

```text
MMR(item) =
    lambda * Sim1(item, query)
    - (1 - lambda) * max(Sim2(item, selected_item))
```

其中：

- `Sim1(item, query)`：候选和用户/query 的相关性。
- `Sim2(item, selected_item)`：候选和已选结果之间的相似度。
- `lambda`：取值范围通常是 `[0, 1]`。
- `lambda` 越大，越偏向相关性。
- `lambda` 越小，越偏向多样性。

如果 `S` 为空，也就是还没有选出任何结果，那么第一条通常只按相关性选择。因此 MMR 排出来的第一条结果，一般还是原排序里分数最高的结果。

## 4. 本项目里的公式

在本项目中，MMR-style 多样性重排序实现在 `recsys/postprocess.py` 里：

```text
rerank_score = rank_score - diversity_lambda * max_similarity(candidate, selected)
```

可以把它理解成经典 MMR 的工程化惩罚版本：

- `rank_score`：排序模型或后排序公式融合后的相关性分数。
- `max_similarity(candidate, selected)`：候选和已选结果中最相似 item 的相似度。
- `diversity_lambda`：多样性惩罚强度。

这里和经典公式有一个重要区别：

- 经典公式里的 `lambda` 是相关性和多样性的插值权重。
- 本项目里的 `diversity_lambda` 是直接乘在相似度上的惩罚系数。

所以在本项目中，`diversity_lambda` 越大，重复内容被扣分越明显，最终列表越偏多样；`diversity_lambda=0` 时，软性的 MMR 多样性惩罚就关闭了，只剩下硬性的窗口和配额约束。

## 5. 选择流程

本项目的完整后处理流程大致是：

1. 过滤掉违反请求时硬规则的候选，比如已曝光内容、屏蔽作者等。
2. 按 `item_id` 去重，保留同一个 item 中排序分最高的候选。
3. 可选地执行 post-rank 公式融合，加入质量、新鲜度、负反馈等因素。
4. 执行 MMR-style 多样性重排序。
5. 应用滑动窗口打散和全局配额约束。
6. 返回最终推荐结果。

核心 MMR 循环可以简化成：

```text
selected = []
remaining = candidates sorted by rank_score desc

while remaining is not empty and len(selected) < limit:
    next_item = item in remaining with max diversified_score
    remove next_item from remaining

    if next_item violates window or quota constraints:
        skip it

    selected.append(next_item)
```

第一条结果通常仍然是排序分最高的 item，因为此时 `selected` 为空，还没有相似度惩罚。

## 6. 相似度设计

MMR 的效果很大程度取决于 item 相似度函数。这个项目里的 `item_similarity` 会综合多个内容侧信号：

```text
same author      + 1.00
same category    + 0.45
same city        + 0.15
tag Jaccard      + 0.35 * jaccard(tags)
embedding cosine + 0.35 * max(0, cosine(embedding))
```

这意味着“同作者”会受到最强惩罚；同类目、同城市、标签重合、向量相似也会增加相似度，但权重相对更低。

相似度函数应该服务于产品目标：

- 如果用户非常讨厌连续刷到同一个作者，作者相似度权重就应该高。
- 如果产品更关注类目多样性，可以提高 category 的权重。
- 如果 embedding 质量很高，可以提高向量余弦相似度权重。
- 如果 embedding 噪声较大，作者、类目、标签这类结构化特征通常更稳定。

## 7. 滑动窗口打散

MMR 是软惩罚。一个相似 item 即使被扣分，只要原始排序分足够高，仍然可能被选中。为了表达更严格的规则，本项目还引入了滑动窗口约束。

滑动窗口只检查最近选出的若干个结果：

```text
window = selected[-window_size:]
```

然后在这个窗口里限制重复：

- `author_window_limit`
- `category_window_limit`
- `city_window_limit`

例如，`author_window_limit=1` 表示：如果最近窗口里已经出现过同一个作者，那么当前候选会被拒绝。

当 `window_size=2` 时，它可以防止相邻两个 item 来自同一个作者；当 `window_size` 更大时，它会防止同一作者出现在更长的近期窗口里。

这类规则在推荐系统里通常叫“打散”。

## 8. 全局配额约束

除了滑动窗口，本项目还支持页面级全局配额：

- `author_global_limit`
- `category_global_limit`
- `source_global_limits`

滑动窗口只看最近一段结果，全局配额则统计整个最终列表。它适合控制一整页里某个作者、某个类目或某个召回源不要出现太多。

例如：

```text
category_global_limit = 2
```

表示最终结果页中，同一个 category 最多只能出现两个 item，不管它们出现在列表的哪个位置。

## 9. 参数调优

常见调参经验：

- `diversity_lambda = 0`：关闭软性的 MMR 多样性惩罚，只保留硬约束。
- 较小的 `diversity_lambda`：基本保留原排序，只减少明显重复。
- 较大的 `diversity_lambda`：更强的探索和多样性，但可能牺牲相关性。
- `author_window_limit = 1`：严格控制作者级打散。
- 较大的 `category_window_limit`：允许同一主题持续一段时间，但避免整页都重复。
- 全局配额适合页面级组成控制，但硬约束太多可能导致结果质量下降，甚至结果数量不足。

最终参数最好通过离线回放和线上 A/B 测试来决定。

## 10. 示例

假设排序后的候选列表如下：

| Item | Rank Score | Author | Category |
| --- | ---: | --- | --- |
| v1 | 0.99 | a1 | sports |
| v2 | 0.98 | a1 | sports |
| v3 | 0.97 | a2 | sports |
| v4 | 0.96 | a3 | food |

如果不做多样性处理，顶部结果可能是 `v1, v2, v3`，其中 `v1` 和 `v2` 都来自作者 `a1`，列表开头会显得重复。

加入 MMR-style 重排序和 `author_window_limit=1` 后，流程会变成：

1. 先选 `v1`，因为它排序分最高。
2. 再考虑 `v2` 时，发现它和 `v1` 是同作者，因此会受到强相似度惩罚，也可能违反作者窗口限制。
3. 接下来会根据排序分、相似度惩罚和硬约束，在 `v3`、`v4` 中选择更合适的结果。

最终列表会减少重复，同时尽量保留高相关性。

## 11. 优点

MMR 常用的原因是：

- 实现简单。
- 逻辑容易解释。
- 不依赖具体模型结构。
- 适合作为最终后处理层。
- 可以和业务配额、滑动窗口打散等规则结合。

当排序模型本身很强，但容易把相似内容扎堆排在顶部时，MMR 很有用。

## 12. 局限

MMR 也有明显局限：

- 它是贪心算法，不保证最终列表全局最优。
- 效果高度依赖相似度函数设计。
- 相似度权重设置不好，会损害相关性或造成生硬的多样性。
- 如果候选集很大、相似度计算很贵，性能开销会比较明显。
- 如果候选本身缺乏多样性，硬约束可能跳过太多 item，导致结果变少。

在生产系统里，MMR 通常会结合候选截断、相似度缓存、向量索引、离线评估和线上实验一起使用。

## 13. 复杂度

假设要从 `N` 个候选中选出 `K` 个最终结果，朴素实现每一轮都要扫描很多剩余候选，并把候选和已选结果做相似度比较。

近似复杂度是：

```text
O(K * N * C)
```

其中 `C` 是计算候选与已选结果最大相似度的成本。如果每次都扫描全部已选结果，那么 `C` 会随着 `K` 增长。

常见优化方式：

- 只对排序后的 top `N` 候选执行 MMR。
- 缓存 pairwise similarity。
- 缓存 item embedding 和元数据相似度组件。
- 一旦选够结果就提前停止。

## 14. 评估指标

相关性指标：

- CTR
- watch time
- completion rate
- NDCG
- MAP

多样性指标：

- author repetition rate
- category repetition rate
- source distribution
- item coverage
- author coverage
- category entropy
- intra-list average distance

护栏指标：

- negative feedback rate
- hide/report rate
- empty or short response rate
- latency

评估 MMR 时不能只看多样性。目标通常不是“多样性越高越好”，而是在相关性损失可控的前提下，改善最终列表的重复问题和探索性。

## 15. 什么时候适合用 MMR

适合使用 MMR 的情况：

- 排序结果顶部有大量近重复或高度相似内容。
- 产品希望首页或 feed 更丰富。
- 需要一个透明、可解释的排序后处理规则。
- 已经有比较可靠的 item 相似度信号。

不适合只依赖 MMR 的情况：

- 召回阶段本身就缺乏足够多样的候选。
- 相似度特征缺失或噪声很大。
- 产品需要复杂的全局列表优化。
- 约束非常严格，已经接近求解器或专门分配算法的问题。

## 16. 和本仓库代码的关系

相关代码：

- `recsys/postprocess.py`
  - `diversity_rerank`
  - `diversified_score`
  - `item_similarity`
  - `violates_window`
  - `violates_global_quota`
- `recsys/types.py`
  - `Item`
  - `Candidate`
  - `RankedCandidate`
  - `Recommendation`

相关测试：

- `tests/test_postprocess.py`
- `tests/test_recommendation_pipeline.py`

在这个仓库里，MMR 不是完整推荐系统本身，而是召回、排序、可选 post-rank 公式融合、去重、过滤之后的最终后处理步骤。
