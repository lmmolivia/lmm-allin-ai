# TIGER 生成式召回详解

> 说明：本文基于 TIGER 原论文《Recommender Systems with Generative Retrieval》、arXiv 页面和论文 HTML 版本整理。论文没有披露的工业实现细节，本文不会当作事实；需要工程外推的地方会明确写成“工程理解”。

## 1. 先给结论

TIGER 全称 **Transformer Index for GEnerative Recommenders**，是 NeurIPS 2023 的一篇生成式召回论文。它把顺序推荐里的“候选生成”从传统的：

```text
user embedding -> ANN / MIPS 检索 item embedding
```

改成：

```text
用户行为序列 -> Transformer 自回归生成下一个 item 的 Semantic ID -> 查表还原 item_id
```

也就是说，TIGER 的召回不是先算用户向量再向量检索，而是让 Transformer 直接生成候选 item 的离散语义标识。这个标识不是随机 item id，而是由 item 内容 embedding 经过 RQ-VAE 量化得到的 **Semantic ID**。

对推荐工程来说，可以把 TIGER 理解成：

```text
TIGER = 内容语义离散化 tokenizer + 序列生成模型 + semantic_id/item_id 查表召回
```

它最核心的变化有两个：

1. item 不再直接用原始 id 或随机 id 表示，而是变成多级语义 token。
2. 召回不再是 `u2i dot product + ANN`，而是 `seq2seq decoding + semantic_id lookup`。

## 2. 资料来源

| 来源             | 用途                         | 链接                                                                                     |
|------------------|------------------------------|------------------------------------------------------------------------------------------|
| arXiv            | 摘要、作者、版本信息         | https://arxiv.org/abs/2305.05065                                                         |
| NeurIPS PDF      | 原论文正文和实验细节         | https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf |
| ar5iv HTML       | 便于核对公式和段落           | https://ar5iv.labs.arxiv.org/html/2305.05065                                              |

我没有在公开资料中确认到作者维护的官方代码仓库；GitHub 上可以搜到若干 unofficial implementation，但本文不把它们作为论文事实来源。

## 3. 它解决什么问题

传统大规模推荐召回常见做法是双塔或多塔：

```text
user tower  -> user embedding
item tower  -> item embedding
ANN / MIPS  -> topK candidate items
```

这套范式很成熟，但有几个天然问题：

1. **item id 原子化**：每个 item 一个 embedding，新 item 或低频 item 很难共享知识。
2. **内容语义利用不充分**：双塔可以加内容特征，但最终仍然通常依赖连续向量空间检索。
3. **召回和生成式序列建模割裂**：用户行为序列模型学到的是下一个 item 分布，但 serving 时还要落到 ANN 检索。
4. **索引维护成本**：item embedding 变化后，需要重建或增量更新 ANN index。

TIGER 想做的是把 item corpus “语言化”：每个 item 是一个由多个 semantic token 组成的短序列，模型像语言模型生成词一样生成下一个 item 的 token 序列。

## 4. 整体流程

TIGER 分成两阶段：

```text
阶段一：给 item 生成 Semantic ID

item 内容特征
-> 预训练内容 encoder，例如 Sentence-T5 / BERT
-> dense content embedding
-> RQ-VAE residual quantization
-> Semantic ID，例如 (7, 1, 4, 0)

阶段二：训练生成式召回模型

用户历史 item 序列
-> item_id 替换成 Semantic ID token 序列
-> Transformer encoder-decoder
-> 预测下一个 item 的 Semantic ID
-> 查 semantic_id_to_item_id 表得到真实 item
```

论文实验中，输入序列还会在最前面加一个 user id token。为了控制词表规模，论文用 Hashing Trick 把原始 user id 映射到 2000 个 user id token 之一。

## 5. Semantic ID 是什么

Semantic ID 是 item 的离散语义标识，形式是一个 codeword tuple：

```text
item_i -> (c_i,0, c_i,1, ..., c_i,m-1)
```

其中每个 `c_i,d` 来自第 `d` 层 codebook。假设每层 codebook 大小分别是 `K_0, K_1, ..., K_{m-1}`，理论可表示的 ID 数量是：

```text
K_0 * K_1 * ... * K_{m-1}
```

论文希望 Semantic ID 具备一个性质：

```text
语义相似的 item，Semantic ID 应该有更多重合前缀或 codeword。
```

比如：

```text
(10, 21, 35) 与 (10, 21, 40) 更相似
(10, 21, 35) 与 (10, 23, 32) 相似度更低
```

这点对推荐召回很重要，因为模型可以在相似 item 之间共享统计强度。对于低频 item、新 item，Semantic ID 可以让它从内容相似 item 那里继承一部分泛化能力。

## 6. RQ-VAE 如何生成 Semantic ID

论文使用 **Residual-Quantized VAE / RQ-VAE** 对 item 内容 embedding 做量化。

假设 item 的内容 embedding 是：

```text
x in R^d
```

RQ-VAE 先用 encoder 得到低维 latent：

```text
z = E(x)
```

然后做多层残差量化。令：

```text
r_0 = z
```

第 `d` 层从 codebook `C_d = {e_k}` 中选距离当前残差最近的 code vector：

```text
c_d = argmin_k || r_d - e_k ||
```

然后更新残差：

```text
r_{d+1} = r_d - e_{c_d}
```

重复 `m` 层以后，Semantic ID 就是：

```text
(c_0, c_1, ..., c_{m-1})
```

量化后的向量可以理解为多个 code vector 的和：

```text
z_hat = e_{c_0} + e_{c_1} + ... + e_{c_{m-1}}
```

再通过 decoder 还原原始内容 embedding：

```text
x_hat = D(z_hat)
```

论文中的训练目标由 reconstruction loss 和 RQ-VAE 量化相关 loss 组成：

```text
L_recon = || x - x_hat ||^2

L_rqvae = sum_d [
    || sg[r_d] - e_{c_d} ||^2
    + beta * || r_d - sg[e_{c_d}] ||^2
]

L = L_recon + L_rqvae
```

其中：

| 符号            | 含义                                                |
|-----------------|-----------------------------------------------------|
| `sg[]`          | stop-gradient，只传数值，不让梯度穿过               |
| `e_{c_d}`       | 第 `d` 层选中的 codebook vector                     |
| `beta`          | commitment 项权重，论文实验里使用 `0.25`            |
| `L_recon`       | 让 decoder 能重建原始内容 embedding                 |
| `L_rqvae`       | 同时更新 encoder、decoder 和 codebook               |

为了避免 codebook collapse，论文采用 k-means 初始化：在第一个训练 batch 上做 k-means，用聚类中心初始化 codebook。

## 7. 为什么是 RQ-VAE，不是随机 ID 或 LSH

论文对比了 Random ID、LSH Semantic ID 和 RQ-VAE Semantic ID。结论是 RQ-VAE SID 在三个 Amazon 数据集上都优于 Random ID 和 LSH SID。

| 方法            | ID 是否有语义         | 是否层级化              | 论文实验结论                         |
|-----------------|----------------------|-------------------------|--------------------------------------|
| Random ID       | 否                   | 否                      | 明显弱于 Semantic ID                 |
| LSH SID         | 有一定局部相似性     | 弱                      | 弱于 RQ-VAE SID                      |
| RQ-VAE SID      | 是                   | 是                      | 三个数据集上效果最好                 |

工程理解：Random ID 更像把 item 当成无意义 token，模型只能靠行为共现学习；RQ-VAE SID 则把内容语义提前编码进 token 空间，让序列模型更容易在相似 item 间泛化。

## 8. collision 怎么处理

多个 item 可能被 RQ-VAE 映射到同一个 Semantic ID。论文把这个问题叫 semantic collisions。

处理方式很直接：在 RQ-VAE 生成的 code tuple 后面追加一个额外 token，用来区分冲突 item。

例如两个 item 都得到：

```text
(12, 24, 52)
```

则追加冲突编号：

```text
item_a -> (12, 24, 52, 0)
item_b -> (12, 24, 52, 1)
```

论文实验里，RQ-VAE 先生成 3-tuple Semantic ID，然后追加第 4 个 codeword 解决冲突；如果没有冲突，第 4 个 codeword 也会设成 `0`。最终每个 item 都有唯一的长度为 4 的 Semantic ID。

需要注意：collision 检测和修复只在 RQ-VAE 训练完成后做一次。之后会冻结两张查表：

```text
item_id      -> semantic_id
semantic_id  -> item_id
```

## 9. 生成式召回模型怎么训练

假设用户历史行为按时间排序为：

```text
(item_1, item_2, ..., item_n)
```

每个 item 的 Semantic ID 是：

```text
item_i -> (c_i,0, c_i,1, ..., c_i,m-1)
```

那么用户历史会被展开成 token 序列：

```text
(c_1,0, ..., c_1,m-1,
 c_2,0, ..., c_2,m-1,
 ...
 c_n,0, ..., c_n,m-1)
```

模型目标是预测下一个 item 的 Semantic ID：

```text
(c_{n+1,0}, ..., c_{n+1,m-1})
```

论文实验使用 Transformer encoder-decoder 架构。输入是：

```text
user_id_token + history_semantic_id_tokens
```

输出是：

```text
next_item_semantic_id_tokens
```

## 10. 论文实验配置

以下是论文公开披露的实验配置，不应理解成线上系统必须照搬。

| 模块                  | 论文实验设置                                                                 |
|-----------------------|------------------------------------------------------------------------------|
| 数据集                | Amazon Product Reviews 的 Beauty、Sports and Outdoors、Toys and Games         |
| 评估指标              | Recall@5、Recall@10、NDCG@5、NDCG@10                                         |
| 数据切分              | leave-one-out，最后一个 item 测试，倒数第二个验证，其余训练                  |
| 历史长度              | 训练时用户历史最多保留 20 个 item                                            |
| 内容 encoder          | Sentence-T5                                                                  |
| item 内容字段         | title、price、brand、category 拼成文本                                        |
| 内容 embedding        | 768 维                                                                       |
| RQ-VAE encoder        | 512、256、128 三个中间层，ReLU，最终 latent 维度 32                           |
| RQ-VAE codebook       | 3 层 residual quantization，每层 codebook size 256，code vector 维度 32        |
| RQ-VAE beta           | 0.25                                                                         |
| RQ-VAE optimizer      | Adagrad，learning rate 0.4，batch size 1024                                   |
| RQ-VAE 训练           | 20k epochs，目标之一是 codebook usage >= 80%                                  |
| 最终 Semantic ID      | 3 个 RQ-VAE codeword + 1 个 collision code，总长度 4                         |
| seq2seq 框架          | T5X                                                                          |
| semantic token 词表   | 256 * 4 = 1024 个 codeword token                                             |
| user token            | 2000 个，通过 Hashing Trick 映射原始 user id                                 |
| Transformer           | encoder 4 层、decoder 4 层、6 个 attention heads，每个 head dim 64            |
| MLP / input dim       | MLP dim 1024，input dim 128                                                  |
| dropout               | 0.1                                                                          |
| 参数量                | 约 13M                                                                       |
| seq2seq 训练          | Beauty/Sports 200k steps，Toys 100k steps，batch size 256                     |
| seq2seq learning rate | 前 10k steps 为 0.01，之后 inverse square root decay                          |

## 11. 和双塔向量召回的核心区别

| 维度                 | 双塔 / ANN 召回                                      | TIGER 生成式召回                                      |
|----------------------|------------------------------------------------------|-------------------------------------------------------|
| item 表示            | 连续 item embedding                                  | 离散 Semantic ID token 序列                           |
| user 表示            | user embedding                                       | 用户行为 token 序列的 Transformer context             |
| 候选生成             | dot product / MIPS / ANN                             | 自回归生成 semantic tokens                            |
| serving 核心依赖     | ANN index                                            | Transformer decoding + semantic_id_to_item_id 查表     |
| 新 item 泛化         | 依赖内容塔、特征和索引更新                           | 可用内容 encoder + RQ-VAE 生成 Semantic ID             |
| item embedding 表    | 通常随 item 数线性增长                               | codeword embedding 表随 codebook 总规模增长            |
| 推理成本             | ANN 高效                                             | beam search / decoding 更贵                           |
| 主要风险             | 语义弱、低频 item 学不稳、索引维护                   | invalid ID、解码成本、Semantic ID 质量依赖内容 encoder |

对视频推荐来说，TIGER 更像一种“生成式 U2I 召回”，但它不是传统 U2I 向量召回。它把 item corpus 变成 token space，然后用用户序列生成目标 item token。

## 12. 推理阶段怎么召回

线上推理可以抽象为：

```text
1. 取用户最近行为序列
2. 把 item_id 转成 Semantic ID token
3. 拼接 user_id_token
4. Transformer decoder 用 beam search 或 sampling 生成多个 Semantic ID
5. 过滤 invalid Semantic ID
6. semantic_id_to_item_id 查表得到 item
7. 去重、过滤已曝光、接粗排/精排/重排
```

因为 Semantic ID 是自回归生成的，模型可能生成不存在的 ID。论文指出，实验中使用长度 4、每个 codeword cardinality 256 的 ID，理论空间是：

```text
256^4 ~= 4 trillion
```

但真实 item 只有 10K-20K。因此 invalid ID 是可能的。论文在三个数据集上观察到 top-10 预测里的 invalid ID 比例约为 `0.1% - 1.6%`，top-20 约为 `0.3% - 6%`。论文建议可以增大 beam size，然后过滤 invalid ID，保证拿到足够多的 valid items。

工程上还可以做 prefix fallback，例如完整 ID 不存在时，用 `(c_0, c_1, c_2)` 前缀召回相近 item。但这在论文里被作为 future work 提到，不是论文已验证方案。

## 13. 冷启动和多样性

论文认为 TIGER 有两个新能力：冷启动召回和可控多样性。

### 13.1 冷启动召回

TIGER 的 Semantic ID 来自内容 embedding，不完全依赖 item 的交互历史。因此新 item 可以先通过内容 encoder 和 RQ-VAE 得到 Semantic ID，再进入生成式召回空间。

论文在 Beauty 数据集上模拟冷启动：从训练集中移除 5% 的测试 item，把它们当成 unseen items。推理时，如果模型生成的前三个 semantic token 与 unseen item 匹配，就可以把 unseen item 放入候选集合，并用参数 `epsilon` 控制 topK 中 unseen items 的最大占比。

这说明 TIGER 对冷启动有潜力，但也要注意：论文是在公开数据集模拟设置上验证，不等于线上新视频一定天然表现好。真实短视频场景还会受封面、标题、ASR/OCR、视觉质量、作者冷启动、审核延迟、供给分发策略等因素影响。

### 13.2 多样性控制

论文使用 temperature-based decoding 控制生成多样性。由于 Semantic ID 是层级化的：

```text
第 1 个 token 更偏 coarse category
第 2/3 个 token 更偏 fine-grained category
```

所以在不同层级上采样会带来不同粒度的多样性。论文用 Entropy@K 衡量 topK item 的 category 分布熵，发现提高 temperature 可以提升 category 多样性。

工程理解：这有点像在召回阶段做“语义簇级别的多样性探索”，但线上仍然需要和多样性打散、作者/类目频控、负反馈过滤、业务规则共同使用。

## 14. 效果结论

论文在三个 Amazon Product Reviews 数据集上与 GRU4Rec、Caser、HGN、SASRec、BERT4Rec、FDSA、S3-Rec、P5 等方法比较。论文报告 TIGER 在 Recall@K 和 NDCG@K 上整体优于这些 baseline。

论文明确提到的相对提升包括：

| 数据集              | 指标                  | 论文报告的相对提升                                      |
|---------------------|-----------------------|---------------------------------------------------------|
| Beauty              | NDCG@5                | 相比 SASRec 最高约 29%                                  |
| Beauty              | Recall@5              | 相比 S3-Rec 约 17.3%                                    |
| Toys and Games      | NDCG@5                | 相比最佳 baseline 约 21%                                |
| Toys and Games      | NDCG@10               | 相比最佳 baseline 约 15%                                |

不要把这些数字直接迁移成工业短视频场景预期收益。Amazon Review 是公开离线 benchmark，行为类型、反馈密度、item 生命周期、用户意图和短视频分发差异都很大。

## 15. 对视频推荐召回的工程启发

如果把 TIGER 思路迁移到视频推荐，可以考虑下面的拆法。

### 15.1 item tokenizer

视频 item 的 Semantic ID 不建议只用标题文本生成，可以融合：

```text
标题、caption、ASR、OCR、封面、视频帧、多模态 embedding、作者、类目、poi/city、协同 item embedding
```

但这里要区分事实和工程推断：TIGER 原论文实验主要使用 Amazon 商品的 title、price、brand、category 文本字段；多模态视频 tokenizer 是短视频工程迁移时的合理方向，不是 TIGER 原论文已经验证的设置。

### 15.2 训练样本

视频推荐可以把用户历史按时间展开：

```text
watch / click / like / follow / comment / share / long_view / dislike
```

目标可以是下一次正反馈视频、下一次长播视频，或高价值行为视频。也可以区分召回目标：

```text
短期兴趣召回：最近 N 次行为 -> next positive item
长期兴趣召回：长期高价值行为 -> next positive item
场景召回：city/poi/时间段/context -> next local item
```

### 15.3 serving 接口

线上可以把 TIGER 作为一个召回源，而不是直接替代全部召回：

```text
tag-based recall
u2i vector recall
i2i recall
author/category/city/poi recall
TIGER generative recall
```

然后在粗排层统一做 dedup、quota、分桶、质量过滤和打分融合。

### 15.4 需要重点监控

| 风险点                 | 需要监控的指标                                      |
|------------------------|-----------------------------------------------------|
| invalid ID             | invalid ratio、valid topK 数量、beam size 需求      |
| token collapse         | codebook usage、Semantic ID 分布、热门 token 占比    |
| 热门偏置               | 头部 item 占比、作者集中度、类目集中度              |
| 冷启动质量             | 新视频曝光后转化、长播率、负反馈率                  |
| 解码成本               | latency、QPS、batching 效率、beam size              |
| 线上衔接               | TIGER 召回源覆盖率、粗排通过率、精排贡献            |

## 16. 局限性

TIGER 论文自己也承认，推理阶段可能比 ANN-based 模型更贵，因为需要 beam search 做自回归解码。论文目标主要是证明生成式召回范式有效，而不是把推理效率优化到工业级最优。

主要局限可以总结为：

1. **推理成本高**：生成多个 token 比一次 ANN 检索更重。
2. **存在 invalid ID**：生成空间远大于真实 item 集合，需要过滤和兜底。
3. **Semantic ID 质量很关键**：content encoder、RQ-VAE 训练、codebook 使用率都会影响召回。
4. **item 更新链路复杂**：新 item 需要跑内容 encoder、RQ-VAE、查表更新，并考虑 collision。
5. **论文验证场景有限**：主要是 Amazon Review benchmark，和短视频、直播、LBS 等工业场景仍有差异。

## 17. 一句话面试版

TIGER 是一种生成式召回方法，它先用内容 encoder 和 RQ-VAE 把 item 转成层级化 Semantic ID，再训练 Transformer 根据用户历史自回归生成下一个 item 的 Semantic ID，最后通过查表还原 item。相比双塔 ANN，它把召回从向量最近邻检索变成 semantic token 生成，优势是能利用内容语义、缓解低频和冷启动 item 泛化问题，并可通过 decoding 控制多样性；代价是推理成本、invalid ID 过滤和 Semantic ID 质量控制更复杂。

## 18. 和你当前推荐系统知识栈的关系

你之前做的双塔、UMA、STAMP、tag recall，基本都还是“召回候选 item”范式。TIGER 可以放在召回层作为一个新的生成式召回源：

| 召回源                 | 主要信号                         | 适合解决的问题                                      |
|------------------------|----------------------------------|-----------------------------------------------------|
| tag/city/poi/cate 召回 | 显式字段匹配                     | 强规则、强场景、可解释、低成本                      |
| 双塔 U2I               | user/item 向量相似               | 泛化召回、海量候选、ANN 高效 serving                |
| I2I / Swing            | item 共现                        | 最近兴趣、相似内容扩展                              |
| STAMP / MIND / UMA     | 多兴趣、session、用户行为模式    | 多兴趣召回、短期意图、长期偏好                      |
| TIGER                  | Semantic ID + 序列生成           | 内容语义泛化、冷启动探索、生成式候选召回            |

如果真落地短视频，比较稳的姿势不是“一把替换双塔”，而是把 TIGER 作为一路召回源，先控制 quota，在粗排/精排后看贡献、互补性和线上收益。

