# OneRec 生成式推荐资料整理

> 说明：本文只基于公开一手资料整理，包括 OneRec 论文、OneRec Technical Report、OneRec-V2 Technical Report 和 Kuaishou-OneRec/OpenOneRec 官方仓库。公开资料中有些系统细节没有披露，本文不会把未披露内容当作事实；必要处会明确写“论文未披露”或“工程上可这样理解”。

## 1. 先给结论

OneRec 的核心思想是把传统“召回 -> 粗排 -> 精排”的级联推荐系统，改造成一个端到端的生成式推荐模型：模型不再只给候选打分，而是直接生成 item 的语义 ID，再把语义 ID 映射回真实视频。

它的关键组件可以概括为：

1. **Semantic ID / Item Tokenizer**：把海量视频 item 映射成可生成的离散 token。公开资料中 OneRec 使用 coarse-to-fine semantic IDs，Technical Report 中一个目标视频对应 3 个 semantic IDs。
2. **用户行为编码 / 上下文建模**：输入用户静态特征、短期行为、正反馈行为、长期行为等多尺度用户行为表示。
3. **生成式模型**：V1/Technical Report 主要描述 encoder-decoder；V2 改为 lazy decoder-only 架构，以减少大量上下文编码成本。
4. **预训练**：用 next-token prediction 预测目标视频的 semantic IDs。
5. **后训练**：用实时曝光流做 RSFT/SFT，让模型跟上用户兴趣变化；同时用 RL 或偏好优化进一步对齐真实业务目标。
6. **RL / 偏好对齐**：V1 主要是 reward model + DPO/ECPO；V2 进一步引入真实用户反馈信号和 GBPO，减少 reward hacking 风险。

一句话：

```text
OneRec = item 语义 token 化 + 用户行为序列建模 + 生成 semantic IDs + reward / feedback 对齐
```

## 2. 公开资料的版本关系

公开资料中至少有四个相关但不同的版本。整理 OneRec 时必须区分，否则很容易混淆。

| 资料 | 重点 | 与训练流程相关的关键信息 |
|---|---|---|
| OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment, 2025-02 | 早期 OneRec 论文 | session-wise list generation、encoder-decoder、MoE、reward model、iterative DPO |
| OneRec Technical Report, 2025-06 | 工业生产版技术报告 | tokenizer、多尺度用户行为、预训练样本、RSFT + RL 后训练、ECPO、线上部署和 MFU |
| OneRec-V2 Technical Report, 2025-08 | V2 架构和 RL 改进 | lazy decoder-only、真实用户反馈 RL、duration-aware reward、GBPO |
| OpenOneRec 官方仓库 / Technical Report, 2026 | 开源 foundation model 和 benchmark | Qwen3 backbone、Itemic Tokens、pretraining + SFT + distillation + RL 的开源训练范式 |

因此本文会把“工业 OneRec V1/Technical Report”的主线讲清楚，再单独说明 V2 和 OpenOneRec 的变化。

## 3. OneRec 解决什么问题

传统大规模推荐系统通常是级联架构：

```text
召回 -> 粗排 -> 精排 -> 重排
```

这种架构的问题是：

- 每一层各自优化，目标不完全一致。
- 后一层上限受前一层候选质量限制。
- 计算资源分散在多个模型和多个服务里。
- 很难直接用 RL 优化最终推荐结果，因为 action space 和链路太复杂。

OneRec 的目标是把推荐改造成端到端生成：

```text
用户上下文 -> 生成 item semantic IDs -> 映射回真实 item
```

它不是“用 LLM 写推荐理由”，而是把推荐结果本身作为 token 序列生成。

## 4. Item 如何变成可生成 token

### 4.1 为什么不能直接生成 item id

工业推荐里的 item 空间可能是亿级，而且持续增长。直接把每个 item id 当作一个 token 会带来问题：

- 词表巨大，softmax 成本高。
- 新 item 冷启动困难。
- item id 本身没有语义，无法在相似 item 之间共享知识。
- 生成非法 item 或长尾 item 的泛化能力弱。

所以 OneRec 使用 semantic ID：把 item 变成多层离散码。

### 4.2 OneRec 的 Semantic ID 思路

OneRec Technical Report 描述的 tokenizer 大致是：

1. 先构造视频的多模态表示。
2. 在表示学习中融合协同信号。
3. 用 RQ-Kmeans 生成 coarse-to-fine 层级 semantic IDs。
4. 训练时让生成模型预测这些 semantic IDs。

公开资料中提到的视频多模态输入包括：

- caption
- tag
- ASR
- OCR
- cover image
- 5 个均匀采样的视频帧

这些多模态输入会经过视觉/语言模型和 QFormer 压缩成 item 表示。

### 4.3 协同信号如何进入 tokenizer

OneRec Technical Report 明确强调，只用内容特征生成 semantic IDs 不够好，因为推荐中 item 相似性不仅来自内容，还来自用户行为共现。

因此它构造高质量 item-pair 数据：

- **User-to-Item Retrieval**：对用户的正反馈目标 item，从用户最近历史正反馈中找协同相似 item 组成 pair。
- **Item-to-Item Retrieval**：用 Swing similarity 等 item 相似度构造 pair。

训练表示时使用两个目标：

- item-to-item contrastive loss：让协同相似 item 表示更接近。
- caption loss：保留内容理解能力，减少表示完全被协同信号带偏。

这样得到的是 collaborative-aware multimodal representation。

### 4.4 RQ-Kmeans / Coarse-to-fine Semantic IDs

OneRec Technical Report 使用 RQ-Kmeans 做 tokenization。可以这样理解：

```text
item embedding
-> 第 1 层聚类得到 coarse semantic id
-> 计算残差
-> 第 2 层对残差聚类
-> 再计算残差
-> 第 3 层对残差聚类
-> 得到 [sid_1, sid_2, sid_3]
```

Technical Report 明确说每个 target item 被 tokenized 成 3 个 semantic identifiers，因此一个训练样本对应 3 个 decoder target tokens。

注意：早期 OneRec 论文还提到 balanced K-means / multi-level balanced quantization，用来缓解 code distribution 不均衡的问题；Technical Report 后续强调 RQ-Kmeans 在 reconstruction quality、codebook utilization 和 balance 上优于常见 RQ-VAE。这里不能简单说“所有版本都完全相同”，只能说公开资料中 OneRec 系列围绕 RQ-Kmeans / balanced semantic ID 做了演进。

## 5. 样本构造

### 5.1 V1 早期论文：session-wise 样本

2025-02 的 OneRec 论文提出 session-wise list generation，而不是单点 next-item prediction。

它把一次请求返回的一批短视频看作一个 session，通常包含 5 到 10 个视频。模型目标不是只预测下一个视频，而是生成一个高价值 session 列表。

论文中用于识别高质量 session 的条件包括：

- 用户在 session 内实际观看的视频数量大于等于 5。
- 用户观看 session 的总时长超过阈值。
- 用户出现点赞、收藏、分享等交互行为。

这个版本的训练样本可以抽象成：

```text
输入：
用户正向历史行为序列

输出：
高质量 session 中多个视频的 semantic IDs 序列
```

如果 session 里有 K 个视频，每个视频有 3 层 semantic IDs，那么目标序列大致是：

```text
sid(v1)_1, sid(v1)_2, sid(v1)_3,
sid(v2)_1, sid(v2)_2, sid(v2)_3,
...
sid(vK)_1, sid(vK)_2, sid(vK)_3
```

这就是它区别于 point-wise next item prediction 的地方：它试图学习一个列表内部 item 之间的相关性、顺序关系、多样性和整体质量。

### 5.2 Technical Report：target item 样本

2025-06 OneRec Technical Report 中，预训练数据描述更偏 target item prediction：

```text
输入：
multi-scale user behavior representations

输出：
target item 的 3 个 semantic IDs
```

公开报告明确说：

- 模型输入是多尺度用户行为表示。
- 预训练目标是预测用户的 target item 序列。
- 每个 training sample 包含一个 target item。
- target item 被 tokenized 成 3 个 semantic identifiers。
- 因此每个样本对应 3 个 next-token prediction target tokens。

这和早期论文的 session-wise 叙述存在粒度差异。合理理解是：OneRec 系列在公开资料中经历了从 session-wise/list-wise 表达到更工程化 point-wise target item 训练描述的演进；不要把两个版本强行合并成同一套样本格式。

### 5.3 用户侧输入特征

OneRec Technical Report 把用户行为建模拆成四类 pathway：

1. **User Static Pathway**
   - uid
   - age
   - gender
   - 其他用户静态特征

2. **Short-term Pathway**
   - 最近用户交互序列
   - vid 或 sid
   - author id
   - tag
   - timestamp
   - playtime
   - duration
   - label，例如 like、follow、forward、dislike、comment、profile entry 等

3. **Positive-feedback Pathway**
   - 高 engagement 行为序列
   - 用来建模更稳定、更强的兴趣偏好

4. **Lifelong Pathway**
   - 超长历史行为，报告中提到可达 100,000 videos
   - 不能直接全量 attention，所以用两阶段 hierarchical compression
   - 通过聚类压缩长期历史，选 cluster center 附近 item 做代表

这些 pathway 最终整合成用户多尺度表示，送入 Transformer encoder 或 V2 的 context processor。

## 6. 模型结构

### 6.1 OneRec V1 / Technical Report：encoder-decoder

OneRec 早期和 Technical Report 中的主结构是 encoder-decoder。

```text
Encoder:
  输入用户历史行为、多尺度用户特征
  输出用户兴趣表示

Decoder:
  输入 BOS + target semantic IDs 前缀
  自回归预测下一个 semantic ID
```

训练时使用 teacher forcing：

```text
decoder input:  BOS, sid_1, sid_2
decoder target: sid_1, sid_2, sid_3
```

目标函数是 semantic IDs 的 cross entropy / next-token prediction loss。

OneRec 还在 decoder 或 encoder/decoder 中使用 sparse MoE，以提升模型容量，同时避免 FLOPs 按参数量等比例增长。

Technical Report 给出的模型规模包括：

- OneRec-0.015B Dense
- OneRec-0.121B Dense
- OneRec-0.935B MoE
- OneRec-2.633B MoE

其中 0.935B MoE 版本在 decoder 使用 24 个专家、每次激活 2 个专家；2.633B MoE 版本在 encoder 和 decoder 中使用 MoE，每次激活 4 个专家。

### 6.2 OneRec-V2：lazy decoder-only

OneRec-V2 认为 V1 的 encoder-decoder 仍有计算效率问题：大部分 FLOPs 用在 context encoding，而不是真正参与 loss 的 target decoding。

V2 改为 lazy decoder-only，核心变化是：

- 用 context processor 处理用户上下文。
- 用 lazy cross-attention 让 decoder 访问上下文。
- 多层 decoder 共享 key-value 表示。
- 使用 Grouped Query Attention 降低上下文表示的内存和计算开销。
- 目标仍然是生成 semantic IDs。

V2 报告称 lazy decoder-only 能在相近 loss 下显著降低 FLOPs 和 activation memory，并支持更大模型扩展，例如 8B dense 和 4B MoE。

## 7. 预训练怎么做

### 7.1 预训练目标

OneRec 的预训练本质是推荐域 next-token prediction：

```text
maximize log P(sid_1, sid_2, sid_3 | user_context)
```

或者展开成：

```text
P(sid_1 | user_context)
P(sid_2 | user_context, sid_1)
P(sid_3 | user_context, sid_1, sid_2)
```

如果是早期 session-wise 版本，则目标序列是一个 session 内多个视频的 semantic IDs。

### 7.2 预训练数据规模

OneRec Technical Report 给出了一些工业规模信息：

- 训练 pipeline 每天处理约 18B samples。
- 每个 sample 产生 3 个 decoder target tokens。
- 因此吞吐约 54B tokens/day。
- OneRec-0.935B 模型约在 100B samples，即 300B tokens 后收敛。

这里的 token 是推荐 semantic ID token，不是自然语言 token。

### 7.3 预训练学习到什么

预训练让模型学到：

- 用户历史行为到目标 item 的映射。
- semantic ID 空间中的 item 生成规律。
- 用户短期兴趣、长期兴趣、正反馈兴趣之间的组合。
- item 内容语义和协同行为的联合结构。

但 Technical Report 也指出一个关键问题：预训练只是在拟合历史曝光 item 分布，而这些曝光 item 来自过去传统推荐系统。因此模型可能被旧系统上限限制住，无法天然突破传统 pipeline。

这就是后面要做 reward / preference alignment / RL 的原因。

## 8. 后训练：SFT / RSFT 怎么做

公开资料中，OneRec Technical Report 把 post-training 描述为实时数据流上的在线训练，同时做 RSFT 和 RL。

### 8.1 RSFT 是什么

RSFT 即 Reject Sampling Fine-Tuning。它不是从所有曝光样本继续训练，而是先根据反馈质量过滤样本。

Technical Report 中的做法是：

- 使用 real-time data streams 做在线训练。
- 对 RSFT，基于 play duration 过滤掉 bottom 50% 的曝光 session。
- 训练 loss 和 pre-training 相同，仍然是 next-token prediction / cross entropy。
- 同时对 sparse parameters 和 dense parameters 降低学习率，即 annealing。

可以理解为：

```text
保留更高质量曝光样本
-> 继续用监督式 next-token prediction 训练
-> 让模型跟上实时兴趣，同时减少低质曝光对模型的污染
```

### 8.2 RSFT 和普通 SFT 的区别

普通 SFT：

```text
用构造好的 target item / target list 做监督学习
```

RSFT：

```text
先用 reward / play duration / 规则筛掉低质量样本
再对剩下的高质量曝光做监督学习
```

所以 RSFT 更像“带样本筛选的 SFT”。

### 8.3 V2 中的 SFT

OneRec-V2 报告说，SFT 阶段和 V1 类似：用 streaming exposure data 做 online loss training，loss 和 pretraining 一致。主要目的有两个：

- 捕捉用户实时兴趣变化。
- 防止模型偏离 pretrained model 太远。

这句话很重要：OneRec 的后训练不是只有 RL，它同时保留监督 loss 来稳定模型。

## 9. Reward Model 和偏好对齐

### 9.1 为什么需要 reward

仅靠 SFT/RSFT，模型学到的是历史曝光分布，而历史曝光来自旧系统。它可能只是复刻旧推荐系统。

RL / preference alignment 的目标是：

```text
让模型在自己生成的 item 空间里优化用户偏好和业务目标
```

### 9.2 V1 早期论文：Reward Model + Iterative DPO

2025-02 OneRec 论文提出 Iterative Preference Alignment。

流程大致是：

1. 先用 session-wise generation 训练出 seed model。
2. 训练一个 session-wise reward model。
3. 对同一个用户，用当前 OneRec 通过 beam search 生成多个 responses / sessions。
4. 用 reward model 给每个 response 打分。
5. 选最高分作为 winner，最低分作为 loser。
6. 构造 preference pair：

```text
(user_context, winner_session, loser_session)
```

7. 用 DPO 类 loss 更新模型。
8. 迭代：新模型继续生成新 preference data，再继续优化。

注意：这里的偏好对不是人工标注的，而是 reward model 自动选出来的。

### 9.3 Reward Model 怎么建

早期论文中的 session-wise reward model：

- 先提取每个 item 的 target-aware representation，例如对用户行为做 target attention。
- session 内 item 之间通过 self-attention 交互，融合列表内信息。
- 用不同 tower 预测多目标 reward。
- 用 recommendation data 和 ground-truth labels，通过 binary cross entropy 训练。

这说明它不是一个简单的单 item CTR 模型，而是能评价一个 session/list 的模型。

## 10. RL 怎么做

### 10.1 Technical Report：Reward System

OneRec Technical Report 把 reward system 分成三类：

1. **Preference Reward**
   - 也叫 P-Score。
   - 用来对齐用户偏好。
   - 来自多目标推荐指标融合。

2. **Format Reward**
   - 用来约束生成格式合法性。
   - 生成式推荐中 semantic ID 序列不一定能映射到真实 item，所以 legality ratio 很重要。

3. **Industrial Reward**
   - 用来对齐具体工业业务需求。
   - 例如社区生态、商业化、冷启动、长尾内容等。

这比单纯“点击率 reward”复杂得多。

### 10.2 ECPO：Early Clipped GRPO

Technical Report 中使用 ECPO 优化 preference reward。

它基于 GRPO，但做了修改：原始 GRPO 中负 advantage 的大 policy ratio 容易导致梯度爆炸，因此 ECPO 对大 ratio 提前裁剪，以提升稳定性。

报告还提到 OneRec 中去掉 KL divergence loss，因为 RL 和 SFT 同时训练，SFT loss 本身起到稳定模型、避免偏离过远的作用。

这里可以用工程语言理解：

```text
RL 负责把模型往高 reward 方向推
SFT / RSFT 负责把模型拉回合法、稳定、贴近真实曝光分布的区域
```

### 10.3 后训练中的 RL 样本生成

Technical Report 描述的工业训练流程：

- 从 RSFT 数据中随机选择 1% 用户生成 RL samples。
- 使用外部 inference service 解耦 RL sample generation 和 training。
- 这 1% 用户访问外部服务，生成 512 个 items。
- 对每个 item 请求 reward model 打分。
- 数据返回训练任务。
- 训练任务每 1000 steps 通过 MQ 把更新后的参数发送到外部 inference service。

这说明 OneRec 的 RL 是高度工程化的 on-policy / near-on-policy 训练，不是简单离线拿历史日志跑个 DPO。

## 11. OneRec-V2 的 RL 改进

OneRec-V2 报告认为 V1 主要依赖 reward model，存在两个问题：

- 受资源限制，V1 只能对少量用户做 on-policy rollout。
- reward model 存在 reward hacking 风险。

随着 OneRec 流量占比提升，V2 可以拿到更多 OneRec 自己生成结果的真实用户反馈。因此 V2 更强调 real-world user interactions。

### 11.1 Duration-aware feedback signals

V2 构造 duration-aware reward score，使用真实用户播放反馈来定义正负样本。

报告中披露的样本筛选方式：

- 按 duration-aware reward score 排序。
- 选 top 25% 作为高质量正样本。
- 对显式负反馈，例如 dislike，设为负样本。
- 其他样本过滤掉。
- advantage 不做 normalization，因为正负定义已经足够严格；进一步归一化可能引入不一致。

这与 V1 的 reward model 打分不同：V2 更直接使用真实用户行为反馈。

### 11.2 GBPO：Gradient-Bounded Policy Optimization

V2 提出 GBPO，用来替代或改进 PPO/GRPO/ECPO 这类 clipping-based 方法。

GBPO 的两个目标：

- **Full Sample Utilization**：保留所有样本梯度，鼓励更充分探索。
- **Bounded Gradient Stabilization**：用 BCE loss 的梯度界约束 RL 梯度，提升训练稳定性。

V2 报告认为传统 clipping 方法会丢掉一些样本，负样本上 policy ratio 过大又容易导致梯度爆炸；GBPO 试图同时保留样本和控制梯度。

### 11.3 V2 的关键结论

V2 的实验结论包括：

- 引入 OneRec 自己生成样本的用户反馈，可以带来自迭代优化。
- 用户真实反馈可以缓解 reward hacking。
- reward model 和 user feedback 都有效，但偏好的目标不同：reward model 更偏多目标融合，真实播放反馈更偏 App Stay Time。

这里不能简单说“V2 不用 reward model”。公开资料中 V2 仍比较了 reward model、user feedback、hybrid 三组，只是强调真实用户反馈的重要性。

## 12. OpenOneRec 与工业 OneRec 的区别

OpenOneRec 是 2026 年官方开源的 foundation model / benchmark 体系，不等同于快手线上工业 OneRec V1/V2。

OpenOneRec 的公开 README 显示：

- 使用 Qwen3 backbone。
- 构造 RecIF-Bench，包含短视频、广告、商品三个域。
- 使用 Itemic Tokens。
- 训练流程包括：
  - Pre-training：Itemic-Text Alignment 和 Full-Parameter Co-Pretraining。
  - Post-training Stage 1：multi-task SFT。
  - Post-training Stage 2：on-policy distillation。
  - Post-training Stage 3：RL。

因此，OpenOneRec 更像把 OneRec 思想迁移到“推荐 foundation model + instruction following + reasoning”的开源体系中。它有助于复现和研究，但不要把 OpenOneRec 的 Qwen3、instruction-following、reasoning 细节直接套到 2025 工业 OneRec V1 上。

## 13. OneRec 与传统推荐系统的关系

OneRec 不是简单替代某个召回模型，而是试图统一召回和排序。

传统系统：

```text
召回：从亿级 item 找几千个
粗排：从几千个筛几百个
精排：对几百个精细打分
重排：多样性、规则、业务约束
```

OneRec：

```text
输入用户上下文
-> 生成 semantic ID
-> semantic ID 映射 item
-> 可选 reward-based selection / filtering
```

但是工程上它并不意味着所有规则都消失。Technical Report 仍然提到 format reward、industrial reward、线上业务目标和约束。工业系统一定还会有合法性过滤、安全策略、商业化策略、冷启策略等。

## 14. 常见误解和纠正

### 14.1 误解：OneRec 就是一个 LLM

不准确。

工业 OneRec 的主线是 generative recommender，它生成的是 item semantic IDs。虽然借鉴了 LLM 的 next-token prediction、MoE、scaling law、RL 等技术，但它不是简单拿一个通用 LLM 来聊天式推荐。

OpenOneRec 则更接近 LLM + recommendation foundation model，但那是后续开源体系。

### 14.2 误解：OneRec 只做召回

不准确。

论文明确目标是 unified retrieve and rank。早期 generative retrieval 可能只用于召回，但 OneRec 的目标是替代级联框架中的多阶段选择，直接生成高质量推荐结果。

### 14.3 误解：OneRec 的 RL 就是 DPO

不完整。

早期论文重点是 Iterative Preference Alignment + DPO。Technical Report 则讲 reward system + ECPO。V2 又提出 user feedback signals + GBPO。因此 OneRec 系列的 RL/偏好优化是不断演进的，不能只等同于 DPO。

### 14.4 误解：SFT 后就够了

不够。

Technical Report 明确指出，预训练/监督训练拟合的是历史曝光分布，而曝光来自传统推荐系统，所以模型可能被旧系统上限限制。RL / reward alignment 的意义是让模型在生成 item 空间中优化更细粒度偏好和业务目标。

### 14.5 误解：RL 越强越好

不一定。

生成式推荐有非法 semantic ID、reward hacking、梯度爆炸、目标跷跷板、线上体验风险。OneRec 的实践里同时使用 SFT/RSFT loss、format reward、industrial reward、ECPO/GBPO 等稳定机制。

## 15. 如果自己要复现一个简化版 OneRec

可以按这个最小闭环做：

### 15.1 Tokenizer

1. 收集 item 内容特征和行为共现特征。
2. 训练 item representation。
3. 用 RQ-Kmeans 或 RQ-VAE 得到 3 层 semantic IDs。
4. 建立：

```text
item_id -> [sid_1, sid_2, sid_3]
[sid_1, sid_2, sid_3] -> item_id
```

需要注意 illegal SID 组合问题：生成出来的 SID 序列未必能映射真实 item。

### 15.2 SFT / Pretrain 数据

构造样本：

```text
用户历史行为 + 用户画像 + 场景上下文 -> target item semantic IDs
```

正样本可以来自：

- 点击且有效播放。
- 高观看时长。
- 点赞、收藏、评论、关注。
- 转化或下单。
- 多目标综合价值高的曝光。

负反馈可用于后续 reward/RL，不一定直接放进普通 next-token SFT。

### 15.3 模型

小规模可以先用 encoder-decoder：

```text
Encoder: 用户历史 item/sid 序列 + 特征
Decoder: BOS + sid prefix -> next sid
Loss: CE(sid target)
```

工业级再考虑：

- MoE
- long sequence compression
- semantic ID input
- lazy decoder-only
- 多机训练、embedding cache、参数服务器

### 15.4 后训练

先做 RSFT：

```text
过滤低质量曝光
保留高时长/高交互样本
继续 CE 训练
```

再做 RL / preference optimization：

```text
对同一 user context 生成多个 candidate items/lists
用 reward model 或真实反馈打分
构造 winner/loser 或 advantage
用 DPO/GRPO/ECPO/GBPO 类方法优化
同时保留 SFT loss 稳定模型
```

### 15.5 评估

离线评估：

- CE loss
- Recall@K / HitRate@K / NDCG@K
- legality ratio
- 多样性
- item 覆盖率
- reward model score

线上评估：

- watch time
- app stay time
- view through rate
- like/follow/comment/share
- 负反馈
- 留存
- 商业化目标

## 16. 面试表达模板

如果面试官问：“OneRec 的生成式推荐是怎么做的？”

可以这样回答：

> OneRec 的核心是把推荐结果转成 semantic ID 序列生成。它先用多模态内容和协同信号构造 item 表示，再通过 RQ-Kmeans 生成 coarse-to-fine semantic IDs。模型输入用户多尺度行为，包括短期行为、正反馈行为和长期行为压缩表示；输出目标 item 的 semantic IDs。预训练阶段用 next-token prediction 的 cross entropy 学习从用户上下文到 semantic IDs 的生成。后训练阶段在实时曝光流上做 RSFT/SFT，过滤低质量曝光并继续监督训练，同时引入 reward system 做 RL 或偏好对齐，以突破只拟合历史曝光分布的限制。V1 主要依赖 reward model、DPO/ECPO，V2 进一步利用真实用户反馈和 GBPO，降低 reward hacking 风险并提升在线指标。

如果想结合工程落地，可以补充：

> 我理解它真正难点不只是模型结构，而是 item tokenization、合法 SID 映射、超长行为压缩、reward 设计、RL 稳定性和线上低延迟服务。尤其在推荐里，SFT 学的是历史曝光分布，RL 才是把模型推向最终 watch time、stay time、互动、负反馈和生态目标的关键阶段，但 RL 必须和 SFT loss、format reward、业务规则一起用，避免非法生成和 reward hacking。

## 17. 参考资料

1. OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment  
   https://arxiv.org/abs/2502.18965

2. OneRec Technical Report  
   https://arxiv.org/abs/2506.13695

3. OneRec-V2 Technical Report  
   https://arxiv.org/abs/2508.20900

4. Kuaishou-OneRec/OpenOneRec 官方仓库  
   https://github.com/Kuaishou-OneRec/OpenOneRec

