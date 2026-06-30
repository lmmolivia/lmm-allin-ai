# OneRec Technical Report 中文翻译

> 原文：OneRec Team, **OneRec Technical Report**, arXiv:2506.13695v4, 2025-09-16。  
> 本文基于仓库内本地文件 `reference/onerec_technical_report_2506.13695.pdf` 翻译整理。为便于推荐算法工程阅读，公式和核心符号保留原文写法，图表以中文说明和关键数值呈现。  
> 说明：这是学习用途的中文译文/精译稿，不替代原论文。原文版权声明为 © 2025 Kuaishou. All rights reserved.

## 摘要

推荐系统已经从简单的启发式规则，演进到复杂的深度学习模型。然而，与人工智能技术的快速发展相比，推荐系统仍然受到传统多阶段级联架构的严重约束。本文介绍 **OneRec**，这是一个端到端生成式推荐框架。OneRec 将现有推荐系统中普遍存在的检索、粗排、精排等级联流程整合到统一的生成模型中。

OneRec 的模型计算量达到当前推荐模型的 10 倍，并在一定边界内观察到了推荐系统中的 scaling laws。同时，强化学习在该框架下表现出显著潜力。OneRec 配套建设了训练和推理基础设施，在旗舰 GPU 上分别达到 23.7% 的训练 MFU 和 28.8% 的推理 MFU；其 OPEX 约为传统推荐流水线的 10.6%。

OneRec 已部署在快手和快手极速版的主场景中，承接总 QPS 的 25%，将 App Stay Time 分别提升 0.54% 和 1.24%，并且正向影响 LT7 等留存指标。论文认为，OneRec 代表了一种推荐系统未来发展的基础性范式。

## 目录

1. 引言  
2. 架构  
   2.1 Tokenizer  
   2.2 Encoder  
   2.3 Decoder  
   2.4 Reward System  
3. 训练框架  
   3.1 训练基础设施  
   3.2 预训练  
   3.3 后训练  
4. 评估  
   4.1 评估指标  
   4.2 Scaling  
   4.3 强化学习  
   4.4 Tokenizer  
   4.5 在线 A/B 测试  
5. 结论、局限与未来方向  
附录 A 贡献  
附录 B 在线 A/B 测试实现细节  
附录 C Tokenization 案例分析  
附录 D 符号表

## 1. 引言

推荐系统是现代互联网平台的核心技术之一。过去二十年中，推荐技术经历了从启发式规则到基于深度学习的复杂模型的演化。尽管深度学习推动推荐系统取得显著进展，但与大语言模型、多模态模型等 AI 方向相比，推荐系统的模型规模、训练范式和系统架构仍然受到较强限制。

工业界主流推荐系统通常采用多阶段级联架构：召回阶段从海量候选中筛出较小候选集，粗排和精排阶段逐级建模和排序，最后再通过重排、规则、约束和业务策略生成最终展示列表。这种架构在过去具有很强工程可行性，但也带来了若干根本瓶颈。

### 1.1 传统级联推荐系统的问题

**第一，计算碎片化。**  
在快手的传统推荐系统中，超过 50% 的在线服务资源消耗在通信和存储访问上，而不是高精度计算本身。排名模型的训练和推理 MFU 分别只有 4.6% 和 11.2%。作为对比，大语言模型在 H100 等 GPU 上的 MFU 往往可达约 40%。同时，推荐系统在线服务通常需要支持超过 400K QPS 和小于 500ms 的延迟，这进一步限制了模型规模扩展。

**第二，目标冲突。**  
推荐系统需要同时优化用户、创作者、平台生态和商业化等多类目标。在快手场景中，需要关注的目标数量可以达到数百个。这些目标会在召回、粗排、精排、重排等不同阶段被反复干预，导致跨阶段建模和目标优化存在冲突。

**第三，难以跟上 AI 技术演进。**  
传统级联推荐架构很难自然接入 scaling laws、强化学习、统一生成建模等现代 AI 技术。每个阶段都需要独立训练、独立部署和独立优化，系统复杂度不断膨胀。

### 1.2 OneRec 的核心思想

OneRec 的目标是把推荐系统从多阶段级联架构推进到端到端生成式架构。它将检索、排序、重排等流程统一为一个 encoder-decoder 生成模型。模型输入用户历史行为和上下文特征，输出目标视频的语义 ID 序列，再通过语义 ID 映射到具体视频。

OneRec 具备两个重要特征：

**端到端优化。**  
OneRec 不再把召回、排序、重排视为割裂模块，而是将它们整合进统一目标中训练，从而缓解级联阶段之间的目标错配。

**计算效率。**  
OneRec 将传统系统中的大量通信、存储访问和多阶段服务调用，压缩为以 GPU 高密度计算为核心的生成推理流程，使计算更接近现代 AI 基础设施的高利用率范式。

## 2. 架构

OneRec 由四个核心模块组成：

| 模块          | 作用                                                                  |
|---------------|-----------------------------------------------------------------------|
| Tokenizer     | 将视频映射为多层 coarse-to-fine semantic IDs。                         |
| Encoder       | 编码用户静态特征、短期行为、正反馈行为和全生命周期行为。              |
| Decoder       | 基于 encoder 输出逐 token 生成目标视频 semantic IDs。                  |
| Reward System | 通过偏好奖励、格式奖励和工业场景奖励，对生成结果进行强化学习对齐。    |

训练阶段，OneRec 以 next-token prediction 的方式预测目标 item 的 semantic ID，同时通过 reward system 做偏好对齐。推理阶段，模型生成 semantic ID 序列，再将其映射回真实视频；在部分在线配置中，还会通过 reward model 对生成结果进行选择。

## 2.1 Tokenizer

推荐系统面对的是海量 item。如果直接把每个视频都作为独立 token，会导致词表极大、参数量膨胀、新视频泛化困难。OneRec 采用 semantic ID 的方式，将视频压缩为由多个离散 token 组成的序列。

每个视频最终表示为：

$$
\{s_m^1, s_m^2, \ldots, s_m^{L_t}\}
$$

其中，$L_t=3$ 表示 semantic ID 的层数。每一层 token 表示从粗到细的语义划分。

### 2.1.1 对齐协同信号的多模态表征

已有生成式推荐方法通常只基于内容特征生成 semantic IDs，容易忽略推荐系统中非常关键的协同过滤信号。OneRec 在多模态内容表征中引入协同信号，使 tokenizer 既理解视频内容，也能捕捉用户行为中体现的 item 相似性。

视频的多模态输入包括：

| 输入类型 | 内容                                      |
|----------|-------------------------------------------|
| 文本     | caption、tag、ASR、OCR                    |
| 图像     | cover image                               |
| 视频帧   | 均匀采样的 5 帧                            |

论文使用 miniCPM-V-8B 提取多模态 token 表征：

$$
M \in \mathbb{R}^{N_M \times d_t}
$$

其中 $N_M=1280$，$d_t=512$。

为了压缩多模态 token，OneRec 使用 QFormer。给定 $\tilde{N}_M=4$ 个可学习 query token：

$$
Q^{(1)} \in \mathbb{R}^{\tilde{N}_M \times d_t}
$$

QFormer 的每层计算为：

$$
Q^{(i+1)} = \mathrm{CrossAttn}(Q^{(i)}, M, M)
$$

$$
Q^{(i+1)} = \mathrm{FFN}(\mathrm{RMSNorm}(Q^{(i+1)}))
$$

其中 $i \in \{1,\ldots,N_c\}$，$N_c=4$。最终压缩表征为：

$$
\tilde{M}=Q^{(N_c+1)}
$$

#### 协同相似样本构造

论文构造 item pair 数据集 $D_{pair}$，用来把协同过滤信号注入多模态表征。pair 来源包括两类：

| 来源                  | 构造方式                                                                 |
|-----------------------|--------------------------------------------------------------------------|
| User-to-Item Retrieval | 对用户正反馈目标视频，选取该用户近期正反馈历史中协同最相似的视频。        |
| Item-to-Item Retrieval | 使用 Swing 等 item 相似度方法，选取高相似 item pair。                    |

#### 训练目标

OneRec 使用两类目标训练 tokenizer 表征：

**Item-to-item contrastive loss：**

$$
\mathcal{L}_{I2I}
=-\frac{1}{|\mathcal{B}|}
\sum_{(i,j)\in\mathcal{B}}
\log
\frac{
\exp(\mathrm{sim}(\tilde{M}_i,\tilde{M}_j)/\tau)
}{
\sum_{(i',j')\in\mathcal{B}}
\exp(\mathrm{sim}(\tilde{M}_i,\tilde{M}_{j'})/\tau)
}
$$

这个目标让协同相似的视频在多模态表示空间中更接近。

**Caption generation loss：**

$$
\mathcal{L}_{caption\_gen}
=-\sum_k \log P(t_{k+1}\mid [t_1,\ldots,t_k])
$$

该目标使用 LLaMA3 decoder 保留内容理解能力，降低表征只拟合协同信号而丢失视频语义的风险。

### 2.1.2 Tokenization

在得到对齐协同信号的多模态表示后，OneRec 使用 **RQ-Kmeans** 生成 semantic IDs。RQ 表示 residual quantization，即逐层量化残差。

第一层残差初始化为：

$$
R^{(1)}=\{\tilde{M}_i \in \mathbb{R}^{\tilde{N}_M \times d_t} \mid \forall \text{ video } i\}
$$

第 $l$ 层 codebook 由 K-means 得到：

$$
C^{(l)}=\mathrm{K\text{-}means}(R^{(l)}, N_t)
$$

$$
C^{(l)}
=\{c_k^{(l)}\in\mathbb{R}^{\tilde{N}_M\times d_t}\mid k=1,\ldots,N_t\}
$$

第 $i$ 个视频在第 $l$ 层的 semantic ID 为最近 centroid 的索引：

$$
s_i^l=\arg\min_k \|R_i^{(l)}-c_k^{(l)}\|
$$

残差更新为：

$$
R_i^{(l+1)}
=R_i^{(l)}-c_{s_i^l}^{(l)}
$$

论文中 $L_t=3$。最终每个视频被表示为 3 个 semantic IDs：

$$
\{s_m^1,s_m^2,s_m^3\}
$$

论文实验显示，与 RQ-VAE 相比，RQ-Kmeans 在重构质量、codebook 利用率和 token 分布均衡性上表现更好。

## 2.2 Encoder

Encoder 的目标是压缩并融合用户不同时间尺度的兴趣。OneRec 将用户侧输入分为四路：

| 输入路径          | 含义                                               |
|-------------------|----------------------------------------------------|
| User Static        | 用户静态属性，例如 uid、age、gender。               |
| Short-term         | 用户最近的短期行为序列。                            |
| Positive-feedback  | 用户较长窗口内的正反馈行为序列。                    |
| Lifelong           | 用户全生命周期行为，最多可达 100,000 个视频。        |

### 2.2.1 输入表示

#### 用户静态特征

用户静态特征由多个 embedding 拼接得到：

$$
f_u=[e_{uid};e_{gender};e_{age};\ldots]
$$

然后通过两层 MLP 投影到模型维度：

$$
h_u=\mathrm{Dense}(\mathrm{LeakyReLU}(\mathrm{Dense}(f_u)))
$$

其中各类静态特征 embedding 维度为 64，最终：

$$
h_u\in\mathbb{R}^{1\times d_{model}}
$$

#### 短期行为序列

短期行为序列长度为 $L_s=20$，表示用户最近的交互。每个行为包含视频标识、作者、标签、时间戳、播放时长、视频时长和交互标签等特征。原文特别说明，这里的视频标识可以用传统 video identifier `vid` 表示，也可以用 2.1.2 中的 semantic identifier `sid` 表示；两种输入表示会在 4.2.2 中对比：

$$
f_s=[e_{vid}^s;e_{aid}^s;e_{tag}^s;e_{ts}^s;e_{playtime}^s;e_{dur}^s;e_{label}^s]
$$

随后同样通过 MLP 投影：

$$
h_s=\mathrm{Dense}(\mathrm{LeakyReLU}(\mathrm{Dense}(f_s)))
$$

输出为：

$$
h_s\in\mathbb{R}^{L_s\times d_{model}}
$$

论文中视频 embedding 与 $d_{model}$ 对齐，作者 embedding 维度为 512，其余特征 embedding 维度为 128。

#### 正反馈行为序列

正反馈序列长度为 $L_p=256$，特征形式与短期行为类似。该序列聚焦用户有较强 engagement 的历史内容：

$$
h_p\in\mathbb{R}^{L_p\times d_{model}}
$$

#### 全生命周期行为序列

全生命周期序列最多包含 100,000 个视频，不能直接输入 Transformer。OneRec 使用两阶段层次压缩方法，思路受到 Twin V2 启发。

第一阶段是层次聚类压缩。每轮聚类会根据当前数据规模动态设定 cluster 数量，并在每个 cluster 中选择离中心最近的视频作为代表。聚类过程持续到 cluster 内 item 数量不超过阈值 $M$。

第二阶段是特征聚合。对于聚类后的代表行为：

| 特征类型     | 聚合方式                                      |
|--------------|-----------------------------------------------|
| 稀疏类别特征 | vid、aid、label 等继承代表视频的特征。          |
| 连续/多值特征 | tag、timestamp、playtime、duration 等做平均。 |

压缩后的长期序列长度为 $L_l=2000$，经 MLP 投影得到：

$$
v_l\in\mathbb{R}^{L_l\times d_{model}}
$$

随后使用 Lifelong QFormer 进一步压缩长期行为。初始化 $N_q=128$ 个可学习 query：

$$
h_l^{(0)}\in\mathbb{R}^{N_q\times d_{model}}
$$

每层计算为：

$$
h_l^{(i+1)}
=\mathrm{CrossAttn}(h_l^{(i)}, v_l, v_l)
$$

$$
h_l^{(i+1)}
=\mathrm{FFN}(\mathrm{RMSNorm}(h_l^{(i+1)}))
$$

论文中 Lifelong QFormer 层数为 $N_l=2$，最终：

$$
h_l\in\mathbb{R}^{N_q\times d_{model}}
$$

### 2.2.2 Encoder 结构

四路用户表示拼接后加位置编码：

$$
z^{(1)}
=[h_u;h_s;h_p;h_l]+e_{pos}
$$

其中：

$$
e_{pos}\in\mathbb{R}^{(1+L_s+L_p+N_q)\times d_{model}}
$$

Encoder 每层包含 self-attention 和 FFN：

$$
z^{(i+1)}
=z^{(i)}
+\mathrm{SelfAttn}(\mathrm{RMSNorm}(z^{(i)}))
$$

$$
z^{(i+1)}
=z^{(i+1)}
+\mathrm{FFN}(\mathrm{RMSNorm}(z^{(i+1)}))
$$

最终输出：

$$
z_{enc}
$$

它是用户多尺度行为的整体表示，会作为 decoder cross-attention 的 key/value。

## 2.3 Decoder

OneRec 的 decoder 使用 point-wise generation paradigm。也就是说，它不是一次生成整个推荐列表，而是对目标视频的 semantic ID 序列进行逐 token 预测。

对于目标视频 $m$，输入 token 序列为：

$$
S_m=\{s_{[BOS]},s_m^1,s_m^2,\ldots,s_m^{L_t}\}
$$

embedding lookup 后得到：

$$
d_m^{(0)}=\mathrm{Emb\_lookup}(S_m)
$$

Decoder 每层由三部分组成：

1. causal self-attention  
2. encoder-decoder cross-attention  
3. MoE feed-forward layer  

计算过程为：

$$
d_m^{(i+1)}
=d_m^{(i)}
+\mathrm{CausalSelfAttn}(d_m^{(i)})
$$

$$
d_m^{(i+1)}
=d_m^{(i+1)}
+\mathrm{CrossAttn}(d_m^{(i+1)},Z_{enc},Z_{enc})
$$

$$
d_m^{(i+1)}
=d_m^{(i+1)}
+\mathrm{MoE}(\mathrm{RMSNorm}(d_m^{(i+1)}))
$$

### MoE 层

MoE 层采用 top-k expert 组合：

$$
\mathrm{MoE}(x)
=\sum_{j=1}^{k}
\mathrm{Gate}_j(x)\cdot \mathrm{Expert}_j(x)
$$

论文采用 DeepSeek-V3 中的 loss-free load balancing 思路，避免额外负载均衡损失干扰主目标。

#### OneRec 里的 MoE 怎么设计

从 OneRec 原文可以确定的设计点如下：

| 设计点       | 原文信息                                                                 |
|--------------|--------------------------------------------------------------------------|
| 放置位置     | Decoder 公式中，每层 cross-attention 后接 MoE；模型表中 0.935B 为 Decoder MoE，2.633B 为 Encoder & Decoder MoE。 |
| 路由方式     | 使用 top-k routing，每个 token 只激活部分 experts。                       |
| expert 结构  | MoE 版本使用 SwiGLU FFN 作为 expert。                                      |
| expert 规模  | expert intermediate hidden size 为 $\frac{2}{3}\times 4d_{model}$，并对齐到 128 的整数倍。 |
| expert 数量  | 0.935B 使用 24 个 experts、激活 2 个；2.633B 使用 24 个 experts、激活 4 个。 |
| 均衡策略     | 使用 DeepSeek 的 loss-free load balancing。                                |

因此，对 decoder 中某个 token hidden state $x$，MoE 层可以理解为：

$$
\mathrm{MoE}(x)
=\sum_{j=1}^{k}
\mathrm{Gate}_j(x)\cdot \mathrm{Expert}_j(x)
$$

这里的 $x$ 是 decoder 中某个位置的 token 表示，例如 $[BOS]$ 或某一层 semantic ID token 的 hidden state。Router 会为这个 token 选择 top-k experts，只计算被选中的 experts 输出，再用 gate 权重加权求和。这样做的核心收益是：**总参数量可以变大，但每个 token 的实际计算量只增长到激活 experts 的规模**。

在推荐场景里，可以把不同 experts 粗略理解为学习不同用户兴趣、内容形态、场景分布或生成模式的子网络。但这只是工程解释，不是 OneRec 原文直接给出的 expert 语义。原文没有披露每个 expert 是否对应明确业务含义，也没有披露容量因子、token dropping、router 具体实现等细节。

#### 为什么需要负载均衡

MoE 的 router 如果不受约束，容易出现 **expert load imbalance**：大量 token 被路由到少数几个 experts，其他 experts 很少被训练。这样会带来两个问题：

| 问题             | 影响                                                                 |
|------------------|----------------------------------------------------------------------|
| routing collapse | 少数 experts 被过度使用，其他 experts 学不到东西，模型容量被浪费。    |
| 计算瓶颈         | 在 expert parallelism 下，高负载 expert 所在设备成为 straggler，拖慢训练或推理。 |

传统 MoE 常用一个额外的 auxiliary load balance loss 来鼓励均衡。例如 loss-free balancing 原论文中给出的典型形式是：

$$
\mathcal{L}_{Balance}
=\alpha\sum_{i=1}^{N} f_iP_i
$$

其中，$f_i$ 表示路由到第 $i$ 个 expert 的 token 比例，$P_i$ 表示第 $i$ 个 expert 的平均 gating score，$\alpha$ 控制负载均衡损失强度。

这个方法的问题是：$\alpha$ 太小，负载均衡不明显；$\alpha$ 太大，auxiliary loss 的梯度会干扰主任务目标。放到 OneRec 里，主任务就是 semantic ID 的 next-token prediction 以及后续偏好对齐。如果额外负载均衡损失过强，模型可能为了“平均使用 experts”牺牲推荐生成质量。

#### DeepSeek-V3 的 loss-free load balancing 是什么

DeepSeek-V3 采用的 **auxiliary-loss-free load balancing** 来自 Wang et al. 2024。它的关键思想是：**不用额外 loss 反向传播负载均衡梯度，而是在路由决策前给每个 expert 的 routing score 加一个动态 bias**。

假设第 $t$ 个 token 到第 $i$ 个 expert 的原始 affinity score 是：

$$
s_{i,t}
$$

为每个 expert 维护一个 bias：

$$
b_i
$$

路由时，用带 bias 的分数决定 top-k：

$$
s_{i,t}+b_i
$$

但注意，DeepSeek-V3 原文强调：**bias 只用于决定选哪些 experts，不用于最终 expert 输出的加权系数**。也就是说，最终 gate 权重仍然来自原始 affinity score $s_{i,t}$，而不是 $s_{i,t}+b_i$。

其形式可以写为：

$$
g'_{i,t}=
\begin{cases}
s_{i,t}, & s_{i,t}+b_i\in \mathrm{Topk}(\{s_{j,t}+b_j\mid 1\le j\le N_r\}, K_r) \\
0, & \text{otherwise}
\end{cases}
$$

训练过程中，每个 step 统计 expert load：

| expert 状态 | bias 更新                         | 效果                                   |
|-------------|-----------------------------------|----------------------------------------|
| 过载        | 降低该 expert 的 $b_i$             | 后续 token 更不容易选到它。             |
| 欠载        | 提高该 expert 的 $b_i$             | 后续 token 更容易选到它。               |

DeepSeek-V3 中这个更新速度由超参数 $\gamma$ 控制。原文描述为：每个训练 step 结束后，如果 expert 过载，则 $b_i$ 减少 $\gamma$；如果 expert 欠载，则 $b_i$ 增加 $\gamma$。

这就是 “loss-free” 的含义：它仍然控制 expert 负载，但不把负载均衡写成一个额外的可微损失项加入总 loss，因此不会直接给主模型参数引入 auxiliary loss 的干扰梯度。

#### 放到 OneRec 里的意义

OneRec 的 decoder 要生成 semantic ID。MoE 扩大 decoder 容量后，不同 experts 可能自然分化去处理不同兴趣簇、内容簇、用户场景或 token 前缀模式。loss-free load balancing 的意义是：

1. 让 experts 在 batch 级别保持相对均衡，避免部分 experts 闲置或过载。  
2. 不额外引入负载均衡 loss，减少对 $\mathcal{L}_{NTP}$ 的干扰。  
3. 在推荐这种目标极其敏感的场景中，避免“为了均衡而均衡”，让主目标仍然主导模型学习。  

OneRec 原文报告，使用该策略后 scaled OneRec 模型的 NTP loss 降低约 0.2。这里能确认的是实验现象；至于每个 expert 具体学到了哪些推荐子分布，论文没有给出可验证分析。

### Next-token prediction

OneRec 使用 semantic ID 的 next-token prediction loss：

$$
\mathcal{L}_{NTP}
=-\sum_{j=0}^{L_t-1}
\log P(s_m^{j+1}\mid [s_{[BOS]},s_m^1,\ldots,s_m^j])
$$

这使得模型能够根据用户表示逐步生成 item semantic ID。

## 2.4 Reward System

仅通过预训练，OneRec 主要学习的是传统推荐系统曝光数据中的分布。由于这些曝光数据来自旧系统，模型容易被旧系统的分布上界限制。为突破该限制，论文引入 reward system，通过强化学习对生成推荐结果进行偏好对齐。

Reward system 包含三类奖励：

| 奖励类型                   | 作用                                                                 |
|----------------------------|----------------------------------------------------------------------|
| User Preference Reward      | 学习用户个性化偏好，替代手工多目标加权。                              |
| Format Reward               | 提升生成 semantic ID 映射到合法 item 的比例。                         |
| Specific Industrial Reward  | 面向内容生态、商业化、冷启动、长尾等工业目标做定向优化。               |

### 2.4.1 用户偏好对齐

推荐系统中的“好推荐”并不容易定义。传统方法通常训练多个 xtr 目标，例如 click、like、comment、watch time 等，然后通过人工权重融合。但这种方法存在两个问题：

1. 不同目标之间可能冲突。  
2. 固定权重难以表达不同用户、不同场景下的个性化价值差异。  

OneRec 提出用神经网络学习个性化融合分数，称为 **P-Score**。P-Score 模型基于 SIM 架构，多个 tower 负责学习不同目标的辅助 BCE loss，随后将各 tower hidden states 与用户、item 表示输入 MLP，得到最终 P-Score。

整体损失为：

$$
\mathcal{L}_{P\text{-}Score}
=\sum_{xtr\in S_o}
w_{xtr}\mathcal{L}_{P\text{-}Score}^{xtr}
$$

其中：

$$
\mathcal{L}_{P\text{-}Score}^{xtr}
=-\left(
y_{xtr}\log p
+(1-y_{xtr})\log(1-p)
\right)
$$

目标集合包括：

$$
S_o=\{ctr,lvtr,ltr,vtr,\ldots\}
$$

论文指出，通过调整 $w_{xtr}$，P-Score 可以偏向不同业务目标，并在实验中改善 AUC。相比简单加权求和，P-Score 更接近个性化的 Pareto 优化。

### Early Clipped GRPO / ECPO

OneRec 使用一种称为 **Early Clipped Policy Optimization, ECPO** 的策略优化方法。对每个用户，旧策略生成 $G$ 个 item，reward system 为每个 item 打分 $r_i$，然后归一化得到 advantage：

$$
A_i
=\frac{r_i-\mathrm{mean}(\{r_1,\ldots,r_G\})}
{\mathrm{std}(\{r_1,\ldots,r_G\})}
$$

ECPO 的目标是 PPO-style clipped objective。核心思想是对 policy ratio 提前裁剪，避免负 advantage 样本产生过大的梯度。

论文定义修正后的旧策略概率：

$$
\pi'_{old}(o_i\mid u)
=\max\left(
\frac{\mathrm{sg}(\pi_\theta(o_i\mid u))}{1+\epsilon+\delta},
\pi_{old}(o_i\mid u)
\right)
$$

其中 $\delta>0$，论文设置为 $\delta=0.1$。ECPO 使用 $\pi'_{old}$ 作为 denominator，提前限制过大的 ratio。论文还移除了 KL divergence 项，因为强化学习和 SFT 同时训练，SFT loss 起到稳定模型的作用。

### 2.4.2 生成格式正则

Semantic ID 空间大小为：

$$
N_t^{L_t}
$$

真实视频数量远小于该空间，因此模型可能生成无法映射到真实视频的非法 semantic ID。论文将 legality ratio 定义为生成序列中能映射到真实 item 的比例。

普通 ECPO 可能因为负样本挤压效应，进一步降低合法 ID 比例。为此 OneRec 引入 format reward。随机从 $G$ 个生成样本中选择 $K$ 个，合法样本 advantage 设置为 1，非法样本丢弃：

$$
A_i=
\begin{cases}
1, & o_i\in I_{legal} \\
0, & \text{otherwise}
\end{cases}
$$

论文强调，非法样本不作为负 advantage 直接压制，以避免 squeezing effect。

### 2.4.3 工业场景对齐

工业推荐系统需要兼顾内容生态、商业化、冷启动、长尾供给等目标。传统系统通常通过补丁式策略处理这些目标，导致系统越来越复杂。OneRec 试图把这些目标纳入 reward system，通过 RL 做统一优化。

论文给出的例子是控制病毒式内容农场的曝光。如果某类 viral content 的比例超过阈值 $f$，则对对应样本的 P-Score reward 做折扣：

$$
r'_i=
\begin{cases}
r_i, & o_i\notin I_{viral} \\
\alpha r_i, & o_i\in I_{viral}
\end{cases}
$$

其中：

$$
\alpha\in(0,1)
$$

实验中，该方法降低了 viral content 的曝光比例，同时保持 Watch Time 和 App Stay Time 基本稳定。

## 3. 训练框架

## 3.1 训练基础设施

OneRec 的训练集群由 90 台服务器组成，每台服务器包含 8 张旗舰 GPU 和 2 个 CPU。单机 GPU 间通过 400Gbps NVLink 互联。跨机网络方面，训练通信使用 400Gbps RDMA，数据和 embedding prefetch 使用 100Gbps TCP。每台机器配备 4 块 NVMe SSD 用于 checkpoint 写入，embedding 和 dense 参数存储在 HDFS 中。

训练加速主要包括四个方面：

| 技术                    | 作用                                                                 |
|-------------------------|----------------------------------------------------------------------|
| Embedding Acceleration  | 使用快手 SKAI GPU 参数服务器、跨 GPU 统一 embedding table、GPU cache 和 prefetch。 |
| Training Parallelism    | 使用 data parallelism、ZERO1 和 gradient accumulation。              |
| Mixed Precision         | 在部分 MLP 计算中使用 BFloat16。                                      |
| Compilation Optimization | 对 attention 等计算进行编译优化。                                    |

论文选择 ZERO1，是因为 dense 参数可以放进单张 GPU，不需要更复杂的模型并行。最终训练 MFU 达到 23.7%。

## 3.2 预训练

预训练输入是多尺度用户行为表示，目标是预测样本对应的目标 item semantic ID 序列。由于每个 item 被 tokenized 为 3 个 semantic IDs，因此每个训练样本对应 3 个目标 token。

论文中的训练流水线每天处理约 18B 样本，即约 54B tokens/day。OneRec-0.935B 在约 100B 样本，也就是约 300B tokens 后收敛。

### 模型结构

论文给出的模型规模如下：

| Model          | Type  | Layers | Hidden | FFN  | Heads | Experts | MoE          |
|----------------|-------|-------:|-------:|-----:|------:|---------|--------------|
| OneRec-0.015B  | Dense |      4 |    128 |  256 |     4 | -       | -            |
| OneRec-0.121B  | Dense |      8 |   1024 | 2048 |     8 | -       | -            |
| OneRec-0.935B  | MoE   |      8 |   1024 | 2048 |     8 | 24/2    | Decoder      |
| OneRec-2.633B  | MoE   |     24 |   1024 | 2048 |     8 | 24/4    | Enc & Dec    |

其中 Layers 表示 encoder 和 decoder 层数之和。Dense FFN 的 hidden dimension 是 $2d_{model}$。MoE 使用 SwiGLU expert，expert hidden dimension 设置为：

$$
\frac{2}{3}\times 4d_{model}
$$

并对齐到 128 的整数倍。

## 3.3 后训练

OneRec 的后训练使用实时数据流，包含两个并行目标：

| 目标 | 含义                                      |
|------|-------------------------------------------|
| RSFT | Rejection Sampling Fine-Tuning。           |
| RL   | 基于 reward system 的强化学习对齐。        |

### RSFT

RSFT 会过滤掉播放时长处于 bottom 50% 的曝光 session，只保留质量更高的数据继续做 next-token prediction。RSFT 使用与预训练相同的 $\mathcal{L}_{NTP}$。

论文中学习率设置为：

| 参数类型 | 学习率  |
|----------|---------|
| sparse   | 1e-4    |
| dense    | 8e-5    |

### RL 训练

RL 从 RSFT 数据中随机选取 1% 用户。为了避免训练任务被生成推理阻塞，论文将 RL 样本生成和训练解耦：训练任务调用外部推理服务为用户生成 512 个 item，再调用 reward service 对每个 item 打分，并将生成结果返回训练任务。

训练任务每 1000 step 通过消息队列将最新参数发送给推理服务，使推理服务的 policy 与训练任务保持近似同步。

## 4. 评估

## 4.1 评估指标

论文主要使用以下指标：

| 指标                      | 含义                                                         |
|---------------------------|--------------------------------------------------------------|
| $\mathcal{L}_{NTP}$       | next-token prediction cross-entropy loss。                   |
| P-Score                   | reward model 输出的用户偏好分数。                            |
| lvtr、vtr、ltr、wtr、cmtr | 由预训练 ranking model 估计的多目标指标。                    |
| 在线指标                  | Watch Time、App Stay Time、Video View、Like、Follow 等。     |

由于流式数据随时间变化，离线实验需要在相同时间窗口对比，并在较长窗口上取平均。

## 4.2 Scaling

### 4.2.1 训练 Scaling

论文从模型规模、数据规模、负载均衡、特征规模、codebook 规模和推理 Pass@K 等角度观察 scaling 现象。

#### 模型参数量

随着模型参数量扩大，$\mathcal{L}_{NTP}$ 持续下降，说明 OneRec 在一定范围内符合推荐系统中的 scaling 规律。

#### 数据规模

训练初期约前 10B 样本收敛速度较快，之后提升放缓，但超过 100B 样本后模型仍然能继续受益。

#### Loss-free load balancing

论文实验显示，loss-free load balancing 可以降低约 0.2 的 NTP loss。

#### 特征规模

论文比较了只使用 256 个正反馈 item ID embedding 的 baseline，与加入更丰富用户行为特征后的模型。加入完整特征后，各类 reward 指标明显提升。

| Metric  | w/o feature | w/ feature | Impr.   |
|---------|------------:|-----------:|--------:|
| lvtr    |      0.4940 |     0.5500 | +11.34% |
| vtr     |      0.8730 |     0.8901 |  +1.96% |
| ltr     |      0.0391 |     0.0441 | +12.79% |
| wtr     |      0.0190 |     0.0224 | +17.89% |
| cmtr    |      0.0919 |     0.1010 |  +9.90% |
| P-score |      0.0749 |     0.0966 | +28.88% |

#### Codebook 规模

论文将 codebook size 从 8K 扩大到 32K。由于候选空间变大，NTP loss 不能直接比较，但 reward 指标均有改善。

| Metric  | 8K     | 32K    | Impr.  |
|---------|-------:|-------:|-------:|
| lvtr    | 0.5118 | 0.5245 | +2.48% |
| vtr     | 0.9384 | 0.9491 | +1.14% |
| ltr     | 0.0298 | 0.0299 | +0.34% |
| wtr     | 0.0153 | 0.0154 | +0.65% |
| cmtr    | 0.0650 | 0.0664 | +2.15% |
| P-score | 0.2516 | 0.2635 | +4.75% |

#### Inference Pass@K

推理时，模型可以生成多个候选 item，再选择更优结果。论文比较了 Pass@8、Pass@64、Pass@512、Pass@1024。K 越大，指标越好，但计算成本也越高。生产中选择 Pass@512 作为权衡。

| Metric  | Pass@8 | Pass@64 | Pass@512 | Pass@1024 | Impr.    |
|---------|-------:|--------:|---------:|----------:|---------:|
| lvtr    | 0.3675 |  0.4927 |   0.5351 |    0.5443 |  +48.11% |
| vtr     | 0.9444 |  0.9462 |   0.9513 |    0.9530 |   +0.91% |
| ltr     | 0.0278 |  0.0346 |   0.0425 |    0.0452 |  +62.59% |
| wtr     | 0.0114 |  0.0138 |   0.0182 |    0.0197 |  +72.81% |
| cmtr    | 0.0350 |  0.0566 |   0.0809 |    0.0891 | +154.57% |
| P-score | 0.0811 |  0.2051 |   0.3375 |    0.3859 | +376.10% |

### 4.2.2 Semantic Identifier 输入表示

论文还探索了在十亿参数规模下，是否可以用 semantic ID 替代用户历史中的 video ID sparse embedding。结果显示，OneRec-2.633B 使用 semantic ID 输入后，整体表现与 sparse video ID embedding 相当，部分指标略优。

| Metric  | VID    | Semantic ID | Impr.  |
|---------|-------:|------------:|-------:|
| lvtr    | 0.4447 |      0.4467 | +0.45% |
| vtr     | 0.8725 |      0.8726 | +0.01% |
| ltr     | 0.0336 |      0.0336 | +0.00% |
| wtr     | 0.0104 |      0.0105 | +0.96% |
| cmtr    | 0.0565 |      0.0573 | +1.42% |
| P-score | 0.0371 |      0.0378 | +1.74% |

论文认为 semantic ID 输入有四个优势：

| 优势                  | 说明                                                                 |
|-----------------------|----------------------------------------------------------------------|
| 参数效率              | 输入和输出可以共享 semantic ID embedding，不需要独立 video ID embedding。 |
| 通信效率              | 减少参数服务器 lookup 和 update 的通信成本。                          |
| 序列容量              | 在同等内存下可以容纳更长用户行为序列。                                |
| 表示一致性            | 输入 item 和输出 item 处于同一 semantic ID 空间。                      |

## 4.3 强化学习

### 4.3.1 用户偏好对齐

为验证 RL 的效果，论文首先使用单目标 vtr reward。vtr 与在线 Watch Time 和 App Stay Time 关系更直接。在线结果以快手传统推荐系统为基线。

#### 采样效率

论文比较不同 Pass@K 下，使用 RL 和不使用 RL 的效果。

| Setting  | RL    | vtr    | Watch Time | App Stay Time | Video View |
|----------|-------|-------:|-----------:|--------------:|-----------:|
| Pass@32  | w/o   | 0.1978 |     +1.62% |        -0.10% |      -4.18 |
| Pass@32  | w/    | 0.2138 |     +3.17% |        +0.39% |      -9.87 |
| Pass@128 | w/o   | 0.2239 |     +4.61% |        +1.11% |     -12.75 |
| Pass@128 | w/    | 0.2387 |     +5.22% |        +1.49% |     -15.06 |
| Pass@512 | w/o   | 0.2444 |     +6.32% |        +1.66% |     -15.54 |
| Pass@512 | w/    | 0.2494 |     +5.88% |        +1.75% |     -13.88 |

实验显示，RL 可以提升生成样本的 reward 质量。在较小 Pass@K 下，RL 的增益更明显；当 K 很大时，采样本身已经能覆盖更多候选，RL 的边际收益下降。

#### 搜索空间和 group size

论文在 Pass@128 下比较不同 RL group size：

| Group Size | vtr    | Watch Time | App Stay Time | Video View |
|-----------:|-------:|-----------:|--------------:|-----------:|
| 0          | 0.2198 |     +4.61% |        +1.11% |     -12.75 |
| 128        | 0.2303 |     +5.22% |        +1.49% |     -15.06 |
| 512        | 0.2350 |     +5.73% |        +1.82% |     -15.49 |
| 2048       | 0.2352 |     +5.84% |        +1.78% |     -15.49 |

论文经验上认为，ECPO 的 group size 约为推理输出数量的 4 倍较合适。

#### 搜索策略

论文比较 Top-k + Top-p sampling 与 Beam Search：

| Strategy      | vtr    | Watch Time | App Stay Time | Video View |
|---------------|-------:|-----------:|--------------:|-----------:|
| Top-k + Top-p | 0.2131 |     +4.45% |        +1.16% |     -13.61 |
| Beam Search   | 0.2162 |     +5.35% |        +1.76% |     -13.30 |

Beam Search 表现更好。论文解释为 semantic ID 具有前缀树结构，Beam Search 能更好利用这种结构。

#### Reference model

论文比较使用预训练模型和当前 policy model 作为 reference：

| Reference Model      | vtr    | Watch Time | App Stay Time | Video View |
|----------------------|-------:|-----------:|--------------:|-----------:|
| Pre-trained Model    | 0.2262 |     +5.35% |        +1.51% |     -13.51 |
| Current Policy Model | 0.2389 |     +6.19% |        +1.56% |     -13.89 |

当前 policy model 作为 reference 时，reward 指标更好。

#### P-Score reward 在线效果

使用 P-Score reward 进行偏好对齐后，在线指标如下：

| App            | Watch Time | App Stay Time | Video View |
|----------------|-----------:|--------------:|-----------:|
| Kuaishou       |     +0.21% |        +0.26% |      +0.17 |
| Kuaishou Lite  |     +0.71% |        +0.22% |      +0.35 |

### 4.3.2 Format reward

没有 format reward 时，模型生成合法 semantic ID 的比例低于 50%。如果只选择 top-k 样本，选中样本的合法率可以达到 100%，但整体合法率先上升后下降。随机选择样本的设定更难，但加入 format reward 后合法率可以稳定提升到约 95%。

在线上，format reward 带来：

| 指标          | 提升   |
|---------------|-------:|
| APP Stay Time | +0.13% |
| Watch Time    | +0.30% |

### 4.3.3 工业场景对齐

论文以 viral content farms 为例。若不加控制，这类内容的曝光比例会上升，可能损害平台内容生态。

OneRec 通过 Specific Industrial Reward 对这类 item 的 reward 做折扣。实验结果显示，该方法将 viral content 曝光比例降低 9.59%，同时保持 Watch Time 和 App Stay Time 基本稳定。

## 4.4 Tokenizer

论文从三个角度评估 tokenizer：

| 指标                 | 含义                                               |
|----------------------|----------------------------------------------------|
| Reconstruction loss  | semantic ID 对原表示的重构误差。                   |
| Codebook utilization | codebook 中被使用 token 的比例。                   |
| Entropy              | token 分布熵，反映 token 使用是否均衡。             |

RQ-VAE 与 RQ-Kmeans 对比如下：

| Metric               | Layer | RQ-VAE | RQ-Kmeans |
|----------------------|------:|-------:|----------:|
| Reconstruction loss  | -     | 0.0548 |    0.0410 |
| Codebook utilization | 1     | 1.0000 |    1.0000 |
| Codebook utilization | 2     | 0.9963 |    1.0000 |
| Codebook utilization | 3     | 0.9958 |    1.0000 |
| Entropy              | 1     | 8.3892 |    8.9191 |
| Entropy              | 2     | 8.4805 |    8.7770 |
| Entropy              | 3     | 8.6037 |    8.7276 |

RQ-Kmeans 的 reconstruction loss 相比 RQ-VAE 降低 25.18%，三层 entropy 分别提升 6.31%、3.50% 和 1.44%。论文据此认为 RQ-Kmeans 更适合 OneRec 的 semantic ID 构造。

## 4.5 在线 A/B 测试

OneRec 部署在快手和快手极速版主 feed 场景。两个 app 合计 DAU 达 400M。论文报告了持续一周、5% 流量的在线实验。

实验包含两个组：

| 组别                    | 说明                                                   |
|-------------------------|--------------------------------------------------------|
| Pure generative OneRec   | 直接使用 OneRec 生成结果。                             |
| OneRec with RM Selection | OneRec 生成候选后，再使用 reward model 选择结果。       |

与当前多阶段推荐系统相比，线上结果如下：

| App           | Method             | App Stay | Watch | Video View | Like  | Follow | Comment | Collect | Forward |
|---------------|--------------------|---------:|------:|-----------:|------:|-------:|--------:|--------:|--------:|
| Kuaishou      | OneRec             |   +0.01% | +0.07%|     +1.98% | -2.00%| -2.88% |  -1.56% |  -0.61% |  +0.27% |
| Kuaishou      | OneRec + RM        |   +0.54% | +1.98%|     +2.52% | +2.43%| +3.24% |  +5.27% |  +2.93% |  +5.90% |
| Kuaishou Lite | OneRec             |   +0.06% | +0.05%|     +2.40% | -2.64%| -2.75% |  -2.23% |  -1.76% |  -1.86% |
| Kuaishou Lite | OneRec + RM        |   +1.24% | +3.28%|     +3.39% | +1.49%| +2.28% |  +3.20% |  +1.91% |  +3.48% |

可以看到，pure generative OneRec 在部分指标上已经接近或超过传统系统，但互动类指标可能下降；加入 RM Selection 后，整体指标显著改善。

### 推理基础设施

在线推理使用 NVIDIA L20 GPU，每台服务器 4 张 GPU 和 2 个 CPU，通过 PCIe 连接。推理系统基于快手 UniPredict，embedding service 与 inference service 部署在 200Gb RDMA 数据中心，通过 RoCE 通信，最大跨机带宽 800Gb。

论文使用 TensorRT 和自定义 plugin 优化 cross-attention、MoE 等算子，并通过 batching 和 MPS 提升吞吐。最终吞吐提升 5 倍，推理 MFU 达到 28.8%。

### 本地生活场景

OneRec 也在 Local Life Service 场景中应用，并取得以下收益：

| 指标                  | 提升    |
|-----------------------|--------:|
| GMV                   | +21.01% |
| Order Volume          | +17.89% |
| Buyer Numbers         | +18.58% |
| New Buyer Acquisition | +23.02% |

在该业务场景中，OneRec 已承接 100% QPS。

## 5. 结论、局限与未来方向

论文提出 OneRec 作为端到端生成式推荐框架，用 encoder 压缩用户全生命周期行为，用 MoE decoder 扩展模型容量，并通过自定义 reward function 做强化学习对齐。系统层面，OneRec 的训练 MFU 达到 23.7%，推理 MFU 约 28.8%，OPEX 约为传统推荐流水线的 10.6%。

论文认为 OneRec 已经证明生成式推荐可以在工业主场景中工作，但仍存在局限：

### 5.1 推理阶段 scaling 不充分

OneRec 在训练阶段已经观察到 scaling 现象，但在推理阶段的 scaling 还不够明显。论文认为，这可能是因为当前推荐生成任务尚未充分引入强 reasoning 能力。

### 5.2 多模态融合仍不充分

OneRec 已经使用多模态 item 表征生成 semantic IDs，但还没有将 LLM/VLM 等模型深度整合到推荐主模型中。论文指出，用户行为本身也可以被视为一种模态，未来可以像视觉、音频和文本对齐一样，让用户行为成为原生多模态建模对象。

### 5.3 Reward system 仍处早期

当前 reward system 仍比较初级。论文认为，OneRec 这种统一架构能够促进 reward 建模的进一步突破，因为平台目标可以通过 reward 更直接地作用到生成模型上。

## 附录 A：贡献

论文附录列出了 OneRec 项目的主要贡献者，包括架构设计、训练系统、推理系统、tokenizer、reward system、在线实验和业务落地等多个方向。本文不逐名翻译贡献列表，读者可参考原文附录 A。

## 附录 B：在线 A/B 测试实现细节

快手主 feed 的在线服务存在缓存机制：每次用户请求会返回 $k$ 个推荐结果，未曝光的剩余结果会被存入 cache pool。高 QPS 请求可以直接读取缓存结果，以降低在线计算成本。

缓存机制的问题是会牺牲推荐结果的实时性，尤其在晚高峰等流量高峰期影响更明显。完全关闭缓存需要消耗大量机器资源。OneRec 的计算效率更高，因此论文选择优先替换 degraded cache traffic。

在线实验设置：

| 设置              | 说明                                                    |
|-------------------|---------------------------------------------------------|
| 实验流量          | 5% 用户。                                                |
| OneRec 覆盖范围   | degraded traffic 中的 25%。                             |
| 对照组            | 当前系统，以及额外 1% 的 cache-disabled baseline。       |

OneRec 与当前系统、关闭缓存 baseline 的对比如下：

| App           | Metric           | vs Current | vs Cache Disabled |
|---------------|------------------|-----------:|------------------:|
| Kuaishou      | App Stay Time    |     +0.54% |            +0.20% |
| Kuaishou      | LT7              |     +0.05% |            +0.03% |
| Kuaishou      | Watch Time       |     +1.98% |            +0.75% |
| Kuaishou      | Video View       |     +2.52% |            +1.79% |
| Kuaishou      | Engagement Depth |     +1.78% |            +1.30% |
| Kuaishou      | Like             |     +2.43% |            +0.88% |
| Kuaishou      | Follow           |     +3.24% |            +1.29% |
| Kuaishou      | Comment          |     +5.27% |            +3.18% |
| Kuaishou      | Collect          |     +2.93% |            +0.73% |
| Kuaishou      | Forward          |     +5.90% |            +4.92% |
| Kuaishou Lite | App Stay Time    |     +1.24% |            +0.55% |
| Kuaishou Lite | LT7              |     +0.08% |            +0.02% |
| Kuaishou Lite | Watch Time       |     +3.28% |            +1.58% |
| Kuaishou Lite | Video View       |     +3.39% |            +1.71% |
| Kuaishou Lite | Engagement Depth |     +2.89% |            +2.49% |
| Kuaishou Lite | Like             |     +1.49% |            -1.71% |
| Kuaishou Lite | Follow           |     +2.28% |            +0.89% |
| Kuaishou Lite | Comment          |     +3.20% |            +0.60% |
| Kuaishou Lite | Collect          |     +1.91% |            -1.03% |
| Kuaishou Lite | Forward          |     +3.48% |            +1.35% |

论文指出，OneRec 当前已替代原有缓存机制，并在主场景承接 25% 流量。

## 附录 C：Tokenization 案例分析

### C.1 表征案例

论文比较了三种表征：

| 表征类型                              | 特点                                                                  |
|---------------------------------------|-----------------------------------------------------------------------|
| Collaborative representation           | 能捕捉用户共现和协同相似，但可能缺少内容语义解释。                    |
| Pure multimodal representation         | 能捕捉视觉、文本等表面语义相似，但可能与推荐行为相似不一致。          |
| Aligned collaborative-aware multimodal | 同时融合内容语义和协同信号，更适合推荐 tokenization。                 |

论文案例显示，纯协同表示容易把用户共同观看但内容差异较大的视频拉近；纯多模态表示容易只关注画面或文本相似；OneRec 的对齐表征能同时捕捉内容相关性和行为相关性。

### C.2 Tokenization 案例

RQ-Kmeans 生成的是 coarse-to-fine semantic IDs。论文案例显示，第一层 token 往往对应较粗粒度主题，第二层和第三层逐渐细化。

例如：

| 层级 | 可能表达的粒度                         |
|------|----------------------------------------|
| 第 1 层 | 大类主题，例如 Food、Sports。          |
| 第 2 层 | 子主题，例如 Mukbang、Basketball。     |
| 第 3 层 | 更细粒度风格或内容簇。                 |

论文中的案例包括：

| 示例             | 粗粒度            | 中粒度                  | 细粒度             |
|------------------|-------------------|-------------------------|--------------------|
| Food 视频         | Food              | Mukbang / Noodles       | 具体面食内容簇      |
| Sports 视频       | Sports            | Basketball & Football   | Basketball 内容簇   |

## 附录 D：符号表

### Tokenizer 相关

| 符号                    | 含义                                                        |
|-------------------------|-------------------------------------------------------------|
| $M$                     | miniCPM-V-8B 输出的多模态 token 表征。                       |
| $N_M$                   | 多模态 token 数量，论文中为 1280。                           |
| $d_t$                   | tokenizer 表征维度，论文中为 512。                           |
| $\tilde{N}_M$           | QFormer 压缩后的 query token 数量，论文中为 4。               |
| $\tilde{M}$             | QFormer 输出的压缩多模态表示。                               |
| $D_{pair}$              | 用于协同对齐的 item pair 数据集。                            |
| $C^{(l)}$               | 第 $l$ 层 RQ-Kmeans codebook。                               |
| $N_t$                   | 每层 codebook size。                                         |
| $L_t$                   | semantic ID 层数，论文中为 3。                               |
| $s_m^l$                 | 视频 $m$ 在第 $l$ 层的 semantic ID。                          |

### Encoder 相关

| 符号                    | 含义                                                        |
|-------------------------|-------------------------------------------------------------|
| $d_{model}$             | OneRec 主模型 hidden size。                                  |
| $h_u$                   | 用户静态特征表示。                                          |
| $h_s$                   | 短期行为序列表示。                                          |
| $h_p$                   | 正反馈行为序列表示。                                        |
| $h_l$                   | 全生命周期行为压缩表示。                                    |
| $L_s$                   | 短期行为序列长度，论文中为 20。                              |
| $L_p$                   | 正反馈行为序列长度，论文中为 256。                           |
| $L_l$                   | 压缩后的长期行为序列长度，论文中为 2000。                    |
| $N_q$                   | Lifelong QFormer query 数量，论文中为 128。                  |
| $z_{enc}$               | Encoder 最终输出，作为 decoder cross-attention 的 key/value。 |

### Decoder 和 RL 相关

| 符号                    | 含义                                                        |
|-------------------------|-------------------------------------------------------------|
| $s_{[BOS]}$             | Decoder 生成 item semantic ID 的起始 token。                 |
| $S_m$                   | 目标视频 $m$ 的 decoder 输入 token 序列。                    |
| $\mathcal{L}_{NTP}$     | next-token prediction loss。                                |
| $G$                     | RL 中每个用户生成的候选数量，也称 group size。               |
| $r_i$                   | reward system 对第 $i$ 个生成 item 的打分。                  |
| $A_i$                   | 归一化后的 advantage。                                      |
| $\pi_\theta$            | 当前 policy model。                                          |
| $\pi_{old}$             | 旧 policy model。                                            |
| $\pi'_{old}$            | ECPO 中修正后的旧策略概率。                                  |
| $I_{legal}$             | 可映射到真实 item 的合法 semantic ID 集合。                  |
| $I_{viral}$             | 工业场景奖励中需要控制的 viral item 集合。                   |

## 工程理解补充

以下内容是基于原文事实的工程理解，不是论文直接陈述。

1. OneRec 的核心价值不是简单把召回模型改成生成模型，而是把“候选生成 + 排序 + 目标对齐”压入统一训练和推理框架。  
2. Semantic ID 是连接生成模型和推荐 item 空间的关键。它既要足够压缩，保证生成空间可控；又要足够语义化和协同化，保证生成结果能表达真实用户兴趣。  
3. Reward system 是 OneRec 能否超越旧系统分布上界的关键。如果只有 SFT/NTP，模型主要学习旧系统曝光分布；只有引入 reward 对齐，才可能主动偏向更优推荐结果。  
4. 从推荐算法工程角度看，OneRec 把传统多目标融合、打散、供给调控、生态约束等后处理问题，部分上移到了 reward 设计和 RL 对齐中。  
5. OneRec 的系统收益来自架构统一和 GPU 高密度计算，但它也要求 tokenizer、生成合法率、reward 稳定性、推理成本和在线容灾能力都达到工业级水平。
