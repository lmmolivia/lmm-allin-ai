# GRPO 和 DPO 的区别

> 公式参考本地论文：`reference/dpo_direct_preference_optimization_2305.18290.pdf` 和 `reference/deepseekmath_grpo_2402.03300.pdf`。

## 1. 一句话区别

**DPO** 是离线偏好学习：给定同一个 prompt 下的 winner / loser 响应对，直接用分类式 loss 让模型更偏向 winner，不需要在线采样、不显式训练 reward model，也不需要 value model。

**GRPO** 是在线强化学习：对同一个 prompt 从当前策略采样一组 responses，用 reward/verifier 打分，在组内做相对优势估计，然后用 PPO 风格的 clipped objective 更新模型，不需要 value model，但需要 rollout 和 reward。

更短地说：

```text
DPO：拿已有偏好对做离线训练，优化“winner 比 loser 更好”。
GRPO：当前模型现场生成一组答案，用 reward 排名，优化“组内高分答案概率更高”。
```

## 2. DPO 是什么

DPO 全称 Direct Preference Optimization，来自论文 **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**。

它的目标是绕过传统 RLHF 的复杂流程。

传统 RLHF 大致是：

```text
SFT model
-> 收集偏好数据
-> 训练 reward model
-> 用 PPO 优化 policy
```

DPO 则把 reward model 隐式写进 policy 和 reference policy 的 log-prob ratio 里，直接训练 policy。

### 2.1 DPO 需要什么数据

DPO 的训练数据是偏好对：

```text
(x, y_w, y_l)
```

含义：

- `x`：prompt / 用户上下文 / 问题。
- `y_w`：winner，被偏好的输出。
- `y_l`：loser，不被偏好的输出。

在推荐里可以类比为：

```text
x: 用户上下文
y_w: 更高收益推荐列表
y_l: 更低收益推荐列表
```

这些偏好对可以来自人工标注、线上反馈、reward model 打分、旧系统排序结果，或者 A/B 中效果更好的列表。

### 2.2 DPO 的核心公式

DPO 的经典 loss 可以写成：

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})
= -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}
\left[
\log \sigma \left(
\beta
\left(
\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)}
-
\log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
\right)
\right)
\right]
$$

其中：

- `pi_theta` 是正在训练的模型。
- `pi_ref` 是 reference model，通常是 SFT model 的冻结副本。
- `beta` 控制模型偏离 reference model 的强度。
- `log pi_theta(y | x)` 是当前 policy 对完整输出 `y` 的 log-prob，通常是 token-level log-prob 之和。
- `log pi_ref(y | x)` 是 reference model 对完整输出 `y` 的 log-prob。

直觉上：

```text
如果 winner 相对 reference 的概率提升
且 loser 相对 reference 的概率降低
loss 就会变小
```

也可以写成更紧凑的形式：

$$
\begin{aligned}
\Delta_\theta
&= \log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x) \\
\Delta_{\mathrm{ref}}
&= \log \pi_{\mathrm{ref}}(y_w \mid x) - \log \pi_{\mathrm{ref}}(y_l \mid x) \\
\mathcal{L}_{\mathrm{DPO}}
&= - \log \sigma \left( \beta(\Delta_\theta - \Delta_{\mathrm{ref}}) \right)
\end{aligned}
$$

注意这里的 `pi_ref` 很关键。DPO 不是单纯让 winner 概率高于 loser，而是让当前模型相对于 reference model 更偏向 winner。这样做相当于保留了一个隐式 KL 约束，避免模型离 SFT model 太远。

### 2.3 DPO 如何训练

DPO 的训练流程：

```text
1. 先准备一个 SFT model。
2. 冻结一份 SFT model 作为 reference model pi_ref。
3. 收集或构造偏好数据：
   (x, y_w, y_l)
4. 用当前 policy pi_theta 计算：
   log pi_theta(y_w | x)
   log pi_theta(y_l | x)
5. 用 frozen reference model 计算：
   log pi_ref(y_w | x)
   log pi_ref(y_l | x)
6. 代入 DPO loss。
7. 只更新 pi_theta，不更新 pi_ref。
```

伪代码：

```text
for batch in preference_data:
    x, y_w, y_l = batch

    pi_yw_logp  = policy.logprob(x, y_w)
    pi_yl_logp  = policy.logprob(x, y_l)
    ref_yw_logp = reference.logprob(x, y_w)
    ref_yl_logp = reference.logprob(x, y_l)

    pi_logratio  = pi_yw_logp  - pi_yl_logp
    ref_logratio = ref_yw_logp - ref_yl_logp

    loss = -log_sigmoid(beta * (pi_logratio - ref_logratio))
    update(policy, loss)
```

DPO 的梯度直觉：

```text
提高 winner 的 log-prob
降低 loser 的 log-prob
如果当前模型已经把 winner 排得很好，更新会变小
如果当前模型仍然偏向 loser，更新会变大
```

### 2.4 DPO 的特点

DPO 的优点：

- 实现简单。
- 训练稳定。
- 不需要 PPO 那样的 rollout pipeline。
- 不需要显式 reward model。
- 不需要 value model / critic。
- 很适合离线偏好数据。

DPO 的局限：

- 强依赖偏好对质量。
- 训练时不一定探索当前模型的新输出。
- 如果偏好对来自旧模型或旧系统，容易受旧分布限制。
- 只知道 winner > loser，不直接知道 reward 绝对值。
- 如果候选 pair 覆盖不足，模型很难学到更优但未出现过的策略。

## 3. GRPO 是什么

GRPO 全称 Group Relative Policy Optimization，来自 DeepSeekMath。论文把它描述为 PPO 的变体，用来提升数学推理能力，同时优化 PPO 的内存开销。

PPO 通常需要：

```text
policy model
reference model
reward model
value model / critic
```

GRPO 去掉了 value model。它不训练一个 critic 去估计 baseline，而是在同一个 prompt 下采样一组 responses，用组内 reward 均值/标准差构造相对 advantage。

### 3.1 GRPO 需要什么数据

GRPO 的训练过程通常是在线生成数据：

```text
给定 prompt x
从 old policy 采样 G 个输出：
y_1, y_2, ..., y_G
```

然后对每个输出打 reward：

```text
r_1, r_2, ..., r_G
```

reward 可以来自：

- 数学题 verifier。
- rule-based checker。
- reward model。
- 推荐场景里的 CTR/时长/CVR/multi-objective reward model。
- 真实用户反馈。

### 3.2 GRPO 的 advantage

GRPO 的关键是组内相对 advantage：

$$
A_i
=
\frac{
r_i - \mathrm{mean}(r_1,\ldots,r_G)
}{
\mathrm{std}(r_1,\ldots,r_G)
}
$$

含义：

- 比组内平均 reward 高，`A_i > 0`，提高概率。
- 比组内平均 reward 低，`A_i < 0`，降低概率。
- 不需要额外 value model 预测 baseline。

这就是 “Group Relative” 的含义：不是拿一个 critic 作为基线，而是拿同组其他输出作为参照。

### 3.3 GRPO 的目标函数

GRPO 使用 PPO 风格的 ratio clipping。

对同一个问题 `q`，从旧策略 `pi_old` 采样 `G` 个输出：

$$
o_1,o_2,\ldots,o_G \sim \pi_{\theta_{\mathrm{old}}}(\cdot \mid q)
$$

对第 `i` 个输出的第 `t` 个 token，定义 ratio：

$$
\rho_{i,t}
=
\frac{
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid q,o_{i,<t})
}
$$

GRPO 的目标函数可以写成：

$$
\begin{aligned}
\mathcal{J}_{\mathrm{GRPO}}(\theta)
=
\mathbb{E}_{q\sim P(Q),\{o_i\}_{i=1}^{G}\sim\pi_{\theta_{\mathrm{old}}}}
\left[
\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{|o_i|}
\sum_{t=1}^{|o_i|}
\left\{
\min \left(
\rho_{i,t} A_{i,t},
\mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon) A_{i,t}
\right)
-
\beta D_{\mathrm{KL}}\left(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\right)
\right\}
\right]
\end{aligned}
$$

训练实现里通常最小化负目标：

$$
\mathcal{L}_{\mathrm{GRPO}} = -\mathcal{J}_{\mathrm{GRPO}}
$$

其中：

- `pi_theta` 是当前要更新的 policy。
- `pi_old` 是 rollout 时用来采样输出的旧 policy。
- `pi_ref` 是 reference model，通常是 SFT model 或某一轮迭代开始时的 policy。
- `epsilon` 是 PPO clipping 超参。
- `beta` 是 KL 惩罚系数。
- `A_i,t` 是第 `i` 个输出第 `t` 个 token 的 advantage。

DeepSeekMath 论文里还给了 KL 的无偏估计形式：

$$
D_{\mathrm{KL}}\left(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\right)
=
\frac{
\pi_{\mathrm{ref}}(o_{i,t}\mid q,o_{i,<t})
}{
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}
-
\log
\frac{
\pi_{\mathrm{ref}}(o_{i,t}\mid q,o_{i,<t})
}{
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}
-1
$$

这个 KL 项用于限制当前 policy 不要偏离 reference policy 太远。

### 3.4 GRPO 的 advantage 怎么算

最常见的是 outcome supervision：每个输出只有一个最终 reward。

给定同一个 `q` 下的 `G` 个输出 reward：

$$
r_1,r_2,\ldots,r_G
$$

先做组内归一化：

$$
A_i
=
\frac{
r_i - \mathrm{mean}(r_1,\ldots,r_G)
}{
\mathrm{std}(r_1,\ldots,r_G)
}
$$

然后把这个输出级 advantage 分配给输出里的所有 token：

$$
A_{i,t}=A_i
$$

也就是说，如果第 `i` 个输出在同组里 reward 高于平均值，它所有 token 的生成概率都会被整体鼓励；如果低于平均值，它所有 token 的生成概率都会被压低。

论文还讨论了 process supervision：如果 reward model 能给每个推理步骤打分，则某个 token 的 advantage 可以由它后续步骤 reward 的归一化和来决定。数学推理里这有意义；推荐列表生成里也可以类比成对列表位置、子序列或 slate 局部结构打 reward，但工程上会复杂很多。

### 3.5 GRPO 如何训练

GRPO 的训练流程：

```text
1. 从 SFT model 初始化 policy pi_theta。
2. 准备 prompt / query 数据集 Q。
3. 对一个 batch 的 q：
   复制当前 policy 为 old policy pi_old。
4. 对每个 q，用 pi_old 采样 G 个输出：
   o_1, ..., o_G
5. 用 reward model / verifier / 真实反馈给每个输出打分：
   r_1, ..., r_G
6. 在同一个 q 的组内计算 advantage：
   A_i = (r_i - mean(r)) / std(r)
7. 计算 token-level ratio：
   rho_i,t = pi_theta(o_i,t | q, o_i,<t) / pi_old(o_i,t | q, o_i,<t)
8. 代入 clipped objective + KL penalty。
9. 更新 pi_theta。
10. 周期性更新 old policy / reference model，或进入下一轮 iterative GRPO。
```

伪代码：

```text
for batch_q in prompts:
    old_policy = copy(policy)

    outputs = []
    rewards = []

    for q in batch_q:
        group = old_policy.sample(q, G)
        score = reward_model(q, group)
        outputs.append(group)
        rewards.append(score)

    advantages = normalize_within_group(rewards)

    for q, group, group_advantages in batch:
        for o_i, A_i in group:
            for token t in o_i:
                rho = policy.prob(o_i[t] | q, o_i[:t]) / old_policy.prob(o_i[t] | q, o_i[:t])
                clipped = min(rho * A_i, clip(rho, 1 - epsilon, 1 + epsilon) * A_i)
                objective += clipped - beta * kl(policy, reference)

    loss = -objective
    update(policy, loss)
```

GRPO 的训练关键点：

- 必须有 rollout：训练数据来自当前或近当前 policy 的采样。
- 必须有 reward：否则没法做组内相对 advantage。
- 不需要 value model：组内均值代替 baseline。
- 仍需要 reference/KL：防止策略过度漂移。

### 3.6 GRPO 的特点

GRPO 的优点：

- 不需要 value model，显著省显存和训练资源。
- 能在线采样当前模型输出，具备探索能力。
- 适合有自动 reward/verifier 的任务，例如数学、代码、可验证推理。
- 可以优化 reward，而不是只学习固定偏好对。

GRPO 的局限：

- 需要 rollout，训练系统比 DPO 复杂。
- reward 设计非常关键。
- reward model/verifier 有漏洞时容易 reward hacking。
- 采样组大小、reward 方差、KL、clip 等超参会影响稳定性。
- 如果 reward 很稀疏或噪声大，训练可能不稳定。

## 4. 核心区别表

| 维度                    | DPO                                  | GRPO                                           |
|-------------------------|--------------------------------------|------------------------------------------------|
| 方法类型                | 离线偏好优化                         | 在线 RL / policy optimization                  |
| 数据形态                | `(x, winner, loser)` 偏好对          | 对同一 `x` 采样一组 outputs，再打 reward       |
| 是否需要 rollout        | 不需要                               | 需要                                           |
| 是否需要 reward model   | 不显式需要                           | 需要 reward/verifier/反馈                      |
| 是否需要 value model    | 不需要                               | 不需要                                         |
| 是否有 reference model  | 通常有                               | 通常有                                         |
| 优化信号                | winner 相对 loser 更好               | 组内 reward 高低形成 advantage                 |
| 目标形式                | binary classification / preference loss | PPO-style clipped objective                 |
| 探索能力                | 弱，依赖已有偏好对                   | 强，可以从当前 policy 采样                     |
| 工程复杂度              | 低                                   | 中高                                           |
| 稳定性                  | 通常更稳                             | 更依赖 reward 和超参                           |
| 适合场景                | 有大量高质量偏好对                   | 有可靠自动 reward，且希望在线探索              |

## 5. 更直观的例子

### 5.1 用数学题理解

同一道题：

```text
x = "求解这个方程..."
```

DPO 数据：

```text
y_w = 正确且推理清楚的答案
y_l = 错误或推理差的答案
```

DPO 直接学习：

```text
P(y_w | x) 应该比 P(y_l | x) 更高
```

GRPO 数据生成：

```text
模型自己生成 8 个答案：
y_1, ..., y_8
```

用 verifier 打分：

```text
r_i = 答案是否正确 / 推理质量分
```

然后组内比较：

```text
高分答案提高概率，低分答案降低概率
```

### 5.2 用推荐理解

同一个用户上下文：

```text
x = 用户最近看了 A/B/C，当前场景是直播推荐
```

DPO：

```text
y_w = 列表 1，历史反馈或 reward model 判断更好
y_l = 列表 2，效果更差
训练目标：让模型更倾向于列表 1
```

GRPO：

```text
模型当前生成 16 个候选推荐列表
每个列表用 reward model 或线上反馈打分
在这 16 个列表内部算相对 advantage
高 reward 列表概率上升，低 reward 列表概率下降
```

推荐里 GRPO 更接近“模型自己探索多个列表，然后根据 reward 学习”；DPO 更接近“给它看一对好坏样本，让它学会偏好好的那个”。

## 6. 和 PPO 的关系

DPO 和 PPO 的关系：

- DPO 不是 PPO。
- DPO 可以被理解为把 KL-constrained RLHF 的最优解重新参数化后，变成一个直接的 preference loss。
- 它避免了显式 reward model 和 PPO 训练。

GRPO 和 PPO 的关系：

- GRPO 是 PPO 的变体。
- 它保留了 PPO 的 ratio clipping 思路。
- 它去掉了 value model，用组内 reward 估计 baseline / advantage。

所以：

```text
DPO：绕开 PPO。
GRPO：改造 PPO。
```

## 7. 在 OneRec / 生成式推荐里怎么选

### 7.1 适合 DPO 的情况

如果你已经有大量偏好对：

- 推荐列表 A 比 B 的线上效果更好。
- 精排/重排系统能给出 winner/loser。
- reward model 能稳定排序 candidate lists。
- 人工或规则能判断哪组推荐更好。

那么 DPO 很适合做离线对齐。

优点是简单、稳定、成本低。

### 7.2 适合 GRPO 的情况

如果你希望模型在当前 policy 下主动探索：

- 同一个用户生成多个候选 item/list。
- 用 reward model 或真实反馈打分。
- 想优化 watch time、CTR、CVR、负反馈、多样性等综合 reward。

那么 GRPO 类方法更合适。

优点是能突破固定偏好对的覆盖范围，但需要更复杂的 rollout/reward 训练系统。

### 7.3 推荐场景的风险

推荐里的 GRPO/RL 风险比数学题更大，因为推荐 reward 通常不是简单对错。

常见问题：

- CTR reward 容易诱导标题党。
- 时长 reward 可能损害多样性或长期满意度。
- reward model 可能被生成模型钻空子。
- 线上反馈有延迟和选择偏差。
- 推荐列表有多目标和业务约束。

因此工业生成式推荐一般会把 SFT/RSFT 和 RL 结合：

```text
SFT/RSFT 负责稳定、合法、贴近真实曝光分布
GRPO/ECPO/GBPO 这类 RL 负责向业务 reward 推进
DPO 负责用偏好对做更轻量的离线对齐
```

## 8. 面试表达模板

如果被问：“GRPO 和 DPO 有什么区别？”

可以这样答：

> DPO 是离线偏好优化，它输入的是同一个 prompt 下的 winner/loser pair，通过 log-prob ratio 的分类式 loss 直接让模型偏向 winner，不需要在线 rollout，也不显式训练 reward model。GRPO 是 PPO 的一种变体，它对同一个 prompt 从当前 policy 采样一组 response，用 reward 或 verifier 打分，通过组内 reward 均值和方差估计 advantage，再用 PPO-style clipped objective 更新模型。GRPO 不需要 critic/value model，但需要 rollout 和 reward。简单说，DPO 是“学已有偏好对”，GRPO 是“当前模型生成一组答案后按 reward 做在线强化学习”。

如果结合生成式推荐，可以补充：

> 在推荐场景里，DPO 适合已有推荐列表 A/B 的偏好对，比如 reward model 或线上反馈判断 A 好于 B；GRPO 更适合让生成式推荐模型对同一用户上下文生成多组候选 item/list，再用点击、时长、转化、负反馈、多样性等 reward 计算组内 advantage 进行优化。DPO 更简单稳定，GRPO 探索能力更强但更依赖 reward 设计和训练系统。

## 9. 参考资料

1. Direct Preference Optimization: Your Language Model is Secretly a Reward Model  
   https://arxiv.org/abs/2305.18290

2. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models  
   https://arxiv.org/abs/2402.03300
