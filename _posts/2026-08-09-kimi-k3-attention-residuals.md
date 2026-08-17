---
layout: post
title: "Kimi K3: An In-Depth Look at KDA"
title_zh: "Kimi K3：KDA 技术路线深读"
subtitle: "Kimi Delta Attention across the sequence, Attention Residuals across depth"
subtitle_zh: "序列方向的 Kimi Delta Attention，深度方向的 Attention Residuals"
date: 2026-08-09
author: Zhejian Peng
bilingual: true
math: true
catalog: true
tags:
  - Kimi
  - KDA
  - Attention
  - LLM
---

{::options parse_block_html="true" /}

<div data-lang-panel="en" markdown="1">

Kimi K3 ([arXiv:2607.24653](https://arxiv.org/abs/2607.24653)) is a 2.8T-parameter MoE with 104B active parameters, 93 layers, and a 1M-token context. Two architectural bets carry the report:

- **Kimi Delta Attention (KDA)** — how information moves along the **sequence**
- **Attention Residuals (AttnRes)** — how information moves along **depth**

## Part 1 — Kimi Delta Attention

### What KDA is replacing

Softmax attention keeps every token it has ever seen: generating token $$t$$ compares the query against $$t$$ cached keys, so per-token cost and KV cache both grow with the context. At 1M tokens that cache *is* the deployment bill.

Linear attention takes the other side of the trade. Drop the softmax and the whole history collapses into one fixed-size matrix $$\mathbf{S} \in \mathbb{R}^{d_k \times d_v}$$ carried from token to token:

$$
\mathbf{S}_t = \mathbf{S}_{t-1} + k_t v_t^\top, \qquad o_t = \mathbf{S}_t^\top q_t
$$

Constant memory, constant work per token, unbounded context. The catch is that the state never gets bigger, so **everything in KDA is about what to write into it and what to erase.**

### The state is an associative memory

Read the update again as a memory. Writing $$kv^\top$$ stores the pair $$(k, v)$$; reading is a dot product against every key at once:

$$
\mathbf{S}^\top q = \sum_i (k_i^\top q)\, v_i
$$

If the keys are orthonormal, querying $$k_1$$ returns exactly $$v_1$$ — a $$d_k \times d_v$$ matrix holds $$d_k$$ clean slots. Reality is messier in two ways.

**Keys collide.** With $$k_3$$ at 45° to $$k_1$$, storing $$(k_1, v_1)$$ then $$(k_3, v_2)$$ and reading at $$k_1$$ gives $$v_1 + 0.707\,v_2$$. Crosstalk.

**Rewrites pile up.** Store `x = 1`, later `x = 3`; plain accumulation hands back *both*:

```text
write (k1,v1), (k2,v2), then (k1,v1')     read at k1
  plain accumulation   ->  v1 + v1'      (stale value still there)
  delta rule, β = 1    ->  v1'           (old value erased)
```

A fixed-size memory that only ever adds becomes an increasingly blurry average of everything. Two fixes exist, and KDA uses both.

### Fix 1 — forgetting, one rate per channel

Multiply the state down before each write:

$$
\mathbf{S}_t = \operatorname{Diag}(\alpha_t)\,\mathbf{S}_{t-1} + k_tv_t^\top
$$

Mamba-2 and Gated DeltaNet use a **scalar** $$\alpha_t$$: the whole memory fades at one rate. KDA makes $$\alpha_t \in (0,1)^{d_k}$$ **channel-wise** — one retention factor per key dimension, chosen per token. Row $$j$$ of the state fades at its own speed, so the same layer can hold a fast, local channel next to one that keeps information for thousands of tokens:

| per-step log-decay $$g$$ | retention $$\alpha = e^{g}$$ | half-life |
|---|---:|---:|
| $$-0.0001$$ | 0.99990 | 6931 tokens |
| $$-0.1$$ | 0.90484 | 7 tokens |

This is also where K3 gets its sense of position. Decay makes "20 tokens ago" measurably weaker than "2 tokens ago", so the model needs **no positional encoding at all** (NoPE) — and a model with no RoPE frequencies has nothing to retune when the context window grows from 8K to 1M.

### Fix 2 — the delta rule, or writing by correction

Instead of asking what to add, ask what the memory currently gets *wrong*. The memory should satisfy $$\mathbf{S}^\top k_t = v_t$$, so define a loss and take one gradient step:

$$
\mathcal{L}(\mathbf{S}) = \tfrac{1}{2}\lVert \mathbf{S}^\top k_t - v_t \rVert^2,
\quad
\nabla_{\mathbf{S}}\mathcal{L} = k_t(\mathbf{S}^\top k_t - v_t)^\top
$$

$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t \nabla_{\mathbf{S}}\mathcal{L}
= \left(\mathbf{I} - \beta_t k_tk_t^\top\right)\mathbf{S}_{t-1} + \beta_t k_tv_t^\top
$$

That is the delta rule, and **a linear-attention layer is doing online gradient descent on its own memory at inference time**, with $$\beta_t$$ as the learning rate.

The geometry is the part worth keeping. Because KDA L2-normalises its keys, $$\lVert k_t\rVert = 1$$ and $$\mathbf{I} - \beta_tk_tk_t^\top$$ is a scaling along $$k_t$$ and the identity everywhere else. It **erases only what was stored at this key** and leaves the rest of the memory untouched:

- $$\beta_t = 1$$ — full overwrite: whatever was at $$k_t$$ is gone, $$v_t$$ takes its place
- $$\beta_t = 0$$ — no write at all
- in between — a partial correction toward $$v_t$$ (with $$\beta = 0.5$$ the numbers above give $$0.25\,v_1 + 0.5\,v_1'$$)

### KDA = both, and here is the whole layer

$$
\mathbf{S}_t=\left(\mathbf{I}-\beta_t k_t k_t^{\top}\right) \operatorname{Diag}(\alpha_t) \mathbf{S}_{t-1}+\beta_t k_t v_t^{\top},
\qquad \tilde{o}_t=\mathbf{S}_t^{\top} q_t
$$

Decay first, then the correction. Everything on the right is produced from the token itself:

| term | what it is | how it is produced |
|---|---|---|
| $$q_t,k_t$$ | probe / address | ShortConv → Swish → **L2Norm** |
| $$v_t$$ | payload | ShortConv → Swish |
| $$\beta_t \in (0,1)$$ | write strength | $$\operatorname{Sigmoid}(\mathbf{W}_\beta x_t)$$ |
| $$\alpha_t \in (0,1)^{d_k}$$ | per-channel retention | low-rank logits + per-head bias, then Eq. below |

The ShortConv gives each token a short local window before it becomes a key or value; the L2Norm is what makes the erase step a clean projection rather than an arbitrary rescaling.

The whole layer, in NumPy:

```python
S = np.zeros((dk, dv))
for t in range(T):
    S = (np.eye(dk) - beta[t] * np.outer(k[t], k[t])) @ (alpha[t][:, None] * S) \
        + beta[t] * np.outer(k[t], v[t])
    o[t] = S.T @ q[t]
```

### Making a serial recurrence run on tensor cores

That loop is correct and hopeless on a GPU: one tiny rank-1 update per token, no parallelism. The fix is the **chunkwise form** — split the sequence into chunks of $$C$$ tokens, stay recurrent *across* chunks, go fully parallel *inside* one. Let $$\gamma^{r} = \prod_{i \le r} \alpha^i$$ be the cumulative decay inside the chunk. Then, given the incoming state $$\mathbf{S}_{[t]}$$,

$$
\mathbf{A}_{[t]}=\operatorname{Tril}\left[(\mathbf{Q}_{[t]} \odot \mathbf{\Gamma}_{[t]})(\mathbf{K}_{[t]} / \mathbf{\Gamma}_{[t]})^{\top}\right]
$$

$$
\mathbf{O}_{[t]}=\underbrace{(\mathbf{\Gamma}_{[t]} \odot \mathbf{Q}_{[t]}) \mathbf{S}_{[t]}}_{\text{inter-chunk}}+\underbrace{\mathbf{A}_{[t]} \widetilde{\mathbf{V}}_{[t]}}_{\text{intra-chunk}}
$$

Two ideas are hiding in there:

1. **Decay becomes a rescaling.** Multiplying queries by $$\gamma^{i}$$ and dividing keys by $$\gamma^{j}$$ makes each entry of $$\mathbf{A}$$ carry exactly the decay $$\gamma^{j+1 \to i}$$ between those two positions — so a whole chunk of gated interactions is one masked matmul.
2. **The delta corrections factor out.** The chain of rank-1 erase operators inside the chunk is folded by the *UT transform* into $$\widetilde{\mathbf{V}}_{[t]} = \mathbf{U}_{[t]} - \mathbf{W}_{[t]}\mathbf{S}_{[t]}$$, a per-chunk "pseudo value" that already accounts for what earlier tokens in the chunk overwrote.

Written out in NumPy, the chunkwise form reproduces the token-by-token loop to $$10^{-16}$$ — it is an exact rewrite, not an approximation, and that is the whole reason KDA can be trained at scale.

### What K3 changed relative to Kimi Linear

KDA comes from Kimi Linear ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)). K3 makes two edits, both small on paper and both about hardware.

**1. Lower-bounded decay.** Look again at $$\mathbf{K}/\mathbf{\Gamma}$$: a *reciprocal* of a product of numbers below 1. Kimi Linear's decay came from a negative softplus, $$g = -e^{A}\operatorname{Softplus}(z) \in (-\infty, 0)$$, so that reciprocal can explode and overflow. K3 bounds it with a scaled sigmoid:

$$
g_t = g_{\min}\operatorname{Sigmoid}(e^{A_h}z_t) \in (g_{\min}, 0), \qquad g_{\min} = -5
$$

Now every step retains at least $$e^{-5} \approx 0.0067$$, cumulative log-decay over a 16-token tile stays in $$(-80, 0)$$, and the rescaling factor is at most $$e^{80} \approx 5.5\times10^{34}$$ — comfortably inside BF16's $$3.4\times10^{38}$$. The payoff is concrete: Kimi Linear had to compute the diagonal tiles with explicit position-pair arithmetic, the main intra-chunk bottleneck. With a bounded range **every tile, diagonal included, becomes a dense tensor-core matmul.** A numerical-stability bound bought a kernel rewrite.

**2. Full-rank output gate.** The low-rank gate becomes an input-dependent full-rank projection, so each token can decide channel by channel how much of the recurrent read to let through:

$$
y_t=\mathbf{W}_o\left[\operatorname{Sigmoid}(\mathbf{W}_g x_t) \odot \operatorname{RMSNorm}(\tilde{o}_t)\right]
$$

### KDA does not work alone: 3:1 with Gated MLA

A fixed state is lossy by construction, so K3 interleaves exact attention. Each block is **3 KDA layers + 1 Gated MLA layer**, and one extra MLA closes the backbone:

$$23 \times (3\,\text{KDA} + 1\,\text{MLA}) + 1\,\text{MLA} = 93$$ layers, i.e. **69 KDA and 24 MLA**.

Only those 24 layers keep a cache that grows with the sequence — against 61 full-attention layers in K2. The MLA layers are NoPE too, and carry the same full-rank output gate.

The division of labour is clean. The 69 KDA layers give recency-weighted, position-aware mixing out of a fixed $$d_k \times d_v$$ state at constant cost per token; the 24 MLA layers keep a latent cache that grows with $$T$$ and buy back what a finite state cannot hold — exact access to any earlier token.

At the Kimi Linear scale the hybrid cut KV-cache usage by up to **75%** and reached up to **6× decoding throughput at 1M context**, while beating full MLA on quality under a matched recipe.

### The state is small — but it is serial

Most of K3's KDA engineering follows from one sentence: *the state is cheap to move and impossible to skip ahead in.*

- **FlashKDA** — a CUTLASS chunkwise kernel that overlaps intra-chunk math with cross-chunk state propagation, so the SMs are not idle during the serial hand-off. It also serves prefill, as a backend of `flash-linear-attention`.
- **KDA Context Parallelism (KCP)** — the interesting one. Vanilla linear attention is a plain sum, so every rank can start from $$\mathbf{S}=\mathbf{0}$$ and the results add up. KDA cannot: in $$\mathbf{S}_t = \mathbf{M}_t\mathbf{S}_{t-1} + \beta_tk_tv_t^\top$$ with $$\mathbf{M}_t = (\mathbf{I}-\beta_tk_tk_t^\top)\operatorname{Diag}(\alpha_t)$$, the incoming state is *transformed*, not just added to. So each rank computes two local quantities — its segment's cumulative transition $$\mathbf{M}$$, and the state its own tokens generate from zero — and one all-gather plus a prefix scan composes them exactly. Messages stay fixed-size at any context length, which is what makes 1M-token training affordable.
- **Prefix caching** — KDA state checkpoints land at 512-token boundaries in the same paged pool as the MLA KV cache; a prefix is reusable only if *both* restore at the same boundary.
- **Speculative decoding** — the state updates in place, so a rejected draft cannot be rolled back. K3 caches the drafts' much smaller projected inputs and replays the accepted prefix on-chip.

### What to remember about KDA

1. A fixed-size matrix used as an associative memory, read with $$\mathbf{S}^\top q$$.
2. **Channel-wise decay** — every key channel picks its own forgetting rate, which also encodes position (hence NoPE, hence painless 1M extension).
3. **The delta rule** — write by erasing what was stored at this key first; it is one step of online gradient descent, and $$\beta_t$$ is the learning rate.
4. **Chunkwise form** — the exact same recurrence expressed as matmuls; K3's bounded decay pushes the last stubborn tile onto tensor cores.
5. **Hybrid by design** — 3:1 with Gated MLA, because a finite state should not be asked to do exact recall.

## Part 2 — Attention Residuals

KDA fixes the sequence axis. AttnRes ([arXiv:2603.15031](https://arxiv.org/abs/2603.15031)) runs the same argument down the depth axis.

### The problem

A PreNorm residual looks innocent:

$$
h_l = h_{l-1} + f_l(h_{l-1}) = h_0 + \sum_{i=1}^{l} f_i(h_{i-1})
$$

Every earlier layer is added with **weight 1**. Depth is an RNN: all history is crushed into one vector. Hidden-state magnitude grows with depth, so each new layer is a smaller and smaller fraction of the stream — PreNorm dilution. Early information cannot be fetched back on demand.

Sequence modeling had the same bottleneck, and softmax attention replaced the RNN. AttnRes does that **for depth**: a standard residual is depth-wise *linear* attention; AttnRes is depth-wise *softmax* attention.

### Three residuals

**Standard.** Each layer sees only $$h_{l-1}$$, with fixed mixing weights and one hidden state travelling between layers — $$N = 1$$ below.

**Full AttnRes.** Every layer output becomes a key/value, and each layer picks among them:

$$
h_l = \sum_{i=0}^{l-1} \alpha_{i \to l}\, v_i, \qquad
\alpha_{i \to l} = \mathrm{softmax}_i\big(w_l^\top \mathrm{RMSNorm}(k_i)\big)
$$

- $$w_l \in \mathbb{R}^d$$: one **learned pseudo-query per layer**, decoupled from that layer's forward pass, so the mix is content-dependent
- RMSNorm on the keys stops large-magnitude layers from dominating; queries start at **zero**, so training begins as a uniform average and does not spike
- Compute $$O(L^2 d)$$, store $$O(Ld)$$. The real cost is that pipeline parallelism must ship every layer output across stages

**Block AttnRes** (what ships). Split $$L$$ layers into $$N$$ blocks: inside a block an ordinary residual, collapsed to one block vector; across blocks Full AttnRes over the $$N$$ summaries plus the embedding, with the unfinished block exposing a **partial sum**. Traffic drops from $$O(Ld)$$ to $$O(Nd)$$. $$N = L$$ is Full, $$N = 1$$ is Standard, and empirically **$$N \approx 8$$ recovers most of Full**. Kimi Linear 48B used 6 layers per block → 9 blocks + embedding = 10 depth sources, for &lt; 4% training overhead and &lt; 2% decode latency.

### Experiments

**Scaling law.** Five sizes, each with Baseline / Block ($$N=8$$) / Full, all under the baseline's hyperparameters — a deliberately conservative test. AttnRes is lower loss along the whole compute curve. Largest size: **Baseline 1.719, Block 1.693, Full 1.692**, and at 5.6 PFLOP/s-days the baseline needs about **25% more compute** to match Block. Full is the ceiling; Block is "almost the same, and you can actually train it."

**48B / 3B active, 1.4T tokens.** Validation loss is lower throughout and the gap widens during decay. Baseline output magnitude grows monotonically with depth, while Block *resets* at block boundaries; gradients even out too, since the softmax makes depth sources compete instead of dumping everything on the earliest layers.

| | Baseline | AttnRes |
|---|---:|---:|
| MMLU | 73.5 | **74.6** |
| GPQA-Diamond | 36.9 | **44.4** (+7.5) |
| Math | 53.5 | **57.1** (+3.6) |
| HumanEval | 59.1 | **62.2** (+3.1) |
| C-Eval | 79.6 | **82.5** |

Knowledge moves a little. **Multi-step reasoning and code move a lot** — consistent with later layers being able to pull earlier representations on demand.

**16-layer ablation** (loss, lower is better): Baseline PreNorm 1.766, DenseFormer 1.767, mHC 1.747, **Full AttnRes 1.737**, input-independent mixing 1.749, sigmoid instead of softmax 1.741, no RMSNorm 1.743, Block $$S=4$$ 1.746. Fixed mixing is clearly worse than learned softmax — **content-dependent depth selection is doing real work**.

### Takeaway

K3 grew depth (K2 had 61 layers; K3 has 93). A unit-weight residual starts to look like an RNN on that axis, exactly as a growing KV cache is the wrong answer on the sequence axis. So both axes get the same treatment: **selective, data-dependent retrieval instead of uniform accumulation** — KDA across tokens, AttnRes across layers. Together with Stable LatentMoE on the width axis, that is where the reported ~2.5× scaling-efficiency gain over K2 comes from.

Papers: [Kimi K3](https://arxiv.org/abs/2607.24653) · [Kimi Linear](https://arxiv.org/abs/2510.26692) · [Attention Residuals](https://arxiv.org/abs/2603.15031). Code: [flash-linear-attention](https://github.com/fla-org/flash-linear-attention).

</div>

<div data-lang-panel="zh" hidden markdown="1">

Kimi K3（[arXiv:2607.24653](https://arxiv.org/abs/2607.24653)）是 2.8T 参数的 MoE，激活 104B，93 层，1M 上下文。报告最核心的两个架构选择：

- **Kimi Delta Attention（KDA）** —— 信息沿 **序列** 方向怎么流
- **Attention Residuals（AttnRes）** —— 信息沿 **深度** 方向怎么流

## 第一部分 —— Kimi Delta Attention

### KDA 要替掉什么

Softmax attention 把见过的每个 token 都留着：生成第 $$t$$ 个 token 要和 $$t$$ 个缓存 key 比较，单 token 开销和 KV cache 都随上下文增长。到 1M 上下文，这块 cache 就是部署成本本身。

线性注意力走另一条路：去掉 softmax，整段历史被压进一个固定大小的矩阵 $$\mathbf{S} \in \mathbb{R}^{d_k \times d_v}$$，在 token 之间传递：

$$
\mathbf{S}_t = \mathbf{S}_{t-1} + k_t v_t^\top, \qquad o_t = \mathbf{S}_t^\top q_t
$$

常数内存、常数单步计算、上下文无上限。代价是状态永远不会变大，所以 **KDA 的全部问题就是：往里写什么、擦掉什么。**

### 状态就是一块联想记忆

把更新式当成记忆来读。写入 $$kv^\top$$ 就是存下 $$(k, v)$$；读取是一次对所有 key 的点积：

$$
\mathbf{S}^\top q = \sum_i (k_i^\top q)\, v_i
$$

key 正交时，用 $$k_1$$ 查询正好取回 $$v_1$$ —— 一个 $$d_k \times d_v$$ 的矩阵相当于 $$d_k$$ 个干净的槽位。现实有两处不干净。

**key 会撞。** 设 $$k_3$$ 与 $$k_1$$ 夹角 45°，先存 $$(k_1, v_1)$$ 再存 $$(k_3, v_2)$$，用 $$k_1$$ 读回来是 $$v_1 + 0.707\,v_2$$，串味了。

**改写会堆积。** 先写 `x = 1`，后写 `x = 3`，纯累加会把两个都还给你：

```text
依次写 (k1,v1), (k2,v2), (k1,v1')      用 k1 读
  纯累加        ->  v1 + v1'          （旧值还在）
  delta rule, β = 1 ->  v1'           （旧值被擦掉）
```

只加不减的定容记忆，最后就是一份越来越糊的平均。解法有两个，KDA 两个都用。

### 解法一 —— 遗忘，而且每个通道一个速率

每次写入前先把状态乘小：

$$
\mathbf{S}_t = \operatorname{Diag}(\alpha_t)\,\mathbf{S}_{t-1} + k_tv_t^\top
$$

Mamba-2 和 Gated DeltaNet 用 **标量** $$\alpha_t$$：整块记忆按同一速率衰减。KDA 把它做成 **channel-wise** 的 $$\alpha_t \in (0,1)^{d_k}$$ —— 每个 key 维度一个保留系数，逐 token 决定。状态的第 $$j$$ 行按自己的速度衰减，于是同一层里可以既有快通道，也有能记住几千 token 的慢通道：

| 单步 log-decay $$g$$ | 保留率 $$\alpha = e^{g}$$ | 半衰期 |
|---|---:|---:|
| $$-0.0001$$ | 0.99990 | 6931 tokens |
| $$-0.1$$ | 0.90484 | 7 tokens |

K3 的位置信息也从这里来。衰减让「20 个 token 之前」明显弱于「2 个 token 之前」，于是模型 **完全不需要位置编码**（NoPE）—— 没有 RoPE 频率，上下文从 8K 拉到 1M 时也就没有东西要重调。

### 解法二 —— delta rule：按误差改写

不要问该加什么，要问这块记忆现在 *错* 在哪。记忆应该满足 $$\mathbf{S}^\top k_t = v_t$$，那就定义损失、走一步梯度：

$$
\mathcal{L}(\mathbf{S}) = \tfrac{1}{2}\lVert \mathbf{S}^\top k_t - v_t \rVert^2,
\quad
\nabla_{\mathbf{S}}\mathcal{L} = k_t(\mathbf{S}^\top k_t - v_t)^\top
$$

$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \beta_t \nabla_{\mathbf{S}}\mathcal{L}
= \left(\mathbf{I} - \beta_t k_tk_t^\top\right)\mathbf{S}_{t-1} + \beta_t k_tv_t^\top
$$

这就是 delta rule。换句话说，**线性注意力层在推理时对自己的记忆做在线梯度下降**，$$\beta_t$$ 就是学习率。

几何意义更值得记住。KDA 对 key 做了 L2 归一化，$$\lVert k_t\rVert = 1$$，所以 $$\mathbf{I} - \beta_tk_tk_t^\top$$ 只沿 $$k_t$$ 方向缩放，其余方向是恒等。它 **只擦掉这个 key 上存过的东西**，不动记忆里的其他内容：

- $$\beta_t = 1$$ —— 完全覆盖：$$k_t$$ 上原来的值消失，换成 $$v_t$$
- $$\beta_t = 0$$ —— 不写
- 中间值 —— 朝 $$v_t$$ 部分修正（上面的例子取 $$\beta = 0.5$$ 得到 $$0.25\,v_1 + 0.5\,v_1'$$）

### KDA = 两个都要，整层长这样

$$
\mathbf{S}_t=\left(\mathbf{I}-\beta_t k_t k_t^{\top}\right) \operatorname{Diag}(\alpha_t) \mathbf{S}_{t-1}+\beta_t k_t v_t^{\top},
\qquad \tilde{o}_t=\mathbf{S}_t^{\top} q_t
$$

先衰减，再修正。右边所有量都由当前 token 算出来：

| 项 | 是什么 | 怎么来的 |
|---|---|---|
| $$q_t,k_t$$ | 探针 / 地址 | ShortConv → Swish → **L2Norm** |
| $$v_t$$ | 内容 | ShortConv → Swish |
| $$\beta_t \in (0,1)$$ | 写入强度 | $$\operatorname{Sigmoid}(\mathbf{W}_\beta x_t)$$ |
| $$\alpha_t \in (0,1)^{d_k}$$ | 每通道保留率 | 低秩 logits + 每头 bias，再走下面的映射 |

ShortConv 让每个 token 在变成 key / value 之前先看一个小局部窗口；L2Norm 则保证擦除那一步是干净的投影，而不是任意缩放。

整层写成 NumPy：

```python
S = np.zeros((dk, dv))
for t in range(T):
    S = (np.eye(dk) - beta[t] * np.outer(k[t], k[t])) @ (alpha[t][:, None] * S) \
        + beta[t] * np.outer(k[t], v[t])
    o[t] = S.T @ q[t]
```

### 怎么让串行递推跑上 Tensor Core

这个循环是对的，但在 GPU 上没救：每个 token 一次小小的 rank-1 更新，毫无并行度。解法是 **chunkwise 形式** —— 把序列切成 $$C$$ 个 token 的 chunk，chunk 之间保持递推，chunk 内部完全并行。记 chunk 内的累积衰减 $$\gamma^{r} = \prod_{i \le r} \alpha^i$$，给定入口状态 $$\mathbf{S}_{[t]}$$：

$$
\mathbf{A}_{[t]}=\operatorname{Tril}\left[(\mathbf{Q}_{[t]} \odot \mathbf{\Gamma}_{[t]})(\mathbf{K}_{[t]} / \mathbf{\Gamma}_{[t]})^{\top}\right]
$$

$$
\mathbf{O}_{[t]}=\underbrace{(\mathbf{\Gamma}_{[t]} \odot \mathbf{Q}_{[t]}) \mathbf{S}_{[t]}}_{\text{inter-chunk}}+\underbrace{\mathbf{A}_{[t]} \widetilde{\mathbf{V}}_{[t]}}_{\text{intra-chunk}}
$$

里面藏着两个技巧：

1. **衰减变成缩放。** query 乘 $$\gamma^{i}$$、key 除 $$\gamma^{j}$$，$$\mathbf{A}$$ 的每个元素就正好带上两个位置之间的衰减 $$\gamma^{j+1 \to i}$$ —— 整个 chunk 的带门控交互变成一次带 mask 的矩阵乘。
2. **delta 修正可以提出来。** chunk 内那一串 rank-1 擦除算子，被 *UT transform* 折叠成 $$\widetilde{\mathbf{V}}_{[t]} = \mathbf{U}_{[t]} - \mathbf{W}_{[t]}\mathbf{S}_{[t]}$$，一个已经把「chunk 内更早的 token 覆盖了什么」算进去的伪 value。

我用 NumPy 把 chunkwise 形式写出来对齐逐 token 循环，误差在 $$10^{-16}$$ 量级 —— 它是精确改写，不是近似，这也正是 KDA 能大规模训练的原因。

### 相对 Kimi Linear，K3 改了什么

KDA 来自 Kimi Linear（[arXiv:2510.26692](https://arxiv.org/abs/2510.26692)）。K3 改了两处，纸面上都很小，但都是冲着硬件去的。

**1. 有下界的衰减。** 再看 $$\mathbf{K}/\mathbf{\Gamma}$$：一串小于 1 的数连乘之后取 *倒数*。Kimi Linear 用负 softplus，$$g = -e^{A}\operatorname{Softplus}(z) \in (-\infty, 0)$$，倒数可以炸掉溢出。K3 改成带缩放的 sigmoid：

$$
g_t = g_{\min}\operatorname{Sigmoid}(e^{A_h}z_t) \in (g_{\min}, 0), \qquad g_{\min} = -5
$$

于是每步至少保留 $$e^{-5} \approx 0.0067$$，16 token 的小 tile 上累积 log-decay 落在 $$(-80, 0)$$，缩放因子最大 $$e^{80} \approx 5.5\times10^{34}$$，稳稳在 BF16 的 $$3.4\times10^{38}$$ 之内。收益很实在：Kimi Linear 的对角 tile 必须走显式的 position-pair 计算，这正是 chunk 内的主要瓶颈；范围有界之后，**包括对角在内的每个 tile 都变成稠密的 Tensor Core 矩阵乘**。一个数值稳定性的下界，换来一次 kernel 重写。

**2. 全秩输出门。** 低秩门换成随输入变化的全秩投影，每个 token 可以逐通道决定放多少递推读出的内容出去：

$$
y_t=\mathbf{W}_o\left[\operatorname{Sigmoid}(\mathbf{W}_g x_t) \odot \operatorname{RMSNorm}(\tilde{o}_t)\right]
$$

### KDA 不单独工作：和 Gated MLA 3:1 混合

定容状态天生有损，所以 K3 周期性插入精确注意力。每个 block 是 **3 层 KDA + 1 层 Gated MLA**，backbone 末尾再补一层 MLA：

$$23 \times (3\,\text{KDA} + 1\,\text{MLA}) + 1\,\text{MLA} = 93$$ 层，也就是 **69 层 KDA + 24 层 MLA**。

只有这 24 层带着随序列增长的 cache —— 对比 K2 的 61 层全注意力。MLA 层同样是 NoPE，并带上同样的全秩输出门。

分工很清楚：69 层 KDA 用固定的 $$d_k \times d_v$$ 状态提供偏近期、带位置感的混合，单 token 开销恒定；24 层 MLA 留一份随 $$T$$ 增长的 latent cache，买回有限状态装不下的东西 —— 对任意早先 token 的精确访问。

在 Kimi Linear 的规模上，这种混合把 KV cache 用量最多降了 **75%**，1M 上下文下 decoding 吞吐最高 **6 倍**，同时在相同配方下质量还优于全 MLA。

### 状态很小 —— 但它是串行的

K3 在 KDA 上的工程，基本都能由一句话推出来：*状态搬运便宜，但没法跳着算。*

- **FlashKDA** —— 基于 CUTLASS 的 chunkwise kernel，把 chunk 内计算和跨 chunk 的状态传递重叠起来，串行交接时 SM 不空转；它同时服务 prefill，是 `flash-linear-attention` 的一个后端。
- **KDA Context Parallelism（KCP）** —— 最有意思的一块。普通线性注意力是纯加法，各 rank 从 $$\mathbf{S}=\mathbf{0}$$ 出发再求和就行。KDA 不行：$$\mathbf{S}_t = \mathbf{M}_t\mathbf{S}_{t-1} + \beta_tk_tv_t^\top$$，其中 $$\mathbf{M}_t = (\mathbf{I}-\beta_tk_tk_t^\top)\operatorname{Diag}(\alpha_t)$$，入口状态是被 *变换*，不只是被加。于是每个 rank 本地算两样东西 —— 本段的累积转移 $$\mathbf{M}$$，以及本地 token 从零生成的状态 —— 一次 all-gather 加一次前缀扫描就能精确拼起来。通信量与上下文长度无关，这才让 1M token 的训练付得起。
- **Prefix cache** —— KDA 状态 checkpoint 按 512 token 边界写入，和 MLA KV cache 共用同一套分页池；两边都能在同一边界恢复，这段前缀才可复用。
- **投机解码** —— 状态原地更新，草稿被拒绝没法回滚。K3 改成缓存草稿那份小得多的投影输入，在片上重放已接受的前缀。

### KDA 记住这五点

1. 一个定容矩阵当联想记忆用，读取就是 $$\mathbf{S}^\top q$$。
2. **channel-wise 衰减** —— 每个 key 通道自己挑遗忘速率，顺带编码位置（所以 NoPE，所以 1M 扩展不用改任何位置编码）。
3. **delta rule** —— 写之前先擦掉这个 key 上的旧值；本质是一步在线梯度下降，$$\beta_t$$ 就是学习率。
4. **chunkwise 形式** —— 同一个递推的精确矩阵乘写法；K3 的有界衰减把最后一块难啃的对角 tile 也推上了 Tensor Core。
5. **混合是设计的一部分** —— 和 Gated MLA 按 3:1 交错，因为不该指望有限状态做精确召回。

## 第二部分 —— Attention Residuals

KDA 解决序列这条轴，AttnRes（[arXiv:2603.15031](https://arxiv.org/abs/2603.15031)）把同一套论证搬到深度这条轴。

### 问题

PreNorm residual 看起来无害：

$$
h_l = h_{l-1} + f_l(h_{l-1}) = h_0 + \sum_{i=1}^{l} f_i(h_{i-1})
$$

前面每一层都以 **权重 1** 加进去。深度方向像 RNN：历史被压成一个向量。hidden 幅度随深度涨，新层占的份额越来越小 —— 这就是 PreNorm dilution。早期信息没法按内容再取回来。

序列建模当年也是这个瓶颈，softmax attention 换掉了 RNN。AttnRes 对 **深度** 做同一件事：标准 residual 是深度方向的 *线性* 注意力，AttnRes 是深度方向的 *softmax* 注意力。

### 三种 residual

**Standard。** 每层只看见 $$h_{l-1}$$，混合权重固定，层间只传一个 hidden —— 也就是下面的 $$N = 1$$。

**Full AttnRes。** 每一层的 output 都变成 key/value，每层自己去挑：

$$
h_l = \sum_{i=0}^{l-1} \alpha_{i \to l}\, v_i, \qquad
\alpha_{i \to l} = \mathrm{softmax}_i\big(w_l^\top \mathrm{RMSNorm}(k_i)\big)
$$

- $$w_l \in \mathbb{R}^d$$：每层一个 **学出来的 pseudo-query**，和本层 forward 解耦，所以混合随内容变
- key 上的 RMSNorm 防止幅度大的层独占权重；query **零初始化**，开训接近均匀平均，不会一上来就炸
- 计算 $$O(L^2 d)$$，存储 $$O(Ld)$$。真正的痛点是 pipeline 并行下每层 output 都要跨 stage 传

**Block AttnRes**（真正上生产的那个）。把 $$L$$ 层切成 $$N$$ 个 block：block 内还是普通 residual，压成一个 block 向量；block 间只对 $$N$$ 个摘要 + embedding 做 Full AttnRes，当前没走完的 block 再多给一个 **partial sum**。通信从 $$O(Ld)$$ 降到 $$O(Nd)$$。$$N = L$$ 是 Full，$$N = 1$$ 是 Standard，经验上 **$$N \approx 8$$ 就能拿回 Full 的大部分收益**。Kimi Linear 48B 每 block 6 层 → 9 个 block + embedding = 10 个 depth source，训练开销 &lt; 4%，decode 延迟 &lt; 2%。

### 实验

**Scaling law。** 五个规模，每档 Baseline / Block（$$N=8$$）/ Full，超参一律按 baseline 选 —— 对 AttnRes 更苛刻。AttnRes 整条 compute 曲线 loss 都更低。最大档：**Baseline 1.719，Block 1.693，Full 1.692**；在 5.6 PFLOP/s-days 上，baseline 大约要多 **25% 算力** 才追平 Block。Full 是上限，Block 是「几乎一样准，而且真能训」。

**48B / 3B 激活，1.4T token。** val loss 全程更低，decay 阶段差距拉大。baseline 的 output magnitude 随深度单调涨，Block 在 block 边界被「重置」；梯度也更均匀，softmax 让各 depth source 相互竞争，而不是把大梯度全砸在最浅几层。

| | Baseline | AttnRes |
|---|---:|---:|
| MMLU | 73.5 | **74.6** |
| GPQA-Diamond | 36.9 | **44.4**（+7.5） |
| Math | 53.5 | **57.1**（+3.6） |
| HumanEval | 59.1 | **62.2**（+3.1） |
| C-Eval | 79.6 | **82.5** |

知识类小涨，**多步推理和代码涨最多** —— 和「后面的层能按需把前面的表示捞回来」一致。

**16 层消融**（loss，越小越好）：Baseline PreNorm 1.766，DenseFormer 1.767，mHC 1.747，**Full AttnRes 1.737**，固定混合 1.749，用 sigmoid 代替 softmax 1.741，去掉 RMSNorm 1.743，Block $$S=4$$ 1.746。固定混合明显差于学出来的 softmax —— **内容相关的深度选择是真的在干活**。

### 小结

K3 把深度加上去了（K2 是 61 层，K3 是 93 层）。权重全是 1 的 residual 在这条轴上开始像 RNN，就像不断增长的 KV cache 在序列这条轴上是错的答案。于是两条轴用同一套办法：**用有选择、随内容变的检索，取代均匀累加** —— 序列上是 KDA，深度上是 AttnRes。再加上宽度方向的 Stable LatentMoE，这就是报告里相对 K2 约 2.5 倍 scaling 效率提升的来源。

论文：[Kimi K3](https://arxiv.org/abs/2607.24653) · [Kimi Linear](https://arxiv.org/abs/2510.26692) · [Attention Residuals](https://arxiv.org/abs/2603.15031)。代码：[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)。

</div>
