---
layout: post
title: "Kimi K3: An In-Depth Look at KDA"
title_zh: "Kimi K3：KDA 技术路线深读"
subtitle: "Part 1 — Attention Residuals [WIP]"
subtitle_zh: "第一篇 — Attention Residuals【WIP】"
date: 2026-08-09
author: Zhejian Peng
bilingual: true
wip: true
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

This is **part 1** of a longer note on the Kimi K3 / KDA stack. KDA itself is the *sequence* mixer (Kimi Linear). This post is about a different axis: **how K3 mixes information across depth** — Attention Residuals ([arXiv:2603.15031](https://arxiv.org/abs/2603.15031)).

**[WIP]** Later parts: Gated DeltaNet → KDA, the 3:1 KDA/MLA layout, and how K3 (2.8T) wires AttnRes into 93 layers.

## The problem

A PreNorm residual looks innocent:

\\[
h_l = h_{l-1} + f_l(h_{l-1}) = h_0 + \sum_{i=1}^{l} f_i(h_{i-1})
\\]

Every earlier layer is added with **weight 1**. Depth is an RNN: all history is crushed into one vector. Hidden-state size grows with depth, so each new layer is a smaller and smaller fraction of the stream — PreNorm dilution. Early information cannot be fetched back on demand.

Sequence modeling had the same bottleneck. Softmax attention replaced the RNN. AttnRes does that **for depth**.

The paper’s claim, in one line: standard residuals (and Highway-style gates) are *depth-wise linear attention*. AttnRes is *depth-wise softmax attention*.

## Three residuals

### Standard

Each layer only sees \(h_{l-1}\). Mixing weights are fixed. Cheap: one hidden state between layers. This is \(N = 1\) in the block picture (embedding isolated as its own source).

### Full AttnRes

\\[
h_l = \sum_{i=0}^{l-1} \alpha_{i \to l}\, v_i, \qquad
\alpha_{i \to l} = \mathrm{softmax}_i\big(w_l^\top \mathrm{RMSNorm}(k_i)\big)
\\]

- \(w_l \in \mathbb{R}^d\): one **learned pseudo-query per layer**, decoupled from that layer’s forward pass
- Softmax over **all previous layer outputs**, so the mix is content-dependent
- Compute \(O(L^2 d)\), store \(O(L d)\)
- At small scale, almost free: those activations are already kept for backprop
- At large scale, **pipeline + rematerialization** must ship every layer output across stages — that \(O(Ld)\) traffic is the real cost
- Queries start at **zero**, so training begins as a uniform average and does not spike

### Block AttnRes (what the 48B run actually uses)

Split \(L\) layers into \(N\) blocks:

- **Inside a block:** ordinary residual, collapsed to one block vector
- **Across blocks:** Full AttnRes over the \(N\) summaries + the embedding
- The unfinished block also exposes a **partial sum**
- Memory / communication drops from \(O(Ld)\) to \(O(Nd)\)
- Extremes: \(N = L\) is Full; \(N = 1\) is Standard
- Empirically, **\(N \approx 8\) recovers most of Full**

Kimi Linear 48B setup: 27 Transformer blocks (54 layers), **6 layers per AttnRes block → 9 blocks + embedding = 10 depth sources**.

Systems extras: cache across pipeline stages, two-phase inference + online softmax. Training overhead with PP &lt; 4%. Decode latency &lt; 2%.

| | Standard | Full AttnRes | Block AttnRes |
|---|---|---|---|
| What it sees | Only the running sum | Every prior layer output | Block summaries + partial |
| Weights | Fixed 1 | Learned softmax | Same, fewer sources |
| Depth selection | None | Finest | Close to Full |
| Extra store / ship | \(O(d)\) | \(O(Ld)\) | \(O(Nd)\) |
| Role in the paper | Baseline | Upper bound | The version that scales |

Per-token per-layer residual I/O (Table 1): Standard \(\sim 3d\), Block \(\sim 5.5d\), mHC (\(m=4\)) \(\sim 34d\). Block is much cheaper than mHC.

## Experiments

### Scaling law

Five sizes. Each size: Baseline / Block (\(N=8\)) / Full. Same hparams as the baseline on purpose — a conservative test.

- Slopes are similar; AttnRes is **lower loss on the whole compute curve**
- Largest size: **Baseline 1.719, Block 1.693, Full 1.692** (Full vs Block: 0.001)
- At 5.6 PFLOP/s-days: Block 1.692 vs Baseline 1.714 ≈ **1.25× compute** (baseline needs ~25% more compute to match)
- Full slightly beats mHC-lite; Block matches mHC with far less I/O

**Full is the ceiling. Block is “almost the same, and you can actually train it.”**

### 48B / 3B active, 1.4T tokens (Block vs same recipe)

Training dynamics (Fig. 5):

- Val loss is lower throughout; the gap widens in decay
- **Output magnitude:** baseline grows monotonically with depth. Block *resets* at block boundaries — bounded, periodic
- **Gradients:** baseline dumps large grads on the earliest layers. Block softmax makes sources compete; grads are more even

Downstream (Table 3 / official README):

| | Baseline | AttnRes |
|---|---:|---:|
| MMLU | 73.5 | **74.6** |
| GPQA-Diamond | 36.9 | **44.4** (+7.5) |
| BBH | 76.3 | **78.0** |
| TriviaQA | 69.9 | **71.8** |
| Math | 53.5 | **57.1** (+3.6) |
| HumanEval | 59.1 | **62.2** (+3.1) |
| MBPP | 72.0 | **73.9** |
| CMMLU | 82.0 | **82.9** |
| C-Eval | 79.6 | **82.5** |

Knowledge moves a little. **Multi-step reasoning and code move a lot** — consistent with later layers being able to pull earlier representations on demand.

### 16-layer ablation (Table 4, lower is better)

| Variant | Loss |
|---|---:|
| Baseline PreNorm | 1.766 |
| DenseFormer | 1.767 |
| mHC | 1.747 |
| **Full AttnRes** | **1.737** |
| + input-dependent query | 1.731 |
| input-independent mixing | 1.749 |
| sigmoid instead of softmax | 1.741 |
| no RMSNorm | 1.743 |
| Block \(S=4\) | 1.746 |

Fixed mixing is clearly worse than learned softmax — **content-dependent depth selection is doing real work**. Block is a bit behind Full, still well above baseline.

## Takeaway

K3 grew depth (K2 had 61 layers; K3 is around 93). A unit-weight residual starts to look like an RNN on that axis. AttnRes is the depth-wise Transformer; Block AttnRes is the production form.

**Next [WIP]:** KDA as a sequence mixer — Gated DeltaNet, channel-wise gates, 3:1 with MLA — and then the K3 report.

Paper: [Attention Residuals, arXiv:2603.15031](https://arxiv.org/abs/2603.15031). Code: [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals).

</div>

<div data-lang-panel="zh" hidden markdown="1">

这是 Kimi K3 / KDA 系列的 **第一篇**。KDA 本身是 *序列* 上的混合器（Kimi Linear）。这篇先讲另一条轴：**K3 怎么在深度上混合信息** — Attention Residuals（[arXiv:2603.15031](https://arxiv.org/abs/2603.15031)）。

**【WIP】** 后面几篇：Gated DeltaNet → KDA、3:1 的 KDA/MLA，以及 K3（2.8T）怎样把 AttnRes 接到约 93 层上。

## 问题

PreNorm residual 看起来无害：

\\[
h_l = h_{l-1} + f_l(h_{l-1}) = h_0 + \sum_{i=1}^{l} f_i(h_{i-1})
\\]

前面每一层都以 **权重 1** 加进去。深度方向像 RNN：历史被压成一个向量。hidden 幅度随深度涨，新层占的份额越来越小 —— 这就是 PreNorm dilution。早期信息没法按内容再取回来。

序列建模当年也是这个瓶颈。Softmax attention 换掉了 RNN。AttnRes 对 **深度** 做同一件事。

论文一句话：标准 residual（以及 Highway 这类门控）是 *depth-wise linear attention*；AttnRes 是 *depth-wise softmax attention*。

## 三种 residual

### Standard

每层只看见 \(h_{l-1}\)。混合权重固定。便宜：层间只传一个 hidden。在 block 图里，这就是 \(N = 1\)（embedding 单独作为一个 source）。

### Full AttnRes

\\[
h_l = \sum_{i=0}^{l-1} \alpha_{i \to l}\, v_i, \qquad
\alpha_{i \to l} = \mathrm{softmax}_i\big(w_l^\top \mathrm{RMSNorm}(k_i)\big)
\\]

- \(w_l \in \mathbb{R}^d\)：每层一个 **学出来的 pseudo-query**，和本层 forward 解耦
- 对 **所有前面层的 output** 做 softmax，混合随内容变
- 计算 \(O(L^2 d)\)，存储 \(O(L d)\)
- 小规模几乎免费：这些 activation 反正要给 backprop 留着
- 大规模：**pipeline + rematerialization** 必须把每层 output 跨 stage 传 —— \(O(Ld)\) 通信才是痛点
- query **零初始化**，开训接近均匀平均，不会一上来就炸

### Block AttnRes（48B 实验真正用的）

把 \(L\) 层切成 \(N\) 个 block：

- **block 内：** 还是普通 residual，压成一个 block 向量
- **block 间：** 只对 \(N\) 个摘要 + embedding 做 Full AttnRes
- 当前还没走完的 block，再多看一个 **partial sum**
- 内存 / 通信从 \(O(Ld)\) 降到 \(O(Nd)\)
- 两个极端：\(N = L\) 就是 Full；\(N = 1\) 就是 Standard
- 经验上 **\(N \approx 8\) 就能拿回 Full 的大部分收益**

Kimi Linear 48B：27 个 Transformer block（54 层），**每个 AttnRes block 6 层 → 9 个 block + embedding = 10 个 depth source**。

工程上还有：pipeline 的 cross-stage cache、推理 two-phase + online softmax。开 PP 时训练开销 &lt; 4%，decode 延迟 &lt; 2%。

| | Standard | Full AttnRes | Block AttnRes |
|---|---|---|---|
| 看见谁 | 只有累加和 | 每一层的 output | block 摘要 + partial |
| 权重 | 固定 1 | 学出来的 softmax | 同上，source 更少 |
| 深度选择 | 无 | 最细 | 接近 Full |
| 额外存 / 传 | \(O(d)\) | \(O(Ld)\) | \(O(Nd)\) |
| 论文定位 | baseline | 上限 | 能上大规模的折中 |

每 token 每层 residual I/O（Table 1）：Standard \(\sim 3d\)，Block \(\sim 5.5d\)，mHC（\(m=4\)）\(\sim 34d\)。Block 比 mHC 省很多带宽。

## 实验

### Scaling law

五个规模。每档：Baseline / Block（\(N=8\)）/ Full。超参按 baseline 选 —— 对 AttnRes 更苛刻。

- 斜率差不多；AttnRes **整条 compute 曲线 loss 都更低**
- 最大档：**Baseline 1.719，Block 1.693，Full 1.692**（Full 和 Block 只差 0.001）
- 在 5.6 PFLOP/s-days：Block 1.692 vs Baseline 1.714 ≈ **1.25× compute**（同样 loss，baseline 大约要多训 25%）
- Full 略优于 mHC-lite；Block 和 mHC 差不多，I/O 低很多

**Full 是上限。Block 是「几乎同样准，而且能真正训」。**

### 48B / 3B active，1.4T token（Block vs 同配方）

训练动态（Fig. 5）：

- val loss 全程更低，decay 阶段差距拉大
- **output magnitude：** baseline 随深度单调涨。Block 在 block 边界被「重置」—— 有界、呈周期
- **gradient：** baseline 最浅层梯度特别大。Block 的 softmax 让 source 抢概率，梯度更均匀

下游（Table 3 / 官方 README）：

| | Baseline | AttnRes |
|---|---:|---:|
| MMLU | 73.5 | **74.6** |
| GPQA-Diamond | 36.9 | **44.4**（+7.5） |
| BBH | 76.3 | **78.0** |
| TriviaQA | 69.9 | **71.8** |
| Math | 53.5 | **57.1**（+3.6） |
| HumanEval | 59.1 | **62.2**（+3.1） |
| MBPP | 72.0 | **73.9** |
| CMMLU | 82.0 | **82.9** |
| C-Eval | 79.6 | **82.5** |

知识类小涨。**多步推理和代码涨最多** —— 和「后面的层能按需把前面的表示捞回来」一致。

### 16 层消融（Table 4，越小越好）

| 变体 | Loss |
|---|---:|
| Baseline PreNorm | 1.766 |
| DenseFormer | 1.767 |
| mHC | 1.747 |
| **Full AttnRes** | **1.737** |
| + input-dependent query | 1.731 |
| input-independent mixing | 1.749 |
| sigmoid 代替 softmax | 1.741 |
| 去掉 RMSNorm | 1.743 |
| Block \(S=4\) | 1.746 |

固定混合明显差于学出来的 softmax —— **内容相关的深度选择是真的在干活**。Block 比 Full 差一点，仍明显好于 baseline。

## 小结

K3 把深度加上去了（K2 是 61 层，K3 大约 93 层）。权重全是 1 的 residual 在这条轴上开始像 RNN。AttnRes 是深度方向的 Transformer；Block AttnRes 是能上生产的形态。

**下一篇【WIP】：** 作为序列混合器的 KDA — Gated DeltaNet、channel-wise gate、和 MLA 的 3:1，然后再接到 K3 报告。

论文：[Attention Residuals, arXiv:2603.15031](https://arxiv.org/abs/2603.15031)。代码：[MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)。

</div>
