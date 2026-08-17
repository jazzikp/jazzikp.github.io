---
layout: post
title: "Pre-Norm vs. Post-Norm: The Identity Path That Stabilizes LLMs"
subtitle: "Why normalization placement changes gradient flow — with a runnable JAX stress test"
description: "A visual explanation of Pre-Norm and Post-Norm Transformers, why Pre-Norm preserves an identity gradient path, and a reproducible CPU/GPU JAX demo."
date: 2026-08-16
author: Zhejian Peng
math: true
catalog: true
tags:
  - LLM
  - Transformer
  - Normalization
  - JAX
---

{::options parse_block_html="true" /}

> **The central idea:** Pre-Norm matters because every Transformer block retains a residual route that normalization does not transform. That direct identity path lets gradients cross many layers. It mainly improves the optimization stability of deep models; it does **not** imply that Pre-Norm always has greater final expressive power.

## The one-line difference

Let $$F_l$$ denote either attention or the feed-forward network, and let $$N$$ denote LayerNorm or RMSNorm. The two arrangements differ by one line:

**Pre-Norm**

$$
x_{l+1}=x_l+F_l(N(x_l))
$$

**Post-Norm**

$$
x_{l+1}=N(x_l+F_l(x_l))
$$

In Pre-Norm, only the branch computation sees normalized input. The residual stream $$x_l$$ reaches the next layer unchanged. In Post-Norm, the residual is still present, but the sum must pass through normalization before becoming $$x_{l+1}$$.

![Pre-Norm preserves a direct identity gradient path, while every Post-Norm path passes through normalization.](/img/pre-vs-post-norm-paths.webp){: loading="lazy" width="960" height="470"}

This diagram is the argument in its most useful form: **Pre-Norm has an unmodified blue highway through depth. Post-Norm has a residual connection, but no unmodified identity highway.**

## Why the identity path changes backpropagation

Differentiate one block. Pre-Norm gives

$$
\frac{\partial x_{l+1}}{\partial x_l}
=I+J_{F_l}J_N.
$$

The $$I$$ term is explicit. When the learned branch is small, as it often is near initialization, the block Jacobian is close to identity:

$$
\frac{\partial x_{l+1}}{\partial x_l}\approx I.
$$

A gradient can therefore travel through dozens of blocks without depending entirely on the attention, MLP, or normalization Jacobians.

Post-Norm instead gives

$$
\frac{\partial x_{l+1}}{\partial x_l}
=J_N(I+J_{F_l}).
$$

Every backward route includes $$J_N$$. Across a deep stack, the gradient repeatedly encounters those normalization Jacobians. Their products can rescale and rotate it, and LayerNorm projects out particular shift and scale directions. The residual addition has not disappeared; what disappeared is the **untransformed identity route**.

That distinction is more precise than saying “Post-Norm cuts the residual connection.” It does not. It makes normalization unavoidable on the path between successive residual states.

## What Xiong et al. actually established

[Xiong et al. (2020), *On Layer Normalization in the Transformer Architecture*](https://proceedings.mlr.press/v119/xiong20b.html), analyzed the models at initialization using mean-field arguments. Their key observation was a layerwise imbalance:

- In **Post-LN**, parameter gradients near the output are large at initialization. Applying the full target learning rate immediately can make the first updates unstable.
- In **Pre-LN**, gradient magnitudes are better behaved across depth, so their translation and BERT experiments could train without learning-rate warmup.

This explains why learning-rate warmup is especially important for classical Post-Norm Transformers: warmup makes the early updates small while the optimizer and network leave the fragile initialization regime.

Two qualifications matter:

1. This is an analysis of initialization under simplifying assumptions, not a theorem that every Post-Norm run diverges.
2. Modern Pre-Norm LLM training still commonly uses warmup. Warmup also helps adaptive-optimizer state, large batches, mixed precision, and other parts of the training system.

The defensible claim is therefore **“Pre-Norm is less sensitive and easier to optimize deeply,”** not “Pre-Norm never needs warmup.”

## A runnable stability stress test

The code accompanying this article trains the same small 12-layer causal Transformer three ways:

1. Pre-Norm at a constant learning rate of $$2\times10^{-3}$$;
2. Post-Norm at the same constant learning rate;
3. Post-Norm with a linear warmup lasting two thirds of the run (120 steps in the default 180-step experiment) to the same target rate.

Everything else—data, initialization seed, optimizer, depth, width, and number of updates—is held fixed. The task is deliberately simple: given an arithmetic token sequence, predict the next token. There is no gradient clipping.

![Loss and global gradient-norm curves from the reproducible Pre-Norm and Post-Norm stress test.](/img/pre-vs-post-norm-stability.webp){: loading="lazy" width="1484" height="600"}

On my CPU run, the results were:

| Configuration | Final loss | Maximum global gradient norm |
|---|---:|---:|
| Pre-Norm, no warmup | **0.00165** | **1.004** |
| Post-Norm, no warmup | 4.21205 | 6.960 |
| Post-Norm, 120-step warmup | **0.01738** | 2.759 |

The result is intentionally not “Post-Norm is incapable of learning.” Warmup rescues it. The result is that **Pre-Norm tolerates the aggressive target learning rate immediately**, while this particular Post-Norm run does not.

This is a pedagogical stress test, not a reproduction of Xiong et al.: the dataset is synthetic, the model is tiny, and a different seed or hyperparameter can move the stability boundary. The useful exercise is to sweep `--layers` and the learning rate and observe how much tuning each topology needs.

### Run it with JAX on CPU or GPU

The complete script lives beside this article:

- [`_posts/pre_norm_vs_post_norm_demo.py`](https://github.com/jazzikp/jazzikp.github.io/blob/main/_posts/pre_norm_vs_post_norm_demo.py)
- [raw measurements used for the figure](/img/pre-vs-post-norm-stability.csv)

From the repository root:

```bash
# CPU, including Apple Silicon. The full run takes about 26 seconds here.
python -m pip install --upgrade jax flax optax matplotlib pillow
python _posts/pre_norm_vs_post_norm_demo.py --device cpu

# NVIDIA GPU on Linux with JAX's CUDA 13 wheel.
python -m pip install --upgrade "jax[cuda13]" flax optax matplotlib pillow
python _posts/pre_norm_vs_post_norm_demo.py --device gpu

# Prefer a JAX GPU backend when present, otherwise use CPU.
python _posts/pre_norm_vs_post_norm_demo.py --device auto
```

The script writes both `img/pre-vs-post-norm-stability.webp` and the underlying CSV. It is JAX end to end: the model is written with Flax, gradients come from `jax.value_and_grad`, updates come from Optax, and the training step is compiled with `jax.jit`. On macOS, the supported default is Apple Silicon **CPU**; the JAX project does not currently ship an official macOS GPU backend. The architectural switch itself is only this:

```python
if self.pre_norm:
    x = x + attention(attention_norm(x))
    return x + mlp(mlp_norm(x))

x = attention_norm(x + attention(x))
return mlp_norm(x + mlp(x))
```

The compiled update is likewise ordinary JAX:

```python
loss, gradients = jax.value_and_grad(loss_fn)(params)
updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
params = optax.apply_updates(params, updates)
```

The full default run is already small enough for CPU use. For a faster smoke test, reduce the work:

```bash
python _posts/pre_norm_vs_post_norm_demo.py \
  --device cpu --layers 6 --steps 30
```

That shorter setting checks that the program and selected backend work; it is not guaranteed to reproduce the full stability separation.

## Why Pre-Norm is not simply “better”

Pre-Norm buys optimization stability by leaving the residual stream itself unconstrained inside the stack. Expanding the recurrence shows the trade-off:

$$
x_L=x_0+\sum_{l=0}^{L-1}F_l(N(x_l)).
$$

Every layer writes directly into the same accumulating stream. As $$\lVert x_l\rVert$$ grows, a new update can become a smaller fraction of the existing state. Later blocks can approach identity transformations and contribute less than their depth suggests. Later work describes related symptoms as representation collapse, residual-stream growth, massive activations, or the “curse of depth.” The exact growth law depends on initialization, training, correlations, and residual scaling; it is not universally exponential.

Post-Norm has the opposite appeal: it re-normalizes the state after every residual addition, tightly controlling forward scale. Its cost is making normalization unavoidable in backpropagation. That is why several newer architectures try to keep the best property of each design rather than treating either as universally optimal:

| Design | What it tries to preserve |
|---|---|
| **Pre-Norm** | Clean identity path and robust optimization |
| **Post-Norm** | Controlled block-output scale |
| **DeepNorm / residual scaling** | Post-Norm quality with bounded updates |
| **NormFormer** | More balanced layerwise gradients in Pre-Norm |
| **Peri-LN / sandwich norms** | Identity residual path plus bounded sublayer output |
| **Final norm** | A stable scale before the language-model head |

## The final norm has a different job

A typical modern Pre-Norm LLM also applies a final normalization:

$$
z=N_{\mathrm{final}}(x_L),\qquad
\mathrm{logits}=W_{\mathrm{vocab}}z.
$$

This does not undo the identity-path benefit. The block-level residual stream remains clean throughout depth; normalization is applied once before the output head.

Ignoring epsilon and affine parameters, LayerNorm and RMSNorm are approximately zero-homogeneous:

$$
N(c x)=N(x),\qquad c>0.
$$

The final norm therefore acts as a **scale anchor**: simply inflating $$\lVert x_L\rVert$$ cannot inflate the logits. This is separate from the role of the internal Pre-Norms:

- **Internal Pre-Norm:** stabilize each branch input and preserve the identity gradient path.
- **Final norm:** stabilize the scale presented to the language-model head.

## The design rule behind the literature

The Pre-Norm versus Post-Norm literature can be condensed into two simultaneous requirements:

1. **Keep a clean residual identity path** so gradients can cross depth.
2. **Control the size of updates written into that path** so the residual stream does not overwhelm later layers.

Plain Pre-Norm solves the first requirement extremely well, which is why it became the default for deep LLM optimization. It only partially solves the second. DeepNorm, NormFormer, Peri-LN, residual scaling, and related designs are attempts to add scale control without losing the identity highway.

## Takeaway

**Pre-Norm is important because it changes the optimization geometry of a deep Transformer.** In

$$
x_{l+1}=x_l+F_l(N(x_l)),
$$

the derivative always contains an identity term. That gives the gradient a direct route across depth and makes large models less sensitive to their earliest updates.

The benefit is primarily **trainability**, not a guarantee of superior final representation. Post-Norm can work—and may sometimes produce stronger representations—when warmup, initialization, or residual scaling controls its optimization. The modern goal is not merely “put normalization first,” but:

> **Preserve the identity path, then control what each learned branch writes into it.**

## References

- [Xiong et al., 2020. *On Layer Normalization in the Transformer Architecture.*](https://proceedings.mlr.press/v119/xiong20b.html)
- [Liu et al., 2020. *Understanding the Difficulty of Training Transformers.*](https://aclanthology.org/2020.emnlp-main.463/)
- [Shleifer et al., 2021. *NormFormer: Improved Transformer Pretraining with Extra Normalization.*](https://arxiv.org/abs/2110.09456)
- [Wang et al., 2022. *DeepNet: Scaling Transformers to 1,000 Layers.*](https://arxiv.org/abs/2203.00555)
- [Kim et al., 2025. *Peri-LN: Revisiting Normalization Layer in the Transformer Architecture.*](https://arxiv.org/abs/2502.02732)
- [Sun et al., 2025. *The Curse of Depth in Large Language Models.*](https://arxiv.org/abs/2502.05795)
