#!/usr/bin/env python3
"""Small JAX Pre-Norm vs Post-Norm optimization stress test.

This is an educational experiment, not a reproduction of Xiong et al. (2020).
It trains the same 12-layer causal Transformer on a synthetic next-token task in
three configurations:

1. Pre-Norm, constant learning rate
2. Post-Norm, constant learning rate
3. Post-Norm, linear learning-rate warmup

The default model runs on CPU and on a JAX GPU backend. It writes a CSV file and
a WebP plot.

CPU install:
    python -m pip install --upgrade jax flax optax matplotlib pillow

NVIDIA GPU install (Linux; CUDA 13 wheel):
    python -m pip install --upgrade "jax[cuda13]" flax optax matplotlib pillow

Run:
    python _posts/pre_norm_vs_post_norm_demo.py --device cpu
    python _posts/pre_norm_vs_post_norm_demo.py --device gpu
    python _posts/pre_norm_vs_post_norm_demo.py --device auto
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

from flax import linen as nn
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from PIL import Image


@dataclass(frozen=True)
class Config:
    vocab_size: int = 64
    sequence_length: int = 32
    d_model: int = 64
    heads: int = 4
    layers: int = 12
    batch_size: int = 24
    steps: int = 180
    learning_rate: float = 2e-3
    seed: int = 7


@dataclass(frozen=True)
class RunSpec:
    name: str
    pre_norm: bool
    warmup_steps: int
    color: str


RUNS = (
    RunSpec("Pre-Norm · no warmup", True, 0, "#2563eb"),
    RunSpec("Post-Norm · no warmup", False, 0, "#dc5a3a"),
    RunSpec("Post-Norm · warmup", False, -1, "#138a72"),
)


class TransformerBlock(nn.Module):
    """One attention + MLP block; pre_norm is the only topology switch."""

    d_model: int
    heads: int
    pre_norm: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        attention_norm = nn.LayerNorm(name="attention_norm")
        mlp_norm = nn.LayerNorm(name="mlp_norm")

        def attention(inputs: jax.Array) -> jax.Array:
            batch, length, _ = inputs.shape
            head_dim = self.d_model // self.heads
            qkv = nn.Dense(3 * self.d_model, name="qkv")(inputs)
            query, key, value = jnp.split(qkv, 3, axis=-1)
            query = query.reshape(batch, length, self.heads, head_dim)
            key = key.reshape(batch, length, self.heads, head_dim)
            value = value.reshape(batch, length, self.heads, head_dim)
            scores = jnp.einsum("bthd,bshd->bhts", query, key)
            scores = scores / math.sqrt(head_dim)
            causal_mask = jnp.tril(jnp.ones((length, length), dtype=jnp.bool_))
            scores = jnp.where(causal_mask[None, None, :, :], scores, -1e30)
            weights = jax.nn.softmax(scores, axis=-1)
            mixed = jnp.einsum("bhts,bshd->bthd", weights, value)
            mixed = mixed.reshape(batch, length, self.d_model)
            return nn.Dense(self.d_model, name="attention_output")(mixed)

        def mlp(inputs: jax.Array) -> jax.Array:
            hidden = nn.Dense(4 * self.d_model, name="mlp_in")(inputs)
            hidden = nn.gelu(hidden)
            return nn.Dense(self.d_model, name="mlp_out")(hidden)

        if self.pre_norm:
            x = x + attention(attention_norm(x))
            return x + mlp(mlp_norm(x))

        x = attention_norm(x + attention(x))
        return mlp_norm(x + mlp(x))


class TinyLanguageModel(nn.Module):
    config: Config
    pre_norm: bool

    @nn.compact
    def __call__(self, token_ids: jax.Array) -> jax.Array:
        embedding = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=nn.initializers.normal(1.0),
            name="embedding",
        )(token_ids)
        position = self.param(
            "position",
            nn.initializers.zeros_init(),
            (1, self.config.sequence_length, self.config.d_model),
        )
        x = embedding + position[:, : token_ids.shape[1]]
        for index in range(self.config.layers):
            x = TransformerBlock(
                self.config.d_model,
                self.config.heads,
                self.pre_norm,
                name=f"block_{index}",
            )(x)
        if self.pre_norm:
            x = nn.LayerNorm(name="final_norm")(x)
        return nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=nn.initializers.normal(0.02),
            name="output",
        )(x)


def select_device(requested: str) -> jax.Device:
    if requested == "cpu":
        return jax.devices("cpu")[0]
    if requested == "gpu":
        try:
            return jax.devices("gpu")[0]
        except RuntimeError as error:
            raise SystemExit(
                "A GPU was requested, but JAX has no GPU backend. "
                "Install the matching accelerator wheel; for NVIDIA CUDA 13, "
                'use: python -m pip install --upgrade "jax[cuda13]"'
            ) from error
    for device in jax.devices():
        if device.platform == "gpu":
            return device
    return jax.devices("cpu")[0]


def make_optimizer() -> optax.GradientTransformation:
    # The learning rate is applied separately so one compiled update can accept
    # either a constant rate or a warmup rate.
    return optax.chain(
        optax.scale_by_adam(b1=0.9, b2=0.95),
        optax.add_decayed_weights(0.01),
        optax.scale(-1.0),
    )


def make_train_step(
    model: TinyLanguageModel,
    optimizer: optax.GradientTransformation,
):
    @jax.jit
    def train_step(params, optimizer_state, source, target, learning_rate):
        def loss_fn(candidate_params):
            logits = model.apply({"params": candidate_params}, source)
            token_losses = optax.softmax_cross_entropy_with_integer_labels(
                logits, target
            )
            return jnp.mean(token_losses)

        loss, gradients = jax.value_and_grad(loss_fn)(params)
        gradient_norm = optax.global_norm(gradients)
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, params
        )
        updates = jax.tree.map(lambda update: learning_rate * update, updates)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state, loss, gradient_norm

    return train_step


def make_batch(
    config: Config, generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Create arithmetic progressions: the next token is current token + 1."""
    starts = generator.integers(
        0, config.vocab_size, size=(config.batch_size, 1), dtype=np.int32
    )
    offsets = np.arange(config.sequence_length + 1, dtype=np.int32)[None, :]
    sequence = (starts + offsets) % config.vocab_size
    return sequence[:, :-1], sequence[:, 1:]


def run_one(
    spec: RunSpec, config: Config, device: jax.Device
) -> list[dict[str, float | int | str]]:
    model = TinyLanguageModel(config, pre_norm=spec.pre_norm)
    optimizer = make_optimizer()
    dummy_tokens = jnp.zeros(
        (config.batch_size, config.sequence_length), dtype=jnp.int32, device=device
    )
    params = model.init(jax.random.PRNGKey(config.seed), dummy_tokens)["params"]
    optimizer_state = optimizer.init(params)
    train_step = make_train_step(model, optimizer)
    generator = np.random.default_rng(config.seed + 116)

    history: list[dict[str, float | int | str]] = []
    for step in range(1, config.steps + 1):
        source, target = make_batch(config, generator)
        source = jax.device_put(source, device)
        target = jax.device_put(target, device)
        learning_rate = config.learning_rate
        if spec.warmup_steps:
            learning_rate *= min(1.0, step / spec.warmup_steps)
        params, optimizer_state, loss, gradient_norm = train_step(
            params,
            optimizer_state,
            source,
            target,
            jnp.asarray(learning_rate, dtype=jnp.float32),
        )
        loss_value = float(loss)
        gradient_norm_value = float(gradient_norm)
        history.append(
            {
                "run": spec.name,
                "step": step,
                "loss": loss_value,
                "gradient_norm": gradient_norm_value,
                "learning_rate": learning_rate,
            }
        )
        if not math.isfinite(loss_value) or not math.isfinite(gradient_norm_value):
            print(f"{spec.name}: stopped at step {step} after a non-finite value")
            break

    return history


def write_csv(histories: list[list[dict[str, float | int | str]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run", "step", "loss", "gradient_norm", "learning_rate"),
        )
        writer.writeheader()
        for history in histories:
            writer.writerows(history)


def plot(
    histories: list[list[dict[str, float | int | str]]],
    run_specs: tuple[RunSpec, ...],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for spec, history in zip(run_specs, histories, strict=True):
        steps = [int(row["step"]) for row in history]
        losses = [float(row["loss"]) for row in history]
        gradients = [float(row["gradient_norm"]) for row in history]
        axes[0].plot(steps, losses, label=spec.name, color=spec.color, linewidth=2)
        axes[1].plot(steps, gradients, label=spec.name, color=spec.color, linewidth=2)

    axes[0].set_title("Optimization without gradient clipping")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_yscale("log")
    axes[1].set_title("Global gradient norm")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("L2 norm")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.grid(True, color="#d8dde6", linewidth=0.7, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle(
        "JAX stress test: same model, data, initialization, and target LR",
        fontsize=14,
        y=1.01,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    png_path = path.with_suffix(".png")
    figure.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    with Image.open(png_path) as image:
        image.convert("RGB").save(path, "WEBP", quality=90, method=6)
    png_path.unlink()


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "gpu"),
        help="auto prefers a JAX GPU backend, then CPU (default: auto)",
    )
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--layers", type=int, default=Config.layers)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "img" / "pre-vs-post-norm-stability.webp",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=repository_root / "img" / "pre-vs-post-norm-stability.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    config = Config(steps=args.steps, layers=args.layers)
    warmup_steps = max(1, 2 * config.steps // 3)
    print(
        f"device={device.platform}:{device.id} layers={config.layers} "
        f"steps={config.steps} target_lr={config.learning_rate:g} "
        f"warmup={warmup_steps}"
    )

    started = time.perf_counter()
    run_specs = tuple(
        RunSpec(
            (
                f"Post-Norm · {warmup_steps}-step warmup"
                if spec.warmup_steps < 0
                else spec.name
            ),
            spec.pre_norm,
            warmup_steps if spec.warmup_steps < 0 else spec.warmup_steps,
            spec.color,
        )
        for spec in RUNS
    )
    histories = [run_one(spec, config, device) for spec in run_specs]
    elapsed = time.perf_counter() - started
    write_csv(histories, args.csv)
    plot(histories, run_specs, args.output)

    print("\nRun summary")
    print("-" * 75)
    for spec, history in zip(run_specs, histories, strict=True):
        final_loss = float(history[-1]["loss"])
        max_gradient = max(float(row["gradient_norm"]) for row in history)
        print(
            f"{spec.name:32s} final loss={final_loss:9.5f}  "
            f"max grad={max_gradient:8.3f}"
        )
    print("-" * 75)
    print(f"finished in {elapsed:.1f}s")
    print(f"plot: {args.output}")
    print(f"data: {args.csv}")


if __name__ == "__main__":
    main()
