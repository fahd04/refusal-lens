"""Plots of refusal directions across layers and prompts.

All functions return a :class:`matplotlib.figure.Figure` and never call
``plt.show()``, so they work headless (CI) and in notebooks alike.
"""

from __future__ import annotations

from pathlib import Path

import torch
from matplotlib.figure import Figure

from refusal_lens.activations import ActivationCache
from refusal_lens.directions import RefusalDirections

HARMFUL_COLOR = "tab:red"
HARMLESS_COLOR = "tab:blue"


def _position_index(positions: list[int], position: int) -> int:
    """Index of ``position`` along the positions dimension."""
    if position not in positions:
        raise ValueError(f"position {position} not recorded; positions={positions}")
    return positions.index(position)


def plot_direction_norms(directions: RefusalDirections, position: int = -1) -> Figure:
    """Line plot of ``||r[l]||`` versus layer ``l``."""
    p = _position_index(directions.positions, position)
    norms = directions.directions[:, p].norm(dim=-1)

    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.subplots()
    ax.plot(directions.layers, norms.numpy(), marker="o")
    ax.set_xlabel("layer")
    ax.set_ylabel("norm of mean(harmful) - mean(harmless)")
    ax.set_title(f"Refusal direction norm per layer (position {position})")
    ax.grid(alpha=0.3)
    return fig


def plot_direction_similarity(directions: RefusalDirections, position: int = -1) -> Figure:
    """Heatmap of pairwise cosine similarity between per-layer directions."""
    p = _position_index(directions.positions, position)
    # normalize() leaves an all-zero vector at zero, so it gets similarity 0, not NaN.
    unit = torch.nn.functional.normalize(directions.directions[:, p], dim=-1)
    similarity = unit @ unit.T

    fig = Figure(figsize=(6, 5), layout="constrained")
    ax = fig.subplots()
    image = ax.imshow(similarity.numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, label="cosine similarity")
    ticks = range(0, len(directions.layers), max(1, len(directions.layers) // 10))
    ax.set_xticks(list(ticks), [str(directions.layers[i]) for i in ticks])
    ax.set_yticks(list(ticks), [str(directions.layers[i]) for i in ticks])
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    ax.set_title(f"Similarity between layer directions (position {position})")
    return fig


def plot_projections(
    harmful: ActivationCache,
    harmless: ActivationCache,
    direction: torch.Tensor,
    position: int = -1,
) -> Figure:
    """Per-layer distribution of each prompt's projection onto ``direction``.

    Shows how well the direction separates harmful from harmless prompts at
    every layer: the line is the mean over prompts, the band is ±1 standard deviation.

    Raises:
        ValueError: If the caches cover different layers, or ``direction`` has the wrong size.
    """
    if harmful.layers != harmless.layers:
        raise ValueError(f"layers differ: {harmful.layers} vs {harmless.layers}")
    d_model = harmful.activations.shape[-1]
    if direction.shape != (d_model,):
        raise ValueError(f"direction shape {tuple(direction.shape)}, expected d_model={d_model}")
    unit = direction.float() / direction.norm()

    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.subplots()
    for cache, label, color in [
        (harmful, "harmful", HARMFUL_COLOR),
        (harmless, "harmless", HARMLESS_COLOR),
    ]:
        p = _position_index(cache.positions, position)
        projections = cache.activations[:, p] @ unit  # [n_layers, n_prompts]
        mean, std = projections.mean(dim=-1).numpy(), projections.std(dim=-1).numpy()
        ax.plot(cache.layers, mean, marker="o", label=label, color=color)
        ax.fill_between(cache.layers, mean - std, mean + std, alpha=0.2, color=color)
    ax.set_xlabel("layer")
    ax.set_ylabel("projection onto refusal direction")
    ax.set_title(f"Harmful vs harmless prompts along the direction (position {position})")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_layer_scores(scores: torch.Tensor, layers: list[int], metric: str) -> Figure:
    """Plot a selection score (e.g. refusal rate after ablation) per candidate layer.

    Raises:
        ValueError: If there is not one score per layer.
    """
    if scores.shape != (len(layers),):
        raise ValueError(f"scores shape {tuple(scores.shape)}, expected ({len(layers)},)")

    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.subplots()
    ax.plot(layers, scores.numpy(), marker="o")
    ax.set_xlabel("layer")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per candidate layer")
    ax.grid(alpha=0.3)
    return fig


def save_figure(fig: Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save ``fig`` to ``path`` (format from the suffix), creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path
