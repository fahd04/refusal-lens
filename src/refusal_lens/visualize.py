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


def plot_direction_norms(directions: RefusalDirections, position: int = -1) -> Figure:
    """Line plot of ``||r[l]||`` versus layer ``l``."""
    raise NotImplementedError


def plot_direction_similarity(directions: RefusalDirections, position: int = -1) -> Figure:
    """Heatmap of pairwise cosine similarity between per-layer directions."""
    raise NotImplementedError


def plot_projections(
    harmful: ActivationCache,
    harmless: ActivationCache,
    direction: torch.Tensor,
    position: int = -1,
) -> Figure:
    """Per-layer distribution of each prompt's projection onto ``direction``.

    Shows how well the direction separates harmful from harmless prompts at
    every layer.
    """
    raise NotImplementedError


def plot_layer_scores(scores: torch.Tensor, layers: list[int], metric: str) -> Figure:
    """Plot a selection score (e.g. refusal rate after ablation) per candidate layer."""
    raise NotImplementedError


def save_figure(fig: Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save ``fig`` to ``path`` (format from the suffix), creating parent dirs."""
    raise NotImplementedError
