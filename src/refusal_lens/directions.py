"""Difference-of-means refusal directions, one per layer and position.

For layer ``l`` and position ``p``::

    r[l, p] = mean(harmful_resid[l, p]) - mean(harmless_resid[l, p])

Following Arditi et al. (2024), the candidate directions are later scored by
ablation/steering effect and a single best one is selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from refusal_lens.activations import ActivationCache


@dataclass(frozen=True)
class RefusalDirections:
    """Candidate refusal directions.

    Attributes:
        directions: Unnormalized difference-of-means vectors,
            shape ``[n_layers, n_positions, d_model]``.
        layers: Layer indices along dim 0.
        positions: Token positions along dim 1.
    """

    directions: torch.Tensor
    layers: list[int]
    positions: list[int]

    def unit(self, layer: int, position: int = -1) -> torch.Tensor:
        """Return the unit-norm direction for ``(layer, position)``."""
        raise NotImplementedError


def difference_of_means(harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    """Compute ``mean(harmful) - mean(harmless)`` over the prompt dimension.

    Args:
        harmful: Activations of shape ``[..., n_harmful, d_model]``.
        harmless: Activations of shape ``[..., n_harmless, d_model]``.

    Returns:
        Tensor of shape ``[..., d_model]``.
    """
    raise NotImplementedError


def compute_refusal_directions(
    harmful: ActivationCache, harmless: ActivationCache
) -> RefusalDirections:
    """Compute a direction for every recorded ``(layer, position)``.

    Raises:
        ValueError: If the two caches cover different layers or positions.
    """
    raise NotImplementedError


def select_direction(
    directions: RefusalDirections,
    scores: torch.Tensor,
) -> tuple[int, int]:
    """Pick the best ``(layer, position)`` given a score per candidate.

    Args:
        directions: Candidate directions.
        scores: Shape ``[n_layers, n_positions]``; higher is better.

    Returns:
        ``(layer, position)`` of the selected direction.
    """
    raise NotImplementedError


def save_directions(directions: RefusalDirections, path: str | Path) -> None:
    """Save directions (tensor + layer/position metadata) as safetensors."""
    raise NotImplementedError


def load_directions(path: str | Path) -> RefusalDirections:
    """Load directions written by :func:`save_directions`."""
    raise NotImplementedError
