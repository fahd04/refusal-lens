"""Difference-of-means refusal directions, one per layer and position.

For layer ``l`` and position ``p``::

    r[l, p] = mean(harmful_resid[l, p]) - mean(harmless_resid[l, p])

Following Arditi et al. (2024), the candidate directions are later scored by
ablation/steering effect and a single best one is selected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from refusal_lens.activations import ActivationCache

# Key of the tensor inside the safetensors file.
_TENSOR_KEY = "directions"


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
        """Return the unit-norm direction for ``(layer, position)``.

        Raises:
            ValueError: If ``(layer, position)`` was not recorded, or the direction is zero.
        """
        if layer not in self.layers or position not in self.positions:
            raise ValueError(
                f"(layer={layer}, position={position}) not recorded; "
                f"layers={self.layers}, positions={self.positions}"
            )
        vector = self.directions[self.layers.index(layer), self.positions.index(position)]
        norm = vector.norm()
        if norm == 0:
            raise ValueError(f"direction at layer {layer}, position {position} is zero")
        return vector.div(norm)


def difference_of_means(harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    """Compute ``mean(harmful) - mean(harmless)`` over the prompt dimension.

    Args:
        harmful: Activations of shape ``[..., n_harmful, d_model]``.
        harmless: Activations of shape ``[..., n_harmless, d_model]``.

    Returns:
        Tensor of shape ``[..., d_model]``.

    Raises:
        ValueError: If the shapes differ in anything but the prompt dimension.
    """
    # Everything except the prompt dimension (second to last) must match.
    if harmful.shape[:-2] != harmless.shape[:-2] or harmful.shape[-1] != harmless.shape[-1]:
        raise ValueError(
            f"shape mismatch: harmful {tuple(harmful.shape)}, harmless {tuple(harmless.shape)}"
        )
    return harmful.mean(dim=-2) - harmless.mean(dim=-2)


def compute_refusal_directions(
    harmful: ActivationCache, harmless: ActivationCache
) -> RefusalDirections:
    """Compute a direction for every recorded ``(layer, position)``.

    Raises:
        ValueError: If the two caches cover different layers or positions.
    """
    if harmful.layers != harmless.layers:
        raise ValueError(f"layers differ: {harmful.layers} vs {harmless.layers}")
    if harmful.positions != harmless.positions:
        raise ValueError(f"positions differ: {harmful.positions} vs {harmless.positions}")
    return RefusalDirections(
        directions=difference_of_means(harmful.activations, harmless.activations),
        layers=list(harmful.layers),
        positions=list(harmful.positions),
    )


def select_direction(
    directions: RefusalDirections,
    scores: torch.Tensor,
) -> tuple[int, int]:
    """Pick the best ``(layer, position)`` given a score per candidate.

    Args:
        directions: Candidate directions.
        scores: Shape ``[n_layers, n_positions]``; higher is better. NaN scores are skipped.

    Returns:
        ``(layer, position)`` of the selected direction.

    Raises:
        ValueError: If ``scores`` does not have one entry per candidate.
    """
    expected = (len(directions.layers), len(directions.positions))
    if tuple(scores.shape) != expected:
        raise ValueError(f"scores shape {tuple(scores.shape)}, expected {expected}")
    # NaN would win argmax; treat it as the worst possible score instead.
    flat = scores.nan_to_num(nan=float("-inf")).flatten()
    layer_idx, position_idx = divmod(int(flat.argmax()), expected[1])
    return directions.layers[layer_idx], directions.positions[position_idx]


def save_directions(directions: RefusalDirections, path: str | Path) -> None:
    """Save directions (tensor + layer/position metadata) as safetensors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # safetensors metadata must be strings, so the lists are stored as JSON.
    metadata = {
        "layers": json.dumps(directions.layers),
        "positions": json.dumps(directions.positions),
    }
    save_file({_TENSOR_KEY: directions.directions.contiguous()}, str(path), metadata=metadata)


def load_directions(path: str | Path) -> RefusalDirections:
    """Load directions written by :func:`save_directions`."""
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata()
        tensor = f.get_tensor(_TENSOR_KEY)
    return RefusalDirections(
        directions=tensor,
        layers=json.loads(metadata["layers"]),
        positions=json.loads(metadata["positions"]),
    )
