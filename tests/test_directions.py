from __future__ import annotations

from pathlib import Path

import pytest
import torch

from refusal_lens.activations import ActivationCache
from refusal_lens.directions import (
    RefusalDirections,
    compute_refusal_directions,
    difference_of_means,
    load_directions,
    save_directions,
    select_direction,
)

D_MODEL = 8


def _cache(activations: torch.Tensor, layers: list[int], positions: list[int]) -> ActivationCache:
    return ActivationCache(activations=activations, layers=layers, positions=positions)


def _directions() -> RefusalDirections:
    torch.manual_seed(0)
    return RefusalDirections(
        directions=torch.randn(3, 2, D_MODEL), layers=[4, 5, 6], positions=[-1, -2]
    )


# --- difference_of_means -------------------------------------------------------


def test_difference_of_means_simple() -> None:
    harmful = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # mean [2, 3]
    harmless = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])  # mean [0, 1]

    torch.testing.assert_close(difference_of_means(harmful, harmless), torch.tensor([2.0, 2.0]))


def test_difference_of_means_keeps_leading_dims() -> None:
    harmful = torch.randn(3, 2, 5, D_MODEL)
    harmless = torch.randn(3, 2, 7, D_MODEL)

    out = difference_of_means(harmful, harmless)

    assert out.shape == (3, 2, D_MODEL)
    torch.testing.assert_close(out[1, 0], harmful[1, 0].mean(0) - harmless[1, 0].mean(0))


def test_difference_of_means_recovers_planted_direction() -> None:
    """Harmful = harmless + a fixed shift: the difference is that shift."""
    torch.manual_seed(0)
    shift = torch.randn(D_MODEL)
    harmless = torch.randn(500, D_MODEL)
    harmful = torch.randn(500, D_MODEL) + shift

    out = difference_of_means(harmful, harmless)

    assert torch.nn.functional.cosine_similarity(out, shift, dim=0) > 0.95


@pytest.mark.parametrize(
    ("harmful_shape", "harmless_shape"),
    [((4, D_MODEL), (4, D_MODEL + 1)), ((2, 4, D_MODEL), (3, 4, D_MODEL))],
)
def test_difference_of_means_shape_mismatch_raises(
    harmful_shape: tuple[int, ...], harmless_shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="shape"):
        difference_of_means(torch.zeros(harmful_shape), torch.zeros(harmless_shape))


# --- compute_refusal_directions --------------------------------------------------


def test_compute_refusal_directions() -> None:
    harmful = torch.randn(3, 2, 5, D_MODEL)
    harmless = torch.randn(3, 2, 6, D_MODEL)

    dirs = compute_refusal_directions(
        _cache(harmful, [4, 5, 6], [-1, -2]), _cache(harmless, [4, 5, 6], [-1, -2])
    )

    assert dirs.directions.shape == (3, 2, D_MODEL)
    assert dirs.layers == [4, 5, 6]
    assert dirs.positions == [-1, -2]
    torch.testing.assert_close(dirs.directions, harmful.mean(2) - harmless.mean(2))


@pytest.mark.parametrize(("layers", "positions"), [([4, 5, 7], [-1, -2]), ([4, 5, 6], [-1, -3])])
def test_compute_mismatched_caches_raise(layers: list[int], positions: list[int]) -> None:
    harmful = _cache(torch.randn(3, 2, 5, D_MODEL), [4, 5, 6], [-1, -2])
    harmless = _cache(torch.randn(3, 2, 5, D_MODEL), layers, positions)

    with pytest.raises(ValueError, match="differ"):
        compute_refusal_directions(harmful, harmless)


# --- unit ------------------------------------------------------------------------


def test_unit_has_norm_one_and_same_direction() -> None:
    dirs = _directions()

    u = dirs.unit(layer=5, position=-2)

    torch.testing.assert_close(u.norm(), torch.tensor(1.0))
    raw = dirs.directions[1, 1]
    torch.testing.assert_close(u * raw.norm(), raw)


def test_unit_defaults_to_last_position() -> None:
    dirs = _directions()

    torch.testing.assert_close(dirs.unit(6), dirs.unit(6, -1))


@pytest.mark.parametrize(("layer", "position"), [(7, -1), (4, -3)])
def test_unit_unknown_layer_or_position_raises(layer: int, position: int) -> None:
    with pytest.raises(ValueError, match="not recorded"):
        _directions().unit(layer, position)


def test_unit_zero_direction_raises() -> None:
    dirs = RefusalDirections(directions=torch.zeros(1, 1, D_MODEL), layers=[0], positions=[-1])

    with pytest.raises(ValueError, match="zero"):
        dirs.unit(0)


# --- select_direction ------------------------------------------------------------


def test_select_direction_picks_highest_score() -> None:
    scores = torch.tensor([[0.1, 0.2], [0.3, 0.9], [0.5, 0.4]])

    assert select_direction(_directions(), scores) == (5, -2)


def test_select_direction_ignores_nan() -> None:
    scores = torch.tensor([[float("nan"), 0.2], [0.3, 0.1], [0.5, 0.4]])

    assert select_direction(_directions(), scores) == (6, -1)


def test_select_direction_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        select_direction(_directions(), torch.zeros(2, 2))


# --- save / load -----------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    dirs = _directions()
    path = tmp_path / "directions.safetensors"

    save_directions(dirs, path)
    loaded = load_directions(path)

    torch.testing.assert_close(loaded.directions, dirs.directions)
    assert loaded.layers == dirs.layers
    assert loaded.positions == dirs.positions


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "run" / "nested" / "directions.safetensors"

    save_directions(_directions(), path)

    assert path.is_file()
