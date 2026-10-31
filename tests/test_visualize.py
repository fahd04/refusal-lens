from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

from refusal_lens.activations import ActivationCache
from refusal_lens.directions import RefusalDirections
from refusal_lens.visualize import (
    plot_direction_norms,
    plot_direction_similarity,
    plot_layer_scores,
    plot_projections,
    save_figure,
)

D_MODEL = 8
LAYERS = [0, 1, 2, 3]
POSITIONS = [-1, -2]


@pytest.fixture
def directions() -> RefusalDirections:
    torch.manual_seed(0)
    return RefusalDirections(
        directions=torch.randn(len(LAYERS), len(POSITIONS), D_MODEL),
        layers=LAYERS,
        positions=POSITIONS,
    )


def _cache(n_prompts: int, shift: float) -> ActivationCache:
    acts = torch.randn(len(LAYERS), len(POSITIONS), n_prompts, D_MODEL)
    acts[..., 0] += shift
    return ActivationCache(activations=acts, layers=LAYERS, positions=POSITIONS)


# --- norms -----------------------------------------------------------------------


def test_norms_plot_one_point_per_layer(directions: RefusalDirections) -> None:
    fig = plot_direction_norms(directions, position=-2)

    assert isinstance(fig, Figure)
    (line,) = fig.axes[0].get_lines()
    np.testing.assert_allclose(line.get_xdata(), LAYERS)
    np.testing.assert_allclose(line.get_ydata(), directions.directions[:, 1].norm(dim=-1).numpy())


def test_norms_unknown_position_raises(directions: RefusalDirections) -> None:
    with pytest.raises(ValueError, match="position"):
        plot_direction_norms(directions, position=-5)


# --- similarity heatmap ------------------------------------------------------------


def test_similarity_heatmap_is_cosine_matrix(directions: RefusalDirections) -> None:
    fig = plot_direction_similarity(directions)

    matrix = np.asarray(fig.axes[0].get_images()[0].get_array())
    unit = torch.nn.functional.normalize(directions.directions[:, 0], dim=-1)
    np.testing.assert_allclose(matrix, (unit @ unit.T).numpy(), atol=1e-6)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-6)


def test_similarity_heatmap_has_colorbar(directions: RefusalDirections) -> None:
    fig = plot_direction_similarity(directions)

    assert len(fig.axes) == 2  # heatmap + colorbar


def test_similarity_handles_zero_direction() -> None:
    """Layer 0 is often exactly zero (same last token); it must not crash or give NaN."""
    vectors = torch.randn(3, 1, D_MODEL)
    vectors[0] = 0
    dirs = RefusalDirections(directions=vectors, layers=[0, 1, 2], positions=[-1])

    matrix = np.asarray(plot_direction_similarity(dirs).axes[0].get_images()[0].get_array())

    assert not np.isnan(matrix).any()


# --- projections -------------------------------------------------------------------


def test_projections_show_mean_per_layer() -> None:
    torch.manual_seed(0)
    harmful, harmless = _cache(6, shift=5.0), _cache(9, shift=0.0)
    direction = torch.zeros(D_MODEL)
    direction[0] = 2.0  # not unit length: the plot must normalize it

    fig = plot_projections(harmful, harmless, direction)

    harmful_line, harmless_line = fig.axes[0].get_lines()
    np.testing.assert_allclose(harmful_line.get_xdata(), LAYERS)
    np.testing.assert_allclose(
        harmful_line.get_ydata(), harmful.activations[:, 0, :, 0].mean(-1).numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(
        harmless_line.get_ydata(), harmless.activations[:, 0, :, 0].mean(-1).numpy(), rtol=1e-5
    )
    assert harmful_line.get_label() == "harmful"
    assert harmless_line.get_label() == "harmless"


def test_projections_mismatched_caches_raise() -> None:
    harmful = _cache(4, 0.0)
    harmless = ActivationCache(
        activations=torch.randn(2, 2, 4, D_MODEL), layers=[0, 1], positions=POSITIONS
    )

    with pytest.raises(ValueError, match="layers"):
        plot_projections(harmful, harmless, torch.ones(D_MODEL))


def test_projections_wrong_direction_size_raises() -> None:
    with pytest.raises(ValueError, match="d_model"):
        plot_projections(_cache(4, 0.0), _cache(4, 0.0), torch.ones(D_MODEL + 1))


# --- layer scores ------------------------------------------------------------------


def test_layer_scores_plot_and_label() -> None:
    scores = torch.tensor([0.2, 0.9, 0.5, 0.1])

    fig = plot_layer_scores(scores, LAYERS, metric="refusal drop")

    ax = fig.axes[0]
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), scores.numpy())
    assert ax.get_ylabel() == "refusal drop"


def test_layer_scores_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="scores"):
        plot_layer_scores(torch.zeros(3), LAYERS, metric="x")


# --- saving ------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".png", ".svg"])
def test_save_figure(tmp_path: Path, directions: RefusalDirections, suffix: str) -> None:
    path = tmp_path / "plots" / f"norms{suffix}"

    out = save_figure(plot_direction_norms(directions), path)

    assert out == path
    assert path.is_file()
    assert path.stat().st_size > 0
