"""Remove a refusal direction from the model.

Two equivalent mechanisms:

* **Hook-based ablation** (reversible): at inference time, project the
  direction out of the residual stream at every layer, ``x <- x - (x . r) r``.
* **Weight orthogonalization** (permanent): edit every matrix that writes to
  the residual stream so it cannot write along ``r``, ``W <- W - r r^T W``.
  The result is a normal checkpoint with no runtime hooks.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType

import torch

from refusal_lens.models import ArchitectureAdapter


def project_out(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove the component of ``x`` along ``direction``.

    Args:
        x: Activations of shape ``[..., d_model]``.
        direction: Shape ``[d_model]``; normalized internally.

    Returns:
        ``x - (x . r_hat) r_hat`` with the same shape and dtype as ``x``.
    """
    raise NotImplementedError


def orthogonalize_matrix(weight: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Return ``weight`` with its output component along ``direction`` removed.

    Args:
        weight: A residual-stream writer. ``nn.Linear`` weights have shape
            ``[d_model, d_in]``; embedding weights have shape ``[vocab, d_model]``.
            The ``d_model`` axis is detected from ``direction``'s size.
        direction: Shape ``[d_model]``; normalized internally.

    Returns:
        A new tensor with the same shape and dtype as ``weight``.
    """
    raise NotImplementedError


class DirectionAblation:
    """Context manager that ablates ``direction`` from the residual stream via hooks.

    Example::

        with DirectionAblation(adapter, direction):
            model.generate(...)

    Args:
        adapter: Adapter wrapping the model.
        direction: Shape ``[d_model]``.
        layers: Layers to hook. Defaults to all layers.
    """

    def __init__(
        self,
        adapter: ArchitectureAdapter,
        direction: torch.Tensor,
        layers: Sequence[int] | None = None,
    ) -> None:
        self.adapter = adapter
        self.direction = direction
        self.layers = list(layers) if layers is not None else None

    def __enter__(self) -> DirectionAblation:
        raise NotImplementedError

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        raise NotImplementedError


@torch.no_grad()
def orthogonalize_weights(
    adapter: ArchitectureAdapter,
    direction: torch.Tensor,
    layers: Sequence[int] | None = None,
    include_embedding: bool = True,
) -> None:
    """Permanently orthogonalize residual-stream writers against ``direction``, in place.

    Args:
        adapter: Adapter wrapping the model to edit.
        direction: Shape ``[d_model]``.
        layers: Layers whose writers are edited. Defaults to all layers.
        include_embedding: Also orthogonalize the token embedding matrix.
    """
    raise NotImplementedError
