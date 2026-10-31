"""Remove a refusal direction from the model.

Two equivalent mechanisms:

* **Hook-based ablation** (temporary): at inference time, project the direction
  out of everything written to the residual stream, ``x <- x - (x . r) r``.
  Nothing in the model changes; leaving the ``with`` block restores it.
* **Weight orthogonalization** (permanent): edit every matrix that writes to
  the residual stream so it cannot write along ``r``, ``W <- W - r r^T W``.
  The result is a normal checkpoint with no runtime hooks.

Both act on the same modules (token embedding, attention ``o_proj`` and MLP
``down_proj``), so they produce the same outputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from refusal_lens.models import ArchitectureAdapter


def _unit(direction: torch.Tensor, d_model: int) -> torch.Tensor:
    """Float32 unit vector of ``direction``, checked against ``d_model``."""
    if direction.shape != (d_model,):
        raise ValueError(f"direction size {tuple(direction.shape)}, expected ({d_model},)")
    norm = direction.norm()
    if norm == 0:
        raise ValueError("direction is zero; it has no direction to remove")
    return direction.float().div(norm)


def _check_layers(adapter: ArchitectureAdapter, layers: Sequence[int] | None) -> list[int]:
    """Default to all layers; reject indices the model doesn't have."""
    if layers is None:
        return list(range(adapter.num_layers))
    bad = [layer for layer in layers if not 0 <= layer < adapter.num_layers]
    if bad:
        raise ValueError(f"layer(s) {bad} out of range 0..{adapter.num_layers - 1}")
    return list(layers)


def project_out(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove the component of ``x`` along ``direction``.

    Args:
        x: Activations of shape ``[..., d_model]``.
        direction: Shape ``[d_model]``; normalized internally.

    Returns:
        ``x - (x . r_hat) r_hat`` with the same shape and dtype as ``x``.

    Raises:
        ValueError: If ``direction`` has the wrong size or is zero.
    """
    r = _unit(direction, x.shape[-1]).to(x.device)
    x32 = x.float()
    # (x . r) is one number per token; multiply it back onto r and subtract.
    return (x32 - (x32 @ r).unsqueeze(-1) * r).to(x.dtype)


def orthogonalize_matrix(
    weight: torch.Tensor, direction: torch.Tensor, output_dim: int = 0
) -> torch.Tensor:
    """Return ``weight`` with its output component along ``direction`` removed.

    Args:
        weight: A residual-stream writer. ``nn.Linear`` weights have shape
            ``[d_model, d_in]`` (``output_dim=0``); embedding weights have shape
            ``[vocab, d_model]`` (``output_dim=1``). The axis must be given
            explicitly because ``o_proj`` can be square.
        direction: Shape ``[d_model]``; normalized internally.
        output_dim: Axis of ``weight`` that lies in the residual stream.

    Returns:
        A new tensor with the same shape and dtype as ``weight``.

    Raises:
        ValueError: If ``direction`` does not match ``weight`` along ``output_dim``.
    """
    r = _unit(direction, weight.shape[output_dim]).to(weight.device)
    w = weight.float()
    # Linear: W - r (r^T W). Embedding: W - (W r) r^T.
    along_r = torch.outer(r, r @ w) if output_dim == 0 else torch.outer(w @ r, r)
    return (w - along_r).to(weight.dtype)


class DirectionAblation:
    """Context manager that ablates ``direction`` from the residual stream via hooks.

    Each residual-stream writer (embedding, ``o_proj``, ``down_proj``) gets a
    forward hook that projects the direction out of its output, which is exactly
    what :func:`orthogonalize_weights` bakes into the weights.

    Example::

        with DirectionAblation(adapter, direction):
            model.generate(...)

    Args:
        adapter: Adapter wrapping the model.
        direction: Shape ``[d_model]``.
        layers: Layers to hook. Defaults to all layers.
        include_embedding: Also ablate the token embedding output.

    Raises:
        ValueError: If a layer is out of range or the direction is unusable.
    """

    def __init__(
        self,
        adapter: ArchitectureAdapter,
        direction: torch.Tensor,
        layers: Sequence[int] | None = None,
        include_embedding: bool = True,
    ) -> None:
        self.adapter = adapter
        self.direction = _unit(direction, adapter.d_model)
        self.layers = _check_layers(adapter, layers)
        self.include_embedding = include_embedding
        self._handles: list[RemovableHandle] = []

    def _hook(self, module: nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        # Returning a value from a forward hook replaces the module's output.
        return project_out(output, self.direction)

    def __enter__(self) -> DirectionAblation:
        modules: list[nn.Module] = [self.adapter.embedding] if self.include_embedding else []
        for layer in self.layers:
            modules.extend(self.adapter.residual_writers(layer))
        for module in modules:
            self._handles.append(module.register_forward_hook(self._hook))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _untie_output_embedding(adapter: ArchitectureAdapter) -> None:
    """Give the output layer its own copy of the weights if it shares the embedding's."""
    lm_head = adapter.model.get_submodule("lm_head")
    if isinstance(lm_head, nn.Linear) and lm_head.weight is adapter.embedding.weight:
        lm_head.weight = nn.Parameter(lm_head.weight.detach().clone(), requires_grad=False)
        # Otherwise save/reload would tie them again.
        adapter.model.config.tie_word_embeddings = False


@torch.no_grad()
def orthogonalize_weights(
    adapter: ArchitectureAdapter,
    direction: torch.Tensor,
    layers: Sequence[int] | None = None,
    include_embedding: bool = True,
) -> None:
    """Permanently orthogonalize residual-stream writers against ``direction``, in place.

    If the output layer shares its weights with the token embedding (tied
    embeddings), it is untied first so that only the embedding is edited.

    Args:
        adapter: Adapter wrapping the model to edit.
        direction: Shape ``[d_model]``.
        layers: Layers whose writers are edited. Defaults to all layers.
        include_embedding: Also orthogonalize the token embedding matrix.

    Raises:
        ValueError: If a layer is out of range or the direction is unusable.
    """
    r = _unit(direction, adapter.d_model)
    selected = _check_layers(adapter, layers)

    if include_embedding:
        _untie_output_embedding(adapter)
        embedding = adapter.embedding.weight
        embedding.copy_(orthogonalize_matrix(embedding, r, output_dim=1))
    for layer in selected:
        for writer in adapter.residual_writers(layer):
            writer.weight.copy_(orthogonalize_matrix(writer.weight, r, output_dim=0))
