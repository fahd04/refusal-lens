"""Record residual-stream activations with forward hooks.

Convention: ``resid[l]`` is the residual stream at the *input* of decoder layer
``l`` (so ``resid[0]`` is the token embedding). Activations are recorded for
the configured token positions, counted from the end of the prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType

import torch
from transformers import PreTrainedTokenizerBase

from refusal_lens.models import ArchitectureAdapter


@dataclass(frozen=True)
class ActivationCache:
    """Residual-stream activations for a batch of prompts.

    Attributes:
        activations: Tensor of shape ``[n_layers, n_positions, n_prompts, d_model]``
            on the CPU, in float32.
        layers: Layer indices along dim 0.
        positions: Token positions along dim 1 (negative = from the end).
    """

    activations: torch.Tensor
    layers: list[int]
    positions: list[int]


class ResidualStreamRecorder:
    """Context manager that registers pre-forward hooks on decoder layers.

    Example::

        with ResidualStreamRecorder(adapter, layers=[10, 12]) as rec:
            model(**batch)
        acts = rec.collect()

    Args:
        adapter: Architecture adapter for the model.
        layers: Layers to record.
        positions: Token positions to keep (negative = from the end).
    """

    def __init__(
        self,
        adapter: ArchitectureAdapter,
        layers: Sequence[int],
        positions: Sequence[int] = (-1,),
    ) -> None:
        self.adapter = adapter
        self.layers = list(layers)
        self.positions = list(positions)

    def __enter__(self) -> ResidualStreamRecorder:
        raise NotImplementedError

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        raise NotImplementedError

    def collect(self) -> ActivationCache:
        """Concatenate everything recorded so far into an :class:`ActivationCache`."""
        raise NotImplementedError


@torch.inference_mode()
def record_activations(
    adapter: ArchitectureAdapter,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    layers: Sequence[int],
    positions: Sequence[int] = (-1,),
    batch_size: int = 8,
) -> ActivationCache:
    """Run ``prompts`` through the model in batches and record the residual stream.

    Args:
        adapter: Adapter wrapping the model.
        tokenizer: Tokenizer (left padding, so position ``-1`` is the last real token).
        prompts: Already chat-formatted prompt strings.
        layers: Layers to record.
        positions: Token positions to record.
        batch_size: Prompts per forward pass.

    Returns:
        The recorded activations.
    """
    raise NotImplementedError
