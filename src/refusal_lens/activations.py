"""Record residual-stream activations with forward hooks.

Convention: ``resid[l]`` is the residual stream at the *input* of decoder layer
``l`` (so ``resid[0]`` is the token embedding). Activations are recorded for
the configured token positions, counted from the end of the prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import torch
from torch.utils.hooks import RemovableHandle
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

    Raises:
        ValueError: If a layer is out of range, or a position is not negative.
    """

    def __init__(
        self,
        adapter: ArchitectureAdapter,
        layers: Sequence[int],
        positions: Sequence[int] = (-1,),
    ) -> None:
        bad_layers = [layer for layer in layers if not 0 <= layer < adapter.num_layers]
        if bad_layers:
            raise ValueError(f"layer(s) {bad_layers} out of range 0..{adapter.num_layers - 1}")
        if not positions or any(p >= 0 for p in positions):
            raise ValueError(f"positions must be non-empty and negative, got {list(positions)}")

        self.adapter = adapter
        self.layers = list(layers)
        self.positions = list(positions)
        # One list of chunks per layer; each chunk is [n_positions, batch, d_model].
        self._chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in self.layers}
        self._handles: list[RemovableHandle] = []

    def _make_hook(self, layer: int) -> Any:
        def hook(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            # The layer's input is the residual stream: [batch, seq, d_model].
            hidden = args[0] if args else kwargs["hidden_states"]
            picked = hidden[:, self.positions, :]  # [batch, n_positions, d_model]
            self._chunks[layer].append(picked.transpose(0, 1).detach().float().cpu())

        return hook

    def __enter__(self) -> ResidualStreamRecorder:
        for layer in self.layers:
            module = self.adapter.decoder_layers[layer]
            handle = module.register_forward_pre_hook(self._make_hook(layer), with_kwargs=True)
            self._handles.append(handle)
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

    def collect(self) -> ActivationCache:
        """Concatenate everything recorded so far into an :class:`ActivationCache`.

        Raises:
            RuntimeError: If no forward pass ran while recording.
        """
        if not self._chunks[self.layers[0]]:
            raise RuntimeError("nothing recorded: run the model inside the `with` block")
        per_layer = [torch.cat(self._chunks[layer], dim=1) for layer in self.layers]
        return ActivationCache(
            activations=torch.stack(per_layer),
            layers=list(self.layers),
            positions=list(self.positions),
        )


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

    Raises:
        ValueError: If there are no prompts or the tokenizer does not pad on the left.
    """
    if not prompts:
        raise ValueError("no prompts to record")
    if tokenizer.padding_side != "left":
        raise ValueError("tokenizer must pad on the left so position -1 is the last real token")

    model = adapter.model
    with ResidualStreamRecorder(adapter, layers, positions) as recorder:
        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start : start + batch_size])
            # The chat template already added any special tokens (e.g. BOS).
            inputs = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            # Count positions from the first real token, so padding doesn't shift them.
            position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
            model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
    return recorder.collect()
