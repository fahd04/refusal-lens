"""Model/tokenizer loading and architecture adapters.

Different Hugging Face architectures name their submodules differently. An
:class:`ArchitectureAdapter` hides those differences behind a small interface
exposing exactly what refusal-direction work needs:

* the list of decoder layers (where residual-stream hooks attach),
* the token embedding matrix (which writes the initial residual stream),
* the modules that write into the residual stream inside each layer: the
  attention output projection and the MLP down projection.

Weight orthogonalization edits exactly these residual-stream writers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from refusal_lens.config import ModelConfig


class ArchitectureAdapter(ABC):
    """Maps one model architecture onto the modules refusal-lens operates on.

    Subclasses set :attr:`model_types` to the ``config.model_type`` values they
    support and are registered with :func:`register_adapter`.

    Args:
        model: The loaded causal LM.
    """

    model_types: ClassVar[tuple[str, ...]] = ()

    def __init__(self, model: PreTrainedModel) -> None:
        self.model = model

    @property
    @abstractmethod
    def decoder_layers(self) -> nn.ModuleList:
        """Decoder blocks in forward order; block ``i`` reads residual stream ``i``."""

    @property
    @abstractmethod
    def embedding(self) -> nn.Embedding:
        """Token embedding module; its ``weight`` is the ``[vocab, d_model]`` matrix."""

    @abstractmethod
    def attn_out_proj(self, layer_idx: int) -> nn.Linear:
        """Attention output projection of layer ``layer_idx`` (writes to the residual stream)."""

    @abstractmethod
    def mlp_down_proj(self, layer_idx: int) -> nn.Linear:
        """MLP down projection of layer ``layer_idx`` (writes to the residual stream)."""

    @property
    def num_layers(self) -> int:
        """Number of decoder layers."""
        return len(self.decoder_layers)

    @property
    def d_model(self) -> int:
        """Residual-stream width."""
        return int(self.embedding.weight.shape[1])

    def residual_writers(self, layer_idx: int) -> list[nn.Linear]:
        """All modules in layer ``layer_idx`` whose output is added to the residual stream."""
        return [self.attn_out_proj(layer_idx), self.mlp_down_proj(layer_idx)]


class Qwen2Adapter(ArchitectureAdapter):
    """Adapter for Qwen2 / Qwen2.5 (``Qwen2ForCausalLM``)."""

    model_types = ("qwen2",)

    @property
    def decoder_layers(self) -> nn.ModuleList:  # noqa: D102 - documented on base class
        raise NotImplementedError

    @property
    def embedding(self) -> nn.Embedding:  # noqa: D102
        raise NotImplementedError

    def attn_out_proj(self, layer_idx: int) -> nn.Linear:  # noqa: D102
        raise NotImplementedError

    def mlp_down_proj(self, layer_idx: int) -> nn.Linear:  # noqa: D102
        raise NotImplementedError


class LlamaAdapter(ArchitectureAdapter):
    """Adapter for Llama 3.x (``LlamaForCausalLM``)."""

    model_types = ("llama",)

    @property
    def decoder_layers(self) -> nn.ModuleList:  # noqa: D102
        raise NotImplementedError

    @property
    def embedding(self) -> nn.Embedding:  # noqa: D102
        raise NotImplementedError

    def attn_out_proj(self, layer_idx: int) -> nn.Linear:  # noqa: D102
        raise NotImplementedError

    def mlp_down_proj(self, layer_idx: int) -> nn.Linear:  # noqa: D102
        raise NotImplementedError


_ADAPTERS: dict[str, type[ArchitectureAdapter]] = {}


def register_adapter(cls: type[ArchitectureAdapter]) -> type[ArchitectureAdapter]:
    """Register an adapter class for each of its ``model_types``.

    Usable as a class decorator.
    """
    for model_type in cls.model_types:
        _ADAPTERS[model_type] = cls
    return cls


for _cls in (Qwen2Adapter, LlamaAdapter):
    register_adapter(_cls)


def supported_model_types() -> list[str]:
    """Return the ``config.model_type`` values with a registered adapter."""
    return sorted(_ADAPTERS)


def get_adapter(model: PreTrainedModel) -> ArchitectureAdapter:
    """Return the adapter for ``model`` based on ``model.config.model_type``.

    Raises:
        KeyError: If no adapter is registered for the model type.
    """
    model_type = model.config.model_type
    try:
        return _ADAPTERS[model_type](model)
    except KeyError:
        raise KeyError(
            f"No adapter for model_type={model_type!r}; supported: {supported_model_types()}"
        ) from None


def load_model_and_tokenizer(
    config: ModelConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM and its tokenizer, in eval mode, on ``device``.

    Args:
        config: Model id, revision and loading options.
        device: Target device.
        dtype: Weight dtype.

    Returns:
        ``(model, tokenizer)``. The tokenizer has a pad token set and left padding.
    """
    raise NotImplementedError
