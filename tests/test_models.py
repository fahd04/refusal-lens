from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from refusal_lens.config import ModelConfig
from refusal_lens.models import (
    LlamaAdapter,
    Qwen2Adapter,
    get_adapter,
    load_model_and_tokenizer,
)

D_MODEL = 32
N_LAYERS = 2

# --- adapters ------------------------------------------------------------------


def test_get_adapter_picks_architecture(tiny_model: PreTrainedModel) -> None:
    expected = {"qwen2": Qwen2Adapter, "llama": LlamaAdapter}[tiny_model.config.model_type]

    assert isinstance(get_adapter(tiny_model), expected)


def test_get_adapter_unknown_type_raises(tiny_model: PreTrainedModel) -> None:
    tiny_model.config.model_type = "gpt2"

    with pytest.raises(KeyError, match="gpt2"):
        get_adapter(tiny_model)


def test_decoder_layers(tiny_model: PreTrainedModel) -> None:
    adapter = get_adapter(tiny_model)

    assert isinstance(adapter.decoder_layers, nn.ModuleList)
    assert adapter.num_layers == N_LAYERS


def test_embedding_is_the_input_embedding(tiny_model: PreTrainedModel) -> None:
    adapter = get_adapter(tiny_model)

    assert adapter.embedding is tiny_model.get_input_embeddings()
    assert adapter.d_model == D_MODEL


@pytest.mark.parametrize("layer_idx", range(N_LAYERS))
def test_residual_writers_project_into_d_model(tiny_model: PreTrainedModel, layer_idx: int) -> None:
    adapter = get_adapter(tiny_model)
    layer = adapter.decoder_layers[layer_idx]
    writers = adapter.residual_writers(layer_idx)

    assert writers == [adapter.attn_out_proj(layer_idx), adapter.mlp_down_proj(layer_idx)]
    for writer in writers:
        assert isinstance(writer, nn.Linear)
        assert writer.out_features == D_MODEL
        # Must live inside the requested layer, not some other one.
        assert any(writer is m for m in layer.modules())
    assert writers[0] is not writers[1]


def test_zeroing_residual_writers_makes_layer_identity(tiny_model: PreTrainedModel) -> None:
    """If the adapter found the right modules, a layer with zeroed writers adds nothing."""
    adapter = get_adapter(tiny_model)
    captured: dict[str, torch.Tensor] = {}
    layer = adapter.decoder_layers[0]
    layer.register_forward_pre_hook(
        lambda _m, args, kwargs: captured.__setitem__(
            "in", (args or (kwargs["hidden_states"],))[0]
        ),
        with_kwargs=True,
    )
    layer.register_forward_hook(
        lambda _m, _args, out: captured.__setitem__(
            "out", out[0] if isinstance(out, tuple) else out
        )
    )
    with torch.no_grad():
        for writer in adapter.residual_writers(0):
            writer.weight.zero_()
            if writer.bias is not None:
                writer.bias.zero_()
        tiny_model(torch.tensor([[1, 2, 3, 4]]))

    torch.testing.assert_close(captured["out"], captured["in"])


# --- loading -------------------------------------------------------------------


def _load(
    model_dir: Path, dtype: torch.dtype = torch.float32
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    config = ModelConfig(id=str(model_dir))
    return load_model_and_tokenizer(config, torch.device("cpu"), dtype)


def test_load_returns_eval_model_on_device(tiny_model_dir: Path) -> None:
    model, _ = _load(tiny_model_dir)

    assert isinstance(model, PreTrainedModel)
    assert not model.training
    assert model.device == torch.device("cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_load_respects_dtype(tiny_model_dir: Path, dtype: torch.dtype) -> None:
    model, _ = _load(tiny_model_dir, dtype)

    assert next(model.parameters()).dtype == dtype


def test_load_freezes_parameters(tiny_model_dir: Path) -> None:
    model, _ = _load(tiny_model_dir)

    assert not any(p.requires_grad for p in model.parameters())


def test_tokenizer_is_ready_for_batching(tiny_model_dir: Path) -> None:
    _, tokenizer = _load(tiny_model_dir)

    assert tokenizer.pad_token is not None
    # Left padding keeps the last real token at position -1 for every prompt.
    assert tokenizer.padding_side == "left"
    batch = tokenizer(["how do i bake bread", "write poem"], padding=True, return_tensors="pt")
    assert (batch["input_ids"][:, -1] != tokenizer.pad_token_id).all()


def test_load_reproduces_saved_weights(tiny_model_dir: Path, tiny_model: PreTrainedModel) -> None:
    model, _ = _load(tiny_model_dir)
    ids = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, tiny_model(ids).logits)
