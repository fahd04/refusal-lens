from __future__ import annotations

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from refusal_lens.activations import ActivationCache, ResidualStreamRecorder, record_activations
from refusal_lens.models import ArchitectureAdapter, get_adapter

PROMPTS = [
    "<|im_start|> user how do i make a cake <|im_end|> <|im_start|> assistant ",
    "<|im_start|> user write a poem about the sea <|im_end|> <|im_start|> assistant ",
    "<|im_start|> user bake bread <|im_end|> <|im_start|> assistant ",
]


@pytest.fixture
def adapter(tiny_model: PreTrainedModel) -> ArchitectureAdapter:
    return get_adapter(tiny_model)


@pytest.fixture
def tokenizer(tiny_tokenizer: PreTrainedTokenizerFast) -> PreTrainedTokenizerFast:
    # Same setup as `load_model_and_tokenizer`.
    tiny_tokenizer.pad_token = tiny_tokenizer.eos_token
    tiny_tokenizer.padding_side = "left"
    return tiny_tokenizer


def _hidden_states(
    adapter: ArchitectureAdapter, tokenizer: PreTrainedTokenizerFast, prompt: str
) -> tuple[torch.Tensor, ...]:
    """Reference: HF's own residual stream for a single unpadded prompt."""
    inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        out = adapter.model(**inputs, output_hidden_states=True)
    return tuple(out.hidden_states)


# --- recorder ------------------------------------------------------------------


def test_recorder_shape_and_dtype(adapter: ArchitectureAdapter) -> None:
    input_ids = torch.randint(0, 20, (3, 7))

    with ResidualStreamRecorder(adapter, layers=[0, 1], positions=[-1, -2]) as rec:
        adapter.model(input_ids=input_ids)
    cache = rec.collect()

    assert isinstance(cache, ActivationCache)
    assert cache.activations.shape == (2, 2, 3, adapter.d_model)
    assert cache.activations.dtype == torch.float32
    assert cache.activations.device.type == "cpu"
    assert cache.layers == [0, 1]
    assert cache.positions == [-1, -2]


def test_recorder_concatenates_batches(adapter: ArchitectureAdapter) -> None:
    with ResidualStreamRecorder(adapter, layers=[1]) as rec:
        adapter.model(input_ids=torch.randint(0, 20, (2, 5)))
        adapter.model(input_ids=torch.randint(0, 20, (3, 6)))

    assert rec.collect().activations.shape == (1, 1, 5, adapter.d_model)


def test_layer_zero_is_the_embedding(adapter: ArchitectureAdapter) -> None:
    input_ids = torch.randint(0, 20, (2, 5))

    with ResidualStreamRecorder(adapter, layers=[0]) as rec:
        adapter.model(input_ids=input_ids)

    expected = adapter.embedding(input_ids)[:, -1].float()
    torch.testing.assert_close(rec.collect().activations[0, 0], expected)


def test_hooks_removed_on_exit(adapter: ArchitectureAdapter) -> None:
    with ResidualStreamRecorder(adapter, layers=[0, 1]):
        assert all(layer._forward_pre_hooks for layer in adapter.decoder_layers)

    assert not any(layer._forward_pre_hooks for layer in adapter.decoder_layers)


def test_hooks_removed_on_error(adapter: ArchitectureAdapter) -> None:
    with pytest.raises(RuntimeError, match="boom"), ResidualStreamRecorder(adapter, layers=[0]):
        raise RuntimeError("boom")

    assert not any(layer._forward_pre_hooks for layer in adapter.decoder_layers)


def test_collect_before_any_forward_raises(adapter: ArchitectureAdapter) -> None:
    with ResidualStreamRecorder(adapter, layers=[0]) as rec:
        pass

    with pytest.raises(RuntimeError, match="nothing recorded"):
        rec.collect()


@pytest.mark.parametrize("layers", [[2], [-1]])
def test_bad_layer_raises(adapter: ArchitectureAdapter, layers: list[int]) -> None:
    with pytest.raises(ValueError, match="layer"):
        ResidualStreamRecorder(adapter, layers=layers)


@pytest.mark.parametrize("positions", [[0], [1], []])
def test_bad_positions_raise(adapter: ArchitectureAdapter, positions: list[int]) -> None:
    with pytest.raises(ValueError, match="position"):
        ResidualStreamRecorder(adapter, layers=[0], positions=positions)


# --- record_activations --------------------------------------------------------


def test_matches_hf_hidden_states(
    adapter: ArchitectureAdapter, tokenizer: PreTrainedTokenizerFast
) -> None:
    """`resid[l]` is HF's `hidden_states[l]`: the input to decoder layer `l`."""
    cache = record_activations(adapter, tokenizer, PROMPTS[:1], layers=[0, 1], positions=[-1, -3])

    hidden = _hidden_states(adapter, tokenizer, PROMPTS[0])
    for i, layer in enumerate([0, 1]):
        for j, pos in enumerate([-1, -3]):
            torch.testing.assert_close(cache.activations[i, j, 0], hidden[layer][0, pos])


def test_left_padding_does_not_change_activations(
    adapter: ArchitectureAdapter, tokenizer: PreTrainedTokenizerFast
) -> None:
    """Short prompts get padded in a batch; their activations must not change."""
    batched = record_activations(adapter, tokenizer, PROMPTS, layers=[0, 1], batch_size=3)
    one_by_one = record_activations(adapter, tokenizer, PROMPTS, layers=[0, 1], batch_size=1)

    torch.testing.assert_close(batched.activations, one_by_one.activations)


def test_prompt_order_is_kept(
    adapter: ArchitectureAdapter, tokenizer: PreTrainedTokenizerFast
) -> None:
    cache = record_activations(adapter, tokenizer, PROMPTS, layers=[1], batch_size=2)

    for i, prompt in enumerate(PROMPTS):
        expected = _hidden_states(adapter, tokenizer, prompt)[1][0, -1]
        torch.testing.assert_close(cache.activations[0, 0, i], expected)


def test_right_padding_raises(
    adapter: ArchitectureAdapter, tokenizer: PreTrainedTokenizerFast
) -> None:
    tokenizer.padding_side = "right"

    with pytest.raises(ValueError, match="left"):
        record_activations(adapter, tokenizer, PROMPTS, layers=[0])


def test_no_prompts_raises(
    adapter: ArchitectureAdapter, tokenizer: PreTrainedTokenizerFast
) -> None:
    with pytest.raises(ValueError, match="prompt"):
        record_activations(adapter, tokenizer, [], layers=[0])
