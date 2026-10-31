from __future__ import annotations

import math

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from refusal_lens.ablation import DirectionAblation
from refusal_lens.evaluate import (
    generate_completions,
    is_refusal,
    kl_divergence,
    next_token_logprobs,
    refusal_rate,
    select_ablation_layer,
)
from refusal_lens.models import get_adapter

MARKERS = ["I'm sorry", "I cannot", "As an AI"]
PROMPTS = [
    "<|im_start|> user how do i make a cake <|im_end|> <|im_start|> assistant ",
    "<|im_start|> user write a poem about the sea <|im_end|> <|im_start|> assistant ",
    "<|im_start|> user bake bread <|im_end|> <|im_start|> assistant ",
]


@pytest.fixture
def tokenizer(tiny_tokenizer: PreTrainedTokenizerFast) -> PreTrainedTokenizerFast:
    # Same setup as `load_model_and_tokenizer`.
    tiny_tokenizer.pad_token = tiny_tokenizer.eos_token
    tiny_tokenizer.padding_side = "left"
    return tiny_tokenizer


# --- refusal detection -----------------------------------------------------------


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("I'm sorry, but I can't help with that.", True),
        ("i'm SORRY, no.", True),  # case-insensitive
        ("Well... as an AI language model I won't.", True),  # anywhere in the text
        ("Sure! Here is a recipe for bread.", False),
        ("", False),
    ],
)
def test_is_refusal(completion: str, expected: bool) -> None:
    assert is_refusal(completion, MARKERS) is expected


def test_refusal_rate() -> None:
    completions = ["I'm sorry.", "Sure, here you go.", "I cannot do that.", "Okay!"]

    assert refusal_rate(completions, MARKERS) == 0.5


def test_refusal_rate_empty_raises() -> None:
    with pytest.raises(ValueError, match="no completions"):
        refusal_rate([], MARKERS)


# --- generation --------------------------------------------------------------------


def test_generate_one_completion_per_prompt(
    tiny_model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast
) -> None:
    completions = generate_completions(tiny_model, tokenizer, PROMPTS, max_new_tokens=5)

    assert len(completions) == len(PROMPTS)
    assert all(isinstance(c, str) for c in completions)


def test_generate_excludes_the_prompt(
    tiny_model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast
) -> None:
    (completion,) = generate_completions(tiny_model, tokenizer, PROMPTS[:1], max_new_tokens=5)

    assert "how do i make a cake" not in completion


def test_generate_respects_max_new_tokens(
    tiny_model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast
) -> None:
    completions = generate_completions(tiny_model, tokenizer, PROMPTS, max_new_tokens=3)

    for completion in completions:
        assert len(completion.split()) <= 3  # word-level tokenizer: 1 word = 1 token


def test_generate_is_greedy_and_batch_independent(
    tiny_model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast
) -> None:
    """Same prompts, different batching (so different padding): same answers."""
    batched = generate_completions(tiny_model, tokenizer, PROMPTS, max_new_tokens=6, batch_size=3)
    single = generate_completions(tiny_model, tokenizer, PROMPTS, max_new_tokens=6, batch_size=1)

    assert batched == single


def test_generate_right_padding_raises(
    tiny_model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast
) -> None:
    tokenizer.padding_side = "right"

    with pytest.raises(ValueError, match="left"):
        generate_completions(tiny_model, tokenizer, PROMPTS)


# --- next-token distributions --------------------------------------------------------


def test_next_token_logprobs_are_distributions(
    tiny_model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast
) -> None:
    logprobs = next_token_logprobs(tiny_model, tokenizer, PROMPTS)

    assert logprobs.shape == (len(PROMPTS), tiny_model.config.vocab_size)
    assert logprobs.dtype == torch.float32
    torch.testing.assert_close(logprobs.exp().sum(-1), torch.ones(len(PROMPTS)))


def test_next_token_logprobs_match_model(
    tiny_model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast
) -> None:
    inputs = tokenizer(PROMPTS[:1], return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        expected = tiny_model(**inputs).logits[0, -1].log_softmax(-1)

    torch.testing.assert_close(next_token_logprobs(tiny_model, tokenizer, PROMPTS[:1])[0], expected)


def test_next_token_logprobs_batch_independent(
    tiny_model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast
) -> None:
    batched = next_token_logprobs(tiny_model, tokenizer, PROMPTS, batch_size=3)
    single = next_token_logprobs(tiny_model, tokenizer, PROMPTS, batch_size=1)

    torch.testing.assert_close(batched, single, atol=1e-5, rtol=1e-5)


# --- KL divergence -------------------------------------------------------------------


def test_kl_of_identical_distributions_is_zero() -> None:
    logprobs = torch.randn(4, 10).log_softmax(-1)

    assert kl_divergence(logprobs, logprobs) == pytest.approx(0.0, abs=1e-6)


def test_kl_known_value() -> None:
    p = torch.tensor([[0.5, 0.5]]).log()
    q = torch.tensor([[0.9, 0.1]]).log()

    expected = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    assert kl_divergence(p, q) == pytest.approx(expected, rel=1e-5)


def test_kl_is_mean_over_prompts() -> None:
    p = torch.tensor([[0.5, 0.5], [0.5, 0.5]]).log()
    q = torch.tensor([[0.5, 0.5], [0.9, 0.1]]).log()

    assert kl_divergence(p, q) == pytest.approx(kl_divergence(p[1:], q[1:]) / 2, rel=1e-5)


def test_kl_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        kl_divergence(torch.zeros(2, 5), torch.zeros(3, 5))


def test_ablation_moves_the_distribution(
    tiny_model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast
) -> None:
    """End to end: base vs hook-ablated model on the same prompts."""
    adapter = get_adapter(tiny_model)
    torch.manual_seed(0)
    direction = torch.randn(adapter.d_model)

    base = next_token_logprobs(tiny_model, tokenizer, PROMPTS)
    with DirectionAblation(adapter, direction):
        edited = next_token_logprobs(tiny_model, tokenizer, PROMPTS)

    assert kl_divergence(base, edited) > 0


# --- choosing the layer to ablate ----------------------------------------------------


def test_select_lowest_refusal_rate_first() -> None:
    """Stopping refusals matters most, even if another layer changes less."""
    assert select_ablation_layer([10, 14, 18], [0.5, 0.0, 0.2], [0.01, 0.9, 0.03]) == 14


def test_select_breaks_ties_by_kl() -> None:
    assert select_ablation_layer([14, 20, 26], [0.0, 0.0, 0.0], [0.88, 1.38, 0.08]) == 26


@pytest.mark.parametrize(
    ("layers", "rates", "kls"),
    [([], [], []), ([1, 2], [0.0], [0.1, 0.2]), ([1], [0.0], [0.1, 0.2])],
)
def test_select_bad_input_raises(layers: list[int], rates: list[float], kls: list[float]) -> None:
    with pytest.raises(ValueError, match="candidate"):
        select_ablation_layer(layers, rates, kls)
