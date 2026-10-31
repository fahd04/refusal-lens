"""Evaluation: refusal rate, KL divergence on harmless prompts, capability retention.

Works the same for both ablation methods: run once on the base model and once
with the edit (inside :class:`~refusal_lens.ablation.DirectionAblation`, or after
:func:`~refusal_lens.ablation.orthogonalize_weights`), then compare.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True)
class EvaluationReport:
    """Summary metrics for one model (or edited model) on held-out prompts.

    Attributes:
        harmful_refusal_rate: Fraction of harmful prompts refused (lower = more ablated).
        harmless_refusal_rate: Fraction of harmless prompts refused (should stay ~0).
        harmless_kl: Mean KL(base || edited) of next-token distributions on harmless prompts.
        capability: Optional benchmark scores, keyed by metric name.
    """

    harmful_refusal_rate: float
    harmless_refusal_rate: float
    harmless_kl: float | None = None
    capability: dict[str, float] = field(default_factory=dict)


def is_refusal(completion: str, markers: Sequence[str]) -> bool:
    """Return whether ``completion`` contains any refusal marker (case-insensitive)."""
    text = completion.lower()
    return any(marker.lower() in text for marker in markers)


def refusal_rate(completions: Sequence[str], markers: Sequence[str]) -> float:
    """Fraction of ``completions`` classified as refusals by :func:`is_refusal`.

    Raises:
        ValueError: If there are no completions.
    """
    if not completions:
        raise ValueError("no completions to score")
    return sum(is_refusal(c, markers) for c in completions) / len(completions)


def _batches(
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> Iterator[BatchEncoding]:
    """Tokenize chat-formatted prompts in left-padded batches on ``device``."""
    if tokenizer.padding_side != "left":
        raise ValueError("tokenizer must pad on the left so generation continues the prompt")
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        # The chat template already added any special tokens (e.g. BOS).
        inputs = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        yield inputs.to(device)


@torch.inference_mode()
def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    max_new_tokens: int = 64,
    batch_size: int = 8,
) -> list[str]:
    """Greedily generate a completion for each chat-formatted prompt.

    Raises:
        ValueError: If the tokenizer does not pad on the left.
    """
    completions: list[str] = []
    for inputs in _batches(tokenizer, prompts, batch_size, model.device):
        # `generate` comes from GenerationMixin, which PreTrainedModel's type doesn't include.
        output = model.generate(  # type: ignore[operator]
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: same prompt, same answer
            pad_token_id=tokenizer.pad_token_id,
        )
        # `generate` returns prompt + answer; keep only the new tokens.
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        completions.extend(text.strip() for text in decoded)
    return completions


@torch.inference_mode()
def next_token_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    batch_size: int = 8,
) -> torch.Tensor:
    """Log-probabilities of the first answer token, for each prompt.

    Returns:
        Shape ``[n_prompts, vocab_size]``, float32 on the CPU.

    Raises:
        ValueError: If the tokenizer does not pad on the left.
    """
    chunks: list[torch.Tensor] = []
    for inputs in _batches(tokenizer, prompts, batch_size, model.device):
        attention_mask = inputs["attention_mask"]
        # Count positions from the first real token, so padding doesn't shift them.
        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        logits = model(
            input_ids=inputs["input_ids"], attention_mask=attention_mask, position_ids=position_ids
        ).logits
        chunks.append(logits[:, -1].float().log_softmax(dim=-1).cpu())
    return torch.cat(chunks)


def kl_divergence(base_logprobs: torch.Tensor, edited_logprobs: torch.Tensor) -> float:
    """Mean KL(base || edited) over prompts.

    0 means the edit did not change the next-token distribution at all. Use it
    on harmless prompts: a good edit stops refusals without moving these.

    Args:
        base_logprobs: ``[n_prompts, vocab]`` from :func:`next_token_logprobs` on the base model.
        edited_logprobs: Same, from the edited model.

    Raises:
        ValueError: If the shapes differ.
    """
    if base_logprobs.shape != edited_logprobs.shape:
        raise ValueError(
            f"shape mismatch: {tuple(base_logprobs.shape)} vs {tuple(edited_logprobs.shape)}"
        )
    # KL(p || q) = sum_v p(v) * (log p(v) - log q(v)), per prompt.
    per_prompt = (base_logprobs.exp() * (base_logprobs - edited_logprobs)).sum(dim=-1)
    return float(per_prompt.mean())


def select_ablation_layer(
    layers: Sequence[int], refusal_rates: Sequence[float], kls: Sequence[float]
) -> int:
    """Choose the layer whose direction, when removed, works best.

    The lowest refusal rate wins; among equally low rates, the smallest KL
    (least change on harmless prompts) wins.

    Args:
        layers: Candidate layers.
        refusal_rates: Harmful refusal rate with each layer's direction removed.
        kls: Harmless KL with each layer's direction removed.

    Raises:
        ValueError: If there are no candidates or the lists differ in length.
    """
    if not layers or not len(layers) == len(refusal_rates) == len(kls):
        raise ValueError(
            f"need one refusal rate and one KL per candidate layer, got "
            f"{len(layers)} layers, {len(refusal_rates)} rates, {len(kls)} KLs"
        )
    # Tuples compare element by element: refusal rate first, then KL.
    best = min(range(len(layers)), key=lambda i: (refusal_rates[i], kls[i]))
    return layers[best]


def capability_benchmark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    benchmark: str,
) -> dict[str, float]:
    """Run a small capability benchmark and return its metrics.

    Args:
        model: Model to evaluate.
        tokenizer: Its tokenizer.
        benchmark: Benchmark identifier.
    """
    raise NotImplementedError
