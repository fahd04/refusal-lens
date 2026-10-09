"""Evaluation: refusal rate, KL divergence on harmless prompts, capability retention."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


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
    raise NotImplementedError


@torch.inference_mode()
def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    max_new_tokens: int = 64,
    batch_size: int = 8,
) -> list[str]:
    """Greedily generate a completion for each chat-formatted prompt."""
    raise NotImplementedError


def refusal_rate(completions: Sequence[str], markers: Sequence[str]) -> float:
    """Fraction of ``completions`` classified as refusals by :func:`is_refusal`."""
    raise NotImplementedError


@torch.inference_mode()
def kl_divergence(
    base: PreTrainedModel,
    edited: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    batch_size: int = 8,
) -> float:
    """Mean KL(base || edited) over last-token next-token distributions.

    Measures how much the edit perturbs behavior on prompts that should be
    unaffected (typically harmless ones).
    """
    raise NotImplementedError


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
