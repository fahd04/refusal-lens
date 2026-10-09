"""Harmful/harmless prompt sets and chat-template formatting."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from refusal_lens.config import PromptSetConfig


@dataclass(frozen=True)
class PromptSplit:
    """Disjoint train/eval instruction lists drawn from one source."""

    train: list[str]
    eval: list[str]


@dataclass(frozen=True)
class PromptSets:
    """Harmful and harmless instructions, each split into train/eval."""

    harmful: PromptSplit
    harmless: PromptSplit


def load_prompt_split(config: PromptSetConfig, seed: int) -> PromptSplit:
    """Load instructions from a dataset and draw disjoint train/eval subsets.

    Args:
        config: Dataset id/path, split, column and subset sizes.
        seed: Seed for the shuffle, so subsets are reproducible.

    Returns:
        ``n_train`` training and ``n_eval`` evaluation instructions.

    Raises:
        ValueError: If the source has fewer than ``n_train + n_eval`` rows.
    """
    raise NotImplementedError


def load_prompt_sets(harmful: PromptSetConfig, harmless: PromptSetConfig, seed: int) -> PromptSets:
    """Load both harmful and harmless prompt splits.

    Args:
        harmful: Harmful-instruction source.
        harmless: Harmless-instruction source.
        seed: Shuffle seed.
    """
    raise NotImplementedError


def format_chat(
    tokenizer: PreTrainedTokenizerBase,
    instructions: list[str],
    system_prompt: str | None = None,
) -> list[str]:
    """Wrap each instruction in the model's chat template, ending at the assistant turn.

    Args:
        tokenizer: Tokenizer providing ``apply_chat_template``.
        instructions: Raw user instructions.
        system_prompt: Optional system message prepended to every conversation.

    Returns:
        Formatted prompt strings (with the generation prompt appended).
    """
    raise NotImplementedError
