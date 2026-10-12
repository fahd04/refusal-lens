"""Harmful/harmless prompt sets and chat-template formatting."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from refusal_lens.config import PromptSetConfig

# Local file suffix -> `datasets` builder that can read it.
_LOCAL_BUILDERS = {".json": "json", ".jsonl": "json", ".csv": "csv", ".txt": "text"}


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


def _read_instructions(config: PromptSetConfig) -> list[str]:
    """Read the instruction column from a local file or a Hugging Face dataset."""
    path = Path(config.dataset)
    if path.is_file():
        builder = _LOCAL_BUILDERS.get(path.suffix.lower())
        if builder is None:
            raise ValueError(f"{path}: unsupported file type; use one of {sorted(_LOCAL_BUILDERS)}")
        dataset = load_dataset(builder, data_files=str(path), split="train")
    else:
        dataset = load_dataset(config.dataset, split=config.split)

    if config.column not in dataset.column_names:
        raise ValueError(
            f"{config.dataset}: no column {config.column!r}; available: {dataset.column_names}"
        )
    return [str(text) for text in dataset[config.column]]


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
    # Strip whitespace, drop blanks, drop duplicates (keeping the first one).
    stripped = (text.strip() for text in _read_instructions(config))
    instructions = list(dict.fromkeys(text for text in stripped if text))

    needed = config.n_train + config.n_eval
    if len(instructions) < needed:
        raise ValueError(
            f"{config.dataset}: need {needed} prompts (n_train + n_eval) "
            f"but only {len(instructions)} usable rows"
        )

    random.Random(seed).shuffle(instructions)
    return PromptSplit(
        train=instructions[: config.n_train],
        eval=instructions[config.n_train : needed],
    )


def load_prompt_sets(harmful: PromptSetConfig, harmless: PromptSetConfig, seed: int) -> PromptSets:
    """Load both harmful and harmless prompt splits.

    Args:
        harmful: Harmful-instruction source.
        harmless: Harmless-instruction source.
        seed: Shuffle seed.
    """
    return PromptSets(
        harmful=load_prompt_split(harmful, seed),
        harmless=load_prompt_split(harmless, seed),
    )


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
    formatted: list[str] = []
    for instruction in instructions:
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": instruction})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not isinstance(text, str):
            raise TypeError(f"apply_chat_template returned {type(text).__name__}, expected str")
        formatted.append(text)
    return formatted
