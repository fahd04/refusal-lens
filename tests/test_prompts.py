from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from refusal_lens import prompts
from refusal_lens.config import PromptSetConfig
from refusal_lens.prompts import PromptSets, format_chat, load_prompt_sets, load_prompt_split

INSTRUCTIONS = [f"instruction number {i}" for i in range(20)]


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _config(path: Path, n_train: int = 8, n_eval: int = 4, column: str = "text") -> PromptSetConfig:
    return PromptSetConfig(dataset=str(path), column=column, n_train=n_train, n_eval=n_eval)


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:
    return _jsonl(tmp_path / "prompts.jsonl", [{"text": t} for t in INSTRUCTIONS])


# --- loading -------------------------------------------------------------------


def test_split_sizes(prompt_file: Path) -> None:
    split = load_prompt_split(_config(prompt_file, n_train=8, n_eval=4), seed=0)

    assert len(split.train) == 8
    assert len(split.eval) == 4


def test_train_and_eval_are_disjoint_and_from_source(prompt_file: Path) -> None:
    split = load_prompt_split(_config(prompt_file, n_train=12, n_eval=8), seed=0)

    assert not set(split.train) & set(split.eval)
    assert set(split.train) | set(split.eval) <= set(INSTRUCTIONS)


def test_same_seed_same_split(prompt_file: Path) -> None:
    a = load_prompt_split(_config(prompt_file), seed=7)
    b = load_prompt_split(_config(prompt_file), seed=7)

    assert a == b


def test_different_seed_different_split(prompt_file: Path) -> None:
    a = load_prompt_split(_config(prompt_file), seed=1)
    b = load_prompt_split(_config(prompt_file), seed=2)

    assert a != b


def test_split_is_shuffled(prompt_file: Path) -> None:
    split = load_prompt_split(_config(prompt_file, n_train=8), seed=0)

    assert split.train != INSTRUCTIONS[:8]


def test_too_few_rows_raises(prompt_file: Path) -> None:
    with pytest.raises(ValueError, match="20"):
        load_prompt_split(_config(prompt_file, n_train=15, n_eval=10), seed=0)


def test_missing_column_raises(prompt_file: Path) -> None:
    with pytest.raises(ValueError, match="prompt"):
        load_prompt_split(_config(prompt_file, column="prompt"), seed=0)


def test_blank_rows_are_dropped_and_text_is_stripped(tmp_path: Path) -> None:
    rows = [{"text": f"  padded {i}  "} for i in range(6)] + [{"text": "   "}, {"text": ""}]
    path = _jsonl(tmp_path / "p.jsonl", rows)

    split = load_prompt_split(_config(path, n_train=4, n_eval=2), seed=0)

    assert sorted(split.train + split.eval) == [f"padded {i}" for i in range(6)]


def test_duplicates_are_dropped(tmp_path: Path) -> None:
    """A duplicate could land in both train and eval and leak between them."""
    rows = [{"text": t} for t in INSTRUCTIONS[:6]] * 3
    path = _jsonl(tmp_path / "p.jsonl", rows)

    split = load_prompt_split(_config(path, n_train=4, n_eval=2), seed=0)

    assert sorted(split.train + split.eval) == sorted(INSTRUCTIONS[:6])


@pytest.mark.parametrize("suffix", [".csv", ".txt"])
def test_other_local_formats(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"p{suffix}"
    header = "text\n" if suffix == ".csv" else ""
    path.write_text(header + "\n".join(INSTRUCTIONS) + "\n", encoding="utf-8")

    split = load_prompt_split(_config(path, n_train=8, n_eval=4), seed=0)

    assert set(split.train + split.eval) <= set(INSTRUCTIONS)


def test_hub_dataset_uses_id_and_split(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_load_dataset(path: str, *, split: str, **_: Any) -> Dataset:
        calls.append((path, split))
        return Dataset.from_list([{"text": t} for t in INSTRUCTIONS])

    monkeypatch.setattr(prompts, "load_dataset", fake_load_dataset)
    config = PromptSetConfig(dataset="someone/some-dataset", split="test", n_train=8, n_eval=4)

    split = load_prompt_split(config, seed=0)

    assert calls == [("someone/some-dataset", "test")]
    assert len(split.train) == 8


def test_load_prompt_sets(tmp_path: Path) -> None:
    harmful = _jsonl(tmp_path / "harmful.jsonl", [{"text": f"bad {i}"} for i in range(10)])
    harmless = _jsonl(tmp_path / "harmless.jsonl", [{"text": f"good {i}"} for i in range(10)])

    sets = load_prompt_sets(_config(harmful, 4, 2), _config(harmless, 4, 2), seed=0)

    assert isinstance(sets, PromptSets)
    assert all(t.startswith("bad") for t in sets.harmful.train + sets.harmful.eval)
    assert all(t.startswith("good") for t in sets.harmless.train + sets.harmless.eval)


# --- chat formatting -----------------------------------------------------------


def test_format_chat_wraps_each_instruction(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    formatted = format_chat(tiny_tokenizer, ["bake bread", "write poem"])

    assert formatted == [
        "<|im_start|> user bake bread <|im_end|> <|im_start|> assistant ",
        "<|im_start|> user write poem <|im_end|> <|im_start|> assistant ",
    ]


def test_format_chat_system_prompt(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    (formatted,) = format_chat(tiny_tokenizer, ["bake bread"], system_prompt="explain to child")

    assert formatted.startswith("<|im_start|> system explain to child <|im_end|> ")
    assert formatted.endswith("<|im_start|> user bake bread <|im_end|> <|im_start|> assistant ")


def test_format_chat_empty(tiny_tokenizer: PreTrainedTokenizerFast) -> None:
    assert format_chat(tiny_tokenizer, []) == []
