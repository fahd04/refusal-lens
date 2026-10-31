"""Shared fixtures. Tests run on CPU with tiny random models only: no downloads."""

from __future__ import annotations

import json
import os

# Fail fast if anything tries to reach the Hugging Face Hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path

import pytest
import torch
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_TINY_DIMS = {
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 64,
}

# ChatML, the format Qwen2.5 uses. Kept minimal; real templates add more.
_CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> "
    "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
)
_SPECIAL_TOKENS = ["<unk>", "<|im_start|>", "<|im_end|>"]
_WORDS = [
    "system",
    "user",
    "assistant",
    "how",
    "do",
    "i",
    "make",
    "a",
    "bomb",
    "cake",
    "write",
    "poem",
    "about",
    "the",
    "sea",
    "steal",
    "car",
    "bake",
    "bread",
    "hack",
    "into",
    "account",
    "explain",
    "photosynthesis",
    "to",
    "child",
]


def _tiny_model(arch: str) -> PreTrainedModel:
    torch.manual_seed(0)
    model: PreTrainedModel
    if arch == "qwen2":
        model = Qwen2ForCausalLM(Qwen2Config(**_TINY_DIMS))
    elif arch == "llama":
        model = LlamaForCausalLM(LlamaConfig(**_TINY_DIMS))
    else:
        raise ValueError(arch)
    return model.to("cpu").eval()


@pytest.fixture(params=["qwen2", "llama"])
def tiny_model(request: pytest.FixtureRequest) -> PreTrainedModel:
    """Tiny randomly initialized causal LM on CPU, once per supported architecture."""
    return _tiny_model(request.param)


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS + _WORDS)}
    assert len(vocab) <= _TINY_DIMS["vocab_size"]
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    # No pad token on purpose: like many chat models, loading code must set one.
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        chat_template=_CHAT_TEMPLATE,
    )


@pytest.fixture
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Word-level tokenizer built in memory, with a ChatML chat template and no pad token."""
    return _tiny_tokenizer()


@pytest.fixture
def tiny_model_dir(tmp_path: Path, tiny_model: PreTrainedModel) -> Path:
    """``tiny_model`` + ``tiny_tokenizer`` saved with ``save_pretrained``, loadable offline."""
    out = tmp_path / f"tiny-{tiny_model.config.model_type}"
    tiny_model.save_pretrained(out)
    _tiny_tokenizer().save_pretrained(out)
    return out


@pytest.fixture
def example_config_path() -> Path:
    return REPO_ROOT / "configs" / "qwen2.5-1.5b.yaml"


def _prompt_file(path: Path, verbs: list[str], nouns: list[str]) -> Path:
    rows = [{"text": f"how do i {v} a {n}"} for v in verbs for n in nouns]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def run_config_path(tmp_path: Path, tiny_model_dir: Path) -> Path:
    """Full run config for ``tiny_model`` with local prompt files; outputs go to ``tmp_path``."""
    harmful = _prompt_file(
        tmp_path / "harmful.jsonl", ["steal", "hack"], ["car", "account", "bomb"]
    )
    harmless = _prompt_file(
        tmp_path / "harmless.jsonl", ["bake", "write"], ["cake", "bread", "poem"]
    )
    config = {
        "name": "tiny",
        "seed": 0,
        "output_dir": str(tmp_path / "outputs"),
        "model": {"id": str(tiny_model_dir), "device": "cpu", "dtype": "float32"},
        "prompts": {
            "harmful": {"dataset": str(harmful), "n_train": 4, "n_eval": 2},
            "harmless": {"dataset": str(harmless), "n_train": 4, "n_eval": 2},
            "batch_size": 3,
        },
        "evaluation": {"max_new_tokens": 4},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path
