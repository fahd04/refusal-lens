"""Shared fixtures. Tests run on CPU with tiny random models only: no downloads."""

from __future__ import annotations

import os

# Fail fast if anything tries to reach the Hugging Face Hub.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path

import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
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


@pytest.fixture
def example_config_path() -> Path:
    return REPO_ROOT / "configs" / "qwen2.5-1.5b.yaml"
