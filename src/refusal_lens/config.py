"""Run configuration: pydantic models loaded from YAML, plus reproducibility helpers.

A single :class:`RunConfig` describes one experiment end to end. Every CLI
command loads it, seeds all RNGs from :attr:`RunConfig.seed`, and writes the
fully resolved config (defaults included) into the run directory so the run
can be reproduced from its outputs alone.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from refusal_lens import __version__
from refusal_lens.device import DeviceName, DTypeName

RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"

GGUFOutType = Literal["f32", "f16", "bf16", "q8_0"]


class _StrictModel(BaseModel):
    """Base model that rejects unknown keys so config typos fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_StrictModel):
    """Which model to load and how."""

    id: str = Field(description="Hugging Face model id or local path.")
    revision: str | None = Field(
        default=None, description="Git revision (commit hash recommended for reproducibility)."
    )
    device: DeviceName = "auto"
    dtype: DTypeName = "auto"
    trust_remote_code: bool = False


class PromptSetConfig(_StrictModel):
    """A source of instructions (harmful or harmless) and how many to draw."""

    dataset: str = Field(description="Hugging Face dataset id or path to a local file.")
    split: str = "train"
    column: str = Field(default="text", description="Column holding the instruction text.")
    n_train: PositiveInt = Field(description="Prompts used to compute directions.")
    n_eval: PositiveInt = Field(description="Held-out prompts used for evaluation.")


class PromptsConfig(_StrictModel):
    """Harmful/harmless prompt sets and chat-template settings."""

    harmful: PromptSetConfig
    harmless: PromptSetConfig
    system_prompt: str | None = None
    batch_size: PositiveInt = 8


class AnalysisConfig(_StrictModel):
    """Where to read activations from."""

    layers: list[int] | Literal["all"] = "all"
    positions: list[int] = Field(
        default=[-1],
        description="Token positions (relative to the end of the prompt) to record.",
    )

    @model_validator(mode="after")
    def _non_empty(self) -> AnalysisConfig:
        if self.layers != "all" and not self.layers:
            raise ValueError("analysis.layers must be 'all' or a non-empty list")
        if not self.positions:
            raise ValueError("analysis.positions must be non-empty")
        return self


class AblationConfig(_StrictModel):
    """How to remove the refusal direction."""

    method: Literal["orthogonalize", "hook"] = "orthogonalize"
    direction_layer: int | None = Field(
        default=None,
        description="Layer whose direction is ablated. None selects it automatically.",
    )
    direction_position: int = -1


class EvaluationConfig(_StrictModel):
    """Generation and metric settings for evaluation."""

    max_new_tokens: PositiveInt = 64
    refusal_markers: list[str] = Field(
        default=["I'm sorry", "I am sorry", "I cannot", "I can't", "As an AI"],
        description="Substrings that mark a completion as a refusal.",
    )
    capability_benchmark: str | None = Field(
        default=None, description="Optional benchmark id for capability retention."
    )


class ExportConfig(_StrictModel):
    """Output formats for the edited model."""

    formats: list[Literal["safetensors", "gguf"]] = ["safetensors"]
    gguf_outtype: GGUFOutType = "q8_0"
    llama_cpp_dir: Path | None = Field(
        default=None, description="Path to a llama.cpp checkout (needed for GGUF)."
    )


class RunConfig(_StrictModel):
    """Top-level configuration for one refusal-lens run."""

    name: str = Field(pattern=r"^[A-Za-z0-9._-]+$", description="Run name; used as a directory.")
    seed: int = 0
    deterministic: bool = Field(
        default=False, description="Request deterministic torch kernels (slower)."
    )
    output_dir: Path = Path("outputs")
    model: ModelConfig
    prompts: PromptsConfig
    analysis: AnalysisConfig = AnalysisConfig()
    ablation: AblationConfig = AblationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    export: ExportConfig = ExportConfig()

    @property
    def run_dir(self) -> Path:
        """Directory where all outputs of this run are written."""
        return self.output_dir / self.name


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a :class:`RunConfig` from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated config.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not a YAML mapping.
        pydantic.ValidationError: If the contents do not match the schema.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return RunConfig.model_validate(raw)


def save_resolved_config(config: RunConfig, run_dir: str | Path | None = None) -> Path:
    """Write the fully resolved config (defaults included) next to run outputs.

    The file can be passed straight back to :func:`load_config`.

    Args:
        config: The config to save.
        run_dir: Target directory. Defaults to :attr:`RunConfig.run_dir`.

    Returns:
        Path of the written file.
    """
    target_dir = Path(run_dir) if run_dir is not None else config.run_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / RESOLVED_CONFIG_FILENAME
    header = (
        f"# Resolved by refusal-lens {__version__} (torch {torch.__version__}) "
        f"at {datetime.now(UTC).isoformat(timespec='seconds')}\n"
    )
    body = yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
    out_path.write_text(header + body, encoding="utf-8")
    return out_path


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python and torch RNGs (CPU, CUDA and MPS).

    Args:
        seed: Seed value.
        deterministic: Also request deterministic torch algorithms. Ops without
            a deterministic implementation warn instead of raising.
    """
    random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS generators
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
