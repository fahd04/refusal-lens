from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from pydantic import ValidationError

from refusal_lens.config import (
    RESOLVED_CONFIG_FILENAME,
    RunConfig,
    load_config,
    save_resolved_config,
    seed_everything,
)


def _write_yaml(path: Path, data: object) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _minimal() -> dict[str, object]:
    prompt_set = {"dataset": "local.jsonl", "n_train": 4, "n_eval": 2}
    return {
        "name": "tiny",
        "model": {"id": "tiny/model"},
        "prompts": {"harmful": prompt_set, "harmless": prompt_set},
    }


def test_example_config_loads(example_config_path: Path) -> None:
    config = load_config(example_config_path)

    assert config.model.id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.seed == 42
    assert config.prompts.harmful.n_train == 128
    assert config.analysis.layers == "all"
    assert config.run_dir == Path("outputs") / "qwen2.5-1.5b-instruct"


def test_defaults_are_filled(tmp_path: Path) -> None:
    config = load_config(_write_yaml(tmp_path / "c.yaml", _minimal()))

    assert config.seed == 0
    assert config.model.device == "auto"
    assert config.ablation.method == "orthogonalize"
    assert config.export.formats == ["safetensors"]


def test_resolved_config_round_trips(tmp_path: Path, example_config_path: Path) -> None:
    config = load_config(example_config_path)

    out = save_resolved_config(config, tmp_path / "run")

    assert out == tmp_path / "run" / RESOLVED_CONFIG_FILENAME
    assert load_config(out) == config


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    data = _minimal()
    data["sede"] = 1  # typo of "seed"

    with pytest.raises(ValidationError, match="sede"):
        load_config(_write_yaml(tmp_path / "c.yaml", data))


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "device", "tpu"),
        ("model", "dtype", "int4"),
        ("analysis", "layers", []),
    ],
)
def test_invalid_values_are_rejected(tmp_path: Path, section: str, key: str, value: object) -> None:
    data = _minimal()
    data[section] = {**data.get(section, {}), key: value}

    with pytest.raises(ValidationError):
        load_config(_write_yaml(tmp_path / "c.yaml", data))


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mapping"):
        load_config(_write_yaml(tmp_path / "c.yaml", ["not", "a", "mapping"]))


def test_seed_everything_is_reproducible() -> None:
    seed_everything(123)
    a = torch.randn(4)
    seed_everything(123)
    b = torch.randn(4)

    assert torch.equal(a, b)


def test_run_dir_uses_name() -> None:
    config = RunConfig.model_validate({**_minimal(), "output_dir": "/tmp/out"})

    assert config.run_dir == Path("/tmp/out/tiny")
