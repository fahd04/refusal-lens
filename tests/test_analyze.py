"""End-to-end ``refusal-lens analyze`` on a tiny random model and local prompt files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from refusal_lens.cli import app
from refusal_lens.directions import load_directions

runner = CliRunner()

N_LAYERS = 2  # tiny_model


def _prompts(path: Path, verbs: list[str], nouns: list[str]) -> Path:
    rows = [{"text": f"how do i {v} a {n}"} for v in verbs for n in nouns]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def config_path(tmp_path: Path, tiny_model_dir: Path) -> Path:
    harmful = _prompts(tmp_path / "harmful.jsonl", ["steal", "hack"], ["car", "account", "bomb"])
    harmless = _prompts(tmp_path / "harmless.jsonl", ["bake", "write"], ["cake", "bread", "poem"])
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
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(config_path: Path) -> Path:
    result = runner.invoke(app, ["analyze", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    return config_path.parent / "outputs" / "tiny"


def test_analyze_writes_directions(config_path: Path) -> None:
    run_dir = _run(config_path)

    directions = load_directions(run_dir / "directions.safetensors")
    assert directions.layers == list(range(N_LAYERS))
    assert directions.positions == [-1]
    assert directions.directions.shape[:2] == (N_LAYERS, 1)


def test_analyze_writes_plots(config_path: Path) -> None:
    run_dir = _run(config_path)

    for name in ["direction_norms", "direction_similarity", "projections", "separation"]:
        plot = run_dir / "plots" / f"{name}.png"
        assert plot.is_file(), name
        assert plot.stat().st_size > 0


def test_analyze_writes_summary(config_path: Path) -> None:
    run_dir = _run(config_path)

    summary = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
    assert summary["model"].endswith("tiny-qwen2") or summary["model"].endswith("tiny-llama")
    assert summary["best_layer"] in range(N_LAYERS)
    assert summary["best_position"] == -1
    assert summary["n_prompts"] == {
        "harmful_train": 4,
        "harmful_eval": 2,
        "harmless_train": 4,
        "harmless_eval": 2,
    }
    assert len(summary["separation"]) == N_LAYERS


def test_analyze_respects_layer_subset(config_path: Path) -> None:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["analysis"] = {"layers": [1]}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    run_dir = _run(config_path)

    assert load_directions(run_dir / "directions.safetensors").layers == [1]
    summary = json.loads((run_dir / "analysis.json").read_text(encoding="utf-8"))
    assert summary["best_layer"] == 1


def test_analyze_is_reproducible(config_path: Path) -> None:
    first = load_directions(_run(config_path) / "directions.safetensors").directions
    second = load_directions(_run(config_path) / "directions.safetensors").directions

    assert first.equal(second)
