"""End-to-end ``refusal-lens ablate`` and ``evaluate`` on a tiny random model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from transformers import AutoModelForCausalLM
from typer.testing import CliRunner

from refusal_lens.ablation import DirectionAblation
from refusal_lens.cli import app
from refusal_lens.directions import load_directions
from refusal_lens.models import get_adapter

runner = CliRunner()


def _invoke(command: str, config_path: Path, exit_code: int = 0) -> str:
    result = runner.invoke(app, [command, "--config", str(config_path)])
    assert result.exit_code == exit_code, result.output
    return result.output


def _set(config_path: Path, section: str, **values: Any) -> None:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw[section] = {**raw.get(section, {}), **values}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")


def _run_dir(config_path: Path) -> Path:
    return config_path.parent / "outputs" / "tiny"


def _read(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


# --- ablate ----------------------------------------------------------------------------


def test_ablate_needs_analyze_first(run_config_path: Path) -> None:
    output = _invoke("ablate", run_config_path, exit_code=1)

    assert "analyze" in output


def test_ablate_orthogonalize_saves_edited_model(run_config_path: Path) -> None:
    _invoke("analyze", run_config_path)
    _invoke("ablate", run_config_path)

    run_dir = _run_dir(run_config_path)
    summary = _read(run_dir / "ablation.json")
    assert summary["method"] == "orthogonalize"
    assert summary["position"] == -1
    assert str(summary["layer"]) in summary["candidates"]
    assert set(summary["candidates"][str(summary["layer"])]) == {"refusal_rate", "kl"}
    assert (run_dir / "plots" / "ablation_refusal.png").is_file()
    assert (run_dir / "plots" / "ablation_kl.png").is_file()
    assert (run_dir / "edited_model" / "config.json").is_file()


def test_saved_edited_model_equals_hook_ablation(
    run_config_path: Path, tiny_model_dir: Path
) -> None:
    """The model file on disk behaves exactly like the base model with hooks on."""
    _invoke("analyze", run_config_path)
    _invoke("ablate", run_config_path)
    run_dir = _run_dir(run_config_path)
    summary = _read(run_dir / "ablation.json")
    direction = load_directions(run_dir / "directions.safetensors").unit(
        summary["layer"], summary["position"]
    )
    input_ids = torch.tensor([[1, 4, 5, 6, 2, 1, 5]])

    base = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32)
    edited = AutoModelForCausalLM.from_pretrained(run_dir / "edited_model", dtype=torch.float32)
    with torch.inference_mode():
        with DirectionAblation(get_adapter(base), direction):
            hooked = base(input_ids=input_ids).logits
        saved = edited(input_ids=input_ids).logits

    torch.testing.assert_close(saved, hooked, atol=1e-4, rtol=1e-4)


def test_ablate_hook_saves_no_model(run_config_path: Path) -> None:
    _set(run_config_path, "ablation", method="hook")
    _invoke("analyze", run_config_path)
    _invoke("ablate", run_config_path)

    run_dir = _run_dir(run_config_path)
    assert _read(run_dir / "ablation.json")["method"] == "hook"
    assert not (run_dir / "edited_model").exists()


def test_ablate_fixed_layer_skips_the_search(run_config_path: Path) -> None:
    _set(run_config_path, "ablation", direction_layer=1)
    _invoke("analyze", run_config_path)
    _invoke("ablate", run_config_path)

    summary = _read(_run_dir(run_config_path) / "ablation.json")
    assert summary["layer"] == 1
    assert summary["candidates"] == {}


# --- evaluate --------------------------------------------------------------------------


def test_evaluate_needs_ablate_first(run_config_path: Path) -> None:
    _invoke("analyze", run_config_path)

    output = _invoke("evaluate", run_config_path, exit_code=1)

    assert "ablate" in output


@pytest.mark.parametrize("method", ["orthogonalize", "hook"])
def test_evaluate_reports_base_and_edited(run_config_path: Path, method: str) -> None:
    _set(run_config_path, "ablation", method=method)
    for command in ["analyze", "ablate", "evaluate"]:
        _invoke(command, run_config_path)

    report = _read(_run_dir(run_config_path) / "evaluation.json")
    assert report["method"] == method
    for model in ["base", "edited"]:
        assert 0 <= report[model]["harmful_refusal_rate"] <= 1
        assert 0 <= report[model]["harmless_refusal_rate"] <= 1
    assert report["base"]["harmless_kl"] is None
    assert report["edited"]["harmless_kl"] >= 0


def test_hook_and_orthogonalize_evaluate_the_same(run_config_path: Path) -> None:
    """Both methods pick the same layer and give the same KL: they are the same edit."""
    reports = {}
    for method in ["hook", "orthogonalize"]:
        _set(run_config_path, "ablation", method=method)
        for command in ["analyze", "ablate", "evaluate"]:
            _invoke(command, run_config_path)
        reports[method] = _read(_run_dir(run_config_path) / "evaluation.json")

    assert reports["hook"]["layer"] == reports["orthogonalize"]["layer"]
    assert reports["hook"]["edited"]["harmless_kl"] == pytest.approx(
        reports["orthogonalize"]["edited"]["harmless_kl"], abs=1e-4
    )
