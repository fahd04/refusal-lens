from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from refusal_lens import __version__
from refusal_lens.cli import app
from refusal_lens.config import RESOLVED_CONFIG_FILENAME, load_config

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_failed_stage_still_saves_resolved_config(
    tmp_path: Path, example_config_path: Path
) -> None:
    """Even a stage that stops early (here: nothing to export yet) records its config."""
    raw = yaml.safe_load(example_config_path.read_text(encoding="utf-8"))
    raw["output_dir"] = str(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = runner.invoke(app, ["export", "--config", str(config_path)])

    assert result.exit_code == 1
    resolved = tmp_path / raw["name"] / RESOLVED_CONFIG_FILENAME
    assert load_config(resolved) == load_config(config_path)
