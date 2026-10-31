from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerFast
from typer.testing import CliRunner

from refusal_lens.cli import app
from refusal_lens.export import (
    convert_to_gguf,
    find_llama_cpp_converter,
    save_safetensors,
    write_ollama_modelfile,
)

runner = CliRunner()

# Stands in for llama.cpp's converter: records how it was called and writes a dummy file.
_FAKE_CONVERTER = """
import json, sys
args = sys.argv[1:]
if "--fail" in open(args[0] + "/config.json").read():
    sys.exit(3)
out = args[args.index("--outfile") + 1]
with open(out, "w") as f:
    json.dump({"python": sys.executable, "args": args}, f)
"""


@pytest.fixture
def llama_cpp_dir(tmp_path: Path) -> Path:
    path = tmp_path / "llama.cpp"
    path.mkdir()
    (path / "convert_hf_to_gguf.py").write_text(_FAKE_CONVERTER, encoding="utf-8")
    return path


@pytest.fixture
def hf_dir(tmp_path: Path) -> Path:
    path = tmp_path / "hf"
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


# --- safetensors -----------------------------------------------------------------------


def test_save_safetensors_round_trip(
    tmp_path: Path, tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerFast
) -> None:
    out = save_safetensors(tiny_model, tiny_tokenizer, tmp_path / "nested" / "model")

    assert out == tmp_path / "nested" / "model"
    assert (out / "config.json").is_file()
    assert list(out.glob("*.safetensors"))
    assert (out / "tokenizer_config.json").is_file()
    reloaded = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(input_ids=ids).logits, tiny_model(input_ids=ids).logits)


# --- llama.cpp converter ------------------------------------------------------------------


def test_find_converter(llama_cpp_dir: Path) -> None:
    assert find_llama_cpp_converter(llama_cpp_dir) == llama_cpp_dir / "convert_hf_to_gguf.py"


def test_find_converter_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"convert_hf_to_gguf\.py"):
        find_llama_cpp_converter(tmp_path)


@pytest.mark.parametrize("outtype", ["q8_0", "bf16"])
def test_convert_to_gguf_calls_converter(
    tmp_path: Path, hf_dir: Path, llama_cpp_dir: Path, outtype: Any
) -> None:
    out_path = tmp_path / "out" / "model.gguf"

    result = convert_to_gguf(hf_dir, out_path, llama_cpp_dir, outtype=outtype)

    assert result == out_path
    call = json.loads(out_path.read_text(encoding="utf-8"))
    assert call["python"] == sys.executable  # runs in this environment (torch etc. available)
    assert call["args"][0] == str(hf_dir)
    assert call["args"][call["args"].index("--outtype") + 1] == outtype


def test_convert_to_gguf_failure_raises(tmp_path: Path, hf_dir: Path, llama_cpp_dir: Path) -> None:
    (hf_dir / "config.json").write_text('{"--fail": 1}', encoding="utf-8")

    with pytest.raises(subprocess.CalledProcessError):
        convert_to_gguf(hf_dir, tmp_path / "model.gguf", llama_cpp_dir)


def test_convert_to_gguf_needs_a_model_dir(tmp_path: Path, llama_cpp_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"config\.json"):
        convert_to_gguf(tmp_path / "missing", tmp_path / "model.gguf", llama_cpp_dir)


def test_ollama_modelfile(tmp_path: Path) -> None:
    gguf = tmp_path / "tiny-q8_0.gguf"

    modelfile = write_ollama_modelfile(gguf)

    assert modelfile == tmp_path / "Modelfile"
    assert modelfile.read_text(encoding="utf-8").startswith("FROM ./tiny-q8_0.gguf\n")


# --- refusal-lens export --------------------------------------------------------------


def _invoke(command: str, config_path: Path, exit_code: int = 0) -> str:
    result = runner.invoke(app, [command, "--config", str(config_path)])
    assert result.exit_code == exit_code, result.output
    return result.output


def _set(config_path: Path, section: str, **values: Any) -> None:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw[section] = {**raw.get(section, {}), **values}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")


def _ablated(config_path: Path) -> Path:
    _invoke("analyze", config_path)
    _invoke("ablate", config_path)
    return config_path.parent / "outputs" / "tiny"


def test_export_needs_ablate_first(run_config_path: Path) -> None:
    assert "ablate" in _invoke("export", run_config_path, exit_code=1)


def test_export_hook_method_has_nothing_to_export(run_config_path: Path) -> None:
    _set(run_config_path, "ablation", method="hook")
    _ablated(run_config_path)

    assert "orthogonalize" in _invoke("export", run_config_path, exit_code=1)


def test_export_safetensors_points_to_edited_model(run_config_path: Path) -> None:
    run_dir = _ablated(run_config_path)

    output = _invoke("export", run_config_path)

    assert "edited_model" in output.replace("\n", "")
    assert not (run_dir / "export").exists()  # nothing copied: it's already safetensors


def test_export_gguf(run_config_path: Path, llama_cpp_dir: Path) -> None:
    _set(
        run_config_path,
        "export",
        formats=["gguf"],
        gguf_outtype="f16",
        llama_cpp_dir=str(llama_cpp_dir),
    )
    run_dir = _ablated(run_config_path)

    _invoke("export", run_config_path)

    gguf = run_dir / "export" / "tiny-f16.gguf"
    call = json.loads(gguf.read_text(encoding="utf-8"))
    assert call["args"][0] == str(run_dir / "edited_model")
    assert (run_dir / "export" / "Modelfile").is_file()


def test_export_gguf_needs_llama_cpp_dir(run_config_path: Path) -> None:
    _set(run_config_path, "export", formats=["gguf"])
    _ablated(run_config_path)

    assert "llama_cpp_dir" in _invoke("export", run_config_path, exit_code=1)
