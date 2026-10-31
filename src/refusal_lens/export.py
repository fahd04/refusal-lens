"""Export edited models as safetensors (Hugging Face format) or GGUF.

GGUF conversion runs llama.cpp's ``convert_hf_to_gguf.py`` with this
environment's Python, so a local llama.cpp checkout is required (no build
needed: the converter writes f32/f16/bf16/q8_0 directly). Edited models remain
subject to the original model's license.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from refusal_lens.config import GGUFOutType

CONVERTER_SCRIPT = "convert_hf_to_gguf.py"


def save_safetensors(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    out_dir: str | Path,
    max_shard_size: str = "2GB",
) -> Path:
    """Save model weights (safetensors) and tokenizer in Hugging Face layout.

    Args:
        model: Model to save.
        tokenizer: Its tokenizer.
        out_dir: Destination directory.
        max_shard_size: Shard size passed to ``save_pretrained``.

    Returns:
        The output directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, max_shard_size=max_shard_size)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def find_llama_cpp_converter(llama_cpp_dir: str | Path) -> Path:
    """Locate ``convert_hf_to_gguf.py`` inside a llama.cpp checkout.

    Raises:
        FileNotFoundError: If the script cannot be found.
    """
    script = Path(llama_cpp_dir) / CONVERTER_SCRIPT
    if not script.is_file():
        raise FileNotFoundError(
            f"{script} not found; point export.llama_cpp_dir at a llama.cpp checkout "
            "(git clone https://github.com/ggml-org/llama.cpp)"
        )
    return script


def convert_to_gguf(
    hf_dir: str | Path,
    out_path: str | Path,
    llama_cpp_dir: str | Path,
    outtype: GGUFOutType = "q8_0",
) -> Path:
    """Convert a Hugging Face model directory to a GGUF file.

    Args:
        hf_dir: Directory produced by :func:`save_safetensors`.
        out_path: Destination ``.gguf`` file.
        llama_cpp_dir: Path to a llama.cpp checkout.
        outtype: GGUF tensor type.

    Returns:
        Path to the written GGUF file.

    Raises:
        FileNotFoundError: If ``hf_dir`` is not a model directory or the converter is missing.
        subprocess.CalledProcessError: If conversion fails.
    """
    hf_dir, out_path = Path(hf_dir), Path(out_path)
    if not (hf_dir / "config.json").is_file():
        raise FileNotFoundError(f"{hf_dir / 'config.json'} not found; not a Hugging Face model")
    converter = find_llama_cpp_converter(llama_cpp_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # This environment's Python already has torch, transformers and numpy, which the
    # converter needs; it brings its own `gguf` package from the checkout.
    command = [
        sys.executable,
        str(converter),
        str(hf_dir),
        "--outfile",
        str(out_path),
        "--outtype",
        outtype,
    ]
    subprocess.run(command, check=True)  # fixed argument list, no shell
    return out_path


def write_ollama_modelfile(gguf_path: str | Path) -> Path:
    """Write an Ollama ``Modelfile`` next to ``gguf_path``.

    The chat template is read from the GGUF metadata, so the file only needs
    to point at the weights. Use it with ``ollama create <name> -f Modelfile``.

    Returns:
        Path to the ``Modelfile``.
    """
    gguf_path = Path(gguf_path)
    modelfile = gguf_path.parent / "Modelfile"
    modelfile.write_text(f"FROM ./{gguf_path.name}\n", encoding="utf-8")
    return modelfile
