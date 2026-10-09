"""Export edited models as safetensors (Hugging Face format) or GGUF.

GGUF conversion shells out to llama.cpp's ``convert_hf_to_gguf.py`` (and
``llama-quantize`` for quantized types), so a local llama.cpp checkout is
required. Edited models remain subject to the original model's license.
"""

from __future__ import annotations

from pathlib import Path

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from refusal_lens.config import GGUFOutType


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
    raise NotImplementedError


def find_llama_cpp_converter(llama_cpp_dir: str | Path) -> Path:
    """Locate ``convert_hf_to_gguf.py`` inside a llama.cpp checkout.

    Raises:
        FileNotFoundError: If the script cannot be found.
    """
    raise NotImplementedError


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
        subprocess.CalledProcessError: If conversion fails.
    """
    raise NotImplementedError
