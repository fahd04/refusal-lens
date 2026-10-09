"""Device and dtype selection.

Preference order for ``"auto"`` is MPS (Apple Silicon), then CUDA, then CPU.

On MPS, some operators are not yet implemented. Set
``PYTORCH_ENABLE_MPS_FALLBACK=1`` **before** Python starts to let PyTorch run
those ops on the CPU instead of raising::

    export PYTORCH_ENABLE_MPS_FALLBACK=1
"""

from __future__ import annotations

import os
import warnings
from typing import Literal

import torch

DeviceName = Literal["auto", "mps", "cuda", "cpu"]
DTypeName = Literal["auto", "float32", "float16", "bfloat16"]

MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def mps_available() -> bool:
    """Return whether the MPS backend is built and usable on this machine."""
    return torch.backends.mps.is_available()


def cuda_available() -> bool:
    """Return whether a CUDA device is usable."""
    return torch.cuda.is_available()


def mps_fallback_enabled() -> bool:
    """Return whether ``PYTORCH_ENABLE_MPS_FALLBACK=1`` is set."""
    return os.environ.get(MPS_FALLBACK_ENV) == "1"


def resolve_device(preference: DeviceName = "auto") -> torch.device:
    """Pick the torch device to run on.

    Args:
        preference: ``"auto"`` picks MPS, then CUDA, then CPU. An explicit
            name requests that backend.

    Returns:
        The selected device.

    Raises:
        RuntimeError: If an explicitly requested backend is unavailable.
        ValueError: If ``preference`` is not a known device name.
    """
    if preference == "auto":
        if mps_available():
            device = torch.device("mps")
        elif cuda_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif preference == "mps":
        if not mps_available():
            raise RuntimeError("MPS requested but not available on this machine")
        device = torch.device("mps")
    elif preference == "cuda":
        if not cuda_available():
            raise RuntimeError("CUDA requested but not available on this machine")
        device = torch.device("cuda")
    elif preference == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device {preference!r}")

    if device.type == "mps" and not mps_fallback_enabled():
        warnings.warn(
            f"Using MPS without {MPS_FALLBACK_ENV}=1; unsupported ops will raise. "
            f"Export {MPS_FALLBACK_ENV}=1 before starting Python to fall back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
    return device


def resolve_dtype(name: DTypeName, device: torch.device) -> torch.dtype:
    """Pick the floating-point dtype for model weights.

    Args:
        name: ``"auto"`` uses bfloat16 on CUDA/MPS and float32 on CPU. Any
            other value maps directly to the torch dtype.
        device: Device the model will run on (used for ``"auto"``).

    Returns:
        The selected dtype.

    Raises:
        ValueError: If ``name`` is not a known dtype name.
    """
    if name == "auto":
        return torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(f"Unknown dtype {name!r}") from None
