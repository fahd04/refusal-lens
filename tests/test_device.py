from __future__ import annotations

import warnings
from collections.abc import Callable

import pytest
import torch

from refusal_lens import device as device_mod
from refusal_lens.device import MPS_FALLBACK_ENV, resolve_device, resolve_dtype


@pytest.fixture
def backends(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Fake which accelerators are available."""

    def set_available(*, mps: bool, cuda: bool) -> None:
        monkeypatch.setattr(device_mod, "mps_available", lambda: mps)
        monkeypatch.setattr(device_mod, "cuda_available", lambda: cuda)

    monkeypatch.setenv(MPS_FALLBACK_ENV, "1")
    return set_available


@pytest.mark.parametrize(
    ("mps", "cuda", "expected"),
    [
        (True, False, "mps"),
        (False, True, "cuda"),
        (True, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_auto_prefers_mps_then_cuda_then_cpu(backends, mps, cuda, expected) -> None:
    backends(mps=mps, cuda=cuda)

    assert resolve_device("auto").type == expected


def test_cpu_is_always_available(backends) -> None:
    backends(mps=False, cuda=False)

    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("name", ["mps", "cuda"])
def test_explicit_unavailable_backend_raises(backends, name) -> None:
    backends(mps=False, cuda=False)

    with pytest.raises(RuntimeError, match=name.upper()):
        resolve_device(name)


def test_unknown_device_raises() -> None:
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")


def test_mps_without_fallback_warns(backends, monkeypatch: pytest.MonkeyPatch) -> None:
    backends(mps=True, cuda=False)
    monkeypatch.delenv(MPS_FALLBACK_ENV)

    with pytest.warns(RuntimeWarning, match=MPS_FALLBACK_ENV):
        resolve_device("auto")


def test_mps_with_fallback_does_not_warn(backends) -> None:
    backends(mps=True, cuda=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolve_device("auto")


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cpu", torch.float32), ("mps", torch.bfloat16), ("cuda", torch.bfloat16)],
)
def test_auto_dtype(device: str, expected: torch.dtype) -> None:
    assert resolve_dtype("auto", torch.device(device)) is expected


def test_explicit_dtype() -> None:
    assert resolve_dtype("float16", torch.device("cpu")) is torch.float16


def test_unknown_dtype_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dtype"):
        resolve_dtype("int4", torch.device("cpu"))
