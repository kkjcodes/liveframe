"""Lazy backend registry: importing LiveFrame never imports MLX or PyTorch."""

from __future__ import annotations

import importlib.util
import platform
from collections.abc import Callable

from liveframe.backends.base import Backend, BackendUnavailableError

BackendFactory = Callable[[], Backend]
_FACTORIES: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("backend name must not be empty")
    if normalized in _FACTORIES:
        raise ValueError(f"backend already registered: {normalized}")
    _FACTORIES[normalized] = factory


def available_backends() -> dict[str, bool]:
    return {
        "synthetic": True,
        "mlx": platform.system() == "Darwin" and importlib.util.find_spec("mlx") is not None,
        "pytorch": importlib.util.find_spec("torch") is not None,
        **{name: True for name in _FACTORIES},
    }


def create_backend(name: str) -> Backend:
    normalized = name.strip().lower()
    if normalized in _FACTORIES:
        return _FACTORIES[normalized]()
    if normalized == "synthetic":
        from liveframe.backends.synthetic import SyntheticBackend

        return SyntheticBackend()
    if normalized == "mlx":
        if not available_backends()["mlx"]:
            raise BackendUnavailableError("MLX backend requires macOS, Apple Silicon, and mlx")
        from liveframe.backends.mlx import MLXBackend

        return MLXBackend()
    if normalized == "pytorch":
        if not available_backends()["pytorch"]:
            raise BackendUnavailableError("PyTorch backend requires the cuda optional dependency")
        from liveframe.backends.pytorch import PyTorchBackend

        return PyTorchBackend()
    raise BackendUnavailableError(f"unknown backend: {name}")
