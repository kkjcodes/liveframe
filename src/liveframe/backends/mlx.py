"""MLX backend shell; model conversion and execution land behind this contract."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from typing import Any

from liveframe.backends.base import (
    BackendCapabilities,
    BackendError,
    GenerationRequest,
    TensorDescriptor,
)


class MLXBackend:
    def __init__(self) -> None:
        try:
            mx = importlib.import_module("mlx.core")
        except ImportError as exc:
            raise BackendError("MLX could not be imported; install liveframe[mlx]") from exc
        self._mx_version = getattr(mx, "__version__", "unknown")

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="mlx",
            device="apple-gpu",
            supports_streaming=False,
            supports_serialized_noise=False,
            supports_intermediates=False,
        )

    def tensor_inventory(self) -> tuple[TensorDescriptor, ...]:
        raise BackendError("CF++ MLX weights have not been converted yet")

    def stream(self, request: GenerationRequest) -> Iterator[Any]:
        del request
        raise BackendError("CF++ MLX streaming is not implemented yet")
        yield
