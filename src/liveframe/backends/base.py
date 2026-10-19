"""Runtime-neutral contracts shared by PyTorch, MLX, and test backends."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from liveframe.engine import GenerationRequest, NativeFrame


class BackendError(RuntimeError):
    """Base backend failure."""


class BackendUnavailableError(BackendError):
    """Raised when an optional runtime cannot be loaded on this host."""


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    name: str
    device: str
    supports_streaming: bool
    supports_serialized_noise: bool
    supports_intermediates: bool


@dataclass(frozen=True, slots=True)
class TensorDescriptor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    layout: str = "contiguous"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor name must not be empty")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("tensor dimensions must be positive")
        if not self.dtype:
            raise ValueError("tensor dtype must not be empty")


@runtime_checkable
class Backend(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities: ...

    def tensor_inventory(self) -> tuple[TensorDescriptor, ...]: ...

    def stream(self, request: GenerationRequest) -> Iterator[NativeFrame]: ...
