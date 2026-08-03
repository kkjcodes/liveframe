"""Runtime-neutral contracts shared by PyTorch, MLX, and test backends."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


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


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: str
    latent_frames: int = 21
    width: int = 832
    height: int = 480
    native_pixel_fps: int = 16
    student_steps: int = 2
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.latent_frames < 1:
            raise ValueError("latent_frames must be positive")
        if self.width < 1 or self.height < 1:
            raise ValueError("resolution must be positive")
        if self.native_pixel_fps != 16:
            raise ValueError("released gates.v1 checkpoint semantics require native 16 FPS")
        if self.student_steps not in {1, 2}:
            raise ValueError("frame-wise CF++ supports a 1-step or 2-step student mode")

    @property
    def pixel_frames(self) -> int:
        return 4 * (self.latent_frames - 1) + 1


@runtime_checkable
class Backend(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities: ...

    def tensor_inventory(self) -> tuple[TensorDescriptor, ...]: ...

    def stream(self, request: GenerationRequest) -> Iterator[Any]: ...
