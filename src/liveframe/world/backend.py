"""WorldBackend protocol, capabilities, and fail-closed contract errors."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from liveframe.engine import NativeFrame


class WorldBackendError(Exception):
    """The backend failed (crash, timeout, upstream error): case status ``failed``."""


class WorldContractError(Exception):
    """The backend violated the measurement contract: case status ``invalid``."""


class FrameAccountingError(WorldContractError):
    """Missing, duplicate, out-of-order, or excess native frames."""


class FramePayloadError(WorldContractError):
    """Frame payload shape, dtype, or value range differs from declared capabilities."""


class ActionNotSupportedError(WorldContractError):
    """An action token outside the backend's declared capabilities was requested."""


class NonMonotonicTimestampError(WorldContractError):
    """A worker-side timestamp went backwards."""


class MissingChunkError(WorldContractError):
    """No frames arrived for a chunk after its action was submitted."""


@dataclass(frozen=True, slots=True)
class WorldCapabilities:
    name: str
    interaction_mode: Literal["interactive", "batch"]
    native_fps: float
    chunk_frames: int
    resolution: tuple[int, int]
    actions: frozenset[str]
    action_magnitudes: Mapping[str, float] = field(default_factory=dict)
    supports_seed: bool = True
    min_vram_gb: float | None = None
    license_notice: str | None = None
    # Causal VAEs commonly emit a shorter first chunk (e.g. 1 + 4*(k-1) pixel
    # frames for the first k-latent block, 4*k thereafter). None = uniform.
    chunk_frames_first: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("backend name must not be empty")
        if self.native_fps <= 0:
            raise ValueError("native_fps must be positive")
        if self.chunk_frames < 1:
            raise ValueError("chunk_frames must be positive")
        if self.chunk_frames_first is not None and self.chunk_frames_first < 1:
            raise ValueError("chunk_frames_first must be positive when set")
        height, width = self.resolution
        if height < 1 or width < 1:
            raise ValueError("resolution must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "interaction_mode": self.interaction_mode,
            "native_fps": self.native_fps,
            "chunk_frames": self.chunk_frames,
            "resolution": list(self.resolution),
            "actions": sorted(self.actions),
            "action_magnitudes": dict(self.action_magnitudes),
            "supports_seed": self.supports_seed,
            "min_vram_gb": self.min_vram_gb,
            "license_notice": self.license_notice,
            "chunk_frames_first": self.chunk_frames_first,
        }


@runtime_checkable
class WorldBackend(Protocol):
    """Contract adapters implement; see the world-handoff spec section 6."""

    @property
    def capabilities(self) -> WorldCapabilities: ...

    def load(self) -> None: ...

    def reset(self, image: Any, prompt: str | None, seed: int) -> None: ...

    def step(self, actions: Sequence[str]) -> Iterator[NativeFrame]: ...

    def run_script(self, script: Any) -> Iterator[NativeFrame]: ...

    def close(self) -> None: ...
