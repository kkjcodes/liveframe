"""Fail-closed contracts for output-only video decoder experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class DecoderContractError(RuntimeError):
    """Raised when a decoder emission violates the frozen output contract."""


@dataclass(frozen=True, slots=True)
class DecoderRequest:
    """One immutable latent-to-RGB decode request."""

    latent_frames: int
    native_frames: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.latent_frames < 1:
            raise ValueError("latent_frames must be positive")
        if self.native_frames < 1:
            raise ValueError("native_frames must be positive")
        if self.width < 1 or self.height < 1:
            raise ValueError("decoder resolution must be positive")


@dataclass(frozen=True, slots=True)
class DecoderChunk:
    """One contiguous uint8 THWC decoder emission."""

    start_index: int
    payload: Any
    shape: tuple[int, int, int, int]
    dtype: str
    finite: bool
    minimum: float
    maximum: float

    @property
    def native_frame_count(self) -> int:
        return self.shape[0]


@runtime_checkable
class VideoDecoder(Protocol):
    """Narrow adapter boundary shared by full and tiny output decoders."""

    def decode(self, latents: Any, request: DecoderRequest) -> Iterable[DecoderChunk]: ...


def validate_decoder_chunks(
    chunks: Iterable[DecoderChunk], request: DecoderRequest
) -> Iterator[DecoderChunk]:
    """Yield validated chunks while enforcing exact contiguous native-frame output."""

    emitted = 0
    for chunk in chunks:
        if not isinstance(chunk, DecoderChunk):
            raise DecoderContractError("decoder must emit DecoderChunk instances")
        if chunk.start_index != emitted:
            raise DecoderContractError(
                f"decoder chunk index mismatch: expected {emitted}, observed {chunk.start_index}"
            )
        if len(chunk.shape) != 4 or chunk.shape[0] < 1:
            raise DecoderContractError("decoder chunk shape must be nonempty THWC")
        if chunk.shape[1:] != (request.height, request.width, 3):
            raise DecoderContractError(
                "decoder chunk dimensions do not match the requested RGB resolution"
            )
        if chunk.dtype != "uint8":
            raise DecoderContractError("decoder chunks must declare uint8 output")
        if not chunk.finite:
            raise DecoderContractError("decoder chunk contains nonfinite values")
        if not math.isfinite(chunk.minimum) or not math.isfinite(chunk.maximum):
            raise DecoderContractError("decoder chunk range must be finite")
        if chunk.minimum < 0 or chunk.maximum > 255 or chunk.minimum > chunk.maximum:
            raise DecoderContractError("decoder chunk values must remain in uint8 range")
        emitted += chunk.native_frame_count
        if emitted > request.native_frames:
            raise DecoderContractError(
                f"decoder emitted {emitted} native frames; expected {request.native_frames}"
            )
        yield chunk
    if emitted != request.native_frames:
        raise DecoderContractError(
            f"decoder emitted {emitted} native frames; expected {request.native_frames}"
        )