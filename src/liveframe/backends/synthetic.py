"""Deterministic CPU-only backend for contract and campaign validation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

from liveframe.backends.base import BackendCapabilities, TensorDescriptor
from liveframe.engine import GenerationRequest, NativeFrame


class SyntheticBackend:
    def __init__(self, *, chunk_size: int = 4) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self._chunk_size = chunk_size

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="synthetic",
            device="cpu",
            supports_streaming=True,
            supports_serialized_noise=True,
            supports_intermediates=False,
        )

    def tensor_inventory(self) -> tuple[TensorDescriptor, ...]:
        return ()

    def stream(self, request: GenerationRequest) -> Iterator[NativeFrame]:
        for index in range(0, request.native_frames, self._chunk_size):
            count = min(self._chunk_size, request.native_frames - index)
            material = (
                f"{request.prompt}\0{request.seed}\0{request.width}x{request.height}"
                f"\0{index}\0{count}"
            ).encode()
            yield NativeFrame(
                index=index,
                payload=hashlib.sha256(material).digest(),
                native_frame_count=count,
                metadata={"materialized": True, "payload_format": "sha256-bytes"},
            )