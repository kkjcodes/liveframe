from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from liveframe.backends import (
    BackendCapabilities,
    BackendUnavailableError,
    GenerationRequest,
    NativeFrame,
    TensorDescriptor,
    create_backend,
)
from liveframe.backends.pytorch import _normalize_tensor_name, _state_dict
from liveframe.backends.registry import register_backend


class FakeBackend:
    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities("fake", "cpu", True, True, True)

    def tensor_inventory(self) -> tuple[TensorDescriptor, ...]:
        return (TensorDescriptor("weight", (2, 2), "float32"),)

    def stream(self, request: GenerationRequest) -> Iterator[Any]:
        yield NativeFrame(
            index=0,
            payload=request.native_frames,
            native_frame_count=request.native_frames,
        )


def test_generation_request_preserves_native_frame_semantics() -> None:
    request = GenerationRequest("a painted animation cel", seed=42)
    positional_metadata = GenerationRequest("prompt", 81, 16.0, 832, 480, {"source": "test"})

    assert request.native_frames == 81
    assert request.playback_seconds == pytest.approx(81 / 16)
    assert request.seed == 42
    assert positional_metadata.metadata == {"source": "test"}
    with pytest.raises(ValueError, match="finite and positive"):
        GenerationRequest("prompt", native_fps=0)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        GenerationRequest("prompt", seed=-1)


def test_tensor_descriptor_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        TensorDescriptor("bad", (1, 0), "float16")


def test_registry_supports_test_backend_without_optional_runtimes() -> None:
    register_backend("fake-test", FakeBackend)

    backend = create_backend("fake-test")

    assert backend.capabilities.supports_serialized_noise
    frames = tuple(backend.stream(GenerationRequest("prompt")))
    assert frames[0].native_frame_count == 81


def test_synthetic_backend_is_deterministic_and_exact() -> None:
    backend = create_backend("synthetic")
    request = GenerationRequest("prompt", native_frames=9, seed=7)

    first = tuple(backend.stream(request))
    second = tuple(backend.stream(request))

    assert backend.capabilities.device == "cpu"
    assert first == second
    assert sum(frame.native_frame_count for frame in first) == 9
    assert all(frame.metadata and frame.metadata["materialized"] for frame in first)


def test_unknown_backend_fails_clearly() -> None:
    with pytest.raises(BackendUnavailableError, match="unknown backend"):
        create_backend("unknown-runtime")


def test_cfpp_checkpoint_envelope_and_fsdp_prefix_are_normalized() -> None:
    state = {"model._fsdp_wrapped_module.patch_embedding.bias": object()}

    assert _state_dict({"generator_ema": state}) is state
    assert (
        _normalize_tensor_name("model._fsdp_wrapped_module.patch_embedding.bias")
        == "patch_embedding.bias"
    )
