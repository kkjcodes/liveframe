from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from liveframe.backends import (
    BackendCapabilities,
    BackendUnavailableError,
    GenerationRequest,
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
        yield request.pixel_frames


def test_generation_request_preserves_native_frame_semantics() -> None:
    request = GenerationRequest("a painted animation cel")

    assert request.pixel_frames == 81
    with pytest.raises(ValueError, match="native 16 FPS"):
        GenerationRequest("prompt", native_pixel_fps=12)


def test_tensor_descriptor_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        TensorDescriptor("bad", (1, 0), "float16")


def test_registry_supports_test_backend_without_optional_runtimes() -> None:
    register_backend("fake-test", FakeBackend)

    backend = create_backend("fake-test")

    assert backend.capabilities.supports_serialized_noise
    assert tuple(backend.stream(GenerationRequest("prompt"))) == (81,)


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
