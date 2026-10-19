"""PyTorch reference backend shell with safe checkpoint inventory support."""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from liveframe.backends.base import (
    BackendCapabilities,
    BackendError,
    TensorDescriptor,
)
from liveframe.engine import GenerationRequest, NativeFrame


class PyTorchBackend:
    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="pytorch",
            device="cuda",
            supports_streaming=False,
            supports_serialized_noise=False,
            supports_intermediates=False,
        )

    def tensor_inventory(self) -> tuple[TensorDescriptor, ...]:
        raise BackendError("load a pinned checkpoint before requesting its inventory")

    def inventory_checkpoint(self, path: Path) -> tuple[TensorDescriptor, ...]:
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise BackendError("PyTorch is not installed") from exc
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise BackendError(f"cannot safely inventory checkpoint {path}: {exc}") from exc
        state = _state_dict(checkpoint)
        descriptors = [
            TensorDescriptor(
                name=_normalize_tensor_name(name),
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
            )
            for name, tensor in state.items()
            if isinstance(name, str) and isinstance(tensor, torch.Tensor)
        ]
        if not descriptors:
            raise BackendError("checkpoint contains no tensor state dictionary")
        return tuple(sorted(descriptors, key=lambda item: item.name))

    def stream(self, request: GenerationRequest) -> Iterator[NativeFrame]:
        del request
        raise BackendError("CF++ PyTorch streaming integration is not implemented yet")
        yield


def _state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("generator_ema", "state_dict", "model", "module"):
            nested = checkpoint.get(key)
            if isinstance(nested, Mapping):
                return _state_dict(nested)
        return checkpoint
    raise BackendError("checkpoint root must be a mapping")


def _normalize_tensor_name(name: str) -> str:
    prefixes = ("model._fsdp_wrapped_module.", "_fsdp_wrapped_module.", "module.")
    for prefix in prefixes:
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name
