"""Backend capability contracts and lazy backend discovery."""

from liveframe.backends.base import (
    Backend,
    BackendCapabilities,
    BackendError,
    BackendUnavailableError,
    TensorDescriptor,
)
from liveframe.backends.registry import available_backends, create_backend
from liveframe.engine import GenerationRequest, NativeFrame

__all__ = [
    "Backend",
    "BackendCapabilities",
    "BackendError",
    "BackendUnavailableError",
    "GenerationRequest",
    "NativeFrame",
    "TensorDescriptor",
    "available_backends",
    "create_backend",
]
