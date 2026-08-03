"""Backend capability contracts and lazy backend discovery."""

from liveframe.backends.base import (
    Backend,
    BackendCapabilities,
    BackendError,
    BackendUnavailableError,
    GenerationRequest,
    TensorDescriptor,
)
from liveframe.backends.registry import available_backends, create_backend

__all__ = [
    "Backend",
    "BackendCapabilities",
    "BackendError",
    "BackendUnavailableError",
    "GenerationRequest",
    "TensorDescriptor",
    "available_backends",
    "create_backend",
]
