"""Versioned prompt state with bounded queues and chunk-boundary swaps."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace


class PromptQueueError(RuntimeError):
    """Raised for invalid asynchronous prompt-pipeline transitions."""


@dataclass(frozen=True, slots=True)
class PromptRequest:
    version: int
    prompt: str
    requested_at: float
    encoding_started_at: float | None = None
    encoding_finished_at: float | None = None
    activated_at: float | None = None
    superseded: bool = False


class PromptCoordinator:
    """State machine for encoding away from generation and swapping only at boundaries."""

    def __init__(self, request_capacity: int = 4, embedding_capacity: int = 2) -> None:
        if request_capacity < 1 or embedding_capacity < 1:
            raise ValueError("prompt queue capacities must be positive")
        self.request_capacity = request_capacity
        self.embedding_capacity = embedding_capacity
        self._next_version = 1
        self._requests: deque[PromptRequest] = deque()
        self._completed: deque[PromptRequest] = deque()
        self._active: PromptRequest | None = None
        self._superseded: list[PromptRequest] = []

    @property
    def active(self) -> PromptRequest | None:
        return self._active

    @property
    def superseded(self) -> tuple[PromptRequest, ...]:
        return tuple(self._superseded)

    def submit(self, prompt: str, requested_at: float) -> PromptRequest:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        self._validate_time(requested_at)
        if len(self._requests) >= self.request_capacity:
            raise PromptQueueError("prompt request queue overflow")
        request = PromptRequest(self._next_version, prompt, requested_at)
        self._next_version += 1
        self._requests.append(request)
        return request

    def start_encoding(self, started_at: float) -> PromptRequest:
        self._validate_time(started_at)
        if not self._requests:
            raise PromptQueueError("no prompt is waiting for encoding")
        request = self._requests.popleft()
        return replace(request, encoding_started_at=started_at)

    def finish_encoding(self, request: PromptRequest, finished_at: float) -> PromptRequest:
        self._validate_time(finished_at)
        if request.encoding_started_at is None:
            raise PromptQueueError("encoding was not started")
        if finished_at < request.encoding_started_at:
            raise PromptQueueError("encoding finish precedes its start")
        if len(self._completed) >= self.embedding_capacity:
            raise PromptQueueError("completed embedding queue overflow")
        completed = replace(request, encoding_finished_at=finished_at)
        self._completed.append(completed)
        return completed

    def activate_at_boundary(self, boundary_at: float) -> PromptRequest | None:
        self._validate_time(boundary_at)
        if not self._completed:
            return None
        newest = self._completed.pop()
        while self._completed:
            self._superseded.append(replace(self._completed.popleft(), superseded=True))
        if newest.encoding_finished_at is None or newest.encoding_finished_at > boundary_at:
            raise PromptQueueError("embedding was not ready at the declared boundary")
        self._active = replace(newest, activated_at=boundary_at)
        return self._active

    @staticmethod
    def _validate_time(value: float) -> None:
        if value < 0:
            raise PromptQueueError("prompt timestamps must be non-negative")
