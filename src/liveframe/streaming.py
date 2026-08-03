"""Native-rate producer/display queue with exact causal accounting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


class QueueError(RuntimeError):
    """Raised when a bounded stream queue would violate its contract."""


@dataclass(frozen=True, slots=True)
class QueueSample:
    elapsed_seconds: float
    occupancy: int
    generated: int
    displayed: int
    underruns: int


class NativeFrameQueue:
    """Bounded FIFO accounting genuinely generated and successfully displayed frames."""

    def __init__(self, capacity: int, initial_frames: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if not 0 <= initial_frames <= capacity:
            raise ValueError("initial_frames must be within queue capacity")
        self.capacity = capacity
        self.q_start = initial_frames
        self._frames: deque[int] = deque(range(-initial_frames, 0))
        self._generated_ids: set[int] = set()
        self.generated = 0
        self.displayed = 0
        self.underruns = 0
        self._last_elapsed = 0.0

    @property
    def occupancy(self) -> int:
        return len(self._frames)

    def generate(self, frame_id: int, elapsed_seconds: float) -> QueueSample:
        self._validate_time(elapsed_seconds)
        if frame_id < 0:
            raise ValueError("post-start generated frame IDs must be non-negative")
        if frame_id in self._generated_ids:
            raise QueueError(f"duplicate frame ID: {frame_id}")
        if self.occupancy >= self.capacity:
            raise QueueError("frame queue overflow")
        self._frames.append(frame_id)
        self._generated_ids.add(frame_id)
        self.generated += 1
        return self.sample(elapsed_seconds)

    def display_tick(self, elapsed_seconds: float) -> tuple[int | None, QueueSample]:
        self._validate_time(elapsed_seconds)
        frame_id: int | None = None
        if self._frames:
            frame_id = self._frames.popleft()
            self.displayed += 1
        else:
            self.underruns += 1
        return frame_id, self.sample(elapsed_seconds)

    def sample(self, elapsed_seconds: float) -> QueueSample:
        self._validate_time(elapsed_seconds)
        expected = self.q_start + self.generated - self.displayed
        if expected != self.occupancy:
            raise QueueError(
                f"queue accounting mismatch: expected {expected}, observed {self.occupancy}"
            )
        return QueueSample(
            elapsed_seconds=elapsed_seconds,
            occupancy=self.occupancy,
            generated=self.generated,
            displayed=self.displayed,
            underruns=self.underruns,
        )

    def _validate_time(self, elapsed_seconds: float) -> None:
        if elapsed_seconds < self._last_elapsed:
            raise QueueError("stream timestamps must be monotonic")
        if elapsed_seconds < 0:
            raise QueueError("stream timestamps must be non-negative")
        self._last_elapsed = elapsed_seconds
