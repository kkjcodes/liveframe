"""Runtime-neutral orchestration for measured frame generation.

Backends own model loading and frame production. This module owns the public
measurement contract: every emitted item declares how many native frames it
contains, clip runs must complete exactly, and benchmark summaries use measured
wall time rather than playback duration.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class EngineError(RuntimeError):
    """Raised when a backend violates the generation contract."""


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """A finite native-frame generation request."""

    prompt: str
    native_frames: int = 81
    native_fps: float = 16.0
    width: int = 832
    height: int = 480
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.native_frames < 1:
            raise ValueError("native_frames must be positive")
        if not math.isfinite(self.native_fps) or self.native_fps <= 0:
            raise ValueError("native_fps must be finite and positive")
        if self.width < 1 or self.height < 1:
            raise ValueError("resolution must be positive")

    @property
    def playback_seconds(self) -> float:
        return self.native_frames / self.native_fps


@dataclass(frozen=True, slots=True)
class NativeFrame:
    """One backend emission containing one or more native generated frames."""

    index: int
    payload: Any
    native_frame_count: int = 1
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("frame index must be non-negative")
        if self.native_frame_count < 1:
            raise ValueError("native_frame_count must be positive")


@runtime_checkable
class StreamingBackend(Protocol):
    """Backend contract consumed by the public engine APIs."""

    def stream(self, request: GenerationRequest) -> Iterator[NativeFrame]: ...


@dataclass(frozen=True, slots=True)
class StreamEvent:
    frame: NativeFrame
    elapsed_seconds: float
    generated_native_frames: int


@dataclass(frozen=True, slots=True)
class ClipMetrics:
    generated_native_frames: int
    wall_seconds: float
    generated_fps: float
    time_to_first_frame_seconds: float
    p95_emission_interval_seconds: float


@dataclass(frozen=True, slots=True)
class ClipResult:
    request: GenerationRequest
    events: tuple[StreamEvent, ...]
    metrics: ClipMetrics


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    request: GenerationRequest
    warmup_runs: int
    trials: tuple[ClipMetrics, ...]
    median_wall_seconds: float
    median_generated_fps: float
    p95_wall_seconds: float


Clock = Callable[[], float]


def stream(
    backend: StreamingBackend,
    request: GenerationRequest,
    *,
    clock: Clock = time.perf_counter,
) -> Iterator[StreamEvent]:
    """Yield measured backend emissions and enforce exact native-frame accounting."""

    return _stream_events(backend, request, clock=clock, started_at=clock())


def clip(
    backend: StreamingBackend,
    request: GenerationRequest,
    *,
    clock: Clock = time.perf_counter,
) -> ClipResult:
    """Generate one finite clip and return its frames with complete-wall metrics."""

    started_at = clock()
    events = tuple(_stream_events(backend, request, clock=clock, started_at=started_at))
    finished_at = clock()
    wall_seconds = _elapsed(started_at, finished_at)
    if wall_seconds == 0:
        raise EngineError("clip wall time must be greater than zero")
    intervals = []
    previous = 0.0
    for event in events:
        intervals.append(event.elapsed_seconds - previous)
        previous = event.elapsed_seconds
    metrics = ClipMetrics(
        generated_native_frames=request.native_frames,
        wall_seconds=wall_seconds,
        generated_fps=request.native_frames / wall_seconds,
        time_to_first_frame_seconds=events[0].elapsed_seconds,
        p95_emission_interval_seconds=_nearest_rank(intervals, 0.95),
    )
    return ClipResult(request=request, events=events, metrics=metrics)


def benchmark(
    backend: StreamingBackend,
    request: GenerationRequest,
    *,
    trials: int = 3,
    warmup_runs: int = 1,
    clock: Clock = time.perf_counter,
) -> BenchmarkResult:
    """Run warmups and measured clips, returning per-trial and aggregate metrics."""

    if trials < 1:
        raise ValueError("trials must be positive")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    for _ in range(warmup_runs):
        clip(backend, request, clock=clock)
    measured = tuple(clip(backend, request, clock=clock).metrics for _ in range(trials))
    walls = [trial.wall_seconds for trial in measured]
    rates = [trial.generated_fps for trial in measured]
    return BenchmarkResult(
        request=request,
        warmup_runs=warmup_runs,
        trials=measured,
        median_wall_seconds=statistics.median(walls),
        median_generated_fps=statistics.median(rates),
        p95_wall_seconds=_nearest_rank(walls, 0.95),
    )


def _stream_events(
    backend: StreamingBackend,
    request: GenerationRequest,
    *,
    clock: Clock,
    started_at: float,
) -> Iterator[StreamEvent]:
    generated = 0
    previous_elapsed = 0.0
    for frame in backend.stream(request):
        if not isinstance(frame, NativeFrame):
            raise EngineError("backend stream must yield NativeFrame instances")
        if frame.index != generated:
            raise EngineError(
                f"backend frame index mismatch: expected {generated}, observed {frame.index}"
            )
        generated += frame.native_frame_count
        if generated > request.native_frames:
            raise EngineError(
                f"backend emitted {generated} native frames; expected {request.native_frames}"
            )
        elapsed = _elapsed(started_at, clock())
        if elapsed < previous_elapsed:
            raise EngineError("clock must be monotonic")
        previous_elapsed = elapsed
        yield StreamEvent(
            frame=frame,
            elapsed_seconds=elapsed,
            generated_native_frames=generated,
        )
    if generated != request.native_frames:
        raise EngineError(
            f"backend emitted {generated} native frames; expected {request.native_frames}"
        )


def _elapsed(started_at: float, finished_at: float) -> float:
    elapsed = finished_at - started_at
    if not math.isfinite(elapsed) or elapsed < 0:
        raise EngineError("clock must return finite monotonic values")
    return elapsed


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]