from __future__ import annotations

from collections.abc import Iterator

import pytest

from liveframe import EngineError, GenerationRequest, NativeFrame, benchmark, clip, stream


class StepClock:
    def __init__(self, step: float = 0.25) -> None:
        self.value = -step
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


class ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeBackend:
    def __init__(self, *, chunk_size: int = 1) -> None:
        self.chunk_size = chunk_size

    def stream(self, request: GenerationRequest) -> Iterator[NativeFrame]:
        for index in range(0, request.native_frames, self.chunk_size):
            count = min(self.chunk_size, request.native_frames - index)
            yield NativeFrame(index=index, payload=f"frame-{index}", native_frame_count=count)


class ShortBackend:
    def stream(self, request: GenerationRequest) -> Iterator[NativeFrame]:
        del request
        yield NativeFrame(index=0, payload=None)


def test_stream_counts_native_frames_inside_chunks() -> None:
    request = GenerationRequest("test prompt", native_frames=5)

    events = tuple(stream(FakeBackend(chunk_size=2), request, clock=StepClock()))

    assert [event.frame.index for event in events] == [0, 2, 4]
    assert events[-1].generated_native_frames == 5


def test_clip_reports_complete_wall_metrics() -> None:
    request = GenerationRequest("test prompt", native_frames=4)

    result = clip(FakeBackend(), request, clock=StepClock())

    assert result.metrics.wall_seconds == 1.25
    assert result.metrics.generated_fps == 3.2
    assert result.metrics.time_to_first_frame_seconds == 0.25


def test_clip_can_sink_payloads_without_retaining_events() -> None:
    clock = ManualClock()
    payloads = []

    def sink(frame: NativeFrame) -> None:
        payloads.append(frame.payload)
        clock.advance(0.5)

    result = clip(
        FakeBackend(),
        GenerationRequest("test prompt", native_frames=2),
        clock=clock,
        frame_sink=sink,
        retain_events=False,
    )

    assert payloads == ["frame-0", "frame-1"]
    assert result.events == ()
    assert result.metrics.wall_seconds == 1.0
    assert result.metrics.time_to_first_frame_seconds == 0.5
    assert result.metrics.p95_emission_interval_seconds == 0.5


def test_stream_fails_closed_on_incomplete_clip() -> None:
    request = GenerationRequest("test prompt", native_frames=2)

    with pytest.raises(EngineError, match="emitted 1 native frames"):
        tuple(stream(ShortBackend(), request, clock=StepClock()))


def test_benchmark_separates_warmups_from_trials() -> None:
    request = GenerationRequest("test prompt", native_frames=2)

    result = benchmark(FakeBackend(), request, warmup_runs=1, trials=3, clock=StepClock())

    assert result.warmup_runs == 1
    assert len(result.trials) == 3
    assert result.median_wall_seconds == 0.75