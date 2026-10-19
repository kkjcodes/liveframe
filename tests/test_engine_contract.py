from __future__ import annotations

import unittest
from collections.abc import Iterator

from liveframe import (
    EngineError,
    GenerationRequest,
    NativeFrame,
    benchmark,
    clip,
    stream,
)


class StepClock:
    def __init__(self, step: float = 0.25) -> None:
        self.value = -step
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


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


class EngineTests(unittest.TestCase):
    def test_stream_counts_native_frames_inside_chunks(self) -> None:
        request = GenerationRequest("test prompt", native_frames=5)

        events = tuple(stream(FakeBackend(chunk_size=2), request, clock=StepClock()))

        self.assertEqual([event.frame.index for event in events], [0, 2, 4])
        self.assertEqual(events[-1].generated_native_frames, 5)

    def test_clip_reports_complete_wall_metrics(self) -> None:
        request = GenerationRequest("test prompt", native_frames=4)

        result = clip(FakeBackend(), request, clock=StepClock())

        self.assertEqual(result.metrics.generated_native_frames, 4)
        self.assertEqual(result.metrics.wall_seconds, 1.25)
        self.assertEqual(result.metrics.generated_fps, 3.2)
        self.assertEqual(result.metrics.time_to_first_frame_seconds, 0.25)

    def test_stream_fails_closed_on_incomplete_clip(self) -> None:
        request = GenerationRequest("test prompt", native_frames=2)

        with self.assertRaisesRegex(EngineError, "emitted 1 native frames"):
            tuple(stream(ShortBackend(), request, clock=StepClock()))

    def test_benchmark_separates_warmups_from_trials(self) -> None:
        request = GenerationRequest("test prompt", native_frames=2)

        result = benchmark(
            FakeBackend(), request, warmup_runs=1, trials=3, clock=StepClock()
        )

        self.assertEqual(result.warmup_runs, 1)
        self.assertEqual(len(result.trials), 3)
        self.assertEqual(result.median_wall_seconds, 0.75)
        self.assertEqual(result.p95_wall_seconds, 0.75)


if __name__ == "__main__":
    unittest.main()