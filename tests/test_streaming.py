from __future__ import annotations

import pytest

from liveframe.streaming import NativeFrameQueue, QueueError


def test_native_queue_exact_accounting() -> None:
    queue = NativeFrameQueue(capacity=8, initial_frames=1)
    frame, first = queue.display_tick(0.0)
    assert frame == -1
    assert first.occupancy == 0

    queue.generate(0, 0.02)
    frame, second = queue.display_tick(0.0625)

    assert frame == 0
    assert second.generated == 1
    assert second.displayed == 2
    assert second.occupancy == 0
    assert second.underruns == 0


def test_queue_records_underrun_without_fake_display() -> None:
    queue = NativeFrameQueue(capacity=4)

    frame, sample = queue.display_tick(0.0625)

    assert frame is None
    assert sample.displayed == 0
    assert sample.underruns == 1
    assert sample.occupancy == 0


def test_queue_rejects_overflow_and_non_monotonic_events() -> None:
    queue = NativeFrameQueue(capacity=1)
    queue.generate(0, 0.1)

    with pytest.raises(QueueError, match="overflow"):
        queue.generate(1, 0.2)
    with pytest.raises(QueueError, match="monotonic"):
        queue.display_tick(0.05)


def test_queue_rejects_replaying_displayed_frame() -> None:
    queue = NativeFrameQueue(capacity=1)
    queue.generate(0, 0.01)
    queue.display_tick(0.02)

    with pytest.raises(QueueError, match="duplicate"):
        queue.generate(0, 0.03)
