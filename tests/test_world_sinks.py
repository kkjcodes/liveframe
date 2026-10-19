from __future__ import annotations

import numpy as np
import pytest

from liveframe.world.sinks import FrameCapture, downscale_short_side, load_captured_frames


def test_downscale_is_deterministic_area_mean() -> None:
    frames = np.arange(2 * 512 * 1024 * 3, dtype=np.int64).astype(np.uint8)
    frames = frames.reshape(2, 512, 1024, 3)
    small = downscale_short_side(frames, target=256)
    assert small.shape == (2, 256, 512, 3)
    assert small.dtype == np.uint8
    block = frames[0, :2, :2, 0].astype(np.float32).mean()
    assert small[0, 0, 0, 0] == np.uint8(np.clip(np.rint(block), 0, 255))
    again = downscale_short_side(frames, target=256)
    assert np.array_equal(small, again)


def test_downscale_leaves_small_frames_unchanged() -> None:
    frames = np.zeros((1, 144, 256, 3), dtype=np.uint8)
    out = downscale_short_side(frames, target=256)
    assert out.shape == frames.shape


def test_capture_round_trips_frames(tmp_path) -> None:
    capture = FrameCapture(tmp_path / "frames", target_short_side=64, frames_per_file=5)
    rng = np.random.default_rng(3)
    emitted = []
    for _ in range(4):
        chunk = rng.integers(0, 256, size=(3, 64, 96, 3), dtype=np.uint8)
        emitted.append(chunk)
        capture.add(chunk)
    files = capture.finalize()
    assert len(files) == 3  # 12 frames at 5 per file
    loaded = load_captured_frames(tmp_path / "frames")
    assert loaded.shape == (12, 64, 96, 3)
    assert np.array_equal(loaded, np.concatenate(emitted, axis=0))


def test_load_rejects_non_contiguous_chunks(tmp_path) -> None:
    capture = FrameCapture(tmp_path / "frames", frames_per_file=2)
    capture.add(np.zeros((4, 32, 48, 3), dtype=np.uint8))
    files = capture.finalize()
    files[0].unlink()
    with pytest.raises(ValueError, match="not contiguous"):
        load_captured_frames(tmp_path / "frames")
