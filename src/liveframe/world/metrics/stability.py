"""Stability metrics over the horizon (spec section 8.3).

Per-frame statistics aggregated per 10-second window so drift is visible,
plus collapse detectors with hard definitions:

- ``black_frames``: mean luminance < 8/255;
- ``frozen_runs``: >= 3 consecutive frames whose mean absolute difference is
  < 0.5/255 while a motion action is commanded;
- ``repeat_loops``: perceptual-hash (8x8 average hash) repeats within a
  64-frame window with a gap of at least 8 frames (adjacent near-identical
  frames are stillness, not loops), confirmed at pixel level (mean absolute
  difference < 2/255) because the 64-bit hash alone collides on genuinely
  different frames (observed during M3 validation);
- ``drift_slope``: per-metric linear-fit slope over window means, normalized
  by the first window's value.

Everything is deterministic for fixed frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from liveframe.world.metrics.control import farneback_flow, to_grayscale

BLACK_LUMA_THRESHOLD = 8.0
FROZEN_DIFF_THRESHOLD = 0.5
FROZEN_MIN_RUN = 3
REPEAT_WINDOW = 64
REPEAT_MIN_GAP = 8
REPEAT_CONFIRM_DIFF = 2.0


def frame_statistics(frames: np.ndarray) -> dict[str, np.ndarray]:
    gray = to_grayscale(frames)
    red = frames[..., 0].astype(np.float32)
    green = frames[..., 1].astype(np.float32)
    blue = frames[..., 2].astype(np.float32)
    rg = red - green
    yb = 0.5 * (red + green) - blue
    colorfulness = np.sqrt(rg.std(axis=(1, 2)) ** 2 + yb.std(axis=(1, 2)) ** 2) + 0.3 * np.sqrt(
        rg.mean(axis=(1, 2)) ** 2 + yb.mean(axis=(1, 2)) ** 2
    )
    sharpness = np.array(
        [float(cv2.Laplacian(gray[i], cv2.CV_32F).var()) for i in range(len(gray))]
    )
    return {
        "brightness_mean": gray.mean(axis=(1, 2)),
        "contrast_std": gray.std(axis=(1, 2)),
        "colorfulness": colorfulness,
        "sharpness": sharpness,
    }


def temporal_flicker(frames: np.ndarray) -> np.ndarray:
    """Mean |difference| per pair after flow-compensated warping."""

    gray = to_grayscale(frames)
    height, width = gray.shape[1:]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    values = []
    for index in range(len(gray) - 1):
        flow = farneback_flow(gray[index], gray[index + 1])
        warped = cv2.remap(
            gray[index],
            grid_x + flow[..., 0],
            grid_y + flow[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        values.append(float(np.abs(warped - gray[index + 1]).mean()))
    return np.array(values)


def _average_hash(gray_frame: np.ndarray) -> int:
    small = cv2.resize(gray_frame, (8, 8), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).ravel()
    return int(np.packbits(bits).view(">u8")[0]) if bits.size == 64 else 0


def collapse_detectors(
    frames: np.ndarray, *, motion_commanded: Sequence[bool]
) -> dict[str, Any]:
    """Collapse flags; ``motion_commanded[i]`` covers frame pair (i, i+1)."""

    gray = to_grayscale(frames)
    luma = gray.mean(axis=(1, 2))
    black = [int(i) for i in np.nonzero(luma < BLACK_LUMA_THRESHOLD)[0]]

    diffs = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))
    frozen_runs: list[dict[str, int]] = []
    run_start: int | None = None
    for index, diff in enumerate(diffs):
        is_frozen = diff < FROZEN_DIFF_THRESHOLD and bool(motion_commanded[index])
        if is_frozen and run_start is None:
            run_start = index
        elif not is_frozen and run_start is not None:
            if index - run_start >= FROZEN_MIN_RUN - 1:
                frozen_runs.append({"start_frame": run_start, "pairs": index - run_start})
            run_start = None
    if run_start is not None and len(diffs) - run_start >= FROZEN_MIN_RUN - 1:
        frozen_runs.append({"start_frame": run_start, "pairs": int(len(diffs) - run_start)})

    hashes = [_average_hash(gray[i]) for i in range(len(gray))]
    repeats: list[dict[str, int]] = []
    # All in-window occurrences per hash: consecutive frames often share a
    # coarse hash, so keying on the most recent occurrence alone would hide
    # true loops behind sub-gap neighbors.
    occurrences: dict[int, list[int]] = {}
    for index, value in enumerate(hashes):
        # Stillness during a commanded no-op is legitimate, exactly as for
        # frozen runs: only flag repeats where the content should be moving.
        moving = index >= 1 and bool(motion_commanded[index - 1])
        candidates = (
            [
                previous
                for previous in occurrences.get(value, [])
                if REPEAT_MIN_GAP <= index - previous <= REPEAT_WINDOW
            ]
            if moving
            else []
        )
        for previous in reversed(candidates):
            pixel_diff = float(np.abs(gray[index] - gray[previous]).mean())
            if pixel_diff < REPEAT_CONFIRM_DIFF:
                repeats.append({"frame": index, "matches_frame": previous})
                break
        occurrences.setdefault(value, []).append(index)
        occurrences[value] = [
            previous for previous in occurrences[value] if index - previous <= REPEAT_WINDOW
        ]
    return {
        "black_frames": black,
        "frozen_runs": frozen_runs,
        "repeat_loops": repeats,
        "collapse_flag_count": len(black) + len(frozen_runs) + len(repeats),
    }


def windowed_drift(
    per_frame: dict[str, np.ndarray], *, native_fps: float, window_seconds: float = 10.0
) -> dict[str, Any]:
    frames_per_window = max(1, round(window_seconds * native_fps))
    windows: dict[str, list[float]] = {name: [] for name in per_frame}
    for name, series in per_frame.items():
        for start in range(0, len(series), frames_per_window):
            chunk = series[start : start + frames_per_window]
            if len(chunk):
                windows[name].append(float(np.mean(chunk)))
    drift: dict[str, Any] = {}
    for name, means in windows.items():
        if len(means) < 2 or abs(means[0]) < 1e-9:
            drift[name] = {"windows": means, "slope_per_window_normalized": None}
            continue
        xs = np.arange(len(means), dtype=np.float64)
        slope = float(np.polyfit(xs, np.array(means), 1)[0])
        drift[name] = {
            "windows": means,
            "slope_per_window_normalized": slope / means[0],
        }
    return drift
