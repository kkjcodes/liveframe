"""Ground-truth validation of the control and stability metrics (spec 8.5).

The synthetic world renders real camera motion, so every clip's commanded
class, onset delay, and injected defect is known exactly. The M3 gate
requires >= 95% classification across the six motion classes plus no-op,
onset recovery within +-1 frame, and every stability defect detected with
zero false collapse flags on clean runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from liveframe.world.adapters.synthetic import (
    FakeClock,
    SyntheticVisualDefects,
    SyntheticWorld,
)
from liveframe.world.metrics.control import (
    analyze_segments,
    classify_pair,
    pair_features,
    to_grayscale,
)
from liveframe.world.metrics.stability import collapse_detectors

MOTION_CLASSES = ("W", "S", "A", "D", "L", "R")
ALL_CLASSES = (*MOTION_CLASSES, "0")
RESOLUTION = (144, 256)


def render_clip(
    action: str,
    frames: int,
    *,
    seed: int = 11,
    motion_scale: float = 1.0,
    noise_amplitude: int = 0,
    visual_defects: SyntheticVisualDefects | None = None,
    lead_in: tuple[str, int] | None = None,
) -> np.ndarray:
    """Render a labeled clip; optional lead-in segment precedes the action."""

    clock = FakeClock()
    world = SyntheticWorld(
        resolution=RESOLUTION,
        chunk_frames=4,
        chunk_delay_seconds=0.01,
        motion_scale=motion_scale,
        visual_defects=visual_defects,
        clock=clock,
    )
    world.load()
    world.reset(np.zeros((*RESOLUTION, 3), dtype=np.uint8), None, seed)
    plan: list[tuple[str, int]] = []
    if lead_in is not None:
        plan.append(lead_in)
    plan.append((action, frames))
    chunks: list[np.ndarray] = []
    for tokens, count in plan:
        remaining = count
        while remaining > 0:
            step_frames = min(4, remaining)
            for frame in world.step(tokens.split("+")):
                chunks.append(np.asarray(frame.payload)[: step_frames])
            remaining -= step_frames
    clip = np.concatenate(chunks, axis=0)
    if noise_amplitude:
        rng = np.random.default_rng(seed + 1)
        noise = rng.integers(
            -noise_amplitude, noise_amplitude + 1, size=clip.shape, dtype=np.int16
        )
        clip = np.clip(clip.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return clip


def classify_clip(clip: np.ndarray, *, settle_pairs: int = 2) -> str:
    """Majority vote over the clip's pairs after a short settle window."""

    gray = to_grayscale(clip)
    votes: dict[str, int] = {}
    for index in range(settle_pairs, len(gray) - 1):
        label = classify_pair(pair_features(gray[index], gray[index + 1]))
        votes[label] = votes.get(label, 0) + 1
    return max(votes, key=lambda key: votes[key])


def calibration_sweep(
    *, speeds: tuple[float, ...] = (0.5, 1.0, 2.0), frames: int = 12
) -> dict[str, Any]:
    """Per-class feature medians: the evidence behind the frozen thresholds.

    Reproduces the separations the classifier constants rely on (notably the
    row-variation ratio: strafe ~5.4 vs yaw <=1.2); committed with the
    validation results so every threshold traces to a measured artifact.
    """

    sweep: dict[str, Any] = {}
    for action in ALL_CLASSES:
        for speed in speeds:
            clip = render_clip(action, frames, motion_scale=speed)
            gray = to_grayscale(clip)
            features = [
                pair_features(gray[i], gray[i + 1]) for i in range(2, len(gray) - 1)
            ]
            sweep[f"{action}@{speed}"] = {
                "median_flow": float(np.median([f.median_flow for f in features])),
                "mean_dx": float(np.median([f.mean_dx for f in features])),
                "divergence": float(np.median([f.divergence for f in features])),
                "row_variation_ratio": float(
                    np.median([f.row_variation_ratio for f in features])
                ),
            }
    return sweep


def validate_control(
    *,
    speeds: tuple[float, ...] = (0.5, 1.0, 2.0),
    noise_levels: tuple[int, ...] = (0, 8),
    frames: int = 20,
) -> dict[str, Any]:
    cells = []
    correct = 0
    for action in ALL_CLASSES:
        for speed in speeds:
            for noise in noise_levels:
                clip = render_clip(
                    action, frames, motion_scale=speed, noise_amplitude=noise
                )
                predicted = classify_clip(clip)
                ok = predicted == action
                correct += ok
                cells.append(
                    {
                        "action": action,
                        "speed": speed,
                        "noise": noise,
                        "predicted": predicted,
                        "correct": ok,
                    }
                )
    total = len(cells)
    per_class: dict[str, dict[str, float]] = {}
    for action in ALL_CLASSES:
        rows = [cell for cell in cells if cell["action"] == action]
        per_class[action] = {
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "cells": float(len(rows)),
        }
    return {
        "cells": cells,
        "total": total,
        "accuracy": correct / total,
        "per_class": per_class,
    }


def validate_onset(
    *, delays: tuple[int, ...] = (0, 4, 8, 16), frames: int = 28
) -> dict[str, Any]:
    rows = []
    for delay in delays:
        lead = 8
        clip = render_clip(
            "W",
            frames,
            visual_defects=SyntheticVisualDefects(onset_delay_frames=delay),
            lead_in=("0", lead),
        )
        segments = [(0, "0", lead), (lead, "W", frames)]
        results, _, _ = analyze_segments(clip, segments)
        measured = results[1].motion_onset_frames
        rows.append(
            {
                "injected_delay": delay,
                "measured_onset_pairs": measured,
                "within_one_frame": measured is not None and abs(measured - delay) <= 1,
            }
        )
    return {"rows": rows, "all_within_one": all(row["within_one_frame"] for row in rows)}


def validate_stability_defects(*, frames: int = 48) -> dict[str, Any]:
    motion = [True] * (frames - 1)
    scenarios = {
        "clean": (None, "none"),
        "freeze": (SyntheticVisualDefects(freeze_from_frame=frames // 2), "frozen_runs"),
        "black": (SyntheticVisualDefects(black_from_frame=frames // 2), "black_frames"),
    }
    report: dict[str, Any] = {}
    for name, (defects, expected_flag) in scenarios.items():
        clip = render_clip("W", frames, visual_defects=defects)
        flags = collapse_detectors(clip, motion_commanded=motion)
        if name == "clean":
            report[name] = {
                "collapse_flag_count": flags["collapse_flag_count"],
                "passed": flags["collapse_flag_count"] == 0,
            }
        else:
            report[name] = {
                "flag": expected_flag,
                "count": len(flags[expected_flag]),
                "passed": len(flags[expected_flag]) > 0,
            }
    # Drift defects are detected through windowed drift slopes.
    from liveframe.world.metrics.stability import frame_statistics, windowed_drift

    bright = render_clip(
        "W", frames, visual_defects=SyntheticVisualDefects(brightness_drift_per_frame=1.5)
    )
    drift = windowed_drift(frame_statistics(bright), native_fps=16.0, window_seconds=1.0)
    slope = drift["brightness_mean"]["slope_per_window_normalized"]
    report["brightness_drift"] = {"slope": slope, "passed": slope is not None and slope > 0.02}

    blurred = render_clip(
        "0",
        frames,
        visual_defects=SyntheticVisualDefects(blur_from_frame=0, blur_growth_per_frame=0.25),
    )
    drift = windowed_drift(frame_statistics(blurred), native_fps=16.0, window_seconds=1.0)
    slope = drift["sharpness"]["slope_per_window_normalized"]
    report["blur_drift"] = {"slope": slope, "passed": slope is not None and slope < -0.02}

    wrong = render_clip(
        "W", 20, visual_defects=SyntheticVisualDefects(wrong_direction=True)
    )
    predicted = classify_clip(wrong)
    report["wrong_direction"] = {"predicted": predicted, "passed": predicted == "S"}

    # Repeat loop: tile a short segment so frames recur with a gap >= 8.
    segment = render_clip("L", 12)
    looped = np.concatenate([segment, segment], axis=0)
    loop_flags = collapse_detectors(looped, motion_commanded=[True] * (len(looped) - 1))
    report["repeat_loop"] = {
        "count": len(loop_flags["repeat_loops"]),
        "passed": len(loop_flags["repeat_loops"]) > 0,
    }

    report["all_passed"] = all(
        bool(entry["passed"]) for entry in report.values() if isinstance(entry, dict)
    )
    return report
