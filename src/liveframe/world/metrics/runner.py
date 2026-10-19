"""Post-pass metric orchestration (spec section 8): read captured frames,
compute control + stability metrics, write ``metrics.json`` into the case
directory.

Runs strictly after the timed pass, from saved frames, so it can be rerun or
extended without regenerating video; results are deterministic for a fixed
capture (the M3 gate pins a byte-identical rerun).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from liveframe.world.metrics.control import analyze_segments, commanded_class
from liveframe.world.metrics.stability import (
    collapse_detectors,
    frame_statistics,
    temporal_flicker,
    windowed_drift,
)
from liveframe.world.sinks import load_captured_frames

METRICS_SCHEMA = "liveframe.world.case-metrics.v1"


def segments_from_case(case_summary: dict[str, Any]) -> list[tuple[int, str, int]]:
    """Reconstruct (t_frames, action, hold) rows from per-chunk timing rows."""

    rows: list[tuple[int, str, int]] = []
    cursor = 0
    for chunk in case_summary.get("chunks", []):
        action = str(chunk["action"])
        frames = int(chunk["native_frames"])
        if rows and rows[-1][1] == action:
            t, previous_action, hold = rows[-1]
            rows[-1] = (t, previous_action, hold + frames)
        else:
            rows.append((cursor, action, frames))
        cursor += frames
    return rows


def _timing_rows(timing_path: Path) -> list[dict[str, Any]]:
    rows = []
    with timing_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _segments_from_timing(timing_rows: list[dict[str, Any]]) -> list[tuple[int, str, int]]:
    rows: list[tuple[int, str, int]] = []
    cursor = 0
    for chunk in timing_rows:
        action = str(chunk["action"])
        frames = int(chunk["native_frames"])
        if rows and rows[-1][1] == action:
            t, previous_action, hold = rows[-1]
            rows[-1] = (t, previous_action, hold + frames)
        else:
            rows.append((cursor, action, frames))
        cursor += frames
    return rows


def cost_per_generated_minute(
    *,
    wall_seconds: float,
    native_frames: int,
    native_fps: float,
    gpu_price_usd_per_hour: float,
) -> float:
    """Dollars of GPU time per minute of generated video (spec 7.3)."""

    if min(wall_seconds, native_frames, native_fps, gpu_price_usd_per_hour) <= 0:
        raise ValueError("cost inputs must be positive")
    generated_minutes = native_frames / native_fps / 60.0
    return (wall_seconds / 3600.0 * gpu_price_usd_per_hour) / generated_minutes


def _perceived_latency(
    timing_rows: list[dict[str, Any]],
    control_segments: list[Any],
    segments: list[tuple[int, str, int]],
    native_fps: float,
) -> dict[str, Any]:
    """Join per-change wall latency with metric-pass motion onset (spec 7.2).

    ``perceived_latency[k] = control_latency_wall[k] + onset_frames[k] / fps``
    over action changes; the wall half comes from the timed pass's per-chunk
    submit/receipt stamps, the onset half from the captured-frame analysis.
    """

    onset_by_start: dict[int, int | None] = {}
    for row, (t_frames, _, _) in zip(control_segments, segments, strict=False):
        onset_by_start[t_frames] = row.motion_onset_frames

    samples = []
    for chunk in timing_rows:
        if not chunk.get("action_changed"):
            continue
        wall = float(chunk["receipt_ts"]) - float(chunk["submit_ts"])
        onset = onset_by_start.get(int(chunk["start_frame"]))
        if onset is None:
            continue
        samples.append(
            {
                "start_frame": int(chunk["start_frame"]),
                "control_latency_wall": wall,
                "motion_onset_frames": onset,
                "perceived_latency_seconds": wall + onset / native_fps,
            }
        )
    values = sorted(s["perceived_latency_seconds"] for s in samples)
    return {
        "samples": samples,
        "count": len(values),
        "p50": values[len(values) // 2] if values else None,
        "max": values[-1] if values else None,
    }


def run_metric_pass(
    case_dir: Path,
    *,
    native_fps: float,
    window_seconds: float = 10.0,
) -> dict[str, Any]:
    """Compute the CPU metric tier for one captured case and persist it."""

    started = time.perf_counter()
    frames = load_captured_frames(case_dir / "frames_256")
    timing_rows = _timing_rows(case_dir / "timing.jsonl")
    segments = _segments_from_timing(timing_rows)
    total_frames = sum(hold for _, _, hold in segments)
    if total_frames != len(frames):
        raise ValueError(
            f"captured frames ({len(frames)}) do not match timing rows ({total_frames})"
        )

    control_segments, pair_classes, _ = analyze_segments(frames, segments)

    motion_commanded = []
    for pair_index in range(len(frames) - 1):
        frame_of_pair = pair_index + 1
        action = next(
            (
                action
                for t, action, hold in segments
                if t <= frame_of_pair < t + hold
            ),
            "0",
        )
        motion_commanded.append(commanded_class(action) not in (None, "0"))

    statistics = frame_statistics(frames)
    flicker = temporal_flicker(frames)
    collapse = collapse_detectors(frames, motion_commanded=motion_commanded)
    drift = windowed_drift(
        {**statistics, "temporal_flicker": flicker},
        native_fps=native_fps,
        window_seconds=window_seconds,
    )

    metrics: dict[str, Any] = {
        "schema": METRICS_SCHEMA,
        "frames_analyzed": len(frames),
        "control": {
            "segments": [segment.as_dict() for segment in control_segments],
            "pair_classes": pair_classes,
        },
        "perceived_latency": _perceived_latency(
            timing_rows, control_segments, segments, native_fps
        ),
        "stability": {
            "windows": drift,
            "collapse": collapse,
            "flicker_mean": float(flicker.mean()) if len(flicker) else None,
        },
        "metric_pass_seconds": None,  # filled below so it is part of the record
    }
    metrics["metric_pass_seconds"] = time.perf_counter() - started

    output = case_dir / "metrics.json"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return metrics


def deterministic_view(metrics: dict[str, Any]) -> dict[str, Any]:
    """The metrics payload minus wall-clock fields, for determinism checks."""

    view: dict[str, Any] = json.loads(json.dumps(metrics))
    view.pop("metric_pass_seconds", None)
    return view


def summarize_for_report(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compact headline view used by gate reports and (later) the HTML report."""

    control = metrics["control"]["segments"]
    accuracies = [
        segment["direction_accuracy"]
        for segment in control
        if segment["direction_accuracy"] is not None and segment["commanded_class"] != "0"
    ]
    onsets = [
        segment["motion_onset_frames"]
        for segment in control
        if segment["motion_onset_frames"] is not None and segment["commanded_class"] != "0"
    ]
    return {
        "direction_accuracy_mean": float(np.mean(accuracies)) if accuracies else None,
        "motion_onset_frames_median": float(np.median(onsets)) if onsets else None,
        "collapse_flag_count": metrics["stability"]["collapse"]["collapse_flag_count"],
        "flicker_mean": metrics["stability"]["flicker_mean"],
        "metric_pass_seconds": metrics["metric_pass_seconds"],
    }
