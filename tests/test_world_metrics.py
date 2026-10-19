"""M3 gate tests: control classification, onset, stability, determinism.

The full validation grid (42 control cells, 4 onset delays, 7 defect
scenarios) runs in tools/validate_world_metrics.py and is committed as a
report; this module reruns a smaller version in CI (spec 8.5) plus the
determinism and runner round-trip gates.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from liveframe.world.actions import ActionScript
from liveframe.world.adapters.synthetic import FakeClock, SyntheticVisualDefects, SyntheticWorld
from liveframe.world.metrics.control import reversal_return, simple_ssim
from liveframe.world.metrics.runner import deterministic_view, run_metric_pass
from liveframe.world.metrics.stability import collapse_detectors
from liveframe.world.metrics.validation import (
    ALL_CLASSES,
    classify_clip,
    render_clip,
    validate_onset,
    validate_stability_defects,
)
from liveframe.world.session import WorldCase, run_case, write_case_directory
from liveframe.world.sinks import FrameCapture


@pytest.mark.parametrize("action", ALL_CLASSES)
def test_control_classifier_mini_grid(action: str) -> None:
    clip = render_clip(action, 16, motion_scale=1.0, noise_amplitude=8)
    assert classify_clip(clip) == action


def test_onset_mini() -> None:
    report = validate_onset(delays=(0, 8), frames=20)
    assert report["all_within_one"], report


def test_stability_defects_mini() -> None:
    report = validate_stability_defects(frames=32)
    assert report["all_passed"], report


def test_reversal_return_distinguishes_return_from_drift() -> None:
    # W then S for equal frames returns near the start pose; W then W drifts.
    forward = render_clip("W", 10)
    returned = render_clip("S", 10, lead_in=("W", 10))
    drifted = render_clip("W", 10, lead_in=("W", 10))
    ssim_return = reversal_return(returned, 9, len(returned) - 1)
    ssim_drift = reversal_return(drifted, 9, len(drifted) - 1)
    assert ssim_return > ssim_drift
    assert simple_ssim(forward[0], forward[0]) == pytest.approx(1.0, abs=1e-6)


def test_metric_pass_is_deterministic_and_round_trips(tmp_path) -> None:
    clock = FakeClock()
    world = SyntheticWorld(
        resolution=(72, 128), chunk_delay_seconds=0.02, clock=clock
    )
    world.load()
    case = WorldCase(
        case_id="metrics-case",
        start_image=np.zeros((72, 128, 3), dtype=np.uint8),
        script=ActionScript.from_segments([(0, "W", 16), (16, "L", 16), (32, "0", 8)]),
        seed=9,
    )
    case_dir = tmp_path / "cases" / case.case_id
    capture = FrameCapture(case_dir / "frames_256")
    result = run_case(world, case, clock=clock, capture=capture)
    assert result.status == "ok", result.error
    write_case_directory(result, case_dir)

    first = run_metric_pass(case_dir, native_fps=16.0)
    second = run_metric_pass(case_dir, native_fps=16.0)
    assert deterministic_view(first) == deterministic_view(second)

    persisted = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    assert persisted["schema"] == "liveframe.world.case-metrics.v1"
    segments = persisted["control"]["segments"]
    assert [segment["action"] for segment in segments] == ["W", "L", "0"]
    walk, yaw, hold = segments
    assert walk["direction_accuracy"] >= 0.9
    assert yaw["direction_accuracy"] >= 0.9
    assert hold["noop_drift"] < 0.08
    assert persisted["stability"]["collapse"]["collapse_flag_count"] == 0


def test_collapse_detector_ignores_stillness_when_commanded() -> None:
    # A no-op hold must not count as frozen: freezing requires commanded motion.
    clip = render_clip("0", 16)
    flags = collapse_detectors(clip, motion_commanded=[False] * (len(clip) - 1))
    assert flags["frozen_runs"] == []


def test_visual_defect_onset_delay_defers_motion() -> None:
    clip = render_clip(
        "W",
        12,
        visual_defects=SyntheticVisualDefects(onset_delay_frames=6),
        lead_in=("0", 8),
    )
    gray_diffs = np.abs(np.diff(clip.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))
    # Frames 8..13 are generated under the delayed action: still near-zero.
    assert float(gray_diffs[8:13].max()) < 0.5
    assert float(gray_diffs[15:].min()) > 0.5
