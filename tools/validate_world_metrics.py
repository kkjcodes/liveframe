"""Full metric validation: control grid, onset, defects, throughput (spec 8.5).

Writes the machine-readable results next to the human report inputs. CI
reruns a smaller version (tests/test_world_metrics.py); this is the full
grid the gate report cites.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.world.actions import ActionScript  # noqa: E402
from liveframe.world.adapters.synthetic import FakeClock, SyntheticWorld  # noqa: E402
from liveframe.world.metrics.runner import run_metric_pass  # noqa: E402
from liveframe.world.metrics.validation import (  # noqa: E402
    calibration_sweep,
    validate_control,
    validate_onset,
    validate_stability_defects,
)
from liveframe.world.session import WorldCase, run_case, write_case_directory  # noqa: E402
from liveframe.world.sinks import FrameCapture  # noqa: E402

# M4 gate (Kumar correction 2026-10-14, replaces the projected %-of-generation
# form, which fails for fast models): the metric pass must sustain at least
# this throughput on the pod CPU (all cores), measured live. Any host can run
# this tool; the gate only counts when measured_on_pod is set.
METRIC_THROUGHPUT_GATE_FPS = 40.0


def measure_metric_throughput(tmp_root: Path) -> dict[str, float]:
    """Live metric-pass throughput on a quick-suite-scale capture at MG2 geometry."""

    frames = 240  # 20 s at 12 native fps, the quick-suite per-case length
    clock = FakeClock()
    world = SyntheticWorld(
        resolution=(176, 320),  # MG2 352x640 after the capture downscale (k=2)
        chunk_frames=4,
        native_fps=12.0,
        chunk_delay_seconds=0.001,
        clock=clock,
    )
    world.load()
    case = WorldCase(
        case_id="cost-probe",
        start_image=np.zeros((176, 320, 3), dtype=np.uint8),
        script=ActionScript.from_segments(
            [(0, "W", 80), (80, "L", 80), (160, "0", 80)]
        ),
        seed=3,
    )
    case_dir = tmp_root / "cases" / case.case_id
    capture = FrameCapture(case_dir / "frames_256")
    result = run_case(world, case, clock=clock, capture=capture)
    assert result.status == "ok", result.error
    write_case_directory(result, case_dir)
    case_metrics = run_metric_pass(case_dir, native_fps=12.0)
    # Same definition run_suite reports per case: analyzed frames over the
    # measured metric-pass wall.
    return {
        "frames": frames,
        "frames_analyzed": case_metrics["frames_analyzed"],
        "metric_pass_seconds": case_metrics["metric_pass_seconds"],
        "throughput_fps": (
            case_metrics["frames_analyzed"] / case_metrics["metric_pass_seconds"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tmp", type=Path, required=True)
    parser.add_argument(
        "--measured-on-pod",
        action="store_true",
        help=(
            "assert this run is on the reference pod CPU; only then does the "
            "throughput measurement count as the M4 gate"
        ),
    )
    args = parser.parse_args()
    args.tmp.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema": "liveframe.world.metric-validation.v2",
        "calibration_sweep": calibration_sweep(),
        "control": validate_control(),
        "onset": validate_onset(),
        "stability": validate_stability_defects(),
        "throughput": measure_metric_throughput(args.tmp),
        "measured_on_pod": args.measured_on_pod,
        "cpu_count": os.cpu_count(),
        "host": platform.platform(),
    }
    report["wall_seconds"] = time.perf_counter() - started
    throughput_ok = report["throughput"]["throughput_fps"] >= METRIC_THROUGHPUT_GATE_FPS
    report["gate"] = {
        "control_accuracy_ge_95": report["control"]["accuracy"] >= 0.95,
        "onset_within_one_frame": report["onset"]["all_within_one"],
        "defects_detected_clean_zero": report["stability"]["all_passed"],
        # M4 restated gate (Kumar): throughput >= 40 frames/s on the pod CPU,
        # measured live. Off-pod runs report the number but the gate stays
        # unmet until measured on the pod.
        "metric_throughput_ge_40fps_on_pod": throughput_ok and args.measured_on_pod,
        "metric_throughput_ge_40fps_this_host": throughput_ok,
    }
    report["gate"]["all"] = all(
        value
        for key, value in report["gate"].items()
        if key != "metric_throughput_ge_40fps_this_host"
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gate": report["gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
