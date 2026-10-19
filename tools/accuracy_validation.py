"""Accuracy validation — proving the measurement mechanism against ground truth.

Two CPU checks that need no GPU (the GPU-dependent "our FPS vs the model's own
upstream timing" cross-check is captured per model on its pod — HY already done:
harness 0.60 vs direct 0.65).

1. Ground-truth exactness: drive the synthetic backend on an EXACT programmed
   emission schedule (FakeClock) and confirm the harness recovers the true FPS,
   intervals, and latency to the decimal. True values are KNOWN, so any error is
   the harness's.
2. Harness overhead: run the busy synthetic backend on a real clock with frame
   capture ON vs OFF and compare generated FPS — proves the instrumentation
   doesn't inflate the number (target <=2%).
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from liveframe.world.actions import ActionScript
from liveframe.world.adapters.synthetic import FakeClock, SyntheticWorld
from liveframe.world.session import WorldCase, run_case
from liveframe.world.sinks import FrameCapture

RES = (36, 64)
START = np.zeros((*RES, 3), dtype=np.uint8)


def ground_truth_exactness() -> dict:
    """Known schedule -> harness must recover the true numbers exactly."""

    clock = FakeClock()
    # 20 chunks of 0.05 s, one 0.6 s stall at chunk 10: true wall 1.6 s, 80
    # frames @ 50 FPS; first frame at 0.05 s; one stall.
    world = SyntheticWorld(
        resolution=RES, chunk_frames=4, chunk_delay_seconds=0.05,
        stall_extra_seconds={10: 0.6}, clock=clock,
    )
    world.load()
    case = WorldCase(
        case_id="gt", start_image=START,
        script=ActionScript.from_segments([(0, "W", 80)]), seed=7,
    )
    result = run_case(world, case, clock=clock)
    m = result.metrics
    truth = {
        "native_frames": 80, "wall_seconds": 1.6, "generated_fps": 50.0,
        "time_to_first_frame_seconds": 0.05, "stalls_count": 1,
    }
    measured = {
        "native_frames": m["native_frames"], "wall_seconds": m["wall_seconds"],
        "generated_fps": m["generated_fps"],
        "time_to_first_frame_seconds": m["time_to_first_frame_seconds"],
        "stalls_count": m["stalls"]["count"],
    }
    exact = all(
        measured[k] == truth[k] or abs(measured[k] - truth[k]) < 1e-9 for k in truth
    )
    return {"truth": truth, "measured": measured, "exact": exact}


def harness_overhead(trials: int = 5) -> dict:
    """Capture ON vs OFF on a real clock; generated FPS must agree within 2%."""

    def one(capture: bool) -> float:
        world = SyntheticWorld(resolution=RES, chunk_frames=4, busy=True,
                               chunk_delay_seconds=0.002)
        world.load()
        case = WorldCase(
            case_id="ovh", start_image=START,
            script=ActionScript.from_segments([(0, "W", 400)]), seed=1,
        )
        cap = None
        tmp = None
        if capture:
            tmp = tempfile.mkdtemp()
            cap = FrameCapture(Path(tmp) / "frames_256")
        result = run_case(world, case, capture=cap)
        return float(result.metrics["generated_fps"])

    on = statistics.median(one(True) for _ in range(trials))
    off = statistics.median(one(False) for _ in range(trials))
    overhead_pct = 100.0 * (off - on) / off if off else 0.0
    return {
        "generated_fps_capture_on": round(on, 2),
        "generated_fps_capture_off": round(off, 2),
        "overhead_pct": round(overhead_pct, 2),
        "within_2pct": abs(overhead_pct) <= 2.0,
    }


def main() -> None:
    report = {
        "schema": "liveframe.world.accuracy-validation.v1",
        "ground_truth_exactness": ground_truth_exactness(),
        "harness_overhead": harness_overhead(),
        # GPU cross-check (our FPS vs the model's own pipeline on the same box):
        "cross_tool": {
            "hy-worldplay": {
                "harness_fps": 0.60, "direct_pipeline_fps": 0.65,
                "agreement_pct": round(100 * 0.60 / 0.65, 1),
                "note": "same A100, same pipe(); harness adds frame capture + a "
                "slightly longer clip (253 vs 125 frames)",
            },
            "matrix-game-2": {"harness_fps": 11.80, "direct_generation_fps": 12.73,
                "agreement_pct": 92.7,
                "note": "same 4090; harness measures frames DELIVERED to host "
                "(incl. subprocess IPC + capture), ~7% below raw in-process "
                "generation -- conservative, never inflating"},
            "matrix-game-3": {"harness_fps": "1.80-2.10", "direct_generation_fps": 1.93,
                "agreement": "~93-100% (harness brackets direct within MG3's "
                "run-to-run int8/SDPA variance)"},
        },
    }
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("accuracy-validation.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
