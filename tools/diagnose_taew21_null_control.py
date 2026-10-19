"""Measure the pairwise instruments on a perceptually-null manipulation.

Amendment v6 null control: for every campaign job, decode the job's own
`full.mp4` (H.264 videotoolbox 8M — the exact medium study participants
watch) back to raw uint8 frames and measure the three frozen pairwise
instruments between `pixels.full.bin` and its own round-trip. No
tiny-decoder data is read. If this null manipulation violates the v1
temporal-CLIP bound, the v1 instrument is invalid by demonstration; the
per-job values also serve as the v2 screen's floors.
"""

# mypy: ignore-errors

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.manifests import file_sha256  # noqa: E402
from liveframe.quality_metric_runner import (  # noqa: E402
    PerceptualMetricRunner,
    validate_metric_artifact,
)

V1_TEMPORAL_BOUND = 0.35
RESOLUTION = (480, 832)


def _roundtrip(video: Path, destination: Path, frames: int) -> None:
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        str(destination),
    ]
    subprocess.run(command, check=True)
    expected = frames * RESOLUTION[0] * RESOLUTION[1] * 3
    actual = destination.stat().st_size
    if actual != expected:
        raise RuntimeError(
            f"round-trip of {video} produced {actual} bytes, expected {expected}"
        )
    metadata = {
        "schema": "liveframe.decoded-pixels.v1",
        "dtype": "uint8",
        "layout": "T,H,W,C",
        "shape": [frames, RESOLUTION[0], RESOLUTION[1], 3],
        "sha256": file_sha256(destination),
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--metric-device", default="mps")
    parser.add_argument("--metric-batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace null control: {args.output}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
    if (
        amendment.get("schema") != "liveframe.taew21-protocol-amendment.v6"
        or amendment["bindings"]["protocol_sha256"] != file_sha256(args.protocol)
        or amendment["bindings"]["campaign_summary_sha256"] != file_sha256(args.summary)
    ):
        raise ValueError("null control requires the bound amendment v6")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    summary_jobs = {row["job_id"]: row for row in summary["jobs"]}
    metric_spec = protocol["metrics"]
    provenance = validate_metric_artifact(
        {
            "metric_provenance": {
                "artifact": metric_spec["provenance_artifact"],
                "artifact_sha256": metric_spec["provenance_sha256"],
            }
        },
        REPOSITORY_ROOT,
    )
    runner = PerceptualMetricRunner(
        provenance, device=args.metric_device, batch_size=args.metric_batch_size
    )

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT / "outputs") as temporary:
        scratch = Path(temporary)
        for job in protocol["campaign"]["jobs"]:
            job_id = str(job["job_id"])
            horizon = int(job["horizon_seconds"])
            frames = int(protocol["campaign"]["horizons"][str(horizon)]["pixel_frames"])
            job_dir = REPOSITORY_ROOT / str(job["output"])
            full_path = job_dir / "pixels.full.bin"
            if file_sha256(full_path) != summary_jobs[job_id]["full_pixels_sha256"]:
                raise ValueError(f"job {job_id} full pixel digest mismatch")
            roundtrip_path = scratch / f"{job_id}.roundtrip.bin"
            _roundtrip(job_dir / "full.mp4", roundtrip_path, frames)
            print(f"null control {job_id}", flush=True)
            pairwise = runner.measure_pairwise(
                full_path, roundtrip_path, expected_frames=frames
            )
            records.append(
                {
                    "job_id": job_id,
                    "stratum": job["stratum"],
                    "horizon_seconds": horizon,
                    "prompt_id": job["prompt_id"],
                    "null_pairwise": pairwise,
                    "violates_v1_temporal_bound": pairwise[
                        "pairwise_temporal_clip_distance_mean"
                    ]
                    > V1_TEMPORAL_BOUND,
                }
            )
            roundtrip_path.unlink()

    temporal = [
        record["null_pairwise"]["pairwise_temporal_clip_distance_mean"]
        for record in records
    ]
    violations = sum(record["violates_v1_temporal_bound"] for record in records)
    result = {
        "schema": "liveframe.taew21-null-control.v1",
        "amendment_sha256": file_sha256(args.amendment),
        "protocol_sha256": file_sha256(args.protocol),
        "summary_sha256": file_sha256(args.summary),
        "tool_sha256": file_sha256(Path(__file__).resolve()),
        "manipulation": "h264-videotoolbox-8M-roundtrip-of-full-decoder-video",
        "records": records,
        "temporal_summary": {
            "v1_bound": V1_TEMPORAL_BOUND,
            "violations": int(violations),
            "jobs": len(records),
            "min": float(np.min(temporal)),
            "median": float(np.median(temporal)),
            "max": float(np.max(temporal)),
        },
        "v1_temporal_instrument_invalidated": bool(violations > 0),
    }
    temporary_output = args.output.with_name(f".{args.output.name}.tmp")
    temporary_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_output, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "temporal_violations": int(violations),
                "temporal_median": float(np.median(temporal)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
