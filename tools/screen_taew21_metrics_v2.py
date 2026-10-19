"""Run the amendment-v6 scale-referenced metric screen (routing authority only).

For each pair and each frozen pairwise instrument, the tiny-vs-full distance
(read digest-bound from the v1 verdict — same pinned metric stack, identical
numbers) is placed between two anchors computed from full-decoder data only:

- floor: the same job's null-control value (full vs its own H.264 round-trip,
  a perceptually-null manipulation identically present in both study arms);
- ceiling: the cross-content value (the job's full video vs the full video of
  the NEXT prompt in the same stratum and horizon, deterministic rotation).

Frozen rule: normalized position (d - floor) / (ceiling - floor) <= 0.5 for
all three instruments on all 32 pairs. This screen ROUTES to the human study;
it is not a quality claim (v1 values were visible when the rule was frozen —
see the amendment's honesty disclosure).
"""

# mypy: ignore-errors

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.manifests import file_sha256  # noqa: E402
from liveframe.quality_metric_runner import (  # noqa: E402
    PerceptualMetricRunner,
    validate_metric_artifact,
)

INSTRUMENTS = (
    "pairwise_lpips_mean",
    "pairwise_clip_distance_mean",
    "pairwise_temporal_clip_distance_mean",
)
RULE_THRESHOLD = 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--v1-verdict", type=Path, required=True)
    parser.add_argument("--null-control", type=Path, required=True)
    parser.add_argument("--metric-device", default="mps")
    parser.add_argument("--metric-batch-size", type=int, required=False, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace v2 screen: {args.output}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
    if (
        amendment.get("schema") != "liveframe.taew21-protocol-amendment.v6"
        or amendment["bindings"]["protocol_sha256"] != file_sha256(args.protocol)
        or amendment["bindings"]["campaign_summary_sha256"] != file_sha256(args.summary)
        or amendment["bindings"]["v1_metric_verdict_sha256"]
        != file_sha256(args.v1_verdict)
    ):
        raise ValueError("v2 screen requires the bound amendment v6")
    null_control = json.loads(args.null_control.read_text(encoding="utf-8"))
    if (
        null_control.get("schema") != "liveframe.taew21-null-control.v1"
        or null_control.get("amendment_sha256") != file_sha256(args.amendment)
    ):
        raise ValueError("v2 screen requires the bound null-control artifact")
    verdict = json.loads(args.v1_verdict.read_text(encoding="utf-8"))
    tiny_pairwise = {
        record["job_id"]: record["pairwise"] for record in verdict["job_records"]
    }
    floors = {
        record["job_id"]: record["null_pairwise"]
        for record in null_control["records"]
    }
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

    jobs = protocol["campaign"]["jobs"]
    by_cell: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for job in jobs:
        by_cell.setdefault((job["stratum"], int(job["horizon_seconds"])), []).append(job)
    partner: dict[str, dict[str, Any]] = {}
    for cell_jobs in by_cell.values():
        ordered = sorted(cell_jobs, key=lambda job: str(job["prompt_id"]))
        for index, job in enumerate(ordered):
            partner[str(job["job_id"])] = ordered[(index + 1) % len(ordered)]

    records: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job["job_id"])
        horizon = int(job["horizon_seconds"])
        frames = int(protocol["campaign"]["horizons"][str(horizon)]["pixel_frames"])
        job_dir = REPOSITORY_ROOT / str(job["output"])
        partner_job = partner[job_id]
        partner_dir = REPOSITORY_ROOT / str(partner_job["output"])
        for candidate_id, path in (
            (job_id, job_dir / "pixels.full.bin"),
            (str(partner_job["job_id"]), partner_dir / "pixels.full.bin"),
        ):
            if file_sha256(path) != summary_jobs[candidate_id]["full_pixels_sha256"]:
                raise ValueError(f"job {candidate_id} full pixel digest mismatch")
        print(f"ceiling {job_id} vs {partner_job['job_id']}", flush=True)
        ceiling = runner.measure_pairwise(
            job_dir / "pixels.full.bin",
            partner_dir / "pixels.full.bin",
            expected_frames=frames,
        )
        checks = {}
        for instrument in INSTRUMENTS:
            floor_value = float(floors[job_id][instrument])
            ceiling_value = float(ceiling[instrument])
            tiny_value = float(tiny_pairwise[job_id][instrument])
            if ceiling_value <= floor_value:
                raise RuntimeError(
                    f"job {job_id} {instrument}: ceiling {ceiling_value} does not "
                    f"exceed floor {floor_value}; anchors degenerate"
                )
            position = (tiny_value - floor_value) / (ceiling_value - floor_value)
            checks[instrument] = {
                "tiny_vs_full": tiny_value,
                "floor_null_control": floor_value,
                "ceiling_cross_content": ceiling_value,
                "normalized_position": position,
                "passed": position <= RULE_THRESHOLD,
            }
        records.append(
            {
                "job_id": job_id,
                "stratum": job["stratum"],
                "horizon_seconds": horizon,
                "partner_job_id": partner_job["job_id"],
                "checks": checks,
                "passed": all(check["passed"] for check in checks.values()),
            }
        )

    passed = all(record["passed"] for record in records)
    result = {
        "schema": "liveframe.taew21-metric-screen.v2",
        "authority": "routing-only-to-human-study-not-a-quality-claim",
        "rule": f"normalized position <= {RULE_THRESHOLD} for all instruments on all pairs",
        "amendment_sha256": file_sha256(args.amendment),
        "protocol_sha256": file_sha256(args.protocol),
        "summary_sha256": file_sha256(args.summary),
        "v1_verdict_sha256": file_sha256(args.v1_verdict),
        "null_control_sha256": file_sha256(args.null_control),
        "tool_sha256": file_sha256(Path(__file__).resolve()),
        "records": records,
        "failing_jobs": [record["job_id"] for record in records if not record["passed"]],
        "passed": passed,
        "verdict": (
            "route-to-human-study" if passed else "stop-tier-publish-as-is"
        ),
    }
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": passed,
                "failing_jobs": result["failing_jobs"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
