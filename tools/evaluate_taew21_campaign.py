"""Evaluate the frozen TAEW2.1 campaign metric screen.

Applies the protocol's frozen metric rules to the 32 paired captures:

- Intrinsic metrics (semantic, motion, temporal, flicker) are computed for
  every full and tiny video with the digest-pinned metric stack. For each
  (stratum, horizon) cell the FOUR full-decoder videos define each metric's
  center and robust scale (median plus scaled MAD, the pre-registered
  `median-plus-scaled-MAD.v1` derivation reused from the frozen holdout
  policy); every tiny video must fall inside the frozen sigma band
  (band_sigma + margin_sigma, scale floored at scale_epsilon, direction per
  spec). The full decoder is the reference implementation, so its own outputs
  define the band centers.
- Identity tripwires (pairwise LPIPS, CLIP distance, temporal-CLIP distance
  means) are computed per pair and must sit below the frozen uppers.
- Decision rule (frozen): all intrinsic bands AND all identity tripwires must
  pass. Screening authority only: a PASS permits the human study; it is not a
  human-noticeability claim.
- Pre-declared diagnostic: a per-frame pairwise LPIPS curve per job, recorded
  to expose local temporal failures hidden by means. Diagnostics carry no
  gate authority.

This evaluator is frozen before any campaign metric is visible.
"""

# mypy: ignore-errors

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.manifests import file_sha256  # noqa: E402
from liveframe.quality_calibration import raw_location_scale  # noqa: E402
from liveframe.quality_metric_runner import (  # noqa: E402
    PerceptualMetricRunner,
    load_pixels,
    validate_metric_artifact,
)

HORIZON_LATENTS = {15: 60, 60: 243}


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _lpips_curve(
    runner: PerceptualMetricRunner, reference_path: Path, candidate_path: Path
) -> list[float]:
    import torch

    reference, _ = load_pixels(reference_path)
    candidate, _ = load_pixels(candidate_path)
    values: list[float] = []
    with torch.inference_mode():
        for start in range(0, reference.shape[0], runner.batch_size):
            end = min(start + runner.batch_size, reference.shape[0])
            values.extend(
                runner.lpips_model(
                    runner._frames(reference[start:end]),
                    runner._frames(candidate[start:end]),
                )
                .reshape(-1)
                .detach()
                .cpu()
                .tolist()
            )
    return values


def _band_rule(spec: dict[str, Any], values: list[float]) -> dict[str, Any]:
    center, raw_scale = raw_location_scale(values)
    scale = max(raw_scale, float(spec["scale_epsilon"]))
    radius = (float(spec["band_sigma"]) + float(spec["margin_sigma"])) * scale
    rule: dict[str, Any] = {
        "direction": spec["direction"],
        "full_center": center,
        "full_robust_scale": scale,
        "raw_full_robust_scale": raw_scale,
        "calibration_videos": len(values),
        "derivation": "median-plus-scaled-MAD.v1-full-decoder-reference",
    }
    if spec["direction"] in {"higher", "two-sided"}:
        rule["lower"] = center - radius
    if spec["direction"] in {"lower", "two-sided"}:
        rule["upper"] = center + radius
    return rule


def _band_pass(rule: dict[str, Any], value: float) -> bool:
    if "lower" in rule and value < rule["lower"]:
        return False
    return not ("upper" in rule and value > rule["upper"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--metric-device", default="mps")
    parser.add_argument("--metric-batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace metric verdict: {args.output}")

    protocol = _object(args.protocol)
    authorization = _object(args.authorization)
    if (
        authorization.get("schema") != "liveframe.taew21-campaign-authorization.v1"
        or not authorization.get("passed")
        or authorization.get("protocol_sha256") != file_sha256(args.protocol)
    ):
        raise ValueError("metric evaluation requires a passed authorization record")
    summary = _object(args.summary)
    if (
        summary.get("schema") != "liveframe.taew21-campaign-summary.v1"
        or summary.get("protocol_sha256") != file_sha256(args.protocol)
        or summary.get("authorization_sha256") != file_sha256(args.authorization)
    ):
        raise ValueError("metric evaluation requires a bound campaign summary")
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
    jobs = {job["job_id"]: job for job in protocol["campaign"]["jobs"]}
    summary_jobs = {row["job_id"]: row for row in summary["jobs"]}
    if set(jobs) != set(summary_jobs):
        raise ValueError("campaign summary does not cover the frozen job list")

    runner = PerceptualMetricRunner(
        provenance, device=args.metric_device, batch_size=args.metric_batch_size
    )

    records: list[dict[str, Any]] = []
    for job_id, job in jobs.items():
        horizon = int(job["horizon_seconds"])
        pixel_frames = int(
            protocol["campaign"]["horizons"][str(horizon)]["pixel_frames"]
        )
        prompt = protocol["campaign"]["prompt_catalog"][str(job["prompt_id"])]
        job_dir = REPOSITORY_ROOT / str(job["output"])
        result = _object(job_dir / "result.json")
        if not result.get("completed"):
            raise ValueError(f"job {job_id} result is incomplete")
        full_path = job_dir / "pixels.full.bin"
        tiny_path = job_dir / "pixels.tiny.bin"
        for which, path in (("full", full_path), ("tiny", tiny_path)):
            expected = summary_jobs[job_id][f"{which}_pixels_sha256"]
            if file_sha256(path) != expected:
                raise ValueError(f"job {job_id} {which} pixel digest mismatch")
        print(f"measuring {job_id}", flush=True)
        intrinsic_full = runner.measure(
            full_path, prompt=prompt, expected_frames=pixel_frames
        )
        intrinsic_tiny = runner.measure(
            tiny_path, prompt=prompt, expected_frames=pixel_frames
        )
        pairwise = runner.measure_pairwise(
            full_path, tiny_path, expected_frames=pixel_frames
        )
        curve = _lpips_curve(runner, full_path, tiny_path)
        tripwires = {
            name: {
                "value": pairwise[name],
                "upper": float(spec["upper"]),
                "passed": pairwise[name] <= float(spec["upper"]),
            }
            for name, spec in metric_spec["identity_tripwires"].items()
        }
        records.append(
            {
                "job_id": job_id,
                "stratum": job["stratum"],
                "horizon_seconds": horizon,
                "prompt_id": job["prompt_id"],
                "intrinsic_full": intrinsic_full,
                "intrinsic_tiny": intrinsic_tiny,
                "pairwise": pairwise,
                "tripwires": tripwires,
                "tripwires_passed": all(row["passed"] for row in tripwires.values()),
                "lpips_curve": {
                    "values": curve,
                    "max": float(np.max(curve)),
                    "argmax_frame": int(np.argmax(curve)),
                    "p95": float(np.quantile(curve, 0.95)),
                    "authority": "diagnostic-only",
                },
            }
        )

    cells: dict[str, dict[str, Any]] = {}
    for stratum in sorted({record["stratum"] for record in records}):
        for horizon in (15, 60):
            cell_records = [
                record
                for record in records
                if record["stratum"] == stratum
                and record["horizon_seconds"] == horizon
            ]
            if len(cell_records) != 4:
                raise ValueError(
                    f"cell {stratum}/{horizon}s must contain 4 jobs, "
                    f"got {len(cell_records)}"
                )
            cell_rules: dict[str, Any] = {}
            cell_checks: list[dict[str, Any]] = []
            for metric, spec in metric_spec["intrinsic_metrics"].items():
                rule = _band_rule(
                    spec,
                    [record["intrinsic_full"][metric] for record in cell_records],
                )
                cell_rules[metric] = rule
                for record in cell_records:
                    value = float(record["intrinsic_tiny"][metric])
                    cell_checks.append(
                        {
                            "job_id": record["job_id"],
                            "metric": metric,
                            "value": value,
                            "passed": _band_pass(rule, value),
                        }
                    )
            cells[f"{stratum}/{horizon}s"] = {
                "stratum": stratum,
                "horizon_seconds": horizon,
                "rules": cell_rules,
                "checks": cell_checks,
                "passed": all(check["passed"] for check in cell_checks),
            }

    intrinsic_passed = all(cell["passed"] for cell in cells.values())
    tripwires_passed = all(record["tripwires_passed"] for record in records)
    passed = intrinsic_passed and tripwires_passed
    verdict = {
        "schema": "liveframe.taew21-metric-verdict.v1",
        "authority": metric_spec["authority"],
        "decision_rule": metric_spec["decision_rule"],
        "protocol_sha256": file_sha256(args.protocol),
        "authorization_sha256": file_sha256(args.authorization),
        "summary_sha256": file_sha256(args.summary),
        "metric_provenance_sha256": metric_spec["provenance_sha256"],
        "evaluator_sha256": file_sha256(Path(__file__).resolve()),
        "job_records": records,
        "cells": cells,
        "intrinsic_passed": intrinsic_passed,
        "tripwires_passed": tripwires_passed,
        "passed": passed,
        "verdict": (
            "metric-screen-PASS-human-study-permitted"
            if passed
            else "metric-screen-FAIL-release-tier-stopped"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "intrinsic_passed": intrinsic_passed,
                "tripwires_passed": tripwires_passed,
                "passed": passed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
