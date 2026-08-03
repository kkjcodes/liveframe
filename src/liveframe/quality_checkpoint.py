"""Fail-closed first-seed checkpoint for the H100 calibration campaign."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from liveframe.perceptual_policy import canonical_digest
from liveframe.quality_capture import calibration_jobs, validate_result
from liveframe.quality_freeze import require_frozen_identities

CHECKPOINT_SCHEMA = "liveframe.quality-calibration-checkpoint.v1"
EXPECTED_PROJECTED_HOURS = 4.23
EXPECTED_PROJECTED_STORAGE_GIB = 211.0
MAX_PROJECTION_MULTIPLIER = 1.25


def first_seed_run_ids(
    protocol: Mapping[str, Any], manifest: Mapping[str, Any]
) -> set[str]:
    first_seed = int(protocol["sampling"]["calibration_seeds"][0])
    return {
        str(job["run_id"])
        for job in calibration_jobs(manifest)
        if int(job["seed"]) == first_seed
    }


def evaluate_first_seed_checkpoint(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    manifest: Mapping[str, Any],
    results: Sequence[tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Validate all 16 prompts at both horizons before the remaining seed sweep."""

    require_frozen_identities(protocol_digest, manifest)
    expected_ids = first_seed_run_ids(protocol, manifest)
    if len(expected_ids) != 32:
        raise ValueError(f"first-seed checkpoint requires 32 jobs, found {len(expected_ids)}")
    jobs = {str(job["run_id"]): job for job in calibration_jobs(manifest)}
    required_metrics = set(protocol["quality_metrics"])
    observed: set[str] = set()
    artifact_bytes = 0
    executor_seconds = 0.0
    for path, result in results:
        run_id = str(result.get("run_id", ""))
        if run_id not in expected_ids:
            raise ValueError(f"unexpected result in first-seed checkpoint: {run_id or path}")
        if run_id in observed:
            raise ValueError(f"duplicate first-seed checkpoint result: {run_id}")
        validate_result(
            result,
            jobs[run_id],
            protocol_digest=protocol_digest,
            manifest_digest=str(manifest["manifest_digest"]),
            required_metrics=required_metrics,
            root=path.parent,
        )
        pixel_metadata_path = path.parent / "pixels.bin.json"
        pixel_metadata = json.loads(pixel_metadata_path.read_text(encoding="utf-8"))
        expected_shape = [int(result["pixel_frames"]), 480, 832, 3]
        if (
            not isinstance(pixel_metadata, dict)
            or pixel_metadata.get("schema") != "liveframe.decoded-pixels.v1"
            or pixel_metadata.get("dtype") != "uint8"
            or pixel_metadata.get("layout") != "T,H,W,C"
            or pixel_metadata.get("shape") != expected_shape
        ):
            raise ValueError(f"pixel metadata mismatch: {run_id}")
        timing = result.get("timing")
        if not isinstance(timing, dict):
            raise ValueError(f"result timing missing: {run_id}")
        wall = timing.get("executor_wall_seconds")
        if isinstance(wall, bool) or not isinstance(wall, (int, float)):
            raise ValueError(f"executor timing missing: {run_id}")
        wall = float(wall)
        if not math.isfinite(wall) or wall <= 0:
            raise ValueError(f"executor timing invalid: {run_id}")
        executor_seconds += wall
        for artifact in result["artifacts"]:
            declared_bytes = artifact.get("bytes")
            artifact_path = path.parent / str(artifact["path"])
            if (
                isinstance(declared_bytes, bool)
                or not isinstance(declared_bytes, int)
                or declared_bytes != artifact_path.stat().st_size
            ):
                raise ValueError(f"artifact byte count mismatch: {artifact_path}")
            artifact_bytes += declared_bytes
        observed.add(run_id)
    missing = sorted(expected_ids - observed)
    if missing:
        raise ValueError(f"first-seed checkpoint incomplete: {len(missing)} missing")
    total_jobs = len(calibration_jobs(manifest))
    projection_factor = total_jobs / len(expected_ids)
    projected_hours = executor_seconds * projection_factor / 3600.0
    projected_storage_gib = artifact_bytes * projection_factor / (1024**3)
    budget = protocol["budget_ceiling"]
    maximum_hours = min(
        float(budget["h100_hours"]),
        EXPECTED_PROJECTED_HOURS * MAX_PROJECTION_MULTIPLIER,
    )
    maximum_storage_gib = min(
        float(budget["storage_gib"]),
        EXPECTED_PROJECTED_STORAGE_GIB * MAX_PROJECTION_MULTIPLIER,
    )
    failures: list[str] = []
    if projected_hours > maximum_hours:
        failures.append(
            f"projected executor time {projected_hours:.3f}h exceeds {maximum_hours:.3f}h"
        )
    if projected_storage_gib > maximum_storage_gib:
        failures.append(
            "projected artifact storage "
            f"{projected_storage_gib:.3f} GiB exceeds {maximum_storage_gib:.3f} GiB"
        )
    report: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "protocol_digest": protocol_digest,
        "manifest_digest": manifest["manifest_digest"],
        "first_calibration_seed": protocol["sampling"]["calibration_seeds"][0],
        "validated_jobs": len(observed),
        "total_calibration_jobs": total_jobs,
        "projected_executor_hours": projected_hours,
        "maximum_projected_hours": maximum_hours,
        "projected_storage_gib": projected_storage_gib,
        "maximum_projected_storage_gib": maximum_storage_gib,
        "passed": not failures,
        "failures": failures,
    }
    report["checkpoint_digest"] = canonical_digest(report)
    return report
