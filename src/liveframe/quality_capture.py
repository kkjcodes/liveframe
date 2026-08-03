"""Strict planning, ingestion, and sanity checks for quality captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from liveframe.perceptual_policy import canonical_digest
from liveframe.quality_calibration import (
    CALIBRATION_SCHEMA,
    PROTOCOL_SCHEMA,
    load_protocol,
    metric_is_authoritative,
    missing_coverage_categories,
)
from liveframe.quality_manifest import RUN_MANIFEST_SCHEMA, quality_run_id

BATCH_SCHEMA = "liveframe.quality-capture-batch.v1"
RESULT_SCHEMA = "liveframe.quality-video-result.v1"
SANITY_SCHEMA = "liveframe.quality-calibration-sanity.v1"
IDENTITY_FIELDS = (
    "split",
    "role",
    "backend",
    "stratum",
    "prompt_id",
    "seed",
    "horizon_seconds",
    "latent_frames",
    "pixel_frames",
    "playback_seconds",
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return raw


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _identity(job: Mapping[str, Any]) -> dict[str, Any]:
    return {name: job[name] for name in IDENTITY_FIELDS}


def load_run_manifest(path: Path, *, protocol_digest: str) -> dict[str, Any]:
    manifest = _read_object(path)
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        raise ValueError("unsupported quality run manifest")
    if manifest.get("protocol_digest") != protocol_digest:
        raise ValueError("run manifest is bound to another protocol")
    recorded_digest = manifest.get("manifest_digest")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if recorded_digest != canonical_digest(unsigned):
        raise ValueError("quality run manifest digest mismatch")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("quality run manifest requires jobs")
    seen: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("quality manifest jobs must be objects")
        expected = quality_run_id(protocol_digest, _identity(job))
        if job.get("run_id") != expected:
            raise ValueError(f"quality run ID mismatch: {job.get('run_id')}")
        if expected in seen:
            raise ValueError(f"duplicate quality run ID: {expected}")
        seen.add(expected)
    counts = manifest.get("job_counts")
    if not isinstance(counts, dict):
        raise ValueError("quality run manifest requires job counts")
    calibration = sum(job["split"] == "calibration" for job in jobs)
    holdout = sum(job["split"] == "holdout" for job in jobs)
    if counts != {"calibration": calibration, "holdout": holdout, "total": len(jobs)}:
        raise ValueError("quality run manifest job counts mismatch")
    return manifest


def calibration_jobs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = [
        dict(job)
        for job in manifest["jobs"]
        if job.get("split") == "calibration"
    ]
    for job in jobs:
        if job.get("role") != "teacher" or job.get("backend") != "cuda":
            raise ValueError(f"invalid calibration job role/backend: {job['run_id']}")
    return jobs


def holdout_candidate_jobs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = [
        dict(job)
        for job in manifest["jobs"]
        if job.get("split") == "holdout" and job.get("role") == "candidate"
    ]
    for job in jobs:
        if job.get("backend") != "mlx":
            raise ValueError(f"invalid holdout candidate backend: {job['run_id']}")
    if not jobs:
        raise ValueError("quality manifest contains no MLX holdout candidates")
    return jobs


def holdout_reference_jobs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = [
        dict(job)
        for job in manifest["jobs"]
        if job.get("split") == "holdout" and job.get("role") == "reference"
    ]
    for job in jobs:
        if job.get("backend") != "cuda":
            raise ValueError(f"invalid holdout reference backend: {job['run_id']}")
    if not jobs:
        raise ValueError("quality manifest contains no CUDA holdout references")
    return jobs


def load_batch_plan(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    batch = _read_object(path)
    if batch.get("schema") != BATCH_SCHEMA:
        raise ValueError("unsupported quality capture batch")
    recorded_digest = batch.get("batch_digest")
    unsigned = {key: value for key, value in batch.items() if key != "batch_digest"}
    if recorded_digest != canonical_digest(unsigned):
        raise ValueError("quality capture batch digest mismatch")
    if batch.get("protocol_digest") != manifest.get("protocol_digest"):
        raise ValueError("quality capture batch protocol mismatch")
    if batch.get("manifest_digest") != manifest.get("manifest_digest"):
        raise ValueError("quality capture batch manifest mismatch")
    expected = {job["run_id"]: job for job in calibration_jobs(manifest)}
    jobs = batch.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("quality capture batch has no jobs")
    seen: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict) or job.get("run_id") not in expected:
            raise ValueError("quality capture batch contains an unknown job")
        run_id = str(job["run_id"])
        if run_id in seen:
            raise ValueError(f"quality capture batch duplicates job: {run_id}")
        if job != expected[run_id]:
            raise ValueError(f"quality capture batch altered job: {run_id}")
        seen.add(run_id)
    return batch


def load_holdout_candidate_batch_plan(
    path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    batch = _read_object(path)
    if batch.get("schema") != BATCH_SCHEMA:
        raise ValueError("unsupported quality capture batch")
    recorded_digest = batch.get("batch_digest")
    unsigned = {key: value for key, value in batch.items() if key != "batch_digest"}
    if recorded_digest != canonical_digest(unsigned):
        raise ValueError("quality capture batch digest mismatch")
    if batch.get("protocol_digest") != manifest.get("protocol_digest"):
        raise ValueError("quality capture batch protocol mismatch")
    if batch.get("manifest_digest") != manifest.get("manifest_digest"):
        raise ValueError("quality capture batch manifest mismatch")
    expected = {job["run_id"]: job for job in holdout_candidate_jobs(manifest)}
    jobs = batch.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("quality capture batch has no jobs")
    seen: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict) or job.get("run_id") not in expected:
            raise ValueError("holdout candidate batch contains an unknown job")
        run_id = str(job["run_id"])
        if run_id in seen:
            raise ValueError(f"holdout candidate batch duplicates job: {run_id}")
        if job != expected[run_id]:
            raise ValueError(f"holdout candidate batch altered job: {run_id}")
        seen.add(run_id)
    return batch


def load_holdout_reference_batch_plan(
    path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    batch = _read_object(path)
    if batch.get("schema") != BATCH_SCHEMA:
        raise ValueError("unsupported quality capture batch")
    recorded_digest = batch.get("batch_digest")
    unsigned = {key: value for key, value in batch.items() if key != "batch_digest"}
    if recorded_digest != canonical_digest(unsigned):
        raise ValueError("quality capture batch digest mismatch")
    if batch.get("protocol_digest") != manifest.get("protocol_digest"):
        raise ValueError("quality capture batch protocol mismatch")
    if batch.get("manifest_digest") != manifest.get("manifest_digest"):
        raise ValueError("quality capture batch manifest mismatch")
    expected = {job["run_id"]: job for job in holdout_reference_jobs(manifest)}
    jobs = batch.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("quality capture batch has no jobs")
    seen: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict) or job.get("run_id") not in expected:
            raise ValueError("holdout reference batch contains an unknown job")
        run_id = str(job["run_id"])
        if run_id in seen:
            raise ValueError(f"holdout reference batch duplicates job: {run_id}")
        if job != expected[run_id]:
            raise ValueError(f"holdout reference batch altered job: {run_id}")
        seen.add(run_id)
    return batch


def result_path(output_root: Path, run_id: str) -> Path:
    return output_root / run_id / "result.json"


def validate_result(
    result: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    protocol_digest: str,
    manifest_digest: str,
    required_metrics: set[str] | None = None,
    root: Path | None = None,
) -> None:
    if result.get("schema") != RESULT_SCHEMA or result.get("status") != "completed":
        raise ValueError(f"quality result is not completed: {job['run_id']}")
    if result.get("protocol_digest") != protocol_digest:
        raise ValueError(f"quality result protocol mismatch: {job['run_id']}")
    if result.get("manifest_digest") != manifest_digest:
        raise ValueError(f"quality result manifest mismatch: {job['run_id']}")
    if result.get("run_id") != job["run_id"]:
        raise ValueError(f"quality result run ID mismatch: {job['run_id']}")
    for name in (*IDENTITY_FIELDS, "prompt"):
        if result.get(name) != job.get(name):
            raise ValueError(f"quality result {name} mismatch: {job['run_id']}")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"quality result requires artifacts: {job['run_id']}")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ValueError(f"invalid quality artifact entry: {job['run_id']}")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ValueError(f"invalid quality artifact digest: {job['run_id']}")
        if root is not None:
            artifact_path = root / artifact["path"]
            if not artifact_path.is_file() or sha256_file(artifact_path) != digest:
                raise ValueError(f"quality artifact checksum mismatch: {artifact_path}")
    metrics = result.get("metrics")
    if required_metrics is not None:
        if not isinstance(metrics, dict) or set(metrics) != required_metrics:
            raise ValueError(f"quality result metric set mismatch: {job['run_id']}")
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"quality result metric {name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"quality result metric {name} must be finite")


def prepare_calibration_batch(
    manifest: Mapping[str, Any],
    output_root: Path,
    *,
    run_ids: set[str] | None = None,
    limit: int | None = None,
    required_metrics: set[str] | None = None,
    include_completed: bool = False,
) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("batch limit must be positive")
    jobs = calibration_jobs(manifest)
    known = {job["run_id"] for job in jobs}
    if run_ids is not None:
        unknown = run_ids - known
        if unknown:
            raise ValueError(f"requested unknown calibration run IDs: {sorted(unknown)}")
        jobs = [job for job in jobs if job["run_id"] in run_ids]
    pending: list[dict[str, Any]] = []
    for job in jobs:
        if include_completed:
            pending.append(job)
            continue
        path = result_path(output_root, job["run_id"])
        if not path.exists():
            pending.append(job)
            continue
        validate_result(
            _read_object(path),
            job,
            protocol_digest=str(manifest["protocol_digest"]),
            manifest_digest=str(manifest["manifest_digest"]),
            required_metrics=required_metrics,
            root=path.parent,
        )
    if limit is not None:
        pending = pending[:limit]
    batch: dict[str, Any] = {
        "schema": BATCH_SCHEMA,
        "protocol_digest": manifest["protocol_digest"],
        "manifest_digest": manifest["manifest_digest"],
        "output_root": str(output_root),
        "jobs": pending,
    }
    batch["batch_digest"] = canonical_digest(batch)
    return batch


def prepare_holdout_candidate_batch(
    manifest: Mapping[str, Any],
    output_root: Path,
    *,
    run_ids: set[str] | None = None,
    limit: int | None = None,
    required_metrics: set[str] | None = None,
    include_completed: bool = False,
) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("batch limit must be positive")
    jobs = holdout_candidate_jobs(manifest)
    known = {job["run_id"] for job in jobs}
    if run_ids is not None:
        unknown = run_ids - known
        if unknown:
            raise ValueError(f"requested unknown candidate run IDs: {sorted(unknown)}")
        jobs = [job for job in jobs if job["run_id"] in run_ids]
    pending: list[dict[str, Any]] = []
    for job in jobs:
        if include_completed:
            pending.append(job)
            continue
        path = result_path(output_root, job["run_id"])
        if not path.exists():
            pending.append(job)
            continue
        validate_result(
            _read_object(path),
            job,
            protocol_digest=str(manifest["protocol_digest"]),
            manifest_digest=str(manifest["manifest_digest"]),
            required_metrics=required_metrics,
            root=path.parent,
        )
    if limit is not None:
        pending = pending[:limit]
    batch: dict[str, Any] = {
        "schema": BATCH_SCHEMA,
        "protocol_digest": manifest["protocol_digest"],
        "manifest_digest": manifest["manifest_digest"],
        "output_root": str(output_root),
        "jobs": pending,
    }
    batch["batch_digest"] = canonical_digest(batch)
    return batch


def prepare_holdout_reference_batch(
    manifest: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    jobs = holdout_reference_jobs(manifest)
    batch: dict[str, Any] = {
        "schema": BATCH_SCHEMA,
        "protocol_digest": manifest["protocol_digest"],
        "manifest_digest": manifest["manifest_digest"],
        "output_root": str(output_root),
        "jobs": jobs,
    }
    batch["batch_digest"] = canonical_digest(batch)
    return batch


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    if path.exists():
        existing = _read_object(path)
        if existing == result:
            return
        raise FileExistsError(f"refusing to replace quality result: {path}")
    _atomic_json(path, result)


def load_results(paths: Iterable[Path]) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, _read_object(path)) for path in sorted(paths)]


def ingest_calibration(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    results: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    expected_jobs = {job["run_id"]: job for job in calibration_jobs(manifest)}
    required_metrics = set(protocol["quality_metrics"])
    observed: dict[str, Mapping[str, Any]] = {}
    for path, result in results:
        run_id = str(result.get("run_id", ""))
        if run_id not in expected_jobs:
            raise ValueError(f"unexpected calibration result: {run_id or path}")
        if run_id in observed:
            raise ValueError(f"duplicate calibration result: {run_id}")
        validate_result(
            result,
            expected_jobs[run_id],
            protocol_digest=manifest["protocol_digest"],
            manifest_digest=manifest["manifest_digest"],
            required_metrics=required_metrics,
            root=path.parent,
        )
        observed[run_id] = result
    missing = sorted(set(expected_jobs) - set(observed))
    if require_complete and missing:
        raise ValueError(f"calibration results incomplete: {len(missing)} missing")
    videos = [
        {
            "run_id": run_id,
            "stratum": result["stratum"],
            "prompt_id": result["prompt_id"],
            "seed_id": str(result["seed"]),
            "horizon_seconds": result["horizon_seconds"],
            "metrics": result["metrics"],
            "result_digest": canonical_digest(result),
        }
        for run_id, result in sorted(observed.items())
    ]
    return {
        "schema": CALIBRATION_SCHEMA,
        "protocol_digest": manifest["protocol_digest"],
        "run_manifest_digest": manifest["manifest_digest"],
        "complete": not missing,
        "missing_run_ids": missing,
        "teacher_videos": videos,
    }


def _raw_center_scale(values: Sequence[float]) -> tuple[float, float]:
    center = float(statistics.median(values))
    deviations = [abs(value - center) for value in values]
    return center, 1.4826 * float(statistics.median(deviations))


def calibration_sanity_report(
    protocol: Mapping[str, Any], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    if calibration.get("schema") != CALIBRATION_SCHEMA or not calibration.get("complete"):
        raise ValueError("sanity evaluation requires complete calibration input")
    videos = calibration.get("teacher_videos")
    if not isinstance(videos, list):
        raise ValueError("calibration input requires teacher videos")
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    warnings: list[str] = []
    physical_bounds = {
        "clip_prompt_mean": (-1.0, 1.0),
        "clip_prompt_p10": (-1.0, 1.0),
        "motion_clip_distance_mean": (0.0, 2.0),
        "motion_lpips_mean": (0.0, math.inf),
        "temporal_delta_cosine_mean": (-1.0, 1.0),
        "luma_flicker_mean": (0.0, math.inf),
    }
    for stratum in protocol["strata"]:
        for prompt_id in stratum["prompt_ids"]:
            for horizon in protocol["horizons_seconds"]:
                matching = [
                    row
                    for row in videos
                    if row["stratum"] == stratum["id"]
                    and row["prompt_id"] == prompt_id
                    and row["horizon_seconds"] == horizon
                ]
                for metric, spec in protocol["quality_metrics"].items():
                    values = [float(row["metrics"][metric]) for row in matching]
                    center, raw_scale = _raw_center_scale(values)
                    epsilon = float(spec["scale_epsilon"])
                    epsilon_dominated = raw_scale <= epsilon
                    authoritative = metric_is_authoritative(protocol, metric, raw_scale)
                    lower, upper = physical_bounds[metric]
                    bounds_valid = all(lower <= value <= upper for value in values)
                    key = f"{stratum['id']}/{prompt_id}/{horizon}s/{metric}"
                    if not bounds_valid:
                        failures.append(f"physical metric bounds failed: {key}")
                    if epsilon_dominated:
                        warnings.append(f"teacher spread is epsilon-dominated: {key}")
                    if (
                        protocol.get("schema") == PROTOCOL_SCHEMA
                        and
                        stratum["id"] == "high-motion-dense-scenes"
                        and metric in {"motion_clip_distance_mean", "motion_lpips_mean"}
                        and epsilon_dominated
                    ):
                        failures.append(f"high-motion teacher spread is degenerate: {key}")
                    rows.append(
                        {
                            "stratum": stratum["id"],
                            "prompt_id": prompt_id,
                            "horizon_seconds": horizon,
                            "metric": metric,
                            "videos": len(values),
                            "teacher_center": center,
                            "raw_teacher_robust_scale": raw_scale,
                            "scale_epsilon": epsilon,
                            "epsilon_dominated": epsilon_dominated,
                            "authoritative": authoritative,
                            "physical_bounds_valid": bounds_valid,
                        }
                    )
                cell_rows = [
                    row
                    for row in rows
                    if row["stratum"] == stratum["id"]
                    and row["prompt_id"] == prompt_id
                    and row["horizon_seconds"] == horizon
                ]
                missing_coverage = missing_coverage_categories(
                    protocol,
                    {
                        str(row["metric"])
                        for row in cell_rows
                        if row["authoritative"]
                    },
                )
                if missing_coverage:
                    failures.append(
                        f"authoritative metric coverage failed: {stratum['id']}/"
                        f"{prompt_id}/{horizon}s missing={missing_coverage}"
                    )
    return {
        "schema": SANITY_SCHEMA,
        "protocol_digest": canonical_digest(protocol),
        "source_protocol_digest": calibration["protocol_digest"],
        "calibration_input_digest": canonical_digest(calibration),
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "cells": rows,
    }


def _result_files(root: Path) -> list[Path]:
    return sorted(root.glob("quality-*/result.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--run-id", action="append")
    prepare.add_argument("--limit", type=int)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--protocol", type=Path, required=True)
    ingest.add_argument("--manifest", type=Path, required=True)
    ingest.add_argument("--result-root", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    sanity = commands.add_parser("sanity")
    sanity.add_argument("--protocol", type=Path, required=True)
    sanity.add_argument("--calibration", type=Path, required=True)
    sanity.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol, protocol_digest = load_protocol(args.protocol)
    if args.command == "prepare":
        manifest = load_run_manifest(args.manifest, protocol_digest=protocol_digest)
        batch = prepare_calibration_batch(
            manifest,
            args.output_root,
            run_ids=set(args.run_id) if args.run_id else None,
            limit=args.limit,
            required_metrics=set(protocol["quality_metrics"]),
        )
        _atomic_json(args.output, batch)
        print(json.dumps({"output": str(args.output), "jobs": len(batch["jobs"])}))
        return
    if args.command == "ingest":
        manifest = load_run_manifest(args.manifest, protocol_digest=protocol_digest)
        calibration = ingest_calibration(
            protocol, manifest, load_results(_result_files(args.result_root))
        )
        _atomic_json(args.output, calibration)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "videos": len(calibration["teacher_videos"]),
                }
            )
        )
        return
    calibration = _read_object(args.calibration)
    report = calibration_sanity_report(protocol, calibration)
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}))


if __name__ == "__main__":
    main()
