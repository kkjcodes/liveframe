"""Fail-closed pairing of frozen CUDA and MLX holdout captures."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from liveframe.perceptual_policy import canonical_digest
from liveframe.quality_calibration import HOLDOUT_SCHEMA
from liveframe.quality_capture import validate_result
from liveframe.quality_stochastic import StochasticPlanReader

HoldoutKey = tuple[str, str, int, int]
PAIRWISE_REPORT_SCHEMA = "liveframe.quality-pairwise-identity.v1"
IDENTITY_METRICS = {
    "pairwise_lpips_mean",
    "pairwise_clip_distance_mean",
    "pairwise_temporal_clip_distance_mean",
}


@dataclass(frozen=True, slots=True)
class HoldoutPair:
    key: HoldoutKey
    reference_job: Mapping[str, Any]
    candidate_job: Mapping[str, Any]
    reference_result: Mapping[str, Any]
    candidate_result: Mapping[str, Any]
    reference_pixels: Path
    candidate_pixels: Path
    stochastic_payload_sha256: str


def _key(row: Mapping[str, Any]) -> HoldoutKey:
    return (
        str(row["stratum"]),
        str(row["prompt_id"]),
        int(row["seed"]),
        int(row["horizon_seconds"]),
    )


def _holdout_jobs(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    jobs: dict[str, Mapping[str, Any]] = {}
    roles: dict[HoldoutKey, set[str]] = {}
    for job in manifest["jobs"]:
        if job.get("split") != "holdout":
            continue
        role = str(job.get("role", ""))
        backend = str(job.get("backend", ""))
        expected_backend = {"reference": "cuda", "candidate": "mlx"}.get(role)
        if backend != expected_backend:
            raise ValueError(
                f"invalid holdout role/backend: {job.get('run_id')} {role}/{backend}"
            )
        run_id = str(job["run_id"])
        if run_id in jobs:
            raise ValueError(f"duplicate holdout run ID: {run_id}")
        jobs[run_id] = job
        roles.setdefault(_key(job), set()).add(role)
    if not jobs:
        raise ValueError("quality manifest contains no holdout jobs")
    incomplete = [key for key, observed in roles.items() if observed != {"reference", "candidate"}]
    if incomplete:
        raise ValueError(f"holdout manifest has incomplete pairs: {len(incomplete)}")
    return jobs


def _pixels_path(result_path: Path, result: Mapping[str, Any]) -> Path:
    artifacts = result["artifacts"]
    matches = [row for row in artifacts if row.get("path") == "pixels.bin"]
    if len(matches) != 1:
        raise ValueError(f"holdout result requires exactly one pixels.bin: {result_path}")
    return result_path.parent / "pixels.bin"


def _stochastic_metadata_path(
    result_path: Path, result: Mapping[str, Any]
) -> Path:
    artifacts = result["artifacts"]
    required = ("stochastic.bf16", "stochastic.bf16.json")
    if any(sum(row.get("path") == name for row in artifacts) != 1 for name in required):
        raise ValueError(
            f"holdout result requires exactly one stochastic payload and metadata: {result_path}"
        )
    return result_path.parent / "stochastic.bf16.json"


def _payloads_equal(reference_path: Path, candidate_path: Path) -> bool:
    if reference_path.stat().st_size != candidate_path.stat().st_size:
        return False
    with reference_path.open("rb") as reference, candidate_path.open("rb") as candidate:
        while reference_chunk := reference.read(1024 * 1024):
            if reference_chunk != candidate.read(len(reference_chunk)):
                return False
        return candidate.read(1) == b""


def require_stochastic_identity(
    reference_path: Path,
    reference: Mapping[str, Any],
    candidate_path: Path,
    candidate: Mapping[str, Any],
) -> str:
    """Validate both streams and require every declared BF16 tensor to match."""

    seed = int(reference["seed"])
    latent_frames = int(reference["latent_frames"])
    reference_metadata_path = _stochastic_metadata_path(reference_path, reference)
    candidate_metadata_path = _stochastic_metadata_path(candidate_path, candidate)
    StochasticPlanReader(
        reference_metadata_path, seed=seed, latent_frames=latent_frames
    )
    StochasticPlanReader(
        candidate_metadata_path, seed=seed, latent_frames=latent_frames
    )
    reference_metadata = json.loads(reference_metadata_path.read_text(encoding="utf-8"))
    candidate_metadata = json.loads(candidate_metadata_path.read_text(encoding="utf-8"))
    fields = ("contract", "seed", "latent_frames", "tensor_count", "tensors")
    for field in fields:
        if reference_metadata.get(field) != candidate_metadata.get(field):
            raise ValueError(f"holdout stochastic {field} mismatch: {candidate['run_id']}")
    if not _payloads_equal(
        reference_metadata_path.with_name("stochastic.bf16"),
        candidate_metadata_path.with_name("stochastic.bf16"),
    ):
        raise ValueError(f"holdout stochastic bytes mismatch: {candidate['run_id']}")
    reference_digest = str(reference_metadata["payload_sha256"])
    if candidate_metadata.get("payload_sha256") != reference_digest:
        raise ValueError(f"holdout stochastic payload mismatch: {candidate['run_id']}")
    return reference_digest


def pair_holdout_results(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    results: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    require_complete: bool = True,
) -> list[HoldoutPair]:
    """Validate captures and pair identical workloads across CUDA and MLX."""

    jobs = _holdout_jobs(manifest)
    required_metrics = set(protocol["quality_metrics"])
    observed: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for path, result in results:
        run_id = str(result.get("run_id", ""))
        if run_id not in jobs:
            raise ValueError(f"unexpected holdout result: {run_id or path}")
        if run_id in observed:
            raise ValueError(f"duplicate holdout result: {run_id}")
        validate_result(
            result,
            jobs[run_id],
            protocol_digest=str(manifest["protocol_digest"]),
            manifest_digest=str(manifest["manifest_digest"]),
            required_metrics=required_metrics,
            root=path.parent,
        )
        observed[run_id] = (path, result)
    missing = set(jobs) - set(observed)
    if require_complete and missing:
        raise ValueError(f"holdout results incomplete: {len(missing)} missing")

    grouped: dict[HoldoutKey, dict[str, tuple[Mapping[str, Any], Path, Mapping[str, Any]]]] = {}
    for run_id, (path, result) in observed.items():
        job = jobs[run_id]
        grouped.setdefault(_key(job), {})[str(job["role"])] = (job, path, result)

    pairs: list[HoldoutPair] = []
    for key, roles in sorted(grouped.items()):
        if set(roles) != {"reference", "candidate"}:
            raise ValueError(f"holdout result pair incomplete: {key}")
        reference_job, reference_path, reference_result = roles["reference"]
        candidate_job, candidate_path, candidate_result = roles["candidate"]
        stochastic_digest = require_stochastic_identity(
            reference_path,
            reference_result,
            candidate_path,
            candidate_result,
        )
        pairs.append(
            HoldoutPair(
                key=key,
                reference_job=reference_job,
                candidate_job=candidate_job,
                reference_result=reference_result,
                candidate_result=candidate_result,
                reference_pixels=_pixels_path(reference_path, reference_result),
                candidate_pixels=_pixels_path(candidate_path, candidate_result),
                stochastic_payload_sha256=stochastic_digest,
            )
        )
    return pairs


def build_holdout_input(
    protocol: Mapping[str, Any],
    bundle: Mapping[str, Any],
    manifest: Mapping[str, Any],
    pairs: Sequence[HoldoutPair],
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind validated captures and pairwise observations to one frozen policy."""

    report_by_candidate: dict[str, Mapping[str, Any]] = {}
    for report in reports:
        if report.get("schema") != PAIRWISE_REPORT_SCHEMA:
            raise ValueError("unsupported holdout pairwise report")
        candidate_run_id = str(report.get("candidate_run_id", ""))
        if not candidate_run_id:
            raise ValueError("pairwise report requires candidate_run_id")
        if candidate_run_id in report_by_candidate:
            raise ValueError(f"duplicate pairwise report: {candidate_run_id}")
        report_by_candidate[candidate_run_id] = report

    videos: list[dict[str, Any]] = []
    consumed: set[str] = set()
    for pair in pairs:
        candidate_run_id = str(pair.candidate_job["run_id"])
        pair_report = report_by_candidate.get(candidate_run_id)
        if pair_report is None:
            raise ValueError(f"missing pairwise report: {candidate_run_id}")
        consumed.add(candidate_run_id)
        if pair_report.get("reference_run_id") != pair.reference_job["run_id"]:
            raise ValueError(f"pairwise reference mismatch: {candidate_run_id}")
        if (
            pair_report.get("stochastic_payload_sha256")
            != pair.stochastic_payload_sha256
        ):
            raise ValueError(f"pairwise stochastic binding mismatch: {candidate_run_id}")
        if pair_report.get("metric_provenance_digest") != protocol["metric_provenance"][
            "artifact_sha256"
        ]:
            raise ValueError(f"pairwise metric provenance mismatch: {candidate_run_id}")
        artifacts = pair_report.get("pixel_artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"pairwise report requires pixel artifacts: {candidate_run_id}")
        expected_reference = next(
            row["sha256"]
            for row in pair.reference_result["artifacts"]
            if row["path"] == "pixels.bin"
        )
        expected_candidate = next(
            row["sha256"]
            for row in pair.candidate_result["artifacts"]
            if row["path"] == "pixels.bin"
        )
        if artifacts != {
            "reference_sha256": expected_reference,
            "candidate_sha256": expected_candidate,
        }:
            raise ValueError(f"pairwise pixel checksum mismatch: {candidate_run_id}")
        identity = pair_report.get("identity_metrics")
        if not isinstance(identity, dict) or set(identity) != IDENTITY_METRICS:
            raise ValueError(f"pairwise identity metric set mismatch: {candidate_run_id}")
        for name, value in identity.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"pairwise identity metric {name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"pairwise identity metric {name} must be finite")
        videos.append(
            {
                "video_id": candidate_run_id,
                "stratum": pair.candidate_job["stratum"],
                "prompt_id": pair.candidate_job["prompt_id"],
                "seed_id": str(pair.candidate_job["seed"]),
                "horizon_seconds": pair.candidate_job["horizon_seconds"],
                "reference_run_id": pair.reference_job["run_id"],
                "candidate_result_digest": canonical_digest(pair.candidate_result),
                "pairwise_report_digest": canonical_digest(pair_report),
                "intrinsic_metrics": dict(pair.candidate_result["metrics"]),
                "identity_metrics": {name: float(value) for name, value in identity.items()},
            }
        )
    extra = set(report_by_candidate) - consumed
    if extra:
        raise ValueError(f"unexpected pairwise reports: {sorted(extra)}")
    return {
        "schema": HOLDOUT_SCHEMA,
        "policy_bundle_digest": bundle["bundle_digest"],
        "capture_protocol_digest": manifest["protocol_digest"],
        "run_manifest_digest": manifest["manifest_digest"],
        "candidate_videos": videos,
    }