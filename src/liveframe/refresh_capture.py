"""Fail-closed ingestion and publication for refresh proof results."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from liveframe.perceptual_policy import canonical_digest
from liveframe.quality_artifacts import atomic_json
from liveframe.quality_capture import sha256_file
from liveframe.refresh_proof import RESULT_SCHEMA, validate_refresh_proof_result

SYNTHETIC_EVIDENCE = "synthetic-contract-test"
H100_EVIDENCE = "h100-prospective"
BATCH_SCHEMA = "liveframe.refresh-addressability-batch.v2"
EVIDENCE_CLASSES = frozenset({SYNTHETIC_EVIDENCE, H100_EVIDENCE})
EVIDENCE_POLICY = {
    "schema": "liveframe.refresh-evidence-eligibility.v1",
    "decision": "split-numerical-and-timing-eligibility",
    "frozen_before_proof_results": "2026-08-04",
    "numerical": {
        "gpu": "NVIDIA H100 80GB HBM3",
        "source_revision": "8db419e341e5fc52542c0b2c4542728420ddfb4a",
        "checkpoint_sha256": (
            "sha256:bdb1b475fc88d528f510158a0990cd457f02c661b527ceb11ca9e4728533e2d0"
        ),
        "causal_model_sha256": (
            "sha256:0566a17f867a7c8ea8ad80d8b79bdfe17aaa4c7859873ffde881a6cdc12c9edd"
        ),
        "wan_wrapper_sha256": (
            "sha256:5120536c7d58fabc75187245a007f0a1486889372386b698d640fb80b280331d"
        ),
        "tf32": False,
        "identity_anchor_passed": True,
        "identity_anchor_report_required": True,
        "driver_equality_required": False,
    },
    "timing": {
        "requires_numerical_eligibility": True,
        "driver": "580.159.04",
        "torch": "2.8.0+cu128",
        "cuda": "12.8",
        "timing_protocol": "paired-interleaved-same-runtime",
        "denominator_sha256": (
            "sha256:82394c2bad77718668c8fbe7b42dd568de5a607679737312a8004321e740f766"
        ),
        "replacement_denominator_policy": (
            "prohibited after any refresh proof result is visible"
        ),
    },
}
EVIDENCE_POLICY_DIGEST = canonical_digest(EVIDENCE_POLICY)
QUALITY_METRICS = frozenset(
    {
        "clip_prompt_mean",
        "clip_prompt_p10",
        "motion_clip_distance_mean",
        "motion_lpips_mean",
        "temporal_delta_cosine_mean",
        "luma_flicker_mean",
    }
)
IDENTITY_METRICS = frozenset(
    {
        "pairwise_lpips_mean",
        "pairwise_clip_distance_mean",
        "pairwise_temporal_clip_distance_mean",
    }
)
REQUIRED_ARTIFACTS = frozenset(
    {
        "stochastic.bf16",
        "reference-latents.bf16",
        "candidate-latents.bf16",
        "reference-pixels.rgb",
        "candidate-pixels.rgb",
    }
)
DRIFT_METRICS = ("playback_seconds", "cosine", "rmse", "max_abs")
TIMING_FIELDS = frozenset(
    {
        "reference_refresh_seconds",
        "candidate_refresh_seconds",
        "reroping_seconds",
        "selection_seconds",
        "cache_movement_seconds",
        "synchronization_seconds",
        "complete_reference_wall_seconds",
        "complete_candidate_wall_seconds",
    }
)
SOURCE_MARKERS = frozenset(
    {
        "class CausalWanSelfAttention",
        "causal_rope_apply",
        "cache_start",
        "global_end_index",
        "local_end_index",
    }
)


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"refresh result requires {name}")
    return value


def evidence_eligibility(
    evidence_class: Any, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate numerical and wall-clock authority as independent layers."""

    numerical_policy = cast(Mapping[str, Any], EVIDENCE_POLICY["numerical"])
    timing_policy = cast(Mapping[str, Any], EVIDENCE_POLICY["timing"])
    numerical_failures = []
    if evidence_class != H100_EVIDENCE:
        numerical_failures.append("evidence class is not h100-prospective")
    for name in (
        "gpu",
        "source_revision",
        "checkpoint_sha256",
        "causal_model_sha256",
        "wan_wrapper_sha256",
        "tf32",
        "identity_anchor_passed",
    ):
        if runtime.get(name) != numerical_policy[name]:
            numerical_failures.append(f"runtime {name} does not match numerical policy")
    harness = runtime.get("harness_sha256")
    if (
        not isinstance(harness, str)
        or not harness.startswith("sha256:")
        or len(harness) != 71
        or harness == "sha256:" + "0" * 64
    ):
        numerical_failures.append("runtime harness_sha256 is absent or placeholder")
    identity_report = runtime.get("identity_anchor_report_sha256")
    if (
        not isinstance(identity_report, str)
        or not identity_report.startswith("sha256:")
        or len(identity_report) != 71
        or identity_report == "sha256:" + "0" * 64
    ):
        numerical_failures.append(
            "runtime identity_anchor_report_sha256 is absent or placeholder"
        )

    timing_failures = []
    if numerical_failures:
        timing_failures.append("numerical eligibility failed")
    for name in (
        "driver",
        "torch",
        "cuda",
        "timing_protocol",
        "denominator_sha256",
    ):
        if runtime.get(name) != timing_policy[name]:
            timing_failures.append(f"runtime {name} does not match timing policy")
    return {
        "policy_digest": EVIDENCE_POLICY_DIGEST,
        "numerical_eligible": not numerical_failures,
        "timing_eligible": not timing_failures,
        "numerical_failures": numerical_failures,
        "timing_failures": timing_failures,
    }


def validate_causal_source_contract(source: str) -> None:
    """Require the cache and temporal RoPE surfaces used by the proof harness."""

    if not isinstance(source, str) or not source.strip():
        raise ValueError("Causal-Forcing source contract requires source text")
    missing = sorted(marker for marker in SOURCE_MARKERS if marker not in source)
    if missing:
        raise ValueError(f"Causal-Forcing source contract is missing markers: {missing}")


def _require_finite_metrics(value: Any, expected: frozenset[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"refresh result {name} metric set mismatch")
    for metric, observed in value.items():
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
        ):
            raise ValueError(f"refresh result {name}.{metric} must be finite")


def _validate_drift_metrics(curves: Mapping[str, Any]) -> None:
    for name, rows in curves.items():
        per_layer = name in {"per-layer-key", "per-layer-value"}
        required = {"window_index", *DRIFT_METRICS}
        if per_layer:
            required.add("layer_index")
        for row in rows:
            if set(row) != required:
                raise ValueError(f"refresh result {name} drift metric set mismatch")
            for metric in DRIFT_METRICS:
                value = row[metric]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"refresh result {name}.{metric} must be finite")
            if not -1 <= float(row["cosine"]) <= 1:
                raise ValueError(f"refresh result {name}.cosine must be in [-1, 1]")
            if any(float(row[metric]) < 0 for metric in ("playback_seconds", "rmse", "max_abs")):
                raise ValueError(f"refresh result {name} drift magnitudes must be non-negative")


def _validate_timing(value: Any, *, claim_eligible: bool) -> None:
    if not isinstance(value, dict) or set(value) != TIMING_FIELDS:
        raise ValueError("refresh result timing field set mismatch")
    for name, observed in value.items():
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) < 0
        ):
            raise ValueError(f"refresh result timing {name} must be finite and non-negative")
    if claim_eligible and any(
        float(value[name]) <= 0
        for name in (
            "reference_refresh_seconds",
            "complete_reference_wall_seconds",
            "complete_candidate_wall_seconds",
        )
    ):
        raise ValueError("claim-eligible refresh result requires positive wall timing")


def validate_capture_result(
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    root: Path | None = None,
    require_numerical_eligible: bool = False,
    require_timing_eligible: bool = False,
) -> None:
    """Validate one completed paired refresh proof result and its artifacts."""

    validate_refresh_proof_result(result, manifest)
    if result.get("status") != "completed":
        raise ValueError("refresh result is not completed")
    evidence_class = result.get("evidence_class")
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError("refresh result evidence class is unsupported")

    recorded_digest = result.get("result_digest")
    unsigned = {key: value for key, value in result.items() if key != "result_digest"}
    if recorded_digest != canonical_digest(unsigned):
        raise ValueError("refresh result digest mismatch")

    stochastic = result.get("stochastic_pairing")
    if not isinstance(stochastic, dict):
        raise ValueError("refresh result requires stochastic pairing")
    reference_digest = _require_digest(
        stochastic.get("reference_sha256"), "stochastic reference_sha256"
    )
    candidate_digest = _require_digest(
        stochastic.get("candidate_sha256"), "stochastic candidate_sha256"
    )
    if reference_digest != candidate_digest:
        raise ValueError("refresh result stochastic bytes differ")
    if stochastic.get("identity") != "same-seed-same-bytes":
        raise ValueError("refresh result stochastic identity policy mismatch")

    _require_finite_metrics(result.get("intrinsic_metrics"), QUALITY_METRICS, "intrinsic")
    _require_finite_metrics(result.get("identity_metrics"), IDENTITY_METRICS, "identity")
    _validate_drift_metrics(result["drift_curves"])
    _validate_timing(result.get("timing"), claim_eligible=evidence_class == H100_EVIDENCE)
    runtime = result.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("evidence_class") != evidence_class:
        raise ValueError("refresh result runtime provenance mismatch")
    harness_digest = _require_digest(runtime.get("harness_sha256"), "runtime harness_sha256")
    eligibility = evidence_eligibility(evidence_class, runtime)
    if result.get("eligibility") != eligibility:
        raise ValueError("refresh result eligibility mismatch")
    if require_numerical_eligible and not eligibility["numerical_eligible"]:
        raise ValueError("refresh result is not numerically eligible")
    if require_timing_eligible and not eligibility["timing_eligible"]:
        raise ValueError("refresh result timing is not eligible")
    if evidence_class == H100_EVIDENCE:
        if runtime.get("gpu") != "NVIDIA H100 80GB HBM3":
            raise ValueError("H100 refresh result requires the frozen GPU class")
        _require_digest(runtime.get("checkpoint_sha256"), "runtime checkpoint_sha256")
        if harness_digest == "sha256:" + "0" * 64:
            raise ValueError("H100 refresh result has a placeholder harness digest")

    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("refresh result requires artifacts")
    observed: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("refresh result artifacts must be objects")
        name = artifact.get("path")
        if not isinstance(name, str) or name in observed:
            raise ValueError("refresh result artifact path is invalid or duplicated")
        observed.add(name)
        digest = _require_digest(artifact.get("sha256"), f"artifact {name} sha256")
        size = artifact.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"refresh result artifact {name} has an invalid size")
        if root is not None:
            path = root / name
            if (
                not path.is_file()
                or path.stat().st_size != size
                or sha256_file(path) != digest
            ):
                raise ValueError(f"refresh result artifact mismatch: {path}")
    if observed != REQUIRED_ARTIFACTS:
        raise ValueError("refresh result artifact set mismatch")
    stochastic_artifact = next(
        artifact for artifact in artifacts if artifact["path"] == "stochastic.bf16"
    )
    if stochastic_artifact["sha256"] != reference_digest:
        raise ValueError("refresh result stochastic artifact digest mismatch")


def build_capture_result(
    job: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    evidence_class: str,
    drift_curves: Mapping[str, Any],
    substitutions: Sequence[Mapping[str, Any]],
    intrinsic_metrics: Mapping[str, float],
    identity_metrics: Mapping[str, float],
    timing: Mapping[str, float],
    runtime: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    """Build and validate the byte-bound envelope shared by dry runs and H100."""

    artifacts = []
    for name in sorted(REQUIRED_ARTIFACTS):
        path = artifact_root / name
        if not path.is_file():
            raise ValueError(f"refresh capture artifact is missing: {path}")
        artifacts.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    stochastic_digest = next(
        row["sha256"] for row in artifacts if row["path"] == "stochastic.bf16"
    )
    result: dict[str, Any] = {
        **dict(job),
        "schema": RESULT_SCHEMA,
        "status": "completed",
        "manifest_digest": manifest["manifest_digest"],
        "evidence_class": evidence_class,
        "stochastic_pairing": {
            "identity": "same-seed-same-bytes",
            "reference_sha256": stochastic_digest,
            "candidate_sha256": stochastic_digest,
        },
        "drift_curves": {name: list(rows) for name, rows in drift_curves.items()},
        "substitutions": [dict(row) for row in substitutions],
        "intrinsic_metrics": dict(intrinsic_metrics),
        "identity_metrics": dict(identity_metrics),
        "timing": dict(timing),
        "runtime": {**dict(runtime), "evidence_class": evidence_class},
        "artifacts": artifacts,
    }
    result["eligibility"] = evidence_eligibility(evidence_class, result["runtime"])
    result["result_digest"] = canonical_digest(result)
    validate_capture_result(result, manifest, root=artifact_root)
    return result


def publish_capture_result(
    path: Path,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Publish once; an existing byte-bound result may not be replaced."""

    validate_capture_result(result, manifest, root=path.parent)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != result:
            raise FileExistsError(f"refusing to replace refresh result: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, result)


def load_capture_results(
    paths: Sequence[Path],
    manifest: Mapping[str, Any],
    *,
    require_complete: bool = False,
    require_numerical_eligible: bool = False,
    require_timing_eligible: bool = False,
) -> list[dict[str, Any]]:
    """Load unique manifest jobs, optionally requiring the complete 64-job campaign."""

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"refresh result must be an object: {path}")
        validate_capture_result(
            value,
            manifest,
            root=path.parent,
            require_numerical_eligible=require_numerical_eligible,
            require_timing_eligible=require_timing_eligible,
        )
        run_id = str(value["run_id"])
        if run_id in seen:
            raise ValueError(f"duplicate refresh result: {run_id}")
        seen.add(run_id)
        results.append(value)
    required = {str(job["run_id"]) for job in manifest["jobs"]}
    if require_complete and seen != required:
        raise ValueError(f"refresh proof results incomplete: {len(required - seen)} missing")
    return results


def prepare_capture_batch(
    manifest: Mapping[str, Any],
    output_root: Path,
    *,
    run_ids: set[str] | None = None,
    limit: int | None = None,
    include_completed: bool = False,
) -> dict[str, Any]:
    """Prepare an immutable H100 batch, skipping only claim-eligible completed jobs."""

    if limit is not None and limit < 1:
        raise ValueError("refresh capture batch limit must be positive")
    jobs = [dict(job) for job in manifest["jobs"]]
    known = {str(job["run_id"]) for job in jobs}
    if run_ids is not None:
        unknown = run_ids - known
        if unknown:
            raise ValueError(f"refresh capture batch requested unknown run IDs: {sorted(unknown)}")
        jobs = [job for job in jobs if job["run_id"] in run_ids]
    pending = []
    for job in jobs:
        path = output_root / str(job["run_id"]) / "result.json"
        if include_completed or not path.exists():
            pending.append(job)
            continue
        load_capture_results([path], manifest, require_numerical_eligible=True)
    if limit is not None:
        pending = pending[:limit]
    batch: dict[str, Any] = {
        "schema": BATCH_SCHEMA,
        "manifest_digest": manifest["manifest_digest"],
        "evidence_policy_digest": EVIDENCE_POLICY_DIGEST,
        "evidence_class": H100_EVIDENCE,
        "output_root": str(output_root),
        "jobs": pending,
    }
    batch["batch_digest"] = canonical_digest(batch)
    return batch


def load_capture_batch(
    path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Load a digest-bound subset containing literal frozen manifest jobs."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != BATCH_SCHEMA:
        raise ValueError("unsupported refresh capture batch")
    recorded_digest = value.get("batch_digest")
    unsigned = {key: row for key, row in value.items() if key != "batch_digest"}
    if recorded_digest != canonical_digest(unsigned):
        raise ValueError("refresh capture batch digest mismatch")
    if value.get("manifest_digest") != manifest.get("manifest_digest"):
        raise ValueError("refresh capture batch manifest mismatch")
    if value.get("evidence_policy_digest") != EVIDENCE_POLICY_DIGEST:
        raise ValueError("refresh capture batch evidence policy mismatch")
    if value.get("evidence_class") != H100_EVIDENCE:
        raise ValueError("refresh capture batch must require H100 evidence")
    expected = {str(job["run_id"]): job for job in manifest["jobs"]}
    jobs = value.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("refresh capture batch jobs must be a list")
    seen: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict) or str(job.get("run_id", "")) not in expected:
            raise ValueError("refresh capture batch contains an unknown job")
        run_id = str(job["run_id"])
        if run_id in seen:
            raise ValueError(f"refresh capture batch duplicates job: {run_id}")
        if job != expected[run_id]:
            raise ValueError(f"refresh capture batch altered job: {run_id}")
        seen.add(run_id)
    return value