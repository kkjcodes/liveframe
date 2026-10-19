"""Digest-bound ledgers for deployment qualification runs."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

from liveframe.backends.base import BackendCapabilities
from liveframe.engine import ClipMetrics, GenerationRequest
from liveframe.events import utc_now
from liveframe.gates import CANONICALIZATION, contract_digest
from liveframe.host import collect_host_provenance
from liveframe.qualification import (
    QualificationEvidence,
    QualificationRun,
    WorkloadProfile,
    evaluate_qualification,
)

QUALIFICATION_LEDGER_SCHEMA: Final = "liveframe.qualification-run.v1"


def build_qualification_ledger(
    capabilities: BackendCapabilities,
    run: QualificationRun,
    *,
    created_at: str | None = None,
    host_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    profile = asdict(run.profile)
    ledger: dict[str, Any] = {
        "schema": QUALIFICATION_LEDGER_SCHEMA,
        "canonicalization": CANONICALIZATION,
        "created_at": created_at or utc_now(),
        "adapter": asdict(capabilities),
        "profile": profile,
        "profile_digest": contract_digest(profile),
        "request": asdict(run.clip.request),
        "metrics": asdict(run.clip.metrics),
        "qualification": run.qualification.as_dict(),
        "host": dict(host_provenance or collect_host_provenance()),
    }
    ledger["ledger_digest"] = contract_digest(ledger)
    return ledger


def verify_qualification_ledger(ledger: Mapping[str, Any]) -> None:
    if ledger.get("schema") != QUALIFICATION_LEDGER_SCHEMA:
        raise ValueError("unsupported qualification ledger schema")
    if ledger.get("canonicalization") != CANONICALIZATION:
        raise ValueError("unsupported qualification ledger canonicalization")
    recorded_digest = ledger.get("ledger_digest")
    unsigned = {key: value for key, value in ledger.items() if key != "ledger_digest"}
    if recorded_digest != contract_digest(unsigned):
        raise ValueError("qualification ledger digest mismatch")

    request = _mapping(ledger.get("request"), "request")
    metrics = _mapping(ledger.get("metrics"), "metrics")
    profile = _mapping(ledger.get("profile"), "profile")
    if ledger.get("profile_digest") != contract_digest(profile):
        raise ValueError("qualification ledger profile digest mismatch")
    requested_frames = _positive_number(request, "native_frames")
    generated_frames = _positive_number(metrics, "generated_native_frames")
    wall_seconds = _positive_number(metrics, "wall_seconds")
    generated_fps = _positive_number(metrics, "generated_fps")
    if requested_frames != generated_frames:
        raise ValueError("qualification ledger native-frame count mismatch")
    expected_fps = generated_frames / wall_seconds
    if not math.isclose(generated_fps, expected_fps, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("qualification ledger generated FPS mismatch")
    _mapping(ledger.get("adapter"), "adapter")
    qualification = _mapping(ledger.get("qualification"), "qualification")
    _mapping(ledger.get("host"), "host")

    request_object = GenerationRequest(**dict(request))
    metrics_object = ClipMetrics(**dict(metrics))
    profile_object = WorkloadProfile(**dict(profile))
    evidence = QualificationEvidence(
        quality_passed=_gate_observed(qualification, "quality"),
        long_horizon_stability_passed=_gate_observed(
            qualification, "long_horizon_stability"
        ),
        memory_eligible=_gate_observed(qualification, "memory_eligibility"),
        thermal_stable=_gate_observed(qualification, "thermal_stability"),
    )
    recomputed = evaluate_qualification(
        profile_object,
        request_object,
        metrics_object,
        evidence,
    ).as_dict()
    if qualification != recomputed:
        raise ValueError("qualification ledger gate recomputation mismatch")


def write_qualification_ledger(path: Path, ledger: Mapping[str, Any]) -> None:
    verify_qualification_ledger(ledger)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(ledger, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_qualification_ledger(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("qualification ledger must be a JSON object")
    verify_qualification_ledger(raw)
    return raw


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"qualification ledger {name} must be an object")
    return value


def _positive_number(container: Mapping[str, Any], name: str) -> float:
    value = container.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"qualification ledger {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"qualification ledger {name} must be finite and positive")
    return result


def _gate_observed(qualification: Mapping[str, Any], name: str) -> bool | None:
    gates = qualification.get("gates")
    if not isinstance(gates, list):
        raise ValueError("qualification ledger gates must be an array")
    matching = [gate for gate in gates if isinstance(gate, Mapping) and gate.get("name") == name]
    if not matching:
        return None
    if len(matching) != 1:
        raise ValueError(f"qualification ledger contains duplicate gate: {name}")
    observed = matching[0].get("observed")
    if observed is not None and not isinstance(observed, bool):
        raise ValueError(f"qualification ledger evidence gate must be boolean: {name}")
    return observed