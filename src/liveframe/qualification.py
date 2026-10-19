"""Deployment qualification profiles and fail-closed gate evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from liveframe.engine import (
    ClipMetrics,
    ClipResult,
    Clock,
    FrameSink,
    GenerationRequest,
    StreamingBackend,
    clip,
)


# str+Enum (not enum.StrEnum) keeps the published core importable on
# Python 3.10, which the bridge worker environments pin.
class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_EVALUATED = "not_evaluated"


class QualificationVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class WorkloadProfile:
    profile_id: str
    min_playback_seconds: float
    min_native_fps: float | None = None
    max_time_to_first_frame_seconds: float | None = None
    max_p95_chunk_latency_seconds: float | None = None
    require_quality: bool = True
    require_long_horizon_stability: bool = True
    require_memory_eligibility: bool = True
    require_thermal_stability: bool = True

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must not be empty")
        _require_positive(self.min_playback_seconds, "min_playback_seconds")
        for name, value in (
            ("min_native_fps", self.min_native_fps),
            ("max_time_to_first_frame_seconds", self.max_time_to_first_frame_seconds),
            ("max_p95_chunk_latency_seconds", self.max_p95_chunk_latency_seconds),
        ):
            if value is not None:
                _require_positive(value, name)


@dataclass(frozen=True, slots=True)
class QualificationEvidence:
    quality_passed: bool | None = None
    long_horizon_stability_passed: bool | None = None
    memory_eligible: bool | None = None
    thermal_stable: bool | None = None


@dataclass(frozen=True, slots=True)
class QualificationGate:
    name: str
    status: GateStatus
    requirement: str
    observed: float | bool | None


@dataclass(frozen=True, slots=True)
class QualificationResult:
    profile_id: str
    verdict: QualificationVerdict
    gates: tuple[QualificationGate, ...]

    @property
    def failed_gates(self) -> tuple[QualificationGate, ...]:
        return tuple(gate for gate in self.gates if gate.status is GateStatus.FAIL)

    @property
    def missing_gates(self) -> tuple[QualificationGate, ...]:
        return tuple(gate for gate in self.gates if gate.status is GateStatus.NOT_EVALUATED)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "verdict": self.verdict.value,
            "gates": [
                {
                    "name": gate.name,
                    "status": gate.status.value,
                    "requirement": gate.requirement,
                    "observed": gate.observed,
                }
                for gate in self.gates
            ],
        }


@dataclass(frozen=True, slots=True)
class QualificationRun:
    profile: WorkloadProfile
    clip: ClipResult
    qualification: QualificationResult


def _require_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


INTERACTIVE_480P: Final = WorkloadProfile(
    profile_id="interactive-480p",
    min_playback_seconds=60.0,
    min_native_fps=16.0,
    max_time_to_first_frame_seconds=2.0,
    max_p95_chunk_latency_seconds=0.25,
)
CREATIVE_PREVIEW: Final = WorkloadProfile(
    profile_id="creative-preview",
    min_playback_seconds=15.0,
    min_native_fps=4.0,
    max_time_to_first_frame_seconds=4.0,
    max_p95_chunk_latency_seconds=1.0,
)
OFFLINE_STREAMING: Final = WorkloadProfile(
    profile_id="offline-streaming",
    min_playback_seconds=60.0,
)

WORKLOAD_PROFILES: Final = MappingProxyType(
    {
        profile.profile_id: profile
        for profile in (INTERACTIVE_480P, CREATIVE_PREVIEW, OFFLINE_STREAMING)
    }
)


def evaluate_qualification(
    profile: WorkloadProfile,
    request: GenerationRequest,
    metrics: ClipMetrics,
    evidence: QualificationEvidence | None = None,
) -> QualificationResult:
    """Evaluate measured results without treating absent evidence as a pass."""

    supplied = evidence or QualificationEvidence()
    gates = [
        _minimum_gate(
            "playback_duration",
            request.playback_seconds,
            profile.min_playback_seconds,
            "seconds",
        )
    ]
    if profile.min_native_fps is not None:
        gates.append(
            _minimum_gate(
                "native_generated_fps",
                metrics.generated_fps,
                profile.min_native_fps,
                "FPS",
            )
        )
    if profile.max_time_to_first_frame_seconds is not None:
        gates.append(
            _maximum_gate(
                "time_to_first_frame",
                metrics.time_to_first_frame_seconds,
                profile.max_time_to_first_frame_seconds,
                "seconds",
            )
        )
    if profile.max_p95_chunk_latency_seconds is not None:
        gates.append(
            _maximum_gate(
                "p95_chunk_latency",
                metrics.p95_emission_interval_seconds,
                profile.max_p95_chunk_latency_seconds,
                "seconds",
            )
        )
    for name, required, observed in (
        ("quality", profile.require_quality, supplied.quality_passed),
        (
            "long_horizon_stability",
            profile.require_long_horizon_stability,
            supplied.long_horizon_stability_passed,
        ),
        ("memory_eligibility", profile.require_memory_eligibility, supplied.memory_eligible),
        ("thermal_stability", profile.require_thermal_stability, supplied.thermal_stable),
    ):
        if required:
            gates.append(_evidence_gate(name, observed))

    gate_tuple = tuple(gates)
    if any(gate.status is GateStatus.FAIL for gate in gate_tuple):
        verdict = QualificationVerdict.FAIL
    elif any(gate.status is GateStatus.NOT_EVALUATED for gate in gate_tuple):
        verdict = QualificationVerdict.INCOMPLETE
    else:
        verdict = QualificationVerdict.PASS
    return QualificationResult(profile.profile_id, verdict, gate_tuple)


def qualify(
    backend: StreamingBackend,
    profile: WorkloadProfile,
    request: GenerationRequest,
    evidence: QualificationEvidence | None = None,
    *,
    clock: Clock | None = None,
    frame_sink: FrameSink | None = None,
    retain_events: bool = False,
) -> QualificationRun:
    """Run one clip and evaluate it against a deployment profile."""

    measured = (
        clip(
            backend,
            request,
            frame_sink=frame_sink,
            retain_events=retain_events,
        )
        if clock is None
        else clip(
            backend,
            request,
            clock=clock,
            frame_sink=frame_sink,
            retain_events=retain_events,
        )
    )
    result = evaluate_qualification(profile, request, measured.metrics, evidence)
    return QualificationRun(profile=profile, clip=measured, qualification=result)


def _minimum_gate(name: str, observed: float, minimum: float, unit: str) -> QualificationGate:
    return QualificationGate(
        name=name,
        status=GateStatus.PASS if observed >= minimum else GateStatus.FAIL,
        requirement=f">= {minimum:g} {unit}",
        observed=observed,
    )


def _maximum_gate(name: str, observed: float, maximum: float, unit: str) -> QualificationGate:
    return QualificationGate(
        name=name,
        status=GateStatus.PASS if observed <= maximum else GateStatus.FAIL,
        requirement=f"<= {maximum:g} {unit}",
        observed=observed,
    )


def _evidence_gate(name: str, observed: bool | None) -> QualificationGate:
    if observed is None:
        status = GateStatus.NOT_EVALUATED
    else:
        status = GateStatus.PASS if observed else GateStatus.FAIL
    return QualificationGate(
        name=name,
        status=status,
        requirement="required to pass",
        observed=observed,
    )