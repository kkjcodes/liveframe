from __future__ import annotations

from liveframe.engine import ClipMetrics, GenerationRequest
from liveframe.qualification import (
    INTERACTIVE_480P,
    GateStatus,
    QualificationEvidence,
    QualificationVerdict,
    evaluate_qualification,
)


def _passing_metrics() -> ClipMetrics:
    return ClipMetrics(
        generated_native_frames=960,
        wall_seconds=50.0,
        generated_fps=19.2,
        time_to_first_frame_seconds=0.8,
        p95_emission_interval_seconds=0.2,
    )


def test_performance_pass_remains_incomplete_without_required_evidence() -> None:
    result = evaluate_qualification(
        INTERACTIVE_480P,
        GenerationRequest("prompt", native_frames=960),
        _passing_metrics(),
    )

    assert result.verdict is QualificationVerdict.INCOMPLETE
    assert {gate.name for gate in result.missing_gates} == {
        "quality",
        "long_horizon_stability",
        "memory_eligibility",
        "thermal_stability",
    }
    assert all(gate.status is GateStatus.PASS for gate in result.gates[:4])


def test_any_failed_gate_makes_qualification_fail() -> None:
    result = evaluate_qualification(
        INTERACTIVE_480P,
        GenerationRequest("prompt", native_frames=960),
        _passing_metrics(),
        QualificationEvidence(
            quality_passed=True,
            long_horizon_stability_passed=False,
            memory_eligible=True,
            thermal_stable=True,
        ),
    )

    assert result.verdict is QualificationVerdict.FAIL
    assert [gate.name for gate in result.failed_gates] == ["long_horizon_stability"]


def test_all_required_gates_can_pass() -> None:
    result = evaluate_qualification(
        INTERACTIVE_480P,
        GenerationRequest("prompt", native_frames=960),
        _passing_metrics(),
        QualificationEvidence(
            quality_passed=True,
            long_horizon_stability_passed=True,
            memory_eligible=True,
            thermal_stable=True,
        ),
    )

    assert result.verdict is QualificationVerdict.PASS
    assert not result.failed_gates
    assert not result.missing_gates


def test_short_clip_cannot_qualify_for_sustained_profile() -> None:
    result = evaluate_qualification(
        INTERACTIVE_480P,
        GenerationRequest("prompt", native_frames=81),
        _passing_metrics(),
    )

    assert result.verdict is QualificationVerdict.FAIL
    assert result.failed_gates[0].name == "playback_duration"