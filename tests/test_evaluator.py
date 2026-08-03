from __future__ import annotations

from typing import Any

from liveframe.evaluator import evaluate_native_proof
from liveframe.gates import load_bundled_gate


def evidence(rate: float) -> dict[str, Any]:
    gate = load_bundled_gate()
    count = int(rate * 60)
    generated = [(index + 0.5) / rate for index in range(count)]
    successful_displays = min(960, count + 8)
    displayed = [(index + 0.5) / 16 for index in range(successful_displays)]
    q_start = 8
    tick_times = [0.0, *[(index + 0.5) / 16 for index in range(960)]]
    queue_samples = [
        {
            "elapsed_seconds": elapsed,
            "occupancy": q_start
            + sum(generated_at <= elapsed for generated_at in generated)
            - sum(displayed_at <= elapsed for displayed_at in displayed),
        }
        for elapsed in tick_times
    ]
    return {
        "gate_id": gate.gate_id,
        "gate_digest": gate.digest,
        "q_start": q_start,
        "q_max": 16,
        "steady_emission_size": 4,
        "generated_frame_times_seconds": generated,
        "displayed_frame_times_seconds": displayed,
        "display_underruns": 960 - successful_displays,
        "queue_samples": queue_samples,
        "emission_latencies_ms": [200.0] * 20,
        "playback_fps": 16,
        "uses_interpolation": False,
        "uses_duplication": False,
        "uses_decimation": False,
        "power_context": {"charger": "connected", "macos_power_mode": "automatic"},
    }


def test_exact_native_rate_passes() -> None:
    result = evaluate_native_proof(evidence(16.0))

    assert result.passed
    assert result.generated_frames == 960
    assert result.queue_end == result.queue_min == 8


def test_15_5_fps_fails_frame_count_underrun_and_reserve() -> None:
    result = evaluate_native_proof(evidence(15.5))

    assert not result.passed
    assert result.generated_frames == 930
    assert "generated_at_least_960" in result.failures
    assert "zero_underruns" in result.failures
    assert "displayed_at_least_960" in result.failures


def test_interpolation_cannot_pass_primary_gate() -> None:
    proof = evidence(16.0)
    proof["uses_interpolation"] = True

    result = evaluate_native_proof(proof)

    assert not result.passed
    assert result.failures == ("generated_not_interpolated",)


def test_sparse_or_fabricated_queue_samples_fail() -> None:
    proof = evidence(16.0)
    proof["queue_samples"] = [
        {"elapsed_seconds": 0, "occupancy": 8},
        {"elapsed_seconds": 60, "occupancy": 9},
    ]

    result = evaluate_native_proof(proof)

    assert not result.passed
    assert "queue_tick_coverage" in result.failures
    assert "exact_queue_accounting" in result.failures
