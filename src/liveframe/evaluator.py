"""Independent evaluator for the gates.v1 native-rate stream proof."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

from liveframe.gates import GateArtifact, GateError, load_bundled_gate


class EvaluationError(ValueError):
    """Raised when proof evidence is malformed or bound to another gate."""


@dataclass(frozen=True, slots=True)
class ProofResult:
    gate_id: str
    gate_digest: str
    passed: bool
    generated_frames: int
    displayed_frames: int
    display_underruns: int
    queue_min: int
    queue_max: int
    queue_end: int
    p95_emission_latency_ms: float | None
    checks: Mapping[str, bool]
    failures: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def evaluate_native_proof(
    evidence: Mapping[str, Any], gate: GateArtifact | None = None
) -> ProofResult:
    artifact = gate or load_bundled_gate()
    artifact.verify()
    if evidence.get("gate_id") != artifact.gate_id:
        raise GateError("proof evidence gate ID mismatch")
    if evidence.get("gate_digest") != artifact.digest:
        raise GateError("proof evidence gate digest mismatch")

    q_start = _integer(evidence, "q_start")
    q_max_allowed = _integer(evidence, "q_max")
    emission_size = _integer(evidence, "steady_emission_size")
    if min(q_start, q_max_allowed, emission_size) < 0 or emission_size == 0:
        message = "queue and emission values must be non-negative; emission must be positive"
        raise EvaluationError(message)

    generated_times = _number_list(evidence, "generated_frame_times_seconds")
    displayed_times = _number_list(evidence, "displayed_frame_times_seconds")
    underruns = _integer(evidence, "display_underruns")
    queue_samples = _queue_samples(evidence.get("queue_samples"))
    emission_latencies = _number_list(evidence, "emission_latencies_ms")
    emission_playback_ms = emission_size * 1000.0 / 16.0

    generated = sum(0.0 <= value <= 60.0 for value in generated_times)
    displayed = sum(0.0 <= value <= 60.0 for value in displayed_times)
    in_window_queue = [occupancy for elapsed, occupancy in queue_samples if 0.0 <= elapsed <= 60.0]
    if not in_window_queue:
        raise EvaluationError("at least one queue sample in the 60-second proof window is required")
    queue_min = min(in_window_queue)
    queue_max = max(in_window_queue)
    queue_end = in_window_queue[-1]

    window_counts: defaultdict[int, int] = defaultdict(int)
    for elapsed in generated_times:
        if 0.0 <= elapsed < 60.0:
            window_counts[min(int(elapsed // 10.0), 5)] += 1
    window_floor = 160 - emission_size
    aligned_windows_pass = all(window_counts[index] >= window_floor for index in range(6))
    p95 = _percentile_95(emission_latencies)
    queue_tick_coverage = len(in_window_queue) >= 961
    exact_queue_accounting = queue_tick_coverage and all(
        occupancy
        == q_start
        + sum(generated_at <= elapsed for generated_at in generated_times)
        - sum(displayed_at <= elapsed for displayed_at in displayed_times)
        for elapsed, occupancy in queue_samples
        if 0.0 <= elapsed <= 60.0
    )

    checks = {
        "generated_at_least_960": generated >= 960,
        "displayed_at_least_960": displayed >= 960,
        "zero_underruns": underruns == 0,
        "queue_non_negative": queue_min >= 0,
        "queue_bounded": queue_max <= q_max_allowed,
        "queue_non_draining": queue_end >= q_start - emission_size,
        "queue_tick_coverage": queue_tick_coverage,
        "exact_queue_accounting": exact_queue_accounting,
        "aligned_windows": aligned_windows_pass,
        "emission_p95_deadline": p95 is not None and p95 < emission_playback_ms,
        "native_playback_fps": evidence.get("playback_fps") == 16,
        "generated_not_interpolated": evidence.get("uses_interpolation") is False,
        "generated_not_duplicated": evidence.get("uses_duplication") is False,
        "generated_not_decimated": evidence.get("uses_decimation") is False,
        "power_context_present": bool(evidence.get("power_context")),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return ProofResult(
        gate_id=artifact.gate_id,
        gate_digest=artifact.digest,
        passed=not failures,
        generated_frames=generated,
        displayed_frames=displayed,
        display_underruns=underruns,
        queue_min=queue_min,
        queue_max=queue_max,
        queue_end=queue_end,
        p95_emission_latency_ms=p95,
        checks=checks,
        failures=failures,
    )


def _integer(evidence: Mapping[str, Any], key: str) -> int:
    value = evidence.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationError(f"{key} must be an integer")
    return value


def _number_list(evidence: Mapping[str, Any], key: str) -> list[float]:
    value = evidence.get(key)
    if not isinstance(value, list):
        raise EvaluationError(f"{key} must be an array")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item):
            raise EvaluationError(f"{key} contains a non-finite number")
        result.append(float(item))
    return result


def _queue_samples(value: Any) -> list[tuple[float, int]]:
    if not isinstance(value, list):
        raise EvaluationError("queue_samples must be an array")
    result: list[tuple[float, int]] = []
    for sample in value:
        if not isinstance(sample, dict):
            raise EvaluationError("queue sample must be an object")
        elapsed = sample.get("elapsed_seconds")
        occupancy = sample.get("occupancy")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, int | float)
            or not math.isfinite(elapsed)
            or isinstance(occupancy, bool)
            or not isinstance(occupancy, int)
        ):
            raise EvaluationError("queue sample has invalid elapsed time or occupancy")
        result.append((float(elapsed), occupancy))
    if any(current[0] < previous[0] for previous, current in pairwise(result)):
        raise EvaluationError("queue samples must be ordered by elapsed time")
    return result


def summarize_rates(times: Iterable[float], duration_seconds: int = 60) -> float:
    """Return observed generated FPS inside a closed proof interval."""
    if duration_seconds <= 0:
        raise EvaluationError("duration must be positive")
    count = sum(0.0 <= elapsed <= duration_seconds for elapsed in times)
    return count / duration_seconds
