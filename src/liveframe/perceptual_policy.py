"""Pre-registered policy primitives for long-horizon perceptual parity."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

POLICY_SCHEMA = "liveframe.perceptual-policy.v1"
LAYERED_POLICY_SCHEMA = "liveframe.perceptual-policy.v2"


def canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def load_policy(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    """Load a frozen policy or explicitly return measurement-only mode."""

    if path is None:
        return None, None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") not in {
        POLICY_SCHEMA,
        LAYERED_POLICY_SCHEMA,
    }:
        raise ValueError("unsupported perceptual policy schema")
    if raw["schema"] == LAYERED_POLICY_SCHEMA:
        _validate_layered_policy(raw)
        return raw, canonical_digest(raw)
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("perceptual policy must define metric floors")
    for name, rule in metrics.items():
        if not isinstance(name, str) or not isinstance(rule, dict):
            raise ValueError("perceptual policy metric rules must be objects")
        operator = rule.get("operator")
        limit = rule.get("limit")
        if operator not in {"<=", ">="} or isinstance(limit, bool) or not isinstance(
            limit, (int, float)
        ):
            raise ValueError(f"invalid perceptual policy rule for {name}")
    return raw, canonical_digest(raw)


def _validate_metric_bounds(name: str, rule: Any) -> None:
    if not isinstance(rule, dict):
        raise ValueError(f"metric rule {name} must be an object")
    bounds = [bound for bound in ("lower", "upper") if bound in rule]
    if not bounds:
        raise ValueError(f"metric rule {name} requires a lower or upper bound")
    for bound in bounds:
        value = rule[bound]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"metric rule {name}.{bound} must be numeric")
    if "lower" in rule and "upper" in rule and rule["lower"] > rule["upper"]:
        raise ValueError(f"metric rule {name} has inverted bounds")


def _validate_layered_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("claim") != "intrinsic-quality-noninferiority":
        raise ValueError("v2 policy must declare intrinsic quality non-inferiority")
    quality = policy.get("quality_metrics")
    identity = policy.get("identity_tripwires")
    if not isinstance(quality, dict) or not quality:
        raise ValueError("v2 policy requires intrinsic quality metrics")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("v2 policy requires pairwise identity tripwires")
    for name, rule in quality.items():
        _validate_metric_bounds(str(name), rule)
    for name, rule in identity.items():
        _validate_metric_bounds(str(name), rule)
    aggregation = policy.get("holdout_aggregation")
    if not isinstance(aggregation, dict):
        raise ValueError("v2 policy requires holdout aggregation rules")
    minimum = aggregation.get("minimum_videos_per_prompt")
    pass_fraction = aggregation.get("minimum_pass_fraction")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 2:
        raise ValueError("minimum_videos_per_prompt must be at least 2")
    if (
        isinstance(pass_fraction, bool)
        or not isinstance(pass_fraction, (int, float))
        or not 0 < pass_fraction <= 1
    ):
        raise ValueError("minimum_pass_fraction must be in (0, 1]")


def evaluate_policy(
    observed: Mapping[str, float], policy: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Evaluate frozen floors; absent floors cannot produce a pass verdict."""

    if policy is None:
        return {
            "status": "measurement-only",
            "authoritative": False,
            "passed": None,
            "failures": ["no pre-registered perceptual policy supplied"],
        }
    if policy.get("schema") == LAYERED_POLICY_SCHEMA:
        raise ValueError("v2 policies require evaluate_layered_policy")
    failures: list[str] = []
    metrics = policy["metrics"]
    for name, rule in metrics.items():
        if name not in observed:
            failures.append(f"missing required metric: {name}")
            continue
        value = float(observed[name])
        limit = float(rule["limit"])
        accepted = value <= limit if rule["operator"] == "<=" else value >= limit
        if not accepted:
            failures.append(f"{name}={value} must be {rule['operator']} {limit}")
    return {
        "status": "passed" if not failures else "failed",
        "authoritative": True,
        "passed": not failures,
        "failures": failures,
    }


def _evaluate_bounds(
    observed: Mapping[str, float], rules: Mapping[str, Any], *, layer: str
) -> dict[str, Any]:
    failures: list[str] = []
    for name, rule in rules.items():
        if name not in observed:
            failures.append(f"{layer}: missing required metric: {name}")
            continue
        value = float(observed[name])
        if not math.isfinite(value):
            failures.append(f"{layer}: {name} must be finite")
            continue
        if "lower" in rule and value < float(rule["lower"]):
            failures.append(f"{layer}: {name}={value} must be >= {rule['lower']}")
        if "upper" in rule and value > float(rule["upper"]):
            failures.append(f"{layer}: {name}={value} must be <= {rule['upper']}")
    return {"passed": not failures, "failures": failures}


def evaluate_layered_policy(
    intrinsic: Mapping[str, float],
    pairwise: Mapping[str, float],
    policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate quality and identity separately; neither layer can mask the other."""

    if policy is None:
        return {
            "status": "measurement-only",
            "authoritative": False,
            "passed": None,
            "quality": None,
            "identity": None,
            "failures": ["no pre-registered layered perceptual policy supplied"],
        }
    if policy.get("schema") != LAYERED_POLICY_SCHEMA:
        raise ValueError("layered evaluation requires a v2 perceptual policy")
    _validate_layered_policy(policy)
    quality = _evaluate_bounds(
        intrinsic, policy["quality_metrics"], layer="quality"
    )
    identity = _evaluate_bounds(
        pairwise, policy["identity_tripwires"], layer="identity"
    )
    failures = [*quality["failures"], *identity["failures"]]
    return {
        "status": "passed" if not failures else "failed",
        "authoritative": True,
        "passed": not failures,
        "quality": quality,
        "identity": identity,
        "failures": failures,
    }


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty metric series")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "minimum": ordered[0],
        "p05": percentile(0.05),
        "p10": percentile(0.10),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "maximum": ordered[-1],
    }


def temporal_relative_error(
    reference_delta: list[float], candidate_delta: list[float], *, epsilon: float = 1e-8
) -> list[float]:
    if len(reference_delta) != len(candidate_delta):
        raise ValueError("temporal metric series must have equal length")
    return [
        abs(candidate - reference) / max(abs(reference), epsilon)
        for reference, candidate in zip(reference_delta, candidate_delta, strict=True)
    ]