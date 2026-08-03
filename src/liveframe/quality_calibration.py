"""Derive and evaluate cross-seed intrinsic quality non-inferiority policies."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from liveframe.perceptual_policy import (
    LAYERED_POLICY_SCHEMA,
    canonical_digest,
    evaluate_layered_policy,
)

PROTOCOL_SCHEMA = "liveframe.perceptual-calibration-protocol.v1"
PROTOCOL_V2_SCHEMA = "liveframe.perceptual-calibration-protocol.v2"
CALIBRATION_SCHEMA = "liveframe.quality-calibration-input.v1"
POLICY_BUNDLE_SCHEMA = "liveframe.perceptual-policy-bundle.v1"
HOLDOUT_SCHEMA = "liveframe.quality-holdout-input.v1"


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _robust_location_scale(values: Sequence[float], *, epsilon: float) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot estimate a teacher band from no videos")
    center = float(statistics.median(values))
    deviations = [abs(value - center) for value in values]
    scale = max(1.4826 * float(statistics.median(deviations)), epsilon)
    return center, scale


def raw_location_scale(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot estimate a teacher band from no videos")
    center = float(statistics.median(values))
    deviations = [abs(value - center) for value in values]
    return center, 1.4826 * float(statistics.median(deviations))


def metric_is_authoritative(protocol: Mapping[str, Any], metric: str, scale: float) -> bool:
    if protocol.get("schema") == PROTOCOL_SCHEMA:
        return True
    epsilon = float(protocol["quality_metrics"][metric]["scale_epsilon"])
    return scale > epsilon


def missing_coverage_categories(
    protocol: Mapping[str, Any], authoritative_metrics: set[str]
) -> list[str]:
    if protocol.get("schema") != PROTOCOL_V2_SCHEMA:
        return []
    authority = protocol["metric_authority_policy"]
    categories = authority["coverage_categories"]
    return [
        category
        for category in authority["required_coverage"]
        if not authoritative_metrics.intersection(categories[category])
    ]


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") not in {
        PROTOCOL_SCHEMA,
        PROTOCOL_V2_SCHEMA,
    }:
        raise ValueError("unsupported perceptual calibration protocol")
    if raw.get("status") != "frozen":
        raise ValueError("calibration protocol must be frozen before use")
    strata = raw.get("strata")
    metrics = raw.get("quality_metrics")
    identity = raw.get("identity_tripwires")
    sampling = raw.get("sampling")
    horizons = raw.get("horizons_seconds")
    horizon_specs = raw.get("horizon_specs")
    prompt_catalog = raw.get("prompt_catalog")
    if not isinstance(strata, list) or not strata:
        raise ValueError("protocol requires content strata")
    if not isinstance(prompt_catalog, dict) or not prompt_catalog:
        raise ValueError("protocol requires a prompt catalog")
    if any(
        not isinstance(prompt_id, str)
        or not prompt_id
        or not isinstance(prompt, str)
        or not prompt.strip()
        for prompt_id, prompt in prompt_catalog.items()
    ):
        raise ValueError("prompt catalog entries must be non-empty strings")
    stratum_ids: set[str] = set()
    assigned_prompts: set[str] = set()
    for stratum in strata:
        if not isinstance(stratum, dict) or not isinstance(stratum.get("id"), str):
            raise ValueError("protocol strata require string ids")
        stratum_id = stratum["id"]
        prompt_ids = stratum.get("prompt_ids")
        if stratum_id in stratum_ids:
            raise ValueError(f"duplicate protocol stratum: {stratum_id}")
        if not isinstance(prompt_ids, list) or len(prompt_ids) < 4:
            raise ValueError(f"stratum {stratum_id} requires at least four prompts")
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError(f"stratum {stratum_id} contains duplicate prompts")
        unknown = set(prompt_ids) - set(prompt_catalog)
        overlap = assigned_prompts & set(prompt_ids)
        if unknown:
            raise ValueError(f"stratum {stratum_id} has unknown prompts: {sorted(unknown)}")
        if overlap:
            raise ValueError(f"prompts assigned to multiple strata: {sorted(overlap)}")
        stratum_ids.add(stratum_id)
        assigned_prompts.update(prompt_ids)
    unassigned = set(prompt_catalog) - assigned_prompts
    if unassigned:
        raise ValueError(f"prompt catalog contains unassigned prompts: {sorted(unassigned)}")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("protocol requires intrinsic quality metrics")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("protocol requires identity tripwires")
    if not isinstance(sampling, dict):
        raise ValueError("protocol requires sampling rules")
    if (
        not isinstance(horizons, list)
        or not horizons
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in horizons
        )
        or len(horizons) != len(set(horizons))
    ):
        raise ValueError("protocol requires unique positive integer horizons_seconds")
    if not isinstance(horizon_specs, dict) or set(horizon_specs) != {
        str(value) for value in horizons
    }:
        raise ValueError("protocol requires one exact horizon spec per horizon")
    for horizon_seconds in horizons:
        spec = horizon_specs[str(horizon_seconds)]
        if not isinstance(spec, dict):
            raise ValueError(f"horizon {horizon_seconds} spec must be an object")
        latent_frames = spec.get("latent_frames")
        pixel_frames = spec.get("pixel_frames")
        playback_seconds = spec.get("playback_seconds")
        if (
            isinstance(latent_frames, bool)
            or not isinstance(latent_frames, int)
            or latent_frames < 3
            or latent_frames % 3
        ):
            raise ValueError(f"horizon {horizon_seconds} requires 3-frame latent blocks")
        if pixel_frames != 4 * (latent_frames - 1) + 1:
            raise ValueError(f"horizon {horizon_seconds} pixel-frame semantics are invalid")
        if _number(playback_seconds, f"horizon {horizon_seconds}.playback_seconds") != (
            pixel_frames / 16
        ):
            raise ValueError(f"horizon {horizon_seconds} playback duration is invalid")
    calibration_seeds = sampling.get("calibration_seeds")
    holdout_seeds = sampling.get("holdout_seeds")
    minimum_calibration = sampling.get("minimum_calibration_videos_per_prompt")
    minimum_holdout = sampling.get("minimum_holdout_videos_per_prompt")
    pass_fraction = sampling.get("minimum_holdout_pass_fraction")
    if (
        isinstance(minimum_calibration, bool)
        or not isinstance(minimum_calibration, int)
        or minimum_calibration < 8
    ):
        raise ValueError("protocol requires at least 8 calibration videos per prompt")
    if (
        isinstance(minimum_holdout, bool)
        or not isinstance(minimum_holdout, int)
        or minimum_holdout < 4
    ):
        raise ValueError("protocol requires at least 4 holdout videos per prompt")
    if (
        not isinstance(calibration_seeds, list)
        or len(calibration_seeds) < minimum_calibration
        or len(calibration_seeds) != len(set(calibration_seeds))
    ):
        raise ValueError("protocol calibration seeds do not meet the declared minimum")
    if (
        not isinstance(holdout_seeds, list)
        or len(holdout_seeds) < minimum_holdout
        or len(holdout_seeds) != len(set(holdout_seeds))
    ):
        raise ValueError("protocol holdout seeds do not meet the declared minimum")
    if set(calibration_seeds) & set(holdout_seeds):
        raise ValueError("calibration and holdout seeds must be disjoint")
    if (
        isinstance(pass_fraction, bool)
        or not isinstance(pass_fraction, (int, float))
        or not 0 < pass_fraction <= 1
    ):
        raise ValueError("minimum holdout pass fraction must be in (0, 1]")
    for name, spec in metrics.items():
        if not isinstance(spec, dict) or spec.get("direction") not in {
            "higher",
            "lower",
            "two-sided",
        }:
            raise ValueError(f"invalid quality metric direction for {name}")
        _number(spec.get("band_sigma"), f"{name}.band_sigma")
        _number(spec.get("margin_sigma"), f"{name}.margin_sigma")
        epsilon = _number(spec.get("scale_epsilon"), f"{name}.scale_epsilon")
        if epsilon <= 0:
            raise ValueError(f"{name}.scale_epsilon must be positive")
        if spec.get("aggregation") not in {"mean", "p10"}:
            raise ValueError(f"invalid quality metric aggregation for {name}")
    for name, spec in identity.items():
        if not isinstance(spec, dict):
            raise ValueError(f"identity tripwire {name} must be an object")
        upper = _number(spec.get("upper"), f"{name}.upper")
        if upper <= 0:
            raise ValueError(f"identity tripwire {name}.upper must be positive")
        if not isinstance(spec.get("rationale"), str) or not spec["rationale"].strip():
            raise ValueError(f"identity tripwire {name} requires a rationale")
    if raw.get("retuning_policy") != "no changes after any holdout result is visible":
        raise ValueError("protocol must prohibit retuning after holdout visibility")
    if raw["schema"] == PROTOCOL_V2_SCHEMA:
        source_protocol_digest = raw.get("source_protocol_digest")
        source_calibration_digest = raw.get("source_calibration_digest")
        source_sanity_digest = raw.get("source_sanity_digest")
        for name, value in (
            ("source_protocol_digest", source_protocol_digest),
            ("source_calibration_digest", source_calibration_digest),
            ("source_sanity_digest", source_sanity_digest),
        ):
            if not isinstance(value, str) or not value.startswith("sha256:"):
                raise ValueError(f"protocol v2 requires {name}")
        authority = raw.get("metric_authority_policy")
        if not isinstance(authority, dict):
            raise ValueError("protocol v2 requires metric_authority_policy")
        if authority.get("epsilon_dominated") != "diagnostic":
            raise ValueError("protocol v2 must demote epsilon-dominated cells")
        if authority.get("identity_tripwires") != "authoritative":
            raise ValueError("protocol v2 must keep identity tripwires authoritative")
        categories = authority.get("coverage_categories")
        required = authority.get("required_coverage")
        if not isinstance(categories, dict) or not categories:
            raise ValueError("protocol v2 requires coverage categories")
        if (
            not isinstance(required, list)
            or not required
            or len(required) != len(set(required))
            or set(required) - set(categories)
        ):
            raise ValueError("protocol v2 required coverage is invalid")
        known_metrics = set(metrics)
        for category, names in categories.items():
            if (
                not isinstance(category, str)
                or not isinstance(names, list)
                or not names
                or len(names) != len(set(names))
                or set(names) - known_metrics
            ):
                raise ValueError(f"protocol v2 coverage category {category} is invalid")
    budget = raw.get("budget_ceiling")
    if not isinstance(budget, dict):
        raise ValueError("protocol requires a budget ceiling")
    if _number(budget.get("h100_hours"), "budget_ceiling.h100_hours") <= 0:
        raise ValueError("H100 budget ceiling must be positive")
    if _number(budget.get("storage_gib"), "budget_ceiling.storage_gib") <= 0:
        raise ValueError("storage budget ceiling must be positive")
    if not isinstance(raw.get("visual_selection"), dict):
        raise ValueError("protocol requires visual selection rules")
    metric_provenance = raw.get("metric_provenance")
    if not isinstance(metric_provenance, dict):
        raise ValueError("protocol requires metric provenance")
    metric_digest = metric_provenance.get("artifact_sha256")
    if not isinstance(metric_digest, str) or not metric_digest.startswith("sha256:"):
        raise ValueError("protocol requires a metric provenance digest")
    return raw, canonical_digest(raw)


def _records_by_prompt(
    records: Sequence[Mapping[str, Any]], *, stratum: str, horizon_seconds: int
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    for record in records:
        if record.get("stratum") != stratum or record.get("horizon_seconds") != horizon_seconds:
            continue
        prompt_id = str(record.get("prompt_id", ""))
        seed_id = str(record.get("seed_id", ""))
        if not prompt_id or not seed_id:
            raise ValueError("calibration records require prompt_id and seed_id")
        key = (prompt_id, seed_id, horizon_seconds)
        if key in seen:
            raise ValueError(
                f"duplicate video: {prompt_id}/{seed_id}/{horizon_seconds}s"
            )
        seen.add(key)
        grouped[prompt_id].append(record)
    return dict(grouped)


def _require_prompt_counts(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_prompts: set[str],
    minimum: int,
    label: str,
    expected_seeds: set[str] | None = None,
) -> None:
    if set(grouped) != expected_prompts:
        missing = sorted(expected_prompts - set(grouped))
        extra = sorted(set(grouped) - expected_prompts)
        raise ValueError(f"{label} prompt mismatch; missing={missing}, extra={extra}")
    for prompt_id, rows in grouped.items():
        if len(rows) < minimum:
            raise ValueError(
                f"{label} prompt {prompt_id} has {len(rows)} videos; requires {minimum}"
            )
        if expected_seeds is not None:
            observed_seeds = {str(row["seed_id"]) for row in rows}
            if observed_seeds != expected_seeds:
                raise ValueError(
                    f"{label} prompt {prompt_id} seed mismatch; "
                    f"missing={sorted(expected_seeds - observed_seeds)}, "
                    f"extra={sorted(observed_seeds - expected_seeds)}"
                )


def derive_policy_bundle(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Mechanically derive per-stratum floors from independent teacher videos."""

    if calibration.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unsupported quality calibration input")
    calibration_protocol_digest = protocol.get("source_protocol_digest", protocol_digest)
    if calibration.get("protocol_digest") != calibration_protocol_digest:
        raise ValueError("calibration input is bound to another protocol")
    if protocol.get("schema") == PROTOCOL_V2_SCHEMA and canonical_digest(
        calibration
    ) != protocol.get("source_calibration_digest"):
        raise ValueError("protocol v2 is bound to another calibration input")
    teacher = calibration.get("teacher_videos")
    if not isinstance(teacher, list):
        raise ValueError("calibration input requires teacher videos")
    sampling = protocol["sampling"]
    minimum_calibration = int(sampling["minimum_calibration_videos_per_prompt"])
    calibration_seeds = {str(value) for value in sampling["calibration_seeds"]}
    policies: dict[str, Any] = {}
    for stratum_row in protocol["strata"]:
        stratum = str(stratum_row["id"])
        prompt_ids = {str(value) for value in stratum_row["prompt_ids"]}
        identity_rules = {
            metric: {
                "upper": float(spec["upper"]),
                "rationale": spec["rationale"],
                "derivation": "pre-registered-same-seed-tripwire.v1",
            }
            for metric, spec in protocol["identity_tripwires"].items()
        }
        prompt_policies: dict[str, Any] = {
            prompt_id: {"prompt_id": prompt_id, "horizons": {}}
            for prompt_id in prompt_ids
        }
        for horizon_seconds in protocol["horizons_seconds"]:
            grouped = _records_by_prompt(
                teacher, stratum=stratum, horizon_seconds=horizon_seconds
            )
            _require_prompt_counts(
                grouped,
                expected_prompts=prompt_ids,
                minimum=minimum_calibration,
                label=f"teacher/{stratum}/{horizon_seconds}s",
                expected_seeds=calibration_seeds,
            )
            for prompt_id, prompt_rows in grouped.items():
                quality_rules: dict[str, Any] = {}
                diagnostic_rules: dict[str, Any] = {}
                for metric, spec in protocol["quality_metrics"].items():
                    values = [
                        _number(row["metrics"].get(metric), f"teacher {metric}")
                        for row in prompt_rows
                    ]
                    center, raw_scale = raw_location_scale(values)
                    scale = max(raw_scale, float(spec["scale_epsilon"]))
                    radius = (
                        float(spec["band_sigma"]) + float(spec["margin_sigma"])
                    ) * scale
                    direction = spec["direction"]
                    rule: dict[str, Any] = {
                        "direction": direction,
                        "teacher_center": center,
                        "teacher_robust_scale": scale,
                        "raw_teacher_robust_scale": raw_scale,
                        "calibration_videos": len(values),
                        "derivation": "median-plus-scaled-MAD.v1",
                    }
                    if direction in {"higher", "two-sided"}:
                        rule["lower"] = center - radius
                    if direction in {"lower", "two-sided"}:
                        rule["upper"] = center + radius
                    if metric_is_authoritative(protocol, metric, raw_scale):
                        quality_rules[metric] = rule
                    else:
                        diagnostic_rules[metric] = {
                            **rule,
                            "diagnostic_reason": "epsilon-dominated-cross-seed-scale",
                            "scale_epsilon": float(spec["scale_epsilon"]),
                        }
                missing_coverage = missing_coverage_categories(
                    protocol, set(quality_rules)
                )
                if missing_coverage:
                    raise ValueError(
                        f"teacher/{stratum}/{prompt_id}/{horizon_seconds}s lacks "
                        f"authoritative coverage: {missing_coverage}"
                    )
                prompt_policies[prompt_id]["horizons"][str(horizon_seconds)] = {
                    "schema": LAYERED_POLICY_SCHEMA,
                    "claim": "intrinsic-quality-noninferiority",
                    "stratum": stratum,
                    "prompt_id": prompt_id,
                    "horizon_seconds": horizon_seconds,
                    "protocol_digest": protocol_digest,
                    "quality_metrics": quality_rules,
                    "diagnostic_metrics": diagnostic_rules,
                    "authoritative_coverage": {
                        category: sorted(
                            set(quality_rules).intersection(metrics)
                        )
                        for category, metrics in protocol.get(
                            "metric_authority_policy", {}
                        ).get("coverage_categories", {}).items()
                    },
                    "identity_tripwires": identity_rules,
                    "holdout_aggregation": {
                        "unit": "independent-video",
                        "prompt_aggregation": "minimum-pass-fraction",
                        "minimum_videos_per_prompt": sampling[
                            "minimum_holdout_videos_per_prompt"
                        ],
                        "minimum_pass_fraction": sampling[
                            "minimum_holdout_pass_fraction"
                        ],
                        "all_prompts_must_pass": True,
                    },
                }
        policies[stratum] = {
            "stratum": stratum,
            "prompts": prompt_policies,
        }
    bundle = {
        "schema": POLICY_BUNDLE_SCHEMA,
        "protocol_digest": protocol_digest,
        "source_protocol_digest": calibration_protocol_digest,
        "calibration_input_digest": canonical_digest(calibration),
        "expected_holdout_seeds": [str(value) for value in sampling["holdout_seeds"]],
        "policies": policies,
    }
    bundle["bundle_digest"] = canonical_digest(bundle)
    return bundle


def evaluate_holdout(
    bundle: Mapping[str, Any], holdout: Mapping[str, Any]
) -> dict[str, Any]:
    """Require every prompt in every stratum to meet the frozen video pass fraction."""

    if holdout.get("schema") != HOLDOUT_SCHEMA:
        raise ValueError("unsupported quality holdout input")
    if holdout.get("policy_bundle_digest") != bundle.get("bundle_digest"):
        raise ValueError("holdout is bound to another policy bundle")
    records = holdout.get("candidate_videos")
    if not isinstance(records, list):
        raise ValueError("holdout requires candidate video records")
    failures: list[str] = []
    strata: dict[str, Any] = {}
    for stratum, policy in bundle["policies"].items():
        expected_prompts = set(policy["prompts"])
        prompt_results: dict[str, Any] = {}
        for prompt_id in expected_prompts:
            horizon_results: dict[str, Any] = {}
            for horizon, prompt_policy in policy["prompts"][prompt_id]["horizons"].items():
                grouped = _records_by_prompt(
                    records, stratum=stratum, horizon_seconds=int(horizon)
                )
                if set(grouped) != expected_prompts:
                    failures.append(
                        f"holdout/{stratum}/{horizon}s: prompt mismatch; "
                        f"missing={sorted(expected_prompts - set(grouped))}, "
                        f"extra={sorted(set(grouped) - expected_prompts)}"
                    )
                rows = grouped.get(prompt_id, [])
                expected_seeds = set(bundle["expected_holdout_seeds"])
                aggregation = prompt_policy["holdout_aggregation"]
                minimum = int(aggregation["minimum_videos_per_prompt"])
                fraction = float(aggregation["minimum_pass_fraction"])
                if len(rows) < minimum:
                    failures.append(
                        f"holdout/{stratum}/{prompt_id}/{horizon}s: "
                        f"{len(rows)} videos < {minimum}"
                    )
                observed_seeds = {str(row["seed_id"]) for row in rows}
                if observed_seeds != expected_seeds:
                    failures.append(
                        f"holdout/{stratum}/{prompt_id}/{horizon}s: seed mismatch; "
                        f"missing={sorted(expected_seeds - observed_seeds)}, "
                        f"extra={sorted(observed_seeds - expected_seeds)}"
                    )
                video_results: list[dict[str, Any]] = []
                for row in rows:
                    intrinsic = row.get("intrinsic_metrics")
                    pairwise = row.get("identity_metrics")
                    if not isinstance(intrinsic, dict) or not isinstance(pairwise, dict):
                        raise ValueError(
                            "holdout records require intrinsic and identity metrics"
                        )
                    verdict = evaluate_layered_policy(intrinsic, pairwise, prompt_policy)
                    video_results.append(
                        {"video_id": row.get("video_id"), "verdict": verdict}
                    )
                passed_count = sum(
                    result["verdict"].get("passed") is True
                    for result in video_results
                )
                observed_fraction = passed_count / len(rows) if rows else 0.0
                horizon_passed = len(rows) >= minimum and observed_fraction >= fraction
                if not horizon_passed:
                    failures.append(
                        f"holdout/{stratum}/{prompt_id}/{horizon}s: pass fraction "
                        f"{observed_fraction} < {fraction}"
                    )
                horizon_results[horizon] = {
                    "passed": horizon_passed,
                    "videos": len(rows),
                    "passed_videos": passed_count,
                    "pass_fraction": observed_fraction,
                    "video_results": video_results,
                }
            prompt_results[prompt_id] = {
                "passed": all(row["passed"] for row in horizon_results.values()),
                "horizons": horizon_results,
            }
        stratum_passed = bool(prompt_results) and all(
            row["passed"] for row in prompt_results.values()
        )
        strata[stratum] = {"passed": stratum_passed, "prompts": prompt_results}
    return {
        "schema": "liveframe.quality-holdout-report.v1",
        "policy_bundle_digest": bundle["bundle_digest"],
        "passed": not failures and all(row["passed"] for row in strata.values()),
        "failures": failures,
        "strata": strata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, protocol_digest = load_protocol(args.protocol)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    bundle = derive_policy_bundle(protocol, protocol_digest, calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "digest": bundle["bundle_digest"]}))


if __name__ == "__main__":
    main()
