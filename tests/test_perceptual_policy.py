from __future__ import annotations

import json

import pytest

from liveframe.perceptual_policy import (
    POLICY_SCHEMA,
    canonical_digest,
    evaluate_policy,
    load_policy,
    summarize,
    temporal_relative_error,
)
from liveframe.rolling_perceptual_parity import _clip_feature_tensor, _intrinsic_metrics


def test_missing_policy_is_measurement_only() -> None:
    verdict = evaluate_policy({"lpips_mean": 0.01}, None)
    assert verdict["authoritative"] is False
    assert verdict["passed"] is None


def test_frozen_policy_evaluates_all_required_metrics(tmp_path) -> None:
    raw = {
        "schema": POLICY_SCHEMA,
        "metrics": {
            "lpips_mean": {"operator": "<=", "limit": 0.1},
            "clip_cosine_mean": {"operator": ">=", "limit": 0.98},
        },
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    policy, digest = load_policy(path)

    assert digest == canonical_digest(raw)
    assert evaluate_policy(
        {"lpips_mean": 0.02, "clip_cosine_mean": 0.99}, policy
    )["passed"]
    assert not evaluate_policy(
        {"lpips_mean": 0.11, "clip_cosine_mean": 0.99}, policy
    )["passed"]


def test_policy_fails_closed_for_missing_metric() -> None:
    policy = {
        "schema": POLICY_SCHEMA,
        "metrics": {"lpips_mean": {"operator": "<=", "limit": 0.1}},
    }
    verdict = evaluate_policy({}, policy)
    assert verdict["passed"] is False
    assert verdict["failures"] == ["missing required metric: lpips_mean"]


def test_temporal_relative_error_and_summary() -> None:
    values = temporal_relative_error([1.0, 2.0], [1.1, 1.0])
    assert values == pytest.approx([0.1, 0.5])
    summary = summarize(values)
    assert summary["mean"] == pytest.approx(0.3)
    assert summary["maximum"] == pytest.approx(0.5)


def test_invalid_policy_is_rejected(tmp_path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"schema": POLICY_SCHEMA, "metrics": {}}))
    with pytest.raises(ValueError, match="metric floors"):
        load_policy(path)


def test_clip_feature_tensor_supports_transformers_api_variants() -> None:
    tensor = object()

    class Output:
        pooler_output = tensor

    assert _clip_feature_tensor(tensor) is tensor
    assert _clip_feature_tensor(Output()) is tensor


def test_intrinsic_metrics_aggregate_one_complete_video() -> None:
    metrics = _intrinsic_metrics(
        prompt_cosines=[0.8, 0.9, 1.0],
        motion_clip_distances=[0.1, 0.2],
        motion_feature_norms=[0.3, 0.4],
        motion_lpips=[0.2, 0.4],
        temporal_delta_cosines=[0.7],
        luma_means=[10.0, 12.0, 11.0],
    )
    assert metrics["clip_prompt_mean"] == pytest.approx(0.9)
    assert metrics["motion_clip_distance_mean"] == pytest.approx(0.15)
    assert metrics["motion_lpips_mean"] == pytest.approx(0.3)
    assert metrics["motion_valid_fraction"] == pytest.approx(1.0)
    assert metrics["luma_flicker_mean"] == pytest.approx(3.0)


def test_intrinsic_metrics_reject_incomplete_or_degenerate_video() -> None:
    values = {
        "prompt_cosines": [0.8, 0.9, 1.0],
        "motion_clip_distances": [0.1, 0.2],
        "motion_feature_norms": [0.3, 0.4],
        "motion_lpips": [0.2, 0.4],
        "temporal_delta_cosines": [0.7],
        "luma_means": [10.0, 12.0, 11.0],
    }
    incomplete = {**values, "motion_lpips": [0.2]}
    with pytest.raises(ValueError, match="motion_lpips has 1 values; expected 2"):
        _intrinsic_metrics(**incomplete)

    degenerate = {**values, "motion_feature_norms": [0.0, 0.0]}
    with pytest.raises(ValueError, match="undefined for zero feature motion"):
        _intrinsic_metrics(**degenerate)

    nonfinite = {**values, "luma_means": [10.0, float("nan"), 11.0]}
    with pytest.raises(ValueError, match="luma_means must be finite"):
        _intrinsic_metrics(**nonfinite)