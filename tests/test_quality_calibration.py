from __future__ import annotations

import copy

import pytest

from liveframe.perceptual_policy import evaluate_layered_policy
from liveframe.quality_calibration import (
    CALIBRATION_SCHEMA,
    HOLDOUT_SCHEMA,
    PROTOCOL_SCHEMA,
    derive_policy_bundle,
    evaluate_holdout,
)

DIGEST = "sha256:frozen-protocol"


def _protocol() -> dict[str, object]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "status": "frozen",
        "horizons_seconds": [15],
        "horizon_specs": {
            "15": {"latent_frames": 60, "pixel_frames": 237, "playback_seconds": 14.8125}
        },
        "strata": [{"id": "animation", "prompt_ids": ["p1"]}],
        "quality_metrics": {
            "alignment": {
                "direction": "higher",
                "band_sigma": 1.0,
                "margin_sigma": 1.0,
                "scale_epsilon": 0.01,
            },
            "flicker": {
                "direction": "lower",
                "band_sigma": 1.0,
                "margin_sigma": 1.0,
                "scale_epsilon": 0.001,
            },
            "motion": {
                "direction": "two-sided",
                "band_sigma": 1.0,
                "margin_sigma": 1.0,
                "scale_epsilon": 0.01,
            },
        },
        "identity_tripwires": {
            "pairwise_lpips_mean": {
                "upper": 0.125,
                "rationale": "Loose same-seed trajectory identity sanity bound.",
            }
        },
        "sampling": {
            "minimum_calibration_videos_per_prompt": 3,
            "minimum_holdout_videos_per_prompt": 2,
            "minimum_holdout_pass_fraction": 1.0,
            "calibration_seeds": [10, 11, 12],
            "holdout_seeds": [20, 21],
        },
    }


def _calibration() -> dict[str, object]:
    videos = [
        {
            "stratum": "animation",
            "prompt_id": "p1",
            "seed_id": str(seed),
            "horizon_seconds": 15,
            "metrics": {
                "alignment": 0.8 + index * 0.01,
                "flicker": 0.02,
                "motion": 0.5,
            },
        }
        for index, seed in enumerate((10, 11, 12))
    ]
    return {
        "schema": CALIBRATION_SCHEMA,
        "protocol_digest": DIGEST,
        "teacher_videos": videos,
    }


def test_derivation_uses_teacher_center_spread_and_metric_direction() -> None:
    bundle = derive_policy_bundle(_protocol(), DIGEST, _calibration())
    quality = bundle["policies"]["animation"]["prompts"]["p1"]["horizons"]["15"][
        "quality_metrics"
    ]

    assert quality["alignment"]["teacher_center"] == pytest.approx(0.81)
    assert quality["alignment"]["teacher_robust_scale"] > 0
    assert "lower" in quality["alignment"] and "upper" not in quality["alignment"]
    assert "upper" in quality["flicker"] and "lower" not in quality["flicker"]
    assert "lower" in quality["motion"] and "upper" in quality["motion"]


def test_derivation_keeps_horizon_floors_separate() -> None:
    protocol = _protocol()
    protocol["horizons_seconds"] = [15, 60]
    protocol["horizon_specs"]["60"] = {
        "latent_frames": 243,
        "pixel_frames": 969,
        "playback_seconds": 60.5625,
    }
    calibration = _calibration()
    calibration["teacher_videos"].extend(
        {
            "stratum": "animation",
            "prompt_id": "p1",
            "seed_id": str(seed),
            "horizon_seconds": 60,
            "metrics": {
                "alignment": 0.6 + index * 0.01,
                "flicker": 0.04,
                "motion": 0.4,
            },
        }
        for index, seed in enumerate((10, 11, 12))
    )

    bundle = derive_policy_bundle(protocol, DIGEST, calibration)
    horizons = bundle["policies"]["animation"]["prompts"]["p1"]["horizons"]

    assert horizons["15"]["quality_metrics"]["alignment"]["teacher_center"] == pytest.approx(
        0.81
    )
    assert horizons["60"]["quality_metrics"]["alignment"]["teacher_center"] == pytest.approx(
        0.61
    )


def test_identity_tripwire_is_loose_but_load_bearing() -> None:
    bundle = derive_policy_bundle(_protocol(), DIGEST, _calibration())
    policy = bundle["policies"]["animation"]["prompts"]["p1"]["horizons"]["15"]
    intrinsic = {"alignment": 0.81, "flicker": 0.02, "motion": 0.5}

    passing = evaluate_layered_policy(
        intrinsic, {"pairwise_lpips_mean": 0.1}, policy
    )
    failing = evaluate_layered_policy(
        intrinsic, {"pairwise_lpips_mean": 0.2}, policy
    )
    assert passing["passed"] is True
    assert failing["passed"] is False
    assert failing["quality"]["passed"] is True
    assert failing["identity"]["passed"] is False


def test_layered_policy_fails_closed_for_nan() -> None:
    bundle = derive_policy_bundle(_protocol(), DIGEST, _calibration())
    policy = bundle["policies"]["animation"]["prompts"]["p1"]["horizons"]["15"]
    verdict = evaluate_layered_policy(
        {"alignment": float("nan"), "flicker": 0.02, "motion": 0.5},
        {"pairwise_lpips_mean": 0.1},
        policy,
    )
    assert verdict["passed"] is False
    assert verdict["failures"] == ["quality: alignment must be finite"]


def test_duplicate_seed_does_not_inflate_independent_sample_count() -> None:
    calibration = _calibration()
    calibration["teacher_videos"].append(
        copy.deepcopy(calibration["teacher_videos"][0])
    )
    with pytest.raises(ValueError, match="duplicate video"):
        derive_policy_bundle(_protocol(), DIGEST, calibration)


def test_insufficient_calibration_videos_fail_closed() -> None:
    calibration = _calibration()
    calibration["teacher_videos"] = calibration["teacher_videos"][:2]
    with pytest.raises(ValueError, match="requires 3"):
        derive_policy_bundle(_protocol(), DIGEST, calibration)


def test_holdout_requires_every_video_when_fraction_is_one() -> None:
    bundle = derive_policy_bundle(_protocol(), DIGEST, _calibration())
    candidate_videos = [
        {
            "stratum": "animation",
            "prompt_id": "p1",
            "seed_id": str(seed),
            "horizon_seconds": 15,
            "video_id": f"v{seed}",
            "intrinsic_metrics": {"alignment": 0.81, "flicker": 0.02, "motion": 0.5},
            "identity_metrics": {"pairwise_lpips_mean": identity},
        }
        for seed, identity in zip((20, 21), (0.1, 0.2), strict=True)
    ]
    holdout = {
        "schema": HOLDOUT_SCHEMA,
        "policy_bundle_digest": bundle["bundle_digest"],
        "candidate_videos": candidate_videos,
    }
    report = evaluate_holdout(bundle, holdout)

    assert report["passed"] is False
    prompt = report["strata"]["animation"]["prompts"]["p1"]
    assert prompt["horizons"]["15"]["pass_fraction"] == 0.5


def test_holdout_binding_prevents_retroactive_promotion() -> None:
    bundle = derive_policy_bundle(_protocol(), DIGEST, _calibration())
    with pytest.raises(ValueError, match="another policy bundle"):
        evaluate_holdout(
            bundle,
            {
                "schema": HOLDOUT_SCHEMA,
                "policy_bundle_digest": "sha256:other",
                "candidate_videos": [],
            },
        )
