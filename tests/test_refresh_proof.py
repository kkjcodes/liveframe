from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from liveframe.quality_calibration import load_protocol
from liveframe.refresh_proof import (
    DENOMINATOR_SHA256,
    HORIZONS,
    MECHANISM_ID,
    POLICY_BUNDLE_DIGEST,
    PROOF_SEEDS,
    PROOF_STRATA,
    RESULT_SCHEMA,
    SUBSTITUTION_SCHEMA,
    build_refresh_proof_manifest,
    load_refresh_proof_manifest,
    reclaim_accounting,
    validate_refresh_proof_result,
    validate_substitution_ancestry,
)

ROOT = Path(__file__).parents[1]
PROTOCOL_PATH = ROOT / "artifacts" / "perceptual-protocol.v1.json"
BUNDLE_PATH = ROOT / "artifacts" / "perceptual-policy-bundle.v2.json"
MANIFEST_PATH = ROOT / "artifacts" / "refresh-addressability-manifest.v1.json"


def _inputs() -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol, digest = load_protocol(PROTOCOL_PATH)
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    return protocol, digest, bundle


def _ancestry() -> dict[str, Any]:
    digest = "sha256:" + "a" * 64
    return {
        "schema": SUBSTITUTION_SCHEMA,
        "mechanism_id": MECHANISM_ID,
        "layer_index": 15,
        "tile_index": 2,
        "source_start_frame": 9,
        "destination_start_frame": 12,
        "frame_count": 3,
        "token_count": 4680,
        "source_key_form": "raw-pre-rope",
        "candidate_key_form": "reroped-from-source-raw-key",
        "value_policy": "reuse-source-value",
        "source_raw_key_sha256": digest,
        "source_value_sha256": digest,
        "candidate_reroped_key_sha256": digest,
        "reference_destination_key_sha256": digest,
        "reference_destination_value_sha256": digest,
    }


def _result(manifest: dict[str, Any]) -> dict[str, Any]:
    job = next(job for job in manifest["jobs"] if job["horizon_seconds"] == 15)
    window_count = job["latent_frames"] // 3 + 3
    return {
        "schema": RESULT_SCHEMA,
        "manifest_digest": manifest["manifest_digest"],
        **{
            name: job[name]
            for name in (
                "run_id",
                "stratum",
                "prompt_id",
                "seed",
                "horizon_seconds",
                "latent_frames",
            )
        },
        "drift_curves": {
            "future_latent": [
                {"window_index": window} for window in range(window_count)
            ],
            "decoded-pixel": [
                {"window_index": window} for window in range(window_count)
            ],
            "per-layer-key": [
                {"window_index": window, "layer_index": layer}
                for window in range(window_count)
                for layer in range(30)
            ],
            "per-layer-value": [
                {"window_index": window, "layer_index": layer}
                for window in range(window_count)
                for layer in range(30)
            ],
        },
        "substitutions": [_ancestry()],
    }


def test_manifest_is_deterministic_complete_and_prospective() -> None:
    protocol, digest, bundle = _inputs()

    manifest = build_refresh_proof_manifest(protocol, digest, bundle)

    assert manifest == build_refresh_proof_manifest(protocol, digest, bundle)
    assert manifest["job_count"] == 64
    assert manifest["denominator"]["sha256"] == DENOMINATOR_SHA256
    assert manifest["policy_bundle_digest"] == POLICY_BUNDLE_DIGEST
    assert manifest["sampling"]["strata"] == list(PROOF_STRATA)
    assert manifest["sampling"]["horizons_seconds"] == list(HORIZONS)
    assert {job["seed"] for job in manifest["jobs"]} == set(PROOF_SEEDS)
    excluded = {
        *manifest["sampling"]["excluded_calibration_seeds"],
        *manifest["sampling"]["excluded_holdout_seeds"],
    }
    assert not excluded.intersection(PROOF_SEEDS)
    cells = {
        (job["stratum"], job["prompt_id"], job["seed"], job["horizon_seconds"])
        for job in manifest["jobs"]
    }
    assert len(cells) == 64
    assert all(job["stochastic_pairing"] == "same-seed-same-bytes" for job in manifest["jobs"])


def test_frozen_manifest_matches_the_deterministic_campaign() -> None:
    protocol, digest, bundle = _inputs()

    manifest = load_refresh_proof_manifest(
        MANIFEST_PATH, protocol, digest, bundle
    )

    assert manifest == build_refresh_proof_manifest(protocol, digest, bundle)


def test_manifest_loader_rejects_any_post_freeze_change(tmp_path: Path) -> None:
    protocol, digest, bundle = _inputs()
    manifest = build_refresh_proof_manifest(protocol, digest, bundle)
    manifest["jobs"][0]["seed"] += 1
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from the frozen campaign"):
        load_refresh_proof_manifest(path, protocol, digest, bundle)


def test_substitution_ancestry_requires_raw_key_value_and_forward_position() -> None:
    validate_substitution_ancestry(_ancestry())

    invalid = copy.deepcopy(_ancestry())
    invalid["source_key_form"] = "already-roped"
    with pytest.raises(ValueError, match="raw pre-RoPE"):
        validate_substitution_ancestry(invalid)


def test_result_requires_complete_drift_curves_and_substitution_ancestry() -> None:
    protocol, digest, bundle = _inputs()
    manifest = build_refresh_proof_manifest(protocol, digest, bundle)
    result = _result(manifest)

    validate_refresh_proof_result(result, manifest)

    incomplete = copy.deepcopy(result)
    incomplete["drift_curves"]["per-layer-key"].pop()
    with pytest.raises(ValueError, match="per-layer-key drift curve is incomplete"):
        validate_refresh_proof_result(incomplete, manifest)

    incomplete = copy.deepcopy(result)
    incomplete["drift_curves"]["decoded-pixel"].pop()
    with pytest.raises(ValueError, match="decoded-pixel drift curve is incomplete"):
        validate_refresh_proof_result(incomplete, manifest)

    missing_ancestry = copy.deepcopy(result)
    missing_ancestry["substitutions"] = []
    with pytest.raises(ValueError, match="requires substitution ancestry"):
        validate_refresh_proof_result(missing_ancestry, manifest)

    invalid = copy.deepcopy(_ancestry())
    invalid["destination_start_frame"] = invalid["source_start_frame"]
    with pytest.raises(ValueError, match="destination must follow"):
        validate_substitution_ancestry(invalid)

    invalid = copy.deepcopy(_ancestry())
    invalid["source_value_sha256"] = "missing"
    with pytest.raises(ValueError, match="source_value_sha256"):
        validate_substitution_ancestry(invalid)


def test_compute_floor_does_not_rescue_wall_floor() -> None:
    result = reclaim_accounting(
        denoise_reclaim_fraction=0.31,
        refresh_reclaim_fraction=0.31,
        overhead_seconds=0.0,
    )

    assert result["compute_floor_passed"] is True
    assert result["complete_wall_reduction_fraction"] == pytest.approx(0.1355066)
    assert result["wall_floor_passed"] is False
    assert result["passed"] is False


def test_combined_reclaim_must_cover_overhead() -> None:
    zero_overhead = reclaim_accounting(
        denoise_reclaim_fraction=0.39,
        refresh_reclaim_fraction=0.39,
        overhead_seconds=0.0,
    )
    with_overhead = reclaim_accounting(
        denoise_reclaim_fraction=0.39,
        refresh_reclaim_fraction=0.39,
        overhead_seconds=0.02,
    )

    assert zero_overhead["passed"] is True
    assert with_overhead["compute_floor_passed"] is True
    assert with_overhead["wall_floor_passed"] is False
    assert with_overhead["passed"] is False


@pytest.mark.parametrize(
    ("denoise", "refresh", "overhead"),
    [(-0.1, 0.0, 0.0), (0.0, 1.1, 0.0), (0.0, 0.0, -0.1)],
)
def test_reclaim_accounting_is_fail_closed(
    denoise: float, refresh: float, overhead: float
) -> None:
    with pytest.raises(ValueError):
        reclaim_accounting(
            denoise_reclaim_fraction=denoise,
            refresh_reclaim_fraction=refresh,
            overhead_seconds=overhead,
        )