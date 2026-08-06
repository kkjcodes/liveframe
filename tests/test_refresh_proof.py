from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from liveframe.perceptual_policy import canonical_digest
from liveframe.quality_calibration import load_protocol
from liveframe.quality_capture import sha256_file
from liveframe.refresh_capture import (
    EVIDENCE_POLICY,
    EVIDENCE_POLICY_DIGEST,
    H100_EVIDENCE,
    IDENTITY_METRICS,
    QUALITY_METRICS,
    REQUIRED_ARTIFACTS,
    SYNTHETIC_EVIDENCE,
    TIMING_FIELDS,
    build_capture_result,
    evidence_eligibility,
    load_capture_batch,
    load_capture_results,
    prepare_capture_batch,
    publish_capture_result,
    validate_capture_result,
    validate_causal_source_contract,
)
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
    is_full_refresh_frame,
    load_refresh_proof_manifest,
    reclaim_accounting,
    source_frame_for_destination,
    validate_refresh_proof_result,
    validate_substitution_ancestry,
)

ROOT = Path(__file__).parents[1]
PROTOCOL_PATH = ROOT / "artifacts" / "perceptual-protocol.v1.json"
BUNDLE_PATH = ROOT / "artifacts" / "perceptual-policy-bundle.v2.json"
MANIFEST_PATH = ROOT / "artifacts" / "refresh-addressability-manifest.v2.json"
H100_BATCH_PATH = ROOT / "artifacts" / "h100-refresh-proof-batch.v2.json"
EVIDENCE_POLICY_PATH = ROOT / "artifacts" / "refresh-evidence-eligibility.v1.json"


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
        "tile_index": 0,
        "source_start_frame": 0,
        "destination_start_frame": 2,
        "frame_count": 1,
        "token_count": 1560,
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
    window_count = job["latent_frames"]
    def point(window: int) -> dict[str, float | int]:
        return {
            "window_index": window,
            "playback_seconds": float(window),
            "cosine": 1.0,
            "rmse": 0.0,
            "max_abs": 0.0,
        }

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
            "future_latent": [point(window) for window in range(window_count)],
            "decoded-pixel": [point(window) for window in range(window_count)],
            "per-layer-key": [
                {**point(window), "layer_index": layer}
                for window in range(window_count)
                for layer in range(30)
            ],
            "per-layer-value": [
                {**point(window), "layer_index": layer}
                for window in range(window_count)
                for layer in range(30)
            ],
        },
        "substitutions": [
            {
                **_ancestry(),
                "layer_index": layer,
                "tile_index": source_frame_for_destination(destination),
                "source_start_frame": source_frame_for_destination(destination),
                "destination_start_frame": destination,
            }
            for destination in range(1, window_count)
            if not is_full_refresh_frame(destination)
            for layer in range(30)
        ],
    }


def _captured_result(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    result = _result(manifest)
    artifacts = []
    for name in sorted(REQUIRED_ARTIFACTS):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        artifacts.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    stochastic_digest = next(
        row["sha256"] for row in artifacts if row["path"] == "stochastic.bf16"
    )
    result.update(
        {
            "status": "completed",
            "evidence_class": SYNTHETIC_EVIDENCE,
            "stochastic_pairing": {
                "identity": "same-seed-same-bytes",
                "reference_sha256": stochastic_digest,
                "candidate_sha256": stochastic_digest,
            },
            "intrinsic_metrics": dict.fromkeys(QUALITY_METRICS, 0.0),
            "identity_metrics": dict.fromkeys(IDENTITY_METRICS, 0.0),
            "runtime": {
                "evidence_class": SYNTHETIC_EVIDENCE,
                "harness_sha256": "sha256:" + "0" * 64,
            },
            "timing": dict.fromkeys(TIMING_FIELDS, 0.0),
            "artifacts": artifacts,
        }
    )
    result["eligibility"] = evidence_eligibility(
        result["evidence_class"], result["runtime"]
    )
    result["result_digest"] = canonical_digest(result)
    return result


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

    incomplete_ancestry = copy.deepcopy(result)
    incomplete_ancestry["substitutions"].pop()
    with pytest.raises(ValueError, match="substitution ancestry is incomplete"):
        validate_refresh_proof_result(incomplete_ancestry, manifest)

    invalid = copy.deepcopy(_ancestry())
    invalid["destination_start_frame"] = invalid["source_start_frame"]
    with pytest.raises(ValueError, match="destination must follow"):
        validate_substitution_ancestry(invalid)

    wrong_anchor = copy.deepcopy(_ancestry())
    wrong_anchor["source_start_frame"] = 1
    wrong_anchor["tile_index"] = 1
    with pytest.raises(ValueError, match="latest periodic anchor"):
        validate_substitution_ancestry(wrong_anchor)

    scheduled_refresh = copy.deepcopy(_ancestry())
    scheduled_refresh["destination_start_frame"] = 3
    with pytest.raises(ValueError, match="scheduled full refresh"):
        validate_substitution_ancestry(scheduled_refresh)


def test_capture_result_round_trip_is_byte_bound_and_not_claim_eligible(
    tmp_path: Path,
) -> None:
    protocol, digest, bundle = _inputs()
    manifest = build_refresh_proof_manifest(protocol, digest, bundle)
    result_dir = tmp_path / str(_result(manifest)["run_id"])
    result = _captured_result(result_dir, manifest)
    path = result_dir / "result.json"

    publish_capture_result(path, result, manifest)
    loaded = load_capture_results([path], manifest)

    assert loaded == [result]
    with pytest.raises(ValueError, match="not numerically eligible"):
        load_capture_results([path], manifest, require_numerical_eligible=True)


def test_capture_result_rejects_tampering_and_duplicate_results(tmp_path: Path) -> None:
    protocol, digest, bundle = _inputs()
    manifest = build_refresh_proof_manifest(protocol, digest, bundle)
    result_dir = tmp_path / str(_result(manifest)["run_id"])
    result = _captured_result(result_dir, manifest)
    path = result_dir / "result.json"
    publish_capture_result(path, result, manifest)

    (result_dir / "candidate-latents.bf16").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact mismatch"):
        load_capture_results([path], manifest)

    (result_dir / "candidate-latents.bf16").write_bytes(b"candidate-latents.bf16")
    with pytest.raises(ValueError, match="duplicate refresh result"):
        load_capture_results([path, path], manifest)


def test_capture_result_requires_stochastic_byte_identity(tmp_path: Path) -> None:
    protocol, digest, bundle = _inputs()
    manifest = build_refresh_proof_manifest(protocol, digest, bundle)
    result_dir = tmp_path / str(_result(manifest)["run_id"])
    result = _captured_result(result_dir, manifest)
    result["stochastic_pairing"]["candidate_sha256"] = "sha256:" + "f" * 64
    result["result_digest"] = canonical_digest(
        {key: value for key, value in result.items() if key != "result_digest"}
    )

    with pytest.raises(ValueError, match="stochastic bytes differ"):
        validate_capture_result(result, manifest, root=result_dir)


def test_capture_result_builder_binds_job_and_artifact_bytes(tmp_path: Path) -> None:
    protocol, digest, bundle = _inputs()
    manifest = build_refresh_proof_manifest(protocol, digest, bundle)
    job = next(job for job in manifest["jobs"] if job["horizon_seconds"] == 15)
    result_dir = tmp_path / str(job["run_id"])
    seed_result = _captured_result(result_dir, manifest)

    built = build_capture_result(
        job,
        manifest,
        evidence_class=SYNTHETIC_EVIDENCE,
        drift_curves=seed_result["drift_curves"],
        substitutions=seed_result["substitutions"],
        intrinsic_metrics=seed_result["intrinsic_metrics"],
        identity_metrics=seed_result["identity_metrics"],
        timing=seed_result["timing"],
        runtime={"harness_sha256": "sha256:" + "0" * 64},
        artifact_root=result_dir,
    )

    assert built["run_id"] == job["run_id"]
    assert built["prompt"] == job["prompt"]
    assert built["result_digest"].startswith("sha256:")


def test_h100_batch_is_digest_bound_and_preserves_literal_jobs(tmp_path: Path) -> None:
    protocol, digest, bundle = _inputs()
    manifest = build_refresh_proof_manifest(protocol, digest, bundle)
    batch = prepare_capture_batch(manifest, Path("artifacts/refresh-results"), limit=2)
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(batch), encoding="utf-8")

    loaded = load_capture_batch(path, manifest)

    assert loaded == batch
    assert loaded["evidence_class"] == "h100-prospective"
    assert loaded["jobs"] == manifest["jobs"][:2]


def test_h100_batch_rejects_tampering_duplicates_and_unknown_ids(tmp_path: Path) -> None:
    protocol, digest, bundle = _inputs()
    manifest = build_refresh_proof_manifest(protocol, digest, bundle)
    batch = prepare_capture_batch(manifest, tmp_path, limit=2)

    altered = copy.deepcopy(batch)
    altered["jobs"][0]["prompt"] = "changed after freeze"
    path = tmp_path / "altered.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="batch digest mismatch"):
        load_capture_batch(path, manifest)

    duplicated = copy.deepcopy(batch)
    duplicated["jobs"] = [duplicated["jobs"][0], duplicated["jobs"][0]]
    duplicated["batch_digest"] = canonical_digest(
        {key: value for key, value in duplicated.items() if key != "batch_digest"}
    )
    path.write_text(json.dumps(duplicated), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates job"):
        load_capture_batch(path, manifest)

    with pytest.raises(ValueError, match="unknown run IDs"):
        prepare_capture_batch(manifest, tmp_path, run_ids={"unknown"})


def test_frozen_h100_batch_matches_all_manifest_jobs() -> None:
    protocol, digest, bundle = _inputs()
    manifest = build_refresh_proof_manifest(protocol, digest, bundle)

    batch = load_capture_batch(H100_BATCH_PATH, manifest)

    assert batch == prepare_capture_batch(
        manifest,
        Path("artifacts/h100-refresh-proof-results-v2"),
        include_completed=True,
    )
    assert len(batch["jobs"]) == 64


def test_frozen_evidence_policy_matches_code() -> None:
    policy = json.loads(EVIDENCE_POLICY_PATH.read_text(encoding="utf-8"))

    assert policy == EVIDENCE_POLICY
    assert canonical_digest(policy) == EVIDENCE_POLICY_DIGEST


def test_driver_mismatch_preserves_numerical_but_rejects_timing_eligibility() -> None:
    runtime = {
        **EVIDENCE_POLICY["numerical"],
        "harness_sha256": "sha256:" + "a" * 64,
        "identity_anchor_report_sha256": "sha256:" + "b" * 64,
        "driver": "570.124.06",
        "torch": EVIDENCE_POLICY["timing"]["torch"],
        "cuda": EVIDENCE_POLICY["timing"]["cuda"],
        "timing_protocol": EVIDENCE_POLICY["timing"]["timing_protocol"],
        "denominator_sha256": EVIDENCE_POLICY["timing"]["denominator_sha256"],
    }

    eligibility = evidence_eligibility(H100_EVIDENCE, runtime)

    assert eligibility["numerical_eligible"] is True
    assert eligibility["timing_eligible"] is False
    assert eligibility["timing_failures"] == [
        "runtime driver does not match timing policy"
    ]


def test_exact_denominator_runtime_is_timing_eligible() -> None:
    runtime = {
        **EVIDENCE_POLICY["numerical"],
        "harness_sha256": "sha256:" + "a" * 64,
        "identity_anchor_report_sha256": "sha256:" + "b" * 64,
        **{
            name: EVIDENCE_POLICY["timing"][name]
            for name in (
                "driver",
                "torch",
                "cuda",
                "timing_protocol",
                "denominator_sha256",
            )
        },
    }

    eligibility = evidence_eligibility(H100_EVIDENCE, runtime)

    assert eligibility["numerical_eligible"] is True
    assert eligibility["timing_eligible"] is True


def test_causal_source_contract_requires_cache_and_rerope_surfaces() -> None:
    source = (
        "class CausalWanSelfAttention causal_rope_apply cache_start "
        "global_end_index local_end_index"
    )
    validate_causal_source_contract(source)

    with pytest.raises(ValueError, match="causal_rope_apply"):
        validate_causal_source_contract(
            "class CausalWanSelfAttention cache_start global_end_index local_end_index"
        )

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