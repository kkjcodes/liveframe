from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from liveframe.quality_artifacts import StochasticWriter  # noqa: E402
from liveframe.quality_calibration import HOLDOUT_SCHEMA, load_protocol  # noqa: E402
from liveframe.quality_capture import RESULT_SCHEMA, load_run_manifest, sha256_file  # noqa: E402
from liveframe.quality_holdout import (  # noqa: E402
    IDENTITY_METRICS,
    PAIRWISE_REPORT_SCHEMA,
    _payloads_equal,
    build_holdout_input,
    pair_holdout_results,
)
from liveframe.quality_stochastic import STOCHASTIC_CONTRACT  # noqa: E402

ROOT = Path(__file__).parents[1]


def _loaded() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol, digest = load_protocol(ROOT / "artifacts" / "perceptual-protocol.v1.json")
    manifest = load_run_manifest(
        ROOT / "artifacts" / "perceptual-run-manifest.v1.json",
        protocol_digest=digest,
    )
    first = next(job for job in manifest["jobs"] if job["split"] == "holdout")
    jobs = [
        job
        for job in manifest["jobs"]
        if job["split"] == "holdout"
        and all(
            job[name] == first[name]
            for name in ("stratum", "prompt_id", "seed", "horizon_seconds")
        )
    ]
    return protocol, manifest, jobs


def _result(
    root: Path,
    manifest: dict[str, Any],
    job: dict[str, Any],
    *,
    stochastic_value: float = 0.0,
) -> tuple[Path, dict[str, Any]]:
    directory = root / job["run_id"]
    directory.mkdir(parents=True)
    pixels = directory / "pixels.bin"
    pixels.write_bytes(job["run_id"].encode())
    writer = StochasticWriter(directory)
    writer.append(
        "initial_noise",
        torch.full((1, 2), stochastic_value, dtype=torch.bfloat16),
    )
    stochastic_paths = writer.finish(
        seed=int(job["seed"]),
        contract=STOCHASTIC_CONTRACT,
        latent_frames=int(job["latent_frames"]),
    )
    result = {
        **job,
        "schema": RESULT_SCHEMA,
        "status": "completed",
        "protocol_digest": manifest["protocol_digest"],
        "manifest_digest": manifest["manifest_digest"],
        "metrics": {
            "clip_prompt_mean": 0.4,
            "clip_prompt_p10": 0.3,
            "motion_clip_distance_mean": 0.02,
            "motion_lpips_mean": 0.1,
            "temporal_delta_cosine_mean": 0.0,
            "luma_flicker_mean": 0.5,
        },
        "artifacts": [
            {
                "path": "pixels.bin",
                "bytes": pixels.stat().st_size,
                "sha256": sha256_file(pixels),
            },
            *[
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in stochastic_paths
            ],
        ],
    }
    return directory / "result.json", result


def test_pairs_exact_cuda_reference_and_mlx_candidate(tmp_path: Path) -> None:
    protocol, manifest, jobs = _loaded()
    results = [_result(tmp_path, manifest, job) for job in jobs]

    pairs = pair_holdout_results(protocol, manifest, results, require_complete=False)

    assert len(pairs) == 1
    assert pairs[0].reference_job["backend"] == "cuda"
    assert pairs[0].candidate_job["backend"] == "mlx"
    assert pairs[0].reference_pixels.name == "pixels.bin"
    assert pairs[0].stochastic_payload_sha256.startswith("sha256:")


def test_rejects_individually_valid_stochastic_mismatch(tmp_path: Path) -> None:
    protocol, manifest, jobs = _loaded()
    results = [
        _result(
            tmp_path,
            manifest,
            job,
            stochastic_value=0.0 if job["role"] == "reference" else 1.0,
        )
        for job in jobs
    ]

    with pytest.raises(ValueError, match="stochastic tensors mismatch"):
        pair_holdout_results(protocol, manifest, results, require_complete=False)


def test_byte_identity_gate_compares_literal_payloads(tmp_path: Path) -> None:
    reference = tmp_path / "reference.bf16"
    candidate = tmp_path / "candidate.bf16"
    reference.write_bytes(b"a" * (1024 * 1024 + 1))
    candidate.write_bytes(reference.read_bytes())

    assert _payloads_equal(reference, candidate)
    candidate.write_bytes(b"a" * (1024 * 1024) + b"b")
    assert not _payloads_equal(reference, candidate)


def test_rejects_incomplete_result_pair(tmp_path: Path) -> None:
    protocol, manifest, jobs = _loaded()
    result = _result(tmp_path, manifest, jobs[0])

    with pytest.raises(ValueError, match="result pair incomplete"):
        pair_holdout_results(protocol, manifest, [result], require_complete=False)


def test_rejects_backend_substitution(tmp_path: Path) -> None:
    protocol, manifest, jobs = _loaded()
    results = [_result(tmp_path, manifest, job) for job in jobs]
    candidate = next(result for _, result in results if result["role"] == "candidate")
    candidate["backend"] = "cuda"

    with pytest.raises(ValueError, match="backend mismatch"):
        pair_holdout_results(protocol, manifest, results, require_complete=False)


def test_rejects_duplicate_and_tampered_results(tmp_path: Path) -> None:
    protocol, manifest, jobs = _loaded()
    results = [_result(tmp_path, manifest, job) for job in jobs]
    with pytest.raises(ValueError, match="duplicate holdout result"):
        pair_holdout_results(
            protocol, manifest, [*results, copy.deepcopy(results[0])], require_complete=False
        )

    results[0][0].parent.joinpath("pixels.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        pair_holdout_results(protocol, manifest, results, require_complete=False)


def _report(protocol: dict[str, Any], pair: Any) -> dict[str, Any]:
    return {
        "schema": PAIRWISE_REPORT_SCHEMA,
        "reference_run_id": pair.reference_job["run_id"],
        "candidate_run_id": pair.candidate_job["run_id"],
        "stochastic_payload_sha256": pair.stochastic_payload_sha256,
        "metric_provenance_digest": protocol["metric_provenance"]["artifact_sha256"],
        "pixel_artifacts": {
            "reference_sha256": next(
                row["sha256"]
                for row in pair.reference_result["artifacts"]
                if row["path"] == "pixels.bin"
            ),
            "candidate_sha256": next(
                row["sha256"]
                for row in pair.candidate_result["artifacts"]
                if row["path"] == "pixels.bin"
            ),
        },
        "identity_metrics": dict.fromkeys(IDENTITY_METRICS, 0.01),
    }


def test_builds_policy_bound_holdout_input(tmp_path: Path) -> None:
    protocol, manifest, jobs = _loaded()
    results = [_result(tmp_path, manifest, job) for job in jobs]
    pair = pair_holdout_results(protocol, manifest, results, require_complete=False)[0]
    bundle = {"bundle_digest": "sha256:frozen-v2"}

    holdout = build_holdout_input(protocol, bundle, manifest, [pair], [_report(protocol, pair)])

    assert holdout["schema"] == HOLDOUT_SCHEMA
    assert holdout["policy_bundle_digest"] == "sha256:frozen-v2"
    assert holdout["capture_protocol_digest"] == manifest["protocol_digest"]
    assert holdout["candidate_videos"][0]["intrinsic_metrics"] == pair.candidate_result[
        "metrics"
    ]


def test_holdout_input_rejects_missing_or_unbound_pairwise_report(tmp_path: Path) -> None:
    protocol, manifest, jobs = _loaded()
    results = [_result(tmp_path, manifest, job) for job in jobs]
    pair = pair_holdout_results(protocol, manifest, results, require_complete=False)[0]
    bundle = {"bundle_digest": "sha256:frozen-v2"}

    with pytest.raises(ValueError, match="missing pairwise report"):
        build_holdout_input(protocol, bundle, manifest, [pair], [])

    report = _report(protocol, pair)
    report["pixel_artifacts"]["candidate_sha256"] = "sha256:wrong"
    with pytest.raises(ValueError, match="pixel checksum mismatch"):
        build_holdout_input(protocol, bundle, manifest, [pair], [report])