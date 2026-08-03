from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from liveframe.quality_calibration import load_protocol
from liveframe.quality_capture import (
    RESULT_SCHEMA,
    calibration_sanity_report,
    ingest_calibration,
    load_batch_plan,
    load_holdout_candidate_batch_plan,
    load_holdout_reference_batch_plan,
    load_results,
    load_run_manifest,
    prepare_calibration_batch,
    prepare_holdout_candidate_batch,
    prepare_holdout_reference_batch,
    result_path,
    sha256_file,
    write_result,
)
from liveframe.quality_freeze import require_frozen_h100_reference_batch

ROOT = Path(__file__).parents[1]
PROTOCOL_PATH = ROOT / "artifacts" / "perceptual-protocol.v1.json"
MANIFEST_PATH = ROOT / "artifacts" / "perceptual-run-manifest.v1.json"


def _loaded() -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol, digest = load_protocol(PROTOCOL_PATH)
    manifest = load_run_manifest(MANIFEST_PATH, protocol_digest=digest)
    return protocol, digest, manifest


def _metric_values(index: int, *, degenerate_high_motion: bool = False) -> dict[str, float]:
    variation = (index % 8) * 0.005
    if degenerate_high_motion:
        variation = 0.0
    return {
        "clip_prompt_mean": 0.7 + variation,
        "clip_prompt_p10": 0.65 + variation,
        "motion_clip_distance_mean": 0.15 + variation,
        "motion_lpips_mean": 0.2 + variation,
        "temporal_delta_cosine_mean": 0.8 + variation,
        "luma_flicker_mean": 1.0 + variation,
    }


def _write_completed(
    root: Path,
    manifest: dict[str, Any],
    job: dict[str, Any],
    metrics: dict[str, float],
) -> Path:
    directory = root / job["run_id"]
    directory.mkdir(parents=True)
    pixels = directory / "pixels.bin"
    pixels.write_bytes(job["run_id"].encode())
    result = {
        "schema": RESULT_SCHEMA,
        "status": "completed",
        "protocol_digest": manifest["protocol_digest"],
        "manifest_digest": manifest["manifest_digest"],
        **{key: job[key] for key in job if key not in {"status"}},
        "metrics": metrics,
        "artifacts": [
            {
                "path": "pixels.bin",
                "bytes": pixels.stat().st_size,
                "sha256": sha256_file(pixels),
            }
        ],
        "runtime": {"kind": "synthetic-test"},
        "timing": {"wall_seconds": 1.0},
    }
    path = result_path(root, job["run_id"])
    write_result(path, result)
    return path


def test_manifest_loader_rejects_tampered_run_identity(tmp_path: Path) -> None:
    _, digest, _ = _loaded()
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    raw["jobs"][0]["seed"] += 1
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest digest mismatch"):
        load_run_manifest(path, protocol_digest=digest)


def test_prepare_is_calibration_only_and_restartable(tmp_path: Path) -> None:
    _, _, manifest = _loaded()
    first_job = next(job for job in manifest["jobs"] if job["split"] == "calibration")
    _write_completed(tmp_path, manifest, first_job, _metric_values(0))

    batch = prepare_calibration_batch(manifest, tmp_path, limit=3)

    assert len(batch["jobs"]) == 3
    assert all(
        job["split"] == "calibration"
        and job["role"] == "teacher"
        and job["backend"] == "cuda"
        for job in batch["jobs"]
    )
    assert first_job["run_id"] not in {job["run_id"] for job in batch["jobs"]}

    immutable = prepare_calibration_batch(
        manifest,
        tmp_path,
        run_ids={first_job["run_id"]},
        include_completed=True,
    )
    assert [job["run_id"] for job in immutable["jobs"]] == [first_job["run_id"]]


def test_prepare_preserves_portable_relative_output_root() -> None:
    _, _, manifest = _loaded()

    batch = prepare_calibration_batch(manifest, Path("artifacts/remote-results"), limit=1)

    assert batch["output_root"] == "artifacts/remote-results"


def test_prepare_holdout_candidates_is_mlx_only(tmp_path: Path) -> None:
    protocol, _, manifest = _loaded()

    batch = prepare_holdout_candidate_batch(
        manifest,
        tmp_path,
        limit=4,
        required_metrics=set(protocol["quality_metrics"]),
    )

    assert len(batch["jobs"]) == 4
    assert all(
        job["split"] == "holdout"
        and job["role"] == "candidate"
        and job["backend"] == "mlx"
        for job in batch["jobs"]
    )


def test_holdout_candidate_batch_rejects_reference_substitution(tmp_path: Path) -> None:
    _, _, manifest = _loaded()
    batch = prepare_holdout_candidate_batch(manifest, tmp_path, limit=1)
    reference = next(
        job
        for job in manifest["jobs"]
        if job["split"] == "holdout" and job["role"] == "reference"
    )
    batch["jobs"] = [reference]
    unsigned = {key: value for key, value in batch.items() if key != "batch_digest"}
    from liveframe.perceptual_policy import canonical_digest

    batch["batch_digest"] = canonical_digest(unsigned)
    path = tmp_path / "candidate-batch.json"
    path.write_text(json.dumps(batch), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown job"):
        load_holdout_candidate_batch_plan(path, manifest)


def test_holdout_reference_batch_is_cuda_only(tmp_path: Path) -> None:
    _, _, manifest = _loaded()
    batch = prepare_holdout_reference_batch(manifest, tmp_path)
    path = tmp_path / "reference-batch.json"
    path.write_text(json.dumps(batch), encoding="utf-8")

    loaded = load_holdout_reference_batch_plan(path, manifest)

    assert len(loaded["jobs"]) == 128
    assert all(
        job["split"] == "holdout"
        and job["role"] == "reference"
        and job["backend"] == "cuda"
        for job in loaded["jobs"]
    )


def test_frozen_h100_reference_batch_is_pinned() -> None:
    batch = json.loads(
        (ROOT / "artifacts" / "h100-holdout-references.v1.json").read_text(
            encoding="utf-8"
        )
    )

    require_frozen_h100_reference_batch(batch)


def test_batch_plan_rejects_an_altered_prompt(tmp_path: Path) -> None:
    _, _, manifest = _loaded()
    batch = prepare_calibration_batch(manifest, tmp_path, limit=1)
    batch["jobs"][0]["prompt"] = "altered after freeze"
    path = tmp_path / "batch.json"
    path.write_text(json.dumps(batch), encoding="utf-8")

    with pytest.raises(ValueError, match="batch digest mismatch"):
        load_batch_plan(path, manifest)


def test_prepare_rejects_corrupt_completed_artifact(tmp_path: Path) -> None:
    _, _, manifest = _loaded()
    job = next(job for job in manifest["jobs"] if job["split"] == "calibration")
    path = _write_completed(tmp_path, manifest, job, _metric_values(0))
    (path.parent / "pixels.bin").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_calibration_batch(manifest, tmp_path, limit=1)


def test_prepare_rejects_completed_result_with_wrong_metrics(tmp_path: Path) -> None:
    protocol, _, manifest = _loaded()
    job = next(job for job in manifest["jobs"] if job["split"] == "calibration")
    _write_completed(tmp_path, manifest, job, {"wrong": 1.0})

    with pytest.raises(ValueError, match="metric set mismatch"):
        prepare_calibration_batch(
            manifest,
            tmp_path,
            limit=1,
            required_metrics=set(protocol["quality_metrics"]),
        )


def test_ingestion_requires_every_frozen_calibration_run(tmp_path: Path) -> None:
    protocol, _, manifest = _loaded()
    jobs = [job for job in manifest["jobs"] if job["split"] == "calibration"]
    path = _write_completed(tmp_path, manifest, jobs[0], _metric_values(0))

    with pytest.raises(ValueError, match="255 missing"):
        ingest_calibration(protocol, manifest, load_results([path]))


def test_complete_ingestion_and_sanity_report(tmp_path: Path) -> None:
    protocol, _, manifest = _loaded()
    jobs = [job for job in manifest["jobs"] if job["split"] == "calibration"]
    paths = [
        _write_completed(tmp_path, manifest, job, _metric_values(index))
        for index, job in enumerate(jobs)
    ]

    calibration = ingest_calibration(protocol, manifest, load_results(paths))
    sanity = calibration_sanity_report(protocol, calibration)

    assert calibration["complete"] is True
    assert len(calibration["teacher_videos"]) == 256
    assert sanity["passed"] is True
    assert len(sanity["cells"]) == 4 * 4 * 2 * 6


def test_sanity_rejects_degenerate_high_motion_spread(tmp_path: Path) -> None:
    protocol, _, manifest = _loaded()
    jobs = [job for job in manifest["jobs"] if job["split"] == "calibration"]
    paths = []
    for index, job in enumerate(jobs):
        degenerate = job["stratum"] == "high-motion-dense-scenes"
        paths.append(
            _write_completed(
                tmp_path,
                manifest,
                job,
                _metric_values(index, degenerate_high_motion=degenerate),
            )
        )
    calibration = ingest_calibration(protocol, manifest, load_results(paths))

    sanity = calibration_sanity_report(protocol, calibration)

    assert sanity["passed"] is False
    assert any("high-motion teacher spread is degenerate" in row for row in sanity["failures"])


def test_write_result_is_idempotent_but_immutable(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    result = {"schema": RESULT_SCHEMA, "status": "completed"}
    write_result(path, result)
    write_result(path, result)

    altered = copy.deepcopy(result)
    altered["status"] = "failed"
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_result(path, altered)
