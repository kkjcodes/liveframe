from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from liveframe.quality_calibration import load_protocol
from liveframe.quality_capture import (
    RESULT_SCHEMA,
    load_run_manifest,
    sha256_file,
)
from liveframe.quality_checkpoint import (
    evaluate_first_seed_checkpoint,
    first_seed_run_ids,
)

ROOT = Path(__file__).parents[1]


def _loaded() -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol, digest = load_protocol(ROOT / "artifacts" / "perceptual-protocol.v1.json")
    manifest = load_run_manifest(
        ROOT / "artifacts" / "perceptual-run-manifest.v1.json",
        protocol_digest=digest,
    )
    return protocol, digest, manifest


def _results(
    root: Path,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    *,
    wall_seconds: float = 1.0,
) -> list[tuple[Path, dict[str, Any]]]:
    expected = first_seed_run_ids(protocol, manifest)
    jobs = {job["run_id"]: job for job in manifest["jobs"]}
    results = []
    for run_id in sorted(expected):
        job = jobs[run_id]
        directory = root / run_id
        directory.mkdir(parents=True)
        pixels = directory / "pixels.bin"
        pixels.write_bytes(run_id.encode())
        metadata = directory / "pixels.bin.json"
        metadata.write_text(
            json.dumps(
                {
                    "schema": "liveframe.decoded-pixels.v1",
                    "dtype": "uint8",
                    "layout": "T,H,W,C",
                    "shape": [job["pixel_frames"], 480, 832, 3],
                }
            ),
            encoding="utf-8",
        )
        artifacts = [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (pixels, metadata)
        ]
        result = {
            **job,
            "schema": RESULT_SCHEMA,
            "status": "completed",
            "protocol_digest": manifest["protocol_digest"],
            "manifest_digest": manifest["manifest_digest"],
            "metrics": {name: 0.1 for name in protocol["quality_metrics"]},
            "artifacts": artifacts,
            "runtime": {"backend": "synthetic"},
            "timing": {"executor_wall_seconds": wall_seconds},
        }
        path = directory / "result.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        results.append((path, result))
    return results


def test_first_seed_checkpoint_passes_complete_finite_results(tmp_path: Path) -> None:
    protocol, digest, manifest = _loaded()

    report = evaluate_first_seed_checkpoint(
        protocol, digest, manifest, _results(tmp_path, protocol, manifest)
    )

    assert report["passed"] is True
    assert report["validated_jobs"] == 32
    assert report["projected_executor_hours"] < report["maximum_projected_hours"]


def test_first_seed_checkpoint_rejects_missing_result(tmp_path: Path) -> None:
    protocol, digest, manifest = _loaded()
    results = _results(tmp_path, protocol, manifest)

    with pytest.raises(ValueError, match="1 missing"):
        evaluate_first_seed_checkpoint(protocol, digest, manifest, results[:-1])


def test_first_seed_checkpoint_fails_excessive_time_projection(tmp_path: Path) -> None:
    protocol, digest, manifest = _loaded()
    results = _results(tmp_path, protocol, manifest, wall_seconds=1000.0)

    report = evaluate_first_seed_checkpoint(protocol, digest, manifest, results)

    assert report["passed"] is False
    assert any("projected executor time" in failure for failure in report["failures"])
