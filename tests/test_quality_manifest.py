from __future__ import annotations

import json
from pathlib import Path

import pytest

from liveframe.quality_calibration import load_protocol
from liveframe.quality_manifest import build_run_manifest

ROOT = Path(__file__).parents[1]
PROTOCOL_PATH = ROOT / "artifacts" / "perceptual-protocol.v1.json"


def test_frozen_protocol_and_manifest_are_deterministic() -> None:
    protocol, digest = load_protocol(PROTOCOL_PATH)

    first = build_run_manifest(protocol, digest)
    second = build_run_manifest(protocol, digest)

    assert first == second
    assert first["protocol_digest"] == digest
    assert first["manifest_digest"].startswith("sha256:")
    assert first["job_counts"] == {
        "calibration": 256,
        "holdout": 256,
        "total": 512,
    }


def test_manifest_has_unique_ids_and_disjoint_seed_splits() -> None:
    protocol, digest = load_protocol(PROTOCOL_PATH)
    manifest = build_run_manifest(protocol, digest)
    jobs = manifest["jobs"]

    assert len({job["run_id"] for job in jobs}) == len(jobs)
    calibration_seeds = {
        job["seed"] for job in jobs if job["split"] == "calibration"
    }
    holdout_seeds = {job["seed"] for job in jobs if job["split"] == "holdout"}
    assert calibration_seeds.isdisjoint(holdout_seeds)


def test_holdout_pairs_cuda_reference_with_mlx_candidate() -> None:
    protocol, digest = load_protocol(PROTOCOL_PATH)
    jobs = build_run_manifest(protocol, digest)["jobs"]
    holdout = [job for job in jobs if job["split"] == "holdout"]
    grouped: dict[tuple[str, str, int, int], set[tuple[str, str]]] = {}
    for job in holdout:
        key = (
            job["stratum"],
            job["prompt_id"],
            job["seed"],
            job["horizon_seconds"],
        )
        grouped.setdefault(key, set()).add((job["role"], job["backend"]))

    assert grouped
    assert all(
        roles == {("reference", "cuda"), ("candidate", "mlx")}
        for roles in grouped.values()
    )


def test_protocol_rejects_retuning_after_holdout_visibility(tmp_path: Path) -> None:
    raw = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    raw["retuning_policy"] = "thresholds may be revised"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="prohibit retuning"):
        load_protocol(path)


def test_protocol_rejects_seed_split_overlap(tmp_path: Path) -> None:
    raw = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    raw["sampling"]["holdout_seeds"][0] = raw["sampling"]["calibration_seeds"][0]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="must be disjoint"):
        load_protocol(path)
