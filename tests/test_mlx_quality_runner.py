from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from liveframe.mlx_quality_runner import (
    HOLDOUT_SEEDS,
    mlx_runtime_provenance,
    validate_candidate_job,
    validate_mlx_assets,
    validate_vae_execution_mode,
)
from liveframe.quality_calibration import load_protocol
from liveframe.quality_capture import load_run_manifest

ROOT = Path(__file__).parents[1]


def _candidate() -> dict[str, Any]:
    protocol, digest = load_protocol(
        ROOT / "artifacts" / "perceptual-protocol.v1.json"
    )
    manifest = load_run_manifest(
        ROOT / "artifacts" / "perceptual-run-manifest.v1.json",
        protocol_digest=digest,
    )
    del protocol
    return next(
        dict(job)
        for job in manifest["jobs"]
        if job["split"] == "holdout" and job["role"] == "candidate"
    )


def test_accepts_only_frozen_mlx_holdout_candidate() -> None:
    job = _candidate()

    validate_candidate_job(job)

    assert job["seed"] in HOLDOUT_SEEDS
    assert job["pixel_frames"] == 4 * (job["latent_frames"] - 1) + 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backend", "cuda", "only accepts holdout candidate"),
        ("role", "reference", "only accepts holdout candidate"),
        ("seed", 42, "frozen holdout set"),
        ("latent_frames", 3, "latent horizon is not frozen"),
        ("pixel_frames", 1, "pixel horizon"),
    ],
)
def test_rejects_unsealed_candidate_identity(
    field: str, value: Any, message: str
) -> None:
    job = _candidate()
    job[field] = value

    with pytest.raises(ValueError, match=message):
        validate_candidate_job(job)


def test_asset_validation_rejects_bundle_before_model_access(tmp_path: Path) -> None:
    bundle = json.loads(
        (ROOT / "artifacts" / "perceptual-policy-bundle.v2.json").read_text()
    )
    altered = copy.deepcopy(bundle)
    altered["bundle_digest"] = "sha256:other"

    with pytest.raises(ValueError, match="frozen v2 policy bundle"):
        validate_mlx_assets(tmp_path / "missing", altered)


def test_runtime_provenance_requires_declared_m5_device() -> None:
    class Metal:
        @staticmethod
        def is_available() -> bool:
            return True

    class Runtime:
        metal = Metal()
        __version__ = "test"

        @staticmethod
        def device_info() -> dict[str, str]:
            return {"device_name": "Other GPU", "architecture": "other"}

    with pytest.raises(ValueError, match="requires Apple M5 Max"):
        mlx_runtime_provenance(Runtime())


def test_vae_execution_mode_is_fail_closed() -> None:
    assert validate_vae_execution_mode("eager") == "eager"
    assert validate_vae_execution_mode("compiled-steady") == "compiled-steady"
    with pytest.raises(ValueError, match="unsupported VAE execution mode"):
        validate_vae_execution_mode("unknown")