from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_h100_pipeline import (  # noqa: E402
    _require_file_digest,
    _require_output_identity,
    _summarize_samples,
    _validate_attention,
)


def _sample(value: float, *, latent: str = "latent", pixel: str = "pixel") -> dict[str, Any]:
    return {
        "wall_seconds": value + 10.0,
        "compute_wall_seconds": value + 7.0,
        "pixel_materialization_seconds": value + 1.0,
        "encoding_seconds": value + 2.0,
        "generated_pixel_fps": value + 3.0,
        "ttff_seconds": value + 4.0,
        "steady_block_mean_seconds": value + 5.0,
        "output_latent_sha256": latent,
        "output_pixel_sha256": pixel,
        "blocks": [
            {
                "denoising_seconds": value,
                "refresh_seconds": value + 1.0,
                "decode_seconds": value + 2.0,
            },
            {
                "denoising_seconds": value + 3.0,
                "refresh_seconds": value + 4.0,
                "decode_seconds": value + 5.0,
            },
        ],
    }


def test_attention_validation_accepts_full_and_valid_rolling_attention() -> None:
    _validate_attention(-1, 0)
    _validate_attention(12, 3)

    with pytest.raises(ValueError, match="invalid rolling attention"):
        _validate_attention(-1, 1)
    with pytest.raises(ValueError, match="invalid rolling attention"):
        _validate_attention(12, 12)


def test_file_digest_validation_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    expected = "sha256:" + hashlib.sha256(b"checkpoint").hexdigest()

    assert _require_file_digest(checkpoint, expected, label="checkpoint") == expected
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        _require_file_digest(checkpoint, "sha256:wrong", label="checkpoint")


def test_output_identity_requires_every_sample_and_both_anchors() -> None:
    with pytest.raises(ValueError, match="at least one measured sample"):
        _require_output_identity(
            [], expected_latents="latent", expected_pixels="pixel"
        )

    samples = [_sample(1.0), _sample(2.0), _sample(3.0)]
    _require_output_identity(
        samples, expected_latents="latent", expected_pixels="pixel"
    )

    samples[1]["output_pixel_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="output identity anchor failed"):
        _require_output_identity(
            samples, expected_latents="latent", expected_pixels="pixel"
        )


def test_summary_uses_trial_medians_and_per_trial_component_totals() -> None:
    summary = _summarize_samples([_sample(1.0), _sample(100.0), _sample(3.0)])

    assert summary == {
        "wall_seconds_median": 13.0,
        "compute_wall_seconds_median": 10.0,
        "denoising_seconds_median": 9.0,
        "refresh_seconds_median": 11.0,
        "vae_seconds_median": 13.0,
        "pixel_materialization_seconds_median": 4.0,
        "encoding_seconds_median": 5.0,
        "generated_pixel_fps_median": 6.0,
        "ttff_seconds_median": 7.0,
        "steady_block_seconds_median": 8.0,
    }