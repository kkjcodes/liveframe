from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from liveframe.quality_capture import sha256_file
from liveframe.quality_metric_runner import (
    PIXEL_SCHEMA,
    PerceptualMetricRunner,
    load_pixels,
    validate_metric_artifact,
)


def _write_pixels(path: Path, frames: np.ndarray) -> None:
    path.write_bytes(frames.tobytes())
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(
            {
                "schema": PIXEL_SCHEMA,
                "dtype": "uint8",
                "layout": "T,H,W,C",
                "shape": list(frames.shape),
                "sha256": sha256_file(path),
            }
        ),
        encoding="utf-8",
    )


def test_load_pixels_validates_lossless_artifact(tmp_path: Path) -> None:
    path = tmp_path / "pixels.bin"
    frames = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
    _write_pixels(path, frames)

    loaded, metadata = load_pixels(path)

    assert np.array_equal(loaded, frames)
    assert metadata["shape"] == [3, 2, 2, 3]


def test_load_pixels_rejects_checksum_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "pixels.bin"
    _write_pixels(path, np.zeros((3, 2, 2, 3), dtype=np.uint8))
    path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_pixels(path)


def test_pairwise_measurement_rejects_shape_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    runner = object.__new__(PerceptualMetricRunner)
    arrays: Any = iter(
        (
            (np.zeros((3, 2, 2, 3), dtype=np.uint8), {}),
            (np.zeros((4, 2, 2, 3), dtype=np.uint8), {}),
        )
    )
    monkeypatch.setattr("liveframe.quality_metric_runner.load_pixels", lambda _: next(arrays))

    with pytest.raises(ValueError, match="pixel shapes differ"):
        runner.measure_pairwise(
            tmp_path / "reference.bin",
            tmp_path / "candidate.bin",
            expected_frames=3,
        )


def test_metric_provenance_is_digest_bound(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text('{"schema":"metrics"}\n', encoding="utf-8")
    protocol = {
        "metric_provenance": {
            "artifact": "metrics.json",
            "artifact_sha256": sha256_file(artifact),
        }
    }

    assert validate_metric_artifact(protocol, tmp_path)["schema"] == "metrics"
    artifact.write_text('{"schema":"altered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_metric_artifact(protocol, tmp_path)
