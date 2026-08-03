from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from liveframe.quality_capture import prepare_calibration_batch
from liveframe.quality_executor import RunnerOutput, execute_batch


class FakeRunner:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[str] = []
        self.fail = fail

    def run(self, job: Mapping[str, Any], output_dir: Path) -> RunnerOutput:
        self.calls.append(str(job["run_id"]))
        if self.fail:
            raise RuntimeError("synthetic runner failure")
        artifact = output_dir / "pixels.bin"
        artifact.write_bytes(str(job["run_id"]).encode())
        return RunnerOutput(
            artifact_paths=[artifact],
            metrics={"quality": 0.75},
            runtime={"backend": "fake"},
            timing={"generation_seconds": 1.0},
        )


def _inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    job = {
        "run_id": "quality-test",
        "split": "calibration",
        "role": "teacher",
        "backend": "cuda",
        "stratum": "animation",
        "prompt_id": "prompt",
        "prompt": "A test prompt.",
        "seed": 10,
        "horizon_seconds": 15,
        "latent_frames": 60,
        "pixel_frames": 237,
        "playback_seconds": 14.8125,
        "status": "planned",
    }
    manifest = {
        "protocol_digest": "sha256:protocol",
        "manifest_digest": "sha256:manifest",
        "jobs": [job],
    }
    protocol = {"quality_metrics": {"quality": {}}}
    return protocol, manifest


def test_executor_writes_valid_result_and_resumes(tmp_path: Path) -> None:
    protocol, manifest = _inputs()
    batch = prepare_calibration_batch(manifest, tmp_path)
    runner = FakeRunner()

    first = execute_batch(protocol, manifest, batch, tmp_path, runner)
    second = execute_batch(protocol, manifest, batch, tmp_path, runner)

    assert first["completed_run_ids"] == ["quality-test"]
    assert second["skipped_run_ids"] == ["quality-test"]
    assert runner.calls == ["quality-test"]
    assert (tmp_path / "quality-test" / "result.json").is_file()


def test_executor_does_not_publish_failed_job(tmp_path: Path) -> None:
    protocol, manifest = _inputs()
    batch = prepare_calibration_batch(manifest, tmp_path)

    with pytest.raises(RuntimeError, match="synthetic runner failure"):
        execute_batch(protocol, manifest, batch, tmp_path, FakeRunner(fail=True))

    assert not (tmp_path / "quality-test" / "result.json").exists()


def test_executor_rejects_wrong_metric_set(tmp_path: Path) -> None:
    protocol, manifest = _inputs()
    protocol["quality_metrics"] = {"different": {}}
    batch = prepare_calibration_batch(manifest, tmp_path)

    with pytest.raises(ValueError, match="metric set mismatch"):
        execute_batch(protocol, manifest, batch, tmp_path, FakeRunner())


def test_executor_rejects_artifacts_outside_job_directory(tmp_path: Path) -> None:
    protocol, manifest = _inputs()
    batch = prepare_calibration_batch(manifest, tmp_path)

    class EscapingRunner(FakeRunner):
        def run(self, job: Mapping[str, Any], output_dir: Path) -> RunnerOutput:
            outside = tmp_path / "outside.bin"
            outside.write_bytes(b"outside")
            output = super().run(job, output_dir)
            return RunnerOutput(
                artifact_paths=[outside],
                metrics=output.metrics,
                runtime=output.runtime,
                timing=output.timing,
            )

    with pytest.raises(ValueError, match="inside its output directory"):
        execute_batch(protocol, manifest, batch, tmp_path, EscapingRunner())
