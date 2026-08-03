"""Backend-neutral execution contract for frozen quality capture batches."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from liveframe.quality_capture import (
    RESULT_SCHEMA,
    result_path,
    sha256_file,
    validate_result,
    write_result,
)


@dataclass(frozen=True)
class RunnerOutput:
    """Artifacts and observations produced by one backend execution."""

    artifact_paths: Sequence[Path]
    metrics: Mapping[str, float]
    runtime: Mapping[str, Any]
    timing: Mapping[str, float]


class QualityJobRunner(Protocol):
    def run(self, job: Mapping[str, Any], output_dir: Path) -> RunnerOutput: ...


def _artifact_entry(path: Path, output_dir: Path) -> dict[str, Any]:
    resolved = path.resolve()
    root = output_dir.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"runner artifact must be a file inside its output directory: {path}")
    relative = resolved.relative_to(root)
    return {
        "path": relative.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def execute_batch(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    batch: Mapping[str, Any],
    output_root: Path,
    runner: QualityJobRunner,
) -> dict[str, Any]:
    """Execute pending jobs and publish a result only after strict validation."""

    required_metrics = set(protocol["quality_metrics"])
    completed: list[str] = []
    skipped: list[str] = []
    started = time.monotonic()
    for job in batch["jobs"]:
        run_id = str(job["run_id"])
        destination = result_path(output_root, run_id)
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                raise ValueError(f"quality result must be an object: {destination}")
            validate_result(
                existing,
                job,
                protocol_digest=str(manifest["protocol_digest"]),
                manifest_digest=str(manifest["manifest_digest"]),
                required_metrics=required_metrics,
                root=destination.parent,
            )
            skipped.append(run_id)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        job_started = time.monotonic()
        output = runner.run(job, destination.parent)
        metrics = {name: float(value) for name, value in output.metrics.items()}
        result: dict[str, Any] = {
            **dict(job),
            "schema": RESULT_SCHEMA,
            "status": "completed",
            "protocol_digest": manifest["protocol_digest"],
            "manifest_digest": manifest["manifest_digest"],
            "metrics": metrics,
            "artifacts": [
                _artifact_entry(path, destination.parent) for path in output.artifact_paths
            ],
            "runtime": dict(output.runtime),
            "timing": {
                **{name: float(value) for name, value in output.timing.items()},
                "executor_wall_seconds": time.monotonic() - job_started,
            },
        }
        validate_result(
            result,
            job,
            protocol_digest=str(manifest["protocol_digest"]),
            manifest_digest=str(manifest["manifest_digest"]),
            required_metrics=required_metrics,
            root=destination.parent,
        )
        write_result(destination, result)
        completed.append(run_id)
    return {
        "completed_run_ids": completed,
        "skipped_run_ids": skipped,
        "jobs": len(completed) + len(skipped),
        "wall_seconds": time.monotonic() - started,
    }
