"""Minimal world-run orchestration for the M2 smoke gate.

Builds a run directory (spec section 10.1 layout), runs the built-in ``smoke``
suite (one case, ~10 s of video at the model's declared rate), captures
frames, writes case artifacts and the digest-bound run ledger. The full suite
system, reports, compare, and export arrive in M4.
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from liveframe.world.actions import ActionScript
from liveframe.world.backend import WorldBackend
from liveframe.world.ledger import write_world_ledger
from liveframe.world.session import WorldCase, run_case, write_case_directory
from liveframe.world.sinks import FrameCapture


def synthetic_start_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Self-made deterministic start image (no upstream demo assets)."""

    generator = np.random.default_rng(seed)
    ys = np.linspace(0, 1, height)[:, None]
    xs = np.linspace(0, 1, width)[None, :]
    red = (ys * 200 + 30).astype(np.uint8)
    green = (xs * 180 + 40).astype(np.uint8)
    blue = ((ys + xs) * 90 + 50).astype(np.uint8)
    frame = np.stack(
        [np.broadcast_to(red, (height, width)),
         np.broadcast_to(green, (height, width)),
         np.broadcast_to(blue, (height, width))],
        axis=-1,
    ).astype(np.uint8)
    noise = generator.integers(0, 24, size=(height, width, 3), dtype=np.uint8)
    boxes = frame.copy()
    for _ in range(6):
        y0 = int(generator.integers(0, height - height // 4))
        x0 = int(generator.integers(0, width - width // 4))
        color = generator.integers(40, 255, size=3)
        boxes[y0 : y0 + height // 5, x0 : x0 + width // 5] = color
    image: np.ndarray = np.clip(boxes.astype(np.int16) + noise - 12, 0, 255).astype(np.uint8)
    return image


def smoke_script(backend: WorldBackend, *, target_seconds: float = 10.0) -> ActionScript:
    """One mixed-motion script sized to the backend's chunk grid."""

    capabilities = backend.capabilities
    first = capabilities.chunk_frames_first or capabilities.chunk_frames
    frames = first
    while frames < target_seconds * capabilities.native_fps:
        frames += capabilities.chunk_frames
    third = max(1, frames // 3)
    segments = [(0, "W", third), (third, "L", third), (2 * third, "0", frames - 2 * third)]
    return ActionScript.from_segments(segments)


def _environment(run_dir: Path | None = None) -> dict[str, Any]:
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    try:
        probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            env["gpu"] = probe.stdout.strip()
            # U-0002 (approved): full identity capture, hashed into the ledger.
            if run_dir is not None:
                full = subprocess.run(
                    ["nvidia-smi", "-q"], capture_output=True, text=True, check=False
                )
                if full.returncode == 0:
                    (run_dir / "env-nvidia-smi.txt").write_text(
                        full.stdout, encoding="utf-8"
                    )
                    env["nvidia_smi_q"] = "env-nvidia-smi.txt"
    except FileNotFoundError:
        env["gpu"] = None
    return env


def run_smoke(
    backend: WorldBackend,
    *,
    run_root: Path,
    model_name: str,
    seed: int = 0,
    metadata: dict[str, Any] | None = None,
    cold_start_storage: str = "unknown",
    gpu_price_usd_per_hour: float | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run the built-in smoke suite; returns (run_dir, case summary)."""

    import time

    capabilities = backend.capabilities
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / model_name / "smoke" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    load_started = time.perf_counter()
    backend.load()
    cold_start_seconds = time.perf_counter() - load_started

    script = smoke_script(backend)
    height, width = capabilities.resolution
    case = WorldCase(
        case_id="smoke-000",
        start_image=synthetic_start_image(height, width, seed),
        script=script,
        seed=seed,
    )
    case_dir = run_dir / "cases" / case.case_id
    capture = FrameCapture(case_dir / "frames_256")
    result = run_case(backend, case, capture=capture)
    write_case_directory(result, case_dir)

    manifest = {
        "schema": "liveframe.world.manifest.v1",
        "suite": "smoke",
        "model": model_name,
        "capabilities": capabilities.as_dict(),
        "script_digest": script.digest(),
        "script_total_frames": script.total_frames,
        # Cold start is diagnostic context, never a headline number (M2.1).
        "cold_start_seconds": cold_start_seconds,
        "cold_start_storage": cold_start_storage,
        "seed": seed,
        "environment": _environment(run_dir),
        "extra": metadata or {},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_world_ledger(run_dir, metadata={"suite": "smoke", "model": model_name})
    summary = {
        "run_dir": str(run_dir),
        "status": result.status,
        "error": result.error,
        "cold_start_seconds": cold_start_seconds,
        "cold_start_storage": cold_start_storage,
        "metrics": result.metrics,
    }
    if gpu_price_usd_per_hour is not None and result.metrics is not None:
        from liveframe.world.metrics.runner import cost_per_generated_minute

        summary["cost_per_generated_minute_usd"] = cost_per_generated_minute(
            wall_seconds=result.metrics["wall_seconds"],
            native_frames=result.metrics["native_frames"],
            native_fps=capabilities.native_fps,
            gpu_price_usd_per_hour=gpu_price_usd_per_hour,
        )
        summary["gpu_price_usd_per_hour"] = gpu_price_usd_per_hour
    return run_dir, summary


def _run_one_case(
    backend: WorldBackend,
    *,
    run_dir: Path,
    case_id: str,
    scene: str,
    seed: int,
    script: ActionScript,
    start_image: np.ndarray,
    resume_key: str,
) -> dict[str, Any]:
    import shutil

    from liveframe.world.metrics.runner import run_metric_pass, summarize_for_report

    case_dir = run_dir / "cases" / case_id
    if case_dir.exists():
        # Re-running (failed earlier attempt or stale key): stale capture
        # chunks would corrupt the metric pass, so start from nothing.
        shutil.rmtree(case_dir)
    case = WorldCase(case_id=case_id, start_image=start_image, script=script, seed=seed)
    capture = FrameCapture(case_dir / "frames_256")
    result = run_case(backend, case, capture=capture)
    write_case_directory(result, case_dir)
    script_rows = [
        [segment.t_frames, segment.action, segment.hold_frames]
        for segment in script.segments
    ]
    row: dict[str, Any] = {
        "case_id": case_id,
        "scene": scene,
        "seed": seed,
        "status": result.status,
        "error": result.error,
        "metrics": result.metrics,
        # (t_frames, action, hold) rows so reports can draw the action timeline.
        "script": script_rows,
    }
    if result.status == "ok":
        try:
            case_metrics = run_metric_pass(
                case_dir, native_fps=backend.capabilities.native_fps
            )
        except Exception as error:  # one bad metric pass must not kill the suite
            row["status"] = "metric-failed"
            row["error"] = f"metric pass failed: {error}"
            return row
        row["metric_summary"] = summarize_for_report(case_metrics)
        row["metric_pass_seconds"] = case_metrics["metric_pass_seconds"]
        row["metric_throughput_fps"] = (
            case_metrics["frames_analyzed"] / case_metrics["metric_pass_seconds"]
        )
        # Resume marker: the key binds suite+model+scene+seed+script digest, so
        # a later resume skips this case only if nothing that defines it changed.
        (case_dir / "resume.json").write_text(
            json.dumps(
                {"key": resume_key, "script": script_rows, "scene": scene, "seed": seed},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return row


def _resume_row(case_dir: Path, resume_key: str) -> dict[str, Any] | None:
    """Rebuild a completed case's report row from its stored artifacts.

    Returns None unless the case completed ok under the SAME resume key —
    stale artifacts (edited suite, different script/model) are re-run, never
    silently accepted.
    """

    from liveframe.world.metrics.runner import summarize_for_report

    required = (case_dir / "case.json", case_dir / "metrics.json", case_dir / "resume.json")
    if not all(path.exists() for path in required):
        return None
    previous = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    marker = json.loads((case_dir / "resume.json").read_text(encoding="utf-8"))
    if previous.get("status") != "ok" or marker.get("key") != resume_key:
        return None
    case_metrics = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    row: dict[str, Any] = {
        "case_id": previous["case_id"],
        "scene": marker.get("scene"),
        "seed": marker.get("seed"),
        "status": "skipped-complete",
        "error": None,
        "metrics": previous.get("metrics"),
        "script": marker.get("script"),
        "resume_key": resume_key,
        "metric_summary": summarize_for_report(case_metrics),
        "metric_pass_seconds": case_metrics.get("metric_pass_seconds"),
    }
    if case_metrics.get("metric_pass_seconds"):
        row["metric_throughput_fps"] = (
            case_metrics["frames_analyzed"] / case_metrics["metric_pass_seconds"]
        )
    return row


def run_suite(
    backend: WorldBackend,
    *,
    suite: str,
    run_root: Path,
    model_name: str,
    metadata: dict[str, Any] | None = None,
    cold_start_storage: str = "unknown",
    gpu_price_usd_per_hour: float | None = None,
    vendor_claim: str | None = None,
    resume_dir: Path | None = None,
    shard: tuple[int, int] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run a full suite: load once, run every case, metric pass, report, ledger.

    ``shard=(index, total)`` runs the deterministic subset of cases with
    ``case_position % total == index`` — multi-GPU is one process per GPU
    (each with its own CUDA_VISIBLE_DEVICES) writing into separate run dirs,
    merged by resuming into one dir afterwards.
    """

    import time

    from liveframe.world.report import build_report
    from liveframe.world.suites import (
        fit_scene,
        load_scene,
        materialize_suite,
        suite_digest,
    )

    capabilities = backend.capabilities
    cases = materialize_suite(suite, capabilities)
    if shard is not None:
        index, total = shard
        if not 0 <= index < total:
            raise ValueError(f"shard index {index} out of range for total {total}")
        cases = [case for pos, case in enumerate(cases) if pos % total == index]
    if resume_dir is not None:
        run_dir = resume_dir
    else:
        run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = run_root / model_name / suite / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

    first_report_started = time.perf_counter()
    load_started = time.perf_counter()
    backend.load()
    cold_start_seconds = time.perf_counter() - load_started

    rows: list[dict[str, Any]] = []
    skipped = 0
    for suite_case in cases:
        case_dir = run_dir / "cases" / suite_case.case_id
        resume_key = suite_case.key(suite=suite, model=model_name)
        if resume_dir is not None:
            resumed = _resume_row(case_dir, resume_key)
            if resumed is not None:
                rows.append(resumed)
                skipped += 1
                continue
        rows.append(
            _run_one_case(
                backend,
                run_dir=run_dir,
                case_id=suite_case.case_id,
                scene=suite_case.scene,
                seed=suite_case.seed,
                script=suite_case.script,
                start_image=fit_scene(
                    load_scene(suite_case.scene), capabilities.resolution
                ),
                resume_key=resume_key,
            )
        )

    report = build_report(
        suite=suite,
        suite_digest=suite_digest(suite),
        model_name=model_name,
        capabilities=capabilities,
        cases=rows,
        cold_start_seconds=cold_start_seconds,
        cold_start_storage=cold_start_storage,
        environment=_environment(run_dir),
        gpu_price_usd_per_hour=gpu_price_usd_per_hour,
        vendor_claim=vendor_claim,
        extra=metadata or {},
        time_to_first_report_seconds=None,
    )
    report["time_to_first_report_seconds"] = time.perf_counter() - first_report_started
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    from liveframe.world.report import render_html

    (run_dir / "report.html").write_text(render_html(report), encoding="utf-8")
    write_world_ledger(run_dir, metadata={"suite": suite, "model": model_name})
    summary = {
        "run_dir": str(run_dir),
        "report_html": str(run_dir / "report.html"),
        "report_json": str(run_dir / "report.json"),
        "suite": suite,
        "cases": len(rows),
        "skipped": skipped,
        "failures": sum(1 for row in rows if row["status"] not in ("ok", "skipped-complete")),
        "headline": report["headline"],
        "time_to_first_report_seconds": report["time_to_first_report_seconds"],
    }
    return run_dir, summary
