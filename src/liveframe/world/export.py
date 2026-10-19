"""Export a world run's videos + action logs for external scorers (spec 10.3).

Formats and the exact layouts targeted (both checked 2026-10-13; neither
repository publishes version tags, so the layout note is dated instead):

- ``raw``: LiveFrame's own layout — ``videos/<case_id>.mp4``,
  ``actions/<case_id>.jsonl`` (one chunk row per line, straight from the run's
  ``timing.jsonl``), ``cases.json``, and a copy of ``report.json``.
- ``worldmark``: https://github.com/AlayaLab/WorldMark —
  ``{domain}/{view}/{MODEL}/{image:03d}_{action:03d}.mp4`` with zero-padded
  indices, native resolution/fps ("evaluation resizes and time-normalises").
  LiveFrame scenes map to image indices and cases to action indices;
  ``worldmark_mapping.json`` records the mapping and the caveat that these are
  LiveFrame suite cases, not WorldMark ``arena_inputs/`` assignments — run
  their ``generation/check_delivery.py`` before scoring.
- ``wbench``: https://github.com/meituan-longcat/WBench —
  ``work_dirs/<model>/videos/case_{id}_combined.mp4`` plus
  ``data/cases/case_{id}.json`` describing the interaction per case.

Videos are rebuilt from the run's capture tier (short side 256) at the model's
native fps; full-resolution export would require re-running generation, which
an export must never silently do.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from liveframe.world.sinks import load_captured_frames

EXPORT_FORMATS = ("raw", "worldmark", "wbench")
LAYOUTS_CHECKED = "2026-10-13"


def _report(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "report.json"
    if not path.exists():
        raise FileNotFoundError(f"{run_dir} has no report.json (not a suite run?)")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _ok_cases(report: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    """Completed cases WITH their suite position.

    Indices in exported filenames come from the case's position in the full
    suite (and the canonical scene list), never from what happens to be
    present — otherwise one failed case would shift every later index and
    silently misalign cross-model comparisons in external scorers.
    """

    cases = [
        (position, case)
        for position, case in enumerate(report["cases"])
        if case["status"] in ("ok", "skipped-complete")
    ]
    if not cases:
        raise ValueError("run has no completed cases to export")
    return cases


def _scene_index(scene: str, known: list[str]) -> int:
    from liveframe.world.suites import SCENE_NAMES

    if scene in SCENE_NAMES:
        return SCENE_NAMES.index(scene)
    # Custom scenes sort after the canonical ones, deterministically.
    return len(SCENE_NAMES) + sorted(known).index(scene)


def open_video_writer(
    path: Path, fps: float, size: tuple[int, int]
) -> tuple[cv2.VideoWriter, str]:
    """H.264 (``avc1``) when the OpenCV build can encode it, else ``mp4v``.

    mp4v is MPEG-4 Part 2, which browsers do not decode — fine for QuickTime
    or VLC, broken in a <video> tag. The codec actually used is returned so
    callers can record it next to the files.
    """

    width, height = size
    for codec in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"could not open any mp4 video writer for {path}")


def _write_video(frames: np.ndarray, path: Path, fps: float) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer, codec = open_video_writer(path, fps, (width, height))
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return codec


def _actions(run_dir: Path, case_id: str) -> list[dict[str, Any]]:
    timing = run_dir / "cases" / case_id / "timing.jsonl"
    return [
        json.loads(line)
        for line in timing.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _frames(run_dir: Path, case_id: str) -> np.ndarray:
    return load_captured_frames(run_dir / "cases" / case_id / "frames_256")


def export_run(
    run_dir: Path,
    output_dir: Path,
    *,
    export_format: str,
    model_tag: str | None = None,
) -> dict[str, Any]:
    """Export one run; returns a summary with the files written."""

    if export_format not in EXPORT_FORMATS:
        raise ValueError(
            f"unknown export format: {export_format} (have {EXPORT_FORMATS})"
        )
    report = _report(run_dir)
    fps = float(report["capabilities"]["native_fps"])
    model = model_tag or str(report["headline"]["model"])
    cases = _ok_cases(report)
    output_dir.mkdir(parents=True, exist_ok=True)
    if export_format == "raw":
        written = _export_raw(run_dir, output_dir, report, cases, fps)
    elif export_format == "worldmark":
        written = _export_worldmark(run_dir, output_dir, cases, fps, model)
    else:
        written = _export_wbench(run_dir, output_dir, cases, fps, model)
    return {
        "format": export_format,
        "layouts_checked": LAYOUTS_CHECKED,
        "cases": len(cases),
        "files": [str(path) for path in written],
        "output_dir": str(output_dir),
    }


def _export_raw(
    run_dir: Path,
    output_dir: Path,
    report: dict[str, Any],
    cases: list[tuple[int, dict[str, Any]]],
    fps: float,
) -> list[Path]:
    written: list[Path] = []
    metadata = []
    for _position, case in cases:
        case_id = case["case_id"]
        video = output_dir / "videos" / f"{case_id}.mp4"
        _write_video(_frames(run_dir, case_id), video, fps)
        written.append(video)
        actions_path = output_dir / "actions" / f"{case_id}.jsonl"
        actions_path.parent.mkdir(parents=True, exist_ok=True)
        actions_path.write_text(
            "\n".join(
                json.dumps(row, sort_keys=True) for row in _actions(run_dir, case_id)
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(actions_path)
        metadata.append(
            {
                "case_id": case_id,
                "scene": case.get("scene"),
                "seed": case.get("seed"),
                "video": str(video.relative_to(output_dir)),
                "actions": str(actions_path.relative_to(output_dir)),
                "capture_tier": "short side 256",
                "fps": fps,
            }
        )
    cases_path = output_dir / "cases.json"
    cases_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written.append(cases_path)
    report_copy = output_dir / "report.json"
    report_copy.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written.append(report_copy)
    return written


def _export_worldmark(
    run_dir: Path,
    output_dir: Path,
    cases: list[tuple[int, dict[str, Any]]],
    fps: float,
    model: str,
) -> list[Path]:
    # domain/view are fixed for LiveFrame suites: procedural scenes, first person.
    base = output_dir / "synthetic" / "first" / model.upper()
    known = sorted({str(case.get("scene")) for _, case in cases})
    written: list[Path] = []
    mapping = []
    for position, case in cases:
        case_id = case["case_id"]
        image = _scene_index(str(case.get("scene")), known)
        # Action index = position in the full suite, stable across models even
        # when another model failed a case.
        video = base / f"{image:03d}_{position:03d}.mp4"
        _write_video(_frames(run_dir, case_id), video, fps)
        written.append(video)
        mapping.append(
            {
                "video": str(video.relative_to(output_dir)),
                "case_id": case_id,
                "scene": case.get("scene"),
                "seed": case.get("seed"),
                "actions": _actions(run_dir, case_id),
            }
        )
    note = {
        "target_layout": "github.com/AlayaLab/WorldMark (unversioned README)",
        "layout_checked": LAYOUTS_CHECKED,
        "caveat": (
            "videos come from LiveFrame suite cases, not WorldMark arena_inputs/ "
            "assignments; run their generation/check_delivery.py before scoring"
        ),
        "mapping": mapping,
    }
    mapping_path = output_dir / "worldmark_mapping.json"
    mapping_path.write_text(
        json.dumps(note, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written.append(mapping_path)
    return written


def _export_wbench(
    run_dir: Path,
    output_dir: Path,
    cases: list[tuple[int, dict[str, Any]]],
    fps: float,
    model: str,
) -> list[Path]:
    written: list[Path] = []
    for position, case in cases:
        index = position + 1  # suite position, stable across models/runs
        case_id = case["case_id"]
        video = output_dir / "work_dirs" / model / "videos" / f"case_{index}_combined.mp4"
        _write_video(_frames(run_dir, case_id), video, fps)
        written.append(video)
        description = {
            "case_id": index,
            "source_case_id": case_id,
            "scene": case.get("scene"),
            "seed": case.get("seed"),
            "fps": fps,
            "turns": [
                {
                    "turn": turn,
                    "tokens": row["action"],
                    "start_frame": row["start_frame"],
                    "frames": row["frames"],
                }
                for turn, row in enumerate(_actions(run_dir, case_id))
            ],
            "layout_checked": LAYOUTS_CHECKED,
            "target_layout": "github.com/meituan-longcat/WBench (unversioned README)",
        }
        case_path = output_dir / "data" / "cases" / f"case_{index}.json"
        case_path.parent.mkdir(parents=True, exist_ok=True)
        case_path.write_text(
            json.dumps(description, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.append(case_path)
    return written
