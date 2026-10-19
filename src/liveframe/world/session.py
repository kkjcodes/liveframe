"""Timed world-model run loop with normative accounting (spec section 7).

The session performs only what timing needs during the timed pass: it submits
actions at chunk granularity, timestamps each emission on host receipt with an
injectable monotonic clock, validates the measurement contract fail-closed,
and optionally hands each emission's pixels to a ``FrameCapture``. All
quantile and sliding-window definitions here are normative and tested against
the synthetic backend's known schedule.

Implementation choices the spec leaves open (recorded, not hidden):
- quantiles use the nearest-rank method on sorted values;
- sustained-FPS windows slide from ``t0`` in 1-second steps, counting frames
  with ``anchor < emission_ts <= anchor + window``; a run shorter than the
  window yields the single whole-run window;
- emission intervals are deltas between consecutive emissions (time to first
  frame is reported separately, not as an interval);
- a case shorter than 60 s reports the 16-FPS real-time verdict as
  ``not applicable`` rather than pass/fail.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from liveframe.engine import Clock, NativeFrame
from liveframe.world.actions import ActionScript, ChunkPlan, plan_chunks
from liveframe.world.backend import (
    ActionNotSupportedError,
    FrameAccountingError,
    FramePayloadError,
    MissingChunkError,
    NonMonotonicTimestampError,
    WorldBackend,
    WorldBackendError,
    WorldContractError,
)
from liveframe.world.sinks import FrameCapture

REALTIME_THRESHOLD_FPS = 16.0
REALTIME_MINIMUM_SECONDS = 60.0
SUSTAINED_WINDOW_SECONDS = 10.0
SUSTAINED_STEP_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class WorldCase:
    case_id: str
    start_image: np.ndarray
    script: ActionScript
    prompt: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer")


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk: ChunkPlan
    submit_ts: float
    receipt_ts: float
    native_frames: int
    worker_ts: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.chunk.as_dict(),
            "submit_ts": self.submit_ts,
            "receipt_ts": self.receipt_ts,
            "native_frames": self.native_frames,
            "worker_ts": self.worker_ts,
        }


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    status: str  # ok | failed | invalid
    error: str | None
    error_type: str | None
    metrics: dict[str, Any] | None
    chunks: tuple[ChunkRecord, ...] = field(default=())
    captured_files: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "error": self.error,
            "error_type": self.error_type,
            "metrics": self.metrics,
            "captured_files": list(self.captured_files),
        }


def _nearest_rank(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of no values")
    rank = max(1, math.ceil(quantile * len(sorted_values)))
    return float(sorted_values[min(rank, len(sorted_values)) - 1])


def _validate_frame(
    frame: NativeFrame,
    *,
    expected_index: int,
    resolution: tuple[int, int],
) -> np.ndarray:
    if frame.index != expected_index:
        kind = "duplicate or out-of-order" if frame.index < expected_index else "missing"
        raise FrameAccountingError(
            f"expected emission index {expected_index}, got {frame.index} ({kind})"
        )
    payload = frame.payload
    if not isinstance(payload, np.ndarray):
        raise FramePayloadError(f"frame payload must be a numpy array, got {type(payload)!r}")
    if payload.dtype != np.uint8:
        raise FramePayloadError(f"frame payload must be uint8, got {payload.dtype}")
    height, width = resolution
    if payload.ndim != 4 or payload.shape[1:] != (height, width, 3):
        raise FramePayloadError(
            f"frame payload shape {payload.shape} does not match declared "
            f"resolution (n, {height}, {width}, 3)"
        )
    if int(payload.shape[0]) != frame.native_frame_count:
        raise FrameAccountingError(
            f"payload holds {payload.shape[0]} frames but declares "
            f"{frame.native_frame_count}"
        )
    return payload


def _worker_ts(frame: NativeFrame) -> float | None:
    if frame.metadata is None:
        return None
    value = frame.metadata.get("worker_ts")
    return None if value is None else float(value)


def _sustained_windows(
    emission_ts: Sequence[float],
    frames_per_emission: Sequence[int],
    t0: float,
    *,
    window: float,
    step: float,
) -> list[float]:
    last = emission_ts[-1]
    if last - t0 <= window:
        total = float(sum(frames_per_emission))
        return [total / (last - t0)] if last > t0 else []
    values: list[float] = []
    anchor = t0
    while anchor + window <= last:
        count = sum(
            frames
            for ts, frames in zip(emission_ts, frames_per_emission, strict=True)
            if anchor < ts <= anchor + window
        )
        values.append(count / window)
        anchor += step
    return values


def _realtime_verdict(
    *,
    window_fps: Sequence[float],
    stalls: int,
    wall_seconds: float,
    threshold: float,
) -> dict[str, Any]:
    if wall_seconds < REALTIME_MINIMUM_SECONDS:
        return {
            "threshold_fps": threshold,
            "applicable": False,
            "passed": None,
            "reason": (
                f"run covers {wall_seconds:.3f}s wall; the real-time rule requires a "
                f"continuous {REALTIME_MINIMUM_SECONDS:.0f}s stream"
            ),
        }
    minimum = min(window_fps)
    passed = minimum >= threshold and stalls == 0
    return {
        "threshold_fps": threshold,
        "applicable": True,
        "passed": passed,
        "min_window_fps": minimum,
        "stalls": stalls,
        "reason": None if passed else "window FPS below threshold or stalled",
    }


def _metrics(
    records: Sequence[ChunkRecord],
    *,
    t0: float,
    reset_seconds: float,
    native_fps: float,
    chunk_frames: int,
    interactive: bool,
) -> dict[str, Any]:
    emission_ts = [record.receipt_ts for record in records]
    frames_per_emission = [record.native_frames for record in records]
    native_frames = sum(frames_per_emission)
    wall_seconds = emission_ts[-1] - t0
    if wall_seconds <= 0:
        raise WorldBackendError("case wall time must be positive")
    intervals = [later - earlier for earlier, later in pairwise(emission_ts)]
    sorted_intervals = sorted(intervals)
    stall_threshold = 2.0 * chunk_frames / native_fps
    stall_seconds = [interval for interval in intervals if interval > stall_threshold]
    window_fps = _sustained_windows(
        emission_ts,
        frames_per_emission,
        t0,
        window=SUSTAINED_WINDOW_SECONDS,
        step=SUSTAINED_STEP_SECONDS,
    )
    sorted_windows = sorted(window_fps)
    generated_fps = native_frames / wall_seconds
    latencies = sorted(
        record.receipt_ts - record.submit_ts
        for record in records
        if record.chunk.action_changed
    )
    quantization = [
        {"chunk": record.chunk.index, "frames": record.chunk.quantization_frames}
        for record in records
        if record.chunk.quantization_frames
    ]
    return {
        "reset_seconds": reset_seconds,
        "native_frames": native_frames,
        "wall_seconds": wall_seconds,
        "generated_fps": generated_fps,
        "time_to_first_frame_seconds": emission_ts[0] - t0,
        "realtime_factor": generated_fps / native_fps,
        "sustained_fps": {
            "window_seconds": SUSTAINED_WINDOW_SECONDS,
            "step_seconds": SUSTAINED_STEP_SECONDS,
            "windows": len(window_fps),
            "min": sorted_windows[0] if sorted_windows else None,
            "p5": _nearest_rank(sorted_windows, 0.05) if sorted_windows else None,
            "median": _nearest_rank(sorted_windows, 0.50) if sorted_windows else None,
        },
        "emission_intervals": {
            "count": len(intervals),
            "p50": _nearest_rank(sorted_intervals, 0.50) if intervals else None,
            "p95": _nearest_rank(sorted_intervals, 0.95) if intervals else None,
            "p99": _nearest_rank(sorted_intervals, 0.99) if intervals else None,
            "max": sorted_intervals[-1] if intervals else None,
            "jitter": (
                _nearest_rank(sorted_intervals, 0.95) - _nearest_rank(sorted_intervals, 0.50)
                if intervals
                else None
            ),
        },
        "stalls": {
            "threshold_seconds": stall_threshold,
            "count": len(stall_seconds),
            "total_seconds": float(sum(stall_seconds)),
        },
        "realtime": _realtime_verdict(
            window_fps=window_fps,
            stalls=len(stall_seconds),
            wall_seconds=wall_seconds,
            threshold=REALTIME_THRESHOLD_FPS,
        ),
        "realtime_vs_model": _realtime_verdict(
            window_fps=window_fps,
            stalls=len(stall_seconds),
            wall_seconds=wall_seconds,
            threshold=native_fps,
        ),
        "control_latency": (
            {
                "samples": len(latencies),
                "p50": _nearest_rank(latencies, 0.50) if latencies else None,
                "p95": _nearest_rank(latencies, 0.95) if latencies else None,
                "max": latencies[-1] if latencies else None,
            }
            if interactive
            else {"applicable": False, "reason": "batch mode cannot support latency claims"}
        ),
        "chunk_quantization": quantization,
        "interaction_mode": "interactive" if interactive else "batch",
    }


def _run_timed_pass(
    backend: WorldBackend,
    case: WorldCase,
    *,
    clock: Clock,
    capture: FrameCapture | None,
) -> tuple[list[ChunkRecord], float, float]:
    capabilities = backend.capabilities
    for segment in case.script.segments:
        for token in segment.action.split("+"):
            if token not in capabilities.actions:
                raise ActionNotSupportedError(
                    f"action token {token!r} is outside the declared capabilities "
                    f"of {capabilities.name}"
                )
    reset_started = clock()
    backend.reset(case.start_image, case.prompt, case.seed)
    reset_seconds = clock() - reset_started

    plans = list(
        plan_chunks(
            case.script,
            capabilities.chunk_frames,
            first_chunk_frames=capabilities.chunk_frames_first,
        )
    )
    interactive = capabilities.interaction_mode == "interactive"
    records: list[ChunkRecord] = []
    previous_worker_ts: float | None = None
    expected_index = 0

    def consume(frame: NativeFrame, plan: ChunkPlan, submit_ts: float) -> None:
        nonlocal expected_index, previous_worker_ts
        # Timestamp first: validation and capture are harness work and must not
        # inflate the receipt time attributed to the model.
        receipt_ts = clock()
        payload = _validate_frame(
            frame, expected_index=expected_index, resolution=capabilities.resolution
        )
        worker_ts = _worker_ts(frame)
        if worker_ts is not None and previous_worker_ts is not None and (
            worker_ts < previous_worker_ts
        ):
            raise NonMonotonicTimestampError(
                f"worker timestamp went backwards at emission {frame.index}: "
                f"{worker_ts} < {previous_worker_ts}"
            )
        if worker_ts is not None:
            previous_worker_ts = worker_ts
        if capture is not None:
            capture.add(payload)
        records.append(
            ChunkRecord(
                chunk=plan,
                submit_ts=submit_ts,
                receipt_ts=receipt_ts,
                native_frames=frame.native_frame_count,
                worker_ts=worker_ts,
            )
        )
        expected_index += 1

    t0 = clock()
    if interactive:
        for plan in plans:
            submit_ts = clock()
            emitted = 0
            for frame in backend.step(plan.action.split("+")):
                consume(frame, plan, submit_ts)
                emitted += frame.native_frame_count
            if emitted == 0:
                raise MissingChunkError(
                    f"no frames arrived for chunk {plan.index} after its action "
                    f"({plan.action!r}) was submitted"
                )
            if emitted != plan.frames:
                raise FrameAccountingError(
                    f"chunk {plan.index} emitted {emitted} native frames, "
                    f"expected {plan.frames}"
                )
    else:
        submit_ts = t0
        plan_by_emission = plans
        for emission, frame in enumerate(backend.run_script(case.script)):
            if emission >= len(plan_by_emission):
                raise FrameAccountingError(
                    f"backend emitted more chunks than planned ({len(plan_by_emission)})"
                )
            consume(frame, plan_by_emission[emission], submit_ts)
    total = sum(record.native_frames for record in records)
    if total != case.script.total_frames:
        raise FrameAccountingError(
            f"case generated {total} native frames, script requires "
            f"{case.script.total_frames}"
        )
    return records, t0, reset_seconds


def run_case(
    backend: WorldBackend,
    case: WorldCase,
    *,
    clock: Clock = time.perf_counter,
    capture: FrameCapture | None = None,
) -> CaseResult:
    """Run one case; contract violations yield ``invalid``, crashes ``failed``."""

    capabilities = backend.capabilities

    def _abort(status: str, error: BaseException) -> CaseResult:
        if capture is not None:
            # The original error stays authoritative; a writer failure during
            # abort must not mask it.
            with contextlib.suppress(Exception):
                capture.finalize()
        return CaseResult(
            case_id=case.case_id,
            status=status,
            error=str(error),
            error_type=type(error).__name__,
            metrics=None,
        )

    try:
        records, t0, reset_seconds = _run_timed_pass(
            backend, case, clock=clock, capture=capture
        )
        metrics = _metrics(
            records,
            t0=t0,
            reset_seconds=reset_seconds,
            native_fps=capabilities.native_fps,
            chunk_frames=capabilities.chunk_frames,
            interactive=capabilities.interaction_mode == "interactive",
        )
    except WorldContractError as error:
        return _abort("invalid", error)
    except Exception as error:
        return _abort("failed", error)
    captured = tuple(str(path) for path in capture.finalize()) if capture is not None else ()
    return CaseResult(
        case_id=case.case_id,
        status="ok",
        error=None,
        error_type=None,
        metrics=metrics,
        chunks=tuple(records),
        captured_files=captured,
    )


def write_case_directory(result: CaseResult, directory: Path) -> list[Path]:
    """Persist one case's timing rows and summary under ``cases/<case_id>/``."""

    directory.mkdir(parents=True, exist_ok=True)
    timing_path = directory / "timing.jsonl"
    with timing_path.open("w", encoding="utf-8") as stream:
        for record in result.chunks:
            stream.write(json.dumps(record.as_dict(), sort_keys=True) + "\n")
    summary_path = directory / "case.json"
    summary_path.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return [timing_path, summary_path]
