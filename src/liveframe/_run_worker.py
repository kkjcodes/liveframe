"""Detached run supervisor. This module is an internal process entry point."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Any

from liveframe.events import Event, EventWriter, utc_now

_CANCEL_GRACE_SECONDS = 10.0


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("run manifest must be an object")
    return value


def supervise(run_dir: Path, command: Sequence[str]) -> int:
    manifest_path = run_dir / "manifest.json"
    events = EventWriter(run_dir / "events.jsonl")
    deadline = time.monotonic() + 5.0
    manifest = _manifest(manifest_path)
    while int(manifest.get("supervisor_pid", 0)) != os.getpid():
        if time.monotonic() >= deadline:
            raise RuntimeError("submitter did not publish supervisor PID")
        time.sleep(0.01)
        manifest = _manifest(manifest_path)
    run_id = str(manifest["run_id"])
    cancelled = False
    child: subprocess.Popen[bytes] | None = None

    def request_cancel(signum: int, _frame: FrameType | None) -> None:
        nonlocal cancelled
        cancelled = True
        events.append(Event.create("run.signal_received", run_id, signal=signum))
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, request_cancel)
    signal.signal(signal.SIGINT, request_cancel)

    try:
        with (run_dir / "stdout.log").open("ab", buffering=0) as stdout, (
            run_dir / "stderr.log"
        ).open("ab", buffering=0) as stderr:
            child = subprocess.Popen(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                cwd=run_dir / "artifacts",
                close_fds=True,
            )
            manifest = _manifest(manifest_path)
            manifest.update(
                {
                    "state": "running",
                    "started_at": utc_now(),
                    "updated_at": utc_now(),
                    "supervisor_pid": os.getpid(),
                    "worker_pid": child.pid,
                }
            )
            _atomic_json(manifest_path, manifest)
            events.append(Event.create("run.started", run_id, worker_pid=child.pid))

            cancel_deadline: float | None = None
            while child.poll() is None:
                if cancelled and cancel_deadline is None:
                    cancel_deadline = time.monotonic() + _CANCEL_GRACE_SECONDS
                if cancel_deadline is not None and time.monotonic() >= cancel_deadline:
                    child.kill()
                    events.append(Event.create("run.force_killed", run_id))
                time.sleep(0.05)
            return_code = int(child.returncode)
    except BaseException as exc:
        manifest = _manifest(manifest_path)
        manifest.update(
            {
                "state": "cancelled" if cancelled else "failed",
                "updated_at": utc_now(),
                "finished_at": utc_now(),
                "failure": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_json(manifest_path, manifest)
        events.append(Event.create("run.supervisor_failed", run_id, error=manifest["failure"]))
        return 1

    if cancelled or return_code < 0:
        final_state = "cancelled"
    else:
        final_state = "completed" if return_code == 0 else "failed"
    manifest = _manifest(manifest_path)
    manifest.update(
        {
            "state": final_state,
            "return_code": return_code,
            "updated_at": utc_now(),
            "finished_at": utc_now(),
        }
    )
    _atomic_json(manifest_path, manifest)
    events.append(Event.create(f"run.{final_state}", run_id, return_code=return_code))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 3 or args[1] != "--":
        usage = "usage: python -m liveframe._run_worker RUN_DIR -- COMMAND [ARG ...]"
        print(usage, file=sys.stderr)
        return 2
    return supervise(Path(args[0]), args[2:])


if __name__ == "__main__":
    raise SystemExit(main())
