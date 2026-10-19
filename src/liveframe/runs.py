"""Durable local run control suitable for key-only SSH from a phone."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from liveframe.events import Event, EventWriter, utc_now
from liveframe.gates import GateArtifact, load_bundled_gate
from liveframe.host import collect_host_provenance

TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "lost"})


class RunError(RuntimeError):
    """Raised for invalid run-control operations."""


def default_home() -> Path:
    configured = os.environ.get("LIVEFRAME_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".local" / "share" / "liveframe"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunError(f"cannot read run manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunError(f"run manifest is not an object: {path}")
    return value


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:10]}"


def _control_context() -> dict[str, Any]:
    return {
        "operator": os.environ.get("USER", "unknown"),
        "remote": bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")),
    }


@dataclass(frozen=True, slots=True)
class RunPaths:
    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def events(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def stdout(self) -> Path:
        return self.root / "stdout.log"

    @property
    def stderr(self) -> Path:
        return self.root / "stderr.log"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"


class RunStore:
    def __init__(self, root: Path | None = None, gate: GateArtifact | None = None) -> None:
        self.root = (root or default_home()).resolve()
        self.runs_root = self.root / "runs"
        self.requests_root = self.root / "requests"
        self.gate = gate or load_bundled_gate()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.requests_root.mkdir(parents=True, exist_ok=True)

    def paths(self, run_id: str) -> RunPaths:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        if not run_id or any(char not in allowed for char in run_id):
            raise RunError(f"invalid run ID: {run_id!r}")
        return RunPaths(self.runs_root / run_id)

    def submit(
        self,
        command: Sequence[str],
        *,
        idempotency_key: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not command or any(not part for part in command):
            raise RunError("submit requires a non-empty command")
        request_path: Path | None = None
        if idempotency_key:
            request_hash = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
            request_path = self.requests_root / f"{request_hash}.json"
            if request_path.exists():
                request = _read_json(request_path)
                return self.status(str(request["run_id"]))

        run_id = _run_id()
        paths = self.paths(run_id)
        paths.artifacts.mkdir(parents=True)
        paths.stdout.touch(mode=0o600)
        paths.stderr.touch(mode=0o600)
        manifest: dict[str, Any] = {
            "schema": "liveframe.run.v1",
            "run_id": run_id,
            "state": "starting",
            "command": list(command),
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "gate_id": self.gate.gate_id,
            "gate_digest": self.gate.digest,
            "metadata": dict(metadata or {}),
            "host": collect_host_provenance(),
            "artifact_dir": str(paths.artifacts),
        }
        _atomic_json(paths.manifest, manifest)
        EventWriter(paths.events).append(
            Event.create(
                "run.submitted",
                run_id,
                command=list(command),
                gate_id=self.gate.gate_id,
                gate_digest=self.gate.digest,
                idempotency_key_sha256=(
                    hashlib.sha256(idempotency_key.encode()).hexdigest()
                    if idempotency_key
                    else None
                ),
            )
        )
        worker_command = [
            sys.executable,
            "-m",
            "liveframe._run_worker",
            str(paths.root),
            "--",
            *command,
        ]
        worker = subprocess.Popen(
            worker_command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        manifest["supervisor_pid"] = worker.pid
        manifest["updated_at"] = utc_now()
        _atomic_json(paths.manifest, manifest)
        if request_path is not None:
            _atomic_json(request_path, {"run_id": run_id, "created_at": utc_now()})
        return manifest

    def status(self, run_id: str, *, audit: bool = True) -> dict[str, Any]:
        paths = self.paths(run_id)
        if not paths.manifest.exists():
            raise RunError(f"unknown run: {run_id}")
        manifest = _read_json(paths.manifest)
        state = str(manifest.get("state", "unknown"))
        supervisor_pid = int(manifest.get("supervisor_pid", 0))
        if state not in TERMINAL_STATES and supervisor_pid and not _pid_exists(supervisor_pid):
            manifest["state"] = "lost"
            manifest["updated_at"] = utc_now()
            manifest["failure"] = "supervisor exited without recording a terminal state"
            _atomic_json(paths.manifest, manifest)
            EventWriter(paths.events).append(Event.create("run.lost", run_id))
        if audit:
            EventWriter(paths.events).append(
                Event.create("control.status_read", run_id, **_control_context())
            )
        return manifest

    def cancel(self, run_id: str) -> dict[str, Any]:
        paths = self.paths(run_id)
        manifest = self.status(run_id, audit=False)
        if manifest["state"] in TERMINAL_STATES:
            return manifest
        pid = int(manifest.get("supervisor_pid", 0))
        manifest["cancel_requested_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        _atomic_json(paths.manifest, manifest)
        EventWriter(paths.events).append(
            Event.create("run.cancel_requested", run_id, signal="SIGTERM")
        )
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return self.status(run_id, audit=False)
        return manifest

    def logs(self, run_id: str, *, stream: str = "stdout", lines: int = 100) -> str:
        paths = self.paths(run_id)
        path = paths.stdout if stream == "stdout" else paths.stderr
        if stream not in {"stdout", "stderr"}:
            raise RunError("stream must be stdout or stderr")
        if lines < 1:
            raise RunError("lines must be positive")
        try:
            content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            raise RunError(f"cannot read {stream} log for {run_id}: {exc}") from exc
        EventWriter(paths.events).append(
            Event.create(
                "control.logs_read", run_id, stream=stream, lines=lines, **_control_context()
            )
        )
        return "\n".join(content[-lines:])

    def artifacts(self, run_id: str) -> list[dict[str, Any]]:
        root = self.paths(run_id).artifacts
        if not root.exists():
            raise RunError(f"unknown run: {run_id}")
        result: list[dict[str, Any]] = []
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            stat = path.stat()
            result.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": stat.st_size,
                    "sha256": _file_digest(path),
                }
            )
        EventWriter(self.paths(run_id).events).append(
            Event.create(
                "control.artifacts_listed", run_id, count=len(result), **_control_context()
            )
        )
        return result

    def iter_events(self, run_id: str) -> Iterator[dict[str, Any]]:
        path = self.paths(run_id).events
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def wait_for_state(
    store: RunStore,
    run_id: str,
    states: set[str],
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        manifest = store.status(run_id, audit=False)
        if manifest["state"] in states:
            return manifest
        time.sleep(0.02)
    raise TimeoutError(f"run {run_id} did not reach {sorted(states)} within {timeout}s")
