"""Host side of the subprocess bridge: a WorldBackend over JSONL + shared memory.

Spawns the worker shim with the interpreter the user names (the model's own
environment), copies the shim (and optionally the adapter module) into a temp
directory on PYTHONPATH so nothing must be installed in the model env, and
exposes the remote adapter as a normal :class:`WorldBackend`.

Timing honesty: frames are yielded to the session only after the host has
copied pixels out of the ring and acknowledged the slot, so every bridge cost
lands inside the measured receipt time (conservative, per spec 6.3). The
worker-side ``perf_counter`` stamp rides along in frame metadata as a
diagnostic; worker and host clocks are not comparable across machines and are
never mixed.
"""

from __future__ import annotations

import base64
import contextlib
import json
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import deque
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from liveframe.engine import NativeFrame
from liveframe.world.actions import ActionScript
from liveframe.world.backend import (
    WorldBackendError,
    WorldCapabilities,
)
from liveframe.world.bridge import protocol

_WORKER_SOURCE = Path(__file__).with_name("worker.py")


class BridgedWorld:
    """Run a plain-Python adapter in its own interpreter as a WorldBackend."""

    def __init__(
        self,
        *,
        python: str | Path,
        adapter: str,
        adapter_file: Path | None = None,
        chunk_timeout_seconds: float = 120.0,
        load_timeout_seconds: float = 1800.0,
        env: dict[str, str] | None = None,
    ) -> None:
        self._chunk_timeout = chunk_timeout_seconds
        self._load_timeout = load_timeout_seconds
        self._stage = Path(tempfile.mkdtemp(prefix="liveframe-bridge-"))
        shutil.copyfile(_WORKER_SOURCE, self._stage / "liveframe_bridge_worker.py")
        if adapter_file is not None:
            shutil.copyfile(adapter_file, self._stage / adapter_file.name)
        import os

        merged_env = dict(os.environ)
        if env:
            merged_env.update(env)
        existing_path = merged_env.get("PYTHONPATH", "")
        merged_env["PYTHONPATH"] = (
            f"{self._stage}{os.pathsep}{existing_path}" if existing_path else str(self._stage)
        )
        merged_env["PYTHONUNBUFFERED"] = "1"
        self._process = subprocess.Popen(
            [str(python), str(self._stage / "liveframe_bridge_worker.py"), "--adapter", adapter],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=merged_env,
        )
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=80)
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        self._ring: Any = None
        self._slot_bytes = 0
        self._send({"type": protocol.MSG_HELLO, "proto": protocol.PROTOCOL})
        hello = self._receive(timeout=60.0)
        if hello.get("type") != protocol.MSG_HELLO_ACK:
            raise WorldBackendError(f"bridge handshake failed: {hello}")
        raw = hello["capabilities"]
        missing = protocol.REQUIRED_CAPABILITY_KEYS - set(raw)
        if missing:
            raise WorldBackendError(f"adapter capabilities missing keys: {sorted(missing)}")
        self._capabilities = WorldCapabilities(
            name=str(raw["name"]),
            interaction_mode=raw["interaction_mode"],
            native_fps=float(raw["native_fps"]),
            chunk_frames=int(raw["chunk_frames"]),
            resolution=(int(raw["resolution"][0]), int(raw["resolution"][1])),
            actions=frozenset(raw["actions"]),
            action_magnitudes=dict(raw.get("action_magnitudes", {})),
            supports_seed=bool(raw.get("supports_seed", True)),
            min_vram_gb=raw.get("min_vram_gb"),
            license_notice=raw.get("license_notice"),
            chunk_frames_first=(
                int(raw["chunk_frames_first"])
                if raw.get("chunk_frames_first") is not None
                else None
            ),
        )
        self._worker_python = str(hello.get("python", "unknown"))

    # -- plumbing -------------------------------------------------------------

    def _pump_stdout(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _pump_stderr(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr_tail.append(line.rstrip())

    def _send(self, message: dict[str, Any]) -> None:
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise self._worker_died(f"writing {message.get('type')}") from error

    def _receive(self, *, timeout: float) -> dict[str, Any]:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise self._worker_died(f"no reply within {timeout:.0f}s") from None
        if line is None:
            raise self._worker_died("worker closed its control channel")
        message: dict[str, Any] = json.loads(line)
        if message.get("type") == protocol.MSG_ERROR:
            raise WorldBackendError(
                f"worker error: {message.get('message')}\n"
                f"worker traceback:\n{message.get('traceback', '')}\n"
                f"stderr tail:\n{self.stderr_tail()}"
            )
        return message

    def _worker_died(self, context: str) -> WorldBackendError:
        code = self._process.poll()
        return WorldBackendError(
            f"bridge worker unavailable ({context}); exit code {code}; "
            f"stderr tail:\n{self.stderr_tail()}"
        )

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    @property
    def worker_python(self) -> str:
        return self._worker_python

    # -- WorldBackend ----------------------------------------------------------

    @property
    def capabilities(self) -> WorldCapabilities:
        return self._capabilities

    def load(self) -> None:
        self._send({"type": protocol.MSG_LOAD})
        reply = self._receive(timeout=self._load_timeout)
        if reply.get("type") != protocol.MSG_LOADED:
            raise WorldBackendError(f"unexpected load reply: {reply}")

    def reset(self, image: Any, prompt: str | None, seed: int) -> None:
        frame = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise WorldBackendError("start image must have shape (h, w, 3)")
        if self._ring is None:
            from multiprocessing import shared_memory

            height, width = self._capabilities.resolution
            # Slots must fit the LARGEST chunk. A causal model can emit a bigger
            # first block than later ones (e.g. MG3: 57 then 40), so size by the
            # max of chunk_frames and chunk_frames_first, not chunk_frames alone.
            max_chunk_frames = max(
                self._capabilities.chunk_frames,
                self._capabilities.chunk_frames_first or 0,
            )
            self._slot_bytes = max_chunk_frames * height * width * 3
            self._ring = shared_memory.SharedMemory(
                create=True, size=protocol.RING_SLOTS * self._slot_bytes
            )
        self._send(
            {
                "type": protocol.MSG_RESET,
                "ring_name": self._ring.name,
                "ring_slots": protocol.RING_SLOTS,
                "slot_bytes": self._slot_bytes,
                "image_shape": list(frame.shape),
                "image_b64": base64.b64encode(frame.tobytes()).decode("ascii"),
                "prompt": prompt,
                "seed": seed,
            }
        )
        reply = self._receive(timeout=self._chunk_timeout)
        if reply.get("type") != protocol.MSG_RESET_DONE:
            raise WorldBackendError(f"unexpected reset reply: {reply}")

    def _frames_until(self, done_type: str) -> Iterator[NativeFrame]:
        while True:
            message = self._receive(timeout=self._chunk_timeout)
            kind = message.get("type")
            if kind == done_type:
                return
            if kind != protocol.MSG_FRAMES:
                raise WorldBackendError(f"unexpected bridge message: {message}")
            slot = int(message["slot"])
            shape = tuple(int(size) for size in message["shape"])
            nbytes = int(message["nbytes"])
            offset = slot * self._slot_bytes
            buffer = self._ring.buf[offset : offset + nbytes]
            payload = np.frombuffer(buffer, dtype=np.uint8).reshape(shape).copy()
            del buffer
            self._send({"type": protocol.MSG_ACK, "slot": slot})
            yield NativeFrame(
                index=int(message["chunk_index"]),
                payload=payload,
                native_frame_count=int(message["count"]),
                metadata={"worker_ts": float(message["worker_ts"])},
            )

    def step(self, actions: Sequence[str]) -> Iterator[NativeFrame]:
        self._send({"type": protocol.MSG_STEP, "actions": list(actions)})
        return self._frames_until(protocol.MSG_CHUNK_DONE)

    def run_script(self, script: ActionScript) -> Iterator[NativeFrame]:
        self._send(
            {
                "type": protocol.MSG_RUN_SCRIPT,
                "segments": [segment.as_dict() for segment in script.segments],
            }
        )
        return self._frames_until(protocol.MSG_SCRIPT_DONE)

    def close(self) -> None:
        try:
            if self._process.poll() is None:
                self._send({"type": protocol.MSG_CLOSE})
                self._receive(timeout=30.0)
        except WorldBackendError:
            pass
        finally:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
            if self._ring is not None:
                self._ring.close()
                # A crashed worker's resource tracker may have unlinked first.
                with contextlib.suppress(FileNotFoundError):
                    self._ring.unlink()
                self._ring = None
            shutil.rmtree(self._stage, ignore_errors=True)


def current_python() -> str:
    """The host interpreter, for same-env bridge tests."""

    return sys.executable
