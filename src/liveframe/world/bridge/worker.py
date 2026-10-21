"""Bridge worker shim: runs INSIDE the model environment.

Constraints (spec section 6.3): one file, standard library only, Python 3.10+,
importable without LiveFrame installed (the host copies this file to a temp
directory on PYTHONPATH). The adapter it hosts is likewise a plain-Python
class found via ``--adapter module:Class``; it may import anything available
in the model environment (torch, numpy, the upstream repo).

Adapter duck-type contract (no imports from LiveFrame):
- ``capabilities() -> dict`` with at least name, interaction_mode, native_fps,
  chunk_frames, resolution [h, w], actions (list of canonical tokens);
  optionally action_magnitudes, supports_seed, min_vram_gb, license_notice.
- ``load() -> None`` (heavy weight loading happens here, timed as cold start).
- ``reset(image, prompt, seed) -> None`` where image is a (h, w, 3) uint8
  array-like built by the worker from raw bytes.
- ``step(actions: list[str]) -> iterable`` of chunk payloads; each payload is
  an object exposing the buffer protocol with shape (n, H, W, 3), dtype uint8
  (a numpy array qualifies), or a tuple ``(payload, n)``.
- optional ``run_script(segments: list[dict]) -> iterable`` of the same, for
  batch-mode models; segments carry t_frames/action/hold_frames.
- ``close() -> None``.

Protocol discipline: this shim owns the real stdout for JSONL control
messages and redirects Python-level stdout to stderr before touching the
adapter, so upstream ``print()`` calls cannot corrupt the channel.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import os
import sys
import time
import traceback
from multiprocessing import shared_memory
from typing import Any

PROTOCOL = "liveframe.world.bridge.v1"


def _fail(out: Any, message: str) -> None:
    _send(out, {"type": "error", "message": message, "traceback": traceback.format_exc()})


def _send(out: Any, message: dict[str, Any]) -> None:
    out.write(json.dumps(message, separators=(",", ":")) + "\n")
    out.flush()


def _as_chunk(payload: Any) -> tuple[Any, int]:
    if isinstance(payload, tuple):
        chunk, count = payload
        return chunk, int(count)
    shape = getattr(payload, "shape", None)
    if shape is None or len(shape) != 4:
        raise ValueError("adapter chunk payload must have shape (n, h, w, 3)")
    return payload, int(shape[0])


def _chunk_bytes(chunk: Any) -> bytes:
    to_bytes = getattr(chunk, "tobytes", None)
    if to_bytes is not None:
        return bytes(to_bytes())
    return bytes(memoryview(chunk).cast("B"))


class _Ring:
    def __init__(self, name: str, slots: int, slot_bytes: int) -> None:
        self.memory = shared_memory.SharedMemory(name=name)
        # The host owns the segment's lifetime. Without unregistering, this
        # process's resource tracker would unlink it when the worker exits
        # (including on a crash), yanking it out from under the host.
        try:
            from multiprocessing import resource_tracker

            resource_tracker.unregister(self.memory._name, "shared_memory")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - tracker internals vary
            pass
        self.slots = slots
        self.slot_bytes = slot_bytes
        self.free = list(range(slots))

    def write(self, data: bytes) -> int:
        if not self.free:
            raise RuntimeError("no free ring slot; host ack missing")
        if len(data) > self.slot_bytes:
            raise ValueError(
                f"chunk needs {len(data)} bytes but ring slots hold {self.slot_bytes}"
            )
        slot = self.free.pop(0)
        offset = slot * self.slot_bytes
        buffer = self.memory.buf
        assert buffer is not None
        buffer[offset : offset + len(data)] = data
        return slot

    def release(self, slot: int) -> None:
        if slot in self.free or not 0 <= slot < self.slots:
            raise RuntimeError(f"bad ack for slot {slot}")
        self.free.append(slot)

    def close(self) -> None:
        self.memory.close()


def _emit_chunks(
    out: Any, inp: Any, ring: _Ring, chunks: Any, start_index: int
) -> int:
    """Stream adapter chunks through the ring, honoring host acks."""

    index = start_index
    for payload in chunks:
        chunk, count = _as_chunk(payload)
        shape = tuple(int(size) for size in chunk.shape)
        data = _chunk_bytes(chunk)
        while not ring.free:
            ack = json.loads(inp.readline())
            if ack.get("type") != "ack":
                raise RuntimeError(f"expected ack while ring is full, got {ack}")
            ring.release(int(ack["slot"]))
        slot = ring.write(data)
        _send(
            out,
            {
                "type": "frames",
                "slot": slot,
                "chunk_index": index,
                "count": count,
                "shape": list(shape),
                "nbytes": len(data),
                "worker_ts": time.perf_counter(),
            },
        )
        index += 1
    return index


def _build_image(message: dict[str, Any]) -> Any:
    height, width, channels = message["image_shape"]
    raw = base64.b64decode(message["image_b64"])
    try:
        import numpy

        return numpy.frombuffer(raw, dtype=numpy.uint8).reshape(height, width, channels)
    except ModuleNotFoundError:
        return {"shape": (height, width, channels), "data": raw}


class _SP:
    """Sequence-parallel fan-out for adapters whose model runs across N ranks.

    Rank 0 (the process the host spawns) stays on the host control channel. It
    spawns N-1 helper copies of this worker, initialises a torch.distributed
    group, and broadcasts each host command so every rank calls the SAME adapter
    method in lockstep -- the model's own SP collectives then coordinate. Only
    rank 0 captures frames and talks to the host; helpers discard output.

    Activated only when ``capabilities["sp_degree"] > 1``; torch is imported
    lazily so the normal single-process path stays stdlib-only.
    """

    def __init__(self, degree: int, adapter_arg: str) -> None:
        import torch
        import torch.distributed as dist

        self.torch, self.dist, self.degree = torch, dist, degree
        self.is_helper = "LIVEFRAME_SP_HELPER_RANK" in os.environ
        self.rank = int(os.environ["LIVEFRAME_SP_HELPER_RANK"]) if self.is_helper else 0
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", os.environ.get("LIVEFRAME_SP_PORT", "29599"))
        os.environ["WORLD_SIZE"] = str(degree)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(self.rank)
        self._procs: list[Any] = []
        if not self.is_helper:
            import subprocess

            worker_file = os.path.abspath(sys.argv[0])
            for r in range(1, degree):
                env = dict(os.environ)
                env.update(LIVEFRAME_SP_HELPER_RANK=str(r), RANK=str(r), LOCAL_RANK=str(r))
                # Redirect helper stdout to a log (NOT the inherited host pipe,
                # which would corrupt the JSONL control channel); keep stderr too.
                hlog = open(f"/tmp/liveframe_sp_helper_{r}.log", "w")  # noqa: SIM115
                self._procs.append(
                    subprocess.Popen(
                        [sys.executable, worker_file, "--adapter", adapter_arg],
                        env=env, stdin=subprocess.DEVNULL, stdout=hlog, stderr=hlog,
                    )
                )
        backend = os.environ.get(
            "LIVEFRAME_SP_BACKEND", "nccl" if torch.cuda.is_available() else "gloo"
        )
        # Bind this rank to its own GPU BEFORE init_process_group, or NCCL maps
        # multiple ranks onto the same device ("Duplicate GPU detected").
        self._dev = None
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            local = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(local)
            self._dev = torch.device("cuda", local)
        dist.init_process_group(backend=backend, rank=self.rank, world_size=degree)

    def bcast(self, obj: Any) -> Any:
        box = [obj]
        # Pass an explicit per-rank device: broadcast_object_list otherwise
        # defaults its size/data tensors to cuda:0 on every rank, so NCCL sees
        # all ranks on one GPU ("Duplicate GPU detected").
        if self._dev is not None:
            self.dist.broadcast_object_list(box, src=0, device=self._dev)
        else:
            self.dist.broadcast_object_list(box, src=0)
        return box[0]

    def shutdown(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            self.dist.destroy_process_group()
        for p in self._procs:
            try:
                p.wait(timeout=60)
            except Exception:
                p.kill()


def _run_sp_helper(adapter: Any, sp: _SP) -> int:
    """Non-rank-0 loop: mirror rank 0's calls, produce no host output."""
    while True:
        cmd = sp.bcast(None)
        kind = cmd.get("type")
        if kind == "load":
            adapter.load()
        elif kind == "reset":
            adapter.reset(cmd["image"], cmd.get("prompt"), int(cmd["seed"]))
        elif kind == "step":
            for _ in adapter.step(list(cmd["actions"])):
                pass
        elif kind == "run_script":
            for _ in adapter.run_script(list(cmd["segments"])):
                pass
        elif kind == "close":
            adapter.close()
            break
    sp.shutdown()
    return 0


def _sp_degree(capabilities: dict[str, Any]) -> int:
    try:
        return max(1, int(capabilities.get("sp_degree", 1)))
    except (TypeError, ValueError):
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="module:Class inside the model env")
    arguments = parser.parse_args()

    # Own the real stdout; route adapter prints to stderr.
    out = open(sys.stdout.fileno(), "w", buffering=1, closefd=False)  # noqa: SIM115
    sys.stdout = sys.stderr
    inp = sys.stdin

    module_name, _, class_name = arguments.adapter.partition(":")
    try:
        adapter_class = getattr(importlib.import_module(module_name), class_name)
        adapter = adapter_class()
        capabilities = dict(adapter.capabilities())
    except Exception:
        _fail(out, f"failed to construct adapter {arguments.adapter}")
        return 1

    # Sequence-parallel adapters fan out across ranks; helpers never touch the
    # host channel and exit via the mirror loop.
    degree = _sp_degree(capabilities)
    sp = _SP(degree, arguments.adapter) if degree > 1 else None
    if sp is not None and sp.is_helper:
        return _run_sp_helper(adapter, sp)

    hello = json.loads(inp.readline())
    if hello.get("type") != "hello" or hello.get("proto") != PROTOCOL:
        _fail(out, f"protocol mismatch: {hello}")
        return 1
    _send(
        out,
        {
            "type": "hello_ack",
            "proto": PROTOCOL,
            "python": sys.version.split()[0],
            "capabilities": capabilities,
        },
    )

    ring: _Ring | None = None
    chunk_index = 0
    try:
        for line in inp:
            message = json.loads(line)
            kind = message.get("type")
            if kind == "load":
                if sp is not None:
                    sp.bcast({"type": "load"})
                adapter.load()
                _send(out, {"type": "loaded"})
            elif kind == "reset":
                if ring is None:
                    ring = _Ring(
                        message["ring_name"],
                        int(message["ring_slots"]),
                        int(message["slot_bytes"]),
                    )
                image = _build_image(message)
                seed = int(message["seed"])
                if sp is not None:
                    sp.bcast(
                        {"type": "reset", "image": image,
                         "prompt": message.get("prompt"), "seed": seed}
                    )
                adapter.reset(image, message.get("prompt"), seed)
                chunk_index = 0
                ring.free = list(range(ring.slots))
                _send(out, {"type": "reset_done"})
            elif kind == "step":
                assert ring is not None, "step before reset"
                if sp is not None:
                    sp.bcast({"type": "step", "actions": list(message["actions"])})
                chunk_index = _emit_chunks(
                    out, inp, ring, adapter.step(list(message["actions"])), chunk_index
                )
                _send(out, {"type": "chunk_done", "next_index": chunk_index})
            elif kind == "run_script":
                assert ring is not None, "run_script before reset"
                if sp is not None:
                    sp.bcast({"type": "run_script", "segments": list(message["segments"])})
                chunk_index = _emit_chunks(
                    out,
                    inp,
                    ring,
                    adapter.run_script(list(message["segments"])),
                    chunk_index,
                )
                _send(out, {"type": "script_done", "next_index": chunk_index})
            elif kind == "ack":
                assert ring is not None
                ring.release(int(message["slot"]))
            elif kind == "close":
                if sp is not None:
                    sp.bcast({"type": "close"})
                adapter.close()
                _send(out, {"type": "closed"})
                break
            else:
                raise RuntimeError(f"unknown message type: {kind}")
    except Exception:
        _fail(out, "worker crashed")
        return 1
    finally:
        if ring is not None:
            ring.close()
        if sp is not None:
            sp.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
