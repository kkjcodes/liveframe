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
                adapter.load()
                _send(out, {"type": "loaded"})
            elif kind == "reset":
                if ring is None:
                    ring = _Ring(
                        message["ring_name"],
                        int(message["ring_slots"]),
                        int(message["slot_bytes"]),
                    )
                adapter.reset(
                    _build_image(message), message.get("prompt"), int(message["seed"])
                )
                chunk_index = 0
                ring.free = list(range(ring.slots))
                _send(out, {"type": "reset_done"})
            elif kind == "step":
                assert ring is not None, "step before reset"
                chunk_index = _emit_chunks(
                    out, inp, ring, adapter.step(list(message["actions"])), chunk_index
                )
                _send(out, {"type": "chunk_done", "next_index": chunk_index})
            elif kind == "run_script":
                assert ring is not None, "run_script before reset"
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
