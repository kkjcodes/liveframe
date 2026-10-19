"""Bridge wire protocol: versioned JSONL control + shared-memory frame ring.

Control messages travel as single JSON lines over the worker's stdin/stdout.
Pixels travel through a ``multiprocessing.shared_memory`` ring sized for two
chunks; a slot is reused only after the host acknowledges it, so the worker
can never overwrite frames the host has not copied out.

The worker-side implementation must stay importable on Python 3.10 with the
standard library only (see ``worker.py``); this module is host-side and may
be imported freely.
"""

from __future__ import annotations

PROTOCOL = "liveframe.world.bridge.v1"
RING_SLOTS = 2

# host -> worker
MSG_HELLO = "hello"
MSG_LOAD = "load"
MSG_RESET = "reset"
MSG_STEP = "step"
MSG_RUN_SCRIPT = "run_script"
MSG_ACK = "ack"
MSG_CLOSE = "close"

# worker -> host
MSG_HELLO_ACK = "hello_ack"
MSG_LOADED = "loaded"
MSG_RESET_DONE = "reset_done"
MSG_FRAMES = "frames"
MSG_CHUNK_DONE = "chunk_done"
MSG_SCRIPT_DONE = "script_done"
MSG_CLOSED = "closed"
MSG_ERROR = "error"

REQUIRED_CAPABILITY_KEYS = frozenset(
    {
        "name",
        "interaction_mode",
        "native_fps",
        "chunk_frames",
        "resolution",
        "actions",
    }
)
