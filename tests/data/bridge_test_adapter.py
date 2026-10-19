"""Standalone bridge-test adapter: numpy only, deliberately no liveframe import.

Emits deterministic gradient chunks with an optional busy-wait per chunk so
bridge overhead can be measured against an in-process equivalent.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

HEIGHT, WIDTH = 48, 64


class GradientWorldAdapter:
    def __init__(self) -> None:
        config = json.loads(os.environ.get("BRIDGE_TEST_CONFIG", "{}"))
        self._chunk_frames = int(config.get("chunk_frames", 4))
        self._busy_seconds = float(config.get("busy_seconds", 0.0))
        self._crash_at_chunk = config.get("crash_at_chunk")
        self._seed = 0
        self._chunk = 0

    def capabilities(self) -> dict:
        return {
            "name": "bridge-test-gradient",
            "interaction_mode": "interactive",
            "native_fps": 16.0,
            "chunk_frames": self._chunk_frames,
            "resolution": [HEIGHT, WIDTH],
            "actions": ["W", "S", "A", "D", "L", "R", "0"],
            "action_magnitudes": {"move_units_per_frame": 1.0},
            "supports_seed": True,
        }

    def load(self) -> None:
        print("adapter prints to stdout must not corrupt the protocol")

    def reset(self, image, prompt, seed) -> None:
        assert getattr(image, "shape", None) is not None
        self._seed = int(seed)
        self._chunk = 0

    def _frame(self, frame_index: int) -> np.ndarray:
        base = (frame_index * 7 + self._seed) % 251
        row = (np.arange(WIDTH, dtype=np.uint16) + base) % 256
        frame = np.broadcast_to(row, (HEIGHT, WIDTH)).astype(np.uint8)
        return np.stack([frame, frame[::-1], np.full_like(frame, base)], axis=-1)

    def step(self, actions):
        if self._crash_at_chunk is not None and self._chunk == int(self._crash_at_chunk):
            raise RuntimeError("injected adapter crash")
        if self._busy_seconds:
            deadline = time.perf_counter() + self._busy_seconds
            while time.perf_counter() < deadline:
                pass
        start = self._chunk * self._chunk_frames
        payload = np.stack(
            [self._frame(start + offset) for offset in range(self._chunk_frames)], axis=0
        )
        self._chunk += 1
        yield payload

    def close(self) -> None:
        pass
