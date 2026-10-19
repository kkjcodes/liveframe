"""A complete minimal WorldBackend adapter — copy this file to start yours.

This "model" is a toy: it scrolls and rotates the start image according to the
commanded action. What matters is the shape of the code — five methods, one
dataclass of declared capabilities — because the harness treats your adapter
exactly like a real model: it validates every frame you emit against what you
declared, times every chunk, and fails the case (not the run) if the contract
is broken.

Run it in-process:

    liveframe world run examples/byo_adapter/drift_world.py:DriftWorld --suite smoke

Run the SAME file inside your model's own environment (your torch, your CUDA)
while the harness stays in its own process:

    liveframe world run examples/byo_adapter/drift_world.py:DriftWorld \
        --suite quick --python /path/to/model-env/bin/python

The three rules that matter when you adapt a real model:

1. ``capabilities`` must tell the truth. ``native_fps`` is the rate your model
   is DESIGNED to be viewed at (not what it achieves), ``chunk_frames`` is how
   many pixel frames one decode step emits, and ``resolution`` is exact — a
   frame with any other shape fails the case as a contract violation.
2. ``step`` must do the work the user would experience. Emit one
   :class:`NativeFrame` per decode step, with the payload a uint8 array of
   shape ``(frames, height, width, 3)``. Don't buffer frames and release them
   at the end — the harness measures the cadence your users would see.
3. Never interpolate or duplicate frames to look faster. The harness counts
   native frames only, and the frame-accounting check will fail the case if
   the count doesn't match the script.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence

import numpy as np

from liveframe.engine import NativeFrame
from liveframe.world.backend import WorldBackendError, WorldCapabilities


class DriftWorld:
    """Scrolls the start image per action: a stand-in for your real model."""

    def __init__(self, *, height: int = 144, width: int = 256) -> None:
        self._height = height
        self._width = width
        self._canvas: np.ndarray | None = None
        self._offset = np.zeros(2)  # (row, col) drift accumulated so far
        self._emission = 0
        self._loaded = False

    # -- 1/5: declared capabilities --------------------------------------
    @property
    def capabilities(self) -> WorldCapabilities:
        return WorldCapabilities(
            name="drift-world",
            interaction_mode="interactive",  # step() is called per chunk
            native_fps=16.0,
            chunk_frames=4,  # one step() emits 4 pixel frames
            resolution=(self._height, self._width),
            actions=frozenset({"W", "S", "A", "D", "L", "R", "0"}),
            action_magnitudes={"W": 2.0, "S": 2.0, "A": 2.0, "D": 2.0},
            supports_seed=True,
        )

    # -- 2/5: one-time weight loading (timed as cold start) ---------------
    def load(self) -> None:
        # A real adapter loads weights here. Cold start is reported as
        # context, never as part of a case's generation time.
        self._loaded = True

    # -- 3/5: per-case reset with the start image and seed ----------------
    def reset(self, image: np.ndarray, prompt: str | None, seed: int) -> None:
        if not self._loaded:
            raise WorldBackendError("reset() before load()")
        # Tile the start image so scrolling never runs off the canvas.
        self._canvas = np.tile(np.asarray(image, dtype=np.uint8), (3, 3, 1))
        self._offset = np.array([float(self._height), float(self._width)])
        self._emission = 0
        self._rng = np.random.default_rng(seed)

    # -- 4/5: generate one chunk for the commanded action -----------------
    def step(self, actions: Sequence[str]) -> Iterator[NativeFrame]:
        if self._canvas is None:
            raise WorldBackendError("step() before reset()")
        deltas = {
            "W": (-2.0, 0.0), "S": (2.0, 0.0),
            "A": (0.0, -2.0), "D": (0.0, 2.0),
            "L": (0.0, -1.0), "R": (0.0, 1.0), "0": (0.0, 0.0),
        }
        frames = []
        for _ in range(self.capabilities.chunk_frames):
            for token in actions:
                self._offset += deltas[token]
            self._offset[0] = float(np.clip(self._offset[0], 0, self._height * 2))
            self._offset[1] = float(np.clip(self._offset[1], 0, self._width * 2))
            row, col = int(self._offset[0]), int(self._offset[1])
            view = self._canvas[row : row + self._height, col : col + self._width]
            noise = self._rng.integers(0, 6, size=view.shape, dtype=np.uint8)
            frames.append(np.clip(view.astype(np.int16) + noise, 0, 255).astype(np.uint8))
        yield NativeFrame(
            index=self._emission,
            payload=np.stack(frames),
            native_frame_count=len(frames),
            metadata={"worker_ts": time.perf_counter()},  # optional but useful
        )
        self._emission += 1

    # Batch-mode models implement run_script instead; interactive ones raise.
    def run_script(self, script: object) -> Iterator[NativeFrame]:
        raise WorldBackendError("drift-world is interactive; the harness calls step()")

    # -- 5/5: release resources -------------------------------------------
    def close(self) -> None:
        self._canvas = None
