"""Deterministic synthetic world model for tests and metric validation.

Renders a procedural 3D scene (checkered ground plane plus colored boxes,
ray-cast in vectorized NumPy at low resolution) so camera motion is real and
ground truth is known. The emission schedule is fully scriptable: fixed or
per-chunk delays, injected stalls, and accounting defects (wrong indices,
wrong counts, wrong shapes/dtypes, empty chunks, non-monotonic worker
timestamps) so every session contract rule can be exercised without a GPU.

Timing modes:
- with a shared :class:`FakeClock`, the backend advances the same clock the
  session reads, so measured numbers must match the schedule exactly;
- with ``busy=True`` and the real clock, each chunk burns its scheduled delay
  in a busy-wait, which supports harness-overhead measurement.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from liveframe.engine import NativeFrame
from liveframe.world.actions import ActionScript
from liveframe.world.backend import WorldBackendError, WorldCapabilities

_YAW_RATE_DEGREES = 3.0
_PITCH_RATE_DEGREES = 1.5
_MOVE_STEP = 0.22
_BOXES = (
    ((-3.0, 0.0, 6.0), (1.2, 1.6, 1.2), (205, 72, 60)),
    ((2.5, 0.0, 9.0), (1.6, 2.4, 1.6), (64, 110, 200)),
    ((0.5, 0.0, 14.0), (2.2, 1.2, 2.2), (70, 170, 90)),
    ((-5.5, 0.0, 12.0), (1.0, 3.0, 1.0), (220, 180, 70)),
)


class FakeClock:
    """A monotonic clock both the session and the backend can share."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot advance a monotonic clock backwards")
        self.now += seconds


@dataclass(frozen=True, slots=True)
class SyntheticDefects:
    """Accounting defects injected at specific chunk indices."""

    index_offset_at: Mapping[int, int] = field(default_factory=dict)
    wrong_count_at: Mapping[int, int] = field(default_factory=dict)
    wrong_shape_at: frozenset[int] = frozenset()
    wrong_dtype_at: frozenset[int] = frozenset()
    empty_chunk_at: frozenset[int] = frozenset()
    nonmonotonic_worker_ts_at: frozenset[int] = frozenset()
    crash_at: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class SyntheticVisualDefects:
    """Visual defects for metric validation (spec 6.4), in frame indices."""

    freeze_from_frame: int | None = None
    black_from_frame: int | None = None
    brightness_drift_per_frame: float = 0.0
    blur_from_frame: int | None = None
    blur_growth_per_frame: float = 0.05
    wrong_direction: bool = False
    onset_delay_frames: int = 0


_OPPOSITE_TOKEN = {"W": "S", "S": "W", "A": "D", "D": "A", "L": "R", "R": "L",
                   "U": "N", "N": "U", "0": "0"}


def _box_blur(frame: np.ndarray, radius: int) -> np.ndarray:
    """Separable numpy box blur (keeps the adapter dependency-free)."""

    if radius < 1:
        return frame
    kernel = 2 * radius + 1
    padded = np.pad(
        frame.astype(np.float32),
        ((radius, radius), (radius, radius), (0, 0)),
        mode="edge",
    )
    zeros_rows = np.zeros((1, padded.shape[1], 3), dtype=np.float32)
    cumulative = np.cumsum(np.concatenate([zeros_rows, padded], axis=0), axis=0)
    vertical = (cumulative[kernel:] - cumulative[:-kernel]) / kernel
    zeros_cols = np.zeros((vertical.shape[0], 1, 3), dtype=np.float32)
    cumulative = np.cumsum(np.concatenate([zeros_cols, vertical], axis=1), axis=1)
    horizontal = (cumulative[:, kernel:] - cumulative[:, :-kernel]) / kernel
    blurred: np.ndarray = np.clip(np.rint(horizontal), 0, 255).astype(np.uint8)
    return blurred


class SyntheticWorld:
    """Scriptable fake world model implementing the WorldBackend protocol."""

    def __init__(
        self,
        *,
        resolution: tuple[int, int] = (144, 256),
        chunk_frames: int = 4,
        native_fps: float = 16.0,
        chunk_delay_seconds: float | Sequence[float] = 0.01,
        stall_extra_seconds: Mapping[int, float] | None = None,
        time_to_first_frame_extra: float = 0.0,
        defects: SyntheticDefects | None = None,
        visual_defects: SyntheticVisualDefects | None = None,
        motion_scale: float = 1.0,
        clock: FakeClock | None = None,
        busy: bool = False,
        interaction_mode: str = "interactive",
    ) -> None:
        if busy and clock is not None:
            raise ValueError("busy mode uses the real clock; do not pass a FakeClock")
        if interaction_mode not in ("interactive", "batch"):
            raise ValueError("interaction_mode must be interactive or batch")
        self._resolution = resolution
        self._chunk_frames = chunk_frames
        self._native_fps = native_fps
        self._delays = chunk_delay_seconds
        self._stalls = dict(stall_extra_seconds or {})
        self._ttff_extra = time_to_first_frame_extra
        self._defects = defects or SyntheticDefects()
        self._visual = visual_defects or SyntheticVisualDefects()
        self._motion_scale = float(motion_scale)
        self._clock = clock
        self._busy = busy
        self._interaction_mode = interaction_mode
        self._loaded = False
        self._frame_counter = 0
        self._active_tokens: tuple[str, ...] = ("0",)
        self._pending_tokens: tuple[str, ...] | None = None
        self._pending_countdown = 0
        self._last_frame: np.ndarray | None = None
        self._chunk_index = 0
        self._position = np.zeros(3, dtype=np.float64)
        self._yaw = 0.0
        self._pitch = 0.0
        self._worker_now = 0.0

    @property
    def capabilities(self) -> WorldCapabilities:
        return WorldCapabilities(
            name="synthetic",
            interaction_mode=self._interaction_mode,  # type: ignore[arg-type]
            native_fps=self._native_fps,
            chunk_frames=self._chunk_frames,
            resolution=self._resolution,
            actions=frozenset({"W", "S", "A", "D", "L", "R", "U", "N", "0"}),
            action_magnitudes={
                "move_units_per_frame": _MOVE_STEP,
                "yaw_degrees_per_frame": _YAW_RATE_DEGREES,
                "pitch_degrees_per_frame": _PITCH_RATE_DEGREES,
            },
            supports_seed=True,
            min_vram_gb=None,
            license_notice=None,
        )

    def load(self) -> None:
        self._loaded = True

    def reset(self, image: Any, prompt: str | None, seed: int) -> None:
        del image, prompt
        if not self._loaded:
            raise WorldBackendError("reset before load")
        generator = np.random.default_rng(seed)
        self._position = np.array(
            [generator.uniform(-1, 1), 1.4, generator.uniform(-1, 1)], dtype=np.float64
        )
        self._yaw = float(generator.uniform(-10, 10))
        self._pitch = 0.0
        self._chunk_index = 0
        self._worker_now = 0.0
        self._frame_counter = 0
        self._active_tokens = ("0",)
        self._pending_tokens = None
        self._pending_countdown = 0
        self._last_frame = None

    def close(self) -> None:
        self._loaded = False

    # -- motion + rendering -------------------------------------------------

    def _effective_tokens(self, tokens: Sequence[str]) -> tuple[str, ...]:
        """Apply onset-delay and wrong-direction defects to the commanded tokens."""

        requested = tuple(tokens)
        if self._visual.wrong_direction:
            requested = tuple(_OPPOSITE_TOKEN.get(token, token) for token in requested)
        delay = self._visual.onset_delay_frames
        if delay <= 0:
            self._active_tokens = requested
            return requested
        if requested != self._active_tokens:
            if self._pending_tokens != requested:
                self._pending_tokens = requested
                self._pending_countdown = delay
            if self._pending_countdown > 0:
                self._pending_countdown -= 1
                return self._active_tokens
            self._active_tokens = requested
            self._pending_tokens = None
        return self._active_tokens

    def _apply_visual_defects(self, frame: np.ndarray) -> np.ndarray:
        visual = self._visual
        index = self._frame_counter
        if visual.freeze_from_frame is not None and index >= visual.freeze_from_frame and (
            self._last_frame is not None
        ):
            return self._last_frame
        if visual.black_from_frame is not None and index >= visual.black_from_frame:
            return np.zeros_like(frame)
        result = frame
        if visual.brightness_drift_per_frame:
            shift = visual.brightness_drift_per_frame * index
            result = np.clip(
                result.astype(np.float32) + shift, 0, 255
            ).astype(np.uint8)
        if visual.blur_from_frame is not None and index >= visual.blur_from_frame:
            radius = int(
                (index - visual.blur_from_frame) * visual.blur_growth_per_frame
            ) + 1
            result = _box_blur(result, min(radius, 12))
        return result

    def _apply_action_frame(self, tokens: Sequence[str]) -> None:
        yaw = np.radians(self._yaw)
        forward = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
        right = np.array([np.cos(yaw), 0.0, -np.sin(yaw)])
        move = _MOVE_STEP * self._motion_scale
        turn = _YAW_RATE_DEGREES * self._motion_scale
        pitch_rate = _PITCH_RATE_DEGREES * self._motion_scale
        for token in self._effective_tokens(tokens):
            if token == "W":
                self._position = self._position + move * forward
            elif token == "S":
                self._position = self._position - move * forward
            elif token == "A":
                self._position = self._position - move * right
            elif token == "D":
                self._position = self._position + move * right
            elif token == "L":
                self._yaw -= turn
            elif token == "R":
                self._yaw += turn
            elif token == "U":
                self._pitch = min(45.0, self._pitch + pitch_rate)
            elif token == "N":
                self._pitch = max(-45.0, self._pitch - pitch_rate)
            elif token == "0":
                pass
            else:
                raise WorldBackendError(f"synthetic world cannot apply token {token!r}")

    def render_frame(self) -> np.ndarray:
        """Ray-cast the scene from the current camera pose (one RGB frame)."""

        height, width = self._resolution
        yaw = np.radians(self._yaw)
        pitch = np.radians(self._pitch)
        xs = np.linspace(-1.0, 1.0, width) * (width / height)
        ys = np.linspace(1.0, -1.0, height)
        grid_x, grid_y = np.meshgrid(xs, ys)
        directions = np.stack(
            [grid_x, grid_y, np.ones_like(grid_x)], axis=-1
        )
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        pitched_y = directions[..., 1] * cos_p + directions[..., 2] * sin_p
        pitched_z = -directions[..., 1] * sin_p + directions[..., 2] * cos_p
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        world_x = directions[..., 0] * cos_y + pitched_z * sin_y
        world_z = -directions[..., 0] * sin_y + pitched_z * cos_y
        direction = np.stack([world_x, pitched_y, world_z], axis=-1)
        direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)

        sky = np.array([150, 190, 235], dtype=np.float32)
        frame = np.broadcast_to(sky, (height, width, 3)).copy()
        depth = np.full((height, width), np.inf)

        # Ground plane y = 0 with a checker texture.
        dy = direction[..., 1]
        hits = dy < -1e-6
        t_plane = np.where(hits, -self._position[1] / np.where(hits, dy, -1.0), np.inf)
        t_safe = np.where(hits, t_plane, 0.0)
        px = self._position[0] + t_safe * direction[..., 0]
        pz = self._position[2] + t_safe * direction[..., 2]
        checker = ((np.floor(px) + np.floor(pz)) % 2).astype(bool)
        ground = np.where(checker[..., None], 95.0, 160.0)
        ground_rgb = np.concatenate(
            [ground, ground * 0.95, ground * 0.8], axis=-1
        ).reshape(height, width, 3)
        ground_mask = hits & (t_plane < depth)
        frame[ground_mask] = ground_rgb[ground_mask]
        depth[ground_mask] = t_plane[ground_mask]

        # Axis-aligned boxes via the slab method.
        for (bx, by, bz), (sx, sy, sz), color in _BOXES:
            low = np.array([bx - sx / 2, by, bz - sz / 2])
            high = np.array([bx + sx / 2, by + sy, bz + sz / 2])
            t_near = np.full((height, width), -np.inf)
            t_far = np.full((height, width), np.inf)
            valid = np.ones((height, width), dtype=bool)
            for axis in range(3):
                d_axis = direction[..., axis]
                o_axis = self._position[axis]
                parallel = np.abs(d_axis) < 1e-9
                inside = (low[axis] <= o_axis) & (o_axis <= high[axis])
                valid &= ~parallel | inside
                safe = np.where(parallel, 1.0, d_axis)
                t1 = (low[axis] - o_axis) / safe
                t2 = (high[axis] - o_axis) / safe
                near = np.minimum(t1, t2)
                far = np.maximum(t1, t2)
                t_near = np.where(parallel, t_near, np.maximum(t_near, near))
                t_far = np.where(parallel, t_far, np.minimum(t_far, far))
            box_hit = valid & (t_near <= t_far) & (t_near > 1e-6) & (t_near < depth)
            shade = np.clip(1.0 - t_near / 40.0, 0.35, 1.0)
            for channel in range(3):
                channel_view = frame[..., channel]
                channel_view[box_hit] = color[channel] * shade[box_hit]
            depth[box_hit] = t_near[box_hit]

        rendered: np.ndarray = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
        return rendered

    # -- emission -----------------------------------------------------------

    def _chunk_delay(self, index: int) -> float:
        base = (
            float(self._delays[index])
            if isinstance(self._delays, Sequence)
            else float(self._delays)
        )
        extra = self._stalls.get(index, 0.0)
        if index == 0:
            extra += self._ttff_extra
        return base + extra

    def _spend(self, seconds: float) -> None:
        if self._clock is not None:
            self._clock.advance(seconds)
        elif self._busy:
            deadline = time.perf_counter() + seconds
            while time.perf_counter() < deadline:
                pass

    def _emit_chunk(self, tokens: Sequence[str], frames: int) -> Iterator[NativeFrame]:
        index = self._chunk_index
        defects = self._defects
        if index in defects.crash_at:
            raise WorldBackendError(f"synthetic crash injected at chunk {index}")
        self._spend(self._chunk_delay(index))
        self._worker_now += self._chunk_delay(index)
        if index in defects.empty_chunk_at:
            self._chunk_index += 1
            return
        rendered = []
        for _ in range(frames):
            self._apply_action_frame(tokens)
            frame = self._apply_visual_defects(self.render_frame())
            self._last_frame = frame
            self._frame_counter += 1
            rendered.append(frame)
        payload = np.stack(rendered, axis=0)
        count = defects.wrong_count_at.get(index, frames)
        if index in defects.wrong_shape_at:
            payload = payload[:, :-2, :, :]
        if index in defects.wrong_dtype_at:
            payload = payload.astype(np.float32)
        worker_ts = self._worker_now
        if index in defects.nonmonotonic_worker_ts_at:
            worker_ts = -1.0
        emission_index = index + defects.index_offset_at.get(index, 0)
        self._chunk_index += 1
        yield NativeFrame(
            index=emission_index,
            payload=payload,
            native_frame_count=count,
            metadata={"worker_ts": worker_ts},
        )

    def step(self, actions: Sequence[str]) -> Iterator[NativeFrame]:
        if self._interaction_mode != "interactive":
            raise WorldBackendError("batch-mode synthetic world requires run_script")
        return self._emit_chunk(actions, self._chunk_frames)

    def run_script(self, script: ActionScript) -> Iterator[NativeFrame]:
        if self._interaction_mode != "batch":
            raise WorldBackendError("interactive synthetic world requires step")
        total = script.total_frames
        for start in range(0, total, self._chunk_frames):
            frames = min(self._chunk_frames, total - start)
            tokens = script.action_at(start).split("+")
            yield from self._emit_chunk(tokens, frames)
