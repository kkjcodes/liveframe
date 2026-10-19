"""Canonical action vocabulary, ActionScript, and chunk planning.

Tokens match WorldMark's six primitives plus explicit no-op and pitch
(world-handoff spec section 5). Time is measured in native frames so scripts
mean the same thing at any generation speed. Combinations are plus-joined in
canonical token order (``W+L``, never ``L+W``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from liveframe.gates import contract_digest

ACTION_TOKENS: tuple[str, ...] = ("W", "S", "A", "D", "L", "R", "U", "N", "0")
_TOKEN_ORDER = {token: index for index, token in enumerate(ACTION_TOKENS)}
_OPPOSITES = (("W", "S"), ("A", "D"), ("L", "R"), ("U", "N"))


def canonical_action(action: str) -> str:
    """Validate and canonicalize one action string (single token or combo)."""

    parts = action.split("+")
    if not parts or any(part not in _TOKEN_ORDER for part in parts):
        raise ValueError(f"unknown action token in: {action!r}")
    if len(set(parts)) != len(parts):
        raise ValueError(f"repeated token in action: {action!r}")
    if len(parts) > 1 and "0" in parts:
        raise ValueError(f"no-op cannot be combined with motion: {action!r}")
    for left, right in _OPPOSITES:
        if left in parts and right in parts:
            raise ValueError(f"opposing tokens in one action: {action!r}")
    return "+".join(sorted(parts, key=_TOKEN_ORDER.__getitem__))


def action_tokens(action: str) -> tuple[str, ...]:
    return tuple(canonical_action(action).split("+"))


@dataclass(frozen=True, slots=True)
class ActionSegment:
    t_frames: int
    action: str
    hold_frames: int

    def __post_init__(self) -> None:
        if self.t_frames < 0:
            raise ValueError("t_frames must be non-negative")
        if self.hold_frames < 1:
            raise ValueError("hold_frames must be positive")
        object.__setattr__(self, "action", canonical_action(self.action))

    def as_dict(self) -> dict[str, Any]:
        return {
            "t_frames": self.t_frames,
            "action": self.action,
            "hold_frames": self.hold_frames,
        }


@dataclass(frozen=True, slots=True)
class ActionScript:
    segments: tuple[ActionSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("an ActionScript requires at least one segment")
        if self.segments[0].t_frames != 0:
            raise ValueError("the first segment must start at frame 0")
        expected = 0
        for segment in self.segments:
            if segment.t_frames != expected:
                raise ValueError(
                    f"segments must be contiguous: expected t_frames={expected}, "
                    f"got {segment.t_frames}"
                )
            expected += segment.hold_frames

    @property
    def total_frames(self) -> int:
        last = self.segments[-1]
        return last.t_frames + last.hold_frames

    def action_at(self, frame: int) -> str:
        if not 0 <= frame < self.total_frames:
            raise ValueError(f"frame {frame} outside script of {self.total_frames} frames")
        for segment in self.segments:
            if segment.t_frames <= frame < segment.t_frames + segment.hold_frames:
                return segment.action
        raise AssertionError("contiguous script must cover every frame")

    def as_jsonl(self) -> str:
        return (
            "\n".join(
                json.dumps(segment.as_dict(), sort_keys=True) for segment in self.segments
            )
            + "\n"
        )

    def digest(self) -> str:
        return contract_digest({"segments": [segment.as_dict() for segment in self.segments]})

    @classmethod
    def from_segments(cls, rows: Iterable[tuple[int, str, int]]) -> ActionScript:
        return cls(
            tuple(
                ActionSegment(t_frames=t, action=action, hold_frames=hold)
                for t, action, hold in rows
            )
        )

    @classmethod
    def from_jsonl(cls, text: str) -> ActionScript:
        segments = []
        for line in text.splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError("each ActionScript line must be a JSON object")
            segments.append(
                ActionSegment(
                    t_frames=int(raw["t_frames"]),
                    action=str(raw["action"]),
                    hold_frames=int(raw["hold_frames"]),
                )
            )
        return cls(tuple(segments))

    @classmethod
    def load(cls, path: Path) -> ActionScript:
        return cls.from_jsonl(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """One backend emission worth of frames with its commanded action."""

    index: int
    start_frame: int
    frames: int
    action: str
    action_changed: bool
    quantization_frames: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_frame": self.start_frame,
            "frames": self.frames,
            "action": self.action,
            "action_changed": self.action_changed,
            "quantization_frames": self.quantization_frames,
        }


def plan_chunks(
    script: ActionScript, chunk_frames: int, *, first_chunk_frames: int | None = None
) -> Iterator[ChunkPlan]:
    """Convert a script to chunk granularity without silently shifting changes.

    A chunk's action is the segment active at the chunk's first frame. When a
    segment boundary falls inside a chunk, the offset of the earliest such
    boundary is recorded as ``quantization_frames`` (0 when every change is
    chunk-aligned); the change itself takes effect at the next chunk.
    ``first_chunk_frames`` supports causal VAEs whose first emission is
    shorter than the steady-state chunk.
    """

    if chunk_frames < 1:
        raise ValueError("chunk_frames must be positive")
    first = chunk_frames if first_chunk_frames is None else first_chunk_frames
    if first < 1:
        raise ValueError("first_chunk_frames must be positive")
    boundaries = {segment.t_frames for segment in script.segments}
    total = script.total_frames
    starts = [0]
    cursor = min(first, total)
    while cursor < total:
        starts.append(cursor)
        cursor += chunk_frames
    previous_action: str | None = None
    for index, start in enumerate(starts):
        size = first if index == 0 else chunk_frames
        frames = min(size, total - start)
        action = script.action_at(start)
        interior = sorted(
            boundary - start
            for boundary in boundaries
            if start < boundary < start + frames
        )
        yield ChunkPlan(
            index=index,
            start_frame=start,
            frames=frames,
            action=action,
            action_changed=previous_action is not None and action != previous_action,
            quantization_frames=interior[0] if interior else 0,
        )
        previous_action = action
