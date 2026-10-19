"""Frame capture for the timed pass: cheap downscale, ring buffer, async writer.

The timed pass does only what timing needs (world-handoff spec section 4): a
host-side downscale of each frame to at most ``target_short_side`` pixels on
the short side, appended to an in-memory buffer that a background thread
drains to compressed ``.npz`` chunks. Metrics run later from the saved
chunks, so they can be rerun without regenerating video.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np


def downscale_short_side(frames: np.ndarray, target: int = 256) -> np.ndarray:
    """Deterministic area-mean downscale of (n, H, W, 3) uint8 frames.

    Uses an integer decimation factor ``k = ceil(short_side / target)`` and
    averages k x k blocks (cropping the sub-block remainder), so results are
    identical across platforms; no interpolation library is involved.
    """

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (n, height, width, 3)")
    if frames.dtype != np.uint8:
        raise ValueError("frames must be uint8")
    if target < 1:
        raise ValueError("target must be positive")
    height, width = frames.shape[1], frames.shape[2]
    short = min(height, width)
    factor = -(-short // target)  # ceil
    if factor <= 1:
        return np.array(frames, copy=True)
    cropped_height = (height // factor) * factor
    cropped_width = (width // factor) * factor
    cropped = frames[:, :cropped_height, :cropped_width, :]
    blocks = cropped.reshape(
        frames.shape[0],
        cropped_height // factor,
        factor,
        cropped_width // factor,
        factor,
        3,
    )
    means = blocks.astype(np.float32).mean(axis=(2, 4))
    result: np.ndarray = np.clip(np.rint(means), 0, 255).astype(np.uint8)
    return result


class FrameCaptureError(RuntimeError):
    """The background writer failed; raised on the caller's thread at finalize."""


class FrameCapture:
    """Capture downscaled frames during the timed pass and persist them async."""

    def __init__(
        self,
        directory: Path,
        *,
        target_short_side: int = 256,
        frames_per_file: int = 256,
    ) -> None:
        if frames_per_file < 1:
            raise ValueError("frames_per_file must be positive")
        self._directory = directory
        self._target = target_short_side
        self._frames_per_file = frames_per_file
        self._buffer: list[np.ndarray] = []
        self._buffered = 0
        self._next_start_index = 0
        self._file_index = 0
        self._queue: queue.Queue[tuple[Path, np.ndarray, int] | None] = queue.Queue()
        self._error: BaseException | None = None
        self._written: list[Path] = []
        directory.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            path, frames, start_index = item
            try:
                temporary = path.with_name(f".{path.name}.tmp.npz")
                np.savez_compressed(temporary, frames=frames, start_index=start_index)
                temporary.replace(path)
                self._written.append(path)
            except BaseException as error:  # surfaced at finalize
                self._error = error
                return

    def add(self, frames: np.ndarray) -> None:
        """Downscale and buffer one emission's frames ((n, H, W, 3) uint8)."""

        self._buffer.append(downscale_short_side(frames, self._target))
        self._buffered += int(frames.shape[0])
        while self._buffered >= self._frames_per_file:
            self._flush(self._frames_per_file)

    def _flush(self, count: int) -> None:
        if count == 0 or not self._buffer:
            return
        stacked = np.concatenate(self._buffer, axis=0)
        chunk, remainder = stacked[:count], stacked[count:]
        self._buffer = [remainder] if remainder.shape[0] else []
        self._buffered = int(remainder.shape[0])
        path = self._directory / f"chunk-{self._file_index:04d}.npz"
        self._queue.put((path, chunk, self._next_start_index))
        self._next_start_index += int(chunk.shape[0])
        self._file_index += 1

    def finalize(self) -> list[Path]:
        """Flush remaining frames, stop the writer, and return written files."""

        self._flush(self._buffered)
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise FrameCaptureError("frame capture writer failed") from self._error
        return list(self._written)


def load_captured_frames(directory: Path) -> np.ndarray:
    """Load every captured chunk back into one (n, h, w, 3) uint8 array."""

    chunks: list[tuple[int, np.ndarray]] = []
    for path in sorted(directory.glob("chunk-*.npz")):
        with np.load(path) as data:
            chunks.append((int(data["start_index"]), np.asarray(data["frames"])))
    chunks.sort(key=lambda item: item[0])
    expected = 0
    arrays: list[np.ndarray] = []
    for start_index, frames in chunks:
        if start_index != expected:
            raise ValueError(
                f"captured chunks are not contiguous: expected start {expected}, "
                f"got {start_index}"
            )
        expected += int(frames.shape[0])
        arrays.append(frames)
    if not arrays:
        raise ValueError(f"no captured chunks under {directory}")
    return np.concatenate(arrays, axis=0)


__all__: list[str] = [
    "FrameCapture",
    "FrameCaptureError",
    "downscale_short_side",
    "load_captured_frames",
]
