"""Deterministic planning for the pinned long-video Rolling Forcing pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RollingWindow:
    """One diagonal denoising window over three-latent blocks."""

    index: int
    start_block: int
    end_block: int
    configured_timesteps: tuple[int, ...]

    @property
    def block_count(self) -> int:
        return self.end_block - self.start_block + 1

    @property
    def start_frame(self) -> int:
        return self.start_block * 3

    @property
    def end_frame(self) -> int:
        return (self.end_block + 1) * 3


@dataclass(frozen=True, slots=True)
class RollingForcingSchedule:
    """The released four-step, three-latent-block long-video schedule."""

    latent_frames: int
    block_frames: int = 3
    configured_steps: tuple[int, ...] = (1000, 750, 500, 250)
    context_noise: int = 0

    def __post_init__(self) -> None:
        if self.latent_frames < self.block_frames:
            raise ValueError("latent_frames must contain at least one complete block")
        if self.latent_frames % self.block_frames:
            raise ValueError("latent_frames must be divisible by block_frames")
        if self.block_frames != 3:
            raise ValueError("the pinned long-video checkpoint requires three-frame blocks")
        if not self.configured_steps:
            raise ValueError("configured_steps cannot be empty")
        if self.context_noise != 0:
            raise ValueError("only explicit context_noise=0 is implemented")

    @property
    def block_count(self) -> int:
        return self.latent_frames // self.block_frames

    @property
    def window_count(self) -> int:
        return self.block_count + len(self.configured_steps) - 1

    @property
    def transformer_forward_count(self) -> int:
        """One diagonal denoise and one clean-cache refresh per window."""

        return 2 * self.window_count

    @property
    def scheduler_noise_draw_count(self) -> int:
        """Every non-final block in every diagonal window is re-noised."""

        return sum(
            sum(step != self.configured_steps[-1] for step in window.configured_timesteps)
            for window in self.windows()
        )

    def windows(self) -> tuple[RollingWindow, ...]:
        step_count = len(self.configured_steps)
        reversed_steps = tuple(reversed(self.configured_steps))
        windows = []
        for index in range(self.window_count):
            start = max(0, index - step_count + 1)
            end = min(self.block_count - 1, index)
            current_blocks = end - start + 1
            if current_blocks == step_count:
                timesteps = reversed_steps
            elif start == 0:
                timesteps = reversed_steps[-current_blocks:]
            elif end == self.block_count - 1:
                timesteps = reversed_steps[:current_blocks]
            else:  # pragma: no cover - construction above makes this unreachable
                raise AssertionError("invalid rolling window")
            windows.append(RollingWindow(index, start, end, timesteps))
        return tuple(windows)
