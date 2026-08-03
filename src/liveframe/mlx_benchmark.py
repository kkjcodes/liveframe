"""Frozen configuration and summaries for MLX Rolling Forcing benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

from liveframe.rolling_forcing import RollingForcingSchedule


@dataclass(frozen=True, slots=True)
class MLXBenchmarkConfig:
    latent_frames: int = 21
    playback_fps: int = 16

    def __post_init__(self) -> None:
        RollingForcingSchedule(latent_frames=self.latent_frames)
        if self.playback_fps <= 0:
            raise ValueError("playback_fps must be positive")

    @property
    def pixel_frames(self) -> int:
        return 4 * (self.latent_frames - 1) + 1

    @property
    def playback_seconds(self) -> float:
        return self.pixel_frames / self.playback_fps

    @property
    def schedule(self) -> RollingForcingSchedule:
        return RollingForcingSchedule(latent_frames=self.latent_frames)
