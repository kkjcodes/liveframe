"""Frozen configuration and fail-closed preflight checks for CF++ MLX timing."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any

from liveframe.scheduler import FramewiseSchedule

_SWAP_PATTERN = re.compile(
    r"total\s*=\s*(?P<total>[0-9.]+)M\s+"
    r"used\s*=\s*(?P<used>[0-9.]+)M\s+"
    r"free\s*=\s*(?P<free>[0-9.]+)M"
)
_FREE_PATTERN = re.compile(r"System-wide memory free percentage:\s*(?P<free>[0-9.]+)%")


@dataclass(frozen=True, slots=True)
class CFPPBenchmarkConfig:
    latent_blocks: int = 21
    student_steps: int = 1
    playback_fps: int = 16
    max_start_swap_mib: float = 128.0
    max_run_swap_delta_mib: float = 128.0
    min_start_free_percent: float = 25.0
    cache_limit_gib: float = 1.0

    def __post_init__(self) -> None:
        FramewiseSchedule(
            latent_blocks=self.latent_blocks, student_steps=self.student_steps
        )
        if self.playback_fps <= 0:
            raise ValueError("playback_fps must be positive")
        if self.max_start_swap_mib < 0:
            raise ValueError("max_start_swap_mib must be nonnegative")
        if self.max_run_swap_delta_mib < 0:
            raise ValueError("max_run_swap_delta_mib must be nonnegative")
        if not 0 <= self.min_start_free_percent <= 100:
            raise ValueError("min_start_free_percent must be in [0, 100]")
        if self.cache_limit_gib < 0:
            raise ValueError("cache_limit_gib must be nonnegative")

    @property
    def pixel_frames(self) -> int:
        return 4 * (self.latent_blocks - 1) + 1

    @property
    def playback_seconds(self) -> float:
        return self.pixel_frames / self.playback_fps

    @property
    def schedule(self) -> FramewiseSchedule:
        return FramewiseSchedule(
            latent_blocks=self.latent_blocks, student_steps=self.student_steps
        )

    @property
    def transformer_forwards(self) -> int:
        return self.schedule.denoising_evaluation_count + self.latent_blocks


def parse_swap_usage(value: str) -> dict[str, float]:
    match = _SWAP_PATTERN.search(value)
    if match is None:
        raise ValueError("could not parse vm.swapusage output")
    return {name + "_mib": float(match.group(name)) for name in ("total", "used", "free")}


def parse_free_memory_percent(value: str) -> float:
    match = _FREE_PATTERN.search(value)
    if match is None:
        raise ValueError("could not parse memory_pressure output")
    return float(match.group("free"))


def require_clean_preflight(
    *, swap_used_mib: float, free_memory_percent: float, config: CFPPBenchmarkConfig
) -> None:
    failures = []
    if swap_used_mib > config.max_start_swap_mib:
        failures.append(
            f"start swap {swap_used_mib:.2f} MiB exceeds "
            f"{config.max_start_swap_mib:.2f} MiB"
        )
    if free_memory_percent < config.min_start_free_percent:
        failures.append(
            f"free memory {free_memory_percent:.2f}% is below "
            f"{config.min_start_free_percent:.2f}%"
        )
    if failures:
        raise RuntimeError("CF++ benchmark preflight failed: " + "; ".join(failures))


def require_swap_stable(*, swap_delta_mib: float, config: CFPPBenchmarkConfig) -> None:
    if swap_delta_mib > config.max_run_swap_delta_mib:
        raise RuntimeError(
            f"CF++ benchmark created {swap_delta_mib:.2f} MiB of swap, exceeding "
            f"{config.max_run_swap_delta_mib:.2f} MiB"
        )


def median_fields(trials: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, float]:
    if not trials:
        raise ValueError("at least one measured trial is required")
    return {
        field: statistics.median(float(trial[field]) for trial in trials)
        for field in fields
    }