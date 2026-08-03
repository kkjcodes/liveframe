from __future__ import annotations

import pytest

from liveframe.mlx_cfpp_benchmark import (
    CFPPBenchmarkConfig,
    median_fields,
    parse_free_memory_percent,
    parse_swap_usage,
    require_clean_preflight,
    require_swap_stable,
)


def test_frozen_cfpp_clip_configuration() -> None:
    config = CFPPBenchmarkConfig()

    assert config.latent_blocks == 21
    assert config.student_steps == 1
    assert config.pixel_frames == 81
    assert config.playback_seconds == pytest.approx(5.0625)
    assert config.schedule.denoising_evaluation_count == 24
    assert config.transformer_forwards == 45
    assert config.cache_limit_gib == 1.0


def test_macos_memory_telemetry_parsers() -> None:
    swap = parse_swap_usage(
        "vm.swapusage: total = 6144.00M  used = 4793.25M  free = 1350.75M  (encrypted)"
    )

    assert swap == {"total_mib": 6144.0, "used_mib": 4793.25, "free_mib": 1350.75}
    assert parse_free_memory_percent("System-wide memory free percentage: 42%") == 42.0


def test_preflight_rejects_swap_and_low_free_memory() -> None:
    config = CFPPBenchmarkConfig()

    with pytest.raises(RuntimeError, match=r"swap.*free memory"):
        require_clean_preflight(
            swap_used_mib=512.0, free_memory_percent=20.0, config=config
        )


def test_run_rejects_material_swap_growth() -> None:
    with pytest.raises(RuntimeError, match=r"created 256\.00 MiB of swap"):
        require_swap_stable(swap_delta_mib=256.0, config=CFPPBenchmarkConfig())


def test_trial_summary_uses_medians() -> None:
    assert median_fields(
        [{"wall_seconds": 4.0}, {"wall_seconds": 2.0}, {"wall_seconds": 100.0}],
        ("wall_seconds",),
    ) == {"wall_seconds": 4.0}