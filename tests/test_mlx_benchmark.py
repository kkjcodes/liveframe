from __future__ import annotations

import pytest

from liveframe.mlx_benchmark import MLXBenchmarkConfig


def test_frozen_m5_clip_configuration() -> None:
    config = MLXBenchmarkConfig()

    assert config.latent_frames == 21
    assert config.pixel_frames == 81
    assert config.playback_seconds == pytest.approx(5.0625)
    assert config.schedule.block_count == 7
    assert config.schedule.window_count == 10
    assert config.schedule.transformer_forward_count == 20


def test_benchmark_configuration_requires_complete_blocks() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MLXBenchmarkConfig(latent_frames=20)