from __future__ import annotations

from liveframe.decoders.taew21_mlx import (
    CONVOLUTION_MODES,
    EXPECTED_TEMPORAL_UPSCALE,
    FP32_MEMORY_LAYERS,
    MEMORY_LAYERS,
    SPATIAL_UPSAMPLE_LAYERS,
    STARTUP_FRAMES_TO_TRIM,
    TEMPORAL_GROWTH,
)


def test_taew21_schedule_matches_pinned_upstream_base_decoder() -> None:
    assert MEMORY_LAYERS == (3, 4, 5, 9, 10, 11, 15, 16, 17)
    assert FP32_MEMORY_LAYERS == (11, 17)
    assert CONVOLUTION_MODES == ("native", "fp32-im2col")
    assert SPATIAL_UPSAMPLE_LAYERS == (6, 12, 18)
    assert TEMPORAL_GROWTH == {7: 1, 13: 2, 19: 2}
    assert EXPECTED_TEMPORAL_UPSCALE == 4
    assert STARTUP_FRAMES_TO_TRIM == 3
    assert 21 * EXPECTED_TEMPORAL_UPSCALE - STARTUP_FRAMES_TO_TRIM == 81