from __future__ import annotations

import pytest

from liveframe.rolling_forcing import RollingForcingSchedule


def test_released_rolling_schedule_builds_diagonal_windows() -> None:
    schedule = RollingForcingSchedule(latent_frames=60)
    windows = schedule.windows()

    assert schedule.block_count == 20
    assert schedule.window_count == 23
    assert schedule.transformer_forward_count == 46
    assert [(row.start_block, row.end_block) for row in windows[:5]] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
    ]
    assert [row.configured_timesteps for row in windows[:4]] == [
        (1000,),
        (750, 1000),
        (500, 750, 1000),
        (250, 500, 750, 1000),
    ]
    assert [row.configured_timesteps for row in windows[-3:]] == [
        (250, 500, 750),
        (250, 500),
        (250,),
    ]
    assert schedule.scheduler_noise_draw_count == 60


def test_rolling_schedule_rejects_nonzero_context_noise() -> None:
    with pytest.raises(ValueError, match="context_noise=0"):
        RollingForcingSchedule(latent_frames=60, context_noise=1)


def test_rolling_schedule_requires_complete_three_frame_blocks() -> None:
    with pytest.raises(ValueError, match="divisible"):
        RollingForcingSchedule(latent_frames=61)
