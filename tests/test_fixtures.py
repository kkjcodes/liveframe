from __future__ import annotations

from pathlib import Path

import pytest

from liveframe.fixtures import FixtureError, load_fixture, write_fixture
from liveframe.scheduler import (
    FramewiseSchedule,
    add_noise_coefficients,
    warp_configured_timestep,
)


def test_fixture_roundtrip_binds_initial_and_scheduler_noise(tmp_path: Path) -> None:
    schedule = FramewiseSchedule(latent_blocks=2, student_steps=2)
    noise_draws = schedule.noise_draws()
    tensors = [("initial-noise", "initial_noise", bytes(8), (2,), "float32", None)]
    tensors.extend(
        (
            f"noise-{index}",
            "scheduler_noise",
            bytes([index]) * 8,
            (2,),
            "float32",
            transition.transition_id,
        )
        for index, transition in enumerate(noise_draws, start=1)
    )

    fixture = write_fixture(
        tmp_path,
        fixture_id="synthetic-h100-golden",
        source_runtime="pytorch-cuda",
        source_revision="runtime-revision",
        model_artifact_id="causal-forcing-plus-plus-framewise-2step",
        tensors=tensors,
    )

    assert len(fixture.tensors) == 1 + schedule.stochastic_draw_count
    assert load_fixture(tmp_path) == fixture


def test_fixture_detects_tensor_tampering(tmp_path: Path) -> None:
    write_fixture(
        tmp_path,
        fixture_id="fixture-1",
        source_runtime="pytorch",
        source_revision="revision",
        model_artifact_id="model",
        tensors=[("initial", "initial_noise", bytes(4), (1,), "float32", None)],
    )
    (tmp_path / "initial.bin").write_bytes(b"evil")

    with pytest.raises(FixtureError, match="checksum mismatch"):
        load_fixture(tmp_path)


def test_fixture_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(FixtureError, match="unsafe tensor name"):
        write_fixture(
            tmp_path,
            fixture_id="fixture-1",
            source_runtime="pytorch",
            source_revision="revision",
            model_artifact_id="model",
            tensors=[("../initial", "initial_noise", bytes(4), (1,), "float32", None)],
        )


def test_framewise_schedule_protects_first_block() -> None:
    one_step = FramewiseSchedule(latent_blocks=21, student_steps=1)
    two_step = FramewiseSchedule(latent_blocks=21, student_steps=2)

    assert one_step.denoising_evaluation_count == 24
    assert two_step.denoising_evaluation_count == 44
    assert one_step.refresh_pass_count == 21
    assert two_step.refresh_pass_count == 21
    assert one_step.total_forward_pass_count == 45
    assert two_step.total_forward_pass_count == 65
    assert one_step.model_evaluation_count == 45
    assert two_step.model_evaluation_count == 65
    assert one_step.stochastic_draw_count == 3
    assert two_step.stochastic_draw_count == 23
    assert len(one_step.noise_draws()) == 3
    assert len(two_step.noise_draws()) == 23
    assert all(item.protected_first_block for item in two_step.denoising_transitions()[:4])
    assert all(
        not item.protected_first_block for item in two_step.denoising_transitions()[4:]
    )
    assert all(item.protected_first_block for item in two_step.noise_draws()[:3])
    assert all(not item.protected_first_block for item in two_step.noise_draws()[3:])


def test_framewise_schedule_matches_upstream_shifted_config() -> None:
    schedule = FramewiseSchedule(latent_blocks=21, student_steps=2)

    assert schedule.configured_timesteps(0) == (1000, 750, 500, 250)
    assert schedule.configured_timesteps(1) == (1000, 500)
    assert schedule.warped_timesteps(0) == pytest.approx(
        (1000.0, 937.5, 833.3333333333334, 625.0)
    )
    assert schedule.warped_timesteps(1) == pytest.approx((1000.0, 833.3333333333334))
    assert warp_configured_timestep(0) == 0.0


def test_add_noise_coefficients_use_warped_sigma() -> None:
    timestep = warp_configured_timestep(500)

    clean, noise = add_noise_coefficients(timestep)

    assert clean == pytest.approx(1.0 / 6.0)
    assert noise == pytest.approx(5.0 / 6.0)
