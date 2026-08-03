"""CF++ frame-wise scheduler transition contract."""

from __future__ import annotations

from dataclasses import dataclass

TRAIN_TIMESTEPS = 1000
DEFAULT_TIMESTEP_SHIFT = 5.0
PROTECTED_FIRST_BLOCK_STEPS = (1000, 750, 500, 250)
STUDENT_STEP_IDS = {1: (1000,), 2: (1000, 500)}


def shifted_sigma(
    unshifted_sigma: float, *, shift: float = DEFAULT_TIMESTEP_SHIFT
) -> float:
    """Apply the Wan/CF++ rational flow-matching timestep shift."""

    if not 0.0 <= unshifted_sigma <= 1.0:
        raise ValueError("unshifted sigma must be in [0, 1]")
    if shift <= 0.0:
        raise ValueError("timestep shift must be positive")
    return shift * unshifted_sigma / (1.0 + (shift - 1.0) * unshifted_sigma)


def warp_configured_timestep(
    configured_timestep: int, *, shift: float = DEFAULT_TIMESTEP_SHIFT
) -> float:
    """Match upstream ``timesteps[1000 - configured_timestep]`` exactly."""

    if not 0 <= configured_timestep <= TRAIN_TIMESTEPS:
        raise ValueError("configured timestep must be in [0, 1000]")
    unshifted_sigma = configured_timestep / TRAIN_TIMESTEPS
    return shifted_sigma(unshifted_sigma, shift=shift) * TRAIN_TIMESTEPS


def add_noise_coefficients(
    warped_timestep: float,
) -> tuple[float, float]:
    """Return clean/noise coefficients for upstream ``scheduler.add_noise``."""

    if not 0.0 <= warped_timestep <= TRAIN_TIMESTEPS:
        raise ValueError("warped timestep must be in [0, 1000]")
    sigma = warped_timestep / TRAIN_TIMESTEPS
    return 1.0 - sigma, sigma


@dataclass(frozen=True, slots=True)
class SchedulerTransition:
    """One CF++ model evaluation at a particular denoising step."""

    transition_id: str
    latent_block: int
    step_index: int
    step_count: int
    protected_first_block: bool


@dataclass(frozen=True, slots=True)
class SchedulerNoiseDraw:
    """A random re-noising draw between two model evaluations."""

    transition_id: str
    latent_block: int
    source_step_index: int
    target_step_index: int
    step_count: int
    protected_first_block: bool


class FramewiseSchedule:
    """Describe denoising, re-noising, and refresh work in execution order."""

    def __init__(self, latent_blocks: int = 21, student_steps: int = 2) -> None:
        if latent_blocks < 1:
            raise ValueError("latent_blocks must be positive")
        if student_steps not in {1, 2}:
            raise ValueError("student_steps must be 1 or 2")
        self.latent_blocks = latent_blocks
        self.student_steps = student_steps

    def denoising_transitions(self) -> tuple[SchedulerTransition, ...]:
        """Return denoising evaluations, including each block's final step."""

        result: list[SchedulerTransition] = []
        for block in range(self.latent_blocks):
            step_count = 4 if block == 0 else self.student_steps
            for step in range(step_count):
                result.append(
                    SchedulerTransition(
                        transition_id=f"block-{block:04d}.step-{step:02d}-of-{step_count:02d}",
                        latent_block=block,
                        step_index=step,
                        step_count=step_count,
                        protected_first_block=block == 0,
                    )
                )
        return tuple(result)

    def transitions(self) -> tuple[SchedulerTransition, ...]:
        """Compatibility alias for :meth:`denoising_transitions`."""

        return self.denoising_transitions()

    def noise_draws(self) -> tuple[SchedulerNoiseDraw, ...]:
        """Return re-noising draws; upstream draws only between denoising steps."""

        result: list[SchedulerNoiseDraw] = []
        for block in range(self.latent_blocks):
            step_count = 4 if block == 0 else self.student_steps
            for source_step in range(step_count - 1):
                target_step = source_step + 1
                result.append(
                    SchedulerNoiseDraw(
                        transition_id=(
                            f"block-{block:04d}.renoise-{source_step:02d}-to-"
                            f"{target_step:02d}-of-{step_count:02d}"
                        ),
                        latent_block=block,
                        source_step_index=source_step,
                        target_step_index=target_step,
                        step_count=step_count,
                        protected_first_block=block == 0,
                    )
                )
        return tuple(result)

    def configured_timesteps(self, latent_block: int) -> tuple[int, ...]:
        """Return the checkpoint config's unwarped step IDs for one block."""

        if not 0 <= latent_block < self.latent_blocks:
            raise ValueError("latent block is outside this schedule")
        if latent_block == 0:
            return PROTECTED_FIRST_BLOCK_STEPS
        return STUDENT_STEP_IDS[self.student_steps]

    def warped_timesteps(
        self, latent_block: int, *, shift: float = DEFAULT_TIMESTEP_SHIFT
    ) -> tuple[float, ...]:
        return tuple(
            warp_configured_timestep(timestep, shift=shift)
            for timestep in self.configured_timesteps(latent_block)
        )

    @property
    def denoising_evaluation_count(self) -> int:
        """Transformer forwards used by the denoising schedules alone."""

        return 4 + (self.latent_blocks - 1) * self.student_steps

    @property
    def refresh_pass_count(self) -> int:
        """Clean-context KV refresh forwards, one after every latent block."""

        return self.latent_blocks

    @property
    def total_forward_pass_count(self) -> int:
        """All transformer forwards required by upstream causal inference."""

        return self.denoising_evaluation_count + self.refresh_pass_count

    @property
    def model_evaluation_count(self) -> int:
        """Backward-compatible name for total, wall-clock-relevant forwards."""

        return self.total_forward_pass_count

    @property
    def stochastic_draw_count(self) -> int:
        return 3 + (self.latent_blocks - 1) * (self.student_steps - 1)
