"""Paired reference/candidate orchestration for framewise refresh proof runs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from liveframe.refresh_proof import (
    MODEL_LAYERS,
    is_full_refresh_frame,
    source_frame_for_destination,
)
from liveframe.torch_refresh_rerope import FramewiseRefreshHarness

StateT = TypeVar("StateT")
InputT = TypeVar("InputT")
PredictionT = TypeVar("PredictionT")
RefreshT = TypeVar("RefreshT")


@dataclass(frozen=True, slots=True)
class PairedStochasticFrame(Generic[InputT]):
    """One byte-identical stochastic input represented in two mutable-safe values."""

    frame_index: int
    reference: InputT
    candidate: InputT
    reference_sha256: str
    candidate_sha256: str


@dataclass(frozen=True, slots=True)
class TimedRefresh(Generic[RefreshT]):
    output: RefreshT
    seconds: float


@dataclass(frozen=True, slots=True)
class PairedFrameResult(Generic[PredictionT, RefreshT]):
    frame_index: int
    source_frame: int | None
    reference_prediction: PredictionT
    candidate_prediction: PredictionT
    reference_refresh: TimedRefresh[RefreshT]
    candidate_refresh: TimedRefresh[RefreshT]


def _require_digest(value: str, name: str) -> None:
    if not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"paired trajectory requires {name}")


def _require_independent_state(
    reference_state: StateT,
    candidate_state: StateT,
    mutable_identities: Callable[[StateT], set[int]],
) -> None:
    if reference_state is candidate_state:
        raise ValueError("paired trajectory states must be distinct")
    reference_ids = mutable_identities(reference_state)
    candidate_ids = mutable_identities(candidate_state)
    if not reference_ids or not candidate_ids:
        raise ValueError("paired trajectory must expose mutable state identities")
    if reference_ids & candidate_ids:
        raise ValueError("paired trajectory mutable state is aliased")


def run_paired_refresh_trajectory(
    harness: FramewiseRefreshHarness,
    frames: Sequence[PairedStochasticFrame[InputT]],
    *,
    reference_state: StateT,
    candidate_state: StateT,
    mutable_identities: Callable[[StateT], set[int]],
    denoise_reference: Callable[[StateT, int, InputT], PredictionT],
    denoise_candidate: Callable[[StateT, int, InputT], PredictionT],
    refresh_reference: Callable[[StateT, int, PredictionT], TimedRefresh[RefreshT]],
    refresh_candidate: Callable[[StateT, int, PredictionT], TimedRefresh[RefreshT]],
    observe_frame: Callable[
        [StateT, StateT, PairedFrameResult[PredictionT, RefreshT]], None
    ]
    | None = None,
    retain_results: bool = True,
) -> list[PairedFrameResult[PredictionT, RefreshT]]:
    """Run paired trajectories while restricting hooks to clean refresh forwards."""

    if not frames:
        raise ValueError("paired trajectory requires at least one frame")
    _require_independent_state(reference_state, candidate_state, mutable_identities)
    results: list[PairedFrameResult[PredictionT, RefreshT]] = []
    starting_substitutions = harness.registry.substitution_count

    for expected_index, frame in enumerate(frames):
        if frame.frame_index != expected_index:
            raise ValueError("paired trajectory frame indices must be contiguous from zero")
        _require_digest(frame.reference_sha256, "reference stochastic digest")
        _require_digest(frame.candidate_sha256, "candidate stochastic digest")
        if frame.reference_sha256 != frame.candidate_sha256:
            raise ValueError("paired trajectory stochastic bytes differ")

        reference_prediction = denoise_reference(
            reference_state, expected_index, frame.reference
        )
        candidate_prediction = denoise_candidate(
            candidate_state, expected_index, frame.candidate
        )

        if is_full_refresh_frame(expected_index):
            reference_refresh = refresh_reference(
                reference_state, expected_index, reference_prediction
            )
            with harness.operation(
                "source", tile_index=expected_index, start_frame=expected_index
            ):
                candidate_refresh = refresh_candidate(
                    candidate_state, expected_index, candidate_prediction
                )
            source_frame = None
        else:
            source_frame = source_frame_for_destination(expected_index)
            with harness.operation(
                "reference", tile_index=expected_index, start_frame=expected_index
            ):
                reference_refresh = refresh_reference(
                    reference_state, expected_index, reference_prediction
                )
            harness.registry.require_layers(
                source_tile_index=source_frame,
                destination_start_frame=expected_index,
            )
            before = harness.registry.substitution_count
            with harness.operation(
                "candidate", tile_index=source_frame, start_frame=expected_index
            ):
                candidate_refresh = refresh_candidate(
                    candidate_state, expected_index, candidate_prediction
                )
            if harness.registry.substitution_count - before != MODEL_LAYERS:
                raise ValueError("paired trajectory candidate did not substitute every layer")

        for name, timed in (
            ("reference", reference_refresh),
            ("candidate", candidate_refresh),
        ):
            if timed.seconds < 0:
                raise ValueError(f"paired trajectory {name} refresh timing is negative")
        _require_independent_state(reference_state, candidate_state, mutable_identities)
        frame_result = PairedFrameResult(
            frame_index=expected_index,
            source_frame=source_frame,
            reference_prediction=reference_prediction,
            candidate_prediction=candidate_prediction,
            reference_refresh=reference_refresh,
            candidate_refresh=candidate_refresh,
        )
        if observe_frame is not None:
            observe_frame(reference_state, candidate_state, frame_result)
        if retain_results:
            results.append(frame_result)

    scheduled_substitutions = sum(
        not is_full_refresh_frame(frame.frame_index) for frame in frames
    ) * MODEL_LAYERS
    if harness.registry.substitution_count - starting_substitutions != scheduled_substitutions:
        raise ValueError("paired trajectory substitution ancestry count mismatch")
    return results