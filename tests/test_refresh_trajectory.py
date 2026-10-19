from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from liveframe.refresh_proof import MODEL_LAYERS, TOKENS_PER_FRAME  # noqa: E402
from liveframe.refresh_trajectory import (  # noqa: E402
    PairedStochasticFrame,
    TimedRefresh,
    run_paired_refresh_trajectory,
)
from liveframe.torch_refresh_rerope import FramewiseRefreshHarness  # noqa: E402

DIGEST = "sha256:" + "a" * 64


class FakeAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 1
        self.head_dim = 2
        self.q = torch.nn.Identity()
        self.k = torch.nn.Identity()
        self.v = torch.nn.Identity()
        self.norm_q = torch.nn.Identity()
        self.norm_k = torch.nn.Identity()
        self.o = torch.nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        _seq_lens: Any,
        grid_sizes: Any,
        freqs: Any,
        _block_mask: Any,
        kv_cache: dict[str, torch.Tensor],
        current_start: int,
        _cache_start: int | None,
    ) -> torch.Tensor:
        query = self.q(x).view(1, TOKENS_PER_FRAME, 1, 2)
        key = self.norm_k(self.k(x)).view_as(query)
        value = self.v(x).view_as(query)
        frame = current_start // TOKENS_PER_FRAME
        roped_key = rope(key, grid_sizes, freqs, start_frame=frame)
        local_end = int(kv_cache["local_end_index"].item())
        global_end = int(kv_cache["global_end_index"].item())
        current_end = current_start + TOKENS_PER_FRAME
        end = local_end + current_end - global_end
        start = end - TOKENS_PER_FRAME
        kv_cache["k"][:, start:end].copy_(roped_key)
        kv_cache["v"][:, start:end].copy_(value)
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(end)
        return attend(query, kv_cache["k"][:, :end], kv_cache["v"][:, :end]).flatten(2)


def rope(
    value: torch.Tensor,
    _grid_sizes: Any,
    _freqs: Any,
    *,
    start_frame: int,
) -> torch.Tensor:
    return value + float(start_frame)


def attend(query: torch.Tensor, _key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return query + value[:, -TOKENS_PER_FRAME:]


def cache() -> dict[str, torch.Tensor]:
    return {
        "k": torch.zeros((1, 6 * TOKENS_PER_FRAME, 1, 2)),
        "v": torch.zeros((1, 6 * TOKENS_PER_FRAME, 1, 2)),
        "global_end_index": torch.tensor([0]),
        "local_end_index": torch.tensor([0]),
    }


@dataclass
class State:
    caches: list[dict[str, torch.Tensor]]
    log: list[str]
    name: str


def identities(state: State) -> set[int]:
    return {
        id(value)
        for layer in state.caches
        for value in layer.values()
    } | {id(state.caches), id(state.log)}


def frames(count: int = 4) -> list[PairedStochasticFrame[torch.Tensor]]:
    return [
        PairedStochasticFrame(
            frame_index=index,
            reference=torch.tensor([float(index)]),
            candidate=torch.tensor([float(index)]),
            reference_sha256=DIGEST,
            candidate_sha256=DIGEST,
        )
        for index in range(count)
    ]


def test_runner_sequences_independent_periodic_anchor_trajectories() -> None:
    modules = [FakeAttention() for _ in range(MODEL_LAYERS)]
    harness = FramewiseRefreshHarness(modules, rope_apply=rope, attention=attend)
    reference = State([cache() for _ in modules], [], "reference")
    candidate = State([cache() for _ in modules], [], "candidate")
    call_log: list[tuple[str, int, str | None]] = []

    def denoise(state: State, frame: int, stochastic: torch.Tensor) -> torch.Tensor:
        operation = harness._operation[0] if harness._operation is not None else None
        call_log.append((f"denoise-{state.name}", frame, operation))
        return torch.full((1, TOKENS_PER_FRAME, 2), stochastic.item() + 1)

    def refresh(
        state: State, frame: int, prediction: torch.Tensor
    ) -> TimedRefresh[torch.Tensor]:
        operation = harness._operation[0] if harness._operation is not None else None
        call_log.append((f"refresh-{state.name}", frame, operation))
        output = prediction
        for module, kv_cache in zip(modules, state.caches, strict=True):
            output = module(
                prediction,
                object(),
                torch.tensor([[1, 1, TOKENS_PER_FRAME]]),
                object(),
                object(),
                kv_cache,
                frame * TOKENS_PER_FRAME,
                None,
            )
        return TimedRefresh(output, frame / 1000)

    def observe(
        _reference: State,
        _candidate: State,
        result: Any,
    ) -> None:
        if result.source_frame is not None:
            harness.registry.release_destination(result.frame_index)

    with harness:
        results = run_paired_refresh_trajectory(
            harness,
            frames(),
            reference_state=reference,
            candidate_state=candidate,
            mutable_identities=identities,
            denoise_reference=denoise,
            denoise_candidate=denoise,
            refresh_reference=refresh,
            refresh_candidate=refresh,
            observe_frame=observe,
        )

    assert [result.source_frame for result in results] == [None, 0, 0, None]
    assert harness.registry.substitution_count == 2 * MODEL_LAYERS
    assert not harness.registry.substitutions
    assert {tile for _, tile in harness.registry.sources} == {3}
    assert len(harness.registry.sources) == MODEL_LAYERS
    assert all(operation is None for name, _, operation in call_log if name.startswith("denoise"))
    assert [
        (name, frame, operation)
        for name, frame, operation in call_log
        if name.startswith("refresh")
    ] == [
        ("refresh-reference", 0, None),
        ("refresh-candidate", 0, "source"),
        ("refresh-reference", 1, "reference"),
        ("refresh-candidate", 1, "candidate"),
        ("refresh-reference", 2, "reference"),
        ("refresh-candidate", 2, "candidate"),
        ("refresh-reference", 3, None),
        ("refresh-candidate", 3, "source"),
    ]
    assert not (identities(reference) & identities(candidate))


def test_runner_rejects_stochastic_mismatch_before_execution() -> None:
    modules = [FakeAttention() for _ in range(MODEL_LAYERS)]
    harness = FramewiseRefreshHarness(modules, rope_apply=rope, attention=attend)
    paired = frames(1)
    paired[0] = PairedStochasticFrame(
        0, torch.tensor([0.0]), torch.tensor([0.0]), DIGEST, "sha256:" + "b" * 64
    )
    state_a = State([cache() for _ in modules], [], "a")
    state_b = State([cache() for _ in modules], [], "b")

    with pytest.raises(ValueError, match="stochastic bytes differ"):
        run_paired_refresh_trajectory(
            harness,
            paired,
            reference_state=state_a,
            candidate_state=state_b,
            mutable_identities=identities,
            denoise_reference=lambda *_: torch.zeros(1),
            denoise_candidate=lambda *_: torch.zeros(1),
            refresh_reference=lambda *_: TimedRefresh(torch.zeros(1), 0.0),
            refresh_candidate=lambda *_: TimedRefresh(torch.zeros(1), 0.0),
        )


def test_runner_rejects_aliased_mutable_state() -> None:
    modules = [FakeAttention() for _ in range(MODEL_LAYERS)]
    harness = FramewiseRefreshHarness(modules, rope_apply=rope, attention=attend)
    shared_cache = [cache() for _ in modules]
    state_a = State(shared_cache, [], "a")
    state_b = State(shared_cache, [], "b")

    with pytest.raises(ValueError, match="mutable state is aliased"):
        run_paired_refresh_trajectory(
            harness,
            frames(1),
            reference_state=state_a,
            candidate_state=state_b,
            mutable_identities=identities,
            denoise_reference=lambda *_: torch.zeros(1),
            denoise_candidate=lambda *_: torch.zeros(1),
            refresh_reference=lambda *_: TimedRefresh(torch.zeros(1), 0.0),
            refresh_candidate=lambda *_: TimedRefresh(torch.zeros(1), 0.0),
        )


def test_runner_can_release_results_after_observation() -> None:
    modules = [FakeAttention() for _ in range(MODEL_LAYERS)]
    harness = FramewiseRefreshHarness(modules, rope_apply=rope, attention=attend)
    state_a = State([cache() for _ in modules], [], "a")
    state_b = State([cache() for _ in modules], [], "b")
    observed: list[int] = []

    def refresh(
        state: State, frame: int, prediction: torch.Tensor
    ) -> TimedRefresh[torch.Tensor]:
        output = prediction
        for module, kv_cache in zip(modules, state.caches, strict=True):
            output = module(
                prediction,
                object(),
                torch.tensor([[1, 1, TOKENS_PER_FRAME]]),
                object(),
                object(),
                kv_cache,
                frame * TOKENS_PER_FRAME,
                None,
            )
        return TimedRefresh(output, 0.0)

    with harness:
        results = run_paired_refresh_trajectory(
            harness,
            frames(),
            reference_state=state_a,
            candidate_state=state_b,
            mutable_identities=identities,
            denoise_reference=lambda _state, _frame, value: torch.full(
                (1, TOKENS_PER_FRAME, 2), value.item() + 1
            ),
            denoise_candidate=lambda _state, _frame, value: torch.full(
                (1, TOKENS_PER_FRAME, 2), value.item() + 1
            ),
            refresh_reference=refresh,
            refresh_candidate=refresh,
            observe_frame=lambda _left, _right, result: observed.append(
                result.frame_index
            ),
            retain_results=False,
        )

    assert results == []
    assert observed == [0, 1, 2, 3]