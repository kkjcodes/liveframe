from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from liveframe.refresh_proof import MODEL_LAYERS, TOKENS_PER_FRAME  # noqa: E402
from liveframe.torch_refresh_rerope import (  # noqa: E402
    FramewiseRefreshHarness,
    build_pinned_framewise_refresh_harness,
)


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
        self.local_attn_size = 21
        self.sink_size = 0
        self.max_attention_size = 21 * TOKENS_PER_FRAME
        self.k_calls = 0
        self.v_calls = 0
        self.k.register_forward_hook(lambda *_: setattr(self, "k_calls", self.k_calls + 1))
        self.v.register_forward_hook(lambda *_: setattr(self, "v_calls", self.v_calls + 1))

    def forward(
        self,
        x: torch.Tensor,
        _seq_lens: object,
        grid_sizes: object,
        freqs: object,
        _block_mask: object,
        kv_cache: dict[str, torch.Tensor],
        current_start: int,
        _cache_start: int | None,
    ) -> torch.Tensor:
        query = self.q(x).view(1, TOKENS_PER_FRAME, 1, 2)
        key = self.norm_k(self.k(x)).view_as(query)
        value = self.v(x).view_as(query)
        start_frame = current_start // TOKENS_PER_FRAME
        key = rope(key, grid_sizes, freqs, start_frame=start_frame)
        start = start_frame * TOKENS_PER_FRAME
        end = start + TOKENS_PER_FRAME
        kv_cache["k"][:, start:end].copy_(key)
        kv_cache["v"][:, start:end].copy_(value)
        kv_cache["global_end_index"].fill_(end)
        kv_cache["local_end_index"].fill_(end)
        return attend(query, kv_cache["k"][:, :end], kv_cache["v"][:, :end]).flatten(2)


def rope(
    value: torch.Tensor,
    _grid_sizes: object,
    _freqs: object,
    *,
    start_frame: int,
) -> torch.Tensor:
    return value + float(start_frame)


def attend(query: torch.Tensor, _key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return query + value[:, -TOKENS_PER_FRAME:]


def cache() -> dict[str, torch.Tensor]:
    return {
        "k": torch.zeros((1, 3 * TOKENS_PER_FRAME, 1, 2)),
        "v": torch.zeros((1, 3 * TOKENS_PER_FRAME, 1, 2)),
        "global_end_index": torch.tensor([0]),
        "local_end_index": torch.tensor([0]),
    }


def call(module: FakeAttention, value: float, start_frame: int, kv_cache: object) -> None:
    module(
        torch.full((1, TOKENS_PER_FRAME, 2), value),
        object(),
        torch.tensor([[1, 1, TOKENS_PER_FRAME]]),
        object(),
        object(),
        kv_cache,
        start_frame * TOKENS_PER_FRAME,
        None,
    )


def test_harness_candidate_skips_key_value_projections_and_restores_methods() -> None:
    modules = [FakeAttention() for _ in range(MODEL_LAYERS)]
    original = modules[0].forward
    harness = FramewiseRefreshHarness(modules, rope_apply=rope, attention=attend)
    caches = [cache() for _ in modules]

    with harness:
        with harness.operation("source", tile_index=0, start_frame=0):
            for module, kv_cache in zip(modules, caches, strict=True):
                call(module, 1.0, 0, kv_cache)
        with harness.operation("reference", tile_index=1, start_frame=1):
            for module, kv_cache in zip(modules, caches, strict=True):
                call(module, 2.0, 1, kv_cache)
        harness.registry.require_layers(source_tile_index=0, destination_start_frame=1)
        projection_counts = [(module.k_calls, module.v_calls) for module in modules]
        with harness.operation("candidate", tile_index=0, start_frame=1):
            for module, kv_cache in zip(modules, caches, strict=True):
                call(module, 3.0, 1, kv_cache)

        assert [(module.k_calls, module.v_calls) for module in modules] == projection_counts
        assert len(harness.registry.substitutions) == MODEL_LAYERS

    assert modules[0].forward == original


def test_harness_rejects_missing_layers_and_restores_after_failure() -> None:
    modules = [FakeAttention() for _ in range(MODEL_LAYERS)]
    original = modules[0].forward
    harness = FramewiseRefreshHarness(modules, rope_apply=rope, attention=attend)

    with (
        pytest.raises(RuntimeError, match="boom"),
        harness,
        harness.operation("source", tile_index=0, start_frame=0),
    ):
        call(modules[0], 1.0, 0, cache())
        raise RuntimeError("boom")
    assert modules[0].forward == original
    with pytest.raises(ValueError, match="every layer"):
        harness.registry.require_layers(source_tile_index=0, destination_start_frame=1)


def test_pinned_generator_adapter_requires_exact_long_horizon_configuration() -> None:
    modules = [FakeAttention() for _ in range(MODEL_LAYERS)]
    generator = SimpleNamespace(
        model=SimpleNamespace(
            blocks=[SimpleNamespace(self_attn=module) for module in modules],
            local_attn_size=21,
        )
    )
    causal_model = SimpleNamespace(causal_rope_apply=rope, attention=attend)

    harness = build_pinned_framewise_refresh_harness(generator, causal_model)

    assert harness.attentions == modules
    generator.model.local_attn_size = -1
    with pytest.raises(ValueError, match="21-frame rolling context"):
        build_pinned_framewise_refresh_harness(generator, causal_model)


def test_pinned_generator_adapter_rejects_layer_and_call_surface_drift() -> None:
    modules = [FakeAttention() for _ in range(MODEL_LAYERS)]
    generator = SimpleNamespace(
        model=SimpleNamespace(
            blocks=[SimpleNamespace(self_attn=module) for module in modules],
            local_attn_size=21,
        )
    )
    modules[12].sink_size = 3
    with pytest.raises(ValueError, match="layer 12 sink mismatch"):
        build_pinned_framewise_refresh_harness(
            generator,
            SimpleNamespace(causal_rope_apply=rope, attention=attend),
        )

    modules[12].sink_size = 0
    with pytest.raises(ValueError, match="call surface mismatch"):
        build_pinned_framewise_refresh_harness(generator, SimpleNamespace())