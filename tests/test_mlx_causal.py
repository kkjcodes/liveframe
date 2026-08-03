from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")

from liveframe.backends.mlx_causal import (  # noqa: E402
    CausalWanForward,
    MLXKVCache,
    RollingForcingKVCache,
    causal_rope_apply,
)
from liveframe.mlx_pipeline_parity import _require_context_noise  # noqa: E402
from liveframe.mlx_rolling_parity import (  # noqa: E402
    FINAL_COSINE_CANARY,
    _runtime_provenance,
    _write_candidate_latents,
)


def test_mlx_cache_refresh_overwrites_speculative_block() -> None:
    cache = MLXKVCache(capacity=8, max_attention_size=8)
    first = mx.ones((1, 2, 1, 1))
    clean = mx.full((1, 2, 1, 1), 7.0)

    cache.update(first, first, current_start=0)
    visible_key, visible_value = cache.update(clean, clean, current_start=0)

    assert cache.global_end == 2
    assert cache.local_end == 2
    assert visible_key.tolist() == [[[[7.0]], [[7.0]]]]
    assert visible_value.tolist() == [[[[7.0]], [[7.0]]]]


def test_mlx_local_cache_rolls_live_arrays_and_preserves_sink() -> None:
    cache = MLXKVCache(
        capacity=8,
        max_attention_size=8,
        local_attention=True,
        sink_tokens=2,
    )
    first = mx.arange(8).reshape(1, 8, 1, 1).astype(mx.float32)
    incoming = mx.arange(8, 12).reshape(1, 4, 1, 1).astype(mx.float32)

    cache.update(first, first, current_start=0)
    visible_key, _ = cache.update(incoming, incoming, current_start=8)

    assert cache.global_end == 12
    assert cache.local_end == 8
    assert visible_key.reshape(-1).tolist() == [0.0, 1.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]


def test_rolling_forcing_cache_evicts_working_block_but_preserves_anchor() -> None:
    cache = RollingForcingKVCache(capacity=6, max_attention_size=6, block_tokens=3)
    freqs = mx.zeros((12, 3, 2), dtype=mx.float32)
    freqs[:, :, 0] = 1

    def append(start: int) -> None:
        value = mx.arange(start, start + 3).reshape(1, 3, 1, 1).astype(mx.float32)
        key = mx.repeat(value, 6, axis=-1)
        cache.update(
            key,
            key,
            value,
            current_start=start,
            grid_sizes=[(3, 1, 1)],
            freqs=freqs,
            updating_cache=True,
        )

    append(0)
    append(3)
    append(6)

    assert cache.global_end == 9
    assert cache.local_end == 6
    assert cache.v is not None
    assert cache.v.reshape(-1).tolist() == [0.0, 1.0, 2.0, 6.0, 7.0, 8.0]


def test_causal_rope_temporal_offset_changes_only_temporal_pairs() -> None:
    # head_dim=6 gives one complex pair to each of temporal, height, and width.
    x = mx.ones((1, 1, 1, 6), dtype=mx.float32)
    angles = mx.zeros((4, 3, 2), dtype=mx.float32)
    angles[:, :, 0] = 1.0
    angles[1, 0, 0] = 0.0
    angles[1, 0, 1] = 1.0

    at_zero = causal_rope_apply(x, [(1, 1, 1)], angles, start_frame=0)
    at_one = causal_rope_apply(x, [(1, 1, 1)], angles, start_frame=1)

    assert at_zero.tolist() == [[[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]]
    assert at_one[0, 0, 0, :2].tolist() == [-1.0, 1.0]
    assert at_one[0, 0, 0, 2:].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_causal_forward_rejects_empty_layer_limit() -> None:
    class Model:
        def __init__(self) -> None:
            self.blocks = [object()]

    with pytest.raises(ValueError, match="layer_limit"):
        CausalWanForward(Model(), layer_limit=0)


def test_causal_forward_reset_preserves_rolling_cache_policy() -> None:
    class Model:
        def __init__(self) -> None:
            self.blocks = [object(), object()]

    causal = CausalWanForward(
        Model(),
        cache_capacity=8,
        max_attention_size=6,
        local_attention=True,
        sink_tokens=2,
    )
    causal.caches[0].global_end = 12
    causal.reset()

    assert len(causal.caches) == 2
    assert causal.caches[0].global_end == 0
    assert causal.caches[0].capacity == 8
    assert causal.caches[0].max_attention_size == 6
    assert causal.caches[0].local_attention is True
    assert causal.caches[0].sink_tokens == 2


def test_pipeline_parity_requires_explicit_zero_context_noise() -> None:
    assert _require_context_noise({"context_noise": 0}) == 0
    assert (
        _require_context_noise({}, fixture_id="cfpp-1step-native-21block-h100") == 0
    )
    with pytest.raises(ValueError, match="explicitly recorded as 0"):
        _require_context_noise({})
    with pytest.raises(ValueError, match="found 3"):
        _require_context_noise({"context_noise": 3})


def test_rolling_parity_records_execution_device() -> None:
    provenance = _runtime_provenance()

    assert provenance["mlx_version"]
    assert provenance["default_device"] == str(mx.default_device())
    assert provenance["runtime"] in {"mlx-metal", "mlx-cpu"}
    assert provenance["platform"]["system"]
    assert provenance["platform"]["machine"]


def test_rolling_numerical_canary_remains_frozen() -> None:
    assert FINAL_COSINE_CANARY == 0.999


def test_candidate_latents_are_exact_fixture_layout_bfloat16(tmp_path) -> None:
    fixture_manifest = tmp_path / "fixture.json"
    model = tmp_path / "model.safetensors"
    fixture_manifest.write_bytes(b"fixture")
    model.write_bytes(b"model")
    output = tmp_path / "candidate.bin"
    value = mx.arange(24).reshape(2, 3, 2, 2).astype(mx.bfloat16)

    artifact = _write_candidate_latents(
        output,
        value,
        fixture_id="fixture-id",
        fixture_manifest=fixture_manifest,
        model_path=model,
    )

    metadata = json.loads((tmp_path / "candidate.bin.json").read_text())
    assert artifact["shape"] == [1, 3, 2, 2, 2]
    assert artifact["layout"] == "B,F,C,H,W"
    assert artifact["dtype"] == "bfloat16"
    assert metadata["sha256"] == artifact["sha256"]
    assert output.stat().st_size == 48