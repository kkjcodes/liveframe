from __future__ import annotations

import pytest
import torch

from liveframe.refresh_proof import TOKENS_PER_FRAME
from liveframe.torch_refresh_rerope import (
    build_refresh_substitution,
    capture_raw_kv_source,
    overwrite_full_attention_cache,
    tensor_sha256,
)

BLOCK_TOKENS = TOKENS_PER_FRAME


def _tile(value: float = 1.0) -> torch.Tensor:
    return torch.full((1, BLOCK_TOKENS, 2, 4), value, dtype=torch.float32)


def _rope(
    raw_key: torch.Tensor,
    _grid_sizes: object,
    _freqs: object,
    *,
    start_frame: int,
) -> torch.Tensor:
    return raw_key + float(start_frame)


def test_source_capture_clones_raw_key_and_value() -> None:
    raw_key = _tile(1.0)
    value = _tile(2.0)

    source = capture_raw_kv_source(
        raw_key,
        value,
        layer_index=4,
        tile_index=2,
        source_start_frame=2,
    )
    raw_key.zero_()
    value.zero_()

    assert torch.all(source.raw_key == 1.0)
    assert torch.all(source.value == 2.0)
    assert source.raw_key_sha256 == tensor_sha256(source.raw_key)
    assert source.value_sha256 == tensor_sha256(source.value)


def test_substitution_reropes_raw_key_and_reuses_unrotated_value() -> None:
    source = capture_raw_kv_source(
        _tile(1.0),
        _tile(2.0),
        layer_index=4,
        tile_index=3,
        source_start_frame=3,
    )
    reference_key = _tile(9.0)
    reference_value = _tile(10.0)

    substitution = build_refresh_substitution(
        source,
        destination_start_frame=4,
        grid_sizes=[(1, 1, TOKENS_PER_FRAME)],
        freqs=object(),
        rope_apply=_rope,
        reference_key=reference_key,
        reference_value=reference_value,
    )

    assert torch.all(substitution.reroped_key == 5.0)
    assert torch.equal(substitution.value, source.value)
    assert substitution.value.data_ptr() != source.value.data_ptr()
    assert substitution.ancestry["source_start_frame"] == 3
    assert substitution.ancestry["destination_start_frame"] == 4
    assert substitution.ancestry["reference_destination_key_sha256"] == tensor_sha256(
        reference_key
    )


def test_cache_overwrite_targets_exact_full_attention_slice() -> None:
    source = capture_raw_kv_source(
        _tile(1.0),
        _tile(2.0),
        layer_index=0,
        tile_index=0,
        source_start_frame=0,
    )
    substitution = build_refresh_substitution(
        source,
        destination_start_frame=1,
        grid_sizes=[(1, 1, TOKENS_PER_FRAME)],
        freqs=object(),
        rope_apply=_rope,
        reference_key=_tile(0.0),
        reference_value=_tile(0.0),
    )
    capacity = 3 * TOKENS_PER_FRAME
    cache = {
        "k": torch.zeros((1, capacity, 2, 4), dtype=torch.float32),
        "v": torch.zeros((1, capacity, 2, 4), dtype=torch.float32),
        "global_end_index": torch.tensor([TOKENS_PER_FRAME]),
        "local_end_index": torch.tensor([TOKENS_PER_FRAME]),
    }

    visible_key, visible_value = overwrite_full_attention_cache(
        cache, substitution, destination_start_frame=1
    )

    assert visible_key.shape[1] == 2 * TOKENS_PER_FRAME
    assert visible_value.shape == visible_key.shape
    assert torch.all(cache["k"][:, :TOKENS_PER_FRAME] == 0.0)
    assert torch.all(cache["k"][:, TOKENS_PER_FRAME : 2 * TOKENS_PER_FRAME] == 2.0)
    assert torch.all(cache["v"][:, TOKENS_PER_FRAME : 2 * TOKENS_PER_FRAME] == 2.0)
    assert cache["global_end_index"].item() == 2 * TOKENS_PER_FRAME
    assert cache["local_end_index"].item() == 2 * TOKENS_PER_FRAME


def test_cache_overwrite_maps_global_frame_into_rolled_local_storage() -> None:
    source = capture_raw_kv_source(
        _tile(1.0),
        _tile(2.0),
        layer_index=0,
        tile_index=21,
        source_start_frame=21,
    )
    substitution = build_refresh_substitution(
        source,
        destination_start_frame=22,
        grid_sizes=[(1, 1, TOKENS_PER_FRAME)],
        freqs=object(),
        rope_apply=_rope,
        reference_key=_tile(0.0),
        reference_value=_tile(0.0),
    )
    capacity = 21 * TOKENS_PER_FRAME
    cache = {
        "k": torch.zeros((1, capacity, 2, 4), dtype=torch.float32),
        "v": torch.zeros((1, capacity, 2, 4), dtype=torch.float32),
        "global_end_index": torch.tensor([23 * TOKENS_PER_FRAME]),
        "local_end_index": torch.tensor([capacity]),
    }

    visible_key, visible_value = overwrite_full_attention_cache(
        cache, substitution, destination_start_frame=22
    )

    assert visible_key.shape[1] == capacity
    assert visible_value.shape == visible_key.shape
    assert torch.all(cache["k"][:, -TOKENS_PER_FRAME:] == 23.0)
    assert torch.all(cache["v"][:, -TOKENS_PER_FRAME:] == 2.0)
    assert cache["global_end_index"].item() == 23 * TOKENS_PER_FRAME
    assert cache["local_end_index"].item() == capacity


def test_substitution_rejects_position_shape_and_cursor_ambiguity() -> None:
    source = capture_raw_kv_source(
        _tile(), _tile(), layer_index=0, tile_index=0, source_start_frame=0
    )
    with pytest.raises(ValueError, match="strictly follow"):
        build_refresh_substitution(
            source,
            destination_start_frame=0,
            grid_sizes=[(1, 1, TOKENS_PER_FRAME)],
            freqs=object(),
            rope_apply=_rope,
            reference_key=_tile(),
            reference_value=_tile(),
        )
    with pytest.raises(ValueError, match="framewise block"):
        capture_raw_kv_source(
            torch.zeros((1, 1, 2, 4)),
            torch.zeros((1, 1, 2, 4)),
            layer_index=0,
            tile_index=0,
            source_start_frame=0,
        )

    substitution = build_refresh_substitution(
        source,
        destination_start_frame=2,
        grid_sizes=[(1, 1, TOKENS_PER_FRAME)],
        freqs=object(),
        rope_apply=_rope,
        reference_key=_tile(),
        reference_value=_tile(),
    )
    cache = {
        "k": torch.zeros((1, 3 * TOKENS_PER_FRAME, 2, 4)),
        "v": torch.zeros((1, 3 * TOKENS_PER_FRAME, 2, 4)),
        "global_end_index": torch.tensor([0]),
        "local_end_index": torch.tensor([0]),
    }
    with pytest.raises(ValueError, match="cursors"):
        overwrite_full_attention_cache(cache, substitution, destination_start_frame=2)