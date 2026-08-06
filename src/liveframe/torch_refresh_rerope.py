"""Torch primitives for framewise refresh K/V substitution with temporal re-RoPE."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any

from liveframe.causal_cache import CachePlanError, plan_cache_update
from liveframe.refresh_proof import (
    BLOCK_FRAMES,
    MAX_CONTEXT_FRAMES,
    MECHANISM_ID,
    MODEL_LAYERS,
    SUBSTITUTION_SCHEMA,
    TOKENS_PER_FRAME,
    is_full_refresh_frame,
    source_frame_for_destination,
    validate_substitution_ancestry,
)

RopeApply = Callable[[Any, Any, Any], Any]


def _torch() -> Any:
    import torch

    return torch


def tensor_sha256(value: Any) -> str:
    """Hash literal contiguous tensor bytes without numeric conversion."""

    torch = _torch()
    tensor = value.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        payload = tensor.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    elif tensor.dtype == torch.float16:
        payload = tensor.numpy().astype("<f2", copy=False).tobytes()
    elif tensor.dtype == torch.float32:
        payload = tensor.numpy().astype("<f4", copy=False).tobytes()
    else:
        raise TypeError(f"unsupported refresh proof tensor dtype: {tensor.dtype}")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class RawKVSource:
    """Immutable raw pre-RoPE key and value captured at one source tile."""

    layer_index: int
    tile_index: int
    source_start_frame: int
    raw_key: Any
    value: Any
    raw_key_sha256: str
    value_sha256: str


@dataclass(frozen=True, slots=True)
class RefreshSubstitution:
    """Candidate K/V plus its validated source/destination ancestry."""

    reroped_key: Any
    value: Any
    ancestry: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReferenceKV:
    layer_index: int
    destination_start_frame: int
    key: Any
    value: Any


class FramewiseRefreshRegistry:
    """Bind source and reference tensors before producing candidate substitutions."""

    def __init__(self) -> None:
        self.sources: dict[tuple[int, int], RawKVSource] = {}
        self.references: dict[tuple[int, int], ReferenceKV] = {}
        self.substitutions: list[RefreshSubstitution] = []
        self.substitution_ancestry: list[dict[str, Any]] = []

    def add_source(self, source: RawKVSource) -> None:
        key = (source.layer_index, source.tile_index)
        if key in self.sources:
            raise ValueError("refresh source capture is duplicated")
        self.sources[key] = source

    def add_reference(
        self,
        *,
        layer_index: int,
        destination_start_frame: int,
        key: Any,
        value: Any,
    ) -> None:
        identity = (layer_index, destination_start_frame)
        if identity in self.references:
            raise ValueError("refresh reference capture is duplicated")
        self.references[identity] = ReferenceKV(
            layer_index,
            destination_start_frame,
            key.detach().clone(),
            value.detach().clone(),
        )

    def substitute(
        self,
        *,
        layer_index: int,
        source_tile_index: int,
        destination_start_frame: int,
        grid_sizes: Sequence[Sequence[int]],
        freqs: Any,
        rope_apply: Callable[..., Any],
    ) -> RefreshSubstitution:
        source = self.sources.get((layer_index, source_tile_index))
        if source is None:
            raise ValueError("refresh source capture is missing for candidate layer")
        reference = self.references.get((layer_index, destination_start_frame))
        if reference is None:
            raise ValueError("refresh reference capture is missing for candidate layer")
        substitution = build_refresh_substitution(
            source,
            destination_start_frame=destination_start_frame,
            grid_sizes=grid_sizes,
            freqs=freqs,
            rope_apply=rope_apply,
            reference_key=reference.key,
            reference_value=reference.value,
        )
        self.substitutions.append(substitution)
        self.substitution_ancestry.append(dict(substitution.ancestry))
        return substitution

    @property
    def substitution_count(self) -> int:
        return len(self.substitution_ancestry)

    def release_destination(self, destination_start_frame: int) -> None:
        """Release observed destination tensors and an anchor after its final reuse."""

        self.references = {
            key: value
            for key, value in self.references.items()
            if key[1] != destination_start_frame
        }
        self.substitutions = [
            value
            for value in self.substitutions
            if value.ancestry["destination_start_frame"] != destination_start_frame
        ]
        if destination_start_frame % 3 == 2:
            source_frame = source_frame_for_destination(destination_start_frame)
            self.sources = {
                key: value for key, value in self.sources.items() if key[1] != source_frame
            }

    def require_layers(
        self,
        *,
        source_tile_index: int,
        destination_start_frame: int,
        layer_count: int = MODEL_LAYERS,
    ) -> None:
        expected = set(range(layer_count))
        source_layers = {
            layer for layer, tile in self.sources if tile == source_tile_index
        }
        reference_layers = {
            layer
            for layer, destination in self.references
            if destination == destination_start_frame
        }
        if source_layers != expected:
            raise ValueError("refresh source capture does not cover every layer")
        if reference_layers != expected:
            raise ValueError("refresh reference capture does not cover every layer")


class FramewiseRefreshHarness:
    """Patch pinned CF++ self-attention modules for paired refresh capture/substitution."""

    def __init__(
        self,
        attentions: Sequence[Any],
        *,
        rope_apply: Callable[..., Any],
        attention: Callable[[Any, Any, Any], Any],
        registry: FramewiseRefreshRegistry | None = None,
    ) -> None:
        if len(attentions) != MODEL_LAYERS:
            raise ValueError(f"framewise refresh harness requires exactly {MODEL_LAYERS} layers")
        self.attentions = list(attentions)
        self.rope_apply = rope_apply
        self.attention = attention
        self.registry = registry or FramewiseRefreshRegistry()
        self._operation: tuple[str, int, int] | None = None
        self._originals: list[Any] = []

    @contextmanager
    def operation(
        self,
        mode: str,
        *,
        tile_index: int,
        start_frame: int,
    ) -> Any:
        if mode not in {"source", "reference", "candidate"}:
            raise ValueError("unsupported framewise refresh operation")
        if mode in {"source", "reference"} and tile_index != start_frame:
            raise ValueError("refresh capture tile must match its frame")
        if mode == "source" and not is_full_refresh_frame(start_frame):
            raise ValueError("refresh source capture must occur at a scheduled anchor")
        if mode == "reference" and is_full_refresh_frame(start_frame):
            raise ValueError("refresh reference capture is unnecessary at an anchor")
        if mode == "candidate":
            try:
                expected_source = source_frame_for_destination(start_frame)
            except ValueError as exc:
                raise ValueError("refresh candidate cannot replace a scheduled anchor") from exc
            if tile_index != expected_source:
                raise ValueError("refresh candidate must use the latest periodic anchor")
        if self._operation is not None:
            raise RuntimeError("framewise refresh operations cannot be nested")
        self._operation = (mode, tile_index, start_frame)
        try:
            yield self
        finally:
            self._operation = None

    def __enter__(self) -> FramewiseRefreshHarness:
        if self._originals:
            raise RuntimeError("framewise refresh harness is already installed")
        for layer_index, module in enumerate(self.attentions):
            original = module.forward
            self._originals.append(original)

            def patched(
                bound_module: Any,
                x: Any,
                seq_lens: Any,
                grid_sizes: Any,
                freqs: Any,
                block_mask: Any,
                kv_cache: dict[str, Any] | None = None,
                current_start: int = 0,
                cache_start: int | None = None,
                *,
                _layer: int = layer_index,
                _original: Any = original,
            ) -> Any:
                return self._forward(
                    _layer,
                    bound_module,
                    _original,
                    x,
                    seq_lens,
                    grid_sizes,
                    freqs,
                    block_mask,
                    kv_cache,
                    current_start,
                    cache_start,
                )

            module.forward = MethodType(patched, module)
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        for module, original in zip(self.attentions, self._originals, strict=True):
            module.forward = original
        self._originals.clear()
        self._operation = None

    def _forward(
        self,
        layer_index: int,
        module: Any,
        original: Any,
        x: Any,
        seq_lens: Any,
        grid_sizes: Any,
        freqs: Any,
        block_mask: Any,
        kv_cache: dict[str, Any] | None,
        current_start: int,
        cache_start: int | None,
    ) -> Any:
        if self._operation is None:
            return original(
                x, seq_lens, grid_sizes, freqs, block_mask, kv_cache, current_start, cache_start
            )
        mode, tile_index, start_frame = self._operation
        frame_tokens = int(grid_sizes[0][1] * grid_sizes[0][2])
        observed_start_frame = current_start // frame_tokens
        if frame_tokens != TOKENS_PER_FRAME or observed_start_frame != start_frame:
            raise ValueError("framewise refresh operation position mismatch")
        if kv_cache is None:
            raise ValueError("framewise refresh operation requires a KV cache")
        if cache_start not in {None, current_start}:
            raise ValueError("framewise refresh cache_start mismatch")
        if mode == "candidate":
            batch, sequence = x.shape[:2]
            query = module.norm_q(module.q(x)).view(
                batch, sequence, module.num_heads, module.head_dim
            )
            roped_query = self.rope_apply(
                query, grid_sizes, freqs, start_frame=start_frame
            ).type_as(query)
            substitution = self.registry.substitute(
                layer_index=layer_index,
                source_tile_index=tile_index,
                destination_start_frame=start_frame,
                grid_sizes=grid_sizes,
                freqs=freqs,
                rope_apply=self.rope_apply,
            )
            visible_key, visible_value = overwrite_full_attention_cache(
                kv_cache,
                substitution,
                destination_start_frame=start_frame,
            )
            output = self.attention(roped_query, visible_key, visible_value)
            return module.o(output.flatten(2))

        captured: dict[str, Any] = {}

        def key_hook(_module: Any, _inputs: Any, output: Any) -> None:
            if "key" in captured:
                raise ValueError("refresh raw key capture is duplicated within a layer")
            captured["key"] = output

        def value_hook(_module: Any, _inputs: Any, output: Any) -> None:
            if "value" in captured:
                raise ValueError("refresh value capture is duplicated within a layer")
            captured["value"] = output

        handles = [
            module.norm_k.register_forward_hook(key_hook),
            module.v.register_forward_hook(value_hook),
        ]
        try:
            output = original(
                x, seq_lens, grid_sizes, freqs, block_mask, kv_cache, current_start, cache_start
            )
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != {"key", "value"}:
            raise ValueError("refresh layer did not emit exactly one raw key and value")
        raw_key = captured["key"].view(
            x.shape[0], x.shape[1], module.num_heads, module.head_dim
        )
        value = captured["value"].view_as(raw_key)
        if mode == "source":
            self.registry.add_source(
                capture_raw_kv_source(
                    raw_key,
                    value,
                    layer_index=layer_index,
                    tile_index=tile_index,
                    source_start_frame=start_frame,
                )
            )
        else:
            reference_key = self.rope_apply(
                raw_key, grid_sizes, freqs, start_frame=start_frame
            ).type_as(value)
            self.registry.add_reference(
                layer_index=layer_index,
                destination_start_frame=start_frame,
                key=reference_key,
                value=value,
            )
        return output


def build_pinned_framewise_refresh_harness(
    generator: Any,
    causal_model: Any,
    *,
    registry: FramewiseRefreshRegistry | None = None,
) -> FramewiseRefreshHarness:
    """Bind the tested harness only to the frozen long-horizon CF++ configuration."""

    model = getattr(generator, "model", None)
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) != MODEL_LAYERS:
        raise ValueError(f"refresh generator must expose exactly {MODEL_LAYERS} blocks")
    if int(getattr(model, "local_attn_size", -1)) != MAX_CONTEXT_FRAMES:
        raise ValueError("refresh generator must use the pinned 21-frame rolling context")
    attentions = [getattr(block, "self_attn", None) for block in blocks]
    for layer_index, module in enumerate(attentions):
        if module is None:
            raise ValueError(f"refresh generator layer {layer_index} has no self-attention")
        if int(getattr(module, "local_attn_size", -1)) != MAX_CONTEXT_FRAMES:
            raise ValueError(f"refresh generator layer {layer_index} context mismatch")
        if int(getattr(module, "sink_size", -1)) != 0:
            raise ValueError(f"refresh generator layer {layer_index} sink mismatch")
        if int(getattr(module, "max_attention_size", -1)) != (
            MAX_CONTEXT_FRAMES * TOKENS_PER_FRAME
        ):
            raise ValueError(f"refresh generator layer {layer_index} token capacity mismatch")
    rope_apply = getattr(causal_model, "causal_rope_apply", None)
    attention = getattr(causal_model, "attention", None)
    if not callable(rope_apply) or not callable(attention):
        raise ValueError("refresh causal module call surface mismatch")
    return FramewiseRefreshHarness(
        attentions,
        rope_apply=rope_apply,
        attention=attention,
        registry=registry,
    )


def _require_tile_tensor(value: Any, name: str) -> None:
    torch = _torch()
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError(f"refresh {name} must be a B,L,H,D tensor")
    if value.shape[0] != 1 or value.shape[1] != BLOCK_FRAMES * TOKENS_PER_FRAME:
        raise ValueError(f"refresh {name} must contain exactly one framewise block")
    if not value.is_floating_point():
        raise ValueError(f"refresh {name} must be floating point")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"refresh {name} must be finite")


def capture_raw_kv_source(
    raw_key: Any,
    value: Any,
    *,
    layer_index: int,
    tile_index: int,
    source_start_frame: int,
) -> RawKVSource:
    """Clone one source tile before RoPE or cache mutation."""

    for name, observed in (
        ("layer_index", layer_index),
        ("tile_index", tile_index),
        ("source_start_frame", source_start_frame),
    ):
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise ValueError(f"refresh source {name} must be a non-negative integer")
    _require_tile_tensor(raw_key, "raw key")
    _require_tile_tensor(value, "value")
    if raw_key.shape != value.shape or raw_key.dtype != value.dtype:
        raise ValueError("refresh source raw key and value must share shape and dtype")
    key_clone = raw_key.detach().clone()
    value_clone = value.detach().clone()
    return RawKVSource(
        layer_index=layer_index,
        tile_index=tile_index,
        source_start_frame=source_start_frame,
        raw_key=key_clone,
        value=value_clone,
        raw_key_sha256=tensor_sha256(key_clone),
        value_sha256=tensor_sha256(value_clone),
    )


def build_refresh_substitution(
    source: RawKVSource,
    *,
    destination_start_frame: int,
    grid_sizes: Sequence[Sequence[int]],
    freqs: Any,
    rope_apply: Callable[..., Any],
    reference_key: Any,
    reference_value: Any,
) -> RefreshSubstitution:
    """Re-RoPE a raw source key and bind it to full-refresh destination K/V."""

    if (
        isinstance(destination_start_frame, bool)
        or not isinstance(destination_start_frame, int)
        or destination_start_frame <= source.source_start_frame
    ):
        raise ValueError("refresh destination must strictly follow the source")
    if len(grid_sizes) != 1 or tuple(int(value) for value in grid_sizes[0])[:1] != (
        BLOCK_FRAMES,
    ):
        raise ValueError("refresh substitution grid must describe one framewise block")
    _require_tile_tensor(reference_key, "reference key")
    _require_tile_tensor(reference_value, "reference value")
    if reference_key.shape != source.raw_key.shape or reference_value.shape != source.value.shape:
        raise ValueError("refresh reference K/V shape does not match the source tile")
    if reference_key.dtype != source.raw_key.dtype or reference_value.dtype != source.value.dtype:
        raise ValueError("refresh reference K/V dtype does not match the source tile")

    candidate_key = rope_apply(
        source.raw_key,
        grid_sizes,
        freqs,
        start_frame=destination_start_frame,
    ).type_as(source.raw_key)
    _require_tile_tensor(candidate_key, "candidate re-RoPE key")
    candidate_value = source.value.detach().clone()
    ancestry = {
        "schema": SUBSTITUTION_SCHEMA,
        "mechanism_id": MECHANISM_ID,
        "layer_index": source.layer_index,
        "tile_index": source.tile_index,
        "source_start_frame": source.source_start_frame,
        "destination_start_frame": destination_start_frame,
        "frame_count": BLOCK_FRAMES,
        "token_count": BLOCK_FRAMES * TOKENS_PER_FRAME,
        "source_key_form": "raw-pre-rope",
        "candidate_key_form": "reroped-from-source-raw-key",
        "value_policy": "reuse-source-value",
        "source_raw_key_sha256": source.raw_key_sha256,
        "source_value_sha256": source.value_sha256,
        "candidate_reroped_key_sha256": tensor_sha256(candidate_key),
        "reference_destination_key_sha256": tensor_sha256(reference_key),
        "reference_destination_value_sha256": tensor_sha256(reference_value),
    }
    validate_substitution_ancestry(ancestry)
    return RefreshSubstitution(candidate_key, candidate_value, ancestry)


def overwrite_full_attention_cache(
    kv_cache: dict[str, Any],
    substitution: RefreshSubstitution,
    *,
    destination_start_frame: int,
) -> tuple[Any, Any]:
    """Overwrite the exact full-attention destination slice and advance cursors."""

    torch = _torch()
    required = {"k", "v", "global_end_index", "local_end_index"}
    if set(kv_cache) != required:
        raise ValueError("refresh cache field set mismatch")
    key_cache = kv_cache["k"]
    value_cache = kv_cache["v"]
    if not isinstance(key_cache, torch.Tensor) or not isinstance(value_cache, torch.Tensor):
        raise ValueError("refresh cache K/V must be tensors")
    if key_cache.shape != value_cache.shape or key_cache.ndim != 4:
        raise ValueError("refresh cache K/V geometry mismatch")
    if (
        key_cache.dtype != substitution.reroped_key.dtype
        or value_cache.dtype != substitution.value.dtype
    ):
        raise ValueError("refresh cache dtype mismatch")
    current_start = destination_start_frame * TOKENS_PER_FRAME
    previous_global_end = int(kv_cache["global_end_index"].item())
    previous_local_end = int(kv_cache["local_end_index"].item())
    try:
        plan = plan_cache_update(
            current_start=current_start,
            token_count=BLOCK_FRAMES * TOKENS_PER_FRAME,
            previous_global_end=previous_global_end,
            previous_local_end=previous_local_end,
            capacity=key_cache.shape[1],
            max_attention_size=key_cache.shape[1],
            local_attention=True,
        )
    except CachePlanError as exc:
        raise ValueError("refresh cache cursors do not address the destination block") from exc
    if not plan.is_overwrite:
        raise ValueError("refresh cache cursors do not address a denoised destination block")

    key_cache[:, plan.local_start : plan.local_end].copy_(substitution.reroped_key)
    value_cache[:, plan.local_start : plan.local_end].copy_(substitution.value)
    kv_cache["global_end_index"].fill_(plan.global_end)
    kv_cache["local_end_index"].fill_(plan.local_end)
    return (
        key_cache[:, plan.attention_start : plan.local_end],
        value_cache[:, plan.attention_start : plan.local_end],
    )