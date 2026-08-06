"""Stage-level timing helpers for the pinned causal MLX Wan VAE."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any


def _profile_stage(
    name: str, operation: Callable[[], Any]
) -> tuple[Any, dict[str, Any]]:
    import mlx.core as mx  # type: ignore[import-not-found]

    build_started = time.perf_counter()
    value = operation()
    build_seconds = time.perf_counter() - build_started
    eval_started = time.perf_counter()
    mx.eval(value)
    eval_seconds = time.perf_counter() - eval_started
    return value, {
        "stage": name,
        "graph_build_seconds": build_seconds,
        "eval_seconds": eval_seconds,
        "shape": list(value.shape),
        "active_mlx_memory_gib": mx.get_active_memory() / 2**30,
        "cache_mlx_memory_gib": mx.get_cache_memory() / 2**30,
        "peak_mlx_memory_gib": mx.get_peak_memory() / 2**30,
    }


def profiled_causal_decode_frame(
    vae: Any, z: Any, cache: list[Any]
) -> tuple[Any, list[dict[str, Any]]]:
    """Decode one latent frame with a synchronization boundary after each decoder stage."""

    import mlx.core as mx
    import mlx.nn as nn

    from liveframe.mlx_vae_decode import _cache_conv, _cached_upsample

    rows: list[dict[str, Any]] = []

    def stage(name: str, operation: Callable[[], Any]) -> Any:
        value, row = _profile_stage(name, operation)
        rows.append(row)
        return value

    mean = vae.mean.reshape(1, -1, 1, 1, 1)
    inv_std = vae.inv_std.reshape(1, -1, 1, 1, 1)
    x = stage("post_quant.conv2", lambda: vae.conv2(z / inv_std + mean))
    decoder = vae.decoder
    index = [0]
    x = stage("decoder.conv1", lambda: _cache_conv(decoder.conv1, x, cache, index))
    for layer_index, layer in enumerate(decoder.middle):
        name = f"middle.{layer_index:02d}.{type(layer).__name__}"
        if hasattr(layer, "residual"):
            def middle_residual(
                current_layer: Any = layer, current_value: Any = x
            ) -> Any:
                return current_layer(current_value, feat_cache=cache, feat_idx=index)

            x = stage(name, middle_residual)
        else:
            def middle_layer(
                current_layer: Any = layer, current_value: Any = x
            ) -> Any:
                return current_layer(current_value)

            x = stage(name, middle_layer)
    for layer_index, layer in enumerate(decoder.upsamples):
        name = f"upsamples.{layer_index:02d}.{type(layer).__name__}"
        if hasattr(layer, "residual"):
            def upsample_residual(
                current_layer: Any = layer, current_value: Any = x
            ) -> Any:
                return current_layer(current_value, feat_cache=cache, feat_idx=index)

            x = stage(name, upsample_residual)
        elif getattr(layer, "mode", "").startswith("upsample"):
            def cached_upsample(
                current_layer: Any = layer, current_value: Any = x
            ) -> Any:
                return _cached_upsample(current_layer, current_value, cache, index)

            x = stage(name, cached_upsample)
        else:
            def upsample_layer(
                current_layer: Any = layer, current_value: Any = x
            ) -> Any:
                return current_layer(current_value)

            x = stage(name, upsample_layer)
    silu = nn.silu  # type: ignore[attr-defined]
    x = stage("head.norm_silu", lambda: silu(decoder.head[0](x)))
    x = stage("head.conv", lambda: _cache_conv(decoder.head[2], x, cache, index))
    x = stage("output.clip", lambda: mx.clip(x, -1, 1))
    if index[0] != len(cache):
        raise RuntimeError(f"VAE cache contract used {index[0]} of {len(cache)} slots")
    return x, rows


def summarize_stage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("stage profiling requires at least one row")
    names = list(dict.fromkeys(str(row["stage"]) for row in rows))
    result = []
    total_eval = sum(float(row["eval_seconds"]) for row in rows)
    for name in names:
        selected = [row for row in rows if row["stage"] == name]
        eval_values = [float(row["eval_seconds"]) for row in selected]
        build_values = [float(row["graph_build_seconds"]) for row in selected]
        result.append(
            {
                "stage": name,
                "samples": len(selected),
                "graph_build_seconds_median": statistics.median(build_values),
                "eval_seconds_median": statistics.median(eval_values),
                "eval_seconds_total": sum(eval_values),
                "eval_share_percent": 100.0 * sum(eval_values) / total_eval,
                "peak_mlx_memory_gib": max(
                    float(row["peak_mlx_memory_gib"]) for row in selected
                ),
            }
        )
    return sorted(result, key=lambda row: float(row["eval_seconds_total"]), reverse=True)