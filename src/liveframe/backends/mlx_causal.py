"""MLX causal Wan primitives layered over the pinned ``mlx-video`` model.

This optional-runtime module intentionally owns the causal behavior instead of
patching site-packages. The loaded stock model remains the weight container.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from liveframe.causal_cache import plan_cache_update

try:
    from mlx_video.models.wan_2.attention import _linear_dtype
except ImportError as exc:  # pragma: no cover - guarded by the backend factory
    raise RuntimeError("pinned mlx-video Wan modules are unavailable") from exc


@dataclass(slots=True)
class MLXKVCache:
    """Populated K/V tensors and the upstream global/local cursors."""

    capacity: int = 32760
    max_attention_size: int = 32760
    local_attention: bool = False
    sink_tokens: int = 0
    k: mx.array | None = None
    v: mx.array | None = None
    global_end: int = 0
    local_end: int = 0

    def update(
        self, key: mx.array, value: mx.array, *, current_start: int
    ) -> tuple[mx.array, mx.array]:
        """Insert or overwrite one block and return the visible attention span."""

        token_count = key.shape[1]
        if value.shape[1] != token_count:
            raise ValueError("causal key/value token counts differ")
        plan = plan_cache_update(
            current_start=current_start,
            token_count=token_count,
            previous_global_end=self.global_end,
            previous_local_end=self.local_end,
            capacity=self.capacity,
            max_attention_size=self.max_attention_size,
            local_attention=self.local_attention,
            sink_tokens=self.sink_tokens,
        )

        if self.k is None or self.v is None:
            new_k, new_v = key, value
        elif plan.evicted_tokens:
            retained_k = mx.concatenate(
                [
                    self.k[:, : plan.sink_tokens],
                    self.k[
                        :,
                        plan.sink_tokens + plan.evicted_tokens : self.local_end,
                    ],
                ],
                axis=1,
            )
            retained_v = mx.concatenate(
                [
                    self.v[:, : plan.sink_tokens],
                    self.v[
                        :,
                        plan.sink_tokens + plan.evicted_tokens : self.local_end,
                    ],
                ],
                axis=1,
            )
            new_k = mx.concatenate([retained_k, key], axis=1)
            new_v = mx.concatenate([retained_v, value], axis=1)
        else:
            new_k = mx.concatenate(
                [self.k[:, : plan.local_start], key, self.k[:, plan.local_end :]],
                axis=1,
            )
            new_v = mx.concatenate(
                [self.v[:, : plan.local_start], value, self.v[:, plan.local_end :]],
                axis=1,
            )

        self.k = new_k[:, : plan.local_end]
        self.v = new_v[:, : plan.local_end]
        self.global_end = plan.global_end
        self.local_end = plan.local_end
        return (
            self.k[:, plan.attention_start : self.local_end],
            self.v[:, plan.attention_start : self.local_end],
        )


@dataclass(slots=True)
class RollingForcingKVCache:
    """Specialized three-frame anchor cache from the pinned long-video fork."""

    capacity: int = 24 * 1560
    max_attention_size: int = 21 * 1560
    block_tokens: int = 3 * 1560
    k: mx.array | None = None
    v: mx.array | None = None
    global_end: int = 0
    local_end: int = 0

    def update(
        self,
        raw_key: mx.array,
        roped_key: mx.array,
        value: mx.array,
        *,
        current_start: int,
        grid_sizes: list[tuple[int, int, int]],
        freqs: mx.array,
        updating_cache: bool,
    ) -> tuple[mx.array, mx.array]:
        if raw_key.shape[1] < self.block_tokens:
            raise ValueError("rolling query must contain a complete three-frame block")
        cache_end = current_start + self.block_tokens
        new_tokens = cache_end - self.global_end
        if new_tokens < 0:
            raise ValueError("rolling cache cannot update expired history")
        if self.k is None or self.v is None:
            self.k = raw_key[:, : self.block_tokens]
            self.v = value[:, : self.block_tokens]
            local_start = 0
            local_end = self.block_tokens
        else:
            evicted = max(0, new_tokens + self.local_end - self.capacity)
            if evicted:
                retained = self.local_end - evicted - self.block_tokens
                if retained < 0:
                    raise ValueError("rolling cache cannot preserve its anchor block")
                self.k = mx.concatenate(
                    [
                        self.k[:, : self.block_tokens],
                        self.k[:, self.block_tokens + evicted : self.local_end],
                    ],
                    axis=1,
                )
                self.v = mx.concatenate(
                    [
                        self.v[:, : self.block_tokens],
                        self.v[:, self.block_tokens + evicted : self.local_end],
                    ],
                    axis=1,
                )
                local_end = self.local_end + new_tokens - evicted
            else:
                local_end = self.local_end + new_tokens
            local_start = local_end - self.block_tokens
            stored_key = (
                raw_key[:, : self.block_tokens]
                if local_start == 0
                else roped_key[:, : self.block_tokens]
            )
            self.k = mx.concatenate(
                [self.k[:, :local_start], stored_key, self.k[:, local_end:]], axis=1
            )[:, :local_end]
            self.v = mx.concatenate(
                [
                    self.v[:, :local_start],
                    value[:, : self.block_tokens],
                    self.v[:, local_end:],
                ],
                axis=1,
            )[:, :local_end]
        if new_tokens:
            self.global_end = cache_end
            self.local_end = local_end

        if local_start == 0:
            return roped_key, value
        frame_tokens = grid_sizes[0][1] * grid_sizes[0][2]
        anchor_grid = [(3, grid_sizes[0][1], grid_sizes[0][2])]
        if updating_cache:
            extract_start = max(0, local_end - self.max_attention_size)
            visible_key = self.k[:, extract_start:local_end]
            visible_value = self.v[:, extract_start:local_end]
            if extract_start == 0:
                anchor = causal_rope_apply(
                    visible_key[:, : self.block_tokens],
                    anchor_grid,
                    freqs,
                    start_frame=0,
                ).astype(visible_key.dtype)
                visible_key = mx.concatenate(
                    [anchor, visible_key[:, self.block_tokens :]], axis=1
                )
            return visible_key, visible_value

        query_tokens = roped_key.shape[1]
        working_limit = self.max_attention_size - query_tokens - self.block_tokens
        if working_limit < 0:
            raise ValueError("rolling query exceeds the 21-frame attention window")
        extract_end = local_start
        extract_start = max(self.block_tokens, local_start - working_limit)
        working_key = self.k[:, extract_start:extract_end]
        working_value = self.v[:, extract_start:extract_end]
        working_frames = working_key.shape[1] // frame_tokens
        current_frame = current_start // frame_tokens
        anchor = causal_rope_apply(
            self.k[:, : self.block_tokens],
            anchor_grid,
            freqs,
            start_frame=current_frame - working_frames - 3,
        ).astype(roped_key.dtype)
        return (
            mx.concatenate([anchor, working_key, roped_key], axis=1),
            mx.concatenate(
                [self.v[:, : self.block_tokens], working_value, value], axis=1
            ),
        )


def causal_rope_apply(
    x: mx.array,
    grid_sizes: list[tuple[int, int, int]],
    freqs: mx.array,
    *,
    start_frame: int,
) -> mx.array:
    """Apply Wan's factorized RoPE with an absolute temporal frame offset."""

    batch, sequence, heads, head_dim = x.shape
    half_dim = head_dim // 2
    if freqs.dtype != x.dtype:
        freqs = freqs.astype(x.dtype)

    temporal_dim = half_dim - 2 * (half_dim // 3)
    height_dim = half_dim // 3
    width_dim = half_dim // 3
    temporal = freqs[:, :temporal_dim]
    height = freqs[:, temporal_dim : temporal_dim + height_dim]
    width = freqs[:, temporal_dim + height_dim :]

    outputs = []
    for index in range(batch):
        frames, rows, columns = grid_sizes[index]
        token_count = frames * rows * columns
        if start_frame + frames > freqs.shape[0]:
            raise ValueError("temporal RoPE offset exceeds the frequency table")
        pairs = x[index, :token_count].reshape(
            token_count, heads, half_dim, 2
        )
        ft = mx.broadcast_to(
            temporal[start_frame : start_frame + frames].reshape(
                frames, 1, 1, temporal_dim, 2
            ),
            (frames, rows, columns, temporal_dim, 2),
        )
        fh = mx.broadcast_to(
            height[:rows].reshape(1, rows, 1, height_dim, 2),
            (frames, rows, columns, height_dim, 2),
        )
        fw = mx.broadcast_to(
            width[:columns].reshape(1, 1, columns, width_dim, 2),
            (frames, rows, columns, width_dim, 2),
        )
        factors = mx.concatenate([ft, fh, fw], axis=3).reshape(
            token_count, 1, half_dim, 2
        )
        real = pairs[..., 0] * factors[..., 0] - pairs[..., 1] * factors[..., 1]
        imaginary = (
            pairs[..., 0] * factors[..., 1] + pairs[..., 1] * factors[..., 0]
        )
        rotated = mx.stack([real, imaginary], axis=-1).reshape(
            token_count, heads, head_dim
        )
        if token_count < sequence:
            rotated = mx.concatenate([rotated, x[index, token_count:]], axis=0)
        outputs.append(rotated)
    return mx.stack(outputs)


def causal_self_attention(
    attention,
    x: mx.array,
    grid_sizes: list[tuple[int, int, int]],
    freqs: mx.array,
    cache: MLXKVCache,
    *,
    current_start: int,
    capture: dict[str, mx.array] | None = None,
    capture_prefix: str = "self_attention",
) -> tuple[mx.array, MLXKVCache]:
    """Run stock Wan Q/K/V projections with CF++ causal cache semantics."""

    batch, sequence, _ = x.shape
    heads, head_dim = attention.num_heads, attention.head_dim
    weight_dtype = _linear_dtype(attention.q)
    weighted_x = x.astype(weight_dtype)
    query = attention.q(weighted_x)
    key = attention.k(weighted_x)
    if attention.norm_q is not None:
        query = attention.norm_q(query)
    if attention.norm_k is not None:
        key = attention.norm_k(key)
    query = query.reshape(batch, sequence, heads, head_dim)
    key = key.reshape(batch, sequence, heads, head_dim)
    value = attention.v(weighted_x).reshape(batch, sequence, heads, head_dim)

    frame_sequence = grid_sizes[0][1] * grid_sizes[0][2]
    if current_start % frame_sequence:
        raise ValueError("current_start must align to a latent-frame token boundary")
    start_frame = current_start // frame_sequence
    query = causal_rope_apply(
        query.astype(mx.float32), grid_sizes, freqs, start_frame=start_frame
    ).astype(weight_dtype)
    key = causal_rope_apply(
        key.astype(mx.float32), grid_sizes, freqs, start_frame=start_frame
    ).astype(weight_dtype)
    visible_key, visible_value = cache.update(key, value, current_start=current_start)
    if capture is not None:
        capture[f"{capture_prefix}.query"] = query
        capture[f"{capture_prefix}.key"] = key
        capture[f"{capture_prefix}.visible_key"] = visible_key
        capture[f"{capture_prefix}.visible_value"] = visible_value

    query = query.transpose(0, 2, 1, 3)
    visible_key = visible_key.transpose(0, 2, 1, 3)
    visible_value = visible_value.transpose(0, 2, 1, 3)
    output = mx.fast.scaled_dot_product_attention(
        query,
        visible_key,
        visible_value,
        scale=attention.scale,
    )
    output = output.transpose(0, 2, 1, 3).reshape(batch, sequence, -1)
    projected = attention.o(output)
    if capture is not None:
        capture[f"{capture_prefix}.output"] = projected
    return projected, cache


def rolling_forcing_self_attention(
    attention,
    x: mx.array,
    grid_sizes: list[tuple[int, int, int]],
    freqs: mx.array,
    cache: RollingForcingKVCache,
    *,
    current_start: int,
    updating_cache: bool,
    capture: dict[str, mx.array] | None = None,
    capture_prefix: str = "self_attention",
) -> tuple[mx.array, RollingForcingKVCache]:
    """Run attention with the released long-video anchor/re-RoPE policy."""

    batch, sequence, _ = x.shape
    heads, head_dim = attention.num_heads, attention.head_dim
    weight_dtype = _linear_dtype(attention.q)
    weighted_x = x.astype(weight_dtype)
    query = attention.q(weighted_x)
    key = attention.k(weighted_x)
    if attention.norm_q is not None:
        query = attention.norm_q(query)
    if attention.norm_k is not None:
        key = attention.norm_k(key)
    query = query.reshape(batch, sequence, heads, head_dim)
    key = key.reshape(batch, sequence, heads, head_dim)
    value = attention.v(weighted_x).reshape(batch, sequence, heads, head_dim)
    frame_sequence = grid_sizes[0][1] * grid_sizes[0][2]
    start_frame = current_start // frame_sequence
    roped_query = causal_rope_apply(
        query.astype(mx.float32), grid_sizes, freqs, start_frame=start_frame
    ).astype(weight_dtype)
    roped_key = causal_rope_apply(
        key.astype(mx.float32), grid_sizes, freqs, start_frame=start_frame
    ).astype(weight_dtype)
    visible_key, visible_value = cache.update(
        key,
        roped_key,
        value,
        current_start=current_start,
        grid_sizes=grid_sizes,
        freqs=freqs,
        updating_cache=updating_cache,
    )
    if capture is not None:
        capture[f"{capture_prefix}.query"] = roped_query
        capture[f"{capture_prefix}.key"] = roped_key
        capture[f"{capture_prefix}.visible_key"] = visible_key
        capture[f"{capture_prefix}.visible_value"] = visible_value
    output = mx.fast.scaled_dot_product_attention(
        roped_query.transpose(0, 2, 1, 3),
        visible_key.transpose(0, 2, 1, 3),
        visible_value.transpose(0, 2, 1, 3),
        scale=attention.scale,
    ).transpose(0, 2, 1, 3)
    projected = attention.o(output.reshape(batch, sequence, -1))
    return projected, cache


class CausalWanForward:
    """One-block-at-a-time causal forward using a loaded stock ``WanModel``."""

    def __init__(
        self,
        model,
        *,
        layer_limit: int | None = None,
        cache_capacity: int = 32760,
        max_attention_size: int = 32760,
        local_attention: bool = False,
        sink_tokens: int = 0,
        rolling_forcing: bool = False,
    ) -> None:
        self.model = model
        if layer_limit is not None and not 1 <= layer_limit <= len(model.blocks):
            raise ValueError("layer_limit must select at least one available block")
        self.layer_limit = layer_limit or len(model.blocks)
        self.cache_capacity = cache_capacity
        self.max_attention_size = max_attention_size
        self.local_attention = local_attention
        self.sink_tokens = sink_tokens
        self.rolling_forcing = rolling_forcing
        self.caches = self._new_caches()

    def _new_caches(self) -> list[MLXKVCache | RollingForcingKVCache]:
        if self.rolling_forcing:
            return [RollingForcingKVCache() for _ in range(self.layer_limit)]
        return [
            MLXKVCache(
                capacity=self.cache_capacity,
                max_attention_size=self.max_attention_size,
                local_attention=self.local_attention,
                sink_tokens=self.sink_tokens,
            )
            for _ in range(self.layer_limit)
        ]

    def reset(self) -> None:
        self.caches = self._new_caches()

    def __call__(
        self,
        x_list: list[mx.array],
        *,
        timestep: mx.array,
        embedded_context: mx.array,
        cross_kv_caches: list[tuple[mx.array, mx.array]],
        current_start: int,
        updating_cache: bool = False,
        capture: dict[str, mx.array] | None = None,
    ) -> list[mx.array]:
        patches = []
        grid_sizes = []
        for latent in x_list:
            patch, grid = self.model._patchify(latent)
            patches.append(patch)
            grid_sizes.append(grid)
        x = mx.concatenate(patches, axis=0)
        if capture is not None:
            capture["patch_embedding"] = x
        batch = len(x_list)

        if timestep.ndim == 1:
            timestep = timestep[:, None]
        sinusoid = timestep[..., None].astype(mx.float32) * self.model._inv_freq
        sin_embedding = mx.concatenate(
            [mx.cos(sinusoid), mx.sin(sinusoid)], axis=-1
        )
        embedding = self.model.time_embedding_1(
            self.model.time_embedding_act(
                self.model.time_embedding_0(sin_embedding)
            )
        )
        modulation = self.model.time_projection(
            self.model.time_projection_act(embedding)
        ).reshape(batch, -1, 6, self.model.dim)
        if capture is not None:
            capture["time_embedding"] = embedding
            capture["time_modulation"] = modulation

        frame_count = modulation.shape[1]
        if x.shape[1] % frame_count:
            raise ValueError("patch sequence does not divide into timestep frames")
        frame_tokens = x.shape[1] // frame_count

        def apply_frame_modulation(
            value: mx.array, scale: mx.array, shift: mx.array
        ) -> mx.array:
            framed = value.reshape(batch, frame_count, frame_tokens, -1)
            return (framed * (1 + scale[:, :, None]) + shift[:, :, None]).reshape(
                value.shape
            )

        for index, block in enumerate(self.model.blocks[: self.layer_limit]):
            mod = block.modulation + modulation
            e0, e1, e2, e3, e4, e5 = (
                mod[:, :, 0, :],
                mod[:, :, 1, :],
                mod[:, :, 2, :],
                mod[:, :, 3, :],
                mod[:, :, 4, :],
                mod[:, :, 5, :],
            )
            self_input = apply_frame_modulation(block.norm1(x), e1, e0)
            if self.rolling_forcing:
                rolling_cache = self.caches[index]
                if not isinstance(rolling_cache, RollingForcingKVCache):
                    raise TypeError("rolling forward requires RollingForcingKVCache")
                self_output, self.caches[index] = rolling_forcing_self_attention(
                    block.self_attn,
                    self_input,
                    grid_sizes,
                    self.model.freqs,
                    rolling_cache,
                    current_start=current_start,
                    updating_cache=updating_cache,
                    capture=capture,
                    capture_prefix=f"block.{index:02d}.self_attention",
                )
            else:
                standard_cache = self.caches[index]
                if not isinstance(standard_cache, MLXKVCache):
                    raise TypeError("causal forward requires MLXKVCache")
                self_output, self.caches[index] = causal_self_attention(
                    block.self_attn,
                    self_input,
                    grid_sizes,
                    self.model.freqs,
                    standard_cache,
                    current_start=current_start,
                    capture=capture,
                    capture_prefix=f"block.{index:02d}.self_attention",
                )
            if capture is not None:
                capture[f"block.{index:02d}.self_attention.output"] = self_output
            x = x + self_output * mx.repeat(e2, frame_tokens, axis=1)
            cross_input = block.norm3(x) if block.norm3 is not None else x
            cross_output = block.cross_attn(
                cross_input,
                embedded_context,
                None,
                kv_cache=cross_kv_caches[index],
            )
            if capture is not None:
                capture[f"block.{index:02d}.cross_attention.output"] = cross_output
            x = x + cross_output
            ffn_input = apply_frame_modulation(block.norm2(x), e4, e3)
            ffn_output = block.ffn(ffn_input)
            if capture is not None:
                capture[f"block.{index:02d}.ffn.output"] = ffn_output
            x = x + ffn_output * mx.repeat(e5, frame_tokens, axis=1)
            if capture is not None:
                capture[f"block.{index:02d}.output"] = x

        head_embedding = mx.repeat(embedding, frame_tokens, axis=1)
        output = self.model.head(x, head_embedding)
        if capture is not None:
            capture["head_output"] = output
        return [item.astype(mx.float32) for item in self.model.unpatchify(output, grid_sizes)]

    def predict_x0(
        self,
        noisy_latents: list[mx.array],
        *,
        warped_timestep: float,
        embedded_context: mx.array,
        cross_kv_caches: list[tuple[mx.array, mx.array]],
        current_start: int,
        capture: dict[str, mx.array] | None = None,
    ) -> list[mx.array]:
        """Run the flow model and convert its output to the CF++ x0 prediction."""

        timestep = mx.full((len(noisy_latents), 1), warped_timestep)
        flow = self(
            noisy_latents,
            timestep=timestep,
            embedded_context=embedded_context,
            cross_kv_caches=cross_kv_caches,
            current_start=current_start,
            capture=capture,
        )
        sigma = warped_timestep / 1000.0
        if capture is not None:
            capture["flow_prediction"] = mx.stack(flow)
        x0 = [
            latent - sigma * prediction
            for latent, prediction in zip(noisy_latents, flow, strict=True)
        ]
        if capture is not None:
            capture["x0_prediction"] = mx.stack(x0)
        return x0