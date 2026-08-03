"""Run one causal layer against a converted CF++ checkpoint on Apple MLX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def run_smoke(
    model_dir: Path,
    *,
    layer_limit: int = 1,
    latent_height: int = 4,
    latent_width: int = 4,
    latent_blocks: int = 1,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_video.models.wan_2.config import WanModelConfig
    from mlx_video.models.wan_2.utils import load_wan_model

    from liveframe.backends.mlx_causal import CausalWanForward

    raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    allowed = WanModelConfig.__dataclass_fields__.keys()
    config = WanModelConfig(**{key: value for key, value in raw_config.items() if key in allowed})
    model = load_wan_model(model_dir / "model.safetensors", config)
    causal = CausalWanForward(model, layer_limit=layer_limit)

    if latent_height <= 0 or latent_width <= 0:
        raise ValueError("latent dimensions must be positive")
    if latent_blocks <= 0:
        raise ValueError("latent_blocks must be positive")
    if latent_height % config.patch_size[1] or latent_width % config.patch_size[2]:
        raise ValueError("latent dimensions must align to the model patch size")
    latent = mx.zeros(
        (config.in_dim, 1, latent_height, latent_width), dtype=mx.bfloat16
    )
    context = mx.zeros((1, config.text_len, config.dim), dtype=mx.bfloat16)
    cross_kv = model.prepare_cross_kv(context)[:layer_limit]
    mx.eval(cross_kv)
    tokens_per_frame = (
        latent_height // config.patch_size[1]
    ) * (latent_width // config.patch_size[2])
    output = latent
    captured: dict[str, Any] = {}
    for block in range(latent_blocks):
        current_start = block * tokens_per_frame
        output = causal.predict_x0(
            [latent],
            warped_timestep=1000.0,
            embedded_context=context,
            cross_kv_caches=cross_kv,
            current_start=current_start,
            capture=captured if block == 0 else None,
        )[0]
        # Upstream Step 3.3: replace speculative K/V with clean-context K/V.
        refreshed = causal(
            [output],
            timestep=mx.zeros((1, 1)),
            embedded_context=context,
            cross_kv_caches=cross_kv,
            current_start=current_start,
        )[0]
        mx.eval(output, refreshed)
    finite = bool(
        (mx.all(mx.isfinite(output)) & mx.all(mx.isfinite(refreshed))).item()
    )
    mx.eval(*captured.values())
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "finite": finite,
        "layers": layer_limit,
        "latent_blocks": latent_blocks,
        "model_forward_passes": latent_blocks * 2,
        "captured_intermediates": len(captured),
        "tokens_per_frame": tokens_per_frame,
        "cache_global_end": causal.caches[0].global_end,
        "cache_local_end": causal.caches[0].local_end,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--latent-height", type=int, default=4)
    parser.add_argument("--latent-width", type=int, default=4)
    parser.add_argument("--latent-blocks", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke(
                args.model_dir,
                layer_limit=args.layers,
                latent_height=args.latent_height,
                latent_width=args.latent_width,
                latent_blocks=args.latent_blocks,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()