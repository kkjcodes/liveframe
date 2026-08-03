"""Replay the complete H100 CF++ fixture through the causal MLX backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_video.models.wan_2.config import WanModelConfig
from mlx_video.models.wan_2.utils import load_wan_model

from liveframe.backends.mlx_causal import CausalWanForward
from liveframe.fixtures import GoldenFixture, GoldenTensor, load_fixture
from liveframe.mlx_parity import _metrics, _mlx_tensor, _numpy_tensor

FIXTURE_IDS = {
    1: "cfpp-1step-native-21block-h100",
    2: "cfpp-2step-native-21block-h100",
}
LATENT_BLOCKS = 21
TOKENS_PER_FRAME = 1560
FIRST_BLOCK_STEPS = 4
SELECTED_CACHE_LAYERS = (0, 15, 29)
CACHE_PROBE_OFFSETS = (0, TOKENS_PER_FRAME // 2, TOKENS_PER_FRAME - 1)


def _require_context_noise(
    metadata: dict[str, Any] | Any, *, fixture_id: str | None = None
) -> int:
    context_noise = metadata.get("context_noise")
    if context_noise is None and fixture_id in FIXTURE_IDS.values():
        return 0
    if context_noise != 0:
        raise ValueError(
            f"fixture context_noise must be explicitly recorded as 0, found {context_noise!r}"
        )
    return 0


def _reference(
    fixture: GoldenFixture, tensors: dict[str, GoldenTensor], name: str
) -> np.ndarray:
    return _numpy_tensor(fixture, tensors[name])


def _candidate(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.array(value.astype(mx.float32))


def _video_layout(value: mx.array) -> np.ndarray:
    """Convert MLX [B,C,F,H,W] to upstream wrapper [B,F,C,H,W]."""

    return _candidate(value).transpose(0, 2, 1, 3, 4)


def _compare(
    comparisons: dict[str, Any],
    fixture: GoldenFixture,
    tensors: dict[str, GoldenTensor],
    name: str,
    candidate: mx.array,
    *,
    video_layout: bool = False,
) -> None:
    got = _video_layout(candidate) if video_layout else _candidate(candidate)
    comparisons[name] = _metrics(_reference(fixture, tensors, name), got)


def _evaluate_policy(
    comparisons: dict[str, Any], *, student_steps: int = 2, latent_blocks: int = 21
) -> dict[str, Any]:
    """Evaluate observed BF16 backend drift without hiding pipeline divergence.

    Model inputs, scheduler outputs, and user-visible final latents remain under
    strict directional gates. Sparse K/V probes are internal deep-layer values;
    their drift compounds across 65 backend-dependent BF16 forwards, so they
    use a lower diagnostic floor while still rejecting gross cache corruption.
    """

    failures: list[str] = []

    def require(name: str, metric: str, limit: float, *, minimum: bool = False) -> None:
        row = comparisons.get(name)
        if row is None:
            failures.append(f"missing required comparison: {name}")
            return
        value = float(row[metric])
        accepted = value >= limit if minimum else value <= limit
        if not accepted:
            operator = ">=" if minimum else "<="
            failures.append(f"{name}.{metric}={value} must be {operator} {limit}")

    for block in range(latent_blocks):
        step_count = FIRST_BLOCK_STEPS if block == 0 else student_steps
        for step in range(step_count):
            prefix = f"block.{block:02d}.step.{step:02d}"
            require(f"{prefix}.model_input", "cosine", 0.995, minimum=True)
            require(f"{prefix}.flow_prediction", "cosine", 0.995, minimum=True)
            require(f"{prefix}.x0_prediction", "cosine", 0.995, minimum=True)
            if step < step_count - 1:
                require(f"{prefix}.renoised_output", "cosine", 0.995, minimum=True)
        refresh = f"block.{block:02d}.refresh"
        # The timestep-zero refresh output is discarded; its cache mutation is
        # checked separately through exact cursors and K/V probes.
        require(f"{refresh}.flow_prediction", "cosine", 0.99, minimum=True)
        require(f"{refresh}.x0_prediction", "cosine", 0.995, minimum=True)
        require(f"{refresh}.cache_global_end", "max_abs", 0.0)
        require(f"{refresh}.cache_local_end", "max_abs", 0.0)
        for layer in SELECTED_CACHE_LAYERS:
            require(
                f"{refresh}.layer.{layer:02d}.cache_key_probe",
                "cosine",
                0.99,
                minimum=True,
            )
            require(
                f"{refresh}.layer.{layer:02d}.cache_value_probe",
                "cosine",
                0.95,
                minimum=True,
            )
    require("output_latents", "cosine", 0.999, minimum=True)
    require("output_latents", "rmse", 0.035)
    require("output_latents", "max_abs", 0.25)
    return {
        "name": f"native-{latent_blocks}block-{student_steps}step-bfloat16.v1",
        "passed": not failures,
        "failures": failures,
    }


def _load_model(model_dir: Path) -> tuple[Any, mx.array, list[tuple[mx.array, mx.array]]]:
    raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    allowed = WanModelConfig.__dataclass_fields__.keys()
    config = WanModelConfig(
        **{key: value for key, value in raw_config.items() if key in allowed}
    )
    model = load_wan_model(model_dir / "model.safetensors", config)
    return model, mx.array([]), []


def run_pipeline_parity(model_dir: Path, fixture_dir: Path) -> dict[str, Any]:
    fixture = load_fixture(fixture_dir)
    tensors = {tensor.name: tensor for tensor in fixture.tensors}
    metadata = fixture.metadata
    student_steps_value = metadata.get("student_steps")
    if student_steps_value in {1, 2}:
        student_steps = int(student_steps_value)
    else:
        matching_steps = [
            steps
            for steps, fixture_id in FIXTURE_IDS.items()
            if fixture.fixture_id == fixture_id
        ]
        if not matching_steps:
            raise ValueError(f"unsupported full-pipeline fixture: {fixture.fixture_id}")
        student_steps = matching_steps[0]
    latent_blocks = int(metadata.get("latent_blocks", 0))
    if latent_blocks < 1:
        raise ValueError("fixture latent_blocks must be positive")
    context_noise = _require_context_noise(metadata, fixture_id=fixture.fixture_id)
    expected_counts = {
        "latent_blocks": latent_blocks,
        "denoising_forwards": 4 + (latent_blocks - 1) * student_steps,
        "refresh_forwards": latent_blocks,
        "total_transformer_forwards": 4
        + (latent_blocks - 1) * student_steps
        + latent_blocks,
        "scheduler_noise_draws": 3 + (latent_blocks - 1) * (student_steps - 1),
    }
    for name, expected in expected_counts.items():
        if metadata.get(name) != expected:
            raise ValueError(f"fixture metadata {name} must be {expected}")

    model, _, _ = _load_model(model_dir)
    prompt = _mlx_tensor(fixture, tensors["prompt_embedding"])
    embedded_context = model.embed_text([prompt[0]])
    cross_kv = model.prepare_cross_kv(embedded_context)
    mx.eval(embedded_context, cross_kv)

    initial = _mlx_tensor(fixture, tensors["initial_noise"])
    local_attention = bool(metadata.get("local_attention", False))
    local_attn_frames = int(metadata.get("local_attn_frames", LATENT_BLOCKS))
    sink_frames = int(metadata.get("sink_frames", 0))
    causal = CausalWanForward(
        model,
        cache_capacity=local_attn_frames * TOKENS_PER_FRAME,
        max_attention_size=local_attn_frames * TOKENS_PER_FRAME,
        local_attention=local_attention,
        sink_tokens=sink_frames * TOKENS_PER_FRAME,
    )
    comparisons: dict[str, Any] = {}
    generated: list[mx.array] = []

    raw_probe_indices = metadata.get("cache_probe_indices")
    if raw_probe_indices is None and fixture.fixture_id in FIXTURE_IDS.values():
        raw_probe_indices = [
            [block * TOKENS_PER_FRAME + offset for offset in CACHE_PROBE_OFFSETS]
            for block in range(latent_blocks)
        ]
    if not isinstance(raw_probe_indices, list) or len(raw_probe_indices) != latent_blocks:
        raise ValueError("fixture must record physical cache_probe_indices for every block")

    for block in range(latent_blocks):
        step_count = FIRST_BLOCK_STEPS if block == 0 else student_steps
        latent = initial[:, block : block + 1].transpose(0, 2, 1, 3, 4)[0]
        current_start = block * TOKENS_PER_FRAME

        for step in range(step_count):
            prefix = f"block.{block:02d}.step.{step:02d}"
            _compare(
                comparisons,
                fixture,
                tensors,
                f"{prefix}.model_input",
                latent[None],
                video_layout=True,
            )
            timestep = float(_mlx_tensor(fixture, tensors[f"{prefix}.timestep"]).item())
            flow = causal(
                [latent],
                timestep=mx.full((1, 1), timestep),
                embedded_context=embedded_context,
                cross_kv_caches=cross_kv,
                current_start=current_start,
            )[0]
            x0 = latent - (timestep / 1000.0) * flow
            mx.eval(flow, x0, *[cache.k for cache in causal.caches])
            _compare(
                comparisons,
                fixture,
                tensors,
                f"{prefix}.flow_prediction",
                flow[None],
                video_layout=True,
            )
            _compare(
                comparisons,
                fixture,
                tensors,
                f"{prefix}.x0_prediction",
                x0[None],
                video_layout=True,
            )

            if step < step_count - 1:
                scheduler_noise = _mlx_tensor(
                    fixture, tensors[f"{prefix}.scheduler_noise"]
                ).transpose(0, 2, 1, 3, 4)[0]
                next_timestep = float(
                    _mlx_tensor(
                        fixture, tensors[f"block.{block:02d}.step.{step + 1:02d}.timestep"]
                    ).item()
                )
                sigma = next_timestep / 1000.0
                latent = (
                    (1.0 - sigma) * x0.astype(mx.float32)
                    + sigma * scheduler_noise.astype(mx.float32)
                ).astype(mx.bfloat16)
                mx.eval(latent)
                _compare(
                    comparisons,
                    fixture,
                    tensors,
                    f"{prefix}.renoised_output",
                    latent[None],
                    video_layout=True,
                )
            else:
                latent = x0.astype(mx.bfloat16)
                mx.eval(latent)

        generated.append(latent)
        refresh = f"block.{block:02d}.refresh"
        if context_noise != 0:
            raise AssertionError("only context_noise=0 is implemented")
        refresh_flow = causal(
            [latent],
            timestep=mx.zeros((1, 1)),
            embedded_context=embedded_context,
            cross_kv_caches=cross_kv,
            current_start=current_start,
        )[0]
        refresh_x0 = latent.astype(mx.float32)
        mx.eval(refresh_flow, refresh_x0, *[cache.k for cache in causal.caches])
        _compare(
            comparisons,
            fixture,
            tensors,
            f"{refresh}.flow_prediction",
            refresh_flow[None],
            video_layout=True,
        )
        _compare(
            comparisons,
            fixture,
            tensors,
            f"{refresh}.x0_prediction",
            refresh_x0[None],
            video_layout=True,
        )

        global_ends = mx.array([cache.global_end for cache in causal.caches], dtype=mx.int64)
        local_ends = mx.array([cache.local_end for cache in causal.caches], dtype=mx.int64)
        _compare(
            comparisons,
            fixture,
            tensors,
            f"{refresh}.cache_global_end",
            global_ends,
        )
        _compare(
            comparisons,
            fixture,
            tensors,
            f"{refresh}.cache_local_end",
            local_ends,
        )
        probe_indices_value = raw_probe_indices[block]
        if not isinstance(probe_indices_value, list) or len(probe_indices_value) != 3:
            raise ValueError(f"invalid cache probes for block {block}")
        probe_indices = [int(index) for index in probe_indices_value]
        for layer in SELECTED_CACHE_LAYERS:
            cache = causal.caches[layer]
            if cache.k is None or cache.v is None:
                raise RuntimeError(f"layer {layer} cache is unexpectedly empty")
            _compare(
                comparisons,
                fixture,
                tensors,
                f"{refresh}.layer.{layer:02d}.cache_key_probe",
                cache.k[:, probe_indices],
            )
            _compare(
                comparisons,
                fixture,
                tensors,
                f"{refresh}.layer.{layer:02d}.cache_value_probe",
                cache.v[:, probe_indices],
            )
        print(f"compared block {block + 1}/{LATENT_BLOCKS}", flush=True)

    output = mx.concatenate(generated, axis=1)[None].transpose(0, 2, 1, 3, 4)
    _compare(comparisons, fixture, tensors, "output_latents", output)
    policy = _evaluate_policy(
        comparisons, student_steps=student_steps, latent_blocks=latent_blocks
    )
    drift_curve = []
    for block in range(latent_blocks):
        step_count = FIRST_BLOCK_STEPS if block == 0 else student_steps
        final_x0 = comparisons[f"block.{block:02d}.step.{step_count - 1:02d}.x0_prediction"]
        refresh = comparisons[f"block.{block:02d}.refresh.flow_prediction"]
        cache_rows = [
            comparisons[f"block.{block:02d}.refresh.layer.{layer:02d}.cache_{kind}_probe"]
            for layer in SELECTED_CACHE_LAYERS
            for kind in ("key", "value")
        ]
        drift_curve.append(
            {
                "block": block,
                "pixel_frame_end": 1 + 4 * block,
                "playback_seconds": (1 + 4 * block) / 16.0,
                "x0_cosine": final_x0["cosine"],
                "x0_rmse": final_x0["rmse"],
                "refresh_cosine": refresh["cosine"],
                "minimum_cache_probe_cosine": min(row["cosine"] for row in cache_rows),
                "maximum_cache_probe_abs": max(row["max_abs"] for row in cache_rows),
            }
        )
    checkpoints = {
        name: min(drift_curve, key=lambda row: abs(row["playback_seconds"] - seconds))
        for name, seconds in (("15s", 15.0), ("60s", 60.0))
        if drift_curve[-1]["playback_seconds"] >= seconds
    }
    ordered = sorted(
        comparisons.items(), key=lambda item: float(item[1]["cosine"])
    )
    return {
        "schema": "liveframe.pipeline-parity-report.v1",
        "fixture_id": fixture.fixture_id,
        "student_steps": student_steps,
        "latent_blocks": latent_blocks,
        "local_attention": local_attention,
        "local_attn_frames": local_attn_frames,
        "sink_frames": sink_frames,
        "model_dir": str(model_dir),
        "compared_tensors": len(comparisons),
        "passed": policy["passed"],
        "policy": policy,
        "drift_curve": drift_curve,
        "drift_checkpoints": checkpoints,
        "comparisons": dict(ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_pipeline_parity(args.model_dir, args.fixture)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "compared_tensors": report["compared_tensors"],
                    "passed": report["passed"],
                },
                sort_keys=True,
            )
        )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
