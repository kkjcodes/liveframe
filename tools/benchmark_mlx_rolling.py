"""Benchmark a frozen 81-frame Rolling Forcing clip on Apple MLX."""

# mypy: ignore-errors

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_video.models.wan_2.utils import load_vae_decoder

from liveframe.backends.mlx_causal import CausalWanForward
from liveframe.fixtures import load_fixture
from liveframe.mlx_benchmark import MLXBenchmarkConfig
from liveframe.mlx_parity import _mlx_tensor
from liveframe.mlx_rolling_parity import _load_model, _runtime_provenance, _sha256
from liveframe.mlx_vae_decode import _causal_decode_frame, _decoder_cache_slots

TOKENS_PER_FRAME = 1560
BLOCK_FRAMES = 3


def _thermal_state() -> list[str]:
    result = subprocess.run(
        ["pmset", "-g", "therm"],
        capture_output=True,
        check=False,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _timed_eval(value: mx.array) -> float:
    started = time.perf_counter()
    mx.eval(value)
    return time.perf_counter() - started


def _encode_h264(pixels: np.ndarray, output: Path, fps: int) -> float:
    height, width = pixels.shape[1:3]
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "h264_videotoolbox",
        "-b:v",
        "8M",
        str(output),
    ]
    started = time.perf_counter()
    subprocess.run(command, input=pixels.tobytes(order="C"), check=True)
    return time.perf_counter() - started


def _generate(
    model: Any,
    cross_kv: Any,
    embedded_context: mx.array,
    fixture: Any,
    tensors: dict[str, Any],
    config: MLXBenchmarkConfig,
) -> tuple[mx.array, dict[str, float]]:
    schedule = config.schedule
    causal = CausalWanForward(model, rolling_forcing=True)
    initial = _mlx_tensor(fixture, tensors["initial_noise"])[
        :, : config.latent_frames
    ].transpose(0, 2, 1, 3, 4)[0]
    initial_blocks = [
        initial[:, index : index + BLOCK_FRAMES]
        for index in range(0, config.latent_frames, BLOCK_FRAMES)
    ]
    noisy_blocks = [mx.zeros_like(row) for row in initial_blocks]
    generated: list[mx.array | None] = [None] * schedule.block_count
    warped_steps = dict(
        zip(schedule.configured_steps, fixture.metadata["warped_steps"], strict=True)
    )
    denoise_seconds = 0.0
    refresh_seconds = 0.0
    scheduler_seconds = 0.0

    for window in schedule.windows():
        block_ids = list(range(window.start_block, window.end_block + 1))
        if window.block_count == len(schedule.configured_steps) or window.start_block == 0:
            blocks = [noisy_blocks[index] for index in block_ids[:-1]] + [
                initial_blocks[block_ids[-1]]
            ]
        else:
            blocks = [noisy_blocks[index] for index in block_ids]
        model_input = mx.concatenate(blocks, axis=1).astype(mx.bfloat16)
        timestep = mx.array(
            [[warped_steps[step] for step in window.configured_timesteps for _ in range(3)]],
            dtype=mx.float32,
        )
        flow = causal(
            [model_input],
            timestep=timestep,
            embedded_context=embedded_context,
            cross_kv_caches=cross_kv,
            current_start=window.start_frame * TOKENS_PER_FRAME,
        )[0]
        sigma = (timestep[0] / 1000.0).reshape(1, -1, 1, 1)
        prediction = model_input.astype(mx.float32) - sigma * flow
        denoise_seconds += _timed_eval(prediction)

        scheduler_started = time.perf_counter()
        for offset, configured_step in enumerate(window.configured_timesteps):
            block_id = window.start_block + offset
            block_prediction = prediction[:, offset * 3 : (offset + 1) * 3]
            if configured_step == schedule.configured_steps[-1]:
                generated[block_id] = block_prediction.astype(mx.bfloat16)
                continue
            next_step = schedule.configured_steps[
                schedule.configured_steps.index(configured_step) + 1
            ]
            name = f"window.{window.index:03d}.block.{offset:02d}.scheduler_noise"
            noise = _mlx_tensor(fixture, tensors[name]).transpose(0, 2, 1, 3, 4)[0]
            next_sigma = float(warped_steps[next_step]) / 1000.0
            noisy_blocks[block_id] = (
                (1 - next_sigma) * block_prediction.astype(mx.float32)
                + next_sigma * noise.astype(mx.float32)
            ).astype(mx.bfloat16)
            mx.eval(noisy_blocks[block_id])
        scheduler_seconds += time.perf_counter() - scheduler_started

        refresh_input = prediction[:, :3].astype(mx.bfloat16)
        refresh_flow = causal(
            [refresh_input],
            timestep=mx.zeros((1, 3)),
            embedded_context=embedded_context,
            cross_kv_caches=cross_kv,
            current_start=window.start_frame * TOKENS_PER_FRAME,
            updating_cache=True,
        )[0]
        refresh_seconds += _timed_eval(refresh_flow)

    if any(row is None for row in generated):
        raise RuntimeError("rolling schedule did not finalize every latent block")
    output = mx.concatenate([row for row in generated if row is not None], axis=1)
    mx.eval(output)
    return output, {
        "dit_denoise_seconds": denoise_seconds,
        "clean_cache_refresh_seconds": refresh_seconds,
        "scheduler_seconds": scheduler_seconds,
    }


def _decode(vae: Any, latents: mx.array) -> tuple[np.ndarray, dict[str, float]]:
    cache: list[Any] = [None] * _decoder_cache_slots(vae.decoder)
    chunks = []
    vae_seconds = 0.0
    materialization_seconds = 0.0
    for frame in range(latents.shape[1]):
        latent = latents[:, frame : frame + 1][None]
        decoded = _causal_decode_frame(vae, latent, cache)
        vae_seconds += _timed_eval(decoded)
        started = time.perf_counter()
        pixels = np.array(decoded[0].transpose(1, 2, 3, 0), dtype=np.float32)
        chunks.append(np.clip((pixels + 1.0) * 127.5, 0, 255).astype(np.uint8))
        materialization_seconds += time.perf_counter() - started
    return np.concatenate(chunks), {
        "vae_seconds": vae_seconds,
        "pixel_materialization_seconds": materialization_seconds,
    }


def _one_trial(
    model: Any,
    vae: Any,
    cross_kv: Any,
    embedded_context: mx.array,
    fixture: Any,
    tensors: dict[str, Any],
    config: MLXBenchmarkConfig,
    output: Path,
) -> dict[str, Any]:
    mx.clear_cache()
    mx.reset_peak_memory()
    thermal_before = _thermal_state()
    started = time.perf_counter()
    latents, generation = _generate(
        model, cross_kv, embedded_context, fixture, tensors, config
    )
    pixels, decode = _decode(vae, latents)
    if pixels.shape[0] != config.pixel_frames:
        raise RuntimeError(
            f"decoder emitted {pixels.shape[0]} frames, expected {config.pixel_frames}"
        )
    encoding_seconds = _encode_h264(pixels, output, config.playback_fps)
    wall_seconds = time.perf_counter() - started
    return {
        **generation,
        **decode,
        "encoding_seconds": encoding_seconds,
        "wall_seconds": wall_seconds,
        "generated_fps": config.pixel_frames / wall_seconds,
        "peak_mlx_memory_gib": mx.get_peak_memory() / 2**30,
        "active_mlx_memory_gib": mx.get_active_memory() / 2**30,
        "thermal_before": thermal_before,
        "thermal_after": _thermal_state(),
        "encoded_bytes": output.stat().st_size,
    }


def _median_fields(trials: list[dict[str, Any]]) -> dict[str, float]:
    fields = (
        "dit_denoise_seconds",
        "clean_cache_refresh_seconds",
        "scheduler_seconds",
        "vae_seconds",
        "pixel_materialization_seconds",
        "encoding_seconds",
        "wall_seconds",
        "generated_fps",
        "peak_mlx_memory_gib",
        "active_mlx_memory_gib",
    )
    return {field: statistics.median(float(row[field]) for row in trials) for field in fields}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.warmups < 0 or args.trials < 1:
        raise ValueError("warmups must be nonnegative and trials must be positive")

    config = MLXBenchmarkConfig()
    fixture = load_fixture(args.fixture)
    if fixture.metadata.get("algorithm") != "rolling_forcing_long_video":
        raise ValueError("benchmark requires a Rolling Forcing fixture")
    if int(fixture.metadata.get("latent_frames", 0)) < config.latent_frames:
        raise ValueError("fixture is shorter than the frozen benchmark")
    tensors = {tensor.name: tensor for tensor in fixture.tensors}
    model = _load_model(args.model)
    vae = load_vae_decoder(args.vae)
    prompt = _mlx_tensor(fixture, tensors["prompt_embedding"])
    embedded_context = model.embed_text([prompt[0]])
    cross_kv = model.prepare_cross_kv(embedded_context)
    mx.eval(embedded_context, cross_kv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        root = Path(temporary)
        for index in range(args.warmups):
            _one_trial(
                model,
                vae,
                cross_kv,
                embedded_context,
                fixture,
                tensors,
                config,
                root / f"warmup-{index}.mp4",
            )
        trials = [
            _one_trial(
                model,
                vae,
                cross_kv,
                embedded_context,
                fixture,
                tensors,
                config,
                root / f"trial-{index}.mp4",
            )
            for index in range(args.trials)
        ]

    report = {
        "schema": "liveframe.mlx-rolling-benchmark.v1",
        "claim_eligible": True,
        "configuration": {
            "latent_frames": config.latent_frames,
            "pixel_frames": config.pixel_frames,
            "playback_fps": config.playback_fps,
            "playback_seconds": config.playback_seconds,
            "block_frames": BLOCK_FRAMES,
            "window_count": config.schedule.window_count,
            "transformer_forwards": config.schedule.transformer_forward_count,
            "context_noise": 0,
            "prompt_encoding": "excluded-precomputed-frozen-fixture",
            "model_loading": "excluded-steady-state",
            "ttff": "not-measured-batch-decode-after-generation",
            "power": "not-measured-powermetrics-requires-privilege",
        },
        "fixture": {
            "id": fixture.fixture_id,
            "manifest_sha256": _sha256(args.fixture / "fixture.json"),
        },
        "model_sha256": _sha256(args.model / "model.safetensors"),
        "vae_sha256": _sha256(args.vae),
        "runtime": _runtime_provenance(),
        "warmups": args.warmups,
        "trials": trials,
        "median": _median_fields(trials),
    }
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_output.replace(args.output)
    print(json.dumps({"output": str(args.output), "median": report["median"]}, sort_keys=True))


if __name__ == "__main__":
    main()