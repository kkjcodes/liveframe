"""Benchmark the frozen one-step CF++ pipeline on Apple MLX."""

# mypy: ignore-errors

from __future__ import annotations

import argparse
import hashlib
import json
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
from liveframe.mlx_cfpp_benchmark import (
    CFPPBenchmarkConfig,
    median_fields,
    parse_free_memory_percent,
    parse_swap_usage,
    require_clean_preflight,
    require_swap_stable,
)
from liveframe.mlx_parity import _metrics, _mlx_tensor, _numpy_tensor
from liveframe.mlx_pipeline_parity import _load_model
from liveframe.mlx_rolling_parity import _runtime_provenance, _sha256
from liveframe.mlx_vae_decode import _causal_decode_frame, _decoder_cache_slots

FIXTURE_ID = "cfpp-1step-native-21block-h100"
TOKENS_PER_FRAME = 1560
FIXTURE_MANIFEST_DIGEST = (
    "sha256:d621037dc84c416dd7182991fc8e3f10a0245894cf5fd74002a517dc65862eb9"
)
SOURCE_CHECKPOINT_DIGEST = (
    "sha256:bdb1b475fc88d528f510158a0990cd457f02c661b527ceb11ca9e4728533e2d0"
)
MODEL_DIGEST = "sha256:40735e0eb76c4a710163d337a5167cee2d7b1ed27dea9b1d8357714bebc1115b"
VAE_DIGEST = "sha256:977530e453dbfabbab31e2972e1577d8d7e2840ba7410c81aa3fd421c0cd7414"
PIXEL_DIGEST = "sha256:4b55f063d0164fd1c44358186af6cd04ae2dba1356dedd5b86403dfd101e8f61"
QUALITY_LIMITS = {"cosine": 0.999, "rmse": 0.035, "max_abs": 0.25}
MEDIAN_FIELDS = (
    "denoising_seconds",
    "refresh_seconds",
    "vae_seconds",
    "pixel_materialization_seconds",
    "encoding_seconds",
    "wall_seconds",
    "generated_fps",
    "ttff_seconds",
    "peak_mlx_memory_gib",
    "active_mlx_memory_gib",
    "cache_mlx_memory_gib",
    "swap_delta_mib",
)


def _command_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _thermal_state() -> list[str]:
    result = subprocess.run(
        ["pmset", "-g", "therm"], capture_output=True, check=False, text=True
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _memory_snapshot() -> dict[str, Any]:
    pressure = _command_output(["memory_pressure", "-Q"])
    swap = parse_swap_usage(_command_output(["sysctl", "vm.swapusage"]))
    return {
        "free_memory_percent": parse_free_memory_percent(pressure),
        "swap": swap,
        "pressure_output": pressure.splitlines(),
    }


def _top_processes(limit: int = 15) -> list[dict[str, Any]]:
    output = _command_output(["ps", "-axo", "pid=,rss=,comm="])
    rows = []
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        rows.append(
            {"pid": int(fields[0]), "rss_mib": int(fields[1]) / 1024.0, "command": fields[2]}
        )
    return sorted(rows, key=lambda row: row["rss_mib"], reverse=True)[:limit]


def _preflight(config: CFPPBenchmarkConfig) -> dict[str, Any]:
    snapshot = _memory_snapshot()
    require_clean_preflight(
        swap_used_mib=float(snapshot["swap"]["used_mib"]),
        free_memory_percent=float(snapshot["free_memory_percent"]),
        config=config,
    )
    return {
        **snapshot,
        "limits": {
            "max_start_swap_mib": config.max_start_swap_mib,
            "max_run_swap_delta_mib": config.max_run_swap_delta_mib,
            "min_start_free_percent": config.min_start_free_percent,
        },
        "top_processes": _top_processes(),
    }


def _configure_allocator(config: CFPPBenchmarkConfig) -> dict[str, Any]:
    device = mx.device_info()
    memory_limit = int(device["max_recommended_working_set_size"])
    cache_limit = int(config.cache_limit_gib * 2**30)
    previous_memory_limit = mx.set_memory_limit(memory_limit)
    previous_cache_limit = mx.set_cache_limit(cache_limit)
    return {
        "memory_limit_bytes": memory_limit,
        "memory_limit_gib": memory_limit / 2**30,
        "cache_limit_bytes": cache_limit,
        "cache_limit_gib": cache_limit / 2**30,
        "previous_memory_limit_bytes": previous_memory_limit,
        "previous_cache_limit_bytes": previous_cache_limit,
        "policy": "device-recommended-working-set-with-bounded-free-cache.v1",
    }


def _timed_eval(value: mx.array, *additional: mx.array) -> float:
    started = time.perf_counter()
    mx.eval(value, *additional)
    return time.perf_counter() - started


def _quality_anchor(
    fixture: Any, tensors: dict[str, Any], latents: mx.array
) -> dict[str, Any]:
    candidate = np.array(
        latents[None].transpose(0, 2, 1, 3, 4).astype(mx.float32)
    )
    reference = _numpy_tensor(fixture, tensors["output_latents"])
    metrics = _metrics(reference, candidate)
    failures = []
    if float(metrics["cosine"]) < QUALITY_LIMITS["cosine"]:
        failures.append("cosine")
    for name in ("rmse", "max_abs"):
        if float(metrics[name]) > QUALITY_LIMITS[name]:
            failures.append(name)
    return {
        "passed": not failures,
        "failures": failures,
        "metrics": metrics,
        "limits": QUALITY_LIMITS,
        "reference": "pinned H100 BF16 output latents; existing MLX parity anchor",
    }


def _encode_h264(pixels: np.ndarray, output: Path, fps: int) -> float:
    height, width = pixels.shape[1:3]
    temporary = output.with_suffix(output.suffix + ".tmp.mp4")
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
        str(temporary),
    ]
    started = time.perf_counter()
    subprocess.run(command, input=pixels.tobytes(order="C"), check=True)
    temporary.replace(output)
    return time.perf_counter() - started


def _one_trial(
    model: Any,
    vae: Any,
    embedded_context: mx.array,
    cross_kv: Any,
    fixture: Any,
    tensors: dict[str, Any],
    config: CFPPBenchmarkConfig,
    video_output: Path,
) -> tuple[dict[str, Any], mx.array]:
    mx.clear_cache()
    mx.reset_peak_memory()
    memory_before = _memory_snapshot()
    thermal_before = _thermal_state()
    causal = CausalWanForward(model)
    decoder_cache: list[Any] = [None] * _decoder_cache_slots(vae.decoder)
    initial = _mlx_tensor(fixture, tensors["initial_noise"])
    generated = []
    pixel_chunks = []
    blocks = []
    emitted_frames = 0
    denoising_seconds = 0.0
    refresh_seconds = 0.0
    vae_seconds = 0.0
    materialization_seconds = 0.0
    started = time.perf_counter()
    ttff_seconds = None

    for block in range(config.latent_blocks):
        block_started = time.perf_counter()
        latent = initial[:, block : block + 1].transpose(0, 2, 1, 3, 4)[0]
        current_start = block * TOKENS_PER_FRAME
        block_denoising = 0.0
        configured_steps = config.schedule.configured_timesteps(block)

        for step, _ in enumerate(configured_steps):
            prefix = f"block.{block:02d}.step.{step:02d}"
            timestep = float(_mlx_tensor(fixture, tensors[f"{prefix}.timestep"]).item())
            flow = causal(
                [latent],
                timestep=mx.full((1, 1), timestep),
                embedded_context=embedded_context,
                cross_kv_caches=cross_kv,
                current_start=current_start,
            )[0]
            x0 = latent - (timestep / 1000.0) * flow
            elapsed = _timed_eval(x0, *[cache.k for cache in causal.caches if cache.k is not None])
            block_denoising += elapsed
            if step < len(configured_steps) - 1:
                noise = _mlx_tensor(fixture, tensors[f"{prefix}.scheduler_noise"]).transpose(
                    0, 2, 1, 3, 4
                )[0]
                next_prefix = f"block.{block:02d}.step.{step + 1:02d}"
                next_timestep = float(
                    _mlx_tensor(fixture, tensors[f"{next_prefix}.timestep"]).item()
                )
                sigma = next_timestep / 1000.0
                latent = (
                    (1.0 - sigma) * x0.astype(mx.float32)
                    + sigma * noise.astype(mx.float32)
                ).astype(mx.bfloat16)
                mx.eval(latent)
            else:
                latent = x0.astype(mx.bfloat16)
                mx.eval(latent)

        generated.append(latent)
        decoded = _causal_decode_frame(vae, latent[None], decoder_cache)
        block_vae = _timed_eval(decoded)
        materialize_started = time.perf_counter()
        pixels = np.array(decoded[0].transpose(1, 2, 3, 0), dtype=np.float32)
        pixels = np.clip((pixels + 1.0) * 127.5, 0, 255).astype(np.uint8)
        block_materialization = time.perf_counter() - materialize_started
        expected_emission = 1 if block == 0 else 4
        if pixels.shape[0] != expected_emission:
            raise RuntimeError(
                f"block {block} emitted {pixels.shape[0]} frames, expected {expected_emission}"
            )
        pixel_chunks.append(pixels)
        emitted_frames += int(pixels.shape[0])
        if ttff_seconds is None:
            ttff_seconds = time.perf_counter() - started

        refresh_flow = causal(
            [latent],
            timestep=mx.zeros((1, 1)),
            embedded_context=embedded_context,
            cross_kv_caches=cross_kv,
            current_start=current_start,
        )[0]
        block_refresh = _timed_eval(
            refresh_flow, *[cache.k for cache in causal.caches if cache.k is not None]
        )
        denoising_seconds += block_denoising
        refresh_seconds += block_refresh
        vae_seconds += block_vae
        materialization_seconds += block_materialization
        blocks.append(
            {
                "block": block,
                "denoising_seconds": block_denoising,
                "refresh_seconds": block_refresh,
                "vae_seconds": block_vae,
                "pixel_materialization_seconds": block_materialization,
                "emitted_pixel_frames": int(pixels.shape[0]),
                "cumulative_pixel_frames": emitted_frames,
                "wall_seconds": time.perf_counter() - block_started,
            }
        )
        print(
            f"block {block + 1}/{config.latent_blocks}; emitted "
            f"{emitted_frames}/{config.pixel_frames}",
            flush=True,
        )

    latents = mx.concatenate(generated, axis=1)
    mx.eval(latents)
    pixels = np.concatenate(pixel_chunks)
    if pixels.shape[0] != config.pixel_frames:
        raise RuntimeError(
            f"pipeline emitted {pixels.shape[0]} frames, expected {config.pixel_frames}"
        )
    pixel_payload = pixels.tobytes(order="C")
    pixel_sha256 = "sha256:" + hashlib.sha256(pixel_payload).hexdigest()
    encoding_seconds = _encode_h264(pixels, video_output, config.playback_fps)
    wall_seconds = time.perf_counter() - started
    memory_after = _memory_snapshot()
    swap_delta = (
        float(memory_after["swap"]["used_mib"])
        - float(memory_before["swap"]["used_mib"])
    )
    return (
        {
            "denoising_seconds": denoising_seconds,
            "refresh_seconds": refresh_seconds,
            "vae_seconds": vae_seconds,
            "pixel_materialization_seconds": materialization_seconds,
            "encoding_seconds": encoding_seconds,
            "wall_seconds": wall_seconds,
            "generated_fps": config.pixel_frames / wall_seconds,
            "ttff_seconds": float(ttff_seconds),
            "peak_mlx_memory_gib": mx.get_peak_memory() / 2**30,
            "active_mlx_memory_gib": mx.get_active_memory() / 2**30,
            "cache_mlx_memory_gib": mx.get_cache_memory() / 2**30,
            "swap_delta_mib": swap_delta,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "thermal_before": thermal_before,
            "thermal_after": _thermal_state(),
            "pixel_bytes": len(pixel_payload),
            "pixel_sha256": pixel_sha256,
            "encoded_bytes": video_output.stat().st_size,
            "encoded_sha256": _sha256(video_output),
            "blocks": blocks,
        },
        latents,
    )


def _validate_fixture(fixture: Any, config: CFPPBenchmarkConfig) -> None:
    metadata = fixture.metadata
    expected = {
        "latent_blocks": config.latent_blocks,
        "denoising_forwards": config.schedule.denoising_evaluation_count,
        "refresh_forwards": config.latent_blocks,
    }
    if fixture.fixture_id != FIXTURE_ID:
        raise ValueError(f"benchmark requires fixture {FIXTURE_ID}")
    for name, value in expected.items():
        if int(metadata.get(name, -1)) != value:
            raise ValueError(f"fixture metadata {name} must be {value}")


def _validate_assets(model_dir: Path, fixture_dir: Path, fixture: Any) -> dict[str, Any]:
    conversion_path = model_dir / "liveframe-conversion.json"
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    actual = {
        "fixture_manifest_sha256": _sha256(fixture_dir / "fixture.json"),
        "model_sha256": _sha256(model_dir / "model.safetensors"),
        "vae_sha256": _sha256(model_dir / "vae.safetensors"),
    }
    expected = {
        "fixture_manifest_sha256": FIXTURE_MANIFEST_DIGEST,
        "model_sha256": MODEL_DIGEST,
        "vae_sha256": VAE_DIGEST,
    }
    failures = [name for name, digest in expected.items() if actual[name] != digest]
    if conversion.get("schema") != "liveframe.cfpp-mlx-conversion.v1":
        failures.append("conversion.schema")
    if conversion.get("source_sha256") != SOURCE_CHECKPOINT_DIGEST:
        failures.append("conversion.source_sha256")
    if conversion.get("model_sha256") != MODEL_DIGEST:
        failures.append("conversion.model_sha256")
    if fixture.metadata.get("checkpoint_sha256") != SOURCE_CHECKPOINT_DIGEST:
        failures.append("fixture.checkpoint_sha256")
    stock = conversion.get("stock_assets", {})
    if not isinstance(stock, dict) or stock.get("vae.safetensors") != VAE_DIGEST:
        failures.append("conversion.stock_assets.vae.safetensors")
    if failures:
        raise ValueError("CF++ benchmark asset ancestry mismatch: " + ", ".join(failures))
    return {
        **actual,
        "conversion_manifest_sha256": "sha256:"
        + hashlib.sha256(conversion_path.read_bytes()).hexdigest(),
        "conversion": conversion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--max-start-swap-mib", type=float, default=128.0)
    parser.add_argument("--max-run-swap-delta-mib", type=float, default=128.0)
    parser.add_argument("--min-start-free-percent", type=float, default=25.0)
    parser.add_argument("--cache-limit-gib", type=float, default=1.0)
    args = parser.parse_args()
    if args.warmups < 0 or args.trials < 1:
        raise ValueError("warmups must be nonnegative and trials must be positive")
    config = CFPPBenchmarkConfig(
        max_start_swap_mib=args.max_start_swap_mib,
        max_run_swap_delta_mib=args.max_run_swap_delta_mib,
        min_start_free_percent=args.min_start_free_percent,
        cache_limit_gib=args.cache_limit_gib,
    )
    preflight = _preflight(config)
    if args.preflight_only:
        print(json.dumps({"passed": True, "preflight": preflight}, indent=2, sort_keys=True))
        return

    allocator = _configure_allocator(config)
    fixture = load_fixture(args.fixture)
    _validate_fixture(fixture, config)
    assets = _validate_assets(args.model_dir, args.fixture, fixture)
    tensors = {tensor.name: tensor for tensor in fixture.tensors}
    model, _, _ = _load_model(args.model_dir)
    vae = load_vae_decoder(args.model_dir / "vae.safetensors")
    prompt = _mlx_tensor(fixture, tensors["prompt_embedding"])
    embedded_context = model.embed_text([prompt[0]])
    cross_kv = model.prepare_cross_kv(embedded_context)
    mx.eval(embedded_context, cross_kv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.video_output.parent.mkdir(parents=True, exist_ok=True)
    trials = []
    quality = None
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        root = Path(temporary)
        for index in range(args.warmups):
            print(f"warmup {index + 1}/{args.warmups}", flush=True)
            warmup, _ = _one_trial(
                model,
                vae,
                embedded_context,
                cross_kv,
                fixture,
                tensors,
                config,
                root / f"warmup-{index}.mp4",
            )
            print(json.dumps({"warmup": index + 1, "telemetry": {
                name: warmup[name]
                for name in (
                    "wall_seconds",
                    "generated_fps",
                    "peak_mlx_memory_gib",
                    "active_mlx_memory_gib",
                    "cache_mlx_memory_gib",
                    "swap_delta_mib",
                )
            }}, sort_keys=True), flush=True)
            require_swap_stable(
                swap_delta_mib=float(warmup["swap_delta_mib"]), config=config
            )
        for index in range(args.trials):
            print(f"measured trial {index + 1}/{args.trials}", flush=True)
            video = args.video_output if index == args.trials - 1 else root / f"trial-{index}.mp4"
            trial, latents = _one_trial(
                model,
                vae,
                embedded_context,
                cross_kv,
                fixture,
                tensors,
                config,
                video,
            )
            trial["quality_anchor"] = _quality_anchor(fixture, tensors, latents)
            require_swap_stable(
                swap_delta_mib=float(trial["swap_delta_mib"]), config=config
            )
            if not trial["quality_anchor"]["passed"]:
                raise RuntimeError(
                    f"trial {index + 1} failed the pinned CF++ latent quality anchor"
                )
            if trial["pixel_sha256"] != PIXEL_DIGEST:
                raise RuntimeError(
                    f"trial {index + 1} failed the pinned raw RGB identity anchor"
                )
            quality = trial["quality_anchor"]
            trials.append(trial)

    report = {
        "schema": "liveframe.mlx-cfpp-benchmark.v1",
        "claim_eligible": not args.diagnostic,
        "claim_ineligibility_reason": (
            "explicit diagnostic run" if args.diagnostic else None
        ),
        "configuration": {
            "student_steps": config.student_steps,
            "latent_blocks": config.latent_blocks,
            "pixel_frames": config.pixel_frames,
            "playback_fps": config.playback_fps,
            "playback_seconds": config.playback_seconds,
            "denoising_forwards": config.schedule.denoising_evaluation_count,
            "refresh_forwards": config.latent_blocks,
            "transformer_forwards": config.transformer_forwards,
            "context_noise": 0,
            "prompt_encoding": "excluded-precomputed-frozen-fixture",
            "model_loading": "excluded-steady-state",
            "memory_limit": "Metal recommended working set",
            "free_cache_limit_gib": config.cache_limit_gib,
            "power": "not-measured-powermetrics-requires-privilege",
        },
        "preflight": preflight,
        "allocator": allocator,
        "fixture": {
            "id": fixture.fixture_id,
            "manifest_sha256": _sha256(args.fixture / "fixture.json"),
        },
        "assets": assets,
        "runtime": _runtime_provenance(),
        "warmups": args.warmups,
        "trials": trials,
        "median": median_fields(trials, MEDIAN_FIELDS),
        "quality_anchor": quality,
        "pixel_identity_anchor": {
            "sha256": PIXEL_DIGEST,
            "bytes": int(trials[-1]["pixel_bytes"]),
            "passed": all(trial["pixel_sha256"] == PIXEL_DIGEST for trial in trials),
        },
        "video": {
            "path": str(args.video_output),
            "sha256": _sha256(args.video_output),
            "bytes": args.video_output.stat().st_size,
        },
    }
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_output.replace(args.output)
    print(json.dumps({"output": str(args.output), "median": report["median"]}, sort_keys=True))


if __name__ == "__main__":
    main()