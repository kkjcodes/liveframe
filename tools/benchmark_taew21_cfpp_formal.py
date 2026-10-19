"""Run the frozen formal TAEW2.1 FP32-im2col decoder + complete-wall benchmark.

Executes the original one-warmup/five-trial formal benchmark authorized by the
screening PASS, per the frozen protocol (`artifacts/taew21-protocol.v1.json`,
`benchmark` section) and precision policy
(`artifacts/taew21-precision-policy.v1.json`). The denoiser, scheduler, KV
handling, fixture, and native decoder default are untouched: this tool reuses
`tools/benchmark_mlx_cfpp.py` verbatim for everything except the per-block
decode, which routes to `TAEW21MLXDecoder` in fp32-im2col mode. Frozen before
any formal timing is visible.
"""

# mypy: ignore-errors

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

import benchmark_mlx_cfpp as native_benchmark  # noqa: E402

from liveframe.backends.mlx_causal import CausalWanForward  # noqa: E402
from liveframe.decoders.taew21_mlx import (  # noqa: E402
    STARTUP_FRAMES_TO_TRIM,
    TAEW21MLXDecoder,
)
from liveframe.fixtures import load_fixture  # noqa: E402
from liveframe.manifests import file_sha256  # noqa: E402
from liveframe.mlx_cfpp_benchmark import (  # noqa: E402
    CFPPBenchmarkConfig,
    median_fields,
    require_swap_stable,
)
from liveframe.mlx_parity import _mlx_tensor  # noqa: E402
from liveframe.mlx_pipeline_parity import _load_model  # noqa: E402
from liveframe.mlx_rolling_parity import _runtime_provenance, _sha256  # noqa: E402
from liveframe.perceptual_policy import canonical_digest  # noqa: E402

TOKENS_PER_FRAME = native_benchmark.TOKENS_PER_FRAME
MEDIAN_FIELDS = native_benchmark.MEDIAN_FIELDS


def _one_tiny_trial(
    model: Any,
    tiny: TAEW21MLXDecoder,
    embedded_context: mx.array,
    cross_kv: Any,
    fixture: Any,
    tensors: dict[str, Any],
    config: CFPPBenchmarkConfig,
    video_output: Path,
) -> tuple[dict[str, Any], mx.array]:
    mx.clear_cache()
    mx.reset_peak_memory()
    memory_before = native_benchmark._memory_snapshot()
    thermal_before = native_benchmark._thermal_state()
    causal = CausalWanForward(model)
    tiny_memory: dict[int, Any] = {}
    initial = _mlx_tensor(fixture, tensors["initial_noise"])
    generated = []
    pixel_chunks = []
    blocks = []
    emitted_frames = 0
    decoded_seen = 0
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
            elapsed = native_benchmark._timed_eval(
                x0, *[cache.k for cache in causal.caches if cache.k is not None]
            )
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
        tiny_latent = latent.transpose(1, 2, 3, 0).astype(mx.float32)
        decode_started = time.perf_counter()
        decoded = tiny._decode_latent(tiny_latent, tiny_memory)
        decoded_seen += len(decoded)
        if decoded_seen <= STARTUP_FRAMES_TO_TRIM:
            raise RuntimeError(
                f"block {block} produced no post-startup frames; "
                f"decoded_seen={decoded_seen}"
            )
        if block == 0:
            decoded = decoded[STARTUP_FRAMES_TO_TRIM:]
        value = mx.stack(decoded, axis=1)
        mx.eval(value)
        block_vae = time.perf_counter() - decode_started
        materialize_started = time.perf_counter()
        pixels = np.asarray(value[0] * 255.0).round().clip(0, 255).astype(np.uint8)
        block_materialization = time.perf_counter() - materialize_started
        expected_emission = 1 if block == 0 else 4
        if pixels.shape[0] != expected_emission:
            raise RuntimeError(
                f"block {block} emitted {pixels.shape[0]} frames, expected {expected_emission}"
            )
        if not np.isfinite(pixels).all():
            raise RuntimeError(f"block {block} emitted non-finite pixels")
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
        block_refresh = native_benchmark._timed_eval(
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
    encoding_seconds = native_benchmark._encode_h264(
        pixels, video_output, config.playback_fps
    )
    wall_seconds = time.perf_counter() - started
    memory_after = native_benchmark._memory_snapshot()
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
            "thermal_after": native_benchmark._thermal_state(),
            "pixel_bytes": len(pixel_payload),
            "pixel_sha256": pixel_sha256,
            "encoded_bytes": video_output.stat().st_size,
            "encoded_sha256": _sha256(video_output),
            "blocks": blocks,
        },
        latents,
    )


def _bootstrap_speedup(
    baseline_walls: list[float],
    candidate_walls: list[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    base = np.asarray(baseline_walls, dtype=np.float64)
    cand = np.asarray(candidate_walls, dtype=np.float64)
    ratios = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        base_sample = rng.choice(base, size=base.size, replace=True)
        cand_sample = rng.choice(cand, size=cand.size, replace=True)
        ratios[index] = np.median(base_sample) / np.median(cand_sample)
    return {
        "resamples": resamples,
        "seed": seed,
        "point_speedup": float(np.median(base) / np.median(cand)),
        "lower_confidence_bound_95": float(np.quantile(ratios, 0.05)),
        "ratio_quantiles": {
            "p05": float(np.quantile(ratios, 0.05)),
            "p50": float(np.quantile(ratios, 0.50)),
            "p95": float(np.quantile(ratios, 0.95)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--parity-result", type=Path, required=True)
    parser.add_argument("--screen-result", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace formal result: {args.output}")

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    policy_payload = dict(policy)
    policy_digest = policy_payload.pop("policy_digest")
    if canonical_digest(policy_payload) != policy_digest:
        raise ValueError("TAEW2.1 precision policy canonical digest mismatch")
    provenance = policy["provenance"]
    if (
        provenance["decoder_sha256"]
        != file_sha256(REPOSITORY_ROOT / "src/liveframe/decoders/taew21_mlx.py")
        or provenance["parity_v5_sha256"] != file_sha256(args.parity_result)
        or provenance["protocol_sha256"] != file_sha256(args.protocol)
        or provenance["baseline_sha256"] != file_sha256(args.baseline)
        or file_sha256(args.weights) != args.weights_sha256
    ):
        raise ValueError("TAEW2.1 formal benchmark provenance mismatch")
    screen = json.loads(args.screen_result.read_text(encoding="utf-8"))
    if (
        screen.get("schema") != "liveframe.taew21-fp32-im2col-screen.v1"
        or not screen.get("passed")
        or screen.get("verdict")
        != "proceed-to-original-five-trial-formal-decoder-and-complete-wall-benchmark"
    ):
        raise ValueError("TAEW2.1 formal benchmark requires a passed screening result")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    spec = protocol["benchmark"]
    warmups = int(spec["warmups"])
    trials_required = int(spec["trials"])
    cache_limit_bytes = int(spec["allocator_cache_limit_bytes"])
    memory_limit_bytes = int(spec["allocator_memory_limit_bytes"])
    maximum_swap_delta = float(spec["maximum_swap_delta_mib"])
    maximum_target_wall = float(spec["maximum_target_wall_seconds"])
    minimum_speedup = float(spec["minimum_complete_wall_speedup"])
    resamples = int(spec["bootstrap_resamples"])
    bootstrap_seed = int(spec["bootstrap_seed"])

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_walls = [float(trial["wall_seconds"]) for trial in baseline["trials"]]
    baseline_vae = [float(trial["vae_seconds"]) for trial in baseline["trials"]]

    config = CFPPBenchmarkConfig(
        max_start_swap_mib=0.0,
        max_run_swap_delta_mib=maximum_swap_delta,
        min_start_free_percent=25.0,
        cache_limit_gib=cache_limit_bytes / 2**30,
    )
    preflight = native_benchmark._preflight(config)
    allocator = native_benchmark._configure_allocator(config)
    if int(allocator["memory_limit_bytes"]) != memory_limit_bytes:
        raise RuntimeError(
            f"allocator memory limit {allocator['memory_limit_bytes']} does not match "
            f"frozen protocol value {memory_limit_bytes}"
        )
    fixture = load_fixture(args.fixture)
    native_benchmark._validate_fixture(fixture, config)
    assets = native_benchmark._validate_assets(args.model_dir, args.fixture, fixture)
    tensors = {tensor.name: tensor for tensor in fixture.tensors}
    model, _, _ = _load_model(args.model_dir)
    tiny = TAEW21MLXDecoder(
        args.weights,
        expected_sha256=args.weights_sha256,
        convolution_mode="fp32-im2col",
    )
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
        for index in range(warmups):
            print(f"warmup {index + 1}/{warmups}", flush=True)
            warmup, _ = _one_tiny_trial(
                model,
                tiny,
                embedded_context,
                cross_kv,
                fixture,
                tensors,
                config,
                root / f"warmup-{index}.mp4",
            )
            require_swap_stable(
                swap_delta_mib=float(warmup["swap_delta_mib"]), config=config
            )
        for index in range(trials_required):
            print(f"measured trial {index + 1}/{trials_required}", flush=True)
            video = (
                args.video_output
                if index == trials_required - 1
                else root / f"trial-{index}.mp4"
            )
            trial, latents = _one_tiny_trial(
                model,
                tiny,
                embedded_context,
                cross_kv,
                fixture,
                tensors,
                config,
                video,
            )
            trial["quality_anchor"] = native_benchmark._quality_anchor(
                fixture, tensors, latents
            )
            require_swap_stable(
                swap_delta_mib=float(trial["swap_delta_mib"]), config=config
            )
            if not trial["quality_anchor"]["passed"]:
                raise RuntimeError(
                    f"trial {index + 1} failed the pinned CF++ latent quality anchor"
                )
            quality = trial["quality_anchor"]
            trials.append(trial)

    pixel_digests = sorted({trial["pixel_sha256"] for trial in trials})
    if len(pixel_digests) != 1:
        raise RuntimeError(
            f"measured trials produced inconsistent pixel digests: {pixel_digests}"
        )

    candidate_walls = [float(trial["wall_seconds"]) for trial in trials]
    candidate_vae = [float(trial["vae_seconds"]) for trial in trials]
    complete_wall = _bootstrap_speedup(
        baseline_walls, candidate_walls, resamples=resamples, seed=bootstrap_seed
    )
    decoder_only = {
        "baseline_vae_median_seconds": statistics.median(baseline_vae),
        "candidate_vae_median_seconds": statistics.median(candidate_vae),
        "point_speedup": statistics.median(baseline_vae) / statistics.median(candidate_vae),
    }
    median_wall = statistics.median(candidate_walls)
    point_pass = median_wall <= maximum_target_wall
    confidence_pass = complete_wall["lower_confidence_bound_95"] >= minimum_speedup
    passed = point_pass and confidence_pass

    report = {
        "schema": "liveframe.taew21-cfpp-formal-benchmark.v1",
        "claim_eligible": True,
        "configuration": {
            "decoder": "taew2_1-mlx-fp32-im2col",
            "convolution_mode": "fp32-im2col",
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
            "model_loading": spec["model_loading"],
            "memory_limit": "Metal recommended working set",
            "free_cache_limit_gib": config.cache_limit_gib,
            "power": "not-measured-powermetrics-requires-privilege",
        },
        "provenance": {
            "policy_sha256": file_sha256(args.policy),
            "policy_digest": policy_digest,
            "protocol_sha256": file_sha256(args.protocol),
            "baseline_sha256": file_sha256(args.baseline),
            "parity_result_sha256": file_sha256(args.parity_result),
            "screen_result_sha256": file_sha256(args.screen_result),
            "taew21_weights_sha256": args.weights_sha256,
        },
        "preflight": preflight,
        "allocator": allocator,
        "fixture": {
            "id": fixture.fixture_id,
            "manifest_sha256": _sha256(args.fixture / "fixture.json"),
        },
        "assets": assets,
        "runtime": _runtime_provenance(),
        "warmups": warmups,
        "trials": trials,
        "median": median_fields(trials, MEDIAN_FIELDS),
        "quality_anchor": quality,
        "pixel_identity": {
            "sha256": pixel_digests[0],
            "bytes": int(trials[-1]["pixel_bytes"]),
            "consistent_across_trials": True,
        },
        "video": {
            "path": str(args.video_output),
            "sha256": _sha256(args.video_output),
            "bytes": args.video_output.stat().st_size,
        },
        "comparison": {
            "baseline_artifact": str(args.baseline),
            "baseline_wall_seconds": baseline_walls,
            "candidate_wall_seconds": candidate_walls,
            "median_candidate_wall_seconds": median_wall,
            "maximum_target_wall_seconds": maximum_target_wall,
            "minimum_complete_wall_speedup": minimum_speedup,
            "confidence_rule": spec["confidence_rule"],
            "complete_wall": complete_wall,
            "decoder_only": decoder_only,
            "point_pass": point_pass,
            "confidence_pass": confidence_pass,
        },
        "passed": passed,
    }
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_output, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "median_wall_seconds": median_wall,
                "complete_wall_point_speedup": complete_wall["point_speedup"],
                "complete_wall_lcb95": complete_wall["lower_confidence_bound_95"],
                "decoder_only_point_speedup": decoder_only["point_speedup"],
                "passed": passed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
