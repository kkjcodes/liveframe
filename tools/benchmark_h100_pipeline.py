"""Benchmark honest blockwise CF++ generation and incremental VAE on H100."""

# mypy: ignore-errors

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import torch

sys.path.insert(0, str(Path.cwd()))

from capture_h100_pipeline import (
    CONFIGURED_STEPS,
    FIRST_BLOCK_CONFIGURED_STEPS,
    TOKENS_PER_FRAME,
    _load_generator,
    _make_caches,
    _warped_steps,
)
from utils.wan_wrapper import WanVAEWrapper  # type: ignore[import-not-found]

T = TypeVar("T")


def _timed(operation: Callable[[], T]) -> tuple[T, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    return result, start.elapsed_time(end) / 1000.0


def _generator_call(
    generator: Any,
    noisy: torch.Tensor,
    conditional: dict[str, torch.Tensor],
    timestep: torch.Tensor,
    kv_cache: list[dict[str, Any]],
    cross_cache: list[dict[str, Any]],
    current_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return generator(
        noisy_image_or_video=noisy,
        conditional_dict=conditional,
        timestep=timestep,
        kv_cache=kv_cache,
        crossattn_cache=cross_cache,
        current_start=current_start,
    )


def _decode_call(vae: Any, latent: torch.Tensor) -> torch.Tensor:
    return vae.decode_to_pixel(latent, use_cache=True)


def _gpu_telemetry() -> dict[str, float]:
    row = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    temperature, power, memory, utilization = (float(value.strip()) for value in row.split(","))
    return {
        "temperature_c": temperature,
        "power_w": power,
        "memory_mib": memory,
        "utilization_percent": utilization,
    }


def _queue_metrics(
    emission_events: list[tuple[float, int]], *, duration_seconds: float = 60.0
) -> dict[str, Any]:
    playback_fps = 16
    deadlines = int(duration_seconds * playback_fps)
    first_emission = emission_events[0][0]
    event_index = 0
    queue_depth = 0
    underruns = 0
    depths = []
    for frame_index in range(deadlines):
        deadline = first_emission + frame_index / playback_fps
        while event_index < len(emission_events) and emission_events[event_index][0] <= deadline:
            queue_depth += emission_events[event_index][1]
            event_index += 1
        if queue_depth:
            queue_depth -= 1
        else:
            underruns += 1
        depths.append(queue_depth)
    return {
        "playback_fps": playback_fps,
        "duration_seconds": duration_seconds,
        "required_frames": deadlines,
        "underruns": underruns,
        "minimum_depth": min(depths),
        "maximum_depth": max(depths),
        "final_depth": depths[-1],
        "non_draining_last_15_seconds": depths[-1] >= depths[-1 - 15 * playback_fps],
    }


def _one_run(
    generator: Any,
    vae: Any,
    *,
    student_steps: int,
    seed: int,
    latent_blocks: int,
    telemetry_interval: int,
) -> dict[str, Any]:
    cpu_rng = torch.Generator(device="cpu").manual_seed(seed)
    initial_noise = torch.randn(
        (1, latent_blocks, 16, 60, 104), generator=cpu_rng, dtype=torch.float32
    ).to(torch.bfloat16)
    prompt = torch.linspace(
        -0.25, 0.25, 512 * 4096, dtype=torch.float32
    ).reshape(1, 512, 4096).to(torch.bfloat16).cuda()
    conditional = {"prompt_embeds": prompt}
    kv_cache, cross_cache = _make_caches(generator)
    vae.model.clear_cache()

    first_steps = _warped_steps(generator, FIRST_BLOCK_CONFIGURED_STEPS)
    later_steps = _warped_steps(generator, CONFIGURED_STEPS[student_steps])
    block_rows = []
    emission_events: list[tuple[float, int]] = []
    thermal_samples = []
    emitted_frames = 0
    run_started = time.perf_counter()

    for block in range(latent_blocks):
        steps = first_steps if block == 0 else later_steps
        noisy = initial_noise[:, block : block + 1].cuda()
        current_start = block * TOKENS_PER_FRAME
        denoising_seconds = 0.0

        for step, warped_timestep in enumerate(steps):
            timestep = torch.full(
                (1, 1), warped_timestep, device="cuda", dtype=torch.float32
            )

            denoise = partial(
                _generator_call,
                generator,
                noisy,
                conditional,
                timestep,
                kv_cache,
                cross_cache,
                current_start,
            )
            (_, prediction), elapsed = _timed(denoise)
            denoising_seconds += elapsed
            if step < len(steps) - 1:
                scheduler_noise = torch.randn(
                    prediction.shape, generator=cpu_rng, dtype=torch.float32
                ).to(torch.bfloat16).cuda()
                sigma = steps[step + 1] / 1000.0
                noisy = (
                    (1.0 - sigma) * prediction.double()
                    + sigma * scheduler_noise.double()
                ).to(torch.bfloat16)

        pixels, decode_seconds = _timed(partial(_decode_call, vae, prediction))
        emitted_frames += pixels.shape[1]
        emission_events.append((time.perf_counter() - run_started, pixels.shape[1]))

        context_timestep = torch.zeros((1, 1), device="cuda", dtype=torch.float32)

        refresh = partial(
            _generator_call,
            generator,
            prediction,
            conditional,
            context_timestep,
            kv_cache,
            cross_cache,
            current_start,
        )
        _, refresh_seconds = _timed(refresh)
        block_rows.append(
            {
                "block": block,
                "denoising_seconds": denoising_seconds,
                "refresh_seconds": refresh_seconds,
                "decode_seconds": decode_seconds,
                "emitted_pixel_frames": pixels.shape[1],
                "ttff_path_seconds": (
                    denoising_seconds + decode_seconds if block == 0 else None
                ),
                "total_seconds": denoising_seconds + refresh_seconds + decode_seconds,
            }
        )
        if block % telemetry_interval == 0 or block == latent_blocks - 1:
            thermal_samples.append(
                {
                    "block": block,
                    "elapsed_seconds": time.perf_counter() - run_started,
                    **_gpu_telemetry(),
                }
            )

    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - run_started
    measured_seconds = sum(row["total_seconds"] for row in block_rows)
    return {
        "student_steps": student_steps,
        "emitted_pixel_frames": emitted_frames,
        "wall_seconds": wall_seconds,
        "measured_gpu_seconds": measured_seconds,
        "generated_pixel_fps": emitted_frames / wall_seconds,
        "gpu_only_pixel_fps": emitted_frames / measured_seconds,
        "ttff_seconds": block_rows[0]["ttff_path_seconds"],
        "first_block": block_rows[0],
        "steady_block_mean_seconds": statistics.mean(
            row["total_seconds"] for row in block_rows[1:]
        ),
        "steady_denoising_mean_seconds": statistics.mean(
            row["denoising_seconds"] for row in block_rows[1:]
        ),
        "steady_refresh_mean_seconds": statistics.mean(
            row["refresh_seconds"] for row in block_rows[1:]
        ),
        "steady_decode_mean_seconds": statistics.mean(
            row["decode_seconds"] for row in block_rows[1:]
        ),
        "queue": _queue_metrics(emission_events),
        "thermal_samples": thermal_samples,
        "blocks": block_rows,
    }


def benchmark(
    checkpoint: Path,
    *,
    student_steps: int,
    repeats: int,
    latent_blocks: int,
    local_attn_frames: int,
    sink_frames: int,
    telemetry_interval: int,
) -> dict[str, Any]:
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load_started = time.perf_counter()
    if latent_blocks < 21:
        raise ValueError("sustained benchmark requires at least 21 latent blocks")
    if local_attn_frames < 2 or not 0 <= sink_frames < local_attn_frames:
        raise ValueError("invalid rolling attention or sink frame count")
    generator = _load_generator(
        checkpoint, local_attn_size=local_attn_frames, sink_size=sink_frames
    )
    vae = WanVAEWrapper().to(device="cuda", dtype=torch.bfloat16).eval()
    load_seconds = time.perf_counter() - load_started

    # The first complete run warms all kernels and allocator paths. It is not
    # included in the steady benchmark samples.
    warmup_blocks = min(21, latent_blocks)
    warmup = _one_run(
        generator,
        vae,
        student_steps=student_steps,
        seed=20260721,
        latent_blocks=warmup_blocks,
        telemetry_interval=telemetry_interval,
    )
    samples = [
        _one_run(
            generator,
            vae,
            student_steps=student_steps,
            seed=20260721,
            latent_blocks=latent_blocks,
            telemetry_interval=telemetry_interval,
        )
        for _ in range(repeats)
    ]
    return {
        "schema": "liveframe.h100-pipeline-benchmark.v1",
        "student_steps": student_steps,
        "latent_blocks": latent_blocks,
        "pixel_frames": 4 * (latent_blocks - 1) + 1,
        "local_attn_frames": local_attn_frames,
        "sink_frames": sink_frames,
        "repeats": repeats,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tf32": False,
        "model_load_seconds": load_seconds,
        "warmup": warmup,
        "samples": samples,
        "summary": {
            "generated_pixel_fps_median": statistics.median(
                sample["generated_pixel_fps"] for sample in samples
            ),
            "ttff_seconds_median": statistics.median(
                sample["ttff_seconds"] for sample in samples
            ),
            "steady_block_seconds_median": statistics.median(
                sample["steady_block_mean_seconds"] for sample in samples
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--student-steps", type=int, choices=(1, 2), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--latent-blocks", type=int, default=360)
    parser.add_argument("--local-attn-frames", type=int, default=21)
    parser.add_argument("--sink-frames", type=int, default=1)
    parser.add_argument("--telemetry-interval", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(
        args.checkpoint,
        student_steps=args.student_steps,
        repeats=args.repeats,
        latent_blocks=args.latent_blocks,
        local_attn_frames=args.local_attn_frames,
        sink_frames=args.sink_frames,
        telemetry_interval=args.telemetry_interval,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
