"""Capture the released three-latent-block Rolling Forcing pipeline on H100.

Run with the pinned ``third_party/Causal-Forcing/long_video`` directory as the
working directory. This intentionally does not reuse the framewise CF++ loop.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from capture_h100_pipeline import Capture, _sha256

from liveframe.rolling_forcing import RollingForcingSchedule

CHECKPOINT_SHA256 = "1f123dce60e60424a9c6d949643176840fceffb947932997bcf1202d93656d98"
TOKENS_PER_FRAME = 1560
BLOCK_FRAMES = 3
BLOCK_TOKENS = BLOCK_FRAMES * TOKENS_PER_FRAME
CACHE_FRAMES = 24
ATTENTION_FRAMES = 21
SELECTED_CACHE_LAYERS = (0, 15, 29)


def _load_generator(checkpoint: Path) -> Any:
    sys.path.insert(0, str(Path.cwd()))
    from utils.wan_wrapper import WanDiffusionWrapper  # type: ignore[import-not-found]

    generator = WanDiffusionWrapper(
        model_name="Wan2.1-T2V-1.3B",
        timestep_shift=5.0,
        is_causal=True,
        local_attn_size=ATTENTION_FRAMES,
        sink_size=BLOCK_FRAMES,
    )
    envelope = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = envelope["generator_ema"]
    normalized = {
        name.replace("model._fsdp_wrapped_module.", "model.", 1): value
        for name, value in state.items()
    }
    generator.load_state_dict(normalized, strict=True)
    generator.model.num_frame_per_block = BLOCK_FRAMES
    return generator.to(device="cuda", dtype=torch.bfloat16).eval()


def _make_caches(generator: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    capacity = CACHE_FRAMES * TOKENS_PER_FRAME
    kv_cache = [
        {
            "k": torch.zeros((1, capacity, 12, 128), device="cuda", dtype=torch.bfloat16),
            "v": torch.zeros((1, capacity, 12, 128), device="cuda", dtype=torch.bfloat16),
            "global_end_index": torch.zeros(1, device="cuda", dtype=torch.long),
            "local_end_index": torch.zeros(1, device="cuda", dtype=torch.long),
        }
        for _ in generator.model.blocks
    ]
    cross_cache = [
        {
            "k": torch.zeros((1, 512, 12, 128), device="cuda", dtype=torch.bfloat16),
            "v": torch.zeros((1, 512, 12, 128), device="cuda", dtype=torch.bfloat16),
            "is_init": False,
        }
        for _ in generator.model.blocks
    ]
    return kv_cache, cross_cache


def _warped_steps(generator: Any, configured: tuple[int, ...]) -> dict[int, float]:
    timeline = torch.cat(
        (generator.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
    )
    return {step: float(timeline[1000 - step].item()) for step in configured}


def _capture_cache(
    capture: Capture, kv_cache: list[dict[str, Any]], window_index: int
) -> list[int]:
    prefix = f"window.{window_index:03d}.refresh"
    global_ends = torch.stack([row["global_end_index"][0] for row in kv_cache])
    local_ends = torch.stack([row["local_end_index"][0] for row in kv_cache])
    capture.tensor(f"{prefix}.cache_global_end", "cache_state", global_ends)
    capture.tensor(f"{prefix}.cache_local_end", "cache_state", local_ends)
    local_end = int(local_ends[0].item())
    probes = sorted({0, BLOCK_TOKENS - 1, local_end // 2, local_end - 1})
    offsets = torch.tensor(probes, device="cuda")
    for layer in SELECTED_CACHE_LAYERS:
        capture.tensor(
            f"{prefix}.layer.{layer:02d}.cache_key_probe",
            "cache_probe",
            kv_cache[layer]["k"].index_select(1, offsets),
        )
        capture.tensor(
            f"{prefix}.layer.{layer:02d}.cache_value_probe",
            "cache_probe",
            kv_cache[layer]["v"].index_select(1, offsets),
        )
    return probes


def capture_rolling(checkpoint: Path, output: Path, *, latent_frames: int) -> None:
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 does not match pinned chunkwise/longvideo.pt")
    schedule = RollingForcingSchedule(latent_frames=latent_frames, context_noise=0)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    generator = _load_generator(checkpoint)
    kv_cache, cross_cache = _make_caches(generator)
    fixture_id = f"rolling-forcing-native-{latent_frames}latent-local21-anchor3-h100"
    capture = Capture(output, fixture_id, "rolling-forcing-long-video")
    cpu_rng = torch.Generator(device="cpu").manual_seed(20260721)
    initial_noise = torch.randn(
        (1, latent_frames, 16, 60, 104), generator=cpu_rng, dtype=torch.float32
    ).to(torch.bfloat16)
    prompt = torch.linspace(-0.25, 0.25, 512 * 4096, dtype=torch.float32).reshape(
        1, 512, 4096
    ).to(torch.bfloat16)
    capture.tensor("initial_noise", "initial_noise", initial_noise)
    capture.tensor("prompt_embedding", "conditioning", prompt)
    noise = initial_noise.cuda()
    conditional = {"prompt_embeds": prompt.cuda()}
    output_latents = torch.zeros_like(noise)
    noisy_cache = torch.zeros_like(noise)
    warped = _warped_steps(generator, schedule.configured_steps)
    scheduler_noise_draws = 0
    cache_probe_indices = []
    started = time.time()

    for window in schedule.windows():
        start, end = window.start_frame, window.end_frame
        current_frames = end - start
        if current_frames == len(schedule.configured_steps) * BLOCK_FRAMES or start == 0:
            model_input = torch.cat(
                [noisy_cache[:, start : end - BLOCK_FRAMES], noise[:, end - BLOCK_FRAMES : end]],
                dim=1,
            )
        else:
            model_input = noisy_cache[:, start:end]
        timestep_values = [
            warped[step]
            for step in window.configured_timesteps
            for _ in range(BLOCK_FRAMES)
        ]
        timestep = torch.tensor([timestep_values], device="cuda", dtype=torch.float32)
        prefix = f"window.{window.index:03d}"
        capture.tensor(f"{prefix}.model_input", "model_input", model_input)
        capture.tensor(f"{prefix}.timestep", "scheduler_state", timestep)
        flow, prediction = generator(
            noisy_image_or_video=model_input,
            conditional_dict=conditional,
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=cross_cache,
            current_start=start * TOKENS_PER_FRAME,
        )
        torch.cuda.synchronize()
        capture.tensor(f"{prefix}.flow_prediction", "model_output", flow)
        capture.tensor(f"{prefix}.x0_prediction", "model_output", prediction)
        output_latents[:, start:end] = prediction

        for offset, configured_step in enumerate(window.configured_timesteps):
            if configured_step == schedule.configured_steps[-1]:
                continue
            next_step = schedule.configured_steps[
                schedule.configured_steps.index(configured_step) + 1
            ]
            block_prediction = prediction[:, offset * BLOCK_FRAMES : (offset + 1) * BLOCK_FRAMES]
            scheduler_noise = torch.randn(
                block_prediction.shape, generator=cpu_rng, dtype=torch.float32
            ).to(torch.bfloat16)
            transition = (
                f"window-{window.index:04d}-block-"
                f"{window.start_block + offset:04d}-{configured_step}-to-{next_step}"
            )
            capture.tensor(
                f"{prefix}.block.{offset:02d}.scheduler_noise",
                "scheduler_noise",
                scheduler_noise,
                scheduler_transition=transition,
            )
            target = torch.full(
                (BLOCK_FRAMES,), warped[next_step], device="cuda", dtype=torch.float32
            )
            block_start = (window.start_block + offset) * BLOCK_FRAMES
            block_end = block_start + BLOCK_FRAMES
            noisy_cache[:, block_start:block_end] = generator.scheduler.add_noise(
                block_prediction.flatten(0, 1),
                scheduler_noise.cuda().flatten(0, 1),
                target,
            ).unflatten(0, block_prediction.shape[:2])
            scheduler_noise_draws += 1

        refresh_input = prediction[:, :BLOCK_FRAMES]
        refresh_timestep = torch.zeros((1, BLOCK_FRAMES), device="cuda", dtype=torch.float32)
        refresh_flow, refresh_prediction = generator(
            noisy_image_or_video=refresh_input,
            conditional_dict=conditional,
            timestep=refresh_timestep,
            kv_cache=kv_cache,
            crossattn_cache=cross_cache,
            current_start=start * TOKENS_PER_FRAME,
            updating_cache=True,
        )
        capture.tensor(f"{prefix}.refresh.flow_prediction", "refresh_output", refresh_flow)
        capture.tensor(f"{prefix}.refresh.x0_prediction", "refresh_output", refresh_prediction)
        cache_probe_indices.append(_capture_cache(capture, kv_cache, window.index))
        capture.progress(
            {
                "fixture_id": fixture_id,
                "completed_windows": window.index + 1,
                "window_count": schedule.window_count,
                "elapsed_seconds": time.time() - started,
            }
        )
        print(f"captured rolling window {window.index + 1}/{schedule.window_count}", flush=True)

    capture.tensor("output_latents", "pipeline_output", output_latents)
    elapsed = time.time() - started
    capture.finish(
        {
            "algorithm": "rolling_forcing_long_video",
            "checkpoint_sha256": "sha256:" + CHECKPOINT_SHA256,
            "latent_frames": latent_frames,
            "pixel_frames": 4 * (latent_frames - 1) + 1,
            "block_frames": BLOCK_FRAMES,
            "configured_steps": list(schedule.configured_steps),
            "warped_steps": [warped[step] for step in schedule.configured_steps],
            "context_noise": 0,
            "local_attention": True,
            "local_attn_frames": ATTENTION_FRAMES,
            "cache_frames": CACHE_FRAMES,
            "anchor_frames": BLOCK_FRAMES,
            "window_count": schedule.window_count,
            "denoising_forwards": schedule.window_count,
            "refresh_forwards": schedule.window_count,
            "total_transformer_forwards": schedule.transformer_forward_count,
            "scheduler_noise_draws": scheduler_noise_draws,
            "cache_probe_indices": cache_probe_indices,
            "selected_cache_layers": list(SELECTED_CACHE_LAYERS),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "tf32": False,
            "elapsed_seconds": elapsed,
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip(),
        }
    )
    print(json.dumps({"fixture": str(output), "elapsed_seconds": elapsed}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-frames", type=int, default=60)
    args = parser.parse_args()
    capture_rolling(args.checkpoint, args.output, latent_frames=args.latent_frames)


if __name__ == "__main__":
    main()
