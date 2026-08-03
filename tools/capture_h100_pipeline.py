"""Capture a complete deterministic CF++ 1-step or 2-step pipeline on an H100.

Run this script from the root of the pinned Causal-Forcing checkout. The
fixture contains exact serialized inputs, all 23 scheduler-noise draws, every
denoising and refresh result, sparse clean-cache probes, and final latents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

FIXTURE_SCHEMA = "liveframe.golden-fixture.v1"
SOURCE_REVISION = "8db419e341e5fc52542c0b2c4542728420ddfb4a"
MODEL_SHA256 = {
    1: "bdb1b475fc88d528f510158a0990cd457f02c661b527ceb11ca9e4728533e2d0",
    2: "8f899a016dffb9539788fac82c6200e43ef6e23b6e61ec6b885bb507a760ce42",
}
CONFIGURED_STEPS = {1: (1000,), 2: (1000, 500)}
FIRST_BLOCK_CONFIGURED_STEPS = (1000, 750, 500, 250)
LATENT_BLOCKS = 21
TOKENS_PER_FRAME = 1560
SELECTED_CACHE_LAYERS = (0, 15, 29)
CACHE_PROBE_OFFSETS = (0, TOKENS_PER_FRAME // 2, TOKENS_PER_FRAME - 1)


class Capture:
    """Write raw little-endian tensors and their portable manifest entries."""

    def __init__(self, output: Path, fixture_id: str, model_artifact_id: str) -> None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"capture output must be empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.fixture_id = fixture_id
        self.model_artifact_id = model_artifact_id
        self.entries: list[dict[str, Any]] = []

    def tensor(
        self,
        name: str,
        role: str,
        value: torch.Tensor,
        *,
        scheduler_transition: str | None = None,
    ) -> None:
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            payload = tensor.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
            dtype_name = "bfloat16"
        elif tensor.dtype == torch.float16:
            payload = tensor.numpy().astype("<f2", copy=False).tobytes()
            dtype_name = "float16"
        elif tensor.dtype == torch.float32:
            payload = tensor.numpy().astype("<f4", copy=False).tobytes()
            dtype_name = "float32"
        elif tensor.dtype == torch.int64:
            payload = tensor.numpy().astype("<i8", copy=False).tobytes()
            dtype_name = "int64"
        else:
            raise TypeError(f"unsupported capture dtype for {name}: {tensor.dtype}")
        filename = f"{name}.bin"
        (self.output / filename).write_bytes(payload)
        self.entries.append(
            {
                "name": name,
                "role": role,
                "file": filename,
                "shape": list(tensor.shape),
                "dtype": dtype_name,
                "byte_order": "little",
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "scheduler_transition": scheduler_transition,
            }
        )

    def progress(self, payload: dict[str, Any]) -> None:
        (self.output / "progress.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def finish(self, metadata: dict[str, Any]) -> None:
        manifest = {
            "schema": FIXTURE_SCHEMA,
            "fixture_id": self.fixture_id,
            "source_runtime": "pytorch-cuda-h100",
            "source_revision": SOURCE_REVISION,
            "model_artifact_id": self.model_artifact_id,
            "metadata": metadata,
            "tensors": self.entries,
        }
        (self.output / "fixture.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (self.output / "progress.json").unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_generator(
    checkpoint: Path, *, local_attn_size: int = -1, sink_size: int = 0
) -> Any:
    sys.path.insert(0, str(Path.cwd()))
    from utils.wan_wrapper import WanDiffusionWrapper  # type: ignore[import-not-found]

    generator = WanDiffusionWrapper(
        model_name="Wan2.1-T2V-1.3B",
        timestep_shift=5.0,
        is_causal=True,
        local_attn_size=local_attn_size,
        sink_size=sink_size,
    )
    envelope = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = envelope["generator_ema"]
    normalized = {}
    for name, tensor in state.items():
        if name.startswith("model._fsdp_wrapped_module."):
            name = name.replace("model._fsdp_wrapped_module.", "model.", 1)
        normalized[name] = tensor
    generator.load_state_dict(normalized, strict=True)
    return generator.to(device="cuda", dtype=torch.bfloat16).eval()


def _make_caches(generator: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocks = len(generator.model.blocks)
    local_attn_size = int(generator.model.local_attn_size)
    capacity = 32760 if local_attn_size == -1 else local_attn_size * TOKENS_PER_FRAME
    kv_cache = [
        {
            "k": torch.zeros((1, capacity, 12, 128), device="cuda", dtype=torch.bfloat16),
            "v": torch.zeros((1, capacity, 12, 128), device="cuda", dtype=torch.bfloat16),
            "global_end_index": torch.tensor([0], device="cuda", dtype=torch.long),
            "local_end_index": torch.tensor([0], device="cuda", dtype=torch.long),
        }
        for _ in range(blocks)
    ]
    cross_cache = [
        {
            "k": torch.zeros((1, 512, 12, 128), device="cuda", dtype=torch.bfloat16),
            "v": torch.zeros((1, 512, 12, 128), device="cuda", dtype=torch.bfloat16),
            "is_init": False,
        }
        for _ in range(blocks)
    ]
    return kv_cache, cross_cache


def _warped_steps(generator: Any, configured: tuple[int, ...]) -> tuple[float, ...]:
    timeline = torch.cat(
        (generator.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
    )
    return tuple(float(timeline[1000 - step].item()) for step in configured)


def _capture_cache_state(
    capture: Capture,
    kv_cache: list[dict[str, Any]],
    block_index: int,
    *,
    local_attention: bool,
) -> list[int]:
    prefix = f"block.{block_index:02d}.refresh"
    global_ends = torch.stack([cache["global_end_index"][0] for cache in kv_cache])
    local_ends = torch.stack([cache["local_end_index"][0] for cache in kv_cache])
    capture.tensor(f"{prefix}.cache_global_end", "cache_state", global_ends)
    capture.tensor(f"{prefix}.cache_local_end", "cache_state", local_ends)

    local_end = int(local_ends[0].item())
    if local_attention:
        probe_indices = [0, local_end // 2, local_end - 1]
    else:
        token_start = block_index * TOKENS_PER_FRAME
        probe_indices = [token_start + offset for offset in CACHE_PROBE_OFFSETS]
    offsets = torch.tensor(probe_indices, device="cuda")
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
    return probe_indices


def capture_pipeline(
    checkpoint: Path,
    output: Path,
    *,
    student_steps: int,
    latent_blocks: int = LATENT_BLOCKS,
    local_attn_frames: int = -1,
    sink_frames: int = 0,
) -> None:
    if latent_blocks < 1:
        raise ValueError("latent_blocks must be positive")
    if local_attn_frames != -1 and local_attn_frames < 2:
        raise ValueError("local_attn_frames must be -1 or at least 2")
    if sink_frames < 0 or (local_attn_frames != -1 and sink_frames >= local_attn_frames):
        raise ValueError("sink_frames must fit inside the local attention window")
    expected_sha256 = MODEL_SHA256[student_steps]
    if _sha256(checkpoint) != expected_sha256:
        raise ValueError(
            f"checkpoint SHA-256 does not match the pinned CF++ {student_steps}-step artifact"
        )

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    started = time.time()
    generator = _load_generator(
        checkpoint, local_attn_size=local_attn_frames, sink_size=sink_frames
    )
    if local_attn_frames == -1 and sink_frames == 0 and latent_blocks == LATENT_BLOCKS:
        fixture_id = f"cfpp-{student_steps}step-native-21block-h100"
    else:
        fixture_id = (
            f"cfpp-{student_steps}step-native-{latent_blocks}block-"
            f"local{local_attn_frames}-sink{sink_frames}-h100"
        )
    model_artifact_id = f"causal-forcing-plus-plus-framewise-{student_steps}step"
    capture = Capture(output, fixture_id, model_artifact_id)
    kv_cache, cross_cache = _make_caches(generator)

    # Construct every stochastic input on CPU, serialize it, and consume that exact
    # tensor on CUDA. Cross-runtime parity never relies on seed equivalence.
    cpu_rng = torch.Generator(device="cpu").manual_seed(20260721)
    initial_noise = torch.randn(
        (1, latent_blocks, 16, 60, 104), generator=cpu_rng, dtype=torch.float32
    ).to(torch.bfloat16)
    prompt_embedding = torch.linspace(
        -0.25, 0.25, 512 * 4096, dtype=torch.float32
    ).reshape(1, 512, 4096).to(torch.bfloat16)
    capture.tensor("initial_noise", "initial_noise", initial_noise)
    capture.tensor("prompt_embedding", "conditioning", prompt_embedding)

    first_steps = _warped_steps(generator, FIRST_BLOCK_CONFIGURED_STEPS)
    later_configured_steps = CONFIGURED_STEPS[student_steps]
    later_steps = _warped_steps(generator, later_configured_steps)
    schedule_rows: list[dict[str, Any]] = []
    output_latents = torch.empty_like(initial_noise, device="cuda")
    conditional = {"prompt_embeds": prompt_embedding.cuda()}

    denoising_forwards = 0
    refresh_forwards = 0
    noise_draws = 0
    cache_probe_indices: list[list[int]] = []
    try:
        for block_index in range(latent_blocks):
            steps = first_steps if block_index == 0 else later_steps
            configured_steps = (
                FIRST_BLOCK_CONFIGURED_STEPS
                if block_index == 0
                else later_configured_steps
            )
            noisy_input = initial_noise[:, block_index : block_index + 1].cuda()
            current_start = block_index * TOKENS_PER_FRAME

            for step_index, (configured_step, warped_step) in enumerate(
                zip(configured_steps, steps, strict=True)
            ):
                prefix = f"block.{block_index:02d}.step.{step_index:02d}"
                timestep = torch.full((1, 1), warped_step, device="cuda", dtype=torch.float32)
                capture.tensor(f"{prefix}.model_input", "model_input", noisy_input)
                capture.tensor(f"{prefix}.timestep", "scheduler_state", timestep)
                flow, prediction = generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional,
                    timestep=timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=cross_cache,
                    current_start=current_start,
                )
                torch.cuda.synchronize()
                capture.tensor(f"{prefix}.flow_prediction", "model_output", flow)
                capture.tensor(f"{prefix}.x0_prediction", "model_output", prediction)
                denoising_forwards += 1
                schedule_rows.append(
                    {
                        "kind": "denoising",
                        "block": block_index,
                        "step": step_index,
                        "configured_timestep": configured_step,
                        "warped_timestep": warped_step,
                    }
                )

                if step_index < len(steps) - 1:
                    transition = (
                        f"block-{block_index:04d}.renoise-{step_index:02d}-to-"
                        f"{step_index + 1:02d}-of-{len(steps):02d}"
                    )
                    scheduler_noise = torch.randn(
                        prediction.shape, generator=cpu_rng, dtype=torch.float32
                    ).to(torch.bfloat16)
                    capture.tensor(
                        f"{prefix}.scheduler_noise",
                        "scheduler_noise",
                        scheduler_noise,
                        scheduler_transition=transition,
                    )
                    target = torch.full(
                        (prediction.shape[0] * prediction.shape[1],),
                        steps[step_index + 1],
                        device="cuda",
                        dtype=torch.float32,
                    )
                    noisy_input = generator.scheduler.add_noise(
                        prediction.flatten(0, 1), scheduler_noise.cuda().flatten(0, 1), target
                    ).unflatten(0, prediction.shape[:2])
                    capture.tensor(
                        f"{prefix}.renoised_output", "scheduler_output", noisy_input
                    )
                    noise_draws += 1

            output_latents[:, block_index : block_index + 1] = prediction
            refresh_prefix = f"block.{block_index:02d}.refresh"
            context_timestep = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
            refresh_flow, refresh_prediction = generator(
                noisy_image_or_video=prediction,
                conditional_dict=conditional,
                timestep=context_timestep,
                kv_cache=kv_cache,
                crossattn_cache=cross_cache,
                current_start=current_start,
            )
            torch.cuda.synchronize()
            capture.tensor(
                f"{refresh_prefix}.flow_prediction", "refresh_output", refresh_flow
            )
            capture.tensor(
                f"{refresh_prefix}.x0_prediction", "refresh_output", refresh_prediction
            )
            cache_probe_indices.append(
                _capture_cache_state(
                    capture,
                    kv_cache,
                    block_index,
                    local_attention=local_attn_frames != -1,
                )
            )
            refresh_forwards += 1
            schedule_rows.append(
                {
                    "kind": "refresh",
                    "block": block_index,
                    "step": None,
                    "configured_timestep": 0,
                    "warped_timestep": 0.0,
                }
            )
            capture.progress(
                {
                    "fixture_id": fixture_id,
                    "completed_blocks": block_index + 1,
                    "latent_blocks": latent_blocks,
                    "denoising_forwards": denoising_forwards,
                    "refresh_forwards": refresh_forwards,
                    "scheduler_noise_draws": noise_draws,
                    "elapsed_seconds": time.time() - started,
                }
            )
            print(
                f"captured block {block_index + 1}/{latent_blocks}: "
                f"{denoising_forwards + refresh_forwards} forwards",
                flush=True,
            )

        capture.tensor("output_latents", "pipeline_output", output_latents)
        torch.cuda.synchronize()
        elapsed = time.time() - started
        capture.finish(
            {
                "checkpoint_sha256": "sha256:" + expected_sha256,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
                "gpu": torch.cuda.get_device_name(0),
                "python": platform.python_version(),
                "native_latent_shape": [1, latent_blocks, 16, 60, 104],
                "tokens_per_frame": TOKENS_PER_FRAME,
                "configured_first_block_steps": list(FIRST_BLOCK_CONFIGURED_STEPS),
                "configured_later_block_steps": list(later_configured_steps),
                "student_steps": student_steps,
                "warped_first_block_steps": list(first_steps),
                "warped_later_block_steps": list(later_steps),
                "latent_blocks": latent_blocks,
                "context_noise": 0,
                "local_attention": local_attn_frames != -1,
                "local_attn_frames": (
                    LATENT_BLOCKS if local_attn_frames == -1 else local_attn_frames
                ),
                "sink_frames": sink_frames,
                "cache_probe_indices": cache_probe_indices,
                "denoising_forwards": denoising_forwards,
                "refresh_forwards": refresh_forwards,
                "total_transformer_forwards": denoising_forwards + refresh_forwards,
                "scheduler_noise_draws": noise_draws,
                "selected_cache_layers": list(SELECTED_CACHE_LAYERS),
                "cache_probe_offsets": list(CACHE_PROBE_OFFSETS),
                "schedule": schedule_rows,
                "elapsed_seconds": elapsed,
                "tf32": False,
                "nvidia_smi": subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    text=True,
                ).strip(),
            }
        )
        print(
            json.dumps(
                {
                    "fixture": str(output),
                    "tensors": len(capture.entries),
                    "forwards": denoising_forwards + refresh_forwards,
                    "noise_draws": noise_draws,
                    "elapsed_seconds": elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception:
        capture.progress(
            {
                "fixture_id": fixture_id,
                "failed": True,
                "denoising_forwards": denoising_forwards,
                "refresh_forwards": refresh_forwards,
                "scheduler_noise_draws": noise_draws,
                "elapsed_seconds": time.time() - started,
            }
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--student-steps", type=int, choices=(1, 2), default=2)
    parser.add_argument("--latent-blocks", type=int, default=LATENT_BLOCKS)
    parser.add_argument("--local-attn-frames", type=int, default=-1)
    parser.add_argument("--sink-frames", type=int, default=0)
    args = parser.parse_args()
    capture_pipeline(
        args.checkpoint,
        args.output,
        student_steps=args.student_steps,
        latent_blocks=args.latent_blocks,
        local_attn_frames=args.local_attn_frames,
        sink_frames=args.sink_frames,
    )


if __name__ == "__main__":
    main()
