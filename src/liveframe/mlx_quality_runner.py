"""Prompt-conditioned MLX runner for sealed Rolling Forcing holdout jobs."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from liveframe.perceptual_policy import canonical_digest
from liveframe.quality_artifacts import StochasticWriter, atomic_json
from liveframe.quality_capture import sha256_file
from liveframe.quality_executor import RunnerOutput
from liveframe.quality_freeze import (
    CHECKPOINT_DIGEST,
    POLICY_BUNDLE_V2_DIGEST,
)
from liveframe.quality_metric_runner import PerceptualMetricRunner
from liveframe.quality_stochastic import (
    STOCHASTIC_CONTRACT,
    StochasticPlanReader,
    plan_directory,
    transition_name,
)
from liveframe.rolling_forcing import RollingForcingSchedule

HOLDOUT_SEEDS = frozenset({81131, 90001, 101111, 120011})
HOLDOUT_LATENT_FRAMES = frozenset({60, 243})
MLX_MODEL_DIGEST = (
    "sha256:1f15d166a8a60bed70c2a8ab2923f40bd2b6b53ed2e0be87905bf45d0666d971"
)
MLX_T5_DIGEST = (
    "sha256:e86ee4199903e00a88dcd43583a43a6eb898cef600e38670f222d7e37d163787"
)
MLX_VAE_DIGEST = (
    "sha256:977530e453dbfabbab31e2972e1577d8d7e2840ba7410c81aa3fd421c0cd7414"
)
TOKENIZER_REPOSITORY = "google/umt5-xxl"
TOKENIZER_REVISION = "66cb9e7e85526fe440a945569e42c72fb6cbc0ad"
WARPED_STEPS = {
    1000: 1000.0,
    750: 937.5,
    500: 833.3333129882812,
    250: 625.0,
}
TOKENS_PER_FRAME = 1560
BLOCK_FRAMES = 3


def mlx_runtime_provenance(mx: Any) -> dict[str, Any]:
    device = mx.device_info() if mx.metal.is_available() else {}
    if (
        not mx.metal.is_available()
        or device.get("device_name") != "Apple M5 Max"
        or device.get("architecture") != "applegpu_g17s"
    ):
        raise ValueError("MLX holdout capture requires Apple M5 Max Metal")
    return {
        "backend": "mlx-metal-m5-max",
        "mlx_version": mx.__version__,
        "mlx_metal_version": importlib.metadata.version("mlx-metal"),
        "metal_available": True,
        "default_device": str(mx.default_device()),
        "device": device,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": platform.python_version(),
    }


def validate_candidate_job(job: Mapping[str, Any]) -> None:
    if (
        job.get("split") != "holdout"
        or job.get("role") != "candidate"
        or job.get("backend") != "mlx"
    ):
        raise ValueError("MLX quality runner only accepts holdout candidate jobs")
    if job.get("seed") not in HOLDOUT_SEEDS:
        raise ValueError("MLX quality runner seed is not in the frozen holdout set")
    latent_frames = int(job["latent_frames"])
    if latent_frames not in HOLDOUT_LATENT_FRAMES:
        raise ValueError("MLX quality runner latent horizon is not frozen")
    schedule = RollingForcingSchedule(latent_frames=latent_frames)
    if int(job["pixel_frames"]) != 4 * (schedule.latent_frames - 1) + 1:
        raise ValueError("MLX quality job pixel horizon is inconsistent")


def validate_mlx_assets(model_dir: Path, bundle: Mapping[str, Any]) -> None:
    if bundle.get("bundle_digest") != POLICY_BUNDLE_V2_DIGEST:
        raise ValueError("MLX quality runner requires the frozen v2 policy bundle")
    unsigned = {key: value for key, value in bundle.items() if key != "bundle_digest"}
    if canonical_digest(unsigned) != POLICY_BUNDLE_V2_DIGEST:
        raise ValueError("MLX quality runner policy bundle payload mismatch")
    conversion_path = model_dir / "liveframe-conversion.json"
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    if not isinstance(conversion, dict):
        raise ValueError("MLX conversion manifest must be an object")
    expected = {
        "source_sha256": CHECKPOINT_DIGEST,
        "model_sha256": MLX_MODEL_DIGEST,
    }
    for name, value in expected.items():
        if conversion.get(name) != value:
            raise ValueError(f"MLX conversion {name} mismatch")
    stock = conversion.get("stock_assets")
    if not isinstance(stock, dict):
        raise ValueError("MLX conversion manifest lacks stock assets")
    expected_files = {
        "model.safetensors": MLX_MODEL_DIGEST,
        "t5_encoder.safetensors": MLX_T5_DIGEST,
        "vae.safetensors": MLX_VAE_DIGEST,
    }
    for name, digest in expected_files.items():
        path = model_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"MLX quality asset checksum mismatch: {name}")
    if stock.get("t5_encoder.safetensors") != MLX_T5_DIGEST:
        raise ValueError("MLX conversion T5 ancestry mismatch")
    if stock.get("vae.safetensors") != MLX_VAE_DIGEST:
        raise ValueError("MLX conversion VAE ancestry mismatch")


def _write_mlx_tensor(
    path: Path, value: Any, *, layout: str, schema: str
) -> list[Path]:
    import mlx.core as mx  # type: ignore[import-not-found]

    mx.eval(value)
    if value.dtype == mx.bfloat16:
        payload = np.array(value.view(mx.uint16), dtype="<u2").tobytes(order="C")
        dtype = "bfloat16"
    elif value.dtype == mx.uint8:
        payload = np.array(value, dtype=np.uint8).tobytes(order="C")
        dtype = "uint8"
    else:
        raise TypeError(f"unsupported MLX quality artifact dtype: {value.dtype}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    metadata_path = path.with_suffix(path.suffix + ".json")
    atomic_json(
        metadata_path,
        {
            "schema": schema,
            "dtype": dtype,
            "layout": layout,
            "shape": list(value.shape),
            "sha256": sha256_file(path),
        },
    )
    return [path, metadata_path]


class MLXQualityRunner:
    """Keep Rolling Forcing, VAE, and metric weights resident across candidate jobs."""

    def __init__(
        self,
        *,
        model_dir: Path,
        bundle: Mapping[str, Any],
        metric_provenance: dict[str, Any],
        metric_batch_size: int,
        noise_root: Path,
        metric_device: str = "mps",
    ) -> None:
        import mlx.core as mx
        from mlx_video.models.wan_2.config import WanModelConfig  # type: ignore[import-untyped]
        from mlx_video.models.wan_2.utils import (  # type: ignore[import-untyped]
            load_vae_decoder,
            load_wan_model,
        )

        validate_mlx_assets(model_dir, bundle)
        runtime = mlx_runtime_provenance(mx)
        raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        allowed = WanModelConfig.__dataclass_fields__.keys()
        self.config = WanModelConfig(
            **{key: value for key, value in raw.items() if key in allowed}
        )
        self.model = load_wan_model(model_dir / "model.safetensors", self.config)
        self.vae = load_vae_decoder(model_dir / "vae.safetensors", self.config)
        self.metrics = PerceptualMetricRunner(
            metric_provenance, device=metric_device, batch_size=metric_batch_size
        )
        self.model_dir = model_dir
        self.noise_root = noise_root
        self.mx = mx
        self.runtime = {
            **runtime,
            "checkpoint_sha256": CHECKPOINT_DIGEST,
            "model_sha256": MLX_MODEL_DIGEST,
            "t5_sha256": MLX_T5_DIGEST,
            "vae_sha256": MLX_VAE_DIGEST,
            "policy_bundle_digest": POLICY_BUNDLE_V2_DIGEST,
            "tokenizer_repository": TOKENIZER_REPOSITORY,
            "tokenizer_revision": TOKENIZER_REVISION,
        }

    @staticmethod
    def _from_torch_bf16(value: Any) -> Any:
        import mlx.core as mx

        raw = value.detach().cpu().contiguous().view(__import__("torch").uint16).numpy()
        return mx.array(raw.astype("<u2", copy=False)).view(mx.bfloat16).reshape(value.shape)

    def _encode_prompt(self, prompt: str) -> tuple[Any, Any, float]:
        from mlx_video.models.wan_2.utils import (
            encode_text,
            load_t5_encoder,
        )
        from transformers import AutoTokenizer

        started = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_REPOSITORY,
            revision=TOKENIZER_REVISION,
            local_files_only=True,
        )
        encoder = load_t5_encoder(
            self.model_dir / "t5_encoder.safetensors", self.config
        )
        prompt_embedding = encode_text(
            encoder, tokenizer, prompt, self.config.text_len
        )
        embedded_context = self.model.embed_text([prompt_embedding])
        cross_kv = self.model.prepare_cross_kv(embedded_context)
        self.mx.eval(embedded_context, cross_kv)
        elapsed = time.monotonic() - started
        del encoder, tokenizer, prompt_embedding
        self.mx.clear_cache()
        return embedded_context, cross_kv, elapsed

    def _generate(
        self,
        job: Mapping[str, Any],
        embedded_context: Any,
        cross_kv: Any,
        stochastic: StochasticWriter,
        noise_plan: StochasticPlanReader,
    ) -> tuple[Any, dict[str, float]]:
        from liveframe.backends.mlx_causal import CausalWanForward

        schedule = RollingForcingSchedule(latent_frames=int(job["latent_frames"]))
        initial_shape = (1, schedule.latent_frames, 16, 60, 104)
        initial_torch = noise_plan.next_torch(
            "initial_noise", initial_shape, device="cpu"
        )
        stochastic.append("initial_noise", initial_torch)
        initial = self._from_torch_bf16(initial_torch).transpose(0, 2, 1, 3, 4)[0]
        initial_blocks = [
            initial[:, index : index + BLOCK_FRAMES]
            for index in range(0, schedule.latent_frames, BLOCK_FRAMES)
        ]
        noisy_blocks = [self.mx.zeros_like(row) for row in initial_blocks]
        generated: list[Any | None] = [None] * schedule.block_count
        causal = CausalWanForward(self.model, rolling_forcing=True)
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
            model_input = self.mx.concatenate(blocks, axis=1).astype(self.mx.bfloat16)
            timestep = self.mx.array(
                [[WARPED_STEPS[step] for step in window.configured_timesteps for _ in range(3)]],
                dtype=self.mx.float32,
            )
            started = time.monotonic()
            flow = causal(
                [model_input],
                timestep=timestep,
                embedded_context=embedded_context,
                cross_kv_caches=cross_kv,
                current_start=window.start_frame * TOKENS_PER_FRAME,
            )[0]
            sigma = (timestep[0] / 1000.0).reshape(1, -1, 1, 1)
            prediction = model_input.astype(self.mx.float32) - sigma * flow
            self.mx.eval(prediction)
            denoise_seconds += time.monotonic() - started
            started = time.monotonic()
            for offset, configured_step in enumerate(window.configured_timesteps):
                block_id = window.start_block + offset
                block_prediction = prediction[:, offset * 3 : (offset + 1) * 3]
                if configured_step == schedule.configured_steps[-1]:
                    generated[block_id] = block_prediction.astype(self.mx.bfloat16)
                    continue
                next_step = schedule.configured_steps[
                    schedule.configured_steps.index(configured_step) + 1
                ]
                transition = transition_name(
                    window.index, block_id, configured_step, next_step
                )
                noise_torch = noise_plan.next_torch(
                    transition, (1, BLOCK_FRAMES, 16, 60, 104), device="cpu"
                )
                stochastic.append(transition, noise_torch)
                noise = self._from_torch_bf16(noise_torch).transpose(0, 2, 1, 3, 4)[0]
                next_sigma = WARPED_STEPS[next_step] / 1000.0
                noisy_blocks[block_id] = (
                    (1 - next_sigma) * block_prediction.astype(self.mx.float32)
                    + next_sigma * noise.astype(self.mx.float32)
                ).astype(self.mx.bfloat16)
                self.mx.eval(noisy_blocks[block_id])
            scheduler_seconds += time.monotonic() - started
            started = time.monotonic()
            refresh = causal(
                [prediction[:, :3].astype(self.mx.bfloat16)],
                timestep=self.mx.zeros((1, 3)),
                embedded_context=embedded_context,
                cross_kv_caches=cross_kv,
                current_start=window.start_frame * TOKENS_PER_FRAME,
                updating_cache=True,
            )[0]
            self.mx.eval(refresh)
            refresh_seconds += time.monotonic() - started
        if any(row is None for row in generated):
            raise RuntimeError("rolling schedule did not finalize every latent block")
        noise_plan.require_consumed()
        output = self.mx.concatenate(
            [row for row in generated if row is not None], axis=1
        )[None].transpose(0, 2, 1, 3, 4)
        self.mx.eval(output)
        return output, {
            "diffusion_seconds": denoise_seconds,
            "clean_cache_refresh_seconds": refresh_seconds,
            "scheduler_seconds": scheduler_seconds,
        }

    def _decode(self, latents: Any, output: Path) -> tuple[list[Path], float]:
        from liveframe.mlx_vae_decode import (
            _causal_decode_frame,
            _decoder_cache_slots,
        )

        cache: list[Any] = [None] * _decoder_cache_slots(self.vae.decoder)
        temporary = output.with_name(f".{output.name}.tmp")
        started = time.monotonic()
        emitted = 0
        with temporary.open("wb") as stream:
            for frame in range(latents.shape[1]):
                decoded = _causal_decode_frame(
                    self.vae, latents[:, frame : frame + 1].transpose(0, 2, 1, 3, 4), cache
                )
                self.mx.eval(decoded)
                pixels = np.array(decoded[0].transpose(1, 2, 3, 0), dtype=np.float32)
                pixels = np.clip((pixels + 1.0) * 127.5, 0, 255).astype(np.uint8)
                stream.write(pixels.tobytes(order="C"))
                emitted += int(pixels.shape[0])
        os.replace(temporary, output)
        metadata_path = output.with_suffix(output.suffix + ".json")
        atomic_json(
            metadata_path,
            {
                "schema": "liveframe.decoded-pixels.v1",
                "dtype": "uint8",
                "layout": "T,H,W,C",
                "shape": [emitted, 480, 832, 3],
                "sha256": sha256_file(output),
            },
        )
        return [output, metadata_path], time.monotonic() - started

    def run(self, job: Mapping[str, Any], output_dir: Path) -> RunnerOutput:
        validate_candidate_job(job)
        stochastic = StochasticWriter(output_dir)
        latent_frames = int(job["latent_frames"])
        seed = int(job["seed"])
        noise_plan = StochasticPlanReader(
            plan_directory(
                self.noise_root, seed=seed, latent_frames=latent_frames
            )
            / "stochastic.bf16.json",
            seed=seed,
            latent_frames=latent_frames,
        )
        try:
            embedded_context, cross_kv, text_seconds = self._encode_prompt(
                str(job["prompt"])
            )
            latents, generation = self._generate(
                job, embedded_context, cross_kv, stochastic, noise_plan
            )
            if latents.shape[1] != int(job["latent_frames"]):
                raise ValueError("generated latent frame count does not match frozen job")
            artifacts = _write_mlx_tensor(
                output_dir / "latents.bf16",
                latents,
                layout="B,T,C,H,W",
                schema="liveframe.output-latents.v1",
            )
            pixel_artifacts, vae_seconds = self._decode(
                latents, output_dir / "pixels.bin"
            )
            pixels_metadata = json.loads(
                (output_dir / "pixels.bin.json").read_text(encoding="utf-8")
            )
            if pixels_metadata["shape"][0] != int(job["pixel_frames"]):
                raise ValueError("decoded pixel frame count does not match frozen job")
            artifacts.extend(pixel_artifacts)
            artifacts.extend(
                stochastic.finish(
                    seed=seed,
                    contract=STOCHASTIC_CONTRACT,
                    latent_frames=latent_frames,
                )
            )
        except BaseException:
            stochastic.abort()
            raise
        metric_started = time.monotonic()
        metrics = self.metrics.measure(
            output_dir / "pixels.bin",
            prompt=str(job["prompt"]),
            expected_frames=int(job["pixel_frames"]),
        )
        return RunnerOutput(
            artifact_paths=artifacts,
            metrics=metrics,
            runtime=self.runtime,
            timing={
                "text_encoding_seconds": text_seconds,
                **generation,
                "vae_seconds": vae_seconds,
                "metric_seconds": time.monotonic() - metric_started,
            },
        )