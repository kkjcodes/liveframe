"""Execute a frozen prompt-conditioned quality batch on one H100.

This script loads the Rolling Forcing pipeline and perceptual models once, then
runs every job in the validated batch with serialized stochastic tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.quality_artifacts import StochasticWriter, write_tensor  # noqa: E402
from liveframe.quality_calibration import load_protocol  # noqa: E402
from liveframe.quality_capture import (  # noqa: E402
    load_batch_plan,
    load_holdout_reference_batch_plan,
    load_run_manifest,
    sha256_file,
)
from liveframe.quality_executor import RunnerOutput, execute_batch  # noqa: E402
from liveframe.quality_freeze import (  # noqa: E402
    CHECKPOINT_DIGEST,
    require_frozen_h100_reference_batch,
    require_frozen_identities,
)
from liveframe.quality_metric_runner import (  # noqa: E402
    PerceptualMetricRunner,
    validate_metric_artifact,
)
from liveframe.quality_stochastic import (  # noqa: E402
    STOCHASTIC_CONTRACT,
    StochasticPlanReader,
    plan_directory,
    transition_name,
)

SOURCE_REVISION = "8db419e341e5fc52542c0b2c4542728420ddfb4a"


class H100QualityRunner:
    def __init__(
        self,
        *,
        checkpoint: Path,
        config_path: Path,
        wan_model_root: Path,
        metric_provenance: dict[str, Any],
        metric_batch_size: int,
        noise_root: Path | None = None,
    ) -> None:
        import torch
        from omegaconf import OmegaConf

        if not torch.cuda.is_available():
            raise RuntimeError("H100 quality runner requires CUDA")
        if sha256_file(checkpoint) != CHECKPOINT_DIGEST:
            raise ValueError("rolling checkpoint checksum mismatch")
        required_model_files = {
            "config.json",
            "diffusion_pytorch_model.safetensors",
            "Wan2.1_VAE.pth",
            "models_t5_umt5-xxl-enc-bf16.pth",
        }
        missing = sorted(
            name for name in required_model_files if not (wan_model_root / name).is_file()
        )
        if missing:
            raise ValueError(f"Wan model root is incomplete: {missing}")
        os.environ["WAN_MODEL_ROOT"] = str(wan_model_root.resolve())
        long_video = REPOSITORY_ROOT / "third_party" / "Causal-Forcing" / "long_video"
        sys.path.insert(0, str(long_video))
        from pipeline import CausalInferencePipeline

        default_config = OmegaConf.load(long_video / "configs" / "default_config.yaml")
        config = OmegaConf.merge(default_config, OmegaConf.load(config_path))
        if int(config.num_frame_per_block) != 3 or int(config.context_noise) != 0:
            raise ValueError("quality capture requires three-frame blocks and context_noise=0")
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.pipeline = CausalInferencePipeline(config, device=torch.device("cuda"))
        envelope = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        state = envelope["generator_ema"]
        normalized: OrderedDict[str, Any] = OrderedDict()
        for name, tensor in state.items():
            name = name.replace("model._fsdp_wrapped_module.", "model.", 1)
            name = name.replace("_fsdp_wrapped_module.", "", 1)
            normalized[name] = tensor
        self.pipeline.generator.load_state_dict(normalized, strict=True)
        self.pipeline = self.pipeline.to(device="cuda", dtype=torch.bfloat16).eval()
        self.metrics = PerceptualMetricRunner(
            metric_provenance, device="cuda", batch_size=metric_batch_size
        )
        self.checkpoint = checkpoint
        self.config_path = config_path
        self.torch = torch
        self.noise_root = noise_root
        self.runtime = {
            "backend": "pytorch-cuda-h100",
            "source_revision": SOURCE_REVISION,
            "checkpoint_sha256": CHECKPOINT_DIGEST,
            "config_sha256": sha256_file(config_path),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "tf32": False,
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip(),
        }

    def run(self, job: Mapping[str, Any], output_dir: Path) -> RunnerOutput:
        if job.get("backend") != "cuda":
            raise ValueError("H100 quality runner only accepts CUDA jobs")
        torch = self.torch
        seed = int(job["seed"])
        latent_frames = int(job["latent_frames"])
        noise_plan = None
        if job.get("split") == "holdout":
            if job.get("role") != "reference" or self.noise_root is None:
                raise ValueError("H100 holdout references require a shared noise root")
            noise_plan = StochasticPlanReader(
                plan_directory(
                    self.noise_root, seed=seed, latent_frames=latent_frames
                )
                / "stochastic.bf16.json",
                seed=seed,
                latent_frames=latent_frames,
            )
        cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
        stochastic = StochasticWriter(output_dir)
        initial_shape = (1, latent_frames, 16, 60, 104)
        initial_noise = (
            noise_plan.next_torch("initial_noise", initial_shape, device="cpu")
            if noise_plan is not None
            else torch.randn(
                initial_shape,
                generator=cpu_generator,
                dtype=torch.float32,
                device="cpu",
            ).to(torch.bfloat16)
        )
        stochastic.append("initial_noise", initial_noise)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        noise_provider = None
        noise_name_provider = None
        if noise_plan is not None:
            def noise_provider(name, shape, dtype, device):
                return noise_plan.next_torch(
                    name, shape, device=str(device)
                ).to(dtype=dtype)

            configured_steps = (1000, 750, 500, 250)

            def noise_name_provider(window_index, block_id, timestep_index):
                return transition_name(
                    window_index,
                    block_id,
                    configured_steps[timestep_index],
                    configured_steps[timestep_index + 1],
                )
        generation_started = time.monotonic()
        try:
            video, latents = self.pipeline.inference_rolling_forcing(
                noise=initial_noise.to(device="cuda"),
                text_prompts=[str(job["prompt"])],
                return_latents=True,
                profile=True,
                noise_generator=cpu_generator if noise_plan is None else None,
                noise_callback=stochastic.append,
                noise_provider=noise_provider,
                noise_name_provider=noise_name_provider,
            )
        except BaseException:
            stochastic.abort()
            raise
        torch.cuda.synchronize()
        if noise_plan is not None:
            noise_plan.require_consumed()
        generation_seconds = time.monotonic() - generation_started
        if latents.shape[1] != int(job["latent_frames"]):
            raise ValueError("generated latent frame count does not match frozen job")
        if video.shape[1] != int(job["pixel_frames"]):
            raise ValueError("decoded pixel frame count does not match frozen job")
        pixels = (
            video.mul(255.0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)[0]
            .permute(0, 2, 3, 1)
            .cpu()
            .contiguous()
        )
        artifacts = write_tensor(
            output_dir / "pixels.bin",
            pixels,
            layout="T,H,W,C",
            schema="liveframe.decoded-pixels.v1",
        )
        latent_artifacts = write_tensor(
            output_dir / "latents.bf16",
            latents.cpu(),
            layout="B,T,C,H,W",
            schema="liveframe.output-latents.v1",
        )
        artifacts.extend(latent_artifacts)
        artifacts.extend(
            stochastic.finish(
                seed=seed,
                contract=STOCHASTIC_CONTRACT if noise_plan is not None else None,
                latent_frames=latent_frames if noise_plan is not None else None,
            )
        )
        del video, latents
        self.pipeline.vae.model.clear_cache()
        torch.cuda.empty_cache()
        metric_started = time.monotonic()
        metrics = self.metrics.measure(
            output_dir / "pixels.bin",
            prompt=str(job["prompt"]),
            expected_frames=int(job["pixel_frames"]),
        )
        torch.cuda.synchronize()
        metric_seconds = time.monotonic() - metric_started
        profile = self.pipeline.last_profile
        if not isinstance(profile, dict):
            raise ValueError("rolling pipeline did not return a timing profile")
        return RunnerOutput(
            artifact_paths=artifacts,
            metrics=metrics,
            runtime=self.runtime,
            timing={
                "generation_seconds": generation_seconds,
                "metric_seconds": metric_seconds,
                "initialization_seconds": float(profile["initialization_ms"]) / 1000.0,
                "diffusion_seconds": float(profile["diffusion_ms"]) / 1000.0,
                "vae_seconds": float(profile["vae_ms"]) / 1000.0,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT
        / "third_party"
        / "Causal-Forcing"
        / "long_video"
        / "configs"
        / "rolling_forcing_dmd.yaml",
    )
    parser.add_argument("--metric-batch-size", type=int, default=8)
    parser.add_argument("--noise-root", type=Path)
    parser.add_argument("--holdout-reference", action="store_true")
    parser.add_argument(
        "--wan-model-root",
        type=Path,
        default=REPOSITORY_ROOT / "wan_models" / "Wan2.1-T2V-1.3B",
    )
    args = parser.parse_args()
    protocol, protocol_digest = load_protocol(args.protocol)
    manifest = load_run_manifest(args.manifest, protocol_digest=protocol_digest)
    require_frozen_identities(protocol_digest, manifest)
    batch = (
        load_holdout_reference_batch_plan(args.batch, manifest)
        if args.holdout_reference
        else load_batch_plan(args.batch, manifest)
    )
    if args.holdout_reference:
        require_frozen_h100_reference_batch(batch)
        if args.noise_root is None:
            raise ValueError("H100 holdout references require --noise-root")
        if Path(batch["output_root"]) != args.output_root:
            raise ValueError("output root does not match the holdout reference batch")
    metric_provenance = validate_metric_artifact(protocol, REPOSITORY_ROOT)
    runner = H100QualityRunner(
        checkpoint=args.checkpoint,
        config_path=args.config,
        wan_model_root=args.wan_model_root,
        metric_provenance=metric_provenance,
        metric_batch_size=args.metric_batch_size,
        noise_root=args.noise_root,
    )
    report = execute_batch(protocol, manifest, batch, args.output_root, runner)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
