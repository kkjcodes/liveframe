"""Preflight or execute the frozen paired H100 framewise refresh campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.quality_artifacts import (  # noqa: E402
    StochasticWriter,
    TensorStreamWriter,
    atomic_json,
)
from liveframe.quality_calibration import load_protocol  # noqa: E402
from liveframe.quality_capture import sha256_file  # noqa: E402
from liveframe.quality_metric_runner import (  # noqa: E402
    PerceptualMetricRunner,
    load_pixels,
    validate_metric_artifact,
)
from liveframe.refresh_capture import (  # noqa: E402
    EVIDENCE_POLICY,
    H100_EVIDENCE,
    build_capture_result,
    load_capture_batch,
    publish_capture_result,
)
from liveframe.refresh_proof import (  # noqa: E402
    DENOMINATOR_SHA256,
    MAX_CONTEXT_FRAMES,
    MODEL_LAYERS,
    TOKENS_PER_FRAME,
    load_refresh_proof_manifest,
)
from liveframe.refresh_trajectory import (  # noqa: E402
    PairedFrameResult,
    PairedStochasticFrame,
    TimedRefresh,
    run_paired_refresh_trajectory,
)
from liveframe.torch_refresh_rerope import (  # noqa: E402
    build_pinned_framewise_refresh_harness,
    tensor_sha256,
)

SOURCE_REVISION = "8db419e341e5fc52542c0b2c4542728420ddfb4a"
MANIFEST_DIGEST = "sha256:c97a6fc1f042a7f56eab08273f0f6bd2f343b938498907c267a7d7ab3334d59b"
BATCH_DIGEST = "sha256:d9541d8a7e77f0495bf80ed8b8536bed4b56ea0974e48ed1b8732d045d3e4934"
NUMERICAL_POLICY = cast(Mapping[str, Any], EVIDENCE_POLICY["numerical"])
CHECKPOINT_DIGEST = str(NUMERICAL_POLICY["checkpoint_sha256"])
FIRST_CONFIGURED_STEPS = (1000, 750, 500, 250)
LATER_CONFIGURED_STEPS = (1000,)


@dataclass(frozen=True, slots=True)
class FrameInputs:
    initial_noise: Any
    scheduler_noise: tuple[Any, ...]


@dataclass(slots=True)
class RuntimeState:
    name: str
    kv_cache: list[dict[str, Any]]
    cross_cache: list[dict[str, Any]]
    conditional: dict[str, Any]
    denoising_seconds: float = 0.0
    refresh_seconds: float = 0.0


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _load_bf16_tensor(path: Path) -> Any:
    import torch

    metadata = _object(path.with_suffix(path.suffix + ".json"))
    if (
        metadata.get("dtype") != "bfloat16"
        or metadata.get("layout") != "B,T,C,H,W"
        or metadata.get("schema") != "liveframe.refresh-latents.v2"
    ):
        raise ValueError("unsupported streamed refresh latent artifact")
    shape = tuple(int(size) for size in metadata["shape"])
    if len(shape) != 5 or shape[0] != 1:
        raise ValueError("streamed refresh latents must be singleton-batch B,T,C,H,W")
    element_count = math.prod(shape)
    if path.stat().st_size != element_count * 2:
        raise ValueError("streamed refresh latent byte size mismatch")
    return torch.from_file(
        str(path), shared=False, size=element_count, dtype=torch.bfloat16
    ).reshape(shape)


def _drift(left: Any, right: Any) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    left_flat = left.detach().float().reshape(-1)
    right_flat = right.detach().float().reshape(-1)
    delta = left_flat - right_flat
    if bool(left_flat.norm().item()) and bool(right_flat.norm().item()):
        cosine = float(functional.cosine_similarity(left_flat, right_flat, dim=0).item())
        cosine = min(1.0, max(-1.0, cosine))
    else:
        cosine = 1.0 if torch.equal(left_flat, right_flat) else 0.0
    return {
        "cosine": cosine,
        "rmse": float(delta.square().mean().sqrt().item()),
        "max_abs": float(delta.abs().max().item()),
    }


def _playback_seconds(frame_index: int) -> float:
    return (4 * frame_index + 1) / 16


def _timed(operation: Any) -> tuple[Any, float]:
    import torch

    torch.cuda.synchronize()
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def _warped_steps(generator: Any, configured: Sequence[int]) -> tuple[float, ...]:
    import torch

    timeline = torch.cat(
        (generator.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
    )
    return tuple(float(timeline[1000 - step].item()) for step in configured)


def _make_caches(generator: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    capacity = MAX_CONTEXT_FRAMES * TOKENS_PER_FRAME
    kv_cache = [
        {
            "k": torch.zeros((1, capacity, 12, 128), device="cuda", dtype=torch.bfloat16),
            "v": torch.zeros((1, capacity, 12, 128), device="cuda", dtype=torch.bfloat16),
            "global_end_index": torch.zeros(1, device="cuda", dtype=torch.long),
            "local_end_index": torch.zeros(1, device="cuda", dtype=torch.long),
        }
        for _ in range(MODEL_LAYERS)
    ]
    cross_cache = [
        {
            "k": torch.zeros((1, 512, 12, 128), device="cuda", dtype=torch.bfloat16),
            "v": torch.zeros((1, 512, 12, 128), device="cuda", dtype=torch.bfloat16),
            "is_init": False,
        }
        for _ in range(MODEL_LAYERS)
    ]
    return kv_cache, cross_cache


def _mutable_identities(state: RuntimeState) -> set[int]:
    identities = {id(state.kv_cache), id(state.cross_cache), id(state.conditional)}
    for cache in (*state.kv_cache, *state.cross_cache):
        identities.add(id(cache))
        identities.update(id(value) for value in cache.values() if hasattr(value, "shape"))
    identities.update(
        id(value) for value in state.conditional.values() if hasattr(value, "shape")
    )
    return identities


def _source_report(causal_root: Path, checkpoint: Path) -> dict[str, Any]:
    source_files = {
        "wan/modules/causal_model.py": str(NUMERICAL_POLICY["causal_model_sha256"]),
        "utils/wan_wrapper.py": str(NUMERICAL_POLICY["wan_wrapper_sha256"]),
    }
    observed = {}
    for relative, expected in source_files.items():
        path = causal_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"refresh source digest mismatch: {relative}")
        observed[relative] = expected
    revision = subprocess.check_output(
        ["git", "-C", str(causal_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("refresh source revision mismatch")
    if sha256_file(checkpoint) != CHECKPOINT_DIGEST:
        raise ValueError("refresh checkpoint digest mismatch")
    return {
        "source_revision": revision,
        "checkpoint_sha256": CHECKPOINT_DIGEST,
        "source_files": observed,
    }


def load_campaign(
    *,
    protocol_path: Path,
    bundle_path: Path,
    manifest_path: Path,
    batch_path: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol, protocol_digest = load_protocol(protocol_path)
    bundle = _object(bundle_path)
    manifest = load_refresh_proof_manifest(
        manifest_path, protocol, protocol_digest, bundle
    )
    if manifest["manifest_digest"] != MANIFEST_DIGEST:
        raise ValueError("refresh runner manifest digest mismatch")
    batch = load_capture_batch(batch_path, manifest)
    if batch["batch_digest"] != BATCH_DIGEST:
        raise ValueError("refresh runner batch digest mismatch")
    if Path(str(batch["output_root"])) != output_root:
        raise ValueError("refresh runner output root differs from the frozen batch")
    return protocol, manifest, batch


def _selected_jobs(
    batch: Mapping[str, Any], run_ids: set[str] | None, limit: int | None
) -> list[dict[str, Any]]:
    jobs = [dict(job) for job in batch["jobs"]]
    known = {str(job["run_id"]) for job in jobs}
    if run_ids is not None:
        unknown = run_ids - known
        if unknown:
            raise ValueError(f"refresh runner requested unknown run IDs: {sorted(unknown)}")
        jobs = [job for job in jobs if str(job["run_id"]) in run_ids]
    if limit is not None:
        if limit < 1:
            raise ValueError("refresh runner limit must be positive")
        jobs = jobs[:limit]
    return jobs


def require_clean_outputs(jobs: Sequence[Mapping[str, Any]], output_root: Path) -> None:
    for job in jobs:
        output = output_root / str(job["run_id"])
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"selected refresh output is not empty: {output}")


def require_execution_acknowledgment(
    acknowledgment: str | None, identity_report: Path | None
) -> Path:
    if acknowledgment != MANIFEST_DIGEST:
        raise ValueError("refresh execution requires the exact manifest visibility acknowledgment")
    if identity_report is None:
        raise ValueError("refresh execution requires --identity-anchor-report")
    return identity_report


def validate_identity_anchor_report(path: Path) -> str:
    identity = _object(path)
    nested_identity = identity.get("identity_anchor")
    observed = [
        value
        for value in (
            identity.get("identity_anchor_passed"),
            identity.get("passed"),
            nested_identity.get("passed") if isinstance(nested_identity, dict) else None,
        )
        if value is not None
    ]
    if not observed or any(value is not True for value in observed):
        raise ValueError("refresh identity-anchor report is not passing")
    return sha256_file(path)


class H100RefreshExecutor:
    def __init__(
        self,
        *,
        causal_root: Path,
        checkpoint: Path,
        metric_provenance: dict[str, Any],
        metric_batch_size: int,
        identity_report: Path,
    ) -> None:
        import torch

        causal_root = causal_root.resolve()
        checkpoint = checkpoint.resolve()
        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_name(0) != "NVIDIA H100 80GB HBM3"
        ):
            raise ValueError("refresh execution requires NVIDIA H100 80GB HBM3")
        self.causal_root = causal_root
        sys.path.insert(0, str(causal_root))
        prior_cwd = Path.cwd()
        os.chdir(causal_root)
        try:
            import wan.modules.causal_model as causal_model  # type: ignore[import-not-found]
            from utils.wan_wrapper import (  # type: ignore[import-not-found]
                WanDiffusionWrapper,
                WanTextEncoder,
                WanVAEWrapper,
            )

            generator = WanDiffusionWrapper(
                model_name="Wan2.1-T2V-1.3B",
                timestep_shift=5.0,
                is_causal=True,
                local_attn_size=MAX_CONTEXT_FRAMES,
                sink_size=0,
            )
            envelope = torch.load(
                checkpoint, map_location="cpu", weights_only=True, mmap=True
            )
            normalized = {}
            for name, tensor in envelope["generator_ema"].items():
                normalized[name.replace("model._fsdp_wrapped_module.", "model.", 1)] = tensor
            generator.load_state_dict(normalized, strict=True)
            self.generator = generator.to(device="cuda", dtype=torch.bfloat16).eval()
            self.text_encoder = WanTextEncoder().to(dtype=torch.bfloat16).eval()
            self.vae = WanVAEWrapper().to(dtype=torch.bfloat16).eval()
            self.causal_model = causal_model
        finally:
            os.chdir(prior_cwd)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.metric_provenance = metric_provenance
        self.metric_batch_size = metric_batch_size
        self.metrics: PerceptualMetricRunner | None = None
        identity_report_digest = validate_identity_anchor_report(identity_report)
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        self.runtime = {
            "gpu": torch.cuda.get_device_name(0),
            "source_revision": SOURCE_REVISION,
            "checkpoint_sha256": CHECKPOINT_DIGEST,
            "causal_model_sha256": sha256_file(causal_root / "wan/modules/causal_model.py"),
            "wan_wrapper_sha256": sha256_file(causal_root / "utils/wan_wrapper.py"),
            "tf32": False,
            "identity_anchor_passed": True,
            "identity_anchor_report_sha256": identity_report_digest,
            "harness_sha256": sha256_file(Path(__file__)),
            "runner_core_sha256": sha256_file(
                REPOSITORY_ROOT / "src/liveframe/refresh_trajectory.py"
            ),
            "refresh_hook_sha256": sha256_file(
                REPOSITORY_ROOT / "src/liveframe/torch_refresh_rerope.py"
            ),
            "refresh_contract_sha256": sha256_file(
                REPOSITORY_ROOT / "src/liveframe/refresh_proof.py"
            ),
            "driver": driver,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "timing_protocol": "paired-interleaved-instrumented-numerical-only",
            "denominator_sha256": DENOMINATOR_SHA256,
        }

    def _prompt(self, prompt: str) -> dict[str, Any]:
        self.text_encoder.to(device="cuda", dtype=self.torch.bfloat16)
        try:
            with self.torch.inference_mode():
                encoded = self.text_encoder([prompt])
        finally:
            self.text_encoder.to(device="cpu")
            self.torch.cuda.empty_cache()
        return {"prompt_embeds": encoded["prompt_embeds"].detach()}

    def _inputs(
        self, output: Path, *, seed: int, latent_frames: int
    ) -> list[PairedStochasticFrame[FrameInputs]]:
        torch = self.torch
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial = torch.randn(
            (1, latent_frames, 16, 60, 104), generator=generator, dtype=torch.float32
        ).to(torch.bfloat16)
        writer = StochasticWriter(output)
        writer.append("initial_noise", initial)
        frames = []
        try:
            for frame_index in range(latent_frames):
                noises = []
                if frame_index == 0:
                    for transition in range(len(FIRST_CONFIGURED_STEPS) - 1):
                        noise = torch.randn(
                            (1, 1, 16, 60, 104),
                            generator=generator,
                            dtype=torch.float32,
                        ).to(torch.bfloat16)
                        writer.append(
                            f"frame-{frame_index:04d}.scheduler-noise-{transition:02d}",
                            noise,
                        )
                        noises.append(noise)
                frame_noise = initial[:, frame_index : frame_index + 1]
                digest = tensor_sha256(frame_noise)
                frame_input = FrameInputs(frame_noise.clone(), tuple(noises))
                frames.append(
                    PairedStochasticFrame(
                        frame_index,
                        frame_input,
                        FrameInputs(frame_noise.clone(), tuple(value.clone() for value in noises)),
                        digest,
                        digest,
                    )
                )
            writer.finish(seed=seed, latent_frames=latent_frames)
        except BaseException:
            writer.abort()
            raise
        return frames

    def _decode_to_pixels(
        self,
        latents: Any,
        output_path: Path,
        *,
        expected_frames: int,
        reference_path: Path | None = None,
    ) -> tuple[float, list[dict[str, Any]]]:
        torch = self.torch
        self.vae.model.clear_cache()
        writer = TensorStreamWriter(
            output_path,
            layout="T,H,W,C",
            schema="liveframe.decoded-pixels.v1",
        )
        reference = load_pixels(reference_path)[0] if reference_path is not None else None
        decode_seconds = 0.0
        pixel_curve: list[dict[str, Any]] = []
        pixel_cursor = 0
        try:
            for frame_index in range(latents.shape[1]):
                pixels, seconds = _timed(
                    lambda index=frame_index: self.vae.decode_to_pixel(
                        latents[:, index : index + 1].to(device="cuda"),
                        use_cache=True,
                    )
                )
                decode_seconds += seconds
                pixels.add_(1).mul_(127.5).round_().clamp_(0, 255)
                rgb = (
                    pixels.to(torch.uint8)[0]
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .contiguous()
                )
                writer.append(rgb)
                next_cursor = pixel_cursor + rgb.shape[0]
                if reference is not None:
                    reference_chunk = torch.from_numpy(
                        reference[pixel_cursor:next_cursor].copy()
                    )
                    pixel_curve.append(
                        {
                            "window_index": frame_index,
                            "playback_seconds": _playback_seconds(frame_index),
                            **_drift(reference_chunk, rgb),
                        }
                    )
                pixel_cursor = next_cursor
                del pixels, rgb
            if pixel_cursor != expected_frames:
                raise ValueError("refresh decoded pixel frame count mismatch")
            writer.finish()
        except BaseException:
            writer.abort()
            raise
        return decode_seconds, pixel_curve

    def _metrics(self) -> PerceptualMetricRunner:
        if self.metrics is None:
            self.metrics = PerceptualMetricRunner(
                self.metric_provenance,
                device="cuda",
                batch_size=self.metric_batch_size,
            )
        else:
            self.metrics.to("cuda")
        return self.metrics

    def run_allocation_smoke(self, output_path: Path) -> dict[str, Any]:
        """Exercise bounded peak-memory phases without loading campaign prompts or seeds."""

        if output_path.exists():
            raise FileExistsError(
                f"refusing to replace refresh allocation smoke: {output_path}"
            )
        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        checkpoints: list[dict[str, Any]] = []

        def record(stage: str) -> None:
            torch.cuda.synchronize()
            checkpoints.append(
                {
                    "stage": stage,
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )

        if self.metrics is not None:
            self.metrics.to("cpu")
        self.generator.to(device="cuda", dtype=torch.bfloat16)
        conditional = self._prompt("synthetic allocation smoke; no proof input")
        reference_kv, reference_cross = _make_caches(self.generator)
        candidate_kv, candidate_cross = _make_caches(self.generator)
        context_tokens = (MAX_CONTEXT_FRAMES - 1) * TOKENS_PER_FRAME
        for cache in (*reference_kv, *candidate_kv):
            cache["global_end_index"].fill_(context_tokens)
            cache["local_end_index"].fill_(context_tokens)
        noisy = torch.zeros((1, 1, 16, 60, 104), device="cuda", dtype=torch.bfloat16)
        timestep = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            _, generated = self.generator(
                noisy_image_or_video=noisy,
                conditional_dict=conditional,
                timestep=timestep,
                kv_cache=reference_kv,
                crossattn_cache=reference_cross,
                current_start=context_tokens,
            )
        record("maximum-context-generation")
        del (
            generated,
            noisy,
            timestep,
            conditional,
            reference_kv,
            reference_cross,
            candidate_kv,
            candidate_cross,
        )
        self.generator.to("cpu")
        torch.cuda.empty_cache()

        self.vae.to(device="cuda", dtype=torch.bfloat16)
        self.vae.model.clear_cache()
        latent = torch.zeros((1, 1, 16, 60, 104), dtype=torch.bfloat16)
        decoded_frames = 0
        with torch.inference_mode():
            for _ in range(243):
                pixels = self.vae.decode_to_pixel(latent.to("cuda"), use_cache=True)
                decoded_frames += int(pixels.shape[1])
                del pixels
        if decoded_frames != 969:
            raise ValueError("allocation smoke VAE frame geometry mismatch")
        record("full-sequence-vae-cache")
        del latent
        self.vae.to("cpu")
        torch.cuda.empty_cache()

        metrics = self._metrics()
        with tempfile.TemporaryDirectory(dir=output_path.parent) as temporary:
            pixel_path = Path(temporary) / "synthetic.rgb"
            writer = TensorStreamWriter(
                pixel_path,
                layout="T,H,W,C",
                schema="liveframe.decoded-pixels.v1",
            )
            base = torch.arange(480 * 832 * 3, dtype=torch.int32).reshape(480, 832, 3)
            frame_offsets = torch.arange(self.metric_batch_size, dtype=torch.int32).mul_(17)
            synthetic_pixels = (
                base.unsqueeze(0).add(frame_offsets[:, None, None, None]).remainder_(256)
            ).to(torch.uint8)
            writer.append(synthetic_pixels)
            writer.finish()
            metrics.measure(
                pixel_path,
                prompt="synthetic allocation smoke; no proof input",
                expected_frames=self.metric_batch_size,
            )
            metrics.measure_pairwise(
                pixel_path,
                pixel_path,
                expected_frames=self.metric_batch_size,
            )
        record("metric-batch")
        metrics.to("cpu")
        torch.cuda.empty_cache()

        report = {
            "schema": "liveframe.h100-refresh-allocation-smoke.v1",
            "proof_inputs_loaded": False,
            "proof_results_generated": False,
            "synthetic_prompt": True,
            "latent_frames": 243,
            "pixel_frames": decoded_frames,
            "metric_batch_size": self.metric_batch_size,
            "memory_protocol": "phase-exclusive-streamed-v1",
            "cuda_memory_checkpoints": checkpoints,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "runtime": self.runtime,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output_path, report)
        return report

    def run(self, job: Mapping[str, Any], manifest: Mapping[str, Any], output: Path) -> None:
        torch = self.torch
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"refusing to replace refresh output: {output}")
        output.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        if self.metrics is not None:
            self.metrics.to("cpu")
        self.generator.to(device="cuda", dtype=torch.bfloat16)
        torch.cuda.empty_cache()
        memory_checkpoints: list[dict[str, Any]] = []

        def record_memory(stage: str) -> None:
            torch.cuda.synchronize()
            memory_checkpoints.append(
                {
                    "stage": stage,
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )

        record_memory("generator-resident")
        conditional = self._prompt(str(job["prompt"]))
        frames = self._inputs(
            output, seed=int(job["seed"]), latent_frames=int(job["latent_frames"])
        )
        reference_kv, reference_cross = _make_caches(self.generator)
        candidate_kv, candidate_cross = _make_caches(self.generator)
        reference_conditional = {
            name: value.detach().clone() for name, value in conditional.items()
        }
        candidate_conditional = {
            name: value.detach().clone() for name, value in conditional.items()
        }
        reference = RuntimeState(
            "reference", reference_kv, reference_cross, reference_conditional
        )
        candidate = RuntimeState(
            "candidate", candidate_kv, candidate_cross, candidate_conditional
        )
        harness = build_pinned_framewise_refresh_harness(
            self.generator, self.causal_model
        )
        registry = harness.registry
        latent_curve: list[dict[str, Any]] = []
        key_curve: list[dict[str, Any]] = []
        value_curve: list[dict[str, Any]] = []
        reference_latent_path = output / "reference-latents.bf16"
        candidate_latent_path = output / "candidate-latents.bf16"
        reference_latent_writer = TensorStreamWriter(
            reference_latent_path,
            layout="B,T,C,H,W",
            schema="liveframe.refresh-latents.v2",
            concat_dim=1,
        )
        candidate_latent_writer = TensorStreamWriter(
            candidate_latent_path,
            layout="B,T,C,H,W",
            schema="liveframe.refresh-latents.v2",
            concat_dim=1,
        )

        first_steps = _warped_steps(self.generator, FIRST_CONFIGURED_STEPS)
        later_steps = _warped_steps(self.generator, LATER_CONFIGURED_STEPS)

        def denoise(
            state: RuntimeState, frame_index: int, inputs: FrameInputs
        ) -> Any:
            steps = first_steps if frame_index == 0 else later_steps

            def operation() -> Any:
                noisy = inputs.initial_noise.to(device="cuda")
                prediction = noisy
                for step_index, warped_step in enumerate(steps):
                    timestep = torch.full(
                        (1, 1), warped_step, device="cuda", dtype=torch.float32
                    )
                    _, prediction = self.generator(
                        noisy_image_or_video=noisy,
                        conditional_dict=state.conditional,
                        timestep=timestep,
                        kv_cache=state.kv_cache,
                        crossattn_cache=state.cross_cache,
                        current_start=frame_index * TOKENS_PER_FRAME,
                    )
                    if step_index < len(steps) - 1:
                        target = torch.full(
                            (prediction.shape[0] * prediction.shape[1],),
                            steps[step_index + 1],
                            device="cuda",
                            dtype=torch.float32,
                        )
                        noisy = self.generator.scheduler.add_noise(
                            prediction.flatten(0, 1),
                            inputs.scheduler_noise[step_index]
                            .to(device="cuda")
                            .flatten(0, 1),
                            target,
                        ).unflatten(0, prediction.shape[:2])
                return prediction

            prediction, seconds = _timed(operation)
            state.denoising_seconds += seconds
            return prediction

        def refresh(
            state: RuntimeState, frame_index: int, prediction: Any
        ) -> TimedRefresh[None]:
            context_timestep = torch.zeros((1, 1), device="cuda", dtype=torch.float32)

            def operation() -> Any:
                return self.generator(
                    noisy_image_or_video=prediction,
                    conditional_dict=state.conditional,
                    timestep=context_timestep,
                    kv_cache=state.kv_cache,
                    crossattn_cache=state.cross_cache,
                    current_start=frame_index * TOKENS_PER_FRAME,
                )

            _, seconds = _timed(operation)
            state.refresh_seconds += seconds
            return TimedRefresh(None, seconds)

        def observe(
            reference_state: RuntimeState,
            candidate_state: RuntimeState,
            result: PairedFrameResult[Any, None],
        ) -> None:
            reference_latent_writer.append(result.reference_prediction)
            candidate_latent_writer.append(result.candidate_prediction)
            playback = _playback_seconds(result.frame_index)
            latent_curve.append(
                {
                    "window_index": result.frame_index,
                    "playback_seconds": playback,
                    **_drift(result.reference_prediction, result.candidate_prediction),
                }
            )
            for layer_index, (reference_cache, candidate_cache) in enumerate(
                zip(reference_state.kv_cache, candidate_state.kv_cache, strict=True)
            ):
                reference_end = int(reference_cache["local_end_index"].item())
                candidate_end = int(candidate_cache["local_end_index"].item())
                if reference_end != candidate_end:
                    raise ValueError("paired refresh cache cursors diverged")
                start = reference_end - TOKENS_PER_FRAME
                for name, rows in (("k", key_curve), ("v", value_curve)):
                    rows.append(
                        {
                            "window_index": result.frame_index,
                            "layer_index": layer_index,
                            "playback_seconds": playback,
                            **_drift(
                                reference_cache[name][:, start:reference_end],
                                candidate_cache[name][:, start:candidate_end],
                            ),
                        }
                    )
            if result.source_frame is not None:
                registry.release_destination(result.frame_index)

        try:
            with harness, torch.inference_mode():
                run_paired_refresh_trajectory(
                    harness,
                    frames,
                    reference_state=reference,
                    candidate_state=candidate,
                    mutable_identities=_mutable_identities,
                    denoise_reference=denoise,
                    denoise_candidate=denoise,
                    refresh_reference=refresh,
                    refresh_candidate=refresh,
                    observe_frame=observe,
                    retain_results=False,
                )
            reference_latent_writer.finish()
            candidate_latent_writer.finish()
        except BaseException:
            reference_latent_writer.abort()
            candidate_latent_writer.abort()
            raise
        record_memory("generation-complete")
        substitutions = list(registry.substitution_ancestry)
        reference_denoising = reference.denoising_seconds
        candidate_denoising = candidate.denoising_seconds
        reference_refresh = reference.refresh_seconds
        candidate_refresh = candidate.refresh_seconds
        del frames, conditional, reference, candidate, harness
        self.generator.to(device="cpu")
        torch.cuda.empty_cache()

        reference_latents = _load_bf16_tensor(reference_latent_path)
        candidate_latents = _load_bf16_tensor(candidate_latent_path)
        self.vae.to(device="cuda", dtype=torch.bfloat16)
        expected_pixels = int(job["pixel_frames"])
        reference_pixel_path = output / "reference-pixels.rgb"
        candidate_pixel_path = output / "candidate-pixels.rgb"
        reference_decode, _ = self._decode_to_pixels(
            reference_latents,
            reference_pixel_path,
            expected_frames=expected_pixels,
        )
        candidate_decode, pixel_curve = self._decode_to_pixels(
            candidate_latents,
            candidate_pixel_path,
            expected_frames=expected_pixels,
            reference_path=reference_pixel_path,
        )
        record_memory("decode-complete")
        del reference_latents, candidate_latents
        self.vae.to(device="cpu")
        torch.cuda.empty_cache()

        metrics = self._metrics()
        intrinsic = metrics.measure(
            output / "candidate-pixels.rgb",
            prompt=str(job["prompt"]),
            expected_frames=expected_pixels,
        )
        identity = metrics.measure_pairwise(
            output / "reference-pixels.rgb",
            output / "candidate-pixels.rgb",
            expected_frames=expected_pixels,
        )
        record_memory("metrics-complete")
        metrics.to("cpu")
        torch.cuda.empty_cache()
        result = build_capture_result(
            job,
            manifest,
            evidence_class=H100_EVIDENCE,
            drift_curves={
                "future_latent": latent_curve,
                "per-layer-key": key_curve,
                "per-layer-value": value_curve,
                "decoded-pixel": pixel_curve,
            },
            substitutions=substitutions,
            intrinsic_metrics=intrinsic,
            identity_metrics=identity,
            timing={
                "reference_refresh_seconds": reference_refresh,
                "candidate_refresh_seconds": candidate_refresh,
                "reroping_seconds": 0.0,
                "selection_seconds": 0.0,
                "cache_movement_seconds": 0.0,
                "synchronization_seconds": 0.0,
                "complete_reference_wall_seconds": (
                    reference_denoising + reference_refresh + reference_decode
                ),
                "complete_candidate_wall_seconds": (
                    candidate_denoising + candidate_refresh + candidate_decode
                ),
            },
            runtime={
                **self.runtime,
                "memory_protocol": "phase-exclusive-streamed-v1",
                "cuda_memory_checkpoints": memory_checkpoints,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            artifact_root=output,
        )
        publish_capture_result(output / "result.json", result, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--policy-bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--identity-anchor-report", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--allocation-smoke-output", type=Path)
    parser.add_argument("--metric-batch-size", type=int, default=8)
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--acknowledge-proof-visibility")
    args = parser.parse_args()

    if args.allocation_smoke_output is not None:
        if args.execute or args.acknowledge_proof_visibility is not None:
            raise ValueError("allocation smoke cannot cross the proof visibility boundary")
        if args.identity_anchor_report is None:
            raise ValueError("allocation smoke requires --identity-anchor-report")
        protocol, _ = load_protocol(args.protocol)
        metric_provenance = validate_metric_artifact(protocol, REPOSITORY_ROOT)
        executor = H100RefreshExecutor(
            causal_root=args.causal_root,
            checkpoint=args.checkpoint,
            metric_provenance=metric_provenance,
            metric_batch_size=args.metric_batch_size,
            identity_report=args.identity_anchor_report,
        )
        report = executor.run_allocation_smoke(args.allocation_smoke_output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    protocol, manifest, batch = load_campaign(
        protocol_path=args.protocol,
        bundle_path=args.policy_bundle,
        manifest_path=args.manifest,
        batch_path=args.batch,
        output_root=args.output_root,
    )
    source = _source_report(args.causal_root, args.checkpoint)
    jobs = _selected_jobs(
        batch, set(args.run_id) if args.run_id else None, args.limit
    )
    require_clean_outputs(jobs, args.output_root)
    report = {
        "schema": "liveframe.h100-refresh-runner-preflight.v2",
        "manifest_digest": manifest["manifest_digest"],
        "batch_digest": batch["batch_digest"],
        "selected_jobs": len(jobs),
        "proof_results_generated": False,
        "runner_sha256": sha256_file(Path(__file__)),
        "runner_core_sha256": sha256_file(
            REPOSITORY_ROOT / "src/liveframe/refresh_trajectory.py"
        ),
        "refresh_hook_sha256": sha256_file(
            REPOSITORY_ROOT / "src/liveframe/torch_refresh_rerope.py"
        ),
        **source,
    }
    if not args.execute:
        if args.preflight_output is not None:
            if args.preflight_output.exists():
                if _object(args.preflight_output) != report:
                    raise FileExistsError(
                        f"refusing to replace refresh runner preflight: {args.preflight_output}"
                    )
            else:
                args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(args.preflight_output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    identity_report = require_execution_acknowledgment(
        args.acknowledge_proof_visibility, args.identity_anchor_report
    )
    metric_provenance = validate_metric_artifact(protocol, REPOSITORY_ROOT)
    executor = H100RefreshExecutor(
        causal_root=args.causal_root,
        checkpoint=args.checkpoint,
        metric_provenance=metric_provenance,
        metric_batch_size=args.metric_batch_size,
        identity_report=identity_report,
    )
    for index, job in enumerate(jobs, start=1):
        output = args.output_root / str(job["run_id"])
        executor.run(job, manifest, output)
        print(f"completed refresh proof job {index}/{len(jobs)}: {job['run_id']}", flush=True)


if __name__ == "__main__":
    main()