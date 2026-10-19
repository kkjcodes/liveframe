"""Run the frozen 32-job TAEW2.1 paired decoder campaign.

For every frozen job in the protocol: generate the latent trajectory once from
the immutable noise plan, save it digest-bound, then decode it with BOTH the
pinned full Wan VAE and the TAEW2.1 fp32-im2col tiny decoder in a
deterministic-randomized interleaved order. Full and tiny outputs differ only
in decoder choice. Timing recorded here carries no claim authority (the formal
benchmark result is the timing authority); the campaign exists to produce the
paired quality captures for metric screening and the human study.

Fail-closed rules: requires a passed authorization record, digest-verified
noise plans, and refuses duplicate or partial results (a crashed job leaves a
`.incomplete` sentinel and is wiped and redone on resume; a completed job is
skipped). Swap is recorded per job but not enforced: the transient T5-XXL
prompt-encoding spike swaps on this host, campaign pixels are deterministic
regardless of paging, and campaign timing carries no claim authority. All 16
prompts are encoded once up front (a single T5 residency) and only the small
prompt embeddings stay resident across jobs.
"""

# mypy: ignore-errors

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from liveframe.decoders.taew21_mlx import (  # noqa: E402
    STARTUP_FRAMES_TO_TRIM,
    TAEW21MLXDecoder,
)
from liveframe.manifests import file_sha256  # noqa: E402
from liveframe.mlx_cfpp_benchmark import parse_swap_usage  # noqa: E402
from liveframe.mlx_quality_runner import (  # noqa: E402
    MLXQualityRunner,
    _write_mlx_tensor,
    mlx_runtime_provenance,
)
from liveframe.mlx_vae_decode import (  # noqa: E402
    _causal_decode_frame,
    _decoder_cache_slots,
)
from liveframe.quality_artifacts import StochasticWriter, atomic_json  # noqa: E402
from liveframe.quality_stochastic import (  # noqa: E402
    STOCHASTIC_CONTRACT,
    StochasticPlanReader,
    plan_directory,
)

HORIZON_LATENTS = {15: 60, 60: 243}
PLAYBACK_FPS = 16
RESOLUTION = (480, 832)
ENCODING_POLICY = {
    "codec": "h264_videotoolbox",
    "bitrate": "8M",
    "container": "mp4",
    "pixel_format": "rgb24",
    "framerate": PLAYBACK_FPS,
}


def _swap() -> dict[str, float]:
    output = subprocess.run(
        ["sysctl", "vm.swapusage"], check=True, capture_output=True, text=True
    ).stdout
    return parse_swap_usage(output)


def _thermal() -> list[str]:
    result = subprocess.run(
        ["pmset", "-g", "therm"], capture_output=True, check=False, text=True
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _encode_video(pixels_path: Path, output: Path, frames: int) -> float:
    height, width = RESOLUTION
    temporary = output.with_name(f".{output.name}.tmp.mp4")
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        ENCODING_POLICY["pixel_format"],
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(PLAYBACK_FPS),
        "-i",
        str(pixels_path),
        "-an",
        "-c:v",
        ENCODING_POLICY["codec"],
        "-b:v",
        ENCODING_POLICY["bitrate"],
        str(temporary),
    ]
    started = time.perf_counter()
    subprocess.run(command, check=True)
    temporary.replace(output)
    return time.perf_counter() - started


def _pixel_metadata(path: Path, frames: int) -> Path:
    metadata_path = path.with_suffix(path.suffix + ".json")
    atomic_json(
        metadata_path,
        {
            "schema": "liveframe.decoded-pixels.v1",
            "dtype": "uint8",
            "layout": "T,H,W,C",
            "shape": [frames, RESOLUTION[0], RESOLUTION[1], 3],
            "sha256": file_sha256(path),
        },
    )
    return metadata_path


class _Telemetry:
    def __init__(self, mx: Any) -> None:
        self.mx = mx
        mx.clear_cache()
        mx.reset_peak_memory()
        self.swap_before = _swap()
        self.thermal_before = _thermal()

    def finish(self) -> dict[str, Any]:
        return {
            "peak_mlx_memory_gib": self.mx.get_peak_memory() / 2**30,
            "swap_before": self.swap_before,
            "swap_after": _swap(),
            "thermal_before": self.thermal_before,
            "thermal_after": _thermal(),
        }


def _decode_full(mx: Any, vae: Any, latents: Any, output: Path) -> dict[str, Any]:
    telemetry = _Telemetry(mx)
    cache: list[Any] = [None] * _decoder_cache_slots(vae.decoder)
    decode_seconds = 0.0
    materialization_seconds = 0.0
    emitted = 0
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as stream:
        for frame in range(int(latents.shape[1])):
            latent = latents[:, frame : frame + 1].transpose(0, 2, 1, 3, 4)
            started = time.perf_counter()
            decoded = _causal_decode_frame(vae, latent, cache)
            mx.eval(decoded)
            decode_seconds += time.perf_counter() - started
            started = time.perf_counter()
            pixels = np.array(decoded[0].transpose(1, 2, 3, 0), dtype=np.float32)
            pixels = np.clip((pixels + 1.0) * 127.5, 0, 255).astype(np.uint8)
            materialization_seconds += time.perf_counter() - started
            stream.write(pixels.tobytes(order="C"))
            emitted += int(pixels.shape[0])
    os.replace(temporary, output)
    return {
        "decoder": "wan-full-vae",
        "emitted_frames": emitted,
        "decode_seconds": decode_seconds,
        "pixel_materialization_seconds": materialization_seconds,
        **telemetry.finish(),
    }


def _decode_tiny(mx: Any, tiny: TAEW21MLXDecoder, latents: Any, output: Path) -> dict[str, Any]:
    telemetry = _Telemetry(mx)
    memory: dict[int, Any] = {}
    decode_seconds = 0.0
    materialization_seconds = 0.0
    emitted = 0
    decoded_seen = 0
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as stream:
        for frame in range(int(latents.shape[1])):
            latent = latents[:, frame].transpose(0, 2, 3, 1).astype(mx.float32)
            started = time.perf_counter()
            decoded = tiny._decode_latent(latent, memory)
            decoded_seen += len(decoded)
            if decoded_seen <= STARTUP_FRAMES_TO_TRIM:
                raise RuntimeError("tiny decode produced no post-startup frames")
            if frame == 0:
                decoded = decoded[STARTUP_FRAMES_TO_TRIM:]
            value = mx.stack(decoded, axis=1)
            mx.eval(value)
            decode_seconds += time.perf_counter() - started
            started = time.perf_counter()
            pixels = np.asarray(value[0] * 255.0).round().clip(0, 255).astype(np.uint8)
            materialization_seconds += time.perf_counter() - started
            if not np.isfinite(pixels).all():
                raise RuntimeError(f"tiny decode frame {frame} produced non-finite pixels")
            stream.write(pixels.tobytes(order="C"))
            emitted += int(pixels.shape[0])
    os.replace(temporary, output)
    return {
        "decoder": "taew2_1-mlx-fp32-im2col",
        "emitted_frames": emitted,
        "decode_seconds": decode_seconds,
        "pixel_materialization_seconds": materialization_seconds,
        **telemetry.finish(),
    }


def _encode_all_prompts(
    runner: MLXQualityRunner, prompts: dict[str, str]
) -> tuple[dict[str, Any], float]:
    """Encode every campaign prompt with one transient T5 residency."""

    from mlx_video.models.wan_2.utils import encode_text, load_t5_encoder
    from transformers import AutoTokenizer

    from liveframe.mlx_quality_runner import (
        TOKENIZER_REPOSITORY,
        TOKENIZER_REVISION,
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_REPOSITORY, revision=TOKENIZER_REVISION, local_files_only=True
    )
    encoder = load_t5_encoder(runner.model_dir / "t5_encoder.safetensors", runner.config)
    embeddings: dict[str, Any] = {}
    for prompt_id, prompt in sorted(prompts.items()):
        embedding = encode_text(encoder, tokenizer, prompt, runner.config.text_len)
        runner.mx.eval(embedding)
        embeddings[prompt_id] = embedding
    elapsed = time.perf_counter() - started
    del encoder, tokenizer
    runner.mx.clear_cache()
    return embeddings, elapsed


def _run_job(
    job: Mapping,
    *,
    runner: MLXQualityRunner,
    tiny: TAEW21MLXDecoder,
    protocol: dict[str, Any],
    noise_root: Path,
    noise_entry: dict[str, Any],
    prompt_embedding: Any,
    text_encoding_seconds: float,
    runner_digest: str,
    authorization_sha256: str,
) -> dict[str, Any]:
    mx = runner.mx
    job_id = str(job["job_id"])
    horizon = int(job["horizon_seconds"])
    latent_frames = HORIZON_LATENTS[horizon]
    pixel_frames = int(protocol["campaign"]["horizons"][str(horizon)]["pixel_frames"])
    prompt = protocol["campaign"]["prompt_catalog"][str(job["prompt_id"])]
    seed = int(job["seed"])
    if protocol["campaign"]["seeds"][str(job["prompt_id"])] != seed:
        raise ValueError(f"job {job_id} seed does not match frozen prompt seed")
    job_dir = REPOSITORY_ROOT / str(job["output"])
    result_path = job_dir / "result.json"
    sentinel = job_dir / ".incomplete"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("schema") != "liveframe.taew21-campaign-job.v1" or not existing.get(
            "completed"
        ):
            raise ValueError(f"job {job_id} has an invalid existing result")
        print(f"job {job_id}: already complete, skipping", flush=True)
        return existing
    if job_dir.exists() and any(job_dir.iterdir()):
        if sentinel.exists():
            print(f"job {job_id}: wiping partial artifacts and redoing", flush=True)
            shutil.rmtree(job_dir)
        else:
            raise RuntimeError(
                f"job {job_id} directory has artifacts but no result and no sentinel"
            )
    job_dir.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("in-progress\n", encoding="utf-8")

    start_swap = _swap()
    job_started = time.perf_counter()
    plan_dir = plan_directory(noise_root, seed=seed, latent_frames=latent_frames)
    payload_digest = file_sha256(plan_dir / "stochastic.bf16")
    if payload_digest != noise_entry["payload_sha256"]:
        raise ValueError(f"job {job_id} noise plan payload digest mismatch")
    noise_plan = StochasticPlanReader(
        plan_dir / "stochastic.bf16.json", seed=seed, latent_frames=latent_frames
    )

    context_started = time.perf_counter()
    embedded_context = runner.model.embed_text([prompt_embedding])
    cross_kv = runner.model.prepare_cross_kv(embedded_context)
    runner.mx.eval(embedded_context, cross_kv)
    context_seconds = time.perf_counter() - context_started
    stochastic = StochasticWriter(job_dir)
    try:
        generation_started = time.perf_counter()
        latents, generation = runner._generate(
            {"latent_frames": latent_frames},
            embedded_context,
            cross_kv,
            stochastic,
            noise_plan,
        )
        generation_wall = time.perf_counter() - generation_started
        if int(latents.shape[1]) != latent_frames:
            raise ValueError(f"job {job_id} generated wrong latent frame count")
        latent_artifacts = _write_mlx_tensor(
            job_dir / "latents.bf16",
            latents,
            layout="B,T,C,H,W",
            schema="liveframe.output-latents.v1",
        )
        stochastic_artifacts = stochastic.finish(
            seed=seed, contract=STOCHASTIC_CONTRACT, latent_frames=latent_frames
        )
    except BaseException:
        stochastic.abort()
        raise
    del embedded_context, cross_kv
    mx.clear_cache()

    order_bit = int(hashlib.sha256(job_id.encode("utf-8")).hexdigest(), 16) % 2
    order = ("full", "tiny") if order_bit == 0 else ("tiny", "full")
    decodes: dict[str, dict[str, Any]] = {}
    for which in order:
        pixels_path = job_dir / f"pixels.{which}.bin"
        video_path = job_dir / f"{which}.mp4"
        if which == "full":
            record = _decode_full(mx, runner.vae, latents, pixels_path)
        else:
            record = _decode_tiny(mx, tiny, latents, pixels_path)
        if record["emitted_frames"] != pixel_frames:
            raise RuntimeError(
                f"job {job_id} {which} decode emitted {record['emitted_frames']} frames, "
                f"expected {pixel_frames}"
            )
        metadata_path = _pixel_metadata(pixels_path, pixel_frames)
        record["encoding_seconds"] = _encode_video(pixels_path, video_path, pixel_frames)
        record["pixels_path"] = str(pixels_path.relative_to(REPOSITORY_ROOT))
        record["pixels_sha256"] = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )["sha256"]
        record["video_path"] = str(video_path.relative_to(REPOSITORY_ROOT))
        record["video_sha256"] = file_sha256(video_path)
        record["video_bytes"] = video_path.stat().st_size
        record["complete_integrated_wall_seconds"] = (
            generation_wall
            + record["decode_seconds"]
            + record["pixel_materialization_seconds"]
            + record["encoding_seconds"]
        )
        decodes[which] = record
    del latents
    mx.clear_cache()

    pair_checks = {
        "frame_count_equal": decodes["full"]["emitted_frames"]
        == decodes["tiny"]["emitted_frames"],
        "dimensions": list(RESOLUTION),
        "playback_fps_equal": True,
        "encoding_policy_equal": True,
        "encoding_policy": ENCODING_POLICY,
    }
    if not pair_checks["frame_count_equal"]:
        raise RuntimeError(f"job {job_id} full/tiny frame counts differ")
    end_swap = _swap()
    result = {
        "schema": "liveframe.taew21-campaign-job.v1",
        "completed": True,
        "job": dict(job),
        "prompt": prompt,
        "timing_authority": "none-campaign-capture-only",
        "authorization_sha256": authorization_sha256,
        "runner_sha256": runner_digest,
        "noise_plan": {
            "directory": str(plan_dir.relative_to(REPOSITORY_ROOT)),
            "payload_sha256": payload_digest,
        },
        "latents_sha256": json.loads(
            (job_dir / "latents.bf16.json").read_text(encoding="utf-8")
        )["sha256"],
        "text_encoding_seconds": text_encoding_seconds,
        "context_preparation_seconds": context_seconds,
        "generation": {**generation, "generation_wall_seconds": generation_wall},
        "decoder_order": list(order),
        "decodes": decodes,
        "pair_checks": pair_checks,
        "job_wall_seconds": time.perf_counter() - job_started,
        "start_swap": start_swap,
        "end_swap": end_swap,
        "swap_delta_mib": end_swap["used_mib"] - start_swap["used_mib"],
        "swap_clean": end_swap["used_mib"] == start_swap["used_mib"] == 0,
        "artifact_count": len(latent_artifacts) + len(stochastic_artifacts) + 6,
        "runtime": runner.runtime,
    }
    temporary = result_path.with_name(f".{result_path.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, result_path)
    sentinel.unlink()
    print(
        json.dumps(
            {
                "job_id": job_id,
                "job_wall_seconds": result["job_wall_seconds"],
                "full_decode_seconds": decodes["full"]["decode_seconds"],
                "tiny_decode_seconds": decodes["tiny"]["decode_seconds"],
                "swap_clean": result["swap_clean"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--noise-index", type=Path, required=True)
    parser.add_argument("--noise-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tiny-weights", type=Path, required=True)
    parser.add_argument("--tiny-weights-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.noise_root = args.noise_root.resolve()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    if (
        authorization.get("schema") != "liveframe.taew21-campaign-authorization.v1"
        or not authorization.get("passed")
        or authorization.get("protocol_sha256") != file_sha256(args.protocol)
        or authorization.get("tiny_weights_sha256") != args.tiny_weights_sha256
    ):
        raise ValueError("campaign requires a passed authorization record")
    noise_index = json.loads(args.noise_index.read_text(encoding="utf-8"))
    if (
        noise_index.get("schema") != "liveframe.taew21-campaign-noise-index.v1"
        or noise_index.get("protocol_sha256") != file_sha256(args.protocol)
        or noise_index.get("authorization_sha256") != file_sha256(args.authorization)
    ):
        raise ValueError("campaign requires a bound noise index")
    noise_entries = {
        (int(plan["seed"]), int(plan["latent_frames"])): plan
        for plan in noise_index["plans"]
    }
    if str(args.output_root) != protocol["campaign"]["output_root"]:
        raise ValueError("output root does not match the frozen campaign root")

    import mlx.core as mx
    from mlx_video.models.wan_2.config import WanModelConfig
    from mlx_video.models.wan_2.utils import load_vae_decoder, load_wan_model

    runtime = mlx_runtime_provenance(mx)
    raw_config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    allowed = WanModelConfig.__dataclass_fields__.keys()
    runner = MLXQualityRunner.__new__(MLXQualityRunner)
    runner.mx = mx
    runner.config = WanModelConfig(
        **{key: value for key, value in raw_config.items() if key in allowed}
    )
    runner.model = load_wan_model(args.model_dir / "model.safetensors", runner.config)
    runner.vae = load_vae_decoder(args.model_dir / "vae.safetensors", runner.config)
    runner.model_dir = args.model_dir
    runner.noise_root = args.noise_root
    runner.vae_execution_mode = "eager"
    runner.runtime = {
        **runtime,
        "model_sha256": authorization["model_assets"]["model.safetensors"],
        "t5_sha256": authorization["model_assets"]["t5_encoder.safetensors"],
        "vae_sha256": authorization["model_assets"]["vae.safetensors"],
        "tiny_weights_sha256": args.tiny_weights_sha256,
        "vae_execution_mode": "eager",
    }
    tiny = TAEW21MLXDecoder(
        args.tiny_weights,
        expected_sha256=args.tiny_weights_sha256,
        convolution_mode="fp32-im2col",
    )
    runner_digest = file_sha256(Path(__file__).resolve())
    authorization_sha256 = file_sha256(args.authorization)

    print("encoding all campaign prompts (single T5 residency)", flush=True)
    embeddings, text_encoding_seconds = _encode_all_prompts(
        runner, dict(protocol["campaign"]["prompt_catalog"])
    )
    print(
        json.dumps(
            {"prompts_encoded": len(embeddings), "seconds": text_encoding_seconds},
            sort_keys=True,
        ),
        flush=True,
    )

    jobs = protocol["campaign"]["jobs"]
    results = []
    for index, job in enumerate(jobs):
        latent_frames = HORIZON_LATENTS[int(job["horizon_seconds"])]
        entry = noise_entries[(int(job["seed"]), latent_frames)]
        print(f"=== job {index + 1}/{len(jobs)}: {job['job_id']} ===", flush=True)
        results.append(
            _run_job(
                job,
                runner=runner,
                tiny=tiny,
                protocol=protocol,
                noise_root=args.noise_root,
                noise_entry=entry,
                prompt_embedding=embeddings[str(job["prompt_id"])],
                text_encoding_seconds=text_encoding_seconds,
                runner_digest=runner_digest,
                authorization_sha256=authorization_sha256,
            )
        )
    if args.summary.exists():
        raise FileExistsError(f"refusing to replace campaign summary: {args.summary}")
    summary = {
        "schema": "liveframe.taew21-campaign-summary.v1",
        "protocol_sha256": file_sha256(args.protocol),
        "authorization_sha256": authorization_sha256,
        "noise_index_sha256": file_sha256(args.noise_index),
        "runner_sha256": runner_digest,
        "job_count": len(results),
        "all_swap_clean": all(result["swap_clean"] for result in results),
        "jobs": [
            {
                "job_id": result["job"]["job_id"],
                "result_sha256": file_sha256(
                    REPOSITORY_ROOT / result["job"]["output"] / "result.json"
                ),
                "latents_sha256": result["latents_sha256"],
                "full_pixels_sha256": result["decodes"]["full"]["pixels_sha256"],
                "tiny_pixels_sha256": result["decodes"]["tiny"]["pixels_sha256"],
                "swap_clean": result["swap_clean"],
            }
            for result in results
        ],
    }
    temporary = args.summary.with_name(f".{args.summary.name}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.summary)
    print(json.dumps({"summary": str(args.summary), "jobs": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
