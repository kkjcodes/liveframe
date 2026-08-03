"""Replay a specialized H100 Rolling Forcing fixture through MLX."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_video.models.wan_2.config import WanModelConfig
from mlx_video.models.wan_2.utils import load_wan_model

from liveframe.backends.mlx_causal import CausalWanForward, RollingForcingKVCache
from liveframe.fixtures import GoldenTensor, load_fixture
from liveframe.mlx_parity import _metrics, _mlx_tensor, _numpy_tensor
from liveframe.rolling_forcing import RollingForcingSchedule

TOKENS_PER_FRAME = 1560
BLOCK_FRAMES = 3
SELECTED_CACHE_LAYERS = (0, 15, 29)
FINAL_COSINE_CANARY = 0.999
FINAL_RMSE_LIMIT = 0.035
FINAL_MAX_ABS_LIMIT = 0.25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _runtime_provenance() -> dict[str, Any]:
    device_info = mx.device_info() if mx.metal.is_available() else {}
    try:
        mlx_metal_version = importlib.metadata.version("mlx-metal")
    except importlib.metadata.PackageNotFoundError:
        mlx_metal_version = None
    return {
        "runtime": "mlx-metal" if mx.metal.is_available() else "mlx-cpu",
        "mlx_version": mx.__version__,
        "mlx_metal_version": mlx_metal_version,
        "default_device": str(mx.default_device()),
        "metal_available": mx.metal.is_available(),
        "device": device_info,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python_version": platform.python_version(),
    }


def _write_candidate_latents(
    path: Path,
    value: mx.array,
    *,
    fixture_id: str,
    fixture_manifest: Path,
    model_path: Path,
) -> dict[str, Any]:
    """Persist exact MLX BF16 output in fixture-compatible video layout."""

    video = value[None].transpose(0, 2, 1, 3, 4).astype(mx.bfloat16)
    mx.eval(video)
    raw = np.array(video.view(mx.uint16), dtype="<u2")
    payload = raw.tobytes(order="C")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    metadata = {
        "schema": "liveframe.candidate-latents.v1",
        "fixture_id": fixture_id,
        "fixture_manifest_sha256": _sha256(fixture_manifest),
        "model_sha256": _sha256(model_path),
        "file": path.name,
        "shape": list(video.shape),
        "layout": "B,F,C,H,W",
        "dtype": "bfloat16",
        "byte_order": "little",
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "runtime": _runtime_provenance(),
    }
    metadata_path = path.with_suffix(path.suffix + ".json")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_metadata.replace(metadata_path)
    return {**metadata, "path": str(path), "metadata_path": str(metadata_path)}


def _load_model(model_dir: Path) -> Any:
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    allowed = WanModelConfig.__dataclass_fields__.keys()
    config = WanModelConfig(**{key: value for key, value in raw.items() if key in allowed})
    return load_wan_model(model_dir / "model.safetensors", config)


def run_rolling_parity(
    model_dir: Path, fixture_dir: Path, *, candidate_output: Path | None = None
) -> dict[str, Any]:
    fixture = load_fixture(fixture_dir)
    metadata = fixture.metadata
    if metadata.get("algorithm") != "rolling_forcing_long_video":
        raise ValueError("fixture is not a specialized Rolling Forcing capture")
    contract = {
        "context_noise": 0,
        "local_attention": True,
        "local_attn_frames": 21,
        "cache_frames": 24,
        "anchor_frames": 3,
        "block_frames": 3,
    }
    for name, expected in contract.items():
        if metadata.get(name) != expected:
            raise ValueError(f"fixture metadata {name} must be {expected!r}")
    latent_frames = int(metadata.get("latent_frames", 0))
    schedule = RollingForcingSchedule(latent_frames=latent_frames, context_noise=0)
    tensors: dict[str, GoldenTensor] = {row.name: row for row in fixture.tensors}
    model = _load_model(model_dir)
    prompt = _mlx_tensor(fixture, tensors["prompt_embedding"])
    embedded_context = model.embed_text([prompt[0]])
    cross_kv = model.prepare_cross_kv(embedded_context)
    mx.eval(embedded_context, cross_kv)
    causal = CausalWanForward(model, rolling_forcing=True)
    initial = _mlx_tensor(fixture, tensors["initial_noise"]).transpose(0, 2, 1, 3, 4)[0]
    initial_blocks = [initial[:, i : i + BLOCK_FRAMES] for i in range(0, latent_frames, 3)]
    noisy_blocks = [mx.zeros_like(row) for row in initial_blocks]
    generated: list[mx.array | None] = [None] * schedule.block_count
    comparisons: dict[str, Any] = {}
    drift_curve = []
    raw_probes = metadata.get("cache_probe_indices")
    if not isinstance(raw_probes, list) or len(raw_probes) != schedule.window_count:
        raise ValueError("fixture must record cache probes for every rolling window")

    def compare(name: str, value: mx.array, *, video: bool = False) -> None:
        mx.eval(value)
        candidate = value.astype(mx.float32)
        if video:
            candidate = candidate.transpose(0, 2, 1, 3, 4)
        comparisons[name] = _metrics(
            _numpy_tensor(fixture, tensors[name]), __import__("numpy").array(candidate)
        )

    for window in schedule.windows():
        block_ids = list(range(window.start_block, window.end_block + 1))
        if window.block_count == len(schedule.configured_steps) or window.start_block == 0:
            blocks = [noisy_blocks[index] for index in block_ids[:-1]] + [
                initial_blocks[block_ids[-1]]
            ]
        else:
            blocks = [noisy_blocks[index] for index in block_ids]
        model_input = mx.concatenate(blocks, axis=1).astype(mx.bfloat16)
        prefix = f"window.{window.index:03d}"
        compare(f"{prefix}.model_input", model_input[None], video=True)
        timestep = _mlx_tensor(fixture, tensors[f"{prefix}.timestep"])
        flow = causal(
            [model_input],
            timestep=timestep,
            embedded_context=embedded_context,
            cross_kv_caches=cross_kv,
            current_start=window.start_frame * TOKENS_PER_FRAME,
        )[0]
        sigma = (timestep[0] / 1000.0).reshape(1, -1, 1, 1)
        prediction = model_input.astype(mx.float32) - sigma * flow
        mx.eval(flow, prediction)
        compare(f"{prefix}.flow_prediction", flow[None], video=True)
        compare(f"{prefix}.x0_prediction", prediction[None], video=True)

        for offset, configured_step in enumerate(window.configured_timesteps):
            block_id = window.start_block + offset
            block_prediction = prediction[:, offset * 3 : (offset + 1) * 3]
            if configured_step == schedule.configured_steps[-1]:
                generated[block_id] = block_prediction.astype(mx.bfloat16)
                continue
            next_step = schedule.configured_steps[
                schedule.configured_steps.index(configured_step) + 1
            ]
            noise = _mlx_tensor(
                fixture, tensors[f"{prefix}.block.{offset:02d}.scheduler_noise"]
            ).transpose(0, 2, 1, 3, 4)[0]
            warped_steps = dict(
                zip(schedule.configured_steps, metadata["warped_steps"], strict=True)
            )
            next_sigma = float(warped_steps[next_step]) / 1000.0
            noisy_blocks[block_id] = (
                (1 - next_sigma) * block_prediction.astype(mx.float32)
                + next_sigma * noise.astype(mx.float32)
            ).astype(mx.bfloat16)
            mx.eval(noisy_blocks[block_id])

        refresh_input = prediction[:, :3].astype(mx.bfloat16)
        refresh_flow = causal(
            [refresh_input],
            timestep=mx.zeros((1, 3)),
            embedded_context=embedded_context,
            cross_kv_caches=cross_kv,
            current_start=window.start_frame * TOKENS_PER_FRAME,
            updating_cache=True,
        )[0]
        compare(f"{prefix}.refresh.flow_prediction", refresh_flow[None], video=True)
        compare(f"{prefix}.refresh.x0_prediction", refresh_input[None], video=True)
        rolling_caches = []
        for cache in causal.caches:
            if not isinstance(cache, RollingForcingKVCache):
                raise TypeError("specialized parity requires rolling caches")
            rolling_caches.append(cache)
        compare(
            f"{prefix}.refresh.cache_global_end",
            mx.array([row.global_end for row in rolling_caches], dtype=mx.int64),
        )
        compare(
            f"{prefix}.refresh.cache_local_end",
            mx.array([row.local_end for row in rolling_caches], dtype=mx.int64),
        )
        probes_value = raw_probes[window.index]
        if not isinstance(probes_value, list):
            raise ValueError(f"invalid cache probes for window {window.index}")
        probes = [int(index) for index in probes_value]
        cache_metrics = []
        for layer in SELECTED_CACHE_LAYERS:
            cache = rolling_caches[layer]
            if cache.k is None or cache.v is None:
                raise RuntimeError("rolling cache unexpectedly empty")
            for kind, value in (("key", cache.k), ("value", cache.v)):
                name = f"{prefix}.refresh.layer.{layer:02d}.cache_{kind}_probe"
                compare(name, value[:, probes])
                cache_metrics.append(comparisons[name])
        drift_curve.append(
            {
                "window": window.index,
                "latent_frame_end": window.end_frame,
                "playback_seconds": (4 * (window.end_frame - 1) + 1) / 16.0,
                "x0_cosine": comparisons[f"{prefix}.x0_prediction"]["cosine"],
                "x0_rmse": comparisons[f"{prefix}.x0_prediction"]["rmse"],
                "minimum_cache_probe_cosine": min(row["cosine"] for row in cache_metrics),
                "maximum_cache_probe_abs": max(row["max_abs"] for row in cache_metrics),
            }
        )
        print(f"compared rolling window {window.index + 1}/{schedule.window_count}", flush=True)

    if any(row is None for row in generated):
        raise RuntimeError("rolling schedule did not finalize every latent block")
    output = mx.concatenate([row for row in generated if row is not None], axis=1)
    compare("output_latents", output[None], video=True)
    candidate_artifact = None
    if candidate_output is not None:
        candidate_artifact = _write_candidate_latents(
            candidate_output,
            output,
            fixture_id=fixture.fixture_id,
            fixture_manifest=fixture_dir / "fixture.json",
            model_path=model_dir / "model.safetensors",
        )
    failures = []
    for name, row in comparisons.items():
        if "cache_global_end" in name or "cache_local_end" in name:
            if row["max_abs"] != 0:
                failures.append(f"{name} cursor mismatch")
        elif "cache_value_probe" in name and row["cosine"] < 0.95:
            failures.append(f"{name} cosine {row['cosine']} < 0.95")
        elif (
            "cache_key_probe" in name or "prediction" in name
        ) and row["cosine"] < 0.99:
            failures.append(f"{name} cosine {row['cosine']} < 0.99")
    final = comparisons["output_latents"]
    if (
        final["cosine"] < FINAL_COSINE_CANARY
        or final["rmse"] > FINAL_RMSE_LIMIT
        or final["max_abs"] > FINAL_MAX_ABS_LIMIT
    ):
        failures.append("output_latents exceeds final parity bounds")
    checkpoints = {
        label: min(drift_curve, key=lambda row: abs(row["playback_seconds"] - seconds))
        for label, seconds in (("15s", 15.0), ("60s", 60.0))
        if drift_curve[-1]["playback_seconds"] >= seconds
    }
    return {
        "schema": "liveframe.rolling-parity-report.v2",
        "fixture_id": fixture.fixture_id,
        "fixture_manifest_sha256": _sha256(fixture_dir / "fixture.json"),
        "model_sha256": _sha256(model_dir / "model.safetensors"),
        "runtime": _runtime_provenance(),
        "local_attention": True,
        "anchor_frames": 3,
        "numerical_canary": {
            "output_latents_cosine_minimum": FINAL_COSINE_CANARY,
            "output_latents_rmse_maximum": FINAL_RMSE_LIMIT,
            "output_latents_max_abs_maximum": FINAL_MAX_ABS_LIMIT,
        },
        "candidate_latents": candidate_artifact,
        "compared_tensors": len(comparisons),
        "passed": not failures,
        "failures": failures,
        "drift_curve": drift_curve,
        "drift_checkpoints": checkpoints,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate-output", type=Path)
    args = parser.parse_args()
    report = run_rolling_parity(
        args.model, args.fixture, candidate_output=args.candidate_output
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "candidate_output": (
                        str(args.candidate_output) if args.candidate_output else None
                    ),
                    "compared_tensors": report["compared_tensors"],
                    "passed": report["passed"],
                    "failures": report["failures"],
                },
                sort_keys=True,
            )
        )
    else:
        print(text, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
