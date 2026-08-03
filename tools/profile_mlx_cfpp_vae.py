"""Profile causal VAE stages and execution modes on frozen CF++ latents."""

# mypy: ignore-errors

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from benchmark_mlx_cfpp import (
    _configure_allocator,
    _load_model,
    _memory_snapshot,
    _one_trial,
    _preflight,
    _sha256,
    _validate_assets,
    _validate_fixture,
)
from mlx_video.models.wan_2.utils import load_vae_decoder

from liveframe.fixtures import load_fixture
from liveframe.mlx_cfpp_benchmark import CFPPBenchmarkConfig, require_swap_stable
from liveframe.mlx_parity import _metrics, _mlx_tensor
from liveframe.mlx_rolling_parity import _runtime_provenance
from liveframe.mlx_vae_decode import _causal_decode_frame, _decoder_cache_slots
from liveframe.mlx_vae_profile import (
    profiled_causal_decode_frame,
    summarize_stage_rows,
)

MODES = ("eager", "stage-eager", "compiled-steady", "async-deferred")


def _pixels(decoded: mx.array) -> np.ndarray:
    value = np.array(decoded[0].transpose(1, 2, 3, 0), dtype=np.float32)
    return np.clip((value + 1.0) * 127.5, 0, 255).astype(np.uint8)


def _pixel_digest(chunks: list[np.ndarray]) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in chunks:
        payload = chunk.tobytes(order="C")
        digest.update(payload)
        size += len(payload)
    return "sha256:" + digest.hexdigest(), size


def _joined_pixels(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks, axis=0)


def _decoded_video_digest(path: Path) -> tuple[str, int]:
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    return "sha256:" + hashlib.sha256(result.stdout).hexdigest(), len(result.stdout)


def _memory() -> dict[str, float]:
    return {
        "active_mlx_memory_gib": mx.get_active_memory() / 2**30,
        "cache_mlx_memory_gib": mx.get_cache_memory() / 2**30,
        "peak_mlx_memory_gib": mx.get_peak_memory() / 2**30,
    }


def _run_eager(vae: Any, latents: mx.array, *, stage_profile: bool) -> dict[str, Any]:
    cache: list[Any] = [None] * _decoder_cache_slots(vae.decoder)
    chunks = []
    chunk_rows = []
    stage_rows = []
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.perf_counter()
    for frame in range(latents.shape[1]):
        latent = latents[:, frame : frame + 1][None]
        build_started = time.perf_counter()
        if stage_profile:
            decoded, rows = profiled_causal_decode_frame(vae, latent, cache)
            stage_rows.extend({**row, "latent_frame": frame} for row in rows)
            build_seconds = sum(float(row["graph_build_seconds"]) for row in rows)
            eval_seconds = sum(float(row["eval_seconds"]) for row in rows)
        else:
            decoded = _causal_decode_frame(vae, latent, cache)
            build_seconds = time.perf_counter() - build_started
            eval_started = time.perf_counter()
            mx.eval(decoded)
            eval_seconds = time.perf_counter() - eval_started
        materialize_started = time.perf_counter()
        chunks.append(_pixels(decoded))
        materialization_seconds = time.perf_counter() - materialize_started
        chunk_rows.append(
            {
                "latent_frame": frame,
                "graph_build_seconds": build_seconds,
                "eval_seconds": eval_seconds,
                "materialization_seconds": materialization_seconds,
                **_memory(),
            }
        )
    digest, byte_count = _pixel_digest(chunks)
    return {
        "mode": "stage-eager" if stage_profile else "eager",
        "wall_seconds": time.perf_counter() - started,
        "pixel_sha256": digest,
        "pixel_bytes": byte_count,
        "_pixels": _joined_pixels(chunks),
        "chunks": chunk_rows,
        "stage_rows": stage_rows,
        "stage_summary": summarize_stage_rows(stage_rows) if stage_rows else [],
        **_memory(),
    }


def _run_compiled(vae: Any, latents: mx.array) -> dict[str, Any]:
    cache: list[Any] = [None] * _decoder_cache_slots(vae.decoder)
    chunks = []
    rows = []
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.perf_counter()

    def decode(value: mx.array) -> mx.array:
        return _causal_decode_frame(vae, value, cache)

    compiled = None
    for frame in range(latents.shape[1]):
        latent = latents[:, frame : frame + 1][None]
        call_started = time.perf_counter()
        if frame < 2:
            decoded = decode(latent)
            execution = "eager-cache-initialization"
        else:
            if compiled is None:
                compiled = mx.compile(decode, inputs=[vae.state, cache], outputs=cache)
            decoded = compiled(latent)
            execution = "compiled-steady"
        build_seconds = time.perf_counter() - call_started
        eval_started = time.perf_counter()
        mx.eval(decoded)
        eval_seconds = time.perf_counter() - eval_started
        materialize_started = time.perf_counter()
        chunks.append(_pixels(decoded))
        rows.append(
            {
                "latent_frame": frame,
                "execution": execution,
                "call_seconds": build_seconds,
                "eval_seconds": eval_seconds,
                "materialization_seconds": time.perf_counter() - materialize_started,
                **_memory(),
            }
        )
    digest, byte_count = _pixel_digest(chunks)
    return {
        "mode": "compiled-steady",
        "wall_seconds": time.perf_counter() - started,
        "pixel_sha256": digest,
        "pixel_bytes": byte_count,
        "_pixels": _joined_pixels(chunks),
        "chunks": rows,
        **_memory(),
    }


def _run_async_deferred(vae: Any, latents: mx.array) -> dict[str, Any]:
    cache: list[Any] = [None] * _decoder_cache_slots(vae.decoder)
    decoded_chunks = []
    rows = []
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.perf_counter()
    for frame in range(latents.shape[1]):
        latent = latents[:, frame : frame + 1][None]
        build_started = time.perf_counter()
        decoded = _causal_decode_frame(vae, latent, cache)
        build_seconds = time.perf_counter() - build_started
        dispatch_started = time.perf_counter()
        mx.async_eval(decoded)
        dispatch_seconds = time.perf_counter() - dispatch_started
        decoded_chunks.append(decoded)
        rows.append(
            {
                "latent_frame": frame,
                "graph_build_seconds": build_seconds,
                "async_dispatch_seconds": dispatch_seconds,
                **_memory(),
            }
        )
    enqueue_seconds = time.perf_counter() - started
    materialize_started = time.perf_counter()
    chunks = [_pixels(decoded) for decoded in decoded_chunks]
    materialization_seconds = time.perf_counter() - materialize_started
    digest, byte_count = _pixel_digest(chunks)
    return {
        "mode": "async-deferred",
        "wall_seconds": time.perf_counter() - started,
        "enqueue_seconds": enqueue_seconds,
        "deferred_materialization_seconds": materialization_seconds,
        "pixel_sha256": digest,
        "pixel_bytes": byte_count,
        "_pixels": _joined_pixels(chunks),
        "chunks": rows,
        **_memory(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--max-start-swap-mib", type=float, default=128.0)
    parser.add_argument("--cache-limit-gib", type=float, default=1.0)
    parser.add_argument("--allow-output-drift", action="store_true")
    args = parser.parse_args()

    config = CFPPBenchmarkConfig(
        max_start_swap_mib=args.max_start_swap_mib,
        cache_limit_gib=args.cache_limit_gib,
    )
    preflight = _preflight(config)
    control_config = CFPPBenchmarkConfig(
        max_start_swap_mib=args.max_start_swap_mib,
        cache_limit_gib=1.0,
    )
    control_allocator = _configure_allocator(control_config)
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    if not baseline.get("claim_eligible"):
        raise ValueError("VAE profiler requires the claim-eligible CF++ baseline")
    baseline_video = Path(baseline["video"]["path"])
    if _sha256(baseline_video) != baseline["video"]["sha256"]:
        raise ValueError("authoritative baseline video checksum mismatch")
    baseline_decoded_digest, baseline_decoded_bytes = _decoded_video_digest(
        baseline_video
    )

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
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        control_video = Path(temporary) / "control.mp4"
        control, latents = _one_trial(
            model,
            vae,
            embedded_context,
            cross_kv,
            fixture,
            tensors,
            config,
            control_video,
        )
        require_swap_stable(
            swap_delta_mib=float(control["swap_delta_mib"]), config=control_config
        )
        control_digest = _sha256(control_video)
        control_decoded_digest, control_decoded_bytes = _decoded_video_digest(
            control_video
        )
        if (
            control_decoded_digest != baseline_decoded_digest
            or control_decoded_bytes != baseline_decoded_bytes
        ):
            raise RuntimeError(
                "control rerun did not reproduce baseline decoded RGB frames"
            )

        del model, prompt, embedded_context, cross_kv
        gc.collect()
        mx.clear_cache()
        allocator = _configure_allocator(config)

        results = []
        for mode in args.modes:
            print(f"profiling VAE mode: {mode}", flush=True)
            memory_before = _memory_snapshot()
            if mode == "eager":
                result = _run_eager(vae, latents, stage_profile=False)
            elif mode == "stage-eager":
                result = _run_eager(vae, latents, stage_profile=True)
            elif mode == "compiled-steady":
                result = _run_compiled(vae, latents)
            else:
                result = _run_async_deferred(vae, latents)
            memory_after = _memory_snapshot()
            result["system_memory_before"] = memory_before
            result["system_memory_after"] = memory_after
            result["swap_delta_mib"] = (
                float(memory_after["swap"]["used_mib"])
                - float(memory_before["swap"]["used_mib"])
            )
            require_swap_stable(
                swap_delta_mib=float(result["swap_delta_mib"]), config=config
            )
            results.append(result)

    eager = next((result for result in results if result["mode"] == "eager"), None)
    if eager is None:
        raise ValueError("profiler modes must include eager as the raw-output identity anchor")
    eager_pixels = eager.pop("_pixels")
    for result in results:
        candidate_pixels = (
            eager_pixels if result is eager else result.pop("_pixels")
        )
        result["identical_to_eager"] = result["pixel_sha256"] == eager["pixel_sha256"]
        result["pixel_metrics_vs_eager"] = _metrics(eager_pixels, candidate_pixels)
        if not result["identical_to_eager"] and not args.allow_output_drift:
            raise RuntimeError(f"{result['mode']} output differs from eager")

    report = {
        "schema": "liveframe.mlx-cfpp-vae-profile.v1",
        "claim_eligible": False,
        "purpose": "diagnostic stage and execution-mode profile",
        "exact_output_required": not args.allow_output_drift,
        "baseline": {
            "report": str(args.baseline_report),
            "report_sha256": _sha256(args.baseline_report),
            "video": str(baseline_video),
            "video_sha256": baseline["video"]["sha256"],
            "control_video_sha256": control_digest,
            "encoded_video_bytes_identical": (
                control_digest == baseline["video"]["sha256"]
            ),
            "decoded_rgb_sha256": baseline_decoded_digest,
            "decoded_rgb_bytes": baseline_decoded_bytes,
            "control_reproduced_baseline_decoded_rgb": True,
        },
        "fixture": {"id": fixture.fixture_id, **assets},
        "runtime": _runtime_provenance(),
        "control_allocator": control_allocator,
        "allocator": allocator,
        "preflight": preflight,
        "control_trial": control,
        "raw_pixel_anchor_sha256": eager["pixel_sha256"],
        "results": results,
    }
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_output.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "modes": {
                    result["mode"]: result["wall_seconds"] for result in results
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()