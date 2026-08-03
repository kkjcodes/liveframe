"""Decode rolling reference/candidate latents with one pinned MLX Wan VAE."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_video.models.wan_2.utils import load_vae_decoder

from liveframe.fixtures import load_fixture
from liveframe.mlx_parity import _mlx_tensor
from liveframe.mlx_rolling_parity import _runtime_provenance, _sha256


def _load_candidate(path: Path) -> tuple[mx.array, dict[str, Any]]:
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "liveframe.candidate-latents.v1":
        raise ValueError("unsupported candidate latent schema")
    if metadata.get("dtype") != "bfloat16" or metadata.get("layout") != "B,F,C,H,W":
        raise ValueError("candidate latents must be BF16 B,F,C,H,W")
    if _sha256(path) != metadata.get("sha256"):
        raise ValueError("candidate latent checksum mismatch")
    shape = tuple(int(value) for value in metadata["shape"])
    expected_bytes = int(np.prod(shape)) * 2
    if path.stat().st_size != expected_bytes:
        raise ValueError("candidate latent byte size mismatch")
    raw = mx.array(np.fromfile(path, dtype="<u2")).view(mx.bfloat16).reshape(shape)
    return raw, metadata


def _cache_conv(
    convolution: Any, x: mx.array, cache: list[Any], index: list[int]
) -> mx.array:
    slot = index[0]
    cache_x = x[:, :, -2:]
    if cache_x.shape[2] < 2 and cache[slot] is not None:
        cache_x = mx.concatenate([cache[slot][:, :, -1:], cache_x], axis=2)
    output = convolution(x, cache_x=cache[slot])
    cache[slot] = cache_x
    index[0] += 1
    return output


def _cached_upsample(layer: Any, x: mx.array, cache: list[Any], index: list[int]) -> mx.array:
    b, c, t, h, w = x.shape
    if layer.mode == "upsample3d":
        slot = index[0]
        previous = cache[slot]
        if previous is None:
            cache[slot] = "repeat"
            index[0] += 1
        else:
            cache_x = x[:, :, -2:]
            if cache_x.shape[2] < 2:
                prefix = (
                    mx.zeros_like(cache_x)
                    if isinstance(previous, str)
                    else previous[:, :, -1:]
                )
                cache_x = mx.concatenate([prefix, cache_x], axis=2)
            x = (
                layer.time_conv(x)
                if isinstance(previous, str)
                else layer.time_conv(x, cache_x=previous)
            )
            cache[slot] = cache_x
            index[0] += 1
            x = x.reshape(b, 2, c, t, h, w)
            x = mx.stack([x[:, 0], x[:, 1]], axis=3).reshape(b, c, t * 2, h, w)
            t *= 2
    x = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)
    x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
    x = layer.resample[1](x)
    return x.reshape(b, t, h * 2, w * 2, x.shape[-1]).transpose(0, 4, 1, 2, 3)


def _decoder_cache_slots(decoder: Any) -> int:
    slots = 2  # decoder conv1 and head conv
    slots += 2 * sum(hasattr(layer, "residual") for layer in decoder.middle)
    slots += 2 * sum(hasattr(layer, "residual") for layer in decoder.upsamples)
    slots += sum(getattr(layer, "mode", None) == "upsample3d" for layer in decoder.upsamples)
    return slots


def _causal_decode_frame(vae: Any, z: mx.array, cache: list[Any]) -> mx.array:
    mean = vae.mean.reshape(1, -1, 1, 1, 1)
    inv_std = vae.inv_std.reshape(1, -1, 1, 1, 1)
    x = vae.conv2(z / inv_std + mean)
    decoder = vae.decoder
    index = [0]
    x = _cache_conv(decoder.conv1, x, cache, index)
    for layer in decoder.middle:
        x = (
            layer(x, feat_cache=cache, feat_idx=index)
            if hasattr(layer, "residual")
            else layer(x)
        )
    for layer in decoder.upsamples:
        if hasattr(layer, "residual"):
            x = layer(x, feat_cache=cache, feat_idx=index)
        elif getattr(layer, "mode", "").startswith("upsample"):
            x = _cached_upsample(layer, x, cache, index)
        else:
            x = layer(x)
    x = nn.silu(decoder.head[0](x))
    x = _cache_conv(decoder.head[2], x, cache, index)
    if index[0] != len(cache):
        raise RuntimeError(f"VAE cache contract used {index[0]} of {len(cache)} slots")
    return mx.clip(x, -1, 1)


def _write_pixels(
    output: Path,
    latents: mx.array,
    *,
    source: dict[str, Any],
    vae_path: Path,
) -> dict[str, Any]:
    """Causally decode one latent at a time and stream uint8 THWC pixels."""

    if latents.ndim != 5 or latents.shape[0] != 1:
        raise ValueError("latents must have shape B,F,C,H,W with B=1")
    vae = load_vae_decoder(vae_path)
    cache: list[Any] = [None] * _decoder_cache_slots(vae.decoder)
    latent_frames = int(latents.shape[1])
    expected_frames = 4 * (latent_frames - 1) + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    emitted = 0
    chunk_shapes: list[list[int]] = []
    frame_shape: list[int] | None = None
    digest = hashlib.sha256()
    with temporary.open("wb") as stream:
        for frame in range(latent_frames):
            latent = latents[:, frame : frame + 1].transpose(0, 2, 1, 3, 4)
            decoded = _causal_decode_frame(vae, latent, cache)
            mx.eval(decoded)
            pixels = np.array(decoded[0].transpose(1, 2, 3, 0), dtype=np.float32)
            pixels = np.clip((pixels + 1.0) * 127.5, 0, 255).astype(np.uint8)
            if frame_shape is None:
                frame_shape = list(pixels.shape[1:])
            elif list(pixels.shape[1:]) != frame_shape:
                raise RuntimeError("decoder changed pixel dimensions between chunks")
            expected_chunk_frames = 1 if frame == 0 else 4
            if pixels.shape[0] != expected_chunk_frames:
                raise RuntimeError(
                    f"latent {frame} emitted {pixels.shape[0]} frames, "
                    f"expected {expected_chunk_frames}"
                )
            payload = pixels.tobytes(order="C")
            stream.write(payload)
            digest.update(payload)
            emitted += int(pixels.shape[0])
            chunk_shapes.append(list(pixels.shape))
            del decoded, pixels
            mx.clear_cache()
            print(
                f"decoded latent frame {frame + 1}/{latent_frames}; "
                f"emitted {emitted}/{expected_frames}",
                flush=True,
            )
    if emitted != expected_frames:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"decoder emitted {emitted}, expected {expected_frames} frames")
    if frame_shape is None:
        raise RuntimeError("decoder emitted no pixel chunks")
    temporary.replace(output)
    metadata = {
        "schema": "liveframe.decoded-pixels.v1",
        "source": source,
        "decoder": {
            "runtime": _runtime_provenance(),
            "vae_path": str(vae_path),
            "vae_sha256": _sha256(vae_path),
            "protocol": "upstream-faithful-causal-latent-frames.v1",
            "chunk_output_shapes": chunk_shapes,
            "first_emission_frames": 1,
            "steady_emission_frames": 4,
            "trimmed_frames": 0,
        },
        "file": output.name,
        "shape": [expected_frames, *frame_shape],
        "layout": "T,H,W,C",
        "dtype": "uint8",
        "sha256": "sha256:" + digest.hexdigest(),
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def decode_pair(
    fixture_dir: Path,
    candidate_path: Path,
    vae_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    fixture = load_fixture(fixture_dir)
    tensors = {tensor.name: tensor for tensor in fixture.tensors}
    reference = _mlx_tensor(fixture, tensors["output_latents"])
    candidate, candidate_metadata = _load_candidate(candidate_path)
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError("reference and candidate latent shapes differ")
    if candidate_metadata.get("fixture_id") != fixture.fixture_id:
        raise ValueError("candidate latents are bound to another fixture")
    fixture_sha = _sha256(fixture_dir / "fixture.json")
    if candidate_metadata.get("fixture_manifest_sha256") != fixture_sha:
        raise ValueError("candidate fixture manifest digest mismatch")
    reference_metadata = _write_pixels(
        output_dir / "reference.rgb8.bin",
        reference,
        source={
            "kind": "h100-reference-latents",
            "fixture_id": fixture.fixture_id,
            "fixture_manifest_sha256": fixture_sha,
            "latent_sha256": tensors["output_latents"].sha256,
        },
        vae_path=vae_path,
    )
    candidate_pixels_metadata = _write_pixels(
        output_dir / "candidate.rgb8.bin",
        candidate,
        source={
            "kind": "mlx-candidate-latents",
            "fixture_id": fixture.fixture_id,
            "latent_sha256": candidate_metadata["sha256"],
        },
        vae_path=vae_path,
    )
    report = {
        "schema": "liveframe.decode-pair-report.v1",
        "fixture_id": fixture.fixture_id,
        "same_decoder": True,
        "reference": reference_metadata,
        "candidate": candidate_pixels_metadata,
    }
    (output_dir / "decode-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = decode_pair(args.fixture, args.candidate, args.vae, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()