"""Append batch and incremental Wan VAE references to a full H100 fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bfloat16(path: Path, shape: list[int]) -> torch.Tensor:
    raw = np.fromfile(path, dtype="<u2").copy()
    return torch.from_numpy(raw).view(torch.bfloat16).reshape(shape)


def _write_tensor(root: Path, name: str, role: str, value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().to(torch.float16).contiguous()
    payload = tensor.numpy().astype("<f2", copy=False).tobytes()
    filename = f"{name}.bin"
    (root / filename).write_bytes(payload)
    return {
        "name": name,
        "role": role,
        "file": filename,
        "shape": list(tensor.shape),
        "dtype": "float16",
        "byte_order": "little",
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "scheduler_transition": None,
    }


def capture_vae(fixture_dir: Path) -> None:
    manifest_path = fixture_dir / "fixture.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fixture_id") not in {
        "cfpp-1step-native-21block-h100",
        "cfpp-2step-native-21block-h100",
    }:
        raise ValueError("VAE capture requires the complete 21-block H100 fixture")
    if any(str(row.get("name", "")).startswith("vae.") for row in manifest["tensors"]):
        raise ValueError("fixture already contains VAE tensors")

    output_entry = next(row for row in manifest["tensors"] if row["name"] == "output_latents")
    latents = _load_bfloat16(
        fixture_dir / output_entry["file"], output_entry["shape"]
    ).cuda()

    sys.path.insert(0, str(Path.cwd()))
    from utils.wan_wrapper import WanVAEWrapper  # type: ignore[import-not-found]

    vae_path = Path("wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    vae = WanVAEWrapper().to(device="cuda", dtype=torch.bfloat16).eval()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.cuda.synchronize()
    batch_started = time.time()
    batch_pixels = vae.decode_to_pixel(latents, use_cache=False)
    torch.cuda.synchronize()
    batch_seconds = time.time() - batch_started

    vae.model.clear_cache()
    chunks = []
    chunk_shapes = []
    chunk_seconds = []
    entries = []
    for block in range(latents.shape[1]):
        torch.cuda.synchronize()
        started = time.time()
        chunk = vae.decode_to_pixel(latents[:, block : block + 1], use_cache=True)
        torch.cuda.synchronize()
        elapsed = time.time() - started
        chunks.append(chunk)
        chunk_shapes.append(list(chunk.shape))
        chunk_seconds.append(elapsed)
        entries.append(
            _write_tensor(
                fixture_dir,
                f"vae.stream.block.{block:02d}",
                "decoded_pixel_chunk",
                chunk,
            )
        )
        print(
            f"decoded latent block {block + 1}/{latents.shape[1]}: "
            f"{chunk.shape[1]} pixel frames in {elapsed:.3f}s",
            flush=True,
        )

    streamed_pixels = torch.cat(chunks, dim=1)
    if streamed_pixels.shape != batch_pixels.shape:
        raise ValueError(
            f"streamed VAE shape {streamed_pixels.shape} != batch shape {batch_pixels.shape}"
        )
    delta = (streamed_pixels.float() - batch_pixels.float()).abs()
    entries.append(_write_tensor(fixture_dir, "vae.batch_pixels", "decoded_pixels", batch_pixels))
    entries.append(
        _write_tensor(fixture_dir, "vae.stream_pixels", "decoded_pixels", streamed_pixels)
    )

    manifest["tensors"].extend(entries)
    manifest["metadata"]["vae"] = {
        "checkpoint_sha256": "sha256:" + _sha256(vae_path),
        "dtype": "bfloat16",
        "batch_output_shape": list(batch_pixels.shape),
        "stream_output_shape": list(streamed_pixels.shape),
        "stream_chunk_shapes": chunk_shapes,
        "batch_decode_seconds": batch_seconds,
        "stream_chunk_seconds": chunk_seconds,
        "stream_total_seconds": sum(chunk_seconds),
        "batch_stream_max_abs": float(delta.max().item()),
        "batch_stream_mean_abs": float(delta.mean().item()),
        "tf32": False,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(json.dumps(manifest["metadata"]["vae"], sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    capture_vae(args.fixture)


if __name__ == "__main__":
    main()
