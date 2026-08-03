"""Shared serialized stochastic inputs for CUDA/MLX holdout identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from liveframe.perceptual_policy import canonical_digest
from liveframe.quality_artifacts import StochasticWriter, atomic_json
from liveframe.quality_capture import sha256_file
from liveframe.rolling_forcing import RollingForcingSchedule

STOCHASTIC_CONTRACT = "liveframe.shared-rolling-noise.v1"
STOCHASTIC_INDEX_SCHEMA = "liveframe.shared-rolling-noise-index.v1"
WARPED_STEPS = {
    1000: 1000.0,
    750: 937.5,
    500: 833.3333129882812,
    250: 625.0,
}
BLOCK_FRAMES = 3
LATENT_SHAPE = (16, 60, 104)


def plan_directory(root: Path, *, seed: int, latent_frames: int) -> Path:
    return root / f"seed-{seed}-latent-{latent_frames}"


def transition_name(
    window_index: int,
    block_id: int,
    configured_step: int,
    next_step: int,
) -> str:
    return (
        f"window-{window_index:04d}-block-{block_id:04d}-"
        f"{WARPED_STEPS[configured_step]}-to-{WARPED_STEPS[next_step]}"
    )


def create_stochastic_plan(output_dir: Path, *, seed: int, latent_frames: int) -> list[Path]:
    """Generate one immutable input stream reused across every prompt and backend."""

    import torch

    schedule = RollingForcingSchedule(latent_frames=latent_frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = StochasticWriter(output_dir)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    try:
        initial = torch.randn(
            (1, latent_frames, *LATENT_SHAPE),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        ).to(torch.bfloat16)
        writer.append("initial_noise", initial)
        for window in schedule.windows():
            for offset, configured_step in enumerate(window.configured_timesteps):
                if configured_step == schedule.configured_steps[-1]:
                    continue
                next_step = schedule.configured_steps[
                    schedule.configured_steps.index(configured_step) + 1
                ]
                block_id = window.start_block + offset
                noise = torch.randn(
                    (1, BLOCK_FRAMES, *LATENT_SHAPE),
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                ).to(torch.bfloat16)
                writer.append(
                    transition_name(
                        window.index, block_id, configured_step, next_step
                    ),
                    noise,
                )
        return writer.finish(
            seed=seed,
            contract=STOCHASTIC_CONTRACT,
            latent_frames=latent_frames,
        )
    except BaseException:
        writer.abort()
        raise


def create_stochastic_plan_set(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Publish or validate the complete manifest-bound shared input set."""

    workloads = sorted(
        {
            (int(job["seed"]), int(job["latent_frames"]))
            for job in manifest["jobs"]
            if job.get("split") == "holdout"
        }
    )
    if len(workloads) != 8:
        raise ValueError(f"shared stochastic plan set requires 8 workloads, got {len(workloads)}")
    root.mkdir(parents=True, exist_ok=True)
    plans: list[dict[str, Any]] = []
    for seed, latent_frames in workloads:
        directory = plan_directory(root, seed=seed, latent_frames=latent_frames)
        payload_path = directory / "stochastic.bf16"
        metadata_path = directory / "stochastic.bf16.json"
        existing = (payload_path.exists(), metadata_path.exists())
        if existing == (False, False):
            create_stochastic_plan(directory, seed=seed, latent_frames=latent_frames)
        elif existing != (True, True):
            raise ValueError(f"incomplete stochastic plan: {directory}")
        StochasticPlanReader(metadata_path, seed=seed, latent_frames=latent_frames)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        plans.append(
            {
                "seed": seed,
                "latent_frames": latent_frames,
                "directory": directory.name,
                "payload_bytes": payload_path.stat().st_size,
                "payload_sha256": metadata["payload_sha256"],
                "metadata_sha256": sha256_file(metadata_path),
                "tensor_count": metadata["tensor_count"],
            }
        )
    index: dict[str, Any] = {
        "schema": STOCHASTIC_INDEX_SCHEMA,
        "contract": STOCHASTIC_CONTRACT,
        "protocol_digest": manifest["protocol_digest"],
        "manifest_digest": manifest["manifest_digest"],
        "plan_count": len(plans),
        "plans": plans,
    }
    index["index_digest"] = canonical_digest(index)
    index_path = root / "index.json"
    if index_path.exists():
        existing_index = json.loads(index_path.read_text(encoding="utf-8"))
        if existing_index != index:
            raise FileExistsError(f"refusing to replace stochastic plan index: {index_path}")
    else:
        atomic_json(index_path, index)
    return index


class StochasticPlanReader:
    """Consume every validated tensor exactly once and in declared order."""

    def __init__(self, metadata_path: Path, *, seed: int, latent_frames: int) -> None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("stochastic metadata must be an object")
        if metadata.get("schema") != "liveframe.serialized-stochastic-tensors.v1":
            raise ValueError("unsupported stochastic artifact schema")
        if metadata.get("contract") != STOCHASTIC_CONTRACT:
            raise ValueError("stochastic artifact contract mismatch")
        if metadata.get("seed") != seed or metadata.get("latent_frames") != latent_frames:
            raise ValueError("stochastic artifact workload mismatch")
        tensors = metadata.get("tensors")
        if not isinstance(tensors, list) or metadata.get("tensor_count") != len(tensors):
            raise ValueError("stochastic tensor table mismatch")
        self.payload_path = metadata_path.with_name("stochastic.bf16")
        if sha256_file(self.payload_path) != metadata.get("payload_sha256"):
            raise ValueError("stochastic payload checksum mismatch")
        self.entries = tensors
        self.index = 0
        self._validate_entries()

    def _validate_entries(self) -> None:
        expected_offset = 0
        payload_bytes = self.payload_path.stat().st_size
        with self.payload_path.open("rb") as stream:
            for entry in self.entries:
                if not isinstance(entry, dict):
                    raise ValueError("stochastic tensor entry must be an object")
                if entry.get("dtype") != "bfloat16":
                    raise ValueError("stochastic tensors must be BF16")
                if entry.get("offset_bytes") != expected_offset:
                    raise ValueError("stochastic tensor offsets are not contiguous")
                size = int(entry.get("bytes", -1))
                shape = entry.get("shape")
                if not isinstance(shape, list) or size != int(np.prod(shape)) * 2:
                    raise ValueError("stochastic tensor byte size mismatch")
                payload = stream.read(size)
                digest = "sha256:" + hashlib.sha256(payload).hexdigest()
                if len(payload) != size or digest != entry.get("sha256"):
                    raise ValueError("stochastic tensor checksum mismatch")
                expected_offset += size
        if expected_offset != payload_bytes:
            raise ValueError("stochastic payload has undeclared bytes")

    def _next_payload(self, name: str, shape: Sequence[int]) -> bytes:
        if self.index >= len(self.entries):
            raise ValueError(f"stochastic plan exhausted before {name}")
        entry = self.entries[self.index]
        if entry.get("name") != name or entry.get("shape") != list(shape):
            raise ValueError(f"stochastic tensor order mismatch at {name}")
        with self.payload_path.open("rb") as stream:
            stream.seek(int(entry["offset_bytes"]))
            payload = stream.read(int(entry["bytes"]))
        self.index += 1
        return payload

    def next_torch(self, name: str, shape: Sequence[int], *, device: str) -> Any:
        import torch

        payload = self._next_payload(name, shape)
        raw = np.frombuffer(payload, dtype="<u2").copy()
        tensor = torch.from_numpy(raw).view(torch.bfloat16).reshape(tuple(shape))
        return tensor.to(device=device)

    def next_mlx(self, name: str, shape: Sequence[int]) -> Any:
        import mlx.core as mx  # type: ignore[import-not-found]

        payload = self._next_payload(name, shape)
        raw = np.frombuffer(payload, dtype="<u2").copy()
        return mx.array(raw).view(mx.bfloat16).reshape(tuple(shape))

    def require_consumed(self) -> None:
        if self.index != len(self.entries):
            raise ValueError(
                f"stochastic plan has {len(self.entries) - self.index} unconsumed tensors"
            )