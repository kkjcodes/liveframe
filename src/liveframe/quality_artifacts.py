"""Atomic lossless artifacts for quality generation and stochastic replay."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from liveframe.quality_capture import sha256_file


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def tensor_payload(value: Any) -> tuple[bytes, str]:
    import torch

    tensor = value.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes(), "bfloat16"
    if tensor.dtype == torch.uint8:
        return tensor.numpy().tobytes(), "uint8"
    raise TypeError(f"unsupported quality artifact dtype: {tensor.dtype}")


def write_tensor(path: Path, value: Any, *, layout: str, schema: str) -> list[Path]:
    payload, dtype = tensor_payload(value)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    metadata = {
        "schema": schema,
        "dtype": dtype,
        "layout": layout,
        "shape": list(value.shape),
        "sha256": sha256_file(path),
    }
    metadata_path = path.with_suffix(path.suffix + ".json")
    atomic_json(metadata_path, metadata)
    return [path, metadata_path]


class StochasticWriter:
    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "stochastic.bf16"
        self.temporary = output_dir / ".stochastic.bf16.tmp"
        self.stream = self.temporary.open("wb")
        self.entries: list[dict[str, Any]] = []

    def append(self, name: str, value: Any) -> None:
        payload, dtype = tensor_payload(value)
        if dtype != "bfloat16":
            raise TypeError("stochastic tensors must be BF16")
        offset = self.stream.tell()
        self.stream.write(payload)
        self.entries.append(
            {
                "name": name,
                "offset_bytes": offset,
                "bytes": len(payload),
                "shape": list(value.shape),
                "dtype": dtype,
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        )

    def finish(
        self,
        *,
        seed: int,
        contract: str | None = None,
        latent_frames: int | None = None,
    ) -> list[Path]:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        os.replace(self.temporary, self.path)
        metadata_path = self.path.with_suffix(self.path.suffix + ".json")
        metadata = {
            "schema": "liveframe.serialized-stochastic-tensors.v1",
            "seed": seed,
            "tensor_count": len(self.entries),
            "payload_sha256": sha256_file(self.path),
            "tensors": self.entries,
        }
        if contract is not None:
            metadata["contract"] = contract
        if latent_frames is not None:
            metadata["latent_frames"] = latent_frames
        atomic_json(metadata_path, metadata)
        return [self.path, metadata_path]

    def abort(self) -> None:
        if not self.stream.closed:
            self.stream.close()
        self.temporary.unlink(missing_ok=True)
