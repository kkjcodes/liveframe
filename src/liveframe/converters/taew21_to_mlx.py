"""Fail-closed TAEW2.1 decoder conversion from PyTorch OIHW to MLX OHWI."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from liveframe.manifests import file_sha256


class TAEW21ConversionError(ValueError):
    """Raised when the pinned TAEW2.1 decoder inventory cannot be mapped exactly."""


@dataclass(frozen=True, slots=True)
class TAEW21Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class TAEW21Mapping:
    source: TAEW21Tensor
    target_name: str
    target_shape: tuple[int, ...]
    transform: str


_MEMORY_LAYERS = (3, 4, 5, 9, 10, 11, 15, 16, 17)
_TGROW_LAYERS = (7, 13, 19)
_PLAIN_CONV_LAYERS = (1, 8, 14, 20, 22)


def expected_decoder_keys() -> frozenset[str]:
    keys = {"decoder.1.weight", "decoder.1.bias", "decoder.22.weight", "decoder.22.bias"}
    for layer in _MEMORY_LAYERS:
        for convolution in (0, 2, 4):
            keys.add(f"decoder.{layer}.conv.{convolution}.weight")
            keys.add(f"decoder.{layer}.conv.{convolution}.bias")
    for layer in _TGROW_LAYERS:
        keys.add(f"decoder.{layer}.conv.weight")
    for layer in _PLAIN_CONV_LAYERS:
        keys.add(f"decoder.{layer}.weight")
    return frozenset(keys)


EXPECTED_DECODER_KEYS = expected_decoder_keys()


def plan_taew21_conversion(
    inventory: Mapping[str, TAEW21Tensor],
) -> tuple[TAEW21Mapping, ...]:
    """Require and map the exact pinned base TAEW2.1 decoder inventory."""

    observed = {name for name in inventory if name.startswith("decoder.")}
    missing = sorted(EXPECTED_DECODER_KEYS - observed)
    unexpected = sorted(observed - EXPECTED_DECODER_KEYS)
    if missing or unexpected:
        raise TAEW21ConversionError(
            f"TAEW2.1 decoder inventory mismatch: missing={missing}, unexpected={unexpected}"
        )
    rules: list[TAEW21Mapping] = []
    for name in sorted(EXPECTED_DECODER_KEYS):
        tensor = inventory[name]
        if tensor.name != name:
            raise TAEW21ConversionError(f"inventory key/name mismatch for {name}")
        if tensor.dtype not in {"float16", "float32"}:
            raise TAEW21ConversionError(f"unsupported TAEW2.1 tensor dtype for {name}")
        if name.endswith(".weight"):
            if len(tensor.shape) != 4:
                raise TAEW21ConversionError(f"TAEW2.1 convolution weight {name} must be OIHW")
            out_channels, in_channels, height, width = tensor.shape
            target_shape = (out_channels, height, width, in_channels)
            transform = "oihw-to-ohwi"
        else:
            if len(tensor.shape) != 1:
                raise TAEW21ConversionError(f"TAEW2.1 bias {name} must be one-dimensional")
            target_shape = tensor.shape
            transform = "identity"
        rules.append(TAEW21Mapping(tensor, name, target_shape, transform))
    return tuple(rules)


def _array_sha256(value: Any) -> str:
    contiguous = value if value.flags.c_contiguous else value.copy(order="C")
    return "sha256:" + hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def execute_taew21_conversion(
    checkpoint: Path,
    output_dir: Path,
    *,
    expected_source_sha256: str,
    source_repository: str,
    source_revision: str,
    target_dtype: str = "float16",
) -> Path:
    """Convert only the base TAEW2.1 decoder and write a digest-bound manifest."""

    if target_dtype not in {"float16", "float32"}:
        raise TAEW21ConversionError(f"unsupported TAEW2.1 target dtype: {target_dtype}")
    actual_source = file_sha256(checkpoint)
    if actual_source != expected_source_sha256:
        raise TAEW21ConversionError(
            f"TAEW2.1 source checksum mismatch: expected {expected_source_sha256}, "
            f"computed {actual_source}"
        )
    try:
        from safetensors import safe_open
        from safetensors.numpy import save_file
    except ImportError as error:
        raise TAEW21ConversionError("TAEW2.1 conversion requires safetensors") from error

    with safe_open(checkpoint, framework="np") as handle:
        import numpy as np

        numpy_dtype = {"float16": np.float16, "float32": np.float32}[target_dtype]
        inventory = {
            name: TAEW21Tensor(
                name,
                tuple(handle.get_slice(name).get_shape()),
                str(handle.get_tensor(name).dtype),
            )
            for name in handle.keys()  # noqa: SIM118 - safe_open is not iterable
        }
        rules = plan_taew21_conversion(inventory)
        converted: dict[str, Any] = {}
        mappings: list[dict[str, Any]] = []
        for rule in rules:
            source = handle.get_tensor(rule.source.name)
            transformed = (
                source.transpose(0, 2, 3, 1)
                if rule.transform == "oihw-to-ohwi"
                else source
            )
            target = transformed.astype(numpy_dtype, copy=True, order="C")
            if tuple(target.shape) != rule.target_shape:
                raise TAEW21ConversionError(f"converted shape mismatch for {rule.source.name}")
            converted[rule.target_name] = target
            mappings.append(
                {
                    "source": rule.source.name,
                    "source_shape": list(rule.source.shape),
                    "source_sha256": _array_sha256(source),
                    "source_dtype": str(source.dtype),
                    "target": rule.target_name,
                    "target_shape": list(rule.target_shape),
                    "target_sha256": _array_sha256(target),
                    "dtype": str(target.dtype),
                    "transform": rule.transform,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "taew2_1-mlx.safetensors"
    temporary_weights = output_dir / ".taew2_1-mlx.safetensors.tmp"
    save_file(converted, temporary_weights)
    os.replace(temporary_weights, weights_path)
    manifest = {
        "schema": "liveframe.taew21-mlx-conversion.v1",
        "source": {
            "repository": source_repository,
            "revision": source_revision,
            "artifact": checkpoint.name,
            "sha256": actual_source,
            "format": "safetensors",
        },
        "target": {
            "artifact": weights_path.name,
            "sha256": file_sha256(weights_path),
            "kernel_layout": "OHWI",
            "target_dtype": target_dtype,
            "tensor_count": len(converted),
        },
        "ignored_source_prefixes": ["encoder."],
        "mappings": mappings,
    }
    manifest_path = output_dir / "taew21-conversion.json"
    temporary_manifest = output_dir / ".taew21-conversion.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest_path