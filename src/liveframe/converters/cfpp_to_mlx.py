"""Pure metadata planning for CF++ PyTorch-to-MLX conversion."""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from liveframe.backends.base import TensorDescriptor
from liveframe.backends.pytorch import _normalize_tensor_name, _state_dict
from liveframe.manifests import file_sha256


class ConversionError(ValueError):
    """Raised when a source tensor cannot be mapped without guessing."""


@dataclass(frozen=True, slots=True)
class MappingRule:
    source: TensorDescriptor
    target_name: str
    target_shape: tuple[int, ...]
    target_dtype: str
    transform: str


@dataclass(frozen=True, slots=True)
class ConversionPlan:
    rules: tuple[MappingRule, ...]
    skipped: tuple[str, ...]

    def require_complete(self) -> None:
        if self.skipped:
            raise ConversionError(f"unmapped source tensors: {', '.join(self.skipped)}")
        targets = [rule.target_name for rule in self.rules]
        if len(targets) != len(set(targets)):
            raise ConversionError("multiple source tensors map to the same MLX target")


def plan_cfpp_conversion(
    inventory: tuple[TensorDescriptor, ...], *, target_dtype: str = "bfloat16"
) -> ConversionPlan:
    """Map stock Wan-compatible tensor metadata without loading tensor bytes.

    CF++-specific keys remain fail-closed until an H100 inventory proves their semantics.
    """
    if target_dtype not in {"float32", "float16", "bfloat16"}:
        raise ConversionError(f"unsupported target dtype: {target_dtype}")
    rules: list[MappingRule] = []
    skipped: list[str] = []
    for tensor in inventory:
        mapped = _map_tensor(tensor, target_dtype)
        if mapped is None:
            skipped.append(tensor.name)
        else:
            rules.append(mapped)
    return ConversionPlan(tuple(rules), tuple(skipped))


def _map_tensor(tensor: TensorDescriptor, target_dtype: str) -> MappingRule | None:
    name = tensor.name
    if name == "freqs":
        return None
    transform = "identity"
    target_name = name
    target_shape = tensor.shape
    if name == "patch_embedding.weight":
        if len(tensor.shape) != 5:
            raise ConversionError("patch embedding must be a five-dimensional Conv3d kernel")
        out_channels, in_channels, depth, height, width = tensor.shape
        target_name = "patch_embedding_proj.weight"
        target_shape = (out_channels, in_channels * depth * height * width)
        transform = "flatten-conv3d-input"
    elif name == "patch_embedding.bias":
        target_name = "patch_embedding_proj.bias"
    elif name.startswith("text_embedding.0."):
        target_name = name.replace("text_embedding.0.", "text_embedding_0.")
    elif name.startswith("text_embedding.2."):
        target_name = name.replace("text_embedding.2.", "text_embedding_1.")
    elif name.startswith("time_embedding.0."):
        target_name = name.replace("time_embedding.0.", "time_embedding_0.")
    elif name.startswith("time_embedding.2."):
        target_name = name.replace("time_embedding.2.", "time_embedding_1.")
    elif name.startswith("time_projection.1."):
        target_name = name.replace("time_projection.1.", "time_projection.")
    elif ".ffn.0." in name:
        target_name = name.replace(".ffn.0.", ".ffn.fc1.")
    elif ".ffn.2." in name:
        target_name = name.replace(".ffn.2.", ".ffn.fc2.")
    elif not _known_passthrough(name):
        return None
    dtype = target_dtype if tensor.dtype in {"torch.float32", "float32"} else tensor.dtype
    return MappingRule(tensor, target_name, target_shape, dtype, transform)


def _known_passthrough(name: str) -> bool:
    prefixes = ("blocks.", "head.")
    suffixes = (".weight", ".bias", ".modulation")
    return name.startswith(prefixes) and name.endswith(suffixes)


def execute_cfpp_conversion(
    checkpoint: Path,
    stock_mlx_dir: Path,
    output_dir: Path,
    *,
    expected_source_sha256: str,
    target_dtype: str = "bfloat16",
) -> Path:
    """Convert a released CF++ student into the pinned MLX Wan tensor layout."""
    actual_source = file_sha256(checkpoint)
    if actual_source != expected_source_sha256:
        raise ConversionError(
            f"CF++ source checksum mismatch: expected {expected_source_sha256}, "
            f"computed {actual_source}"
        )
    required = ("config.json", "t5_encoder.safetensors", "vae.safetensors")
    missing = [name for name in required if not (stock_mlx_dir / name).is_file()]
    if missing:
        raise ConversionError(f"stock MLX directory is missing: {', '.join(missing)}")
    try:
        torch = importlib.import_module("torch")
        mx = importlib.import_module("mlx.core")
    except ImportError as exc:
        raise ConversionError("CF++ conversion requires both PyTorch and MLX") from exc

    try:
        checkpoint_root = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConversionError(f"cannot safely load CF++ checkpoint: {exc}") from exc
    state = _state_dict(checkpoint_root)
    normalized = {
        _normalize_tensor_name(name): tensor
        for name, tensor in state.items()
        if isinstance(name, str) and isinstance(tensor, torch.Tensor)
    }
    inventory = tuple(
        TensorDescriptor(name, tuple(tensor.shape), str(tensor.dtype))
        for name, tensor in normalized.items()
    )
    plan = plan_cfpp_conversion(inventory, target_dtype=target_dtype)
    plan.require_complete()
    dtype = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }[target_dtype]
    converted: dict[str, Any] = {}
    for rule in plan.rules:
        tensor = normalized[rule.source.name]
        array = mx.array(tensor.detach().float().numpy())
        if rule.transform == "flatten-conv3d-input":
            array = array.reshape(rule.target_shape)
        converted[rule.target_name] = array.astype(dtype)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.safetensors"
    mx.save_safetensors(str(model_path), converted)
    for name in required:
        destination = output_dir / name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        os.symlink((stock_mlx_dir / name).resolve(), destination)
    metadata = {
        "schema": "liveframe.cfpp-mlx-conversion.v1",
        "source_checkpoint": str(checkpoint.resolve()),
        "source_sha256": actual_source,
        "target_dtype": target_dtype,
        "tensor_count": len(converted),
        "model_sha256": file_sha256(model_path),
        "stock_mlx_dir": str(stock_mlx_dir.resolve()),
        "stock_assets": {name: file_sha256(stock_mlx_dir / name) for name in required},
    }
    metadata_path = output_dir / "liveframe-conversion.json"
    temporary = metadata_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, metadata_path)
    del checkpoint_root, state, normalized, converted
    return metadata_path
