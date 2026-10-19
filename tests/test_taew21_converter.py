from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("safetensors")

from safetensors import safe_open
from safetensors.numpy import save_file

from liveframe.converters.taew21_to_mlx import (
    EXPECTED_DECODER_KEYS,
    TAEW21ConversionError,
    TAEW21Tensor,
    execute_taew21_conversion,
    plan_taew21_conversion,
)


def _inventory() -> dict[str, TAEW21Tensor]:
    inventory: dict[str, TAEW21Tensor] = {}
    for name in EXPECTED_DECODER_KEYS:
        shape = (8,) if name.endswith(".bias") else (8, 4, 3, 3)
        inventory[name] = TAEW21Tensor(name, shape, "float16")
    inventory["encoder.0.weight"] = TAEW21Tensor(
        "encoder.0.weight", (64, 3, 3, 3), "float16"
    )
    return inventory


def test_taew21_plan_maps_every_decoder_tensor_and_ignores_encoder() -> None:
    rules = plan_taew21_conversion(_inventory())

    assert len(rules) == 64
    assert {rule.target_name for rule in rules} == EXPECTED_DECODER_KEYS
    weight = next(rule for rule in rules if rule.source.name == "decoder.1.weight")
    assert weight.target_shape == (8, 3, 3, 4)
    assert weight.transform == "oihw-to-ohwi"


def test_taew21_plan_rejects_missing_or_unexpected_decoder_tensor() -> None:
    missing = _inventory()
    missing.pop("decoder.1.weight")
    with pytest.raises(TAEW21ConversionError, match=r"missing=.*decoder\.1\.weight"):
        plan_taew21_conversion(missing)

    unexpected = _inventory()
    unexpected["decoder.99.weight"] = TAEW21Tensor(
        "decoder.99.weight", (8, 4, 3, 3), "float16"
    )
    with pytest.raises(
        TAEW21ConversionError, match=r"unexpected=.*decoder\.99\.weight"
    ):
        plan_taew21_conversion(unexpected)


def test_taew21_plan_rejects_nonconvolution_weight_shape() -> None:
    inventory = _inventory()
    inventory["decoder.1.weight"] = TAEW21Tensor(
        "decoder.1.weight", (8, 4), "float16"
    )

    with pytest.raises(TAEW21ConversionError, match="must be OIHW"):
        plan_taew21_conversion(inventory)


def test_taew21_converter_executes_real_safetensors_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    tensors = {
        name: np.arange(8, dtype=np.float16)
        if name.endswith(".bias")
        else np.arange(8 * 4 * 3 * 3, dtype=np.float16).reshape(8, 4, 3, 3)
        for name in EXPECTED_DECODER_KEYS
    }
    tensors["encoder.0.weight"] = np.zeros((64, 3, 3, 3), dtype=np.float16)
    save_file(tensors, source)
    digest = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()

    manifest = execute_taew21_conversion(
        source,
        tmp_path / "output",
        expected_source_sha256=digest,
        source_repository="https://example.invalid/taehv",
        source_revision="pinned",
    )

    assert manifest.is_file()
    with safe_open(manifest.parent / "taew2_1-mlx.safetensors", framework="np") as handle:
        assert set(handle.keys()) == EXPECTED_DECODER_KEYS
        converted = handle.get_tensor("decoder.1.weight")
        assert converted.shape == (8, 3, 3, 4)
        assert np.array_equal(
            converted, tensors["decoder.1.weight"].transpose(0, 2, 3, 1)
        )


def test_taew21_converter_upcasts_to_float32_and_records_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.safetensors"
    tensors = {
        name: np.arange(8, dtype=np.float16)
        if name.endswith(".bias")
        else np.arange(8 * 4 * 3 * 3, dtype=np.float16).reshape(8, 4, 3, 3)
        for name in EXPECTED_DECODER_KEYS
    }
    save_file(tensors, source)
    digest = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()

    manifest_path = execute_taew21_conversion(
        source,
        tmp_path / "output",
        expected_source_sha256=digest,
        source_repository="https://example.invalid/taehv",
        source_revision="pinned",
        target_dtype="float32",
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["target"]["target_dtype"] == "float32"
    assert {row["source_dtype"] for row in manifest["mappings"]} == {"float16"}
    assert {row["dtype"] for row in manifest["mappings"]} == {"float32"}
    with safe_open(manifest_path.parent / "taew2_1-mlx.safetensors", framework="np") as handle:
        converted = handle.get_tensor("decoder.1.weight")
        assert converted.dtype == np.float32
        assert np.array_equal(
            converted,
            tensors["decoder.1.weight"].transpose(0, 2, 3, 1).astype(np.float32),
        )


def test_taew21_converter_rejects_unknown_target_dtype(tmp_path: Path) -> None:
    with pytest.raises(TAEW21ConversionError, match=r"unsupported.*target dtype"):
        execute_taew21_conversion(
            tmp_path / "missing.safetensors",
            tmp_path / "output",
            expected_source_sha256="sha256:unused",
            source_repository="https://example.invalid/taehv",
            source_revision="pinned",
            target_dtype="bfloat16",
        )