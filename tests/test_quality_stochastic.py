from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from liveframe.quality_stochastic import (
    STOCHASTIC_CONTRACT,
    StochasticPlanReader,
    create_stochastic_plan,
    transition_name,
)


def test_shared_plan_round_trips_identical_torch_and_mlx_bytes(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    first = tmp_path / "first"
    second = tmp_path / "second"
    create_stochastic_plan(first, seed=42, latent_frames=12)
    create_stochastic_plan(second, seed=42, latent_frames=12)
    first_metadata = json.loads((first / "stochastic.bf16.json").read_text())
    second_metadata = json.loads((second / "stochastic.bf16.json").read_text())

    assert first_metadata["contract"] == STOCHASTIC_CONTRACT
    assert first_metadata["payload_sha256"] == second_metadata["payload_sha256"]
    assert first_metadata["tensors"] == second_metadata["tensors"]

    torch_reader = StochasticPlanReader(
        first / "stochastic.bf16.json", seed=42, latent_frames=12
    )
    mlx_reader = StochasticPlanReader(
        first / "stochastic.bf16.json", seed=42, latent_frames=12
    )
    torch_value = torch_reader.next_torch(
        "initial_noise", (1, 12, 16, 60, 104), device="cpu"
    )
    mlx_value = mlx_reader.next_mlx("initial_noise", (1, 12, 16, 60, 104))

    torch_bytes = torch_value.view(__import__("torch").uint16).numpy().tobytes()
    mlx_bytes = np.array(mlx_value.view(__import__("mlx.core").core.uint16)).tobytes()
    assert torch_bytes == mlx_bytes


def test_shared_plan_rejects_tensor_tampering(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    create_stochastic_plan(tmp_path, seed=42, latent_frames=12)
    payload = tmp_path / "stochastic.bf16"
    altered = bytearray(payload.read_bytes())
    altered[-1] ^= 1
    payload.write_bytes(altered)
    metadata_path = tmp_path / "stochastic.bf16.json"
    metadata = json.loads(metadata_path.read_text())
    from liveframe.quality_capture import sha256_file

    metadata["payload_sha256"] = sha256_file(payload)
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="tensor checksum mismatch"):
        StochasticPlanReader(metadata_path, seed=42, latent_frames=12)


def test_shared_plan_fails_closed_on_order_and_completeness(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    create_stochastic_plan(tmp_path, seed=42, latent_frames=12)
    reader = StochasticPlanReader(
        tmp_path / "stochastic.bf16.json", seed=42, latent_frames=12
    )

    with pytest.raises(ValueError, match="order mismatch"):
        reader.next_torch(
            transition_name(0, 0, 1000, 750),
            (1, 3, 16, 60, 104),
            device="cpu",
        )
    with pytest.raises(ValueError, match="unconsumed tensors"):
        reader.require_consumed()
