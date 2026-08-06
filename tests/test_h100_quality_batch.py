from __future__ import annotations

import json
from pathlib import Path

import torch

from liveframe.quality_artifacts import StochasticWriter, TensorStreamWriter, write_tensor
from liveframe.quality_capture import sha256_file


def test_tensor_writer_publishes_lossless_payload_and_metadata(tmp_path: Path) -> None:
    pixels = torch.arange(36, dtype=torch.uint8).reshape(3, 2, 2, 3)

    artifacts = write_tensor(
        tmp_path / "pixels.bin",
        pixels,
        layout="T,H,W,C",
        schema="liveframe.decoded-pixels.v1",
    )

    metadata = json.loads(artifacts[1].read_text(encoding="utf-8"))
    assert artifacts[0].read_bytes() == pixels.numpy().tobytes()
    assert metadata["shape"] == [3, 2, 2, 3]
    assert metadata["sha256"] == sha256_file(artifacts[0])


def test_stochastic_writer_records_offsets_and_tensor_digests(tmp_path: Path) -> None:
    writer = StochasticWriter(tmp_path)
    writer.append("initial_noise", torch.ones((1, 2), dtype=torch.bfloat16))
    writer.append("scheduler_noise", torch.zeros((1, 3), dtype=torch.bfloat16))

    payload_path, metadata_path = writer.finish(seed=11003)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload_path.stat().st_size == 10
    assert metadata["seed"] == 11003
    assert metadata["tensor_count"] == 2
    assert [row["offset_bytes"] for row in metadata["tensors"]] == [0, 4]
    assert metadata["payload_sha256"] == sha256_file(payload_path)


def test_stochastic_writer_abort_removes_partial_payload(tmp_path: Path) -> None:
    writer = StochasticWriter(tmp_path)
    writer.append("partial", torch.ones((1, 2), dtype=torch.bfloat16))

    writer.abort()

    assert not (tmp_path / ".stochastic.bf16.tmp").exists()


def test_stream_writer_matches_whole_tensor_artifact(tmp_path: Path) -> None:
    pixels = torch.arange(72, dtype=torch.uint8).reshape(6, 2, 2, 3)
    writer = TensorStreamWriter(
        tmp_path / "pixels.rgb",
        layout="T,H,W,C",
        schema="liveframe.decoded-pixels.v1",
    )
    writer.append(pixels[:1])
    writer.append(pixels[1:5])
    writer.append(pixels[5:])

    payload_path, metadata_path = writer.finish()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload_path.read_bytes() == pixels.numpy().tobytes()
    assert metadata["shape"] == [6, 2, 2, 3]
    assert metadata["sha256"] == sha256_file(payload_path)


def test_stream_writer_concatenates_singleton_batch_time_axis(tmp_path: Path) -> None:
    latents = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 2, 2).to(
        torch.bfloat16
    )
    writer = TensorStreamWriter(
        tmp_path / "latents.bf16",
        layout="B,T,C,H,W",
        schema="liveframe.refresh-latents.v2",
        concat_dim=1,
    )
    for frame in range(latents.shape[1]):
        writer.append(latents[:, frame : frame + 1])

    payload_path, metadata_path = writer.finish()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload_path.read_bytes() == latents.view(torch.uint16).numpy().tobytes()
    assert metadata["shape"] == [1, 3, 2, 2, 2]
