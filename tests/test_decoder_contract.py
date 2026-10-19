from __future__ import annotations

import pytest

from liveframe.decoder_contract import (
    DecoderChunk,
    DecoderContractError,
    DecoderRequest,
    validate_decoder_chunks,
)


def _chunk(start: int, frames: int) -> DecoderChunk:
    return DecoderChunk(
        start_index=start,
        payload=object(),
        shape=(frames, 480, 832, 3),
        dtype="uint8",
        finite=True,
        minimum=0,
        maximum=255,
    )


def test_decoder_contract_accepts_exact_causal_chunk_accounting() -> None:
    request = DecoderRequest(latent_frames=21, native_frames=81, width=832, height=480)
    chunks = [_chunk(0, 1), *(_chunk(1 + 4 * index, 4) for index in range(20))]

    assert sum(chunk.native_frame_count for chunk in validate_decoder_chunks(chunks, request)) == 81


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        ([_chunk(0, 1), _chunk(2, 4)], "index mismatch"),
        ([_chunk(0, 1)], "emitted 1 native frames"),
        ([_chunk(0, 82)], "emitted 82 native frames"),
    ],
)
def test_decoder_contract_rejects_bad_frame_accounting(
    chunks: list[DecoderChunk], message: str
) -> None:
    request = DecoderRequest(latent_frames=21, native_frames=81, width=832, height=480)

    with pytest.raises(DecoderContractError, match=message):
        tuple(validate_decoder_chunks(chunks, request))


def test_decoder_contract_rejects_nonfinite_or_wrong_resolution() -> None:
    request = DecoderRequest(latent_frames=1, native_frames=1, width=832, height=480)
    nonfinite = DecoderChunk(0, object(), (1, 480, 832, 3), "uint8", False, 0, 255)
    wrong_shape = DecoderChunk(0, object(), (1, 512, 512, 3), "uint8", True, 0, 255)

    with pytest.raises(DecoderContractError, match="nonfinite"):
        tuple(validate_decoder_chunks([nonfinite], request))
    with pytest.raises(DecoderContractError, match="requested RGB resolution"):
        tuple(validate_decoder_chunks([wrong_shape], request))