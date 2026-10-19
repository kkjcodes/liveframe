from __future__ import annotations

import pytest

from liveframe.world.actions import (
    ActionScript,
    ActionSegment,
    canonical_action,
    plan_chunks,
)


def _script(rows):
    return ActionScript.from_segments(rows)


def test_canonical_action_sorts_combo_tokens() -> None:
    assert canonical_action("L+W") == "W+L"
    assert canonical_action("0") == "0"


@pytest.mark.parametrize(
    "action", ["X", "W+W", "W+0", "W+S", "L+R", "U+N", "", "w"]
)
def test_canonical_action_rejects_invalid(action: str) -> None:
    with pytest.raises(ValueError):
        canonical_action(action)


def test_script_requires_contiguous_segments() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        _script([(0, "W", 8), (10, "L", 8)])
    with pytest.raises(ValueError, match="frame 0"):
        _script([(4, "W", 8)])


def test_script_round_trips_jsonl_and_digests_stably() -> None:
    script = _script([(0, "W", 32), (32, "L", 16), (48, "0", 16)])
    assert script.total_frames == 64
    clone = ActionScript.from_jsonl(script.as_jsonl())
    assert clone == script
    assert clone.digest() == script.digest()
    assert script.action_at(0) == "W"
    assert script.action_at(33) == "L"
    assert script.action_at(63) == "0"


def test_plan_chunks_records_quantization_without_shifting() -> None:
    script = _script([(0, "W", 6), (6, "L", 10)])
    plans = list(plan_chunks(script, 4))
    assert [plan.frames for plan in plans] == [4, 4, 4, 4]
    assert [plan.action for plan in plans] == ["W", "W", "L", "L"]
    # The W->L boundary at frame 6 falls 2 frames into chunk 1.
    assert plans[1].quantization_frames == 2
    assert plans[0].quantization_frames == 0
    assert [plan.action_changed for plan in plans] == [False, False, True, False]


def test_plan_chunks_handles_partial_final_chunk() -> None:
    script = _script([(0, "W", 10)])
    plans = list(plan_chunks(script, 4))
    assert [plan.frames for plan in plans] == [4, 4, 2]
    assert ActionSegment(t_frames=0, action="L+W", hold_frames=1).action == "W+L"


def test_plan_chunks_supports_shorter_first_chunk() -> None:
    # Causal-VAE pattern: first block 9 pixel frames, then 12 per block.
    script = _script([(0, "W", 9), (9, "L", 24)])
    plans = list(plan_chunks(script, 12, first_chunk_frames=9))
    assert [plan.frames for plan in plans] == [9, 12, 12]
    assert [plan.start_frame for plan in plans] == [0, 9, 21]
    assert [plan.action for plan in plans] == ["W", "L", "L"]
    assert plans[1].action_changed is True
    assert all(plan.quantization_frames == 0 for plan in plans)
