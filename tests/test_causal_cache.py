from __future__ import annotations

import pytest

from liveframe.causal_cache import CachePlanError, plan_cache_update


def test_causal_cache_appends_then_overwrites_same_block() -> None:
    first = plan_cache_update(
        current_start=0,
        token_count=1560,
        previous_global_end=0,
        previous_local_end=0,
        capacity=32760,
        max_attention_size=32760,
    )
    refresh = plan_cache_update(
        current_start=0,
        token_count=1560,
        previous_global_end=first.global_end,
        previous_local_end=first.local_end,
        capacity=32760,
        max_attention_size=32760,
    )

    assert (first.local_start, first.local_end, first.global_end) == (0, 1560, 1560)
    assert refresh.is_overwrite
    assert (refresh.local_start, refresh.local_end, refresh.global_end) == (0, 1560, 1560)


def test_local_cache_rolls_oldest_non_sink_tokens() -> None:
    plan = plan_cache_update(
        current_start=8,
        token_count=4,
        previous_global_end=8,
        previous_local_end=8,
        capacity=10,
        max_attention_size=8,
        local_attention=True,
        sink_tokens=2,
    )

    assert plan.evicted_tokens == 2
    assert plan.rolled_tokens == 4
    assert (plan.local_start, plan.local_end) == (6, 10)
    assert plan.attention_start == 2


def test_sink_may_be_configured_before_initial_cache_fill() -> None:
    plan = plan_cache_update(
        current_start=0,
        token_count=4,
        previous_global_end=0,
        previous_local_end=0,
        capacity=8,
        max_attention_size=8,
        local_attention=True,
        sink_tokens=2,
    )

    assert (plan.local_start, plan.local_end) == (0, 4)
    assert plan.evicted_tokens == 0


def test_cache_plan_rejects_global_gaps() -> None:
    with pytest.raises(CachePlanError, match="global token gap"):
        plan_cache_update(
            current_start=1560,
            token_count=1560,
            previous_global_end=0,
            previous_local_end=0,
            capacity=32760,
            max_attention_size=32760,
        )