from __future__ import annotations

import pytest

from liveframe.mlx_vae_profile import summarize_stage_rows


def test_stage_summary_aggregates_and_orders_by_total_eval() -> None:
    rows = [
        {
            "stage": "small",
            "graph_build_seconds": 0.01,
            "eval_seconds": 1.0,
            "peak_mlx_memory_gib": 2.0,
        },
        {
            "stage": "large",
            "graph_build_seconds": 0.02,
            "eval_seconds": 3.0,
            "peak_mlx_memory_gib": 4.0,
        },
        {
            "stage": "small",
            "graph_build_seconds": 0.03,
            "eval_seconds": 1.0,
            "peak_mlx_memory_gib": 3.0,
        },
    ]

    summary = summarize_stage_rows(rows)

    assert [row["stage"] for row in summary] == ["large", "small"]
    assert summary[0]["eval_share_percent"] == pytest.approx(60.0)
    assert summary[1]["graph_build_seconds_median"] == pytest.approx(0.02)
    assert summary[1]["peak_mlx_memory_gib"] == pytest.approx(3.0)


def test_stage_summary_requires_samples() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_stage_rows([])