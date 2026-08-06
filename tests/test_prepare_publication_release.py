from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

TOOL_PATH = Path(__file__).parents[1] / "tools" / "prepare_publication_release.py"
SPEC = importlib.util.spec_from_file_location("prepare_publication_release", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load publication release tool")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
selected_rows: Any = MODULE.selected_rows


def test_selected_rows_uses_pre_registered_seed_ranks_and_horizons() -> None:
    protocol = {
        "sampling": {"holdout_seeds": [11, 22, 33, 44]},
        "visual_selection": {
            "required_holdout_seed_ranks_per_prompt": [0, 2],
            "include_horizons_seconds": [15, 60],
        },
    }
    holdout = {
        "candidate_videos": [
            {
                "stratum": "dialogue",
                "prompt_id": "prompt",
                "horizon_seconds": horizon,
                "seed_id": str(seed),
            }
            for horizon in (15, 60)
            for seed in (11, 22, 33, 44)
        ]
    }

    rows = selected_rows(holdout, protocol)

    assert [(row["horizon_seconds"], row["seed_id"]) for row in rows] == [
        (15, "11"),
        (15, "33"),
        (60, "11"),
        (60, "33"),
    ]