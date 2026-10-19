"""Empirical power for the ACTUAL v6.1/v6.2 collection state machine.

Frozen before its values are visible (amendment v6.2). Unlike the v1
simulation (fixed 30-participant cohort, no replacement), this executes the
recruitment procedure that will really run:

1. Collect all 30 primary slots (P00-P29). Each participant independently
   fails the attention check with probability 0.10 (excluded, contributes 0).
2. While usable study judgments < 486 and replacements remain, activate the
   next replacement slot in ID order (same exclusion probability), exactly as
   the QC tool's necessity rule requires.
3. Stop at the first count >= 486 (or exhaustion of R30-R35).
4. Apply the analyzer's exact two-gate verdict: target met (usable >= 486)
   AND two-way cluster-robust one-sided 95% UCB <= 0.55.

Heterogeneity model, analysis machinery, and constants are imported from the
frozen v1 simulator so the two artifacts cannot drift apart. Judgment
assignment uses the builder's real slot arithmetic (offset = slot * 18 mod 32
round-robin over the 32 items).

This artifact REPLACES the retracted v6.1 claim that the v1 value (0.7485)
lower-bounds the executed design. The operative usable-judgment target (486)
and the verdict rule are unchanged; this is transparency evidence about the
design's error rates, computed before recruitment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from simulate_taew21_study_power import (  # noqa: E402
    ATTENTION_FAILURE_RATE,
    CEILING,
    P_TRUE_GRID,
    SIGMA_PARTICIPANT,
    SIGMA_VIDEO,
    VIDEO_PAIRS,
    Z_ONE_SIDED_95,
    _expit,
    _logit,
    _two_way_se,
)

from liveframe.manifests import file_sha256  # noqa: E402

SEED = 20260823
REPLICATES = 2000
PRIMARIES = 30
REPLACEMENTS = 6
TRIALS = 18


def _simulate_state_machine(
    rng: np.random.Generator, p_true: float, target: int
) -> dict[str, Any]:
    base = _logit(p_true)
    passes = 0
    usable_totals = []
    replacements_activated = []
    exhausted_count = 0
    slots_total = PRIMARIES + REPLACEMENTS
    for _ in range(REPLICATES):
        keep = rng.random(slots_total) >= ATTENTION_FAILURE_RATE
        participant_effects = rng.normal(0.0, SIGMA_PARTICIPANT, slots_total)
        video_effects = rng.normal(0.0, SIGMA_VIDEO, VIDEO_PAIRS)
        collected: list[int] = list(range(PRIMARIES))
        usable = sum(TRIALS for slot in collected if keep[slot])
        activated = 0
        while usable < target and activated < REPLACEMENTS:
            slot = PRIMARIES + activated
            collected.append(slot)
            activated += 1
            if keep[slot]:
                usable += TRIALS
        replacements_activated.append(activated)
        usable_totals.append(usable)
        if usable < target:
            exhausted_count += 1
            continue
        rows_participant = []
        rows_video = []
        for slot in collected:
            if not keep[slot]:
                continue
            offset = (slot * TRIALS) % VIDEO_PAIRS
            rows_participant.append(np.full(TRIALS, slot))
            rows_video.append((offset + np.arange(TRIALS)) % VIDEO_PAIRS)
        participant_column = np.concatenate(rows_participant)
        video_column = np.concatenate(rows_video)
        probabilities = _expit(
            base
            * np.exp(
                participant_effects[participant_column]
                + video_effects[video_column]
            )
        )
        correct = (rng.random(probabilities.size) < probabilities).astype(np.float64)
        _, participant_index = np.unique(participant_column, return_inverse=True)
        p_hat = float(correct.mean())
        upper = p_hat + Z_ONE_SIDED_95 * _two_way_se(
            correct, participant_index, video_column, p_hat
        )
        passes += upper <= CEILING
    return {
        "p_true": p_true,
        "pass_probability": passes / REPLICATES,
        "expected_usable_judgments": float(np.mean(usable_totals)),
        "mean_replacements_activated": float(np.mean(replacements_activated)),
        "exhaustion_probability": exhausted_count / REPLICATES,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--power-v1", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace v2 power artifact: {args.output}")
    power_v1 = json.loads(args.power_v1.read_text(encoding="utf-8"))
    if power_v1.get("schema") != "liveframe.taew21-study-power.v1":
        raise ValueError("unsupported v1 power artifact schema")
    target = int(power_v1["selected_design"]["usable_judgment_target"])
    amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
    if (
        amendment.get("schema") != "liveframe.taew21-protocol-amendment.v6.2"
        or amendment["bindings"]["power_v1_sha256"] != file_sha256(args.power_v1)
        or amendment["bindings"]["protocol_sha256"] != file_sha256(args.protocol)
    ):
        raise ValueError("v2 power simulation requires the bound amendment v6.2")

    rng = np.random.default_rng(SEED)
    cells = [
        _simulate_state_machine(rng, p_true, target) for p_true in P_TRUE_GRID
    ]
    chance = next(cell for cell in cells if cell["p_true"] == 0.50)
    result = {
        "schema": "liveframe.taew21-study-power.v2",
        "authority": (
            "transparency evidence for the executed v6.1/v6.2 design; the operative "
            "target (486) and the analyzer verdict rule come from the frozen v1 "
            "artifact and are unchanged"
        ),
        "protocol_sha256": file_sha256(args.protocol),
        "power_v1_sha256": file_sha256(args.power_v1),
        "amendment_sha256": file_sha256(args.amendment),
        "simulator_sha256": file_sha256(Path(__file__).resolve()),
        "state_machine": (
            "collect 30 primaries -> sequential necessity-gated replacement "
            "activation in ID order (max 6) -> stop at first usable >= target -> "
            "two-gate verdict (target met AND two-way clustered UCB <= 0.55)"
        ),
        "assumptions": {
            "seed": SEED,
            "replicates": REPLICATES,
            "sigma_participant_logit": SIGMA_PARTICIPANT,
            "sigma_video_logit": SIGMA_VIDEO,
            "attention_failure_rate": ATTENTION_FAILURE_RATE,
            "usable_judgment_target": target,
            "heterogeneity_and_analysis": "imported from frozen v1 simulator",
        },
        "cells": cells,
        "pass_probability_at_chance": chance["pass_probability"],
        "supersedes": (
            "the RETRACTED v6.1 claim that the v1 value 0.7485 lower-bounds the "
            "executed design"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pass_probability_at_chance": chance["pass_probability"],
                "cells": [
                    {"p_true": cell["p_true"], "pass": cell["pass_probability"]}
                    for cell in cells
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
