"""Pre-registered clustered power simulation for the TAEW2.1 human study.

Fixes the usable-judgment target BEFORE recruitment, per the frozen protocol
(`human_study.sample_size = fixed-by-clustered-simulation-before-recruitment`,
stopping rule = stop at the simulation-fixed target, no sequential peeking).

Model (all parameters frozen here, before any response is visible):
- True discrimination accuracy p_true on a grid including chance (0.50), the
  sensitivity point (0.53), and the ceiling (0.55).
- Heterogeneity scales with discriminability (multiplicative d'-style model):
  logit(p_ij) = logit(p_true) * exp(u_i + v_j) with participant effects
  u_i ~ N(0, sigma_participant^2) and video-pair effects
  v_j ~ N(0, sigma_video^2). At true chance the discriminative signal is
  zero, so every cluster sits exactly at 0.5 and judgments are independent
  Bernoulli(0.5) — there is no signal for participants or videos to differ
  on. An additive-offset model at chance would instead simulate a world
  where individual videos ARE partially distinguishable (per-cluster
  accuracy deviating from 0.5 in both directions), which is a different
  hypothesis than the one the chance cell tests. Above chance the
  multiplicative effects spread per-cluster accuracy exactly as intended.
- Balanced incomplete block design: each participant judges `trials` of the 32
  frozen video pairs, assigned round-robin from a rotating offset so video
  exposure counts differ by at most one.
- Exclusions: each participant independently fails the pre-declared attention
  check with probability 0.10 and is dropped entirely (usable judgments vary
  per replicate; the design target is the EXPECTED usable judgments).

Analysis rule (identical to the study's primary analysis): accuracy p_hat with
two-way cluster-robust standard error (participant and video clusters,
Cameron-Gelbach-Miller combination, each component the cluster-robust variance
of a proportion), one-sided 95% upper confidence bound
UCB = p_hat + 1.645 * SE. PASS if UCB <= 0.55.

The selected design is the smallest candidate whose pass probability at
p_true = 0.50 is at least POWER_TARGET with expected usable judgments inside
the frozen 400-600 planning envelope and at least 271 minimum usable
judgments. POWER_TARGET is 0.70: the protocol's own planning envelope caps
nominal judgments at 600, where even independent-judgment arithmetic yields
only ~0.79 pass probability at true chance (achieving 0.90 would need ~860+
independent judgments, outside the frozen envelope), so the envelope
implicitly accepts an inconclusive-fail risk of roughly one in four at true
chance. That risk is reported, not hidden: every cell of the sensitivity grid
is written to the artifact.
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

from liveframe.manifests import file_sha256  # noqa: E402

SEED = 20260818
REPLICATES = 2000
SIGMA_PARTICIPANT = 0.5
SIGMA_VIDEO = 0.5
ATTENTION_FAILURE_RATE = 0.10
VIDEO_PAIRS = 32
Z_ONE_SIDED_95 = 1.645
CEILING = 0.55
POWER_TARGET = 0.70
P_TRUE_GRID = (0.50, 0.53, 0.55, 0.58)
DESIGNS = ({"participants": 30, "trials": 16}, {"participants": 30, "trials": 18},
           {"participants": 30, "trials": 20}, {"participants": 36, "trials": 18})


def _expit(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _logit(value: float) -> float:
    return float(np.log(value / (1.0 - value)))


def _cluster_variance(correct: np.ndarray, clusters: np.ndarray, p_hat: float) -> float:
    residuals = correct - p_hat
    sums = np.bincount(clusters, weights=residuals)
    return float(np.sum(sums**2)) / (correct.size**2)


def _two_way_se(
    correct: np.ndarray, participants: np.ndarray, videos: np.ndarray, p_hat: float
) -> float:
    pair_ids = participants.astype(np.int64) * VIDEO_PAIRS + videos.astype(np.int64)
    _, pair_index = np.unique(pair_ids, return_inverse=True)
    variance = (
        _cluster_variance(correct, participants, p_hat)
        + _cluster_variance(correct, videos, p_hat)
        - _cluster_variance(correct, pair_index, p_hat)
    )
    return float(np.sqrt(max(variance, 0.0)))


def _simulate_design(
    rng: np.random.Generator, participants: int, trials: int, p_true: float
) -> dict[str, Any]:
    base = _logit(p_true)
    passes = 0
    usable_totals = []
    realized = []
    for _ in range(REPLICATES):
        keep = rng.random(participants) >= ATTENTION_FAILURE_RATE
        participant_effects = rng.normal(0.0, SIGMA_PARTICIPANT, participants)
        video_effects = rng.normal(0.0, SIGMA_VIDEO, VIDEO_PAIRS)
        rows_participant = []
        rows_video = []
        for index in range(participants):
            if not keep[index]:
                continue
            offset = (index * trials) % VIDEO_PAIRS
            videos = (offset + np.arange(trials)) % VIDEO_PAIRS
            rows_participant.append(np.full(trials, index))
            rows_video.append(videos)
        if not rows_participant:
            continue
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
        usable_totals.append(correct.size)
        realized.append(float(probabilities.mean()))
    return {
        "p_true": p_true,
        "realized_mean_accuracy": float(np.mean(realized)),
        "pass_probability": passes / REPLICATES,
        "expected_usable_judgments": float(np.mean(usable_totals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace power simulation: {args.output}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    study = protocol["human_study"]
    if study["noticeability_ceiling"] != CEILING or study["chance_accuracy"] != 0.5:
        raise ValueError("frozen simulation constants do not match the protocol")

    rng = np.random.default_rng(SEED)
    design_reports = []
    for design in DESIGNS:
        cells = [
            _simulate_design(rng, design["participants"], design["trials"], p_true)
            for p_true in P_TRUE_GRID
        ]
        chance_cell = next(cell for cell in cells if cell["p_true"] == 0.50)
        design_reports.append(
            {
                **design,
                "nominal_judgments": design["participants"] * design["trials"],
                "expected_usable_judgments": chance_cell["expected_usable_judgments"],
                "cells": cells,
                "pass_probability_at_chance": chance_cell["pass_probability"],
            }
        )
    selected = None
    for report in design_reports:
        if (
            report["pass_probability_at_chance"] >= POWER_TARGET
            and report["expected_usable_judgments"] >= study["minimum_usable_judgments"]
            and study["planning_judgments"][0]
            <= report["nominal_judgments"]
            <= study["planning_judgments"][1]
        ):
            selected = report
            break
    if selected is None:
        raise RuntimeError(
            "no candidate design achieves the POWER_TARGET pass probability at "
            "chance inside the frozen planning envelope; larger heterogeneity "
            "than assumed"
        )
    result = {
        "schema": "liveframe.taew21-study-power.v1",
        "protocol_sha256": file_sha256(args.protocol),
        "simulator_sha256": file_sha256(Path(__file__).resolve()),
        "assumptions": {
            "seed": SEED,
            "replicates": REPLICATES,
            "sigma_participant_logit": SIGMA_PARTICIPANT,
            "sigma_video_logit": SIGMA_VIDEO,
            "attention_failure_rate": ATTENTION_FAILURE_RATE,
            "analysis": "two-way-cluster-robust-CGM-one-sided-95-UCB",
            "power_target_at_chance": POWER_TARGET,
            "power_target_rationale": (
                "0.90 is arithmetically unreachable inside the protocol's frozen "
                "400-600 planning envelope even for independent judgments; the "
                "envelope implicitly accepts roughly one-in-four inconclusive-fail "
                "risk at true chance, reported in the sensitivity grid"
            ),
            "derivation": "frozen-before-any-response-visibility",
        },
        "designs": design_reports,
        "selected_design": {
            "participants": selected["participants"],
            "trials_per_participant": selected["trials"],
            "nominal_judgments": selected["nominal_judgments"],
            "usable_judgment_target": round(selected["expected_usable_judgments"]),
            "pass_probability_at_chance": selected["pass_probability_at_chance"],
        },
        "stopping_rule": (
            "stop at the usable_judgment_target above; no sequential peeking; "
            "excluded judgments are replaced only by pre-registered replacement "
            "participants"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {"output": str(args.output), "selected": result["selected_design"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
