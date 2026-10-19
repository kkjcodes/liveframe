"""Analyze the TAEW2.1 discrimination-study responses (frozen pre-response).

Applies the pre-registered pipeline exactly:

1. Exclusions, in order: withdrawn-consent, duplicate-participant,
   incomplete-session (any scheduled trial missing), failed pre-declared
   attention check (any attention item answered incorrectly). Excluded
   participants contribute NO judgments.
2. Primary sample: study-role trials from included participants, first
   occurrence per (participant, item) only (repeat-consistency duplicates are
   reported separately and never enter the primary sample).
3. Primary rule: discrimination accuracy p_hat with two-way cluster-robust
   standard error (participant and item clusters, Cameron-Gelbach-Miller),
   one-sided 95% upper confidence bound p_hat + 1.645 * SE. Tier quality GO
   iff UCB <= 0.55.
4. Reported alongside: per-stratum and per-horizon estimates,
   repeat-consistency rate, secondary preference summary, and the
   pre-registered secondary metric correlation (per-item accuracy vs the v1
   pairwise LPIPS mean, participant-cluster bootstrap and
   leave-one-prompt-out stability).

Opening `assignments.sealed.json` is what un-blinds the analysis, so this
tool refuses to run unless `--responses-frozen` is passed, asserting that the
response file is final and no further data will be collected.

Response file format (`liveframe.taew21-study-responses.v1`):
{
  "schema": "liveframe.taew21-study-responses.v1",
  "withdrawn_participants": ["P03", ...],
  "duplicate_participants": ["P07", ...],
  "responses": [
    {"participant_id": "P00", "item_id": "item-00", "role": "study",
     "answer": "A", "preference": "A"|"B"|"none" (optional)},
    ...
  ]
}
Every response row must match a scheduled trial for that participant.
"""

# mypy: ignore-errors

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.manifests import file_sha256  # noqa: E402

Z_ONE_SIDED_95 = 1.645
CEILING = 0.55
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260821


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _cluster_variance(values: np.ndarray, clusters: np.ndarray, p_hat: float) -> float:
    residuals = values - p_hat
    sums = np.bincount(clusters, weights=residuals)
    return float(np.sum(sums**2)) / (values.size**2)


def _two_way_ucb(
    correct: np.ndarray, participants: np.ndarray, items: np.ndarray
) -> dict[str, float]:
    p_hat = float(correct.mean())
    _, participant_index = np.unique(participants, return_inverse=True)
    _, item_index = np.unique(items, return_inverse=True)
    pair_ids = participant_index.astype(np.int64) * (item_index.max() + 1) + item_index
    _, pair_index = np.unique(pair_ids, return_inverse=True)
    variance = (
        _cluster_variance(correct, participant_index, p_hat)
        + _cluster_variance(correct, item_index, p_hat)
        - _cluster_variance(correct, pair_index, p_hat)
    )
    se = float(np.sqrt(max(variance, 0.0)))
    return {
        "accuracy": p_hat,
        "se_two_way_cluster": se,
        "ucb_95_one_sided": p_hat + Z_ONE_SIDED_95 * se,
        "n": int(correct.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--v1-verdict", type=Path, required=True)
    parser.add_argument("--power", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument(
        "--responses-frozen",
        action="store_true",
        help="assert the response file is final; required to open sealed assignments",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.responses_frozen:
        raise SystemExit(
            "refusing to open sealed assignments before the response freeze; "
            "pass --responses-frozen only when data collection is complete"
        )
    if args.output.exists():
        raise FileExistsError(f"refusing to replace study analysis: {args.output}")

    manifest = _object(args.bundle_root / "bundle-manifest.json")
    schedules = _object(args.bundle_root / "participant-schedules.json")
    sealed_path = args.bundle_root / "assignments.sealed.json"
    if file_sha256(sealed_path) != manifest["sealed_assignments_sha256"]:
        raise ValueError("sealed assignments digest mismatch vs bundle manifest")
    if file_sha256(args.bundle_root / "participant-schedules.json") != manifest[
        "schedules_sha256"
    ]:
        raise ValueError("schedules digest mismatch vs bundle manifest")
    sealed = _object(sealed_path)
    answers = {row["item_id"]: row["correct_answer"] for row in sealed["assignments"]}
    item_meta = {
        row["item_id"]: row for row in manifest["items"] if row["kind"] == "study"
    }
    payload = _object(args.responses)
    if payload.get("schema") != "liveframe.taew21-study-responses.v1":
        raise ValueError("unsupported response schema")
    if file_sha256(args.power) != manifest["power_sha256"]:
        raise ValueError("power artifact digest mismatch vs bundle manifest")
    power = _object(args.power)
    if power.get("schema") != "liveframe.taew21-study-power.v1":
        raise ValueError("unsupported power artifact schema")
    target = int(power["selected_design"]["usable_judgment_target"])
    if target != int(schedules["usable_judgment_target"]):
        raise ValueError("power target does not match the schedules' pinned target")
    if file_sha256(args.amendment) != manifest["amendment_sha256"]:
        raise ValueError("amendment digest mismatch vs bundle manifest")
    amendment = _object(args.amendment)
    if amendment.get("schema") != "liveframe.taew21-protocol-amendment.v6.3":
        raise ValueError("unsupported amendment schema")

    scheduled: dict[str, list[dict[str, str]]] = {
        row["participant_id"]: row["trials"] for row in schedules["schedules"]
    }
    responses = payload["responses"]
    by_participant: dict[str, list[dict[str, Any]]] = {}
    for row in responses:
        by_participant.setdefault(str(row["participant_id"]), []).append(row)

    withdrawn = set(payload.get("withdrawn_participants", []))
    duplicates = set(payload.get("duplicate_participants", []))
    contradictory = sorted(withdrawn & duplicates)
    if contradictory:
        raise ValueError(
            f"contradictory exclusion metadata: {contradictory} appear in both "
            "withdrawn_participants and duplicate_participants"
        )
    unscheduled_exclusions = sorted(
        pid for pid in withdrawn | duplicates if pid not in scheduled
    )
    if unscheduled_exclusions:
        raise ValueError(
            "exclusion metadata names unscheduled participants: "
            f"{unscheduled_exclusions}"
        )
    exclusions: dict[str, str] = {}
    for participant in sorted(by_participant):
        if participant not in scheduled:
            raise ValueError(f"unscheduled participant in responses: {participant}")
        if participant in withdrawn:
            exclusions[participant] = "withdrawn-consent"
            continue
        if participant in duplicates:
            exclusions[participant] = "duplicate-participant"
            continue
        rows = by_participant[participant]
        expected = Counter(
            (trial["item_id"], trial["role"]) for trial in scheduled[participant]
        )
        observed = Counter((str(row["item_id"]), str(row["role"])) for row in rows)
        if observed != expected:
            exclusions[participant] = "incomplete-session"
            continue
        attention_wrong = any(
            str(row["role"]) == "attention"
            and str(row["answer"]) != answers[str(row["item_id"])]
            for row in rows
        )
        if attention_wrong:
            exclusions[participant] = "failed-predeclared-attention-check"
    # Withdrawn/duplicate participants who never submitted rows still account
    # for their slot and appear in the exclusion report.
    for participant in sorted(withdrawn | duplicates):
        if participant in scheduled and participant not in by_participant:
            exclusions[participant] = (
                "withdrawn-consent" if participant in withdrawn else "duplicate-participant"
            )
    included = [
        participant
        for participant in sorted(by_participant)
        if participant not in exclusions
    ]

    # Amendment v6.1/v6.2 target-based stopping enforcement (hard errors).
    primary_ids = [
        row["participant_id"]
        for row in schedules["schedules"]
        if row.get("cohort") != "replacement"
    ]
    replacement_ids = sorted(
        row["participant_id"]
        for row in schedules["schedules"]
        if row.get("cohort") == "replacement"
    )
    def _accounted(pid: str) -> bool:
        return pid in by_participant or pid in withdrawn or pid in duplicates

    # v6.2 slot semantics: a slot is accounted for (and a replacement slot is
    # ACTIVATED) by a submitted session OR a withdrawn/duplicate entry; usable
    # judgments still come only from included submitted sessions.
    replacements_activated = [pid for pid in replacement_ids if _accounted(pid)]
    # v6.2: every primary slot must be accounted for, independent of
    # replacement presence and of the usable-judgment count.
    unaccounted_primaries = [pid for pid in primary_ids if not _accounted(pid)]
    if unaccounted_primaries:
        raise ValueError(
            "authoritative analysis requires the full primary cohort; "
            f"unaccounted primary slots: {unaccounted_primaries}"
        )
    def _usable_of(ids: list[str]) -> int:
        return sum(
            sum(
                1
                for row in by_participant[pid]
                if str(row["role"]) == "study"
            )
            for pid in ids
            if pid in by_participant and pid not in exclusions
        )

    if replacements_activated:
        if replacements_activated != replacement_ids[: len(replacements_activated)]:
            raise ValueError(
                f"activated replacements {replacements_activated} are not a "
                f"contiguous ID-ordered prefix of {replacement_ids}"
            )
        for index, replacement in enumerate(replacements_activated):
            prior = _usable_of(primary_ids + replacements_activated[:index])
            if prior >= target:
                raise ValueError(
                    f"{replacement} activated although usable judgments from prior "
                    f"participants ({prior}) already met the target ({target}); "
                    "target-based stopping violated"
                )

    # Amendment v6.3 terminal-state rule: freezing responses is legal only at
    # target, or at valid exhaustion (every replacement slot accounted for).
    usable_total = _usable_of(primary_ids + replacement_ids)
    unactivated_replacements = [pid for pid in replacement_ids if not _accounted(pid)]
    if usable_total < target and unactivated_replacements:
        raise ValueError(
            "premature response freeze: usable judgments "
            f"({usable_total}) are below the target ({target}) while replacement "
            f"slots remain unactivated: {unactivated_replacements}; collection "
            "must continue per the frozen replacement procedure"
        )

    primary_rows = []
    repeat_pairs = []
    preferences = Counter()
    for participant in included:
        seen: dict[str, str] = {}
        for row in by_participant[participant]:
            item_id = str(row["item_id"])
            role = str(row["role"])
            answer = str(row["answer"])
            if role == "study":
                primary_rows.append(
                    {
                        "participant_id": participant,
                        "item_id": item_id,
                        "correct": float(answer == answers[item_id]),
                    }
                )
                seen[item_id] = answer
                if "preference" in row:
                    preferences[str(row["preference"])] += 1
            elif role == "repeat-consistency":
                repeat_pairs.append((participant, item_id, answer))
    consistent = sum(
        1
        for participant, item_id, answer in repeat_pairs
        if any(
            row["participant_id"] == participant
            and row["item_id"] == item_id
            for row in primary_rows
        )
        and answer
        == next(
            str(r["answer"])
            for r in by_participant[participant]
            if str(r["item_id"]) == item_id and str(r["role"]) == "study"
        )
    )

    correct = np.array([row["correct"] for row in primary_rows])
    participants_column = np.array([row["participant_id"] for row in primary_rows])
    items_column = np.array([row["item_id"] for row in primary_rows])
    primary = _two_way_ucb(correct, participants_column, items_column)
    target_met = primary["n"] >= target
    bound_met = primary["ucb_95_one_sided"] <= CEILING
    # The frozen stopping rule fixes the usable-judgment target; an
    # under-target sample must not authorize the tier no matter what its
    # bound says.
    quality_go = target_met and bound_met

    strata: dict[str, dict[str, float]] = {}
    for key_fn, bucket in (
        (lambda item: item_meta[item]["stratum"], strata),
    ):
        for stratum in sorted({key_fn(item) for item in items_column}):
            mask = np.array([key_fn(item) == stratum for item in items_column])
            bucket[stratum] = _two_way_ucb(
                correct[mask], participants_column[mask], items_column[mask]
            )
    horizons: dict[str, dict[str, float]] = {}
    for horizon in sorted({item_meta[item]["horizon_seconds"] for item in items_column}):
        mask = np.array(
            [item_meta[item]["horizon_seconds"] == horizon for item in items_column]
        )
        horizons[str(horizon)] = _two_way_ucb(
            correct[mask], participants_column[mask], items_column[mask]
        )

    verdict_v1 = _object(args.v1_verdict)
    lpips_by_job = {
        record["job_id"]: record["pairwise"]["pairwise_lpips_mean"]
        for record in verdict_v1["job_records"]
    }
    item_accuracy = {}
    item_lpips = {}
    for item in sorted(set(items_column)):
        mask = items_column == item
        item_accuracy[item] = float(correct[mask].mean())
        item_lpips[item] = float(lpips_by_job[item_meta[item]["job_id"]])
    accuracy_values = np.array([item_accuracy[item] for item in sorted(item_accuracy)])
    lpips_values = np.array([item_lpips[item] for item in sorted(item_accuracy)])
    correlation = float(np.corrcoef(lpips_values, accuracy_values)[0, 1])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    participant_list = sorted(set(participants_column))
    boot = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = rng.choice(participant_list, size=len(participant_list), replace=True)
        mask = np.isin(participants_column, sample)
        per_item = {}
        for item in sorted(set(items_column[mask])):
            item_mask = mask & (items_column == item)
            per_item[item] = float(correct[item_mask].mean())
        if len(per_item) < 3:
            continue
        boot.append(
            float(
                np.corrcoef(
                    [item_lpips[item] for item in sorted(per_item)],
                    [per_item[item] for item in sorted(per_item)],
                )[0, 1]
            )
        )
    prompt_of_item = {
        item: item_meta[item]["job_id"].rsplit("-", 1)[0] for item in item_accuracy
    }
    loo = []
    for prompt in sorted(set(prompt_of_item.values())):
        kept = [item for item in sorted(item_accuracy) if prompt_of_item[item] != prompt]
        loo.append(
            float(
                np.corrcoef(
                    [item_lpips[item] for item in kept],
                    [item_accuracy[item] for item in kept],
                )[0, 1]
            )
        )

    result = {
        "schema": "liveframe.taew21-study-analysis.v1",
        "bundle_manifest_sha256": file_sha256(args.bundle_root / "bundle-manifest.json"),
        "responses_sha256": file_sha256(args.responses),
        "power_sha256": file_sha256(args.power),
        "v1_verdict_sha256": file_sha256(args.v1_verdict),
        "tool_sha256": file_sha256(Path(__file__).resolve()),
        "exclusions": exclusions,
        "included_participants": included,
        "usable_judgments": primary["n"],
        "usable_judgment_target": target,
        "target_met": bool(target_met),
        "primary": primary,
        "noticeability_ceiling": CEILING,
        "quality_go": bool(quality_go),
        "verdict": (
            "tier-quality-GO-discrimination-bounded"
            if quality_go
            else (
                "no-authority-insufficient-usable-judgments"
                if not target_met
                else "tier-quality-NOT-ESTABLISHED-discrimination-bound-not-shown"
            )
        ),
        "verdict_semantics": (
            "GO means the one-sided 95% clustered upper bound on discrimination "
            "accuracy is at or below the ceiling at the frozen sample target. A "
            "non-GO does NOT prove discrimination exceeds the ceiling; it means "
            "the bound was not established (the pre-registered design accepts "
            "roughly a one-in-four inconclusive outcome even at true chance)."
        ),
        "per_stratum": strata,
        "per_horizon": horizons,
        "repeat_consistency": {
            "pairs": len(repeat_pairs),
            "consistent": consistent,
            "rate": consistent / len(repeat_pairs) if repeat_pairs else None,
        },
        "preference_counts": dict(preferences),
        "secondary_metric_correlation": {
            "instrument": "pairwise_lpips_mean (v1 verdict)",
            "pearson_r_item_level": correlation,
            "bootstrap": {
                "draws": len(boot),
                "seed": BOOTSTRAP_SEED,
                "p05": float(np.quantile(boot, 0.05)) if boot else None,
                "p95": float(np.quantile(boot, 0.95)) if boot else None,
            },
            "leave_one_prompt_out": {
                "min": float(np.min(loo)),
                "max": float(np.max(loo)),
            },
            "authority": "calibration-evidence-only",
        },
    }
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "usable_judgments": primary["n"],
                "accuracy": primary["accuracy"],
                "ucb": primary["ucb_95_one_sided"],
                "quality_go": bool(quality_go),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
