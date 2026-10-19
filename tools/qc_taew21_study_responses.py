"""Blinded pre-freeze QC for the TAEW2.1 study: usable-judgment accounting.

Safe to run DURING data collection: reads only the participant schedules and
the attention key (attention items are copy-vs-degraded and reveal nothing
about decoder identity). It never touches `assignments.sealed.json` and never
computes any accuracy on study items, so it cannot un-blind the analysis or
enable outcome peeking. It answers exactly one question: after the
pre-registered exclusions, how many usable judgments exist, and how many
replacement participants (strictly in R-ID order) still need to be recruited
to reach the frozen target.

Amendment v6.1 enforcement: the supplied power artifact is verified against
the bundle manifest's `power_sha256` and its target against the schedules'
pinned target; observed replacements are checked against the frozen
target-based stopping rule (all primaries first, contiguous ID-ordered
prefix, each activation necessary) and violations are reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.manifests import file_sha256  # noqa: E402


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--power", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    args = parser.parse_args()

    manifest = _object(args.bundle_root / "bundle-manifest.json")
    schedules_path = args.bundle_root / "participant-schedules.json"
    attention_key_path = args.bundle_root / "attention-key.json"
    if file_sha256(schedules_path) != manifest["schedules_sha256"]:
        raise ValueError("schedules digest mismatch vs bundle manifest")
    if file_sha256(attention_key_path) != manifest["attention_key_sha256"]:
        raise ValueError("attention key digest mismatch vs bundle manifest")
    schedules = _object(schedules_path)
    attention_answers = _object(attention_key_path)["answers"]
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
    if _object(args.amendment).get("schema") != "liveframe.taew21-protocol-amendment.v6.3":
        raise ValueError("unsupported amendment schema")

    payload = _object(args.responses)
    if payload.get("schema") != "liveframe.taew21-study-responses.v1":
        raise ValueError("unsupported response schema")
    withdrawn = set(payload.get("withdrawn_participants", []))
    duplicates = set(payload.get("duplicate_participants", []))
    scheduled = {row["participant_id"]: row for row in schedules["schedules"]}
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
    by_participant: dict[str, list[dict[str, Any]]] = {}
    for row in payload["responses"]:
        by_participant.setdefault(str(row["participant_id"]), []).append(row)

    statuses: dict[str, str] = {}
    usable = 0
    for participant in sorted(by_participant):
        if participant not in scheduled:
            raise ValueError(f"unscheduled participant in responses: {participant}")
        rows = by_participant[participant]
        if participant in withdrawn:
            statuses[participant] = "excluded:withdrawn-consent"
            continue
        if participant in duplicates:
            statuses[participant] = "excluded:duplicate-participant"
            continue
        expected = Counter(
            (trial["item_id"], trial["role"])
            for trial in scheduled[participant]["trials"]
        )
        observed = Counter((str(row["item_id"]), str(row["role"])) for row in rows)
        if observed != expected:
            statuses[participant] = "excluded:incomplete-session"
            continue
        attention_wrong = any(
            str(row["role"]) == "attention"
            and str(row["answer"]) != attention_answers[str(row["item_id"])]
            for row in rows
        )
        if attention_wrong:
            statuses[participant] = "excluded:failed-predeclared-attention-check"
            continue
        statuses[participant] = "included"
        usable += sum(1 for row in rows if str(row["role"]) == "study")

    # Absent-row withdrawn/duplicate scheduled slots are excluded slots too
    # (amendment v6.3 accounting parity with the analyzer's exclusion report).
    scheduled_ids = {row["participant_id"] for row in schedules["schedules"]}
    for participant in sorted(withdrawn | duplicates):
        if participant in scheduled_ids and participant not in by_participant:
            statuses[participant] = (
                "excluded:withdrawn-consent"
                if participant in withdrawn
                else "excluded:duplicate-participant"
            )

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

    # A replacement slot is ACTIVATED once it is accounted for in any way —
    # a submitted session OR a withdrawn/duplicate entry (v6.2 slot
    # semantics). Usable judgments still come only from included sessions.
    replacements_activated = [pid for pid in replacement_ids if _accounted(pid)]

    def _usable_of(ids: list[str]) -> int:
        return sum(
            sum(1 for row in by_participant[pid] if str(row["role"]) == "study")
            for pid in ids
            if pid in by_participant and statuses.get(pid) == "included"
        )

    unaccounted_primaries = [pid for pid in primary_ids if not _accounted(pid)]
    violations: list[str] = []
    if replacements_activated:
        if unaccounted_primaries:
            violations.append(
                "replacements present before all primary slots were accounted "
                f"for: missing {unaccounted_primaries}"
            )
        if replacements_activated != replacement_ids[: len(replacements_activated)]:
            violations.append(
                f"activated replacements {replacements_activated} are not a "
                f"contiguous ID-ordered prefix of {replacement_ids}"
            )
        for index, replacement in enumerate(replacements_activated):
            prior = _usable_of(primary_ids + replacements_activated[:index])
            if prior >= target:
                violations.append(
                    f"{replacement} activated although usable judgments from prior "
                    f"participants ({prior}) already met the target ({target})"
                )

    excluded = sum(1 for status in statuses.values() if status != "included")
    all_primaries_accounted = not unaccounted_primaries
    target_met = usable >= target
    replacements_exhausted = (
        len(replacements_activated) == len(replacement_ids) and not target_met
    )
    # Fail-closed routing field: freezing responses is authorized ONLY by
    # collection_complete, never by raw target_met. Per amendment v6.3, valid
    # exhaustion also completes collection, solely for the analyzer's
    # no-authority verdict.
    collection_complete = (
        all_primaries_accounted
        and (target_met or replacements_exhausted)
        and not violations
    )
    next_replacement = (
        []
        if target_met or not all_primaries_accounted
        else [pid for pid in replacement_ids if not _accounted(pid)][:1]
    )
    report = {
        "schema": "liveframe.taew21-study-qc.v1",
        "sealed_assignments_read": False,
        "study_accuracy_computed": False,
        "participants_responded": len(by_participant),
        "participants_included": sum(
            1 for status in statuses.values() if status == "included"
        ),
        "participants_excluded": excluded,
        "statuses": statuses,
        "usable_judgments": usable,
        "usable_judgment_target": target,
        "target_met": target_met,
        "all_primaries_accounted": all_primaries_accounted,
        "unaccounted_primaries": unaccounted_primaries,
        "replacements_activated": replacements_activated,
        "replacements_exhausted": replacements_exhausted,
        "next_replacement_in_order": next_replacement,
        "policy_violations": violations,
        "collection_complete": collection_complete,
        "guidance": (
            "account for all 30 primary slots first (submitted session, or a "
            "withdrawn/duplicate entry; a recruit who never starts is re-slotted "
            "to the same ID); if target is unmet, activate exactly one "
            "replacement at a time, strictly in ID order, re-running this QC "
            "between activations; freeze responses ONLY at the first "
            "collection_complete: true (never on raw target_met); if "
            "replacements are exhausted below target the study returns "
            "no-authority"
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
