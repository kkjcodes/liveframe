"""Render the blinded TAEW2.1 discrimination-study bundle.

Runs ONLY on a metric-screen PASS. Produces, without opening decoder labels
into the participant-facing materials:

- 32 study items, one per frozen campaign pair: `item-XX/reference.mp4` (the
  first-generation full-decoder encode) and `item-XX/A.mp4` / `item-XX/B.mp4`
  (one full and one tiny decode of the same latents, side assignment
  randomized per item by a seeded RNG). BOTH candidates are uniform
  second-generation transcodes (identical command applied to full.mp4 and
  tiny.mp4), so no bundle file is byte-identical to any other, hash-matching
  against the reference cannot reveal the answer, and the extra codec
  generation is symmetric across arms (no side cue; common-mode for the
  discrimination task). The builder VERIFIES that every digest in the bundle
  is unique and aborts otherwise.
- Two attention-check items reused from distinct pairs: the correct option is
  a second-generation transcode of the reference (perceptually matching, not
  byte-identical) and the foil is an objectively degraded re-encode (16x
  spatial round-trip), so failure is unambiguous.
- `participant-schedules.json`: 30 primary participants (P00-P29) plus 6
  pre-registered replacement participants (R30-R35, activated strictly in ID
  order, one per excluded primary), each x (18 study trials + 2 attention
  checks + 2 repeat-consistency duplicates), balanced incomplete-block over
  the 32 items per the frozen power artifact.
- `attention-key.json`: the correct sides of the two attention items ONLY
  (copy-vs-degraded; reveals nothing about decoder identity). Enables the
  sealed-free pre-freeze QC tool.
- `bundle-manifest.json`: structure and video digests, NO decoder labels.
  Manifests, the attention key, schedules, and sealed files are NEVER
  distributed to participants — participants receive item directories and
  their own trial order only.
- `assignments.sealed.json`: the item -> (A,B) decoder labels, stored
  digest-bound and pre-registered as closed until the response freeze. The
  primary task is reference-then-A/B source matching; the correct answer is
  the side holding the full decode.

Randomization seed is frozen here before any participant exists.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.manifests import file_sha256  # noqa: E402

BUNDLE_SEED = 20260819
PARTICIPANTS = 30
REPLACEMENTS = 6
TRIALS = 18
REPEAT_TRIALS = 2
ATTENTION_ITEMS = 2


def _transcode(source: Path, output: Path, tag: str) -> None:
    """Uniform second-generation transcode applied identically to both sides.

    Every transcode embeds a unique container-metadata tag (the item/side
    identity, already visible in the filename) so its bytes are distinct even
    if the encoder is bit-deterministic for identical inputs — digest
    uniqueness must not depend on encoder nondeterminism.
    """

    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "h264_videotoolbox",
        "-b:v",
        "8M",
        "-metadata",
        f"title={tag}",
        str(output),
    ]
    subprocess.run(command, check=True)


def _degrade(source: Path, output: Path, tag: str) -> None:
    # Carries the same style of metadata tag as every transcode: no bundle
    # video may be distinguishable by presence/absence of container metadata.
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        "scale=104:60:flags=bilinear,scale=832:480:flags=bilinear",
        "-an",
        "-c:v",
        "h264_videotoolbox",
        "-b:v",
        "8M",
        "-metadata",
        f"title={tag}",
        str(output),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--metric-verdict", type=Path, required=True)
    parser.add_argument("--power", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    verdict = json.loads(args.metric_verdict.read_text(encoding="utf-8"))
    # Amendment v6: a v2 routing screen PASS authorizes the study in place of a
    # v1 verdict PASS (the v1 FAIL remains on record; see the amendment's
    # honesty disclosure).
    if verdict.get("schema") == "liveframe.taew21-metric-verdict.v1":
        if not verdict.get("passed") or verdict.get("protocol_sha256") != file_sha256(
            args.protocol
        ):
            raise ValueError("study bundle requires a PASSED metric verdict")
    elif verdict.get("schema") == "liveframe.taew21-metric-screen.v2":
        if (
            not verdict.get("passed")
            or verdict.get("protocol_sha256") != file_sha256(args.protocol)
            or verdict.get("verdict") != "route-to-human-study"
        ):
            raise ValueError("study bundle requires a route-to-human-study v2 screen")
    else:
        raise ValueError("study bundle requires a v1 verdict or v2 screen artifact")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("schema") != "liveframe.taew21-campaign-summary.v1" or file_sha256(
        args.summary
    ) != verdict.get("summary_sha256"):
        raise ValueError("study bundle requires the verdict-bound campaign summary")
    power = json.loads(args.power.read_text(encoding="utf-8"))
    if (
        power.get("schema") != "liveframe.taew21-study-power.v1"
        or power.get("protocol_sha256") != file_sha256(args.protocol)
    ):
        raise ValueError("study bundle requires the bound power simulation")
    design = power["selected_design"]
    if design["participants"] != PARTICIPANTS or design["trials_per_participant"] != TRIALS:
        raise ValueError("frozen bundle constants do not match the selected design")
    amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
    if (
        amendment.get("schema") != "liveframe.taew21-protocol-amendment.v6.3"
        or amendment["bindings"]["protocol_sha256"] != file_sha256(args.protocol)
        or amendment["bindings"]["power_v1_sha256"] != file_sha256(args.power)
    ):
        raise ValueError("study bundle requires the bound amendment v6.3")
    # Mechanically verify the full amendment ancestry chain, hop by hop, at
    # the canonical artifact paths: v6.3 -> v6.2 -> v6.1 -> v6. Without this,
    # a v6.3 file with an altered ancestry digest could be bound into the
    # manifest and the chain would be declarative at that hop.
    current = amendment
    for binding_name, ancestor_name in (
        ("amendment_v6_2_sha256", "taew21-protocol-amendment.v6.2.json"),
        ("amendment_v6_1_sha256", "taew21-protocol-amendment.v6.1.json"),
        ("amendment_v6_sha256", "taew21-protocol-amendment.v6.json"),
    ):
        ancestor_path = REPOSITORY_ROOT / "artifacts" / ancestor_name
        if current["bindings"][binding_name] != file_sha256(ancestor_path):
            raise ValueError(
                f"amendment ancestry mismatch: {binding_name} does not match "
                f"{ancestor_name}"
            )
        current = json.loads(ancestor_path.read_text(encoding="utf-8"))

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty bundle root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(BUNDLE_SEED)
    jobs = protocol["campaign"]["jobs"]
    if len(jobs) != 32:
        raise ValueError("campaign must contain exactly 32 jobs")
    items: list[dict[str, Any]] = []
    sealed: list[dict[str, Any]] = []
    for index, job in enumerate(jobs):
        job_dir = REPOSITORY_ROOT / str(job["output"])
        full_video = job_dir / "full.mp4"
        tiny_video = job_dir / "tiny.mp4"
        item_id = f"item-{index:02d}"
        item_dir = args.output_root / item_id
        item_dir.mkdir()
        shutil.copyfile(full_video, item_dir / "reference.mp4")
        full_side = "A" if rng.random() < 0.5 else "B"
        tiny_side = "B" if full_side == "A" else "A"
        _transcode(full_video, item_dir / f"{full_side}.mp4", f"{item_id}/{full_side}")
        _transcode(tiny_video, item_dir / f"{tiny_side}.mp4", f"{item_id}/{tiny_side}")
        items.append(
            {
                "item_id": item_id,
                "kind": "study",
                "job_id": job["job_id"],
                "stratum": job["stratum"],
                "horizon_seconds": job["horizon_seconds"],
                "reference_sha256": file_sha256(item_dir / "reference.mp4"),
                "a_sha256": file_sha256(item_dir / "A.mp4"),
                "b_sha256": file_sha256(item_dir / "B.mp4"),
            }
        )
        sealed.append(
            {
                "item_id": item_id,
                "job_id": job["job_id"],
                "full_side": full_side,
                "tiny_side": tiny_side,
                "correct_answer": full_side,
            }
        )

    attention_source_indices = rng.choice(len(jobs), size=ATTENTION_ITEMS, replace=False)
    attention_items = []
    for rank, source_index in enumerate(sorted(int(i) for i in attention_source_indices)):
        job = jobs[source_index]
        job_dir = REPOSITORY_ROOT / str(job["output"])
        item_id = f"attention-{rank:02d}"
        item_dir = args.output_root / item_id
        item_dir.mkdir()
        # The attention reference is itself a tagged transcode: a byte-copy of
        # full.mp4 would collide with the study item built from the same job.
        _transcode(job_dir / "full.mp4", item_dir / "reference.mp4", f"{item_id}/reference")
        copy_side = "A" if rng.random() < 0.5 else "B"
        foil_side = "B" if copy_side == "A" else "A"
        _transcode(job_dir / "full.mp4", item_dir / f"{copy_side}.mp4", f"{item_id}/{copy_side}")
        _degrade(job_dir / "full.mp4", item_dir / f"{foil_side}.mp4", f"{item_id}/{foil_side}")
        attention_items.append(
            {
                "item_id": item_id,
                "kind": "attention",
                "source_job_id": job["job_id"],
                "reference_sha256": file_sha256(item_dir / "reference.mp4"),
                "a_sha256": file_sha256(item_dir / "A.mp4"),
                "b_sha256": file_sha256(item_dir / "B.mp4"),
            }
        )
        sealed.append(
            {
                "item_id": item_id,
                "job_id": job["job_id"],
                "attention_correct": copy_side,
                "correct_answer": copy_side,
            }
        )

    digest_rows = [
        (row["item_id"], name, row[f"{name}_sha256"])
        for row in items + attention_items
        for name in ("reference", "a", "b")
    ]
    seen_digests: dict[str, tuple[str, str]] = {}
    for item_id, name, digest in digest_rows:
        if digest in seen_digests:
            raise RuntimeError(
                f"bundle digest collision: {item_id}/{name} matches "
                f"{seen_digests[digest][0]}/{seen_digests[digest][1]}; a shared "
                "digest would let hash-matching reveal the answer key"
            )
        seen_digests[digest] = (item_id, name)

    schedules = []
    for slot in range(PARTICIPANTS + REPLACEMENTS):
        offset = (slot * TRIALS) % len(items)
        study_ids = [
            items[(offset + position) % len(items)]["item_id"]
            for position in range(TRIALS)
        ]
        repeats = [
            study_ids[int(value)]
            for value in rng.choice(TRIALS, size=REPEAT_TRIALS, replace=False)
        ]
        sequence = (
            [{"item_id": item, "role": "study"} for item in study_ids]
            + [{"item_id": row["item_id"], "role": "attention"} for row in attention_items]
            + [{"item_id": item, "role": "repeat-consistency"} for item in repeats]
        )
        order = rng.permutation(len(sequence))
        participant_id = (
            f"P{slot:02d}" if slot < PARTICIPANTS else f"R{slot:02d}"
        )
        schedules.append(
            {
                "participant_id": participant_id,
                "cohort": "primary" if slot < PARTICIPANTS else "replacement",
                "trials": [sequence[int(position)] for position in order],
            }
        )

    sealed_payload = {
        "schema": "liveframe.taew21-study-assignments.v1",
        "status": "sealed-until-response-freeze",
        "seed": BUNDLE_SEED,
        "assignments": sealed,
    }
    sealed_path = args.output_root / "assignments.sealed.json"
    sealed_path.write_text(
        json.dumps(sealed_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    schedules_path = args.output_root / "participant-schedules.json"
    schedules_path.write_text(
        json.dumps(
            {
                "schema": "liveframe.taew21-study-schedules.v1",
                "participants": PARTICIPANTS,
                "trials_per_participant": TRIALS,
                "repeat_trials": REPEAT_TRIALS,
                "attention_items": ATTENTION_ITEMS,
                "usable_judgment_target": design["usable_judgment_target"],
                "replacements": REPLACEMENTS,
                "replacement_rule": (
                    "replacement participants R30-R35 are activated strictly in "
                    "ID order, exactly one per excluded primary participant; a "
                    "replacement uses its OWN pre-registered schedule row and its "
                    "own participant_id"
                ),
                "schedules": schedules,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    attention_key_path = args.output_root / "attention-key.json"
    attention_key_path.write_text(
        json.dumps(
            {
                "schema": "liveframe.taew21-study-attention-key.v1",
                "note": (
                    "correct sides of the ATTENTION items only (copy vs degraded); "
                    "reveals nothing about decoder identity; used by the sealed-free "
                    "pre-freeze QC tool; never distributed to participants"
                ),
                "answers": {
                    row["item_id"]: row["correct_answer"]
                    for row in sealed
                    if "attention_correct" in row
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "liveframe.taew21-study-bundle.v1",
        "blinding": (
            "decoder labels appear ONLY in assignments.sealed.json, pre-registered "
            "closed until the response freeze; participant materials and analysis "
            "labels use item/side identifiers only; every bundle video is a "
            "distinct encode (verified unique digests) so hash-matching cannot "
            "recover the answer key"
        ),
        "distribution_rule": (
            "participants receive item directories and their own trial order "
            "ONLY; manifests, schedules, the attention key, and sealed files are "
            "never participant-accessible"
        ),
        "candidate_encoding": (
            "both A/B candidates are uniform second-generation "
            "h264_videotoolbox 8M transcodes (identical command for both sides); "
            "the extra generation is symmetric across arms"
        ),
        "seed": BUNDLE_SEED,
        "protocol_sha256": file_sha256(args.protocol),
        "metric_verdict_sha256": file_sha256(args.metric_verdict),
        "power_sha256": file_sha256(args.power),
        "amendment_sha256": file_sha256(args.amendment),
        "builder_sha256": file_sha256(Path(__file__).resolve()),
        "items": items + attention_items,
        "sealed_assignments_sha256": file_sha256(sealed_path),
        "schedules_sha256": file_sha256(schedules_path),
        "attention_key_sha256": file_sha256(attention_key_path),
    }
    manifest_path = args.output_root / "bundle-manifest.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(
        json.dumps(
            {
                "output": str(args.output_root),
                "items": len(items),
                "attention_items": len(attention_items),
                "participants": PARTICIPANTS,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
