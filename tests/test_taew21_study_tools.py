"""Regression tests for the TAEW2.1 study QC and analysis tools.

Pins the launch-blocker repairs: power-artifact digest binding, target-based
replacement enforcement, the under-target no-authority verdict, and (against
the real bundle when present) media digest uniqueness and attention-candidate
metadata parity.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.manifests import file_sha256  # noqa: E402

QC_TOOL = REPOSITORY_ROOT / "tools" / "qc_taew21_study_responses.py"
ANALYZER = REPOSITORY_ROOT / "tools" / "analyze_taew21_study_responses.py"
BUNDLE = REPOSITORY_ROOT / "outputs" / "taew21-study-bundle"

STUDY_ITEMS = [
    ("item-00", "taew21-alpha-15s", "s1", 15),
    ("item-01", "taew21-alpha-60s", "s1", 60),
    ("item-02", "taew21-beta-15s", "s2", 15),
    ("item-03", "taew21-beta-60s", "s2", 60),
]
ATTENTION_ITEM = "attention-00"
PRIMARIES = ["P00", "P01", "P02"]
REPLACEMENTS = ["R03", "R04"]
TRIALS_PER_PARTICIPANT = 4  # study trials; plus 1 attention + 1 repeat
TARGET = 12  # 3 primaries x 4 study trials


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_fixture(tmp_path: Path, target: int = TARGET) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    sealed_rows = []
    items = []
    for index, (item_id, job_id, stratum, horizon) in enumerate(STUDY_ITEMS):
        full_side = "A" if index % 2 == 0 else "B"
        sealed_rows.append(
            {
                "item_id": item_id,
                "job_id": job_id,
                "full_side": full_side,
                "tiny_side": "B" if full_side == "A" else "A",
                "correct_answer": full_side,
            }
        )
        items.append(
            {
                "item_id": item_id,
                "kind": "study",
                "job_id": job_id,
                "stratum": stratum,
                "horizon_seconds": horizon,
            }
        )
    sealed_rows.append(
        {
            "item_id": ATTENTION_ITEM,
            "job_id": STUDY_ITEMS[0][1],
            "attention_correct": "A",
            "correct_answer": "A",
        }
    )
    items.append(
        {
            "item_id": ATTENTION_ITEM,
            "kind": "attention",
            "source_job_id": STUDY_ITEMS[0][1],
        }
    )
    schedules = []
    for slot, participant in enumerate(PRIMARIES + REPLACEMENTS):
        study_ids = [
            STUDY_ITEMS[(slot + position) % len(STUDY_ITEMS)][0]
            for position in range(TRIALS_PER_PARTICIPANT)
        ]
        trials = (
            [{"item_id": item, "role": "study"} for item in study_ids]
            + [{"item_id": ATTENTION_ITEM, "role": "attention"}]
            + [{"item_id": study_ids[0], "role": "repeat-consistency"}]
        )
        schedules.append(
            {
                "participant_id": participant,
                "cohort": "primary" if participant in PRIMARIES else "replacement",
                "trials": trials,
            }
        )
    _write(bundle / "assignments.sealed.json", {"assignments": sealed_rows})
    _write(
        bundle / "participant-schedules.json",
        {
            "usable_judgment_target": target,
            "schedules": schedules,
        },
    )
    _write(bundle / "attention-key.json", {"answers": {ATTENTION_ITEM: "A"}})
    power_path = tmp_path / "power.json"
    _write(
        power_path,
        {
            "schema": "liveframe.taew21-study-power.v1",
            "selected_design": {"usable_judgment_target": target},
        },
    )
    amendment_path = tmp_path / "amendment.json"
    _write(
        amendment_path,
        {"schema": "liveframe.taew21-protocol-amendment.v6.3"},
    )
    _write(
        bundle / "bundle-manifest.json",
        {
            "schema": "liveframe.taew21-study-bundle.v1",
            "items": items,
            "sealed_assignments_sha256": file_sha256(bundle / "assignments.sealed.json"),
            "schedules_sha256": file_sha256(bundle / "participant-schedules.json"),
            "attention_key_sha256": file_sha256(bundle / "attention-key.json"),
            "power_sha256": file_sha256(power_path),
            "amendment_sha256": file_sha256(amendment_path),
        },
    )
    verdict_path = tmp_path / "v1-verdict.json"
    _write(
        verdict_path,
        {
            "job_records": [
                {"job_id": job_id, "pairwise": {"pairwise_lpips_mean": 0.01 * (index + 1)}}
                for index, (_, job_id, _, _) in enumerate(STUDY_ITEMS)
            ]
        },
    )
    return bundle, power_path


def _responses(
    participants: list[str],
    bundle: Path,
    *,
    correct_probability: float = 0.5,
    fail_attention: set[str] | None = None,
    seed: int = 5,
) -> dict:
    sealed = json.loads((bundle / "assignments.sealed.json").read_text())
    answers = {row["item_id"]: row["correct_answer"] for row in sealed["assignments"]}
    schedules = json.loads((bundle / "participant-schedules.json").read_text())
    scheduled = {row["participant_id"]: row["trials"] for row in schedules["schedules"]}
    rng = np.random.default_rng(seed)
    rows = []
    for participant in participants:
        for trial in scheduled[participant]:
            item, role = trial["item_id"], trial["role"]
            good = answers[item]
            wrong = "B" if good == "A" else "A"
            if role == "attention":
                answer = wrong if participant in (fail_attention or set()) else good
            else:
                answer = good if rng.random() < correct_probability else wrong
            rows.append(
                {
                    "participant_id": participant,
                    "item_id": item,
                    "role": role,
                    "answer": answer,
                }
            )
    return {
        "schema": "liveframe.taew21-study-responses.v1",
        "withdrawn_participants": [],
        "duplicate_participants": [],
        "responses": rows,
    }


def _run_qc(bundle: Path, responses_path: Path, power_path: Path, amendment: Path | None = None):
    return subprocess.run(
        [
            sys.executable,
            str(QC_TOOL),
            "--bundle-root",
            str(bundle),
            "--responses",
            str(responses_path),
            "--power",
            str(power_path),
            "--amendment",
            str(amendment if amendment is not None else bundle.parent / "amendment.json"),
        ],
        capture_output=True,
        text=True,
    )


def _run_analyzer(
    bundle: Path,
    responses_path: Path,
    power_path: Path,
    verdict_path: Path,
    output: Path,
    amendment: Path | None = None,
):
    return subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--bundle-root",
            str(bundle),
            "--responses",
            str(responses_path),
            "--v1-verdict",
            str(verdict_path),
            "--power",
            str(power_path),
            "--amendment",
            str(amendment if amendment is not None else bundle.parent / "amendment.json"),
            "--responses-frozen",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )


def test_altered_power_artifact_is_rejected(tmp_path):
    bundle, _power_path = _build_fixture(tmp_path)
    responses_path = tmp_path / "responses.json"
    _write(responses_path, _responses(PRIMARIES, bundle))
    tampered = tmp_path / "power-tampered.json"
    _write(
        tampered,
        {
            "schema": "liveframe.taew21-study-power.v1",
            "selected_design": {"usable_judgment_target": 1},
        },
    )
    qc = _run_qc(bundle, responses_path, tampered)
    assert qc.returncode != 0
    assert "power artifact digest mismatch" in (qc.stderr + qc.stdout)
    analysis = _run_analyzer(
        bundle, responses_path, tampered, tmp_path / "v1-verdict.json", tmp_path / "a.json"
    )
    assert analysis.returncode != 0
    assert "power artifact digest mismatch" in (analysis.stderr + analysis.stdout)


def test_out_of_order_replacement_is_rejected(tmp_path):
    bundle, power_path = _build_fixture(tmp_path)
    responses_path = tmp_path / "responses.json"
    # P00 fails attention (excluded, usable 8 < 12) but the SECOND replacement
    # is used while the first is skipped.
    _write(
        responses_path,
        _responses([*PRIMARIES, "R04"], bundle, fail_attention={"P00"}),
    )
    analysis = _run_analyzer(
        bundle, responses_path, power_path, tmp_path / "v1-verdict.json", tmp_path / "a.json"
    )
    assert analysis.returncode != 0
    assert "contiguous ID-ordered prefix" in (analysis.stderr + analysis.stdout)
    qc = _run_qc(bundle, responses_path, power_path)
    assert qc.returncode == 0
    assert any(
        "contiguous ID-ordered prefix" in violation
        for violation in json.loads(qc.stdout)["policy_violations"]
    )


def test_unnecessary_replacement_is_rejected(tmp_path):
    bundle, power_path = _build_fixture(tmp_path)
    responses_path = tmp_path / "responses.json"
    # No exclusions: primaries alone give 12 >= 12, so R03 must not exist.
    _write(responses_path, _responses([*PRIMARIES, "R03"], bundle))
    analysis = _run_analyzer(
        bundle, responses_path, power_path, tmp_path / "v1-verdict.json", tmp_path / "a.json"
    )
    assert analysis.returncode != 0
    assert "target-based stopping violated" in (analysis.stderr + analysis.stdout)


def test_justified_replacement_reaches_go(tmp_path):
    bundle, power_path = _build_fixture(tmp_path)
    responses_path = tmp_path / "responses.json"
    # P00 excluded (attention) -> 8 usable < 12 -> R03 justified -> 12 usable.
    # Zero correct answers force accuracy 0, so the UCB rule passes at target.
    _write(
        responses_path,
        _responses(
            [*PRIMARIES, "R03"],
            bundle,
            correct_probability=0.0,
            fail_attention={"P00"},
        ),
    )
    output = tmp_path / "analysis.json"
    analysis = _run_analyzer(
        bundle, responses_path, power_path, tmp_path / "v1-verdict.json", output
    )
    assert analysis.returncode == 0, analysis.stderr
    report = json.loads(output.read_text())
    assert report["target_met"] is True
    assert report["usable_judgments"] == TARGET
    assert report["quality_go"] is True
    assert report["verdict"] == "tier-quality-GO-discrimination-bounded"
    assert report["exclusions"] == {"P00": "failed-predeclared-attention-check"}


def test_under_target_sample_has_no_authority(tmp_path):
    bundle, power_path = _build_fixture(tmp_path)
    responses_path = tmp_path / "responses.json"
    # All primaries accounted, two fail attention (4 usable < 12), and BOTH
    # replacement slots are accounted (submitted R03 also fails attention;
    # R04 withdrew) - a valid terminal exhaustion state. Zero-accuracy answers
    # would satisfy the bound, but the target gate must refuse authority.
    payload = _responses(
        [*PRIMARIES, "R03"],
        bundle,
        correct_probability=0.0,
        fail_attention={"P00", "P01", "R03"},
    )
    payload["withdrawn_participants"] = ["R04"]
    _write(responses_path, payload)
    output = tmp_path / "analysis.json"
    analysis = _run_analyzer(
        bundle, responses_path, power_path, tmp_path / "v1-verdict.json", output
    )
    assert analysis.returncode == 0, analysis.stderr
    report = json.loads(output.read_text())
    assert report["target_met"] is False
    assert report["quality_go"] is False
    assert report["verdict"] == "no-authority-insufficient-usable-judgments"


def test_premature_under_target_freeze_is_rejected(tmp_path):
    bundle, power_path = _build_fixture(tmp_path)
    responses_path = tmp_path / "responses.json"
    # Under target (4 usable < 12) with BOTH replacement slots unactivated:
    # not a terminal state, so authoritative analysis must be refused even
    # though the sample would only have yielded a no-authority verdict.
    _write(
        responses_path,
        _responses(
            PRIMARIES,
            bundle,
            correct_probability=0.0,
            fail_attention={"P00", "P01"},
        ),
    )
    output = tmp_path / "analysis.json"
    analysis = _run_analyzer(
        bundle, responses_path, power_path, tmp_path / "v1-verdict.json", output
    )
    assert analysis.returncode != 0
    assert "premature response freeze" in (analysis.stderr + analysis.stdout)
    assert not output.exists()
    qc = _run_qc(bundle, responses_path, power_path)
    assert qc.returncode == 0, qc.stderr
    report = json.loads(qc.stdout)
    assert report["collection_complete"] is False
    assert report["next_replacement_in_order"] == ["R03"]


def test_qc_recommends_single_next_replacement(tmp_path):
    bundle, power_path = _build_fixture(tmp_path)
    responses_path = tmp_path / "responses.json"
    _write(responses_path, _responses(PRIMARIES, bundle, fail_attention={"P00"}))
    qc = _run_qc(bundle, responses_path, power_path)
    assert qc.returncode == 0, qc.stderr
    report = json.loads(qc.stdout)
    assert report["target_met"] is False
    assert report["policy_violations"] == []
    assert report["next_replacement_in_order"] == ["R03"]


def test_target_met_with_missing_primary_blocks_authority(tmp_path):
    # Target 8 is reachable from two of the three primaries: the exact state
    # the third review round demonstrated (target met, cohort incomplete).
    bundle, power_path = _build_fixture(tmp_path, target=8)
    responses_path = tmp_path / "responses.json"
    _write(responses_path, _responses(["P00", "P01"], bundle, correct_probability=0.0))
    output = tmp_path / "analysis.json"
    analysis = _run_analyzer(
        bundle, responses_path, power_path, tmp_path / "v1-verdict.json", output
    )
    assert analysis.returncode != 0
    assert "unaccounted primary slots" in (analysis.stderr + analysis.stdout)
    assert not output.exists()
    qc = _run_qc(bundle, responses_path, power_path)
    assert qc.returncode == 0, qc.stderr
    report = json.loads(qc.stdout)
    assert report["target_met"] is True
    assert report["all_primaries_accounted"] is False
    assert report["unaccounted_primaries"] == ["P02"]
    assert report["collection_complete"] is False


def test_withdrawn_primary_accounts_for_its_slot(tmp_path):
    bundle, power_path = _build_fixture(tmp_path, target=8)
    responses_path = tmp_path / "responses.json"
    payload = _responses(["P00", "P01"], bundle, correct_probability=0.0)
    payload["withdrawn_participants"] = ["P02"]
    _write(responses_path, payload)
    output = tmp_path / "analysis.json"
    analysis = _run_analyzer(
        bundle, responses_path, power_path, tmp_path / "v1-verdict.json", output
    )
    assert analysis.returncode == 0, analysis.stderr
    report = json.loads(output.read_text())
    assert report["exclusions"] == {"P02": "withdrawn-consent"}
    assert report["target_met"] is True
    assert report["quality_go"] is True
    qc = _run_qc(bundle, responses_path, power_path)
    assert json.loads(qc.stdout)["collection_complete"] is True


def test_withdrawn_replacement_slot_counts_as_activated(tmp_path):
    bundle, power_path = _build_fixture(tmp_path)
    responses_path = tmp_path / "responses.json"
    # Legal sequence: P00 excluded (8 < 12), R03 withdraws WITHOUT rows,
    # R04 responds. R03's slot is activated by its withdrawn entry, so R04 is
    # a valid contiguous successor, not an out-of-order violation.
    payload = _responses(
        [*PRIMARIES, "R04"],
        bundle,
        correct_probability=0.0,
        fail_attention={"P00"},
    )
    payload["withdrawn_participants"] = ["R03"]
    _write(responses_path, payload)
    qc = _run_qc(bundle, responses_path, power_path)
    assert qc.returncode == 0, qc.stderr
    report = json.loads(qc.stdout)
    assert report["policy_violations"] == []
    assert report["replacements_activated"] == ["R03", "R04"]
    assert report["collection_complete"] is True
    output = tmp_path / "analysis.json"
    analysis = _run_analyzer(
        bundle, responses_path, power_path, tmp_path / "v1-verdict.json", output
    )
    assert analysis.returncode == 0, analysis.stderr
    result = json.loads(output.read_text())
    assert result["quality_go"] is True
    assert result["exclusions"]["R03"] == "withdrawn-consent"


def test_replacement_exhaustion_is_recognized(tmp_path):
    bundle, power_path = _build_fixture(tmp_path)
    responses_path = tmp_path / "responses.json"
    # P00 and P01 excluded (4 usable < 12); both replacements withdraw
    # without rows. The pool is exhausted: QC must not recommend another
    # replacement and must mark collection complete for a no-authority freeze.
    payload = _responses(
        PRIMARIES,
        bundle,
        correct_probability=0.0,
        fail_attention={"P00", "P01"},
    )
    payload["withdrawn_participants"] = ["R03", "R04"]
    _write(responses_path, payload)
    qc = _run_qc(bundle, responses_path, power_path)
    assert qc.returncode == 0, qc.stderr
    report = json.loads(qc.stdout)
    assert report["policy_violations"] == []
    assert report["replacements_activated"] == ["R03", "R04"]
    assert report["replacements_exhausted"] is True
    assert report["next_replacement_in_order"] == []
    assert report["target_met"] is False
    assert report["collection_complete"] is True
    assert report["statuses"]["R03"] == "excluded:withdrawn-consent"
    assert report["statuses"]["R04"] == "excluded:withdrawn-consent"
    assert report["participants_excluded"] == 4
    output = tmp_path / "analysis.json"
    analysis = _run_analyzer(
        bundle, responses_path, power_path, tmp_path / "v1-verdict.json", output
    )
    assert analysis.returncode == 0, analysis.stderr
    result = json.loads(output.read_text())
    assert result["verdict"] == "no-authority-insufficient-usable-judgments"


def test_contradictory_exclusion_metadata_is_rejected(tmp_path):
    bundle, power_path = _build_fixture(tmp_path)
    responses_path = tmp_path / "responses.json"
    # The demonstrated attack: every replacement ID in BOTH lists would
    # manufacture a valid-looking exhaustion state. Must hard-error instead.
    payload = _responses(PRIMARIES, bundle, fail_attention={"P00", "P01"})
    payload["withdrawn_participants"] = ["R03", "R04"]
    payload["duplicate_participants"] = ["R03", "R04"]
    _write(responses_path, payload)
    qc = _run_qc(bundle, responses_path, power_path)
    assert qc.returncode != 0
    assert "contradictory exclusion metadata" in (qc.stderr + qc.stdout)
    analysis = _run_analyzer(
        bundle, responses_path, power_path, tmp_path / "v1-verdict.json", tmp_path / "a.json"
    )
    assert analysis.returncode != 0
    assert "contradictory exclusion metadata" in (analysis.stderr + analysis.stdout)
    assert not (tmp_path / "a.json").exists()


def test_unscheduled_exclusion_metadata_is_rejected(tmp_path):
    bundle, power_path = _build_fixture(tmp_path)
    responses_path = tmp_path / "responses.json"
    payload = _responses(PRIMARIES, bundle)
    payload["withdrawn_participants"] = ["P99"]
    _write(responses_path, payload)
    qc = _run_qc(bundle, responses_path, power_path)
    assert qc.returncode != 0
    assert "unscheduled participants" in (qc.stderr + qc.stdout)
    analysis = _run_analyzer(
        bundle, responses_path, power_path, tmp_path / "v1-verdict.json", tmp_path / "a.json"
    )
    assert analysis.returncode != 0
    assert "unscheduled participants" in (analysis.stderr + analysis.stdout)
    assert not (tmp_path / "a.json").exists()


def test_builder_rejects_altered_ancestry_chain(tmp_path):
    # A v6.3-shaped amendment with correct protocol/power digests but a wrong
    # v6.2 ancestry digest must be refused by the builder's chain walk.
    import copy

    real = json.loads(
        (REPOSITORY_ROOT / "artifacts" / "taew21-protocol-amendment.v6.3.json").read_text()
    )
    forged = copy.deepcopy(real)
    forged["bindings"]["amendment_v6_2_sha256"] = "sha256:" + "0" * 64
    forged_path = tmp_path / "forged-v63.json"
    _write(forged_path, forged)
    build = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "tools" / "build_taew21_study_bundle.py"),
            "--protocol",
            str(REPOSITORY_ROOT / "artifacts" / "taew21-protocol.v1.json"),
            "--authorization",
            str(REPOSITORY_ROOT / "outputs" / "taew21-campaign-authorization.v1.json"),
            "--summary",
            str(REPOSITORY_ROOT / "outputs" / "taew21-campaign" / "campaign-summary.v1.json"),
            "--metric-verdict",
            str(REPOSITORY_ROOT / "artifacts" / "taew21-metric-screen.v2.json"),
            "--power",
            str(REPOSITORY_ROOT / "artifacts" / "taew21-study-power.v1.json"),
            "--amendment",
            str(forged_path),
            "--output-root",
            str(tmp_path / "bundle-out"),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode != 0
    assert "amendment ancestry mismatch" in (build.stderr + build.stdout)
    assert not (tmp_path / "bundle-out" / "bundle-manifest.json").exists()


def test_altered_amendment_is_rejected(tmp_path):
    bundle, power_path = _build_fixture(tmp_path)
    responses_path = tmp_path / "responses.json"
    _write(responses_path, _responses(PRIMARIES, bundle))
    tampered = tmp_path / "amendment-tampered.json"
    _write(tampered, {"schema": "liveframe.taew21-protocol-amendment.v6.3", "extra": 1})
    qc = _run_qc(bundle, responses_path, power_path, amendment=tampered)
    assert qc.returncode != 0
    assert "amendment digest mismatch" in (qc.stderr + qc.stdout)
    analysis = _run_analyzer(
        bundle,
        responses_path,
        power_path,
        tmp_path / "v1-verdict.json",
        tmp_path / "a.json",
        amendment=tampered,
    )
    assert analysis.returncode != 0
    assert "amendment digest mismatch" in (analysis.stderr + analysis.stdout)


@pytest.mark.skipif(not BUNDLE.exists(), reason="rendered study bundle not present")
def test_bundle_manifest_binds_current_amendment():
    manifest = json.loads((BUNDLE / "bundle-manifest.json").read_text())
    amendment_path = REPOSITORY_ROOT / "artifacts" / "taew21-protocol-amendment.v6.3.json"
    assert manifest["amendment_sha256"] == file_sha256(amendment_path)


@pytest.mark.skipif(not BUNDLE.exists(), reason="rendered study bundle not present")
def test_bundle_media_digests_are_unique_and_leak_free():
    digests: dict[str, str] = {}
    item_dirs = sorted(path for path in BUNDLE.iterdir() if path.is_dir())
    videos = 0
    for item_dir in item_dirs:
        for video in sorted(item_dir.glob("*.mp4")):
            digest = hashlib.sha256(video.read_bytes()).hexdigest()
            assert digest not in digests, f"{video} collides with {digests[digest]}"
            digests[digest] = str(video)
            videos += 1
    assert videos == 102
    for item_dir in item_dirs:
        reference = hashlib.sha256((item_dir / "reference.mp4").read_bytes()).hexdigest()
        for side in ("A", "B"):
            option = hashlib.sha256((item_dir / f"{side}.mp4").read_bytes()).hexdigest()
            assert option != reference, f"{item_dir.name}/{side} equals its reference"


@pytest.mark.skipif(not BUNDLE.exists(), reason="rendered study bundle not present")
def test_attention_candidates_have_equivalent_metadata():
    item_dirs = [path for path in sorted(BUNDLE.glob("attention-*")) if path.is_dir()]
    assert len(item_dirs) == 2
    for item_dir in item_dirs:
        titles = {}
        for side in ("A", "B"):
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format_tags=title",
                    "-of",
                    "json",
                    str(item_dir / f"{side}.mp4"),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            tags = json.loads(probe.stdout).get("format", {}).get("tags", {})
            titles[side] = tags.get("title")
        assert titles["A"] and titles["B"], (
            f"{item_dir.name}: both candidates must carry a metadata tag; got {titles}"
        )
        assert titles["A"] != titles["B"]
        for side, title in titles.items():
            assert title == f"{item_dir.name}/{side}"
