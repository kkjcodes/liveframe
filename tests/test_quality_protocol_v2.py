from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from liveframe.perceptual_policy import canonical_digest
from liveframe.quality_calibration import derive_policy_bundle, load_protocol
from liveframe.quality_capture import calibration_sanity_report

ROOT = Path(__file__).parents[1]
PROTOCOL = ROOT / "artifacts" / "perceptual-protocol.v2.json"
CALIBRATION = ROOT / "artifacts" / "h100-calibration-teacher.v1.json"


def _loaded() -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol, digest = load_protocol(PROTOCOL)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    return protocol, digest, calibration


def test_v2_is_bound_to_failed_v1_calibration() -> None:
    protocol, _, calibration = _loaded()

    assert protocol["source_protocol_digest"] == calibration["protocol_digest"]
    assert protocol["source_calibration_digest"] == canonical_digest(calibration)
    assert protocol["source_sanity_digest"] == (
        "sha256:320953225c1e4775ec10285d27ec38f48a504f0cba4fd89161ea828f11e3f6e3"
    )


def test_v2_sanity_demotes_all_epsilon_dominated_cells() -> None:
    protocol, _, calibration = _loaded()

    report = calibration_sanity_report(protocol, calibration)
    diagnostics = [row for row in report["cells"] if not row["authoritative"]]

    assert report["passed"] is True
    assert report["failures"] == []
    assert len(diagnostics) == 39
    assert all(row["epsilon_dominated"] for row in diagnostics)


def test_v2_bundle_omits_diagnostics_and_preserves_coverage() -> None:
    protocol, digest, calibration = _loaded()

    bundle = derive_policy_bundle(protocol, digest, calibration)

    for stratum in bundle["policies"].values():
        for prompt in stratum["prompts"].values():
            for policy in prompt["horizons"].values():
                authoritative = set(policy["quality_metrics"])
                diagnostic = set(policy["diagnostic_metrics"])
                assert authoritative.isdisjoint(diagnostic)
                assert set(policy["authoritative_coverage"]) == {
                    "semantic-fidelity",
                    "frame-dynamics",
                    "temporal-behavior",
                }
                assert all(policy["authoritative_coverage"].values())
                assert len(policy["identity_tripwires"]) == 3


def test_v2_rejects_another_calibration_input() -> None:
    protocol, digest, calibration = _loaded()
    altered = copy.deepcopy(calibration)
    altered["teacher_videos"][0]["metrics"]["clip_prompt_mean"] += 0.001

    with pytest.raises(ValueError, match="another calibration input"):
        derive_policy_bundle(protocol, digest, altered)
