from __future__ import annotations

import json
from pathlib import Path

from liveframe.cli import main
from liveframe.qualification_ledger import load_qualification_ledger


def test_synthetic_qualification_cli_reports_missing_evidence(
    capsys: object,
    tmp_path: Path,
) -> None:
    output = tmp_path / "qualification.json"
    result = main(
        [
            "qualify",
            "--backend",
            "synthetic",
            "--profile",
            "interactive-480p",
            "--seed",
            "7",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert payload == load_qualification_ledger(output)
    assert payload["schema"] == "liveframe.qualification-run.v1"
    assert payload["request"]["seed"] == 7
    assert payload["metrics"]["generated_native_frames"] == 960
    assert payload["qualification"]["verdict"] == "incomplete"
    missing = {
        gate["name"]
        for gate in payload["qualification"]["gates"]
        if gate["status"] == "not_evaluated"
    }
    assert missing == {
        "quality",
        "long_horizon_stability",
        "memory_eligibility",
        "thermal_stability",
    }

    assert main(["qualification", "verify", str(output)]) == 0
    verification = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert verification == {
        "ledger_digest": payload["ledger_digest"],
        "schema": "liveframe.qualification-run.v1",
        "verified": True,
    }