from __future__ import annotations

import json
from pathlib import Path

import pytest

from liveframe.gates import GateError, load_bundled_gate, load_gate


def test_bundled_gate_is_frozen_and_digest_bound() -> None:
    gate = load_bundled_gate()

    assert gate.gate_id == "gates.v1"
    assert gate.digest == "sha256:809cba47d33942e6ee7fabf0943647eced12a8f61a2568fe719d0233fb375813"
    assert gate.contract["status"] == "frozen"


def test_tampered_gate_is_rejected(tmp_path: Path) -> None:
    gate = load_bundled_gate()
    raw = json.loads(gate.source.read_text(encoding="utf-8"))
    raw["contract"]["primary_proof"]["min_generated_frames"] = 959
    path = tmp_path / "gates.v1.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(GateError, match="digest mismatch"):
        load_gate(path)


def test_unknown_gate_is_rejected(tmp_path: Path) -> None:
    gate = load_bundled_gate()
    raw = json.loads(gate.source.read_text(encoding="utf-8"))
    raw["gate_id"] = "gates.v2"
    path = tmp_path / "gates.v2.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(GateError, match="unknown gate ID"):
        load_gate(path)
