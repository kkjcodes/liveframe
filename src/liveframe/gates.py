"""Immutable gate artifact loading, canonicalization, and verification."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

CANONICALIZATION = "rfc8785-compatible-json-v1"


class GateError(ValueError):
    """Raised when a gate is unknown, malformed, or has been altered."""


def canonical_json(value: Mapping[str, Any]) -> bytes:
    """Return the project's frozen canonical JSON encoding.

    Gate contracts intentionally contain no floating-point values. Sorted UTF-8 JSON with
    compact separators is therefore stable across supported Python runtimes.
    """
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def contract_digest(contract: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(contract)).hexdigest()


@dataclass(frozen=True, slots=True)
class GateArtifact:
    gate_id: str
    digest: str
    contract: Mapping[str, Any]
    source: Path

    def verify(self) -> None:
        actual = contract_digest(self.contract)
        if self.digest != actual:
            raise GateError(f"gate digest mismatch: expected {self.digest}, computed {actual}")
        _validate_contract(self.gate_id, self.contract)


def _validate_contract(gate_id: str, contract: Mapping[str, Any]) -> None:
    if gate_id != "gates.v1":
        raise GateError(f"unknown gate ID: {gate_id}")
    required = {"status", "temporal", "latency", "primary_proof", "secondary_proof", "rejections"}
    missing = sorted(required - contract.keys())
    if missing:
        raise GateError(f"gate contract missing fields: {', '.join(missing)}")
    if contract["status"] != "frozen":
        raise GateError("gate contract is not frozen")
    temporal = cast(Mapping[str, Any], contract["temporal"])
    proof = cast(Mapping[str, Any], contract["primary_proof"])
    latency = cast(Mapping[str, Any], contract["latency"])
    if temporal.get("native_pixel_fps") != 16:
        raise GateError("gates.v1 requires native 16 FPS")
    if temporal.get("pixel_frames_formula") != "4*(latent_frames-1)+1":
        raise GateError("gates.v1 latent-to-pixel formula changed")
    if latency.get("warm_ttff_max_ms") != 3000:
        raise GateError("gates.v1 warm TTFF threshold changed")
    if proof.get("duration_seconds") != 60 or proof.get("min_generated_frames") != 960:
        raise GateError("gates.v1 primary proof duration/frame threshold changed")
    if proof.get("max_display_underruns") != 0:
        raise GateError("gates.v1 requires zero display underruns")


def load_gate(path: Path) -> GateArtifact:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot load gate artifact {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("canonicalization") != CANONICALIZATION:
        raise GateError("unsupported gate artifact or canonicalization")
    contract = raw.get("contract")
    if not isinstance(contract, dict):
        raise GateError("gate contract must be an object")
    artifact = GateArtifact(
        gate_id=str(raw.get("gate_id", "")),
        digest=str(raw.get("digest", "")),
        contract=contract,
        source=path,
    )
    artifact.verify()
    return artifact


def load_bundled_gate(gate_id: str = "gates.v1") -> GateArtifact:
    if gate_id != "gates.v1":
        raise GateError(f"unknown bundled gate ID: {gate_id}")
    resource = files("liveframe").joinpath("data").joinpath("gates").joinpath(f"{gate_id}.json")
    with resource.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    contract = raw.get("contract")
    if not isinstance(contract, dict):
        raise GateError("bundled gate contract must be an object")
    artifact = GateArtifact(
        gate_id=str(raw.get("gate_id", "")),
        digest=str(raw.get("digest", "")),
        contract=contract,
        source=Path(str(resource)),
    )
    if raw.get("canonicalization") != CANONICALIZATION:
        raise GateError("unsupported bundled gate canonicalization")
    artifact.verify()
    return artifact
