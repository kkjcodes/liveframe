from __future__ import annotations

import json
from pathlib import Path

import pytest

from liveframe.manifests import (
    ManifestError,
    file_sha256,
    load_upstream_manifest,
    verify_local_file,
)

ROOT = Path(__file__).parents[1]


def test_phase_zero_manifest_has_immutable_public_pins() -> None:
    manifest = load_upstream_manifest(ROOT / "artifacts" / "upstream.v1.json")

    assert manifest.status == "pinned"
    assert manifest.unresolved_ids == ()
    manifest.require_resolved()
    assert manifest.artifact("causal-forcing-source").revision == (
        "8db419e341e5fc52542c0b2c4542728420ddfb4a"
    )


def test_local_checksum_must_be_frozen_and_match(tmp_path: Path) -> None:
    payload = tmp_path / "fixture.bin"
    payload.write_bytes(b"golden")
    digest = file_sha256(payload)
    source = ROOT / "artifacts" / "upstream.v1.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    row = raw["artifacts"][0]
    row["local_sha256"] = digest
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    artifact = load_upstream_manifest(path).artifact("causal-forcing-source")

    assert verify_local_file(artifact, payload) == digest
    payload.write_bytes(b"tampered")
    with pytest.raises(ManifestError, match="checksum mismatch"):
        verify_local_file(artifact, payload)
