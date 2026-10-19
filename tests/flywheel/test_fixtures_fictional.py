"""Fictional-data-only enforcement (addendum A section 6).

Fails CI when fixture provenance, origins, PII hygiene, or data-source rules
are violated. The scan roots are tests/fixtures/, the suite scenes directory,
and examples/; test-support code under tests/data/ is source code, not
fixture data, and is covered by the PII scan below.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO / "tests" / "fixtures" / "MANIFEST.json"
FIXTURE_ROOTS = (
    REPO / "tests" / "fixtures",
    REPO / "src" / "liveframe" / "world" / "suites" / "scenes",
    REPO / "examples",
)
TEXT_SCAN_ROOTS = (
    REPO / "tests" / "fixtures",
    REPO / "tests" / "data",
    REPO / "src" / "liveframe" / "world" / "suites",
    REPO / "examples",
)
ALLOWED_ORIGINS = {"procedural", "self-made", "cc0"}
FORBIDDEN_DATASET_MARKERS = ("open bandit", "obd_", "wbench", "worldmark")

# PII patterns (kept in one module so they can be extended).
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"),
    "phone": re.compile(r"(?<![\w/.-])\+?\d[\d\s().-]{8,}\d(?![\w.-])"),
    "street-address": re.compile(
        r"\b\d{1,5}\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln)\b",
        re.IGNORECASE,
    ),
    "government-id": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}
URL_PATTERN = re.compile(r"https?://([\w.-]+)")
URL_ALLOWLIST = (
    "localhost",
    "example.com",
    "example.org",
    "github.com",
    "huggingface.co",
    "json-schema.org",
)


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _fixture_files() -> list[Path]:
    files: list[Path] = []
    for root in FIXTURE_ROOTS:
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.name not in ("MANIFEST.json", "LICENSES.md")
            and "__pycache__" not in path.parts
        )
    return files


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_fixture_is_manifested_with_allowed_origin() -> None:
    manifest = _manifest()
    entries = {entry["path"]: entry for entry in manifest["entries"]}
    for path in _fixture_files():
        relative = path.relative_to(REPO).as_posix()
        assert relative in entries, f"unmanifested fixture: {relative}"
        entry = entries[relative]
        assert entry["origin"] in ALLOWED_ORIGINS, relative
        assert entry.get("contains_people") is False, relative
        if entry["origin"] == "procedural":
            assert entry.get("generator"), f"{relative}: procedural needs generator"
        if entry["origin"] == "cc0":
            assert entry.get("source_url") and entry.get("license"), relative
        assert entry["sha256"] == _sha256(path), f"hash mismatch: {relative}"
    # Manifest must not reference files that no longer exist.
    for listed in entries:
        assert (REPO / listed).is_file(), f"manifest lists missing file: {listed}"


def test_no_pii_or_disallowed_urls_in_text_fixtures() -> None:
    violations: list[str] = []
    for root in TEXT_SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in (".json", ".jsonl", ".txt", ".md", ".py"):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            relative = path.relative_to(REPO).as_posix()
            for name, pattern in PII_PATTERNS.items():
                if pattern.search(text):
                    violations.append(f"{relative}: {name}")
            for host in URL_PATTERN.findall(text):
                allowed_host = any(
                    host == allowed or host.endswith("." + allowed)
                    for allowed in URL_ALLOWLIST
                )
                if not allowed_host:
                    violations.append(f"{relative}: url host {host}")
    assert not violations, violations


def test_no_external_benchmark_data_committed() -> None:
    offenders: list[str] = []
    for root in FIXTURE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            lowered = path.name.lower()
            if any(marker in lowered for marker in FORBIDDEN_DATASET_MARKERS):
                offenders.append(path.relative_to(REPO).as_posix())
    assert not offenders, offenders


def test_network_access_is_blocked_during_tests() -> None:
    import socket

    import pytest

    with pytest.raises(RuntimeError, match="network access is blocked"):
        socket.create_connection(("example.com", 80), timeout=1)
