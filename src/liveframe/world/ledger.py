"""World run ledgers: every file in a run directory, digest-bound and verifiable.

``liveframe.world.ledger.v1`` reuses the repository's canonicalization (the
same helper the gate and qualification ledgers use) so a ledger carries a
self-digest, and verification re-hashes every listed file fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from liveframe.gates import CANONICALIZATION, contract_digest

WORLD_LEDGER_SCHEMA = "liveframe.world.ledger.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def build_world_ledger(
    run_directory: Path, *, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Hash every file under the run directory (except the ledger itself)."""

    if not run_directory.is_dir():
        raise ValueError(f"run directory does not exist: {run_directory}")
    files: dict[str, str] = {}
    for path in sorted(run_directory.rglob("*")):
        if not path.is_file() or path.name == "ledger.json" or path.name.startswith("."):
            continue
        files[path.relative_to(run_directory).as_posix()] = _sha256(path)
    if not files:
        raise ValueError(f"run directory holds no files to bind: {run_directory}")
    ledger: dict[str, Any] = {
        "schema": WORLD_LEDGER_SCHEMA,
        "canonicalization": CANONICALIZATION,
        "metadata": metadata or {},
        "files": files,
    }
    ledger["ledger_digest"] = contract_digest(ledger)
    return ledger


def write_world_ledger(
    run_directory: Path, *, metadata: dict[str, Any] | None = None
) -> Path:
    ledger = build_world_ledger(run_directory, metadata=metadata)
    path = run_directory / "ledger.json"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def verify_world_ledger(ledger_path: Path) -> dict[str, Any]:
    """Fail closed on schema, self-digest, or any file digest mismatch."""

    raw = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != WORLD_LEDGER_SCHEMA:
        raise ValueError("unsupported world ledger schema")
    recorded = raw.get("ledger_digest")
    unsigned = {key: value for key, value in raw.items() if key != "ledger_digest"}
    if contract_digest(unsigned) != recorded:
        raise ValueError("world ledger self-digest mismatch")
    root = ledger_path.parent
    files = raw["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("world ledger lists no files")
    for relative, expected in sorted(files.items()):
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"ledger path escapes the run directory: {relative}")
        target = root / candidate
        if not target.is_file():
            raise ValueError(f"ledger file missing: {relative}")
        actual = _sha256(target)
        if actual != expected:
            raise ValueError(
                f"digest mismatch for {relative}: expected {expected}, got {actual}"
            )
    return raw
