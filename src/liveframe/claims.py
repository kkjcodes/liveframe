"""Publication-claims ledger verification (the PyPI liveframe 0.2.0 verifier).

The verify and recompute command implementations are carried over
verbatim from the published liveframe 0.2.0 cli.py so their behavior,
output, and exit codes are unchanged; the unified CLI wires them as
subcommands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

LEDGER_SCHEMA = "liveframe.publication-claims.v1"
SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _load_ledger(path: Path) -> dict[str, Any]:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict):
        raise ValueError("claims ledger must be a JSON object")
    if ledger.get("schema") != LEDGER_SCHEMA:
        raise ValueError(
            f"unsupported claims ledger schema: {ledger.get('schema', '<missing>')}"
        )
    return ledger


def _iter_artifact_refs(node: object, refs: dict[str, str]) -> dict[str, str]:
    if isinstance(node, dict):
        artifact = node.get("artifact")
        digest = node.get("sha256")
        if artifact is not None or digest is not None:
            if not isinstance(artifact, str) or not isinstance(digest, str):
                raise ValueError(
                    "artifact references require string artifact and sha256 fields"
                )
            if not SHA256_PATTERN.fullmatch(digest):
                raise ValueError(f"invalid SHA-256 digest for artifact: {artifact}")
            previous = refs.get(artifact)
            if previous is not None and previous != digest:
                raise ValueError(f"conflicting digests for artifact: {artifact}")
            refs[artifact] = digest
        for value in node.values():
            _iter_artifact_refs(value, refs)
    elif isinstance(node, list):
        for value in node:
            _iter_artifact_refs(value, refs)
    return refs


def cmd_verify(args: argparse.Namespace) -> int:
    ledger_path = Path(args.ledger)
    ledger = _load_ledger(ledger_path)
    print(f"ledger: {ledger_path}  schema: {ledger['schema']}")
    root = (
        Path(args.artifacts_root) if args.artifacts_root else ledger_path.parent.parent
    ).resolve()
    refs = _iter_artifact_refs(ledger, {})
    if not refs:
        print("FAIL: ledger references no digest-bound artifacts")
        return 2
    failures = 0
    for rel, expected in sorted(refs.items()):
        relative = Path(rel)
        if relative.is_absolute() or ".." in relative.parts:
            print(f"  INVALID  {rel} (artifact path must remain below root)")
            failures += 1
            continue
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            print(f"  INVALID  {rel} (artifact path resolves outside root)")
            failures += 1
            continue
        if not target.is_file():
            print(f"  MISSING  {rel}")
            failures += 1
            continue
        actual = _sha256(target)
        status = "ok" if actual == expected else "DIGEST MISMATCH"
        if actual != expected:
            failures += 1
            print(f"  {status}  {rel}\n    expected {expected}\n    actual   {actual}")
        else:
            print(f"  ok       {rel}")
    if failures:
        print(
            f"FAIL: {failures}/{len(refs)} artifact reference(s) "
            "did not verify (fail-closed)"
        )
        return 1
    print(f"PASS: all {len(refs)} unique digest-bound artifacts verified")
    return 0


def _number(container: Mapping[str, Any], name: str, context: str) -> float:
    value = container.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}.{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}.{name} must be finite")
    return result


def cmd_recompute(args: argparse.Namespace) -> int:
    ledger = _load_ledger(Path(args.ledger))
    problems = 0
    checks = 0
    engineering_speedups = ledger.get("engineering_speedups")
    negative_results = ledger.get("negative_results")
    if not isinstance(engineering_speedups, list):
        raise ValueError("engineering_speedups must be an array")
    if not isinstance(negative_results, list):
        raise ValueError("negative_results must be an array")
    for claim in engineering_speedups:
        if not isinstance(claim, dict):
            raise ValueError("each engineering speedup must be an object")
        base, cand = claim.get("baseline", {}), claim.get("candidate", {})
        if not isinstance(base, dict) or not isinstance(cand, dict):
            raise ValueError("engineering speedup baseline and candidate must be objects")
        claim_id = str(claim.get("id", "<missing>"))
        bw = _number(base, "wall_seconds", f"{claim_id}.baseline")
        cw = _number(cand, "wall_seconds", f"{claim_id}.candidate")
        if bw <= 0 or cw <= 0:
            raise ValueError(f"{claim_id} wall times must be positive")
        speedup = bw / cw
        reduction = (bw - cw) / bw
        bv = _number(base, "vae_seconds", f"{claim_id}.baseline")
        cv = _number(cand, "vae_seconds", f"{claim_id}.candidate")
        if bv <= 0 or cv < 0:
            raise ValueError(f"{claim_id} VAE times must be non-negative with positive baseline")
        vae_reduction = (bv - cv) / bv
        for name, computed in (
            ("wall_speedup", speedup),
            ("wall_reduction_fraction", reduction),
            ("vae_reduction_fraction", vae_reduction),
        ):
            recorded = _number(claim, name, claim_id)
            ok = abs(recorded - computed) < 1e-9
            checks += 1
            problems += 0 if ok else 1
            print(
                f"  {claim_id}: {name} recorded={recorded:.9f} "
                f"recomputed={computed:.9f} {'ok' if ok else 'MISMATCH'}"
            )
    for neg in negative_results:
        if not isinstance(neg, dict):
            raise ValueError("each negative result must be an object")
        ev = neg.get("evidence", {})
        if not isinstance(ev, dict) or not any(
            name in ev
            for name in ("generated_fps", "target_fps", "target_gap_factor")
        ):
            continue
        claim_id = str(neg.get("id", "<missing>"))
        fps = _number(ev, "generated_fps", f"{claim_id}.evidence")
        target = _number(ev, "target_fps", f"{claim_id}.evidence")
        gap = _number(ev, "target_gap_factor", f"{claim_id}.evidence")
        if fps <= 0 or target <= 0:
            raise ValueError(f"{claim_id} FPS values must be positive")
        computed = target / fps
        ok = abs(gap - computed) < 1e-9
        checks += 1
        problems += 0 if ok else 1
        print(
            f"  {claim_id}: target_gap_factor recorded={gap:.9f} "
            f"recomputed={computed:.9f} {'ok' if ok else 'MISMATCH'}"
        )
    if checks == 0:
        print("FAIL: ledger contains no supported arithmetic claims")
        return 2
    if problems:
        print(f"FAIL: {problems} arithmetic inconsistency(ies)")
        return 1
    print("PASS: ledger arithmetic reproduces")
    return 0
