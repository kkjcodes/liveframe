"""Inspect pinned H100 refresh hook surfaces without generating proof results."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

SOURCE_REVISION = "8db419e341e5fc52542c0b2c4542728420ddfb4a"
CHECKPOINT_SHA256 = "bdb1b475fc88d528f510158a0990cd457f02c661b527ceb11ca9e4728533e2d0"
EXPECTED_GPU = "NVIDIA H100 80GB HBM3"
DENOMINATOR_DRIVER = "580.159.04"
SOURCE_FILES = (
    Path("wan/modules/causal_model.py"),
    Path("utils/wan_wrapper.py"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _signature(node: ast.FunctionDef) -> str:
    arguments = [argument.arg for argument in node.args.args]
    defaults = [None] * (len(arguments) - len(node.args.defaults)) + node.args.defaults
    rendered = []
    for name, default in zip(arguments, defaults, strict=True):
        rendered.append(name if default is None else f"{name}={ast.unparse(default)}")
    return "(" + ", ".join(rendered) + ")"


def _source_signatures(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    signatures: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "causal_rope_apply":
            signatures[node.name] = _signature(node)
        if isinstance(node, ast.ClassDef) and node.name == "CausalWanSelfAttention":
            forward = next(
                child
                for child in node.body
                if isinstance(child, ast.FunctionDef) and child.name == "forward"
            )
            signatures[f"{node.name}.forward"] = _signature(forward)
    if set(signatures) != {"causal_rope_apply", "CausalWanSelfAttention.forward"}:
        raise ValueError("refresh preflight could not resolve framewise hook signatures")
    return signatures


def _validate_source(source: str) -> None:
    markers = (
        "class CausalWanSelfAttention",
        "causal_rope_apply",
        "cache_start",
        "global_end_index",
        "local_end_index",
    )
    missing = [marker for marker in markers if marker not in source]
    if missing:
        raise ValueError(f"refresh preflight source markers are missing: {missing}")


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    revision = _git("rev-parse", "HEAD")
    if revision != SOURCE_REVISION:
        raise ValueError("refresh preflight requires the pinned Causal-Forcing revision")
    if _sha256(args.checkpoint) != "sha256:" + CHECKPOINT_SHA256:
        raise ValueError("refresh preflight checkpoint digest mismatch")
    for path in SOURCE_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"refresh preflight source file is missing: {path}")
        if _git("hash-object", str(path)) != _git("rev-parse", f"HEAD:{path}"):
            raise ValueError(f"refresh preflight source differs from pinned Git blob: {path}")
    causal_source = SOURCE_FILES[0].read_text(encoding="utf-8")
    _validate_source(causal_source)

    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != EXPECTED_GPU:
        raise ValueError("refresh preflight requires NVIDIA H100 80GB HBM3")
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    report = {
        "schema": "liveframe.h100-refresh-source-preflight.v1",
        "claim_eligible": False,
        "proof_results_generated": False,
        "source_revision": revision,
        "source_files": {str(path): _sha256(path) for path in SOURCE_FILES},
        "source_git_blobs": {
            str(path): _git("rev-parse", f"HEAD:{path}") for path in SOURCE_FILES
        },
        "signatures": _source_signatures(causal_source),
        "tracked_changes_outside_source_files": _git(
            "status", "--short", "--untracked-files=no"
        ).splitlines(),
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "driver": driver,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "checkpoint_sha256": "sha256:" + CHECKPOINT_SHA256,
            "preflight_harness_sha256": _sha256(Path(__file__)),
            "denominator_driver": DENOMINATOR_DRIVER,
            "denominator_driver_match": driver == DENOMINATOR_DRIVER,
        },
    }
    if args.output.exists():
        if _object(args.output) != report:
            raise FileExistsError(f"refusing to replace refresh preflight: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()