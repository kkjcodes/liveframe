"""Publish the immutable 64-job H100 framewise refresh-addressability batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.quality_artifacts import atomic_json  # noqa: E402
from liveframe.quality_calibration import load_protocol  # noqa: E402
from liveframe.refresh_capture import (  # noqa: E402
    load_capture_batch,
    prepare_capture_batch,
)
from liveframe.refresh_proof import load_refresh_proof_manifest  # noqa: E402


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--policy-bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol, protocol_digest = load_protocol(args.protocol)
    bundle = _object(args.policy_bundle)
    manifest = load_refresh_proof_manifest(
        args.manifest, protocol, protocol_digest, bundle
    )
    batch = prepare_capture_batch(
        manifest, args.result_root, include_completed=True
    )
    if len(batch["jobs"]) != 64:
        raise ValueError("frozen H100 refresh campaign must contain exactly 64 jobs")
    if args.output.exists():
        if load_capture_batch(args.output, manifest) != batch:
            raise FileExistsError(f"refusing to replace H100 refresh batch: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, batch)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "jobs": len(batch["jobs"]),
                "batch_digest": batch["batch_digest"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()