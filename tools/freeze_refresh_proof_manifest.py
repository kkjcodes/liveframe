"""Publish the pre-registered refresh-addressability proof campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.quality_artifacts import atomic_json  # noqa: E402
from liveframe.quality_calibration import load_protocol  # noqa: E402
from liveframe.refresh_proof import (  # noqa: E402
    build_refresh_proof_manifest,
    validate_refresh_proof_manifest,
)


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--policy-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol, protocol_digest = load_protocol(args.protocol)
    policy_bundle = _object(args.policy_bundle)
    manifest = build_refresh_proof_manifest(protocol, protocol_digest, policy_bundle)
    if args.output.exists():
        validate_refresh_proof_manifest(
            _object(args.output), protocol, protocol_digest, policy_bundle
        )
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest_digest": manifest["manifest_digest"],
                "jobs": manifest["job_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()