"""Create immutable first-seed and full calibration execution batches."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.quality_calibration import load_protocol  # noqa: E402
from liveframe.quality_capture import (  # noqa: E402
    load_run_manifest,
    prepare_calibration_batch,
)
from liveframe.quality_checkpoint import first_seed_run_ids  # noqa: E402
from liveframe.quality_freeze import require_frozen_identities  # noqa: E402


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--first-seed-output", type=Path, required=True)
    parser.add_argument("--full-output", type=Path, required=True)
    args = parser.parse_args()
    protocol, protocol_digest = load_protocol(args.protocol)
    manifest = load_run_manifest(args.manifest, protocol_digest=protocol_digest)
    require_frozen_identities(protocol_digest, manifest)
    first_seed = prepare_calibration_batch(
        manifest,
        args.output_root,
        run_ids=first_seed_run_ids(protocol, manifest),
        required_metrics=set(protocol["quality_metrics"]),
        include_completed=True,
    )
    full = prepare_calibration_batch(
        manifest,
        args.output_root,
        required_metrics=set(protocol["quality_metrics"]),
        include_completed=True,
    )
    if len(first_seed["jobs"]) != 32 or len(full["jobs"]) != 256:
        raise ValueError("frozen campaign batch cardinality mismatch")
    _write_json(args.first_seed_output, first_seed)
    _write_json(args.full_output, full)
    print(
        json.dumps(
            {
                "first_seed_jobs": len(first_seed["jobs"]),
                "first_seed_batch_digest": first_seed["batch_digest"],
                "full_jobs": len(full["jobs"]),
                "full_batch_digest": full["batch_digest"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
