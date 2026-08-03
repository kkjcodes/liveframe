from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.quality_artifacts import atomic_json  # noqa: E402
from liveframe.quality_calibration import load_protocol  # noqa: E402
from liveframe.quality_capture import (  # noqa: E402
    load_run_manifest,
    prepare_holdout_reference_batch,
)
from liveframe.quality_freeze import require_frozen_identities  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, protocol_digest = load_protocol(args.protocol)
    manifest = load_run_manifest(args.manifest, protocol_digest=protocol_digest)
    require_frozen_identities(protocol_digest, manifest)
    batch = prepare_holdout_reference_batch(manifest, args.output_root)
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != batch:
            raise FileExistsError(f"refusing to replace reference batch: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, batch)
    print(json.dumps({"batch_digest": batch["batch_digest"], "jobs": len(batch["jobs"])}))


if __name__ == "__main__":
    main()