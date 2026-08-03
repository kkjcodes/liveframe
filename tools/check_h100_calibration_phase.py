"""Validate the frozen first-seed H100 checkpoint before phase two."""

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
from liveframe.quality_capture import load_results, load_run_manifest  # noqa: E402
from liveframe.quality_checkpoint import (  # noqa: E402
    evaluate_first_seed_checkpoint,
    first_seed_run_ids,
)


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
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, protocol_digest = load_protocol(args.protocol)
    manifest = load_run_manifest(args.manifest, protocol_digest=protocol_digest)
    expected = first_seed_run_ids(protocol, manifest)
    paths = [
        args.result_root / run_id / "result.json"
        for run_id in sorted(expected)
        if (args.result_root / run_id / "result.json").is_file()
    ]
    report = evaluate_first_seed_checkpoint(
        protocol,
        protocol_digest,
        manifest,
        load_results(paths),
    )
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
