from __future__ import annotations

import argparse
import json
from pathlib import Path

from liveframe.quality_calibration import load_protocol
from liveframe.quality_capture import load_run_manifest
from liveframe.quality_stochastic import create_stochastic_plan_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    _, protocol_digest = load_protocol(args.protocol)
    manifest = load_run_manifest(args.manifest, protocol_digest=protocol_digest)
    index = create_stochastic_plan_set(args.output_root, manifest)
    print(json.dumps(index, sort_keys=True))


if __name__ == "__main__":
    main()