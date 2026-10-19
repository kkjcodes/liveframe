"""Convert the pinned upstream TAEW2.1 decoder checkpoint to MLX layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.converters.taew21_to_mlx import execute_taew21_conversion  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--source-repository", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--target-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = execute_taew21_conversion(
        args.checkpoint,
        args.output_dir,
        expected_source_sha256=args.expected_sha256,
        source_repository=args.source_repository,
        source_revision=args.source_revision,
        target_dtype=args.target_dtype,
    )
    print(json.dumps({"manifest": str(manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()