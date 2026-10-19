"""Freeze all seed sets already opened by LiveFrame experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.quality_artifacts import atomic_json  # noqa: E402
from liveframe.taew21_protocol import build_seed_exclusions  # noqa: E402


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quality-protocol",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/perceptual-protocol.v1.json",
    )
    parser.add_argument(
        "--refresh-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/refresh-addressability-manifest.v2.json",
    )
    parser.add_argument(
        "--stock-wan-artifact",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/mlx-stock-wan.v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/taew21-seed-exclusions.v1.json",
    )
    args = parser.parse_args()
    quality = _object(args.quality_protocol)
    refresh = _object(args.refresh_manifest)
    stock_wan = _object(args.stock_wan_artifact)
    artifact = build_seed_exclusions(
        {
            "quality-calibration": (
                args.quality_protocol.relative_to(REPOSITORY_ROOT),
                "sampling.calibration_seeds",
                quality["sampling"]["calibration_seeds"],
            ),
            "quality-opened-holdout": (
                args.quality_protocol.relative_to(REPOSITORY_ROOT),
                "sampling.holdout_seeds",
                quality["sampling"]["holdout_seeds"],
            ),
            "refresh-proof": (
                args.refresh_manifest.relative_to(REPOSITORY_ROOT),
                "sampling.seeds",
                refresh["sampling"]["seeds"],
            ),
            "stock-wan-diagnostic": (
                args.stock_wan_artifact.relative_to(REPOSITORY_ROOT),
                "smoke.seed",
                [stock_wan["smoke"]["seed"]],
            ),
        }
    )
    if args.output.exists():
        existing = _object(args.output)
        if existing != artifact:
            raise FileExistsError(f"refusing to replace seed exclusions: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, artifact)
    print(json.dumps({"output": str(args.output), "excluded_seeds": artifact["excluded_seeds"]}))


if __name__ == "__main__":
    main()