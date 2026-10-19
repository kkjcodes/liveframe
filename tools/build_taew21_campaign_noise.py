"""Materialize the immutable stochastic noise plans for the TAEW2.1 campaign.

Creates one deterministic, digest-bound noise plan per unique
(seed, latent_frames) workload in the frozen protocol's campaign job list and
writes an index artifact binding every plan payload digest to the protocol.
Plans are pure functions of (seed, latent_frames); re-running validates rather
than regenerates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.manifests import file_sha256  # noqa: E402
from liveframe.quality_stochastic import (  # noqa: E402
    StochasticPlanReader,
    create_stochastic_plan,
    plan_directory,
)
from liveframe.rolling_forcing import RollingForcingSchedule  # noqa: E402

HORIZON_LATENTS = {15: 60, 60: 243}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    if (
        authorization.get("schema") != "liveframe.taew21-campaign-authorization.v1"
        or not authorization.get("passed")
        or authorization.get("protocol_sha256") != file_sha256(args.protocol)
    ):
        raise ValueError("campaign noise build requires a passed authorization record")

    workloads = sorted(
        {
            (int(job["seed"]), HORIZON_LATENTS[int(job["horizon_seconds"])])
            for job in protocol["campaign"]["jobs"]
        }
    )
    if len(workloads) != 32:
        raise ValueError(f"campaign requires 32 unique workloads, got {len(workloads)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    plans: list[dict[str, Any]] = []
    for seed, latent_frames in workloads:
        RollingForcingSchedule(latent_frames=latent_frames)
        directory = plan_directory(args.output_root, seed=seed, latent_frames=latent_frames)
        payload_path = directory / "stochastic.bf16"
        metadata_path = directory / "stochastic.bf16.json"
        existing = (payload_path.exists(), metadata_path.exists())
        if existing == (False, False):
            create_stochastic_plan(directory, seed=seed, latent_frames=latent_frames)
            print(f"created plan seed={seed} latent_frames={latent_frames}", flush=True)
        elif existing != (True, True):
            raise ValueError(f"incomplete stochastic plan: {directory}")
        StochasticPlanReader(metadata_path, seed=seed, latent_frames=latent_frames)
        plans.append(
            {
                "seed": seed,
                "latent_frames": latent_frames,
                "directory": str(directory.relative_to(REPOSITORY_ROOT)),
                "payload_sha256": file_sha256(payload_path),
                "metadata_sha256": file_sha256(metadata_path),
            }
        )
    index = {
        "schema": "liveframe.taew21-campaign-noise-index.v1",
        "protocol_sha256": file_sha256(args.protocol),
        "authorization_sha256": file_sha256(args.authorization),
        "builder_sha256": file_sha256(Path(__file__).resolve()),
        "plans": plans,
    }
    temporary = args.index.with_name(f".{args.index.name}.tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.index)
    print(json.dumps({"index": str(args.index), "plans": len(plans)}, sort_keys=True))


if __name__ == "__main__":
    main()
