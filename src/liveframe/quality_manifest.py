"""Generate digest-bound quality calibration and blind-holdout run manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from liveframe.perceptual_policy import canonical_digest
from liveframe.quality_calibration import load_protocol

RUN_MANIFEST_SCHEMA = "liveframe.quality-run-manifest.v1"


def quality_run_id(protocol_digest: str, identity: Mapping[str, Any]) -> str:
    payload = {
        "protocol_digest": protocol_digest,
        **identity,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "quality-" + hashlib.sha256(encoded).hexdigest()[:20]


def _job(
    *,
    protocol_digest: str,
    split: str,
    role: str,
    backend: str,
    stratum: str,
    prompt_id: str,
    prompt: str,
    seed: int,
    horizon_seconds: int,
    latent_frames: int,
    pixel_frames: int,
    playback_seconds: float,
) -> dict[str, Any]:
    identity = {
        "split": split,
        "role": role,
        "backend": backend,
        "stratum": stratum,
        "prompt_id": prompt_id,
        "seed": seed,
        "horizon_seconds": horizon_seconds,
        "latent_frames": latent_frames,
        "pixel_frames": pixel_frames,
        "playback_seconds": playback_seconds,
    }
    return {
        "run_id": quality_run_id(protocol_digest, identity),
        **identity,
        "prompt": prompt,
        "status": "planned",
    }


def build_run_manifest(
    protocol: Mapping[str, Any], protocol_digest: str
) -> dict[str, Any]:
    """Enumerate every pre-registered calibration and holdout workload."""

    jobs: list[dict[str, Any]] = []
    catalog = protocol["prompt_catalog"]
    horizons = protocol["horizons_seconds"]
    sampling = protocol["sampling"]
    for stratum in protocol["strata"]:
        for prompt_id in stratum["prompt_ids"]:
            common = {
                "protocol_digest": protocol_digest,
                "stratum": stratum["id"],
                "prompt_id": prompt_id,
                "prompt": catalog[prompt_id],
            }
            for seed in sampling["calibration_seeds"]:
                for horizon_seconds in horizons:
                    horizon = protocol["horizon_specs"][str(horizon_seconds)]
                    jobs.append(
                        _job(
                            split="calibration",
                            role="teacher",
                            backend="cuda",
                            seed=seed,
                            horizon_seconds=horizon_seconds,
                            latent_frames=horizon["latent_frames"],
                            pixel_frames=horizon["pixel_frames"],
                            playback_seconds=horizon["playback_seconds"],
                            **common,
                        )
                    )
            for seed in sampling["holdout_seeds"]:
                for horizon_seconds in horizons:
                    horizon = protocol["horizon_specs"][str(horizon_seconds)]
                    for role, backend in (
                        ("reference", "cuda"),
                        ("candidate", "mlx"),
                    ):
                        jobs.append(
                            _job(
                                split="holdout",
                                role=role,
                                backend=backend,
                                seed=seed,
                                horizon_seconds=horizon_seconds,
                                latent_frames=horizon["latent_frames"],
                                pixel_frames=horizon["pixel_frames"],
                                playback_seconds=horizon["playback_seconds"],
                                **common,
                            )
                        )
    run_ids = [job["run_id"] for job in jobs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("quality run ID collision")
    manifest: dict[str, Any] = {
        "schema": RUN_MANIFEST_SCHEMA,
        "protocol_digest": protocol_digest,
        "job_counts": {
            "calibration": sum(job["split"] == "calibration" for job in jobs),
            "holdout": sum(job["split"] == "holdout" for job in jobs),
            "total": len(jobs),
        },
        "execution_order": [
            "calibration teacher captures",
            "frozen policy derivation",
            "holdout reference and candidate captures",
        ],
        "jobs": jobs,
    }
    manifest["manifest_digest"] = canonical_digest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, protocol_digest = load_protocol(args.protocol)
    manifest = build_run_manifest(protocol, protocol_digest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "protocol_digest": protocol_digest,
                "manifest_digest": manifest["manifest_digest"],
                "job_counts": manifest["job_counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
