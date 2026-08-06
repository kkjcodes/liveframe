"""Generate one claim-ineligible refresh result to test the capture contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.quality_calibration import load_protocol  # noqa: E402
from liveframe.refresh_capture import (  # noqa: E402
    IDENTITY_METRICS,
    QUALITY_METRICS,
    REQUIRED_ARTIFACTS,
    SYNTHETIC_EVIDENCE,
    TIMING_FIELDS,
    build_capture_result,
    publish_capture_result,
)
from liveframe.refresh_proof import (  # noqa: E402
    BLOCK_FRAMES,
    MECHANISM_ID,
    MODEL_LAYERS,
    SUBSTITUTION_SCHEMA,
    is_full_refresh_frame,
    load_refresh_proof_manifest,
    source_frame_for_destination,
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _point(window: int, latent_frames: int) -> dict[str, float | int]:
    generated = min((window + 1) * BLOCK_FRAMES, latent_frames)
    return {
        "window_index": window,
        "playback_seconds": (4 * (generated - 1) + 1) / 16,
        "cosine": 1.0,
        "rmse": 0.0,
        "max_abs": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--policy-bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()

    protocol, protocol_digest = load_protocol(args.protocol)
    bundle = _object(args.policy_bundle)
    manifest = load_refresh_proof_manifest(
        args.manifest, protocol, protocol_digest, bundle
    )
    jobs = {str(job["run_id"]): job for job in manifest["jobs"]}
    job = jobs[args.run_id] if args.run_id else manifest["jobs"][0]
    output = args.output_root / str(job["run_id"])
    output.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_ARTIFACTS:
        path = output / name
        if path.exists() and path.read_bytes() != f"synthetic:{name}".encode():
            raise FileExistsError(f"refusing to replace synthetic artifact: {path}")
        path.write_bytes(f"synthetic:{name}".encode())

    window_count = int(job["latent_frames"])
    base = [_point(window, int(job["latent_frames"])) for window in range(window_count)]
    layered = [
        {**point, "layer_index": layer}
        for point in base
        for layer in range(MODEL_LAYERS)
    ]
    digest = "sha256:" + "0" * 64
    substitutions = [
        {
            "schema": SUBSTITUTION_SCHEMA,
            "mechanism_id": MECHANISM_ID,
            "layer_index": layer,
            "tile_index": source_frame_for_destination(destination),
            "source_start_frame": source_frame_for_destination(destination),
            "destination_start_frame": destination,
            "frame_count": BLOCK_FRAMES,
            "token_count": BLOCK_FRAMES * 1560,
            "source_key_form": "raw-pre-rope",
            "candidate_key_form": "reroped-from-source-raw-key",
            "value_policy": "reuse-source-value",
            "source_raw_key_sha256": digest,
            "source_value_sha256": digest,
            "candidate_reroped_key_sha256": digest,
            "reference_destination_key_sha256": digest,
            "reference_destination_value_sha256": digest,
        }
        for destination in range(1, window_count)
        if not is_full_refresh_frame(destination)
        for layer in range(MODEL_LAYERS)
    ]
    result = build_capture_result(
        job,
        manifest,
        evidence_class=SYNTHETIC_EVIDENCE,
        drift_curves={
            "future_latent": base,
            "decoded-pixel": base,
            "per-layer-key": layered,
            "per-layer-value": layered,
        },
        substitutions=substitutions,
        intrinsic_metrics=dict.fromkeys(QUALITY_METRICS, 0.0),
        identity_metrics=dict.fromkeys(IDENTITY_METRICS, 0.0),
        timing=dict.fromkeys(TIMING_FIELDS, 0.0),
        runtime={"harness_sha256": digest, "kind": "synthetic-only"},
        artifact_root=output,
    )
    result_path = output / "result.json"
    publish_capture_result(result_path, result, manifest)
    print(
        json.dumps(
            {
                "evidence_class": SYNTHETIC_EVIDENCE,
                "result": str(result_path),
                "result_digest": result["result_digest"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()