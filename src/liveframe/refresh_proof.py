"""Pre-registered refresh-addressability proof contract for Delta Scheduling."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from liveframe.perceptual_policy import canonical_digest

MANIFEST_SCHEMA = "liveframe.refresh-addressability-manifest.v1"
RESULT_SCHEMA = "liveframe.refresh-addressability-result.v1"
SUBSTITUTION_SCHEMA = "liveframe.refresh-kv-substitution.v1"
MECHANISM_ID = "reroped-refresh-kv-v1"
PROOF_STRATA = (
    "limited-animation-held-frames",
    "dialogue-minimal-motion",
)
PROOF_SEEDS = (130003, 140009, 150011, 160001)
HORIZONS = (15, 60)
DENOMINATOR_ARTIFACT = "artifacts/h100-cfpp-1step-denominator.v1.json"
DENOMINATOR_SHA256 = (
    "sha256:82394c2bad77718668c8fbe7b42dd568de5a607679737312a8004321e740f766"
)
POLICY_BUNDLE_DIGEST = (
    "sha256:41a228ceac16a026c7771d2e2c496714e5476c8e9287d3da3d15a78e993a9870"
)
WALL_SECONDS = 5.960915993899107
DENOISE_SECONDS = 1.4401073913574218
REFRESH_SECONDS = 1.1655172805786134
COMPUTE_RECLAIM_FLOOR = 0.30
WALL_REDUCTION_FLOOR = 0.17
BLOCK_FRAMES = 3
TOKENS_PER_FRAME = 1560
MODEL_LAYERS = 30


def _job_id(protocol_digest: str, identity: Mapping[str, Any]) -> str:
    payload = {
        "protocol_digest": protocol_digest,
        "mechanism_id": MECHANISM_ID,
        **identity,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "refresh-proof-" + hashlib.sha256(encoded).hexdigest()[:20]


def _quality_bundle_digest(bundle: Mapping[str, Any]) -> str:
    digest = bundle.get("bundle_digest")
    if digest != POLICY_BUNDLE_DIGEST:
        raise ValueError("refresh proof requires the frozen v2 quality bundle")
    unsigned = {key: value for key, value in bundle.items() if key != "bundle_digest"}
    if canonical_digest(unsigned) != POLICY_BUNDLE_DIGEST:
        raise ValueError("refresh proof quality bundle payload mismatch")
    return str(digest)


def build_refresh_proof_manifest(
    protocol: Mapping[str, Any],
    protocol_digest: str,
    policy_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the one-shot prospective full-refresh versus re-RoPE campaign."""

    bundle_digest = _quality_bundle_digest(policy_bundle)
    if policy_bundle.get("source_protocol_digest") != protocol_digest:
        raise ValueError("refresh proof quality bundle descends from another protocol")
    sampling = protocol["sampling"]
    excluded_seeds = {
        *map(int, sampling["calibration_seeds"]),
        *map(int, sampling["holdout_seeds"]),
    }
    if excluded_seeds.intersection(PROOF_SEEDS):
        raise ValueError("refresh proof seeds overlap calibration or opened holdout")

    strata = {
        str(row["id"]): row
        for row in protocol["strata"]
        if bool(row.get("ds_gate_class"))
    }
    if tuple(strata) != PROOF_STRATA:
        raise ValueError("refresh proof requires the two frozen DS gate strata")
    catalog = protocol["prompt_catalog"]
    jobs: list[dict[str, Any]] = []
    for stratum_id in PROOF_STRATA:
        for prompt_id in strata[stratum_id]["prompt_ids"]:
            for seed in PROOF_SEEDS:
                for horizon_seconds in HORIZONS:
                    horizon = protocol["horizon_specs"][str(horizon_seconds)]
                    identity = {
                        "stratum": stratum_id,
                        "prompt_id": prompt_id,
                        "seed": seed,
                        "horizon_seconds": horizon_seconds,
                        "latent_frames": int(horizon["latent_frames"]),
                    }
                    jobs.append(
                        {
                            "run_id": _job_id(protocol_digest, identity),
                            **identity,
                            "pixel_frames": int(horizon["pixel_frames"]),
                            "playback_seconds": float(horizon["playback_seconds"]),
                            "prompt": catalog[prompt_id],
                            "backend": "cuda-h100",
                            "split": "prospective-optimization-validation",
                            "reference_mode": "full-refresh",
                            "candidate_mode": MECHANISM_ID,
                            "stochastic_pairing": "same-seed-same-bytes",
                            "status": "planned",
                        }
                    )

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "frozen-before-refresh-oracle-visibility",
        "frozen_at": "2026-08-04",
        "claim": "refresh-addressability-with-horizon-indexed-drift",
        "source_protocol_digest": protocol_digest,
        "policy_bundle_digest": bundle_digest,
        "denominator": {
            "artifact": DENOMINATOR_ARTIFACT,
            "sha256": DENOMINATOR_SHA256,
            "wall_seconds": WALL_SECONDS,
            "denoising_seconds": DENOISE_SECONDS,
            "refresh_seconds": REFRESH_SECONDS,
        },
        "mechanism": {
            "id": MECHANISM_ID,
            "reference": "full-refresh",
            "candidate": "reuse-source-value-and-rerope-source-raw-key",
            "source_key_form": "raw-pre-rope",
            "candidate_key_form": "reroped-from-source-raw-key",
            "value_policy": "reuse-source-value",
            "attention_scope": "full",
            "block_frames": BLOCK_FRAMES,
            "tokens_per_frame": TOKENS_PER_FRAME,
            "prohibited": [
                "relabel-already-roped-key",
                "unresolved-source-position",
                "short-horizon-extrapolation",
                "post-result-mechanism-tuning",
            ],
        },
        "sampling": {
            "seeds": list(PROOF_SEEDS),
            "excluded_calibration_seeds": list(map(int, sampling["calibration_seeds"])),
            "excluded_holdout_seeds": list(map(int, sampling["holdout_seeds"])),
            "minimum_videos_per_prompt_horizon": len(PROOF_SEEDS),
            "horizons_seconds": list(HORIZONS),
            "strata": list(PROOF_STRATA),
            "result_visibility_policy": (
                "one-shot validation; no mechanism changes after any result is visible"
            ),
        },
        "required_drift_curves": [
            "future_latent",
            "per-layer-key",
            "per-layer-value",
            "decoded-pixel",
        ],
        "accounting": {
            "denoise_only": True,
            "refresh_only": True,
            "combined_weighted_by_frozen_seconds": True,
            "complete_wall_including_overhead": True,
            "compute_reclaim_floor": COMPUTE_RECLAIM_FLOOR,
            "wall_reduction_floor": WALL_REDUCTION_FLOOR,
            "both_floors_must_pass": True,
        },
        "quality": {
            "intrinsic_policy_bundle_digest": bundle_digest,
            "identity_tripwires_authoritative": True,
            "all_prompt_horizon_cells_must_pass": True,
        },
        "job_count": len(jobs),
        "jobs": jobs,
    }
    manifest["manifest_digest"] = canonical_digest(manifest)
    return manifest


def validate_refresh_proof_manifest(
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_digest: str,
    policy_bundle: Mapping[str, Any],
) -> None:
    """Require literal equality with the deterministic frozen campaign."""

    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported refresh proof manifest")
    expected = build_refresh_proof_manifest(protocol, protocol_digest, policy_bundle)
    if dict(manifest) != expected:
        raise ValueError("refresh proof manifest differs from the frozen campaign")


def load_refresh_proof_manifest(
    path: Path,
    protocol: Mapping[str, Any],
    protocol_digest: str,
    policy_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("refresh proof manifest must be an object")
    validate_refresh_proof_manifest(value, protocol, protocol_digest, policy_bundle)
    return value


def validate_substitution_ancestry(record: Mapping[str, Any]) -> None:
    """Reject refresh substitutions without explicit raw-key/value ancestry."""

    if record.get("schema") != SUBSTITUTION_SCHEMA:
        raise ValueError("unsupported refresh substitution record")
    if record.get("mechanism_id") != MECHANISM_ID:
        raise ValueError("refresh substitution mechanism mismatch")
    integer_fields = (
        "layer_index",
        "tile_index",
        "source_start_frame",
        "destination_start_frame",
        "frame_count",
        "token_count",
    )
    values: dict[str, int] = {}
    for name in integer_fields:
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"refresh substitution {name} must be a non-negative integer")
        values[name] = value
    if values["destination_start_frame"] <= values["source_start_frame"]:
        raise ValueError("refresh substitution destination must follow its source")
    if values["frame_count"] != BLOCK_FRAMES:
        raise ValueError("refresh substitution must cover one three-frame block")
    if values["token_count"] != BLOCK_FRAMES * TOKENS_PER_FRAME:
        raise ValueError("refresh substitution token count mismatch")
    if record.get("source_key_form") != "raw-pre-rope":
        raise ValueError("refresh substitution requires a raw pre-RoPE source key")
    if record.get("candidate_key_form") != "reroped-from-source-raw-key":
        raise ValueError("refresh substitution candidate key ancestry is unresolved")
    if record.get("value_policy") != "reuse-source-value":
        raise ValueError("refresh substitution value ancestry is unresolved")
    for name in (
        "source_raw_key_sha256",
        "source_value_sha256",
        "candidate_reroped_key_sha256",
        "reference_destination_key_sha256",
        "reference_destination_value_sha256",
    ):
        value = record.get(name)
        if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
            raise ValueError(f"refresh substitution requires {name}")


def _curve_keys(
    rows: Any, *, name: str, window_count: int, per_layer: bool
) -> set[int] | set[tuple[int, int]]:
    if not isinstance(rows, list):
        raise ValueError(f"refresh result {name} drift curve must be a list")
    keys: set[int] | set[tuple[int, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"refresh result {name} drift rows must be objects")
        window_index = row.get("window_index")
        if (
            isinstance(window_index, bool)
            or not isinstance(window_index, int)
            or not 0 <= window_index < window_count
        ):
            raise ValueError(f"refresh result {name} has an invalid window index")
        if per_layer:
            layer_index = row.get("layer_index")
            if (
                isinstance(layer_index, bool)
                or not isinstance(layer_index, int)
                or not 0 <= layer_index < MODEL_LAYERS
            ):
                raise ValueError(f"refresh result {name} has an invalid layer index")
            key: int | tuple[int, int] = (window_index, layer_index)
        else:
            key = window_index
        if key in keys:
            raise ValueError(f"refresh result {name} duplicates a drift point")
        keys.add(key)  # type: ignore[arg-type]
    return keys


def validate_refresh_proof_result(
    result: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    """Require a manifest-bound result with complete long-horizon drift curves."""

    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError("unsupported refresh proof result")
    if result.get("manifest_digest") != manifest.get("manifest_digest"):
        raise ValueError("refresh proof result manifest mismatch")
    jobs = {str(job["run_id"]): job for job in manifest["jobs"]}
    run_id = str(result.get("run_id", ""))
    if run_id not in jobs:
        raise ValueError("refresh proof result has an unknown run ID")
    job = jobs[run_id]
    for name in (
        "stratum",
        "prompt_id",
        "seed",
        "horizon_seconds",
        "latent_frames",
    ):
        if result.get(name) != job[name]:
            raise ValueError(f"refresh proof result {name} mismatch")

    window_count = int(job["latent_frames"]) // BLOCK_FRAMES + 3
    curves = result.get("drift_curves")
    if not isinstance(curves, dict) or set(curves) != {
        "future_latent",
        "per-layer-key",
        "per-layer-value",
        "decoded-pixel",
    }:
        raise ValueError("refresh proof result requires all four drift curves")
    windows = set(range(window_count))
    layer_windows = {
        (window_index, layer_index)
        for window_index in range(window_count)
        for layer_index in range(MODEL_LAYERS)
    }
    if _curve_keys(
        curves["future_latent"],
        name="future_latent",
        window_count=window_count,
        per_layer=False,
    ) != windows:
        raise ValueError("refresh result future_latent drift curve is incomplete")
    if _curve_keys(
        curves["decoded-pixel"],
        name="decoded-pixel",
        window_count=window_count,
        per_layer=False,
    ) != windows:
        raise ValueError("refresh result decoded-pixel drift curve is incomplete")
    for name in ("per-layer-key", "per-layer-value"):
        if _curve_keys(
            curves[name], name=name, window_count=window_count, per_layer=True
        ) != layer_windows:
            raise ValueError(f"refresh result {name} drift curve is incomplete")

    substitutions = result.get("substitutions")
    if not isinstance(substitutions, list) or not substitutions:
        raise ValueError("refresh proof result requires substitution ancestry")
    seen: set[tuple[int, int, int]] = set()
    for record in substitutions:
        if not isinstance(record, dict):
            raise ValueError("refresh proof substitutions must be objects")
        validate_substitution_ancestry(record)
        key = (
            int(record["layer_index"]),
            int(record["tile_index"]),
            int(record["destination_start_frame"]),
        )
        if key in seen:
            raise ValueError("refresh proof result duplicates substitution ancestry")
        seen.add(key)


def reclaim_accounting(
    *,
    denoise_reclaim_fraction: float,
    refresh_reclaim_fraction: float,
    overhead_seconds: float,
) -> dict[str, float | bool]:
    """Apply the four-way frozen H100 reclaim accounting contract."""

    values = {
        "denoise_reclaim_fraction": denoise_reclaim_fraction,
        "refresh_reclaim_fraction": refresh_reclaim_fraction,
        "overhead_seconds": overhead_seconds,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values.values()
    ):
        raise ValueError("refresh reclaim accounting values must be numeric")
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError("refresh reclaim accounting values must be finite")
    if not 0 <= denoise_reclaim_fraction <= 1 or not 0 <= refresh_reclaim_fraction <= 1:
        raise ValueError("refresh reclaim fractions must be in [0, 1]")
    if overhead_seconds < 0:
        raise ValueError("refresh reclaim overhead cannot be negative")

    denoise_saved = DENOISE_SECONDS * denoise_reclaim_fraction
    refresh_saved = REFRESH_SECONDS * refresh_reclaim_fraction
    gross_saved = denoise_saved + refresh_saved
    combined_fraction = gross_saved / (DENOISE_SECONDS + REFRESH_SECONDS)
    net_saved = gross_saved - overhead_seconds
    wall_reduction = net_saved / WALL_SECONDS
    compute_passed = combined_fraction >= COMPUTE_RECLAIM_FLOOR
    wall_passed = wall_reduction >= WALL_REDUCTION_FLOOR
    return {
        "denoise_saved_seconds": denoise_saved,
        "refresh_saved_seconds": refresh_saved,
        "gross_saved_seconds": gross_saved,
        "overhead_seconds": overhead_seconds,
        "net_saved_seconds": net_saved,
        "denoise_only_wall_reduction_fraction": denoise_saved / WALL_SECONDS,
        "refresh_only_wall_reduction_fraction": refresh_saved / WALL_SECONDS,
        "combined_reclaim_fraction": combined_fraction,
        "complete_wall_reduction_fraction": wall_reduction,
        "compute_floor_passed": compute_passed,
        "wall_floor_passed": wall_passed,
        "passed": compute_passed and wall_passed,
    }