"""Build the complete TAEW2.1 campaign protocol before first MLX output."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.manifests import file_sha256  # noqa: E402
from liveframe.perceptual_policy import canonical_digest  # noqa: E402
from liveframe.quality_artifacts import atomic_json  # noqa: E402
from liveframe.taew21_protocol import (  # noqa: E402
    TAEW21_PROTOCOL_SCHEMA,
    load_port_parity_policy,
    validate_taew21_protocol,
)

FRESH_SEEDS = (
    170003,
    180001,
    190027,
    200003,
    210011,
    220009,
    230003,
    240007,
    250013,
    260003,
    270001,
    280001,
    290011,
    300007,
    310019,
    320009,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"protocol source must be an object: {path}")
    return value


def build_protocol() -> dict[str, Any]:
    artifacts = REPOSITORY_ROOT / "artifacts"
    golden_root = REPOSITORY_ROOT / "outputs/golden/taew21-pytorch-reference.v1"
    quality_protocol_path = artifacts / "perceptual-protocol.v2.json"
    metric_path = artifacts / "perceptual-metrics.v1.json"
    exclusions_path = artifacts / "taew21-seed-exclusions.v1.json"
    parity_path = artifacts / "taew21-port-parity-policy.v1.json"
    quality = _load(quality_protocol_path)
    exclusions = _load(exclusions_path)
    parity, parity_digest = load_port_parity_policy(parity_path)
    prompts = quality["prompt_catalog"]
    if not isinstance(prompts, dict) or len(prompts) != len(FRESH_SEEDS):
        raise ValueError("frozen prompt catalog does not contain exactly 16 prompts")
    seeds = dict(zip(prompts, FRESH_SEEDS, strict=True))
    stratum_by_prompt = {
        prompt_id: stratum["id"]
        for stratum in quality["strata"]
        for prompt_id in stratum["prompt_ids"]
    }
    jobs = [
        {
            "job_id": f"taew21-{prompt_id}-{horizon}s",
            "prompt_id": prompt_id,
            "stratum": stratum_by_prompt[prompt_id],
            "seed": seeds[prompt_id],
            "horizon_seconds": horizon,
            "output": f"outputs/taew21-campaign/{prompt_id}/{horizon}s",
        }
        for prompt_id in prompts
        for horizon in (15, 60)
    ]
    protocol: dict[str, Any] = {
        "schema": TAEW21_PROTOCOL_SCHEMA,
        "status": "frozen-before-first-mlx-taew21-output",
        "provenance": {
            "upstream": {
                "repository": parity["source"]["repository"],
                "revision": parity["source"]["revision"],
                "license": "MIT",
                "implementation_sha256": parity["source"]["implementation_sha256"],
                "weights_sha256": parity["source"]["weights_sha256"],
                "license_sha256": (
                    "sha256:532f9e394518ffddecd294a517d5b41d79d3d3866c3fb95a6cb0e8bcc02370bf"
                ),
            },
            "conversion": {
                "converter_sha256": file_sha256(
                    REPOSITORY_ROOT / "src/liveframe/converters/taew21_to_mlx.py"
                ),
                "mlx_decoder_sha256": file_sha256(
                    REPOSITORY_ROOT / "src/liveframe/decoders/taew21_mlx.py"
                ),
                "parity_checker_sha256": file_sha256(
                    REPOSITORY_ROOT / "tools/check_taew21_port_parity.py"
                ),
                "manifest_sha256": file_sha256(
                    artifacts / "taew21-mlx-weights.v1/taew21-conversion.json"
                ),
                "weights_sha256": file_sha256(
                    artifacts / "taew21-mlx-weights.v1/taew2_1-mlx.safetensors"
                ),
            },
            "parity": {
                "policy_digest": parity_digest,
                "policy_sha256": file_sha256(parity_path),
                "calibration_sha256": file_sha256(
                    artifacts / "taew21-mlx-primitive-calibration.v1.json"
                ),
                "golden_manifest_sha256": file_sha256(
                    golden_root / "taew21-pytorch-golden.json"
                ),
                "golden_tensors_sha256": file_sha256(
                    golden_root / "taew21-pytorch-golden.safetensors"
                ),
                "fixture_sha256": parity["fixture"]["sha256"],
            },
            "baseline": {
                "artifact": "artifacts/m5-cfpp-1step-cache4-baseline.v1.json",
                "artifact_sha256": file_sha256(
                    artifacts / "m5-cfpp-1step-cache4-baseline.v1.json"
                ),
                "complete_wall_seconds": 57.12651099999948,
                "vae_wall_seconds": 41.31944479000049,
            },
            "full_vae": {
                "artifact": "weights/Wan2.1-T2V-1.3B-MLX/vae.safetensors",
                "weights_sha256": file_sha256(
                    REPOSITORY_ROOT / "weights/Wan2.1-T2V-1.3B-MLX/vae.safetensors"
                ),
            },
            "seed_exclusions": {
                "artifact": "artifacts/taew21-seed-exclusions.v1.json",
                "artifact_sha256": file_sha256(exclusions_path),
                "exclusions_digest": exclusions["exclusions_digest"],
            },
        },
        "decoder_contract": {
            "latent_layout": "B,F,C,H,W",
            "latent_dtype": "bfloat16-cast-to-float16",
            "latent_channels": 16,
            "latent_scaling": "no-additional-scaling",
            "temporal_upscale": 4,
            "startup_frames_trimmed": 3,
            "output_layout": "T,H,W,C",
            "output_dtype": "uint8",
            "output_range": [0, 255],
            "resolution": [480, 832],
            "color_conversion": "clamp-[0,1]-multiply-255-round-nearest",
        },
        "campaign": {
            "output_root": "outputs/taew21-campaign",
            "prompt_catalog": prompts,
            "strata": quality["strata"],
            "horizons": {key: quality["horizon_specs"][key] for key in ("15", "60")},
            "seeds": seeds,
            "jobs": jobs,
        },
        "benchmark": {
            "warmups": 1,
            "trials": 5,
            "model_loading": "excluded-steady-state",
            "timing_boundaries": ["decoder", "pixel_materialization", "complete-wall"],
            "allocator_cache_limit_bytes": 4294967296,
            "allocator_memory_limit_bytes": 30150672384,
            "maximum_swap_delta_mib": 0.0,
            "thermal_capture": "required-before-and-after-each-trial",
            "minimum_complete_wall_speedup": 1.5,
            "maximum_target_wall_seconds": 38.08434066666632,
            "confidence_rule": "one-sided-95%-lower-confidence-bound-on-complete-wall-speedup>=1.5",
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 20260809,
        },
        "metrics": {
            "source_protocol": "artifacts/perceptual-protocol.v2.json",
            "source_protocol_sha256": file_sha256(quality_protocol_path),
            "provenance_artifact": "artifacts/perceptual-metrics.v1.json",
            "provenance_sha256": file_sha256(metric_path),
            "intrinsic_metrics": quality["quality_metrics"],
            "identity_tripwires": quality["identity_tripwires"],
            "authority": "screening-only-not-a-human-noticeability-threshold",
            "decision_rule": "all-intrinsic-bands-and-all-identity-tripwires-must-pass",
            "retuning": "forbidden-after-candidate-visibility",
        },
        "human_study": {
            "task": "reference-then-randomized-A-B-source-matching",
            "blinding": "decoder-identity-hidden-from-participants-and-analysis-labels",
            "chance_accuracy": 0.5,
            "noticeability_ceiling": 0.55,
            "primary_rule": "one-sided-95%-upper-confidence-bound-on-discrimination-accuracy<=0.55",
            "clustering": ["participant", "video"],
            "minimum_usable_judgments": 271,
            "planning_judgments": [400, 600],
            "planned_participants": 30,
            "trials_per_participant": [16, 20],
            "sample_size": "fixed-by-clustered-simulation-before-recruitment",
            "exclusions": [
                "withdrawn-consent",
                "duplicate-participant",
                "incomplete-session",
                "failed-predeclared-attention-check",
            ],
            "stopping_rule": (
                "stop-at-simulation-fixed-usable-judgment-target-no-sequential-peeking"
            ),
            "secondary_analysis": (
                "bootstrap-metric-correlation-with-leave-one-prompt-out-stability"
            ),
        },
    }
    protocol["protocol_digest"] = canonical_digest(protocol)
    validate_taew21_protocol(protocol, exclusions)
    return protocol


def main() -> None:
    output = REPOSITORY_ROOT / "artifacts/taew21-protocol.v1.json"
    protocol = build_protocol()
    if output.exists():
        existing = _load(output)
        if existing != protocol:
            raise FileExistsError(f"refusing to replace different protocol: {output}")
    else:
        atomic_json(output, protocol)
    print(
        json.dumps(
            {
                "artifact": str(output.relative_to(REPOSITORY_ROOT)),
                "protocol_digest": protocol["protocol_digest"],
                "jobs": len(protocol["campaign"]["jobs"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()