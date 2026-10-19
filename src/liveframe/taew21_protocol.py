"""Prospective protocol primitives for the taew2_1 decoder campaign."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from liveframe.perceptual_policy import canonical_digest
from liveframe.quality_capture import sha256_file

SEED_EXCLUSIONS_SCHEMA = "liveframe.taew21-seed-exclusions.v1"
PORT_PARITY_POLICY_SCHEMA = "liveframe.taew21-port-parity-policy.v1"
TAEW21_PROTOCOL_SCHEMA = "liveframe.taew21-protocol.v1"


def _strict_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise ValueError(f"{context} keys mismatch: missing={missing}, unknown={unknown}")


def _digest(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{context} must be a sha256 digest")
    try:
        int(value[7:], 16)
    except ValueError as error:
        raise ValueError(f"{context} must be a sha256 digest") from error
    return value


def build_seed_exclusions(
    sources: Mapping[str, tuple[Path, str, Iterable[int]]]
) -> dict[str, Any]:
    """Build a digest-bound union of seeds from explicit authoritative selectors."""

    if not sources:
        raise ValueError("seed exclusions require at least one source")
    rows: list[dict[str, Any]] = []
    union: set[int] = set()
    for name, (path, selector, values) in sorted(sources.items()):
        if not name or not selector:
            raise ValueError("seed exclusion sources require names and selectors")
        seeds = sorted(set(values))
        if not seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        ):
            raise ValueError(f"seed exclusion source {name} contains invalid seeds")
        overlap = union.intersection(seeds)
        rows.append(
            {
                "name": name,
                "artifact": str(path),
                "artifact_sha256": sha256_file(path),
                "selector": selector,
                "seeds": seeds,
                "already_excluded_by_prior_sources": sorted(overlap),
            }
        )
        union.update(seeds)
    artifact: dict[str, Any] = {
        "schema": SEED_EXCLUSIONS_SCHEMA,
        "status": "frozen-before-taew21-campaign-seed-selection",
        "sources": rows,
        "excluded_seeds": sorted(union),
    }
    artifact["exclusions_digest"] = canonical_digest(artifact)
    return artifact


def validate_seed_exclusions(value: Mapping[str, Any]) -> None:
    _strict_keys(
        value,
        {"schema", "status", "sources", "excluded_seeds", "exclusions_digest"},
        context="seed exclusions",
    )
    if value["schema"] != SEED_EXCLUSIONS_SCHEMA:
        raise ValueError("unsupported taew2_1 seed exclusion schema")
    sources = value["sources"]
    excluded = value["excluded_seeds"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("seed exclusions require source records")
    if not isinstance(excluded, list) or excluded != sorted(set(excluded)):
        raise ValueError("excluded seeds must be sorted and unique")
    unsigned = {key: item for key, item in value.items() if key != "exclusions_digest"}
    if canonical_digest(unsigned) != value["exclusions_digest"]:
        raise ValueError("seed exclusion digest mismatch")


def require_fresh_seeds(candidates: Iterable[int], exclusions: Mapping[str, Any]) -> None:
    validate_seed_exclusions(exclusions)
    seeds = list(candidates)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("candidate seeds must be nonempty and unique")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("candidate seeds must be nonnegative integers")
    overlap = sorted(set(seeds).intersection(exclusions["excluded_seeds"]))
    if overlap:
        raise ValueError(f"candidate seeds overlap frozen exclusions: {overlap}")


def load_port_parity_policy(path: Path) -> tuple[dict[str, Any], str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("port parity policy must be an object")
    _strict_keys(
        raw,
        {"schema", "status", "source", "fixture", "comparison", "limits"},
        context="port parity policy",
    )
    if raw["schema"] != PORT_PARITY_POLICY_SCHEMA:
        raise ValueError("unsupported taew2_1 port parity policy schema")
    source = raw["source"]
    fixture = raw["fixture"]
    comparison = raw["comparison"]
    limits = raw["limits"]
    if not all(isinstance(item, dict) for item in (source, fixture, comparison, limits)):
        raise ValueError("port parity policy sections must be objects")
    _strict_keys(
        source,
        {"repository", "revision", "implementation_sha256", "weights_sha256", "pytorch_version"},
        context="port parity source",
    )
    _digest(source["implementation_sha256"], context="implementation_sha256")
    _digest(source["weights_sha256"], context="weights_sha256")
    _strict_keys(
        fixture,
        {"artifact", "sha256", "dtype", "layout", "shape"},
        context="port parity fixture",
    )
    _digest(fixture["sha256"], context="fixture sha256")
    shape = fixture["shape"]
    if not isinstance(shape, list) or len(shape) != 5 or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in shape
    ):
        raise ValueError("port parity fixture shape must contain five positive dimensions")
    _strict_keys(
        comparison,
        {"activation_anchors", "final_outputs", "metrics"},
        context="port parity comparison",
    )
    anchors = comparison["activation_anchors"]
    outputs = comparison["final_outputs"]
    metrics = comparison["metrics"]
    if (
        not isinstance(anchors, list)
        or not anchors
        or anchors != list(dict.fromkeys(anchors))
        or not all(isinstance(name, str) and name for name in anchors)
    ):
        raise ValueError("activation anchors must be nonempty unique names")
    if outputs != ["float_decoder_output", "rgb_uint8"]:
        raise ValueError("port parity final outputs must bind float and RGB output")
    expected_metrics = ["maximum_absolute_error", "mean_absolute_error", "cosine_similarity"]
    if metrics != expected_metrics:
        raise ValueError("port parity metrics do not match the frozen contract")
    _strict_keys(
        limits,
        {"maximum_absolute_error", "mean_absolute_error", "minimum_cosine_similarity"},
        context="port parity limits",
    )
    for name, limit in limits.items():
        if (
            isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not math.isfinite(limit)
        ):
            raise ValueError(f"port parity limit {name} must be finite")
    if limits["maximum_absolute_error"] < 0 or limits["mean_absolute_error"] < 0:
        raise ValueError("absolute-error limits must be nonnegative")
    if not -1 <= limits["minimum_cosine_similarity"] <= 1:
        raise ValueError("minimum cosine similarity must be in [-1, 1]")
    return raw, canonical_digest(raw)


def validate_taew21_protocol(
    value: Mapping[str, Any], exclusions: Mapping[str, Any]
) -> None:
    """Validate the full pre-visibility decoder campaign contract."""

    _strict_keys(
        value,
        {
            "schema",
            "status",
            "provenance",
            "decoder_contract",
            "campaign",
            "benchmark",
            "metrics",
            "human_study",
            "protocol_digest",
        },
        context="taew21 protocol",
    )
    if value["schema"] != TAEW21_PROTOCOL_SCHEMA:
        raise ValueError("unsupported taew2_1 protocol schema")
    if value["status"] != "frozen-before-first-mlx-taew21-output":
        raise ValueError("taew2_1 protocol status is not prospectively frozen")
    unsigned = {key: item for key, item in value.items() if key != "protocol_digest"}
    if canonical_digest(unsigned) != value["protocol_digest"]:
        raise ValueError("taew2_1 protocol digest mismatch")

    provenance = value["provenance"]
    _strict_keys(
        provenance,
        {
            "upstream",
            "conversion",
            "parity",
            "baseline",
            "full_vae",
            "seed_exclusions",
        },
        context="taew21 provenance",
    )
    for section_name, section in provenance.items():
        if not isinstance(section, dict) or not section:
            raise ValueError(f"taew21 provenance {section_name} must be a nonempty object")
        digest_fields = [
            name
            for name in section
            if name.endswith("sha256") or name.endswith("digest")
        ]
        if not digest_fields:
            raise ValueError(f"taew21 provenance {section_name} requires a digest")
        for field in digest_fields:
            _digest(section[field], context=f"{section_name}.{field}")

    decoder = value["decoder_contract"]
    _strict_keys(
        decoder,
        {
            "latent_layout",
            "latent_dtype",
            "latent_channels",
            "latent_scaling",
            "temporal_upscale",
            "startup_frames_trimmed",
            "output_layout",
            "output_dtype",
            "output_range",
            "resolution",
            "color_conversion",
        },
        context="taew21 decoder contract",
    )
    if decoder != {
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
    }:
        raise ValueError("taew2_1 decoder contract does not match the frozen implementation")

    campaign = value["campaign"]
    _strict_keys(
        campaign,
        {"output_root", "prompt_catalog", "strata", "horizons", "seeds", "jobs"},
        context="taew21 campaign",
    )
    prompts = campaign["prompt_catalog"]
    seeds = campaign["seeds"]
    jobs = campaign["jobs"]
    if not isinstance(prompts, dict) or len(prompts) != 16:
        raise ValueError("taew2_1 campaign requires exactly 16 prompts")
    if set(seeds) != set(prompts) or len(set(seeds.values())) != 16:
        raise ValueError("taew2_1 campaign requires one unique seed per prompt")
    require_fresh_seeds(seeds.values(), exclusions)
    if not isinstance(jobs, list) or len(jobs) != 32:
        raise ValueError("taew2_1 campaign requires exactly 32 jobs")
    expected_pairs = {(prompt_id, horizon) for prompt_id in prompts for horizon in (15, 60)}
    observed_pairs: set[tuple[str, int]] = set()
    output_root = PurePosixPath(campaign["output_root"])
    if output_root != PurePosixPath("outputs/taew21-campaign"):
        raise ValueError("taew2_1 campaign output root is not frozen")
    for job in jobs:
        _strict_keys(
            job,
            {"job_id", "prompt_id", "stratum", "seed", "horizon_seconds", "output"},
            context="taew21 campaign job",
        )
        prompt_id = job["prompt_id"]
        pair = (prompt_id, job["horizon_seconds"])
        if pair in observed_pairs or pair not in expected_pairs:
            raise ValueError(f"duplicate or unknown taew2_1 campaign job: {pair}")
        if job["seed"] != seeds.get(prompt_id):
            raise ValueError(f"taew2_1 campaign job seed mismatch: {job['job_id']}")
        output = PurePosixPath(job["output"])
        try:
            output.relative_to(output_root)
        except ValueError as error:
            raise ValueError("taew2_1 campaign output escapes campaign root") from error
        if ".." in output.parts:
            raise ValueError("taew2_1 campaign output contains parent traversal")
        observed_pairs.add(pair)
    if observed_pairs != expected_pairs:
        raise ValueError("taew2_1 campaign prompt/horizon coverage is incomplete")

    benchmark = value["benchmark"]
    if benchmark.get("minimum_complete_wall_speedup") != 1.5:
        raise ValueError("taew2_1 minimum complete-wall speedup must be 1.5x")
    if benchmark.get("maximum_target_wall_seconds") != 38.08434066666632:
        raise ValueError("taew2_1 complete-wall target does not match frozen baseline")
    human = value["human_study"]
    if (
        human.get("chance_accuracy") != 0.5
        or human.get("noticeability_ceiling") != 0.55
        or human.get("minimum_usable_judgments") != 271
        or human.get("planning_judgments") != [400, 600]
        or human.get("primary_rule")
        != "one-sided-95%-upper-confidence-bound-on-discrimination-accuracy<=0.55"
    ):
        raise ValueError("taew2_1 human-study decision rule is not frozen")