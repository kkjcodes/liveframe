from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from liveframe.quality_capture import sha256_file

ROOT = Path(__file__).parents[1]
CLAIMS_PATH = ROOT / "artifacts" / "liveframe-publication-claims.v1.json"


def _claims() -> dict[str, Any]:
    return json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))


def test_publication_claim_sources_are_digest_bound() -> None:
    claims = _claims()
    assert claims["schema"] == "liveframe.publication-claims.v1"
    assert claims["package_status"] == "publication-fallback-active"

    evidence = [
        item
        for speedup in claims["engineering_speedups"]
        for item in (speedup["baseline"], speedup["candidate"])
    ]
    evidence.extend(result["evidence"] for result in claims["negative_results"])
    for item in evidence:
        assert sha256_file(ROOT / item["artifact"]) == item["sha256"]


def test_publication_speedup_is_derived_from_frozen_rows() -> None:
    speedup = _claims()["engineering_speedups"][0]
    baseline = speedup["baseline"]
    candidate = speedup["candidate"]
    candidate_artifact = json.loads(
        (ROOT / candidate["artifact"]).read_text(encoding="utf-8")
    )

    assert math.isclose(
        speedup["wall_speedup"],
        baseline["wall_seconds"] / candidate["wall_seconds"],
    )
    assert math.isclose(
        speedup["wall_reduction_fraction"],
        1 - candidate["wall_seconds"] / baseline["wall_seconds"],
    )
    assert math.isclose(
        speedup["vae_reduction_fraction"],
        1 - candidate["vae_seconds"] / baseline["vae_seconds"],
    )
    assert candidate["recorded_raw_pixel_anchor_sha256"] == (
        candidate_artifact["pixel_identity_anchor"]["sha256"]
    )
    assert candidate["raw_pixel_payload_retained"] is False
    assert candidate["encoded_video_sha256"] == candidate_artifact["video"]["sha256"]
    assert candidate["swap_delta_mib"] == 0.0


def test_publication_has_three_scoped_negative_results() -> None:
    results = _claims()["negative_results"]
    assert {result["id"] for result in results} == {
        "cross-runtime-ar-divergence",
        "m5-native-fps-target-gap",
        "periodic-refresh-kv-reuse",
    }
    assert all(result["scope"] and result["not_claimed"] for result in results)

    target_gap = next(
        result for result in results if result["id"] == "m5-native-fps-target-gap"
    )["evidence"]
    assert math.isclose(
        target_gap["target_gap_factor"],
        target_gap["target_fps"] / target_gap["generated_fps"],
    )
