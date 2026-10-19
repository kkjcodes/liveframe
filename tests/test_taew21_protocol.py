from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from liveframe.perceptual_policy import canonical_digest
from liveframe.taew21_protocol import (
    PORT_PARITY_POLICY_SCHEMA,
    build_seed_exclusions,
    load_port_parity_policy,
    require_fresh_seeds,
    validate_seed_exclusions,
    validate_taew21_protocol,
)


def test_seed_exclusions_are_digest_bound_and_reject_reuse(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    exclusions = build_seed_exclusions(
        {
            "calibration": (source, "sampling.calibration_seeds", [11, 17]),
            "opened-holdout": (source, "sampling.holdout_seeds", [23, 29]),
        }
    )

    validate_seed_exclusions(exclusions)
    require_fresh_seeds([31, 37], exclusions)
    with pytest.raises(ValueError, match="overlap frozen exclusions"):
        require_fresh_seeds([31, 23], exclusions)

    exclusions["excluded_seeds"].append(41)
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_seed_exclusions(exclusions)


def _parity_policy() -> dict[str, object]:
    digest = "sha256:" + "a" * 64
    return {
        "schema": PORT_PARITY_POLICY_SCHEMA,
        "status": "frozen-before-pytorch-golden-capture",
        "source": {
            "repository": "https://example.invalid/taehv",
            "revision": "abc123",
            "implementation_sha256": digest,
            "weights_sha256": digest,
            "pytorch_version": "2.8.0",
        },
        "fixture": {
            "artifact": "fixtures/latent.bf16",
            "sha256": digest,
            "dtype": "bfloat16",
            "layout": "B,F,C,H,W",
            "shape": [1, 21, 16, 60, 104],
        },
        "comparison": {
            "activation_anchors": ["decoder.0", "decoder.4", "decoder.output"],
            "final_outputs": ["float_decoder_output", "rgb_uint8"],
            "metrics": [
                "maximum_absolute_error",
                "mean_absolute_error",
                "cosine_similarity",
            ],
        },
        "limits": {
            "maximum_absolute_error": 0.02,
            "mean_absolute_error": 0.002,
            "minimum_cosine_similarity": 0.999,
        },
    }


def test_port_parity_policy_is_strict_and_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_parity_policy()), encoding="utf-8")

    first, first_digest = load_port_parity_policy(path)
    second, second_digest = load_port_parity_policy(path)

    assert first == second
    assert first_digest == second_digest


def test_port_parity_policy_rejects_unknown_keys(tmp_path: Path) -> None:
    policy = _parity_policy()
    policy["post_result_tuning"] = True
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unknown=.*post_result_tuning"):
        load_port_parity_policy(path)


def test_complete_taew21_protocol_rejects_recomputed_unsafe_mutations() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "artifacts/taew21-protocol.v1.json").read_text())
    exclusions = json.loads(
        (root / "artifacts/taew21-seed-exclusions.v1.json").read_text()
    )
    validate_taew21_protocol(protocol, exclusions)

    escaped = deepcopy(protocol)
    escaped["campaign"]["jobs"][0]["output"] = "outputs/outside-campaign"
    escaped["protocol_digest"] = canonical_digest(
        {key: value for key, value in escaped.items() if key != "protocol_digest"}
    )
    with pytest.raises(ValueError, match="escapes campaign root"):
        validate_taew21_protocol(escaped, exclusions)

    reused = deepcopy(protocol)
    prompt_id = next(iter(reused["campaign"]["seeds"]))
    reused["campaign"]["seeds"][prompt_id] = 42
    for job in reused["campaign"]["jobs"]:
        if job["prompt_id"] == prompt_id:
            job["seed"] = 42
    reused["protocol_digest"] = canonical_digest(
        {key: value for key, value in reused.items() if key != "protocol_digest"}
    )
    with pytest.raises(ValueError, match="overlap frozen exclusions"):
        validate_taew21_protocol(reused, exclusions)