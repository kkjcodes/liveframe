"""Validate the frozen TAEW2.1 campaign contract and its gate chain.

Runs the pre-registered seed-exclusion and protocol validation, then verifies
the full authorization chain (parity v5 GO -> smoke PASS -> screening PASS ->
formal benchmark PASS) before any campaign job may start. Emits an immutable
authorization record; the campaign runner refuses to start without it.
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
from liveframe.perceptual_policy import canonical_digest  # noqa: E402
from liveframe.taew21_protocol import (  # noqa: E402
    validate_seed_exclusions,
    validate_taew21_protocol,
)

MLX_MODEL_DIGEST = (
    "sha256:1f15d166a8a60bed70c2a8ab2923f40bd2b6b53ed2e0be87905bf45d0666d971"
)
MLX_T5_DIGEST = (
    "sha256:e86ee4199903e00a88dcd43583a43a6eb898cef600e38670f222d7e37d163787"
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parity-result", type=Path, required=True)
    parser.add_argument("--smoke-result", type=Path, required=True)
    parser.add_argument("--screen-result", type=Path, required=True)
    parser.add_argument("--formal-result", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tiny-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace authorization record: {args.output}")

    protocol = _object(args.protocol)
    exclusions = _object(args.exclusions)
    validate_seed_exclusions(exclusions)
    validate_taew21_protocol(protocol, exclusions)
    if (
        protocol["provenance"]["seed_exclusions"]["artifact_sha256"]
        != file_sha256(args.exclusions)
    ):
        raise ValueError("seed exclusion artifact digest does not match protocol")

    policy = _object(args.policy)
    policy_payload = dict(policy)
    policy_digest = policy_payload.pop("policy_digest")
    if canonical_digest(policy_payload) != policy_digest:
        raise ValueError("precision policy canonical digest mismatch")
    provenance = policy["provenance"]
    if provenance["protocol_sha256"] != file_sha256(args.protocol):
        raise ValueError("precision policy does not bind this protocol file")
    if provenance["parity_v5_sha256"] != file_sha256(args.parity_result):
        raise ValueError("parity v5 result digest mismatch")
    if provenance["decoder_sha256"] != file_sha256(
        REPOSITORY_ROOT / "src/liveframe/decoders/taew21_mlx.py"
    ):
        raise ValueError("frozen decoder source digest mismatch")

    parity = _object(args.parity_result)
    if not parity.get("passed"):
        raise ValueError("parity v5 result is not a GO")
    smoke = _object(args.smoke_result)
    if (
        smoke.get("schema") != "liveframe.taew21-fp32-im2col-smoke.v1"
        or not smoke.get("passed")
    ):
        raise ValueError("smoke result is not a PASS")
    screen = _object(args.screen_result)
    if (
        screen.get("schema") != "liveframe.taew21-fp32-im2col-screen.v1"
        or not screen.get("passed")
        or screen.get("smoke_result_sha256") != file_sha256(args.smoke_result)
    ):
        raise ValueError("screening result is not a PASS bound to this smoke result")
    formal = _object(args.formal_result)
    if (
        formal.get("schema") != "liveframe.taew21-cfpp-formal-benchmark.v1"
        or not formal.get("passed")
        or formal["provenance"]["screen_result_sha256"] != file_sha256(args.screen_result)
        or formal["provenance"]["policy_sha256"] != file_sha256(args.policy)
    ):
        raise ValueError("formal benchmark result is not a PASS bound to this chain")

    tiny_expected = provenance_tiny = None
    for candidate in policy["candidate_priority"]:
        if candidate.get("configuration") == "float32-explicit-im2col":
            provenance_tiny = candidate
    if provenance_tiny is None:
        raise ValueError("policy does not prioritize the fp32-im2col candidate")
    tiny_expected = formal["provenance"]["taew21_weights_sha256"]
    if file_sha256(args.tiny_weights) != tiny_expected:
        raise ValueError("tiny decoder weights digest mismatch vs formal benchmark")

    asset_digests = {
        "model.safetensors": (
            MLX_MODEL_DIGEST,
            file_sha256(args.model_dir / "model.safetensors"),
        ),
        "t5_encoder.safetensors": (
            MLX_T5_DIGEST,
            file_sha256(args.model_dir / "t5_encoder.safetensors"),
        ),
        "vae.safetensors": (
            protocol["provenance"]["full_vae"]["weights_sha256"],
            file_sha256(args.model_dir / "vae.safetensors"),
        ),
    }
    mismatched = [
        name for name, (expected, actual) in asset_digests.items() if expected != actual
    ]
    if mismatched:
        raise ValueError(f"campaign model asset digest mismatch: {mismatched}")

    record = {
        "schema": "liveframe.taew21-campaign-authorization.v1",
        "protocol_sha256": file_sha256(args.protocol),
        "protocol_digest": protocol["protocol_digest"],
        "exclusions_sha256": file_sha256(args.exclusions),
        "exclusions_digest": exclusions["exclusions_digest"],
        "policy_sha256": file_sha256(args.policy),
        "policy_digest": policy_digest,
        "parity_result_sha256": file_sha256(args.parity_result),
        "smoke_result_sha256": file_sha256(args.smoke_result),
        "screen_result_sha256": file_sha256(args.screen_result),
        "formal_result_sha256": file_sha256(args.formal_result),
        "tiny_weights_sha256": tiny_expected,
        "model_dir": str(args.model_dir),
        "model_assets": {
            name: actual for name, (_, actual) in asset_digests.items()
        },
        "job_count": len(protocol["campaign"]["jobs"]),
        "validator_sha256": file_sha256(Path(__file__).resolve()),
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "passed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
