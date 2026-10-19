"""Run the frozen Branch A FP32-im2col decoder timing screen.

Screening authority is routing-only per the frozen precision policy
(`artifacts/taew21-precision-policy.v1.json`): a PASS routes to the original
one-warmup/five-trial formal benchmark; a FAIL stops the FP32-im2col shipping
track with no timing claim. This runner is frozen before any timing is visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.decoder_contract import DecoderRequest, validate_decoder_chunks  # noqa: E402
from liveframe.manifests import file_sha256  # noqa: E402
from liveframe.mlx_cfpp_benchmark import parse_swap_usage  # noqa: E402
from liveframe.perceptual_policy import canonical_digest  # noqa: E402


def _swap() -> dict[str, float]:
    output = subprocess.run(
        ["sysctl", "vm.swapusage"], check=True, capture_output=True, text=True
    ).stdout
    return parse_swap_usage(output)


def _decode_once(decoder, latent, request) -> tuple[float, str, int]:
    """Decode the fixture once; return (wall seconds, pixel digest, frames).

    The timing boundary covers the full decoder wall: the decode call plus
    materialization of every chunk to contiguous host memory, matching the
    boundary of the frozen 41.319 s full-VAE comparator.
    """
    digest = hashlib.sha256()
    frames = 0
    start = time.perf_counter()
    decoded = decoder.decode(decoder.mx.array(latent), request)
    for chunk in validate_decoder_chunks(decoded, request):
        payload = np.asarray(chunk.payload)
        if not payload.flags.c_contiguous:
            raise ValueError("TAEW2.1 screen chunk payload must be contiguous")
        digest.update(payload.tobytes(order="C"))
        frames += chunk.shape[0]
    wall = time.perf_counter() - start
    return wall, "sha256:" + digest.hexdigest(), frames


def run(args: argparse.Namespace) -> dict[str, object]:
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    policy_payload = dict(policy)
    policy_digest = policy_payload.pop("policy_digest")
    if canonical_digest(policy_payload) != policy_digest:
        raise ValueError("TAEW2.1 precision policy canonical digest mismatch")
    provenance = policy["provenance"]
    if (
        provenance["decoder_sha256"]
        != file_sha256(REPOSITORY_ROOT / "src/liveframe/decoders/taew21_mlx.py")
        or provenance["parity_v5_sha256"] != file_sha256(args.parity_result)
        or file_sha256(args.weights) != args.weights_sha256
        or file_sha256(args.fixture) != args.fixture_sha256
    ):
        raise ValueError("TAEW2.1 screen provenance mismatch")
    smoke = json.loads(args.smoke_result.read_text(encoding="utf-8"))
    if not smoke.get("passed") or smoke.get("schema") != "liveframe.taew21-fp32-im2col-smoke.v1":
        raise ValueError("TAEW2.1 screen requires a passed smoke result")
    expected_pixels = smoke["pixel_sha256"]

    screen = policy["screening"]
    warmups = int(screen["warmups"])
    timed_trials = int(screen["timed_trials"])
    ceiling = float(screen["maximum_candidate_median_seconds"])

    from liveframe.decoders.taew21_mlx import TAEW21MLXDecoder

    shape = (1, 21, 16, 60, 104)
    latent = torch.from_file(
        str(args.fixture), shared=False, size=math.prod(shape), dtype=torch.bfloat16
    ).reshape(shape).float().numpy()
    decoder = TAEW21MLXDecoder(
        args.weights,
        expected_sha256=args.weights_sha256,
        convolution_mode="fp32-im2col",
    )
    request = DecoderRequest(latent_frames=21, native_frames=81, width=832, height=480)

    trials: list[dict[str, object]] = []
    for index in range(warmups + timed_trials):
        role = "warmup" if index < warmups else "timed"
        start_swap = _swap()
        if start_swap["used_mib"] != 0:
            raise RuntimeError(
                f"TAEW2.1 screen trial {index} requires zero start swap; "
                f"observed {start_swap['used_mib']} MiB"
            )
        wall, pixels, frames = _decode_once(decoder, latent, request)
        end_swap = _swap()
        swap_delta = end_swap["used_mib"] - start_swap["used_mib"]
        if swap_delta != 0:
            raise RuntimeError(
                f"TAEW2.1 screen trial {index} requires zero swap delta; "
                f"observed {swap_delta} MiB"
            )
        if pixels != expected_pixels:
            raise ValueError(
                f"TAEW2.1 screen trial {index} pixel digest {pixels} does not "
                f"match smoke digest {expected_pixels}"
            )
        if frames != request.native_frames:
            raise ValueError(
                f"TAEW2.1 screen trial {index} emitted {frames} frames; "
                f"expected {request.native_frames}"
            )
        trials.append(
            {
                "index": index,
                "role": role,
                "wall_seconds": wall,
                "start_swap": start_swap,
                "end_swap": end_swap,
                "swap_delta_mib": swap_delta,
                "pixel_sha256": pixels,
                "emitted_frames": frames,
            }
        )

    timed = [trial["wall_seconds"] for trial in trials if trial["role"] == "timed"]
    median = statistics.median(timed)
    passed = median <= ceiling
    return {
        "schema": "liveframe.taew21-fp32-im2col-screen.v1",
        "policy_sha256": file_sha256(args.policy),
        "policy_digest": policy_digest,
        "parity_result_sha256": file_sha256(args.parity_result),
        "smoke_result_sha256": file_sha256(args.smoke_result),
        "weights_sha256": args.weights_sha256,
        "fixture_sha256": args.fixture_sha256,
        "configuration": "float32-explicit-im2col",
        "request": {
            "latent_frames": 21,
            "native_frames": 81,
            "height": 480,
            "width": 832,
        },
        "warmups": warmups,
        "timed_trials": timed_trials,
        "trials": trials,
        "timed_wall_seconds": timed,
        "median_wall_seconds": median,
        "maximum_candidate_median_seconds": ceiling,
        "baseline_vae_median_seconds": screen["baseline_vae_median_seconds"],
        "authority": screen["authority"],
        "passed": passed,
        "verdict": (
            "proceed-to-original-five-trial-formal-decoder-and-complete-wall-benchmark"
            if passed
            else "stop-fp32-im2col-shipping-track-no-timing-claim"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parity-result", type=Path, required=True)
    parser.add_argument("--smoke-result", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace screen result: {args.output}")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "median_wall_seconds": result["median_wall_seconds"],
                "passed": result["passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
