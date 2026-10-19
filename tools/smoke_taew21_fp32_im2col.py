"""Run the frozen Branch A FP32-im2col decoder smoke gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
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
        raise ValueError("TAEW2.1 smoke provenance mismatch")

    start_swap = _swap()
    if start_swap["used_mib"] != 0:
        raise RuntimeError(
            f"TAEW2.1 smoke requires zero start swap; "
            f"observed {start_swap['used_mib']} MiB"
        )
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
    digest = hashlib.sha256()
    chunks = []
    decoded = decoder.decode(decoder.mx.array(latent), request)
    for chunk in validate_decoder_chunks(decoded, request):
        payload = np.asarray(chunk.payload)
        if not payload.flags.c_contiguous:
            raise ValueError("TAEW2.1 smoke chunk payload must be contiguous")
        digest.update(payload.tobytes(order="C"))
        chunks.append(
            {
                "start_index": chunk.start_index,
                "shape": list(chunk.shape),
                "minimum": chunk.minimum,
                "maximum": chunk.maximum,
            }
        )
    end_swap = _swap()
    swap_delta = end_swap["used_mib"] - start_swap["used_mib"]
    if start_swap["used_mib"] != 0 or swap_delta != 0:
        raise RuntimeError(
            f"TAEW2.1 smoke requires zero start swap and zero delta; "
            f"start={start_swap['used_mib']} MiB delta={swap_delta} MiB"
        )
    return {
        "schema": "liveframe.taew21-fp32-im2col-smoke.v1",
        "policy_sha256": file_sha256(args.policy),
        "policy_digest": policy_digest,
        "parity_result_sha256": file_sha256(args.parity_result),
        "weights_sha256": args.weights_sha256,
        "fixture_sha256": args.fixture_sha256,
        "configuration": "float32-explicit-im2col",
        "request": {
            "latent_frames": 21,
            "native_frames": 81,
            "height": 480,
            "width": 832,
        },
        "chunks": chunks,
        "emitted_frames": sum(chunk["shape"][0] for chunk in chunks),
        "pixel_sha256": "sha256:" + digest.hexdigest(),
        "start_swap": start_swap,
        "end_swap": end_swap,
        "swap_delta_mib": swap_delta,
        "passed": True,
        "timing_authority": "none-smoke-only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parity-result", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace smoke result: {args.output}")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "passed": result["passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
