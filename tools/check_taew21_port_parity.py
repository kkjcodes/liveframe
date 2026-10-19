"""Check the first TAEW2.1 MLX output against the frozen PyTorch golden."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.decoders.taew21_mlx import TAEW21MLXDecoder  # noqa: E402
from liveframe.manifests import file_sha256  # noqa: E402
from liveframe.taew21_protocol import validate_taew21_protocol  # noqa: E402
from liveframe.taew21_reference import (  # noqa: E402
    parse_activation_probes,
)

EXPECTED_FULL_CALL_COUNTS = {
    **{layer: 21 for layer in range(14)},
    **{layer: 42 for layer in range(14, 20)},
    **{layer: 84 for layer in range(20, 23)},
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"parity shape mismatch: reference={reference.shape}, candidate={candidate.shape}"
        )
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    maximum = 0.0
    absolute_sum = 0.0
    dot = 0.0
    reference_norm = 0.0
    candidate_norm = 0.0
    chunk_size = 1_000_000
    for start in range(0, reference_flat.size, chunk_size):
        stop = min(start + chunk_size, reference_flat.size)
        reference_chunk = reference_flat[start:stop].astype(np.float64)
        candidate_chunk = candidate_flat[start:stop].astype(np.float64)
        difference = np.abs(reference_chunk - candidate_chunk)
        maximum = max(maximum, float(difference.max(initial=0)))
        absolute_sum += float(difference.sum())
        dot += float(np.dot(reference_chunk, candidate_chunk))
        reference_norm += float(np.dot(reference_chunk, reference_chunk))
        candidate_norm += float(np.dot(candidate_chunk, candidate_chunk))
    denominator = math.sqrt(reference_norm * candidate_norm)
    cosine = 1.0 if denominator == 0 else dot / denominator
    return {
        "maximum_absolute_error": maximum,
        "mean_absolute_error": absolute_sum / reference_flat.size,
        "cosine_similarity": cosine,
        "element_count": int(reference_flat.size),
    }


def _verdict(metrics: dict[str, float | int], limits: dict[str, float]) -> list[str]:
    failures = []
    if metrics["maximum_absolute_error"] > limits["maximum_absolute_error"]:
        failures.append("maximum_absolute_error")
    if metrics["mean_absolute_error"] > limits["mean_absolute_error"]:
        failures.append("mean_absolute_error")
    if metrics["cosine_similarity"] < limits["minimum_cosine_similarity"]:
        failures.append("cosine_similarity")
    return failures


def check(args: argparse.Namespace) -> dict[str, Any]:
    protocol = _load_json(args.protocol)
    exclusions = _load_json(args.seed_exclusions)
    validate_taew21_protocol(protocol, exclusions)
    provenance = protocol["provenance"]
    amendment = _load_json(args.amendment)
    implementation_amendment = _load_json(args.implementation_amendment)
    fp32_amendment = _load_json(args.fp32_amendment)
    execution_amendment = _load_json(args.execution_amendment)
    expected_amendment_keys = {
        "schema",
        "status",
        "protocol_digest",
        "original_checker_sha256",
        "amended_checker_sha256",
        "scope",
        "visibility_at_amendment",
    }
    if set(amendment) != expected_amendment_keys:
        raise ValueError("TAEW2.1 protocol amendment keys mismatch")
    if (
        amendment["schema"] != "liveframe.taew21-protocol-amendment.v1"
        or amendment["status"] != "frozen-before-first-parity-metric-visibility"
        or amendment["protocol_digest"] != protocol["protocol_digest"]
        or amendment["original_checker_sha256"]
        != provenance["conversion"]["parity_checker_sha256"]
        or amendment["amended_checker_sha256"]
        != implementation_amendment.get("previous_checker_sha256")
        or amendment["scope"] != "full-layer-call-count-validation-only"
        or amendment["visibility_at_amendment"]
        != "mlx-computation-completed-no-candidate-tensor-metrics-read-or-reported"
    ):
        raise ValueError("TAEW2.1 protocol amendment does not authorize this checker")
    expected_implementation_keys = {
        "schema",
        "status",
        "protocol_digest",
        "previous_amendment_sha256",
        "failed_result_sha256",
        "failure_class",
        "visibility_at_amendment",
        "scope",
        "original_converter_sha256",
        "repaired_converter_sha256",
        "original_weights_sha256",
        "repaired_weights_sha256",
        "repaired_manifest_sha256",
        "previous_checker_sha256",
        "amended_checker_sha256",
        "policy_sha256",
        "limits_unchanged",
    }
    if set(implementation_amendment) != expected_implementation_keys:
        raise ValueError("TAEW2.1 implementation amendment keys mismatch")
    if (
        implementation_amendment["schema"]
        != "liveframe.taew21-protocol-amendment.v2"
        or implementation_amendment["status"]
        != "frozen-after-v1-no-go-before-repaired-candidate-metrics"
        or implementation_amendment["protocol_digest"] != protocol["protocol_digest"]
        or implementation_amendment["previous_amendment_sha256"]
        != file_sha256(args.amendment)
        or implementation_amendment["failed_result_sha256"]
        != file_sha256(args.failed_result)
        or implementation_amendment["failure_class"]
        != "noncontiguous-numpy-transpose-serialized-with-incorrect-element-order"
        or implementation_amendment["visibility_at_amendment"]
        != "v1-parity-no-go-visible-repair-selected-from-first-divergent-layer"
        or implementation_amendment["scope"]
        != "contiguous-weight-materialization-and-parity-rerun-only"
        or implementation_amendment["original_converter_sha256"]
        != provenance["conversion"]["converter_sha256"]
        or implementation_amendment["original_weights_sha256"]
        != provenance["conversion"]["weights_sha256"]
        or implementation_amendment["repaired_converter_sha256"]
        != file_sha256(REPOSITORY_ROOT / "src/liveframe/converters/taew21_to_mlx.py")
        or implementation_amendment["repaired_weights_sha256"]
        != file_sha256(args.weights)
        or implementation_amendment["repaired_manifest_sha256"]
        != file_sha256(args.conversion_manifest)
        or implementation_amendment["amended_checker_sha256"]
        != fp32_amendment.get("pre_repair_checker_sha256")
        or implementation_amendment["policy_sha256"]
        != file_sha256(args.previous_policy)
        or implementation_amendment["limits_unchanged"] is not True
    ):
        raise ValueError("TAEW2.1 implementation amendment does not authorize repair")
    expected_files = {
        args.weights: implementation_amendment["repaired_weights_sha256"],
        args.golden: provenance["parity"]["golden_tensors_sha256"],
        args.previous_policy: provenance["parity"]["policy_sha256"],
    }
    for path, expected in expected_files.items():
        if file_sha256(path) != expected:
            raise ValueError(f"TAEW2.1 parity artifact digest mismatch: {path}")
    previous_policy = _load_json(args.previous_policy)
    policy = _load_json(args.policy)
    expected_policy_keys = {
        "schema",
        "status",
        "supersedes",
        "change_control",
        "source",
        "fixture",
        "comparison",
        "limits",
    }
    if set(policy) != expected_policy_keys:
        raise ValueError("TAEW2.1 parity policy v2 keys mismatch")
    if (
        policy["schema"] != "liveframe.taew21-port-parity-policy.v2"
        or policy["status"]
        != "frozen-before-fp32-block-11-and-17-candidate-execution"
        or policy["supersedes"]
        != {
            "artifact": "artifacts/taew21-port-parity-policy.v1.json",
            "sha256": file_sha256(args.previous_policy),
        }
        or policy["source"] != previous_policy["source"]
        or policy["fixture"] != previous_policy["fixture"]
        or policy["comparison"] != previous_policy["comparison"]
        or policy["limits"]["float"] != previous_policy["limits"]
        or policy["limits"]["rgb_uint8"]
        != {
            "maximum_absolute_error": 13.0,
            "mean_absolute_error": 2.275,
            "minimum_cosine_similarity": 0.999,
        }
        or policy["change_control"]["float_limits_unchanged"] is not True
        or policy["change_control"]["changed_surface"] != "rgb_uint8-limits-only"
    ):
        raise ValueError("TAEW2.1 parity policy v2 changes exceed preregistered scope")
    expected_fp32_keys = {
        "schema",
        "status",
        "base_protocol_digest",
        "previous_amendment_sha256",
        "previous_result_sha256",
        "pre_repair_decoder_sha256",
        "pre_repair_checker_sha256",
        "policy_v2_sha256",
        "allowed_implementation_change",
        "unchanged_constraints",
        "visibility_at_freeze",
    }
    if set(fp32_amendment) != expected_fp32_keys:
        raise ValueError("TAEW2.1 FP32 amendment keys mismatch")
    if (
        fp32_amendment["schema"] != "liveframe.taew21-protocol-amendment.v3"
        or fp32_amendment["status"]
        != "frozen-before-fp32-block-11-and-17-candidate-execution"
        or fp32_amendment["base_protocol_digest"] != protocol["protocol_digest"]
        or fp32_amendment["previous_amendment_sha256"]
        != file_sha256(args.implementation_amendment)
        or fp32_amendment["previous_result_sha256"]
        != file_sha256(args.previous_result)
        or fp32_amendment["pre_repair_decoder_sha256"]
        != provenance["conversion"]["mlx_decoder_sha256"]
        or fp32_amendment["pre_repair_checker_sha256"]
        != implementation_amendment["amended_checker_sha256"]
        or fp32_amendment["policy_v2_sha256"] != file_sha256(args.policy)
        or fp32_amendment["allowed_implementation_change"]
        != {
            "accumulation_dtype": "float32",
            "cast_boundary": (
                "cast-each-selected-memory-block-input-prior-and-weights-to-"
                "float32-then-cast-block-output-to-input-dtype"
            ),
            "layers": [11, 17],
            "op_order": "unchanged",
        }
    ):
        raise ValueError("TAEW2.1 FP32 amendment does not authorize this repair")
    expected_execution_keys = {
        "schema",
        "status",
        "fp32_amendment_sha256",
        "policy_v2_sha256",
        "decoder_sha256",
        "checker_sha256",
        "weights_sha256",
        "golden_sha256",
        "output",
    }
    if set(execution_amendment) != expected_execution_keys:
        raise ValueError("TAEW2.1 execution amendment keys mismatch")
    if (
        execution_amendment["schema"] != "liveframe.taew21-execution-amendment.v1"
        or execution_amendment["status"] != "frozen-before-fp32-candidate-execution"
        or execution_amendment["fp32_amendment_sha256"]
        != file_sha256(args.fp32_amendment)
        or execution_amendment["policy_v2_sha256"] != file_sha256(args.policy)
        or execution_amendment["decoder_sha256"]
        != file_sha256(REPOSITORY_ROOT / "src/liveframe/decoders/taew21_mlx.py")
        or execution_amendment["checker_sha256"] != file_sha256(Path(__file__))
        or execution_amendment["weights_sha256"] != file_sha256(args.weights)
        or execution_amendment["golden_sha256"] != file_sha256(args.golden)
        or execution_amendment["output"] != str(args.output)
    ):
        raise ValueError("TAEW2.1 execution amendment does not bind this run")
    probes = parse_activation_probes(policy["comparison"]["activation_anchors"])
    selected = {(probe.layer, call) for probe in probes for call in probe.calls}

    fixture = policy["fixture"]
    latent_path = REPOSITORY_ROOT / fixture["artifact"]
    if file_sha256(latent_path) != fixture["sha256"]:
        raise ValueError("TAEW2.1 parity latent digest mismatch")
    latent = torch.from_file(
        str(latent_path),
        shared=False,
        size=math.prod(fixture["shape"]),
        dtype=torch.bfloat16,
    ).reshape(fixture["shape"])
    latent_numpy = latent.to(torch.float16).numpy()

    decoder = TAEW21MLXDecoder(
        args.weights,
        expected_sha256=implementation_amendment["repaired_weights_sha256"],
    )
    captured: dict[str, np.ndarray] = {}

    def capture(layer: int, call: int, output: Any) -> None:
        if (layer, call) not in selected:
            return
        decoder.mx.eval(output)
        value = np.asarray(output).copy().transpose(0, 3, 1, 2)
        captured[f"decoder.{layer}.call.{call:03d}"] = value

    candidate_float, call_counts = decoder.decode_float(
        decoder.mx.array(latent_numpy), capture=capture
    )
    decoder.mx.eval(candidate_float)
    candidate_float_numpy = np.asarray(candidate_float).copy().transpose(0, 1, 4, 2, 3)
    candidate_rgb = (
        np.asarray(candidate_float)[0].copy() * 255.0
    ).round().clip(0, 255).astype(np.uint8)
    if call_counts != EXPECTED_FULL_CALL_COUNTS:
        raise ValueError(f"TAEW2.1 MLX call-count mismatch: {call_counts}")
    if len(captured) != len(selected):
        raise ValueError("TAEW2.1 MLX activation capture is incomplete")

    rows: dict[str, Any] = {}
    limits = policy["limits"]
    with safe_open(args.golden, framework="np") as handle:
        candidates = {
            **captured,
            "float_decoder_output": candidate_float_numpy,
            "rgb_uint8": candidate_rgb,
        }
        if set(candidates) != set(handle.keys()):
            raise ValueError("TAEW2.1 candidate and golden tensor inventories differ")
        for name, candidate in sorted(candidates.items()):
            metrics = _metrics(handle.get_tensor(name), candidate)
            tensor_limits = limits["rgb_uint8"] if name == "rgb_uint8" else limits["float"]
            failures = _verdict(metrics, tensor_limits)
            rows[name] = {
                "shape": list(candidate.shape),
                "dtype": str(candidate.dtype),
                "metrics": metrics,
                "limits": tensor_limits,
                "failures": failures,
                "passed": not failures,
            }
    failed = [name for name, row in rows.items() if not row["passed"]]
    return {
        "schema": "liveframe.taew21-port-parity-result.v2",
        "protocol_digest": protocol["protocol_digest"],
        "amendment_sha256": file_sha256(args.amendment),
        "implementation_amendment_sha256": file_sha256(args.implementation_amendment),
        "fp32_amendment_sha256": file_sha256(args.fp32_amendment),
        "execution_amendment_sha256": file_sha256(args.execution_amendment),
        "checker_sha256": file_sha256(Path(__file__)),
        "policy_sha256": file_sha256(args.policy),
        "weights_sha256": implementation_amendment["repaired_weights_sha256"],
        "golden_sha256": provenance["parity"]["golden_tensors_sha256"],
        "limits": limits,
        "call_counts": {f"decoder.{layer}": count for layer, count in call_counts.items()},
        "tensors": rows,
        "failed_tensors": failed,
        "passed": not failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--implementation-amendment", type=Path, required=True)
    parser.add_argument("--fp32-amendment", type=Path, required=True)
    parser.add_argument("--execution-amendment", type=Path, required=True)
    parser.add_argument("--failed-result", type=Path, required=True)
    parser.add_argument("--previous-result", type=Path, required=True)
    parser.add_argument("--seed-exclusions", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--previous-policy", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace parity result: {args.output}")
    result = check(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "failed_tensors": result["failed_tensors"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()