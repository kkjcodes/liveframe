"""Run the amendment-v4.1 same-precision TAEW2.1 FP32 diagnostic."""

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

from check_taew21_port_parity import (  # noqa: E402
    EXPECTED_FULL_CALL_COUNTS,
    _metrics,
    _verdict,
)

from liveframe.decoders.taew21_mlx import TAEW21MLXDecoder  # noqa: E402
from liveframe.manifests import file_sha256  # noqa: E402
from liveframe.perceptual_policy import canonical_digest  # noqa: E402
from liveframe.taew21_reference import parse_activation_probes  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_amendments(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    amendment_v4 = _load_json(args.amendment_v4)
    amendment_v4_1 = _load_json(args.amendment_v4_1)
    amendment_v5 = _load_json(args.amendment_v5)
    for value, schema in (
        (amendment_v4, "liveframe.taew21-protocol-amendment.v4"),
        (amendment_v4_1, "liveframe.taew21-protocol-amendment.v4.1"),
        (amendment_v5, "liveframe.taew21-protocol-amendment.v5"),
    ):
        payload = dict(value)
        digest = payload.pop("amendment_digest")
        if value.get("schema") != schema or canonical_digest(payload) != digest:
            raise ValueError(f"invalid canonical amendment: {schema}")
    bindings = amendment_v4_1["bindings"]
    if (
        bindings["previous_amendment_sha256"] != file_sha256(args.amendment_v4)
        or bindings["previous_amendment_digest"] != amendment_v4["amendment_digest"]
        or amendment_v4_1["status"]
        != "frozen-before-same-precision-full-fp32-diagnostic-implementation"
    ):
        raise ValueError("amendment v4.1 does not bind amendment v4")
    if (
        amendment_v5["bindings"]["amendment_v4_1_sha256"]
        != file_sha256(args.amendment_v4_1)
        or amendment_v5["bindings"]["previous_result_sha256"]
        != file_sha256(args.previous_result)
        or amendment_v5["status"]
        != "frozen-before-fp32-im2col-fix-iteration-1-implementation"
        or amendment_v5["allowed_fix_iteration_1"]["convolution_scope"]
        != {
            "kernel": [3, 3],
            "padding": 1,
            "replacement": "explicit-padded-im2col-plus-matmul",
            "flatten_order": "kernel-y-kernel-x-input-channel",
        }
    ):
        raise ValueError("amendment v5 does not authorize this repair")
    return amendment_v4, amendment_v4_1, amendment_v5


def _validate_binding(args: argparse.Namespace) -> dict[str, Any]:
    binding = _load_json(args.execution_binding)
    expected_keys = {
        "schema",
        "status",
        "amendment_v4_sha256",
        "amendment_v4_1_sha256",
        "amendment_v5_sha256",
        "capture_tool_sha256",
        "diagnostic_runner_sha256",
        "comparison_engine_sha256",
        "decoder_sha256",
        "converter_sha256",
        "weights_sha256",
        "conversion_manifest_sha256",
        "fp32_reference_manifest_sha256",
        "fp32_reference_tensors_sha256",
        "policy_v2_sha256",
        "fixture_sha256",
        "output",
    }
    if set(binding) != expected_keys:
        raise ValueError("FP32 execution binding keys mismatch")
    expected = {
        "amendment_v4_sha256": file_sha256(args.amendment_v4),
        "amendment_v4_1_sha256": file_sha256(args.amendment_v4_1),
        "amendment_v5_sha256": file_sha256(args.amendment_v5),
        "capture_tool_sha256": file_sha256(
            REPOSITORY_ROOT / "tools/capture_taew21_pytorch_reference.py"
        ),
        "diagnostic_runner_sha256": file_sha256(Path(__file__)),
        "comparison_engine_sha256": file_sha256(
            REPOSITORY_ROOT / "tools/check_taew21_port_parity.py"
        ),
        "decoder_sha256": file_sha256(
            REPOSITORY_ROOT / "src/liveframe/decoders/taew21_mlx.py"
        ),
        "converter_sha256": file_sha256(
            REPOSITORY_ROOT / "src/liveframe/converters/taew21_to_mlx.py"
        ),
        "weights_sha256": file_sha256(args.weights),
        "conversion_manifest_sha256": file_sha256(args.conversion_manifest),
        "fp32_reference_manifest_sha256": file_sha256(args.golden_manifest),
        "fp32_reference_tensors_sha256": file_sha256(args.golden),
        "policy_v2_sha256": file_sha256(args.policy),
        "fixture_sha256": file_sha256(args.latent_fixture),
        "output": str(args.output),
    }
    if (
        binding["schema"] != "liveframe.taew21-fp32-fix1-execution-binding.v1"
        or binding["status"] != "frozen-before-mlx-fp32-im2col-candidate-execution"
    ):
        raise ValueError("unsupported FP32 execution binding")
    mismatches = {
        key: (binding[key], actual)
        for key, actual in expected.items()
        if binding[key] != actual
    }
    if mismatches:
        raise ValueError(f"FP32 execution binding mismatch: {mismatches}")
    return binding


def _validate_artifacts(args: argparse.Namespace, amendment_v4_1: dict[str, Any]) -> None:
    policy = _load_json(args.policy)
    conversion = _load_json(args.conversion_manifest)
    golden = _load_json(args.golden_manifest)
    if conversion["target"].get("target_dtype") != "float32":
        raise ValueError("diagnostic conversion must target float32")
    if conversion["target"]["sha256"] != file_sha256(args.weights):
        raise ValueError("diagnostic conversion manifest does not bind weights")
    if {row["dtype"] for row in conversion["mappings"]} != {"float32"}:
        raise ValueError("diagnostic conversion contains non-float32 tensors")
    if (
        golden["runtime"]["execution_dtype"] != "float32"
        or golden["fixture"]["input_cast"] != "bfloat16-to-float32"
        or golden["tensor_artifact"]["sha256"] != file_sha256(args.golden)
        or golden["source"]["weights_sha256"]
        != amendment_v4_1["bindings"]["source_weights_sha256"]
        or golden["fixture"]["sha256"] != file_sha256(args.latent_fixture)
    ):
        raise ValueError("diagnostic golden is not the authorized FP32 reference")
    if policy["limits"]["float"] != {
        "maximum_absolute_error": 0.05,
        "mean_absolute_error": 0.005,
        "minimum_cosine_similarity": 0.999,
    }:
        raise ValueError("FP32 diagnostic float limits changed")


def check(args: argparse.Namespace) -> dict[str, Any]:
    _amendment_v4, amendment_v4_1, _amendment_v5 = _validate_amendments(args)
    binding = _validate_binding(args)
    _validate_artifacts(args, amendment_v4_1)
    policy = _load_json(args.policy)
    probes = parse_activation_probes(policy["comparison"]["activation_anchors"])
    selected = {(probe.layer, call) for probe in probes for call in probe.calls}

    fixture = policy["fixture"]
    if Path(fixture["artifact"]).resolve() != args.latent_fixture.resolve():
        raise ValueError("FP32 diagnostic fixture path mismatch")
    latent = torch.from_file(
        str(args.latent_fixture),
        shared=False,
        size=math.prod(fixture["shape"]),
        dtype=torch.bfloat16,
    ).reshape(fixture["shape"])
    latent_numpy = latent.to(torch.float32).numpy()

    decoder = TAEW21MLXDecoder(
        args.weights,
        expected_sha256=binding["weights_sha256"],
        convolution_mode="fp32-im2col",
    )
    captured: dict[str, np.ndarray] = {}

    def capture(layer: int, call: int, output: Any) -> None:
        if (layer, call) not in selected:
            return
        decoder.mx.eval(output)
        value = np.asarray(output).copy().transpose(0, 3, 1, 2)
        if value.dtype != np.float32:
            raise ValueError(f"non-float32 activation at decoder.{layer}.call.{call:03d}")
        captured[f"decoder.{layer}.call.{call:03d}"] = value

    candidate_float, call_counts = decoder.decode_float(
        decoder.mx.array(latent_numpy), capture=capture
    )
    decoder.mx.eval(candidate_float)
    candidate_nfhwc = np.asarray(candidate_float).copy()
    if candidate_nfhwc.dtype != np.float32:
        raise ValueError("diagnostic decoder output is not float32")
    candidate_float_numpy = candidate_nfhwc.transpose(0, 1, 4, 2, 3)
    candidate_rgb = (candidate_nfhwc[0] * 255.0).round().clip(0, 255).astype(np.uint8)
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
            raise ValueError("FP32 candidate and reference tensor inventories differ")
        for name, candidate in sorted(candidates.items()):
            reference = handle.get_tensor(name)
            if name != "rgb_uint8" and reference.dtype != np.float32:
                raise ValueError(f"non-float32 diagnostic reference tensor: {name}")
            metrics = _metrics(reference, candidate)
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
        "schema": "liveframe.taew21-fp32-im2col-result.v1",
        "amendment_v4_sha256": file_sha256(args.amendment_v4),
        "amendment_v4_1_sha256": file_sha256(args.amendment_v4_1),
        "amendment_v5_sha256": file_sha256(args.amendment_v5),
        "execution_binding_sha256": file_sha256(args.execution_binding),
        "comparison_engine_sha256": binding["comparison_engine_sha256"],
        "diagnostic_runner_sha256": binding["diagnostic_runner_sha256"],
        "policy_v2_sha256": binding["policy_v2_sha256"],
        "weights_sha256": binding["weights_sha256"],
        "conversion_manifest_sha256": binding["conversion_manifest_sha256"],
        "fp32_reference_manifest_sha256": binding["fp32_reference_manifest_sha256"],
        "fp32_reference_tensors_sha256": binding["fp32_reference_tensors_sha256"],
        "call_counts": {f"decoder.{layer}": count for layer, count in call_counts.items()},
        "tensors": rows,
        "failed_tensors": failed,
        "passed": not failed,
        "timing_authority": "none-diagnostic-only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment-v4", type=Path, required=True)
    parser.add_argument("--amendment-v4-1", type=Path, required=True)
    parser.add_argument("--amendment-v5", type=Path, required=True)
    parser.add_argument("--execution-binding", type=Path, required=True)
    parser.add_argument("--previous-result", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--golden-manifest", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--latent-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace FP32 diagnostic result: {args.output}")
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
