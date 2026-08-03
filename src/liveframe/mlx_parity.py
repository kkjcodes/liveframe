"""Compare MLX causal Wan intermediates to a serialized H100 reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx_video.models.wan_2.config import WanModelConfig
from mlx_video.models.wan_2.utils import load_wan_model

from liveframe.backends.mlx_causal import CausalWanForward
from liveframe.fixtures import GoldenFixture, GoldenTensor, load_fixture


def _numpy_tensor(fixture: GoldenFixture, tensor: GoldenTensor) -> np.ndarray:
    path = fixture.root / tensor.file
    if tensor.dtype == "bfloat16":
        raw = np.fromfile(path, dtype="<u2").astype(np.uint32)
        return (raw << 16).view(np.float32).reshape(tensor.shape)
    dtypes = {
        "float16": "<f2",
        "float32": "<f4",
        "int64": "<i8",
    }
    return np.fromfile(path, dtype=dtypes[tensor.dtype]).reshape(tensor.shape)


def _mlx_tensor(fixture: GoldenFixture, tensor: GoldenTensor) -> mx.array:
    path = fixture.root / tensor.file
    if tensor.dtype == "bfloat16":
        raw = mx.array(np.fromfile(path, dtype="<u2")).view(mx.bfloat16)
    elif tensor.dtype == "float16":
        raw = mx.array(np.fromfile(path, dtype="<f2"))
    elif tensor.dtype == "float32":
        raw = mx.array(np.fromfile(path, dtype="<f4"))
    elif tensor.dtype == "int64":
        raw = mx.array(np.fromfile(path, dtype="<i8"))
    else:
        raise ValueError(f"unsupported parity dtype: {tensor.dtype}")
    return raw.reshape(tensor.shape)


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: reference={reference.shape}, candidate={candidate.shape}"
        )
    ref = reference.astype(np.float64, copy=False).reshape(-1)
    got = candidate.astype(np.float64, copy=False).reshape(-1)
    delta = np.abs(ref - got)
    denominator = np.maximum(np.abs(ref), 1e-6)
    ref_norm = float(np.linalg.norm(ref))
    got_norm = float(np.linalg.norm(got))
    cosine = float(np.dot(ref, got) / (ref_norm * got_norm)) if ref_norm and got_norm else 1.0
    return {
        "elements": ref.size,
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()),
        "rmse": float(math.sqrt(np.square(delta).mean())),
        "max_rel": float((delta / denominator).max(initial=0.0)),
        "cosine": cosine,
    }


def _evaluate_policy(comparisons: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []

    def require(name: str, metric: str, limit: float, *, minimum: bool = False) -> None:
        row = comparisons.get(name)
        if row is None:
            failures.append(f"missing required comparison: {name}")
            return
        value = float(row[metric])
        accepted = value >= limit if minimum else value <= limit
        if not accepted:
            operator = ">=" if minimum else "<="
            failures.append(f"{name}.{metric}={value} must be {operator} {limit}")

    require("patch_embedding", "max_abs", 0.01)
    require("time_embedding", "max_abs", 0.005)
    require("time_modulation", "max_abs", 0.01)
    for layer in range(30):
        name = f"block.{layer:02d}.output"
        require(name, "cosine", 0.9995, minimum=True)
        require(name, "rmse", 0.04)
    for layer in (0, 1, 15, 29):
        prefix = f"block.{layer:02d}"
        for suffix in (
            "self_attention.query",
            "self_attention.key",
            "self_attention.visible_key",
            "self_attention.visible_value",
            "self_attention.output",
            "cross_attention.output",
            "ffn.output",
        ):
            require(f"{prefix}.{suffix}", "cosine", 0.9995, minimum=True)
    for name in ("head_output", "flow_prediction", "x0_prediction"):
        require(name, "cosine", 0.9995, minimum=True)
        require(name, "rmse", 0.03)
        require(name, "max_abs", 0.1)
    return {
        "name": "native-block0-bfloat16.v1",
        "passed": not failures,
        "failures": failures,
    }


def run_parity(model_dir: Path, fixture_dir: Path) -> dict[str, Any]:
    fixture = load_fixture(fixture_dir)
    tensors = {tensor.name: tensor for tensor in fixture.tensors}
    initial = _mlx_tensor(fixture, tensors["initial_noise"])
    prompt = _mlx_tensor(fixture, tensors["prompt_embedding"])
    timestep = _mlx_tensor(fixture, tensors["timestep"])

    raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    allowed = WanModelConfig.__dataclass_fields__.keys()
    config = WanModelConfig(**{key: value for key, value in raw_config.items() if key in allowed})
    model = load_wan_model(model_dir / "model.safetensors", config)
    embedded_context = model.embed_text([prompt[0]])
    cross_kv = model.prepare_cross_kv(embedded_context)
    mx.eval(embedded_context, cross_kv)

    capture: dict[str, mx.array] = {}
    causal = CausalWanForward(model)
    latent = initial[0].transpose(1, 0, 2, 3)  # [F,C,H,W] -> [C,F,H,W]
    output = causal.predict_x0(
        [latent],
        warped_timestep=float(timestep.item()),
        embedded_context=embedded_context,
        cross_kv_caches=cross_kv,
        current_start=0,
        capture=capture,
    )[0]
    mx.eval(output, *capture.values())

    comparisons: dict[str, Any] = {}
    for name, tensor in tensors.items():
        candidate = capture.get(name)
        if candidate is None:
            continue
        candidate_np = np.array(candidate.astype(mx.float32))
        reference_np = _numpy_tensor(fixture, tensor)
        if name == "time_embedding" and candidate_np.ndim == reference_np.ndim + 1:
            candidate_np = candidate_np[:, 0]
        if name == "head_output" and reference_np.ndim == candidate_np.ndim + 1:
            reference_np = reference_np[:, 0]
        if name in {"flow_prediction", "x0_prediction"}:
            # MLX capture is [B,C,F,H,W], upstream wrapper emits [B,F,C,H,W].
            reference_np = reference_np.transpose(0, 2, 1, 3, 4)
        comparisons[name] = _metrics(reference_np, candidate_np)

    ordered = sorted(
        comparisons.items(), key=lambda item: float(item[1]["max_abs"]), reverse=True
    )
    policy = _evaluate_policy(comparisons)
    return {
        "schema": "liveframe.parity-report.v1",
        "fixture_id": fixture.fixture_id,
        "model_dir": str(model_dir),
        "compared_tensors": len(comparisons),
        "passed": policy["passed"],
        "policy": policy,
        "comparisons": dict(ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_parity(args.model_dir, args.fixture)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "compared_tensors": report["compared_tensors"],
                    "passed": report["passed"],
                },
                sort_keys=True,
            )
        )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()