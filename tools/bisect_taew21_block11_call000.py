"""Compare PyTorch and MLX block-11 call-000 intermediates in FP32."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "tools"))

from check_taew21_port_parity import _metrics  # noqa: E402

from liveframe.decoders.taew21_mlx import TAEW21MLXDecoder  # noqa: E402
from liveframe.manifests import file_sha256  # noqa: E402

STAGES = (
    "input",
    "memory",
    "concatenated",
    "conv0",
    "relu0",
    "conv2",
    "relu2",
    "conv4",
    "residual_sum",
    "output",
)


def _load_upstream(path: Path) -> Any:
    specification = importlib.util.spec_from_file_location("taew21_bisection_upstream", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import upstream implementation: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _pytorch_stages(block: Any, value: torch.Tensor) -> dict[str, np.ndarray]:
    memory = value * 0
    concatenated = torch.cat([value, memory], dim=1)
    conv0 = block.conv[0](concatenated)
    relu0 = torch.relu(conv0)
    conv2 = block.conv[2](relu0)
    relu2 = torch.relu(conv2)
    conv4 = block.conv[4](relu2)
    residual_sum = conv4 + value
    output = torch.relu(residual_sum)
    values = {
        "input": value,
        "memory": memory,
        "concatenated": concatenated,
        "conv0": conv0,
        "relu0": relu0,
        "conv2": conv2,
        "relu2": relu2,
        "conv4": conv4,
        "residual_sum": residual_sum,
        "output": output,
    }
    return {name: tensor.detach().cpu().contiguous().numpy() for name, tensor in values.items()}


def _mlx_stages(decoder: TAEW21MLXDecoder, value: Any) -> dict[str, np.ndarray]:
    mx = decoder.mx
    memory = value * 0
    concatenated = mx.concatenate([value, memory], axis=-1)
    conv0 = decoder._convolution(
        concatenated, "decoder.11.conv.0", padding=1, accumulation_dtype=mx.float32
    )
    relu0 = mx.maximum(conv0, 0)
    conv2 = decoder._convolution(
        relu0, "decoder.11.conv.2", padding=1, accumulation_dtype=mx.float32
    )
    relu2 = mx.maximum(conv2, 0)
    conv4 = decoder._convolution(
        relu2, "decoder.11.conv.4", padding=1, accumulation_dtype=mx.float32
    )
    residual_sum = conv4 + value
    output = mx.maximum(residual_sum, 0)
    values = {
        "input": value,
        "memory": memory,
        "concatenated": concatenated,
        "conv0": conv0,
        "relu0": relu0,
        "conv2": conv2,
        "relu2": relu2,
        "conv4": conv4,
        "residual_sum": residual_sum,
        "output": output,
    }
    mx.eval(*values.values())
    return {
        name: np.asarray(tensor).copy().transpose(0, 3, 1, 2)
        for name, tensor in values.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    expected = {
        args.upstream_source: args.upstream_source_sha256,
        args.source_weights: args.source_weights_sha256,
        args.mlx_weights: args.mlx_weights_sha256,
        args.fixture: args.fixture_sha256,
    }
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise ValueError(f"bisection artifact digest mismatch: {path}")

    upstream = _load_upstream(args.upstream_source)
    model = upstream.TAEHV(checkpoint_path=None)
    model.load_state_dict(load_file(str(args.source_weights), device="cpu"), strict=True)
    model = model.eval().to(dtype=torch.float32)

    shape = (1, 21, 16, 60, 104)
    latent = torch.from_file(
        str(args.fixture),
        shared=False,
        size=math.prod(shape),
        dtype=torch.bfloat16,
    ).reshape(shape)[:, 0].to(torch.float32)

    pytorch_input: torch.Tensor | None = None
    pytorch_upstream: dict[str, np.ndarray] = {}

    def block11_hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal pytorch_input
        if pytorch_input is None:
            pytorch_input = inputs[0].detach().clone()

    def upstream_hook(layer: int) -> Any:
        def capture(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            name = f"decoder.{layer}.output"
            if name not in pytorch_upstream:
                pytorch_upstream[name] = output.detach().cpu().contiguous().numpy()

        return capture

    handles = [model.decoder[11].register_forward_pre_hook(block11_hook)]
    handles.extend(
        model.decoder[layer].register_forward_hook(upstream_hook(layer))
        for layer in range(6, 11)
    )
    try:
        with torch.inference_mode():
            model.decode_video(latent.unsqueeze(1), parallel=False, show_progress_bar=False)
    finally:
        for handle in handles:
            handle.remove()
    if pytorch_input is None:
        raise RuntimeError("PyTorch block-11 call-000 input was not captured")
    pytorch = _pytorch_stages(model.decoder[11], pytorch_input)

    decoder = TAEW21MLXDecoder(args.mlx_weights, expected_sha256=args.mlx_weights_sha256)
    mlx_input: Any | None = None
    mlx_upstream: dict[str, np.ndarray] = {}
    original_memory_block = decoder._memory_block

    def capture_memory_block(value: Any, prior: Any | None, layer: int) -> Any:
        nonlocal mlx_input
        if layer == 11 and mlx_input is None:
            mlx_input = value
        return original_memory_block(value, prior, layer)

    decoder._memory_block = capture_memory_block  # type: ignore[method-assign]
    latent_nhwc = decoder.mx.array(latent.numpy().transpose(0, 2, 3, 1))

    def capture_upstream(layer: int, call: int, output: Any) -> None:
        if layer not in range(6, 11) or call != 0:
            return
        decoder.mx.eval(output)
        mlx_upstream[f"decoder.{layer}.output"] = (
            np.asarray(output).copy().transpose(0, 3, 1, 2)
        )

    decoder._decode_latent(latent_nhwc, {}, {}, capture_upstream)
    if mlx_input is None:
        raise RuntimeError("MLX block-11 call-000 input was not captured")
    mlx = _mlx_stages(decoder, mlx_input)

    upstream_rows = {}
    for layer in range(6, 11):
        name = f"decoder.{layer}.output"
        upstream_rows[name] = {
            "shape": list(pytorch_upstream[name].shape),
            "metrics": _metrics(pytorch_upstream[name], mlx_upstream[name]),
        }

    rows = {}
    first_nonzero = None
    for name in STAGES:
        metrics = _metrics(pytorch[name], mlx[name])
        if first_nonzero is None and metrics["maximum_absolute_error"] != 0:
            first_nonzero = name
        rows[name] = {
            "shape": list(pytorch[name].shape),
            "pytorch_dtype": str(pytorch[name].dtype),
            "mlx_dtype": str(mlx[name].dtype),
            "metrics": metrics,
        }
    return {
        "schema": "liveframe.taew21-block11-call000-bisection.v1",
        "scope": "block-11-call-000-only",
        "execution_dtype": "float32",
        "bindings": {str(path): digest for path, digest in expected.items()},
        "first_nonzero_stage": first_nonzero,
        "upstream_stages": upstream_rows,
        "stages": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--upstream-source-sha256", required=True)
    parser.add_argument("--source-weights", type=Path, required=True)
    parser.add_argument("--source-weights-sha256", required=True)
    parser.add_argument("--mlx-weights", type=Path, required=True)
    parser.add_argument("--mlx-weights-sha256", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace bisection result: {args.output}")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "first_nonzero_stage": result["first_nonzero_stage"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
