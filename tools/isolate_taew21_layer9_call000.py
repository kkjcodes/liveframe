"""Isolate TAEW2.1 layer-9 call-000 semantics with a shared FP32 input."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections.abc import Callable
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

STAGES = ("input", "conv0", "relu0", "conv2", "relu2", "conv4", "residual_sum", "output")


def _load_upstream(path: Path) -> Any:
    specification = importlib.util.spec_from_file_location("taew21_layer9_upstream", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import upstream implementation: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _torch_block(block: Any, value: torch.Tensor) -> dict[str, np.ndarray]:
    concatenated = torch.cat([value, value * 0], dim=1)
    conv0 = block.conv[0](concatenated)
    relu0 = torch.relu(conv0)
    conv2 = block.conv[2](relu0)
    relu2 = torch.relu(conv2)
    conv4 = block.conv[4](relu2)
    residual_sum = conv4 + value
    output = torch.relu(residual_sum)
    tensors = {
        "input": value,
        "conv0": conv0,
        "relu0": relu0,
        "conv2": conv2,
        "relu2": relu2,
        "conv4": conv4,
        "residual_sum": residual_sum,
        "output": output,
    }
    return {name: tensor.detach().cpu().numpy() for name, tensor in tensors.items()}


def _im2col_convolution(decoder: TAEW21MLXDecoder, value: Any, name: str) -> Any:
    mx = decoder.mx
    weight = decoder.weights[f"{name}.weight"].astype(mx.float32)
    bias = decoder.weights[f"{name}.bias"].astype(mx.float32)
    height, width = int(value.shape[1]), int(value.shape[2])
    padded = mx.pad(value, [(0, 0), (1, 1), (1, 1), (0, 0)])
    patches = mx.concatenate(
        [
            padded[:, row : row + height, column : column + width, :]
            for row in range(3)
            for column in range(3)
        ],
        axis=-1,
    )
    flattened_weight = weight.reshape(int(weight.shape[0]), -1).transpose(1, 0)
    return patches @ flattened_weight + bias


def _mlx_block(
    decoder: TAEW21MLXDecoder,
    value_nchw: np.ndarray,
    convolution: Callable[[TAEW21MLXDecoder, Any, str], Any],
) -> dict[str, np.ndarray]:
    mx = decoder.mx
    value = mx.array(value_nchw.transpose(0, 2, 3, 1))
    concatenated = mx.concatenate([value, value * 0], axis=-1)
    conv0 = convolution(decoder, concatenated, "decoder.9.conv.0")
    relu0 = mx.maximum(conv0, 0)
    conv2 = convolution(decoder, relu0, "decoder.9.conv.2")
    relu2 = mx.maximum(conv2, 0)
    conv4 = convolution(decoder, relu2, "decoder.9.conv.4")
    residual_sum = conv4 + value
    output = mx.maximum(residual_sum, 0)
    tensors = {
        "input": value,
        "conv0": conv0,
        "relu0": relu0,
        "conv2": conv2,
        "relu2": relu2,
        "conv4": conv4,
        "residual_sum": residual_sum,
        "output": output,
    }
    mx.eval(*tensors.values())
    return {
        name: np.asarray(tensor).copy().transpose(0, 3, 1, 2)
        for name, tensor in tensors.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    bindings = {
        "upstream_source_sha256": (args.upstream_source, args.upstream_source_sha256),
        "source_weights_sha256": (args.source_weights, args.source_weights_sha256),
        "mlx_weights_sha256": (args.mlx_weights, args.mlx_weights_sha256),
        "fixture_sha256": (args.fixture, args.fixture_sha256),
    }
    for path, digest in bindings.values():
        if file_sha256(path) != digest:
            raise ValueError(f"layer-9 isolation artifact digest mismatch: {path}")

    upstream = _load_upstream(args.upstream_source)
    model = upstream.TAEHV(checkpoint_path=None)
    model.load_state_dict(load_file(str(args.source_weights), device="cpu"), strict=True)
    model = model.eval().float()
    shape = (1, 21, 16, 60, 104)
    latent = torch.from_file(
        str(args.fixture), shared=False, size=math.prod(shape), dtype=torch.bfloat16
    ).reshape(shape)[:, 0].float()

    layer9_input: torch.Tensor | None = None

    def hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        nonlocal layer9_input
        if layer9_input is None:
            layer9_input = inputs[0].detach().clone()

    handle = model.decoder[9].register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            model.decode_video(latent.unsqueeze(1), parallel=False, show_progress_bar=False)
    finally:
        handle.remove()
    if layer9_input is None:
        raise RuntimeError("PyTorch layer-9 call-000 input was not captured")

    decoder = TAEW21MLXDecoder(args.mlx_weights, expected_sha256=args.mlx_weights_sha256)
    reference = _torch_block(model.decoder[9], layer9_input)
    native_candidate = _mlx_block(
        decoder,
        layer9_input.numpy(),
        lambda model, value, name: model._convolution(
            value, name, padding=1, accumulation_dtype=model.mx.float32
        ),
    )
    im2col_candidate = _mlx_block(
        decoder, layer9_input.numpy(), _im2col_convolution
    )

    def compare(candidate: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            name: {
                "shape": list(reference[name].shape),
                "metrics": _metrics(reference[name], candidate[name]),
            }
            for name in STAGES
        }

    return {
        "schema": "liveframe.taew21-layer9-call000-shared-input-isolation.v1",
        "scope": "layer-9-call-000-shared-pytorch-input",
        "execution_dtype": "float32",
        "bindings": {
            key: {"artifact": str(path), "sha256": digest}
            for key, (path, digest) in bindings.items()
        },
        "native_conv2d_stages": compare(native_candidate),
        "im2col_matmul_stages": compare(im2col_candidate),
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
        raise FileExistsError(f"refusing to replace layer-9 isolation result: {args.output}")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
