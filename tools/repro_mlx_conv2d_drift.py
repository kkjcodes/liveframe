"""Standalone reproduction: MLX fp32 conv2d accumulation drift vs PyTorch.

Self-contained (no project imports; only mlx + torch + numpy). Builds a
synthetic chain of 3x3 fp32 convolutions with ReLU - the shape of a tiny-VAE
decoder trunk - from one shared set of random weights, then runs it three
ways:

1. PyTorch CPU fp32 ``torch.nn.functional.conv2d`` (reference);
2. native MLX ``mx.conv2d`` on GPU;
3. MLX explicit padded im2col + matmul on GPU (mathematically the same
   convolution, different reduction order).

It reports, per depth, the max |diff| of (2) and (3) against (1), and of (2)
against (3) within MLX.

ROOT CAUSE (identified 2026-10-09): TF32 accumulation is enabled by default
for fp32 on M5 (NAX) GPUs. Run this script twice to demonstrate:

    python tools/repro_mlx_conv2d_drift.py            # TF32 default ON
    MLX_ENABLE_TF32=0 python tools/repro_mlx_conv2d_drift.py

Observed on Apple M5 Max, mlx 0.32.0, torch 2.13.0, single 3x3 fp32 conv vs
a float64 ground truth (output scale ~1): torch fp32 max err 1.4e-05; MLX
native conv2d 9.7e-02 default -> 1.4e-04 with TF32 off; MLX im2col matmul
5.3e-03 default -> 1.5e-05 with TF32 off. The native conv path (Winograd:
two transform-domain matmuls) is hit harder than one im2col matmul. This is
the mechanism behind the 0.0854-vs-0.00143 decoder-level drift recorded in
the LiveFrame TAEW2.1 bisection (2026-08), and matches ml-explore/mlx-swift
issue #492 (fp32 conv3d drops to TF32 on NAX since the Winograd path routes
through steel_matmul).

Usage: python repro_mlx_conv2d_drift.py [--depth 12] [--channels 256]
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np
import torch

SEED = 20260817
HEIGHT, WIDTH = 60, 104


def _im2col_conv(value: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    """3x3 same-padding convolution as explicit patch gather + matmul (NHWC)."""
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
    flattened = weight.reshape(int(weight.shape[0]), -1).transpose(1, 0)
    return patches @ flattened + bias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--channels", type=int, default=256)
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)
    channels = args.channels
    value_np = rng.standard_normal((1, HEIGHT, WIDTH, channels)).astype(np.float32)
    weights = [
        rng.standard_normal((channels, 3, 3, channels)).astype(np.float32)
        * (2.0 / (9 * channels)) ** 0.5
        for _ in range(args.depth)
    ]
    biases = [rng.standard_normal(channels).astype(np.float32) * 0.01 for _ in range(args.depth)]

    # PyTorch reference: NCHW, weight OIHW.
    torch_value = torch.from_numpy(value_np).permute(0, 3, 1, 2)
    # MLX: NHWC, weight OHWI (mx.conv2d convention: out, kh, kw, in).
    native = mx.array(value_np)
    im2col = mx.array(value_np)

    print(f"mlx {mx.__version__} | torch {torch.__version__} | device {mx.default_device()}")
    print(f"depth={args.depth} channels={channels} spatial={HEIGHT}x{WIDTH} dtype=float32")
    print(f"{'layer':>5} {'native_vs_torch':>16} {'im2col_vs_torch':>16} {'native_vs_im2col':>17}")
    for index, (weight_np, bias_np) in enumerate(zip(weights, biases, strict=True)):
        torch_weight = torch.from_numpy(weight_np).permute(0, 3, 1, 2).contiguous()
        torch_value = torch.nn.functional.conv2d(
            torch_value, torch_weight, torch.from_numpy(bias_np), padding=1
        ).relu()
        mlx_weight = mx.array(weight_np)
        bias = mx.array(bias_np)
        native = mx.maximum(mx.conv2d(native, mlx_weight, padding=1) + bias, 0)
        im2col = mx.maximum(_im2col_conv(im2col, mlx_weight, bias), 0)
        mx.eval(native, im2col)
        reference = torch_value.permute(0, 2, 3, 1).numpy()
        native_np = np.asarray(native)
        im2col_np = np.asarray(im2col)
        print(
            f"{index:>5} {np.abs(native_np - reference).max():>16.3e} "
            f"{np.abs(im2col_np - reference).max():>16.3e} "
            f"{np.abs(native_np - im2col_np).max():>17.3e}"
        )


if __name__ == "__main__":
    main()
