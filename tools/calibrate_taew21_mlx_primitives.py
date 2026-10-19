"""Calibrate PyTorch-versus-MLX numerical error without decoding model output."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional

SEED = 20260320
DTYPE = np.float16
LONGEST_PATH_CONVOLUTIONS = 35
ACCUMULATION_HEADROOM = 1.25
PROSPECTIVE_LIMITS = {
    "maximum_absolute_error": 0.05,
    "mean_absolute_error": 0.005,
    "minimum_cosine_similarity": 0.999,
}


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference64 = reference.astype(np.float64).reshape(-1)
    candidate64 = candidate.astype(np.float64).reshape(-1)
    difference = np.abs(reference64 - candidate64)
    denominator = np.linalg.norm(reference64) * np.linalg.norm(candidate64)
    cosine = (
        1.0
        if denominator == 0
        else float(np.dot(reference64, candidate64) / denominator)
    )
    return {
        "maximum_absolute_error": float(difference.max(initial=0)),
        "mean_absolute_error": float(difference.mean()),
        "cosine_similarity": cosine,
        "element_count": int(difference.size),
    }


def _torch_conv(
    value: np.ndarray, weight: np.ndarray, bias: np.ndarray | None
) -> np.ndarray:
    tensor = torch.from_numpy(value).permute(0, 3, 1, 2)
    kernel = torch.from_numpy(weight).permute(0, 3, 1, 2)
    torch_bias = None if bias is None else torch.from_numpy(bias)
    output = torch_functional.conv2d(
        tensor, kernel, torch_bias, padding=weight.shape[1] // 2
    )
    return output.permute(0, 2, 3, 1).numpy()


def _mlx_conv(
    mx: Any, value: np.ndarray, weight: np.ndarray, bias: np.ndarray | None
) -> np.ndarray:
    output = mx.conv2d(mx.array(value), mx.array(weight), padding=weight.shape[1] // 2)
    if bias is not None:
        output = output + mx.array(bias)
    mx.eval(output)
    return np.asarray(output)


def _random(
    rng: np.random.Generator, shape: tuple[int, ...], scale: float = 0.25
) -> np.ndarray:
    return (rng.standard_normal(shape) * scale).astype(DTYPE)


def calibrate() -> dict[str, Any]:
    import mlx.core as mx

    rng = np.random.default_rng(SEED)
    cases: dict[str, dict[str, float]] = {}

    clamp_input = _random(rng, (1, 5, 7, 4), scale=3.0)
    torch_clamp = (torch.tanh(torch.from_numpy(clamp_input) / 3) * 3).numpy()
    mlx_clamp = mx.tanh(mx.array(clamp_input) / 3) * 3
    mx.eval(mlx_clamp)
    cases["clamp"] = _metrics(torch_clamp, np.asarray(mlx_clamp))

    conv_input = _random(rng, (1, 7, 9, 4))
    conv_weight = _random(rng, (6, 3, 3, 4))
    conv_bias = _random(rng, (6,))
    cases["conv2d"] = _metrics(
        _torch_conv(conv_input, conv_weight, conv_bias),
        _mlx_conv(mx, conv_input, conv_weight, conv_bias),
    )

    torch_upsample = torch_functional.interpolate(
        torch.from_numpy(conv_input).permute(0, 3, 1, 2),
        scale_factor=2,
        mode="nearest",
    ).permute(0, 2, 3, 1).numpy()
    mlx_upsample = mx.repeat(mx.repeat(mx.array(conv_input), 2, axis=1), 2, axis=2)
    mx.eval(mlx_upsample)
    cases["nearest_upsample"] = _metrics(torch_upsample, np.asarray(mlx_upsample))

    tgrow_input = _random(rng, (1, 4, 5, 4))
    tgrow_weight = _random(rng, (8, 1, 1, 4))
    torch_tgrow_raw = torch_functional.conv2d(
        torch.from_numpy(tgrow_input).permute(0, 3, 1, 2),
        torch.from_numpy(tgrow_weight).permute(0, 3, 1, 2),
    )
    torch_tgrow = torch_tgrow_raw.reshape(2, 4, 4, 5).permute(0, 2, 3, 1).numpy()
    mlx_tgrow_raw = mx.conv2d(mx.array(tgrow_input), mx.array(tgrow_weight))
    mlx_tgrow = mx.concatenate(mx.split(mlx_tgrow_raw, 2, axis=-1), axis=0)
    mx.eval(mlx_tgrow)
    cases["temporal_growth"] = _metrics(torch_tgrow, np.asarray(mlx_tgrow))

    frames = [_random(rng, (1, 5, 6, 4)) for _ in range(3)]
    first_weight = _random(rng, (4, 3, 3, 8))
    first_bias = _random(rng, (4,))
    second_weight = _random(rng, (4, 3, 3, 4))
    second_bias = _random(rng, (4,))
    third_weight = _random(rng, (4, 3, 3, 4))
    third_bias = _random(rng, (4,))
    torch_memory: np.ndarray | None = None
    mlx_memory: np.ndarray | None = None
    torch_outputs = []
    mlx_outputs = []
    for frame in frames:
        torch_prior = np.zeros_like(frame) if torch_memory is None else torch_memory
        torch_value = np.concatenate([frame, torch_prior], axis=-1)
        torch_value = np.maximum(_torch_conv(torch_value, first_weight, first_bias), 0)
        torch_value = np.maximum(_torch_conv(torch_value, second_weight, second_bias), 0)
        torch_value = _torch_conv(torch_value, third_weight, third_bias)
        torch_outputs.append(np.maximum(torch_value + frame, 0))
        torch_memory = frame

        mlx_prior = np.zeros_like(frame) if mlx_memory is None else mlx_memory
        mlx_value = mx.concatenate([mx.array(frame), mx.array(mlx_prior)], axis=-1)
        mlx_value = mx.maximum(
            mx.conv2d(mlx_value, mx.array(first_weight), padding=1) + mx.array(first_bias), 0
        )
        mlx_value = mx.maximum(
            mx.conv2d(mlx_value, mx.array(second_weight), padding=1) + mx.array(second_bias), 0
        )
        mlx_value = mx.conv2d(mlx_value, mx.array(third_weight), padding=1) + mx.array(third_bias)
        mlx_value = mx.maximum(mlx_value + mx.array(frame), 0)
        mx.eval(mlx_value)
        mlx_outputs.append(np.asarray(mlx_value))
        mlx_memory = frame
    cases["memory_block_three_steps"] = _metrics(
        np.concatenate(torch_outputs, axis=0), np.concatenate(mlx_outputs, axis=0)
    )

    aggregate = {
        "maximum_absolute_error": max(
            case["maximum_absolute_error"] for case in cases.values()
        ),
        "maximum_mean_absolute_error": max(
            case["mean_absolute_error"] for case in cases.values()
        ),
        "minimum_cosine_similarity": min(
            case["cosine_similarity"] for case in cases.values()
        ),
    }
    return {
        "schema": "liveframe.taew21-mlx-primitive-calibration.v1",
        "status": "synthetic-before-pytorch-golden-capture",
        "seed": SEED,
        "dtype": "float16",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "mlx": importlib.metadata.version("mlx"),
        },
        "cases": cases,
        "aggregate": aggregate,
        "limit_derivation": {
            "longest_path_convolutions": LONGEST_PATH_CONVOLUTIONS,
            "accumulation_headroom": ACCUMULATION_HEADROOM,
            "unrounded_maximum_absolute_error": aggregate["maximum_absolute_error"]
            * LONGEST_PATH_CONVOLUTIONS
            * ACCUMULATION_HEADROOM,
            "unrounded_mean_absolute_error": aggregate["maximum_mean_absolute_error"]
            * LONGEST_PATH_CONVOLUTIONS
            * ACCUMULATION_HEADROOM,
            "rounding": "outward-to-simple-conservative-limits",
        },
        "prospective_limits": PROSPECTIVE_LIMITS,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = calibrate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(report["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()