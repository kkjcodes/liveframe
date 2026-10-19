"""Capture the pinned TAEW2.1 PyTorch golden after parity policy freeze."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.fixtures import load_fixture  # noqa: E402
from liveframe.manifests import file_sha256  # noqa: E402
from liveframe.taew21_protocol import load_port_parity_policy  # noqa: E402
from liveframe.taew21_reference import (  # noqa: E402
    EXPECTED_LAYER_CALLS,
    parse_activation_probes,
)


def _load_upstream(path: Path) -> Any:
    specification = importlib.util.spec_from_file_location("liveframe_pinned_taehv", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import pinned TAEW2.1 implementation from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _fixture_tensor(fixture_dir: Path, policy: dict[str, Any]) -> torch.Tensor:
    fixture = load_fixture(fixture_dir)
    matches = [tensor for tensor in fixture.tensors if tensor.name == "output_latents"]
    if len(matches) != 1:
        raise ValueError("fixture must contain exactly one output_latents tensor")
    entry = matches[0]
    expected = policy["fixture"]
    path = fixture_dir / entry.file
    if (
        str(path.resolve().relative_to(REPOSITORY_ROOT)) != expected["artifact"]
        or entry.sha256 != expected["sha256"]
        or entry.dtype != expected["dtype"]
        or list(entry.shape) != expected["shape"]
    ):
        raise ValueError("fixture output_latents does not match frozen parity policy")
    if expected["layout"] != "B,F,C,H,W" or entry.dtype != "bfloat16":
        raise ValueError("unsupported TAEW2.1 parity fixture layout or dtype")
    return torch.from_file(
        str(path), shared=False, size=math.prod(entry.shape), dtype=torch.bfloat16
    ).reshape(entry.shape)


def capture(args: argparse.Namespace) -> Path:
    policy, policy_digest = load_port_parity_policy(args.policy)
    source = policy["source"]
    if file_sha256(args.upstream_source) != source["implementation_sha256"]:
        raise ValueError("pinned TAEW2.1 implementation digest mismatch")
    if file_sha256(args.checkpoint) != source["weights_sha256"]:
        raise ValueError("pinned TAEW2.1 weight digest mismatch")
    if torch.__version__ != source["pytorch_version"]:
        raise ValueError("PyTorch version does not match frozen parity policy")
    probes = parse_activation_probes(policy["comparison"]["activation_anchors"])
    if args.output.exists():
        raise FileExistsError(f"refusing to replace TAEW2.1 golden: {args.output}")

    upstream = _load_upstream(args.upstream_source)
    execution_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.execution_dtype]
    model = upstream.TAEHV(checkpoint_path=None)
    state = load_file(str(args.checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    model = model.eval().to(dtype=execution_dtype)
    latent = _fixture_tensor(args.fixture_dir, policy).to(dtype=execution_dtype)

    captured: dict[str, torch.Tensor] = {}
    call_counts = {probe.layer: 0 for probe in probes}
    handles = []
    for probe in probes:
        selected = frozenset(probe.calls)

        def hook(
            _module: Any,
            _inputs: Any,
            output: torch.Tensor,
            *,
            layer: int = probe.layer,
            selected_calls: frozenset[int] = selected,
        ) -> None:
            call = call_counts[layer]
            if call in selected_calls:
                captured[f"decoder.{layer}.call.{call:03d}"] = output.detach().cpu().contiguous()
            call_counts[layer] = call + 1

        handles.append(model.decoder[probe.layer].register_forward_hook(hook))
    try:
        with torch.inference_mode():
            output = model.decode_video(latent, parallel=False, show_progress_bar=True)
    finally:
        for handle in handles:
            handle.remove()

    if call_counts != EXPECTED_LAYER_CALLS:
        raise ValueError(f"TAEW2.1 causal call-count mismatch: {call_counts}")
    expected_capture_count = sum(len(probe.calls) for probe in probes)
    if len(captured) != expected_capture_count:
        raise ValueError("TAEW2.1 activation capture is incomplete")
    expected_output_shape = (1, 81, 3, 480, 832)
    if tuple(output.shape) != expected_output_shape:
        raise ValueError(
            f"TAEW2.1 output shape mismatch: expected {expected_output_shape}, "
            f"observed {tuple(output.shape)}"
        )
    captured["float_decoder_output"] = output.cpu().contiguous()
    captured["rgb_uint8"] = (
        output[0]
        .permute(0, 2, 3, 1)
        .mul(255)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .contiguous()
    )

    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        tensors_path = temporary / "taew21-pytorch-golden.safetensors"
        save_file(captured, tensors_path)
        manifest = {
            "schema": "liveframe.taew21-pytorch-golden.v1",
            "status": "captured-under-frozen-port-parity-policy",
            "policy": {
                "artifact": str(args.policy.resolve().relative_to(REPOSITORY_ROOT)),
                "canonical_digest": policy_digest,
                "sha256": file_sha256(args.policy),
            },
            "source": {
                "repository": source["repository"],
                "revision": source["revision"],
                "implementation_sha256": source["implementation_sha256"],
                "weights_sha256": source["weights_sha256"],
            },
            "fixture": {
                **policy["fixture"],
                "manifest_sha256": file_sha256(args.fixture_dir / "fixture.json"),
                "input_cast": f"bfloat16-to-{args.execution_dtype}",
            },
            "runtime": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "device": "cpu",
                "execution_dtype": args.execution_dtype,
                "capture_tool_sha256": file_sha256(Path(__file__)),
            },
            "call_counts": {f"decoder.{layer}": count for layer, count in call_counts.items()},
            "tensors": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in sorted(captured.items())
            },
            "tensor_artifact": {
                "file": tensors_path.name,
                "sha256": file_sha256(tensors_path),
            },
        }
        manifest_path = temporary / "taew21-pytorch-golden.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return args.output / "taew21-pytorch-golden.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--execution-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--output", type=Path, required=True)
    manifest = capture(parser.parse_args())
    print(json.dumps({"manifest": str(manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()