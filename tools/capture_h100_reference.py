"""Capture deterministic native-resolution CF++ intermediates on an H100.

Run from the root of the pinned Causal-Forcing checkout. The output follows
``liveframe.golden-fixture.v1`` and contains only raw little-endian tensors plus
JSON metadata; no pickle payloads are emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

FIXTURE_SCHEMA = "liveframe.golden-fixture.v1"
SOURCE_REVISION = "8db419e341e5fc52542c0b2c4542728420ddfb4a"
MODEL_ARTIFACT_ID = "causal-forcing-plus-plus-framewise-2step"
MODEL_SHA256 = "8f899a016dffb9539788fac82c6200e43ef6e23b6e61ec6b885bb507a760ce42"
SELECTED_ATTENTION_LAYERS = frozenset({0, 1, 15, 29})


class Capture:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict[str, Any]] = []

    def tensor(self, name: str, role: str, value: torch.Tensor) -> None:
        tensor = value.detach().cpu().contiguous()
        dtype_name: str
        if tensor.dtype == torch.bfloat16:
            payload = tensor.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
            dtype_name = "bfloat16"
        elif tensor.dtype == torch.float16:
            payload = tensor.numpy().astype("<f2", copy=False).tobytes()
            dtype_name = "float16"
        elif tensor.dtype == torch.float32:
            payload = tensor.numpy().astype("<f4", copy=False).tobytes()
            dtype_name = "float32"
        elif tensor.dtype == torch.int64:
            payload = tensor.numpy().astype("<i8", copy=False).tobytes()
            dtype_name = "int64"
        else:
            raise TypeError(f"unsupported capture dtype for {name}: {tensor.dtype}")
        filename = f"{name}.bin"
        (self.output / filename).write_bytes(payload)
        self.entries.append(
            {
                "name": name,
                "role": role,
                "file": filename,
                "shape": list(tensor.shape),
                "dtype": dtype_name,
                "byte_order": "little",
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "scheduler_transition": None,
            }
        )

    def finish(self, metadata: dict[str, Any]) -> None:
        manifest = {
            "schema": FIXTURE_SCHEMA,
            "fixture_id": "cfpp-2step-native-block0-h100",
            "source_runtime": "pytorch-cuda-h100",
            "source_revision": SOURCE_REVISION,
            "model_artifact_id": MODEL_ARTIFACT_ID,
            "metadata": metadata,
            "tensors": self.entries,
        }
        (self.output / "fixture.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_generator(checkpoint: Path) -> Any:
    sys.path.insert(0, str(Path.cwd()))
    from utils.wan_wrapper import WanDiffusionWrapper

    generator = WanDiffusionWrapper(
        model_name="Wan2.1-T2V-1.3B", timestep_shift=5.0, is_causal=True
    )
    envelope = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = envelope["generator_ema"]
    normalized = {}
    for name, tensor in state.items():
        if name.startswith("model._fsdp_wrapped_module."):
            name = name.replace("model._fsdp_wrapped_module.", "model.", 1)
        normalized[name] = tensor
    generator.load_state_dict(normalized, strict=True)
    return generator.to(device="cuda", dtype=torch.bfloat16).eval()


def _install_capture_hooks(generator: Any, capture: Capture) -> tuple[list[Any], Any, Any]:
    import wan.modules.causal_model as causal_model

    handles = []

    def patch_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        flattened = output.flatten(2).transpose(1, 2)
        capture.tensor("patch_embedding", "intermediate", flattened)

    def time_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        capture.tensor("time_embedding", "intermediate", output)

    def projection_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        capture.tensor("time_modulation", "intermediate", output.reshape(1, 1, 6, -1))

    def head_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        capture.tensor("head_output", "intermediate", output)

    handles.append(generator.model.patch_embedding.register_forward_hook(patch_hook))
    handles.append(generator.model.time_embedding.register_forward_hook(time_hook))
    handles.append(generator.model.time_projection.register_forward_hook(projection_hook))
    handles.append(generator.model.head.register_forward_hook(head_hook))

    for index, block in enumerate(generator.model.blocks):
        def block_hook(
            _module: Any, _inputs: Any, output: torch.Tensor, *, layer: int = index
        ) -> None:
            capture.tensor(f"block.{layer:02d}.output", "intermediate", output)

        handles.append(block.register_forward_hook(block_hook))
        if index in SELECTED_ATTENTION_LAYERS:
            def self_attention_hook(
                _module: Any, _inputs: Any, output: torch.Tensor, *, layer: int = index
            ) -> None:
                capture.tensor(
                    f"block.{layer:02d}.self_attention.output", "intermediate", output
                )

            def cross_attention_hook(
                _module: Any, _inputs: Any, output: torch.Tensor, *, layer: int = index
            ) -> None:
                capture.tensor(
                    f"block.{layer:02d}.cross_attention.output", "intermediate", output
                )

            def ffn_hook(
                _module: Any, _inputs: Any, output: torch.Tensor, *, layer: int = index
            ) -> None:
                capture.tensor(f"block.{layer:02d}.ffn.output", "intermediate", output)

            handles.append(block.self_attn.register_forward_hook(self_attention_hook))
            handles.append(block.cross_attn.register_forward_hook(cross_attention_hook))
            handles.append(block.ffn.register_forward_hook(ffn_hook))

    original_rope = causal_model.causal_rope_apply
    rope_call = 0

    def captured_rope(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal rope_call
        output = original_rope(*args, **kwargs)
        layer, operation = divmod(rope_call, 2)
        if layer in SELECTED_ATTENTION_LAYERS:
            suffix = "query" if operation == 0 else "key"
            capture.tensor(f"block.{layer:02d}.self_attention.{suffix}", "intermediate", output)
        rope_call += 1
        return output

    original_attention = causal_model.attention
    attention_call = 0

    def captured_attention(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        nonlocal attention_call
        layer = attention_call
        if layer in SELECTED_ATTENTION_LAYERS:
            capture.tensor(
                f"block.{layer:02d}.self_attention.visible_key", "intermediate", key
            )
            capture.tensor(
                f"block.{layer:02d}.self_attention.visible_value", "intermediate", value
            )
        output = original_attention(query, key, value, *args, **kwargs)
        attention_call += 1
        return output

    causal_model.causal_rope_apply = captured_rope
    causal_model.attention = captured_attention
    return handles, original_rope, original_attention


def capture_reference(checkpoint: Path, output: Path) -> None:
    if _sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 does not match the pinned CF++ 2-step artifact")
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    generator = _load_generator(checkpoint)
    capture = Capture(output)
    handles, original_rope, original_attention = _install_capture_hooks(generator, capture)

    # Deterministic CPU construction; serialized values, not seed equivalence,
    # are the cross-runtime contract.
    initial_noise = torch.linspace(
        -1.0, 1.0, 16 * 60 * 104, dtype=torch.float32
    ).reshape(1, 1, 16, 60, 104).to(dtype=torch.bfloat16)
    prompt_embedding = torch.linspace(
        -0.25, 0.25, 512 * 4096, dtype=torch.float32
    ).reshape(1, 512, 4096).to(dtype=torch.bfloat16)
    timestep = torch.tensor([[1000.0]], dtype=torch.float32)
    capture.tensor("initial_noise", "initial_noise", initial_noise)
    capture.tensor("prompt_embedding", "conditioning", prompt_embedding)
    capture.tensor("timestep", "scheduler_state", timestep)

    blocks = len(generator.model.blocks)
    kv_cache = [
        {
            "k": torch.zeros((1, 32760, 12, 128), device="cuda", dtype=torch.bfloat16),
            "v": torch.zeros((1, 32760, 12, 128), device="cuda", dtype=torch.bfloat16),
            "global_end_index": torch.tensor([0], device="cuda", dtype=torch.long),
            "local_end_index": torch.tensor([0], device="cuda", dtype=torch.long),
        }
        for _ in range(blocks)
    ]
    cross_cache = [
        {
            "k": torch.zeros((1, 512, 12, 128), device="cuda", dtype=torch.bfloat16),
            "v": torch.zeros((1, 512, 12, 128), device="cuda", dtype=torch.bfloat16),
            "is_init": False,
        }
        for _ in range(blocks)
    ]

    try:
        flow, prediction = generator(
            noisy_image_or_video=initial_noise.cuda(),
            conditional_dict={"prompt_embeds": prompt_embedding.cuda()},
            timestep=timestep.cuda(),
            kv_cache=kv_cache,
            crossattn_cache=cross_cache,
            current_start=0,
        )
        torch.cuda.synchronize()
        capture.tensor("flow_prediction", "model_output", flow)
        capture.tensor("x0_prediction", "model_output", prediction)
    finally:
        for handle in handles:
            handle.remove()
        import wan.modules.causal_model as causal_model

        causal_model.causal_rope_apply = original_rope
        causal_model.attention = original_attention

    capture.finish(
        {
            "checkpoint_sha256": "sha256:" + MODEL_SHA256,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "native_latent_shape": [1, 1, 16, 60, 104],
            "tokens_per_frame": 1560,
            "warped_timestep": 1000.0,
            "tf32": False,
            "selected_attention_layers": sorted(SELECTED_ATTENTION_LAYERS),
            "nvidia_smi": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture_reference(args.checkpoint, args.output)


if __name__ == "__main__":
    main()