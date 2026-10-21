"""ABot-World-0-5B bridge adapter (runs INSIDE the model env /opt/envs/abot).

Drives amap-cvlab/ABot-World's CausalInferencePipeline block-by-block, mirroring
scripts/inference.py::_run_single_quant. One step() = one causal block of
num_frame_per_block(3) latents -> 12 decoded frames (frames_per_latent=4).
Released-pipeline defaults: vae_type from configs/default_config.yaml (taew2_2),
no fp8, 704x1280, native fps 12. We measure the open weights on an A100 (their
claim is 16 fps on an RTX 5090).
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Any

ABOT_REPO = os.environ.get("ABOT_REPO", "/opt/code/ABot-World")
STREAM_H, STREAM_W = 704, 1280
FRAMES_PER_LATENT = 4
NUM_FPB = 3  # configs/long_forcing_dmd.yaml num_frame_per_block
CHUNK_FRAMES = NUM_FPB * FRAMES_PER_LATENT  # 12 (later blocks: 4n)
CHUNK_FRAMES_FIRST = NUM_FPB * FRAMES_PER_LATENT - 3  # 9 (causal VAE first block: 4n-3)

# ABot KEY_ORDER = ["W","A","S","D","I","J","K","L"] (WASD move + IJKL camera look)
# LiveFrame token -> ABot keys.
TOKEN_TO_KEYS = {
    "W": ["W"], "S": ["S"], "A": ["A"], "D": ["D"],
    "U": ["I"], "N": ["K"], "L": ["J"], "R": ["L"],
    "0": [],
}
KEY_ORDER = ["W", "A", "S", "D", "I", "J", "K", "L"]


class ABotAdapter:
    def __init__(self) -> None:
        self._loaded = False

    def capabilities(self) -> dict[str, Any]:
        return {
            "name": "abot-world-0",
            "interaction_mode": "interactive",
            "native_fps": 12.0,
            "chunk_frames": CHUNK_FRAMES,
            "chunk_frames_first": CHUNK_FRAMES_FIRST,
            "resolution": [STREAM_H, STREAM_W],
            "actions": ["W", "S", "A", "D", "L", "R", "U", "N", "0"],
            "supports_seed": True,
            "license_notice": "Apache-2.0 (acvlab/ABot-World-0-5B-LF); claims 16fps on RTX 5090",
        }

    def load(self) -> None:
        os.chdir(ABOT_REPO)
        sys.path.insert(0, ABOT_REPO)
        import torch
        from web_client.pipeline_loader import decode_block_to_frames, get_pipeline

        self._torch = torch
        self._decode = decode_block_to_frames
        pipeline, _cfg, device = get_pipeline(vae_type=None, use_fp8_gemm=False, quant_type="none")
        self._pipe = pipeline
        self._device = device
        self._dtype = torch.bfloat16
        self._num_fpb = int(pipeline.num_frame_per_block)
        vae = pipeline.encoder if getattr(pipeline, "encoder", None) is not None else pipeline.vae
        up = getattr(vae, "upsampling_factor", 8)
        self._latent_shape = (1, self._num_fpb, vae.z_dim, STREAM_H // up, STREAM_W // up)
        self._loaded = True

    def reset(self, image: Any, prompt: str | None, seed: int) -> None:
        assert self._loaded, "reset before load"
        import numpy as np
        from PIL import Image

        self._start_path = "/tmp/abot_start.png"
        Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((STREAM_W, STREAM_H)).save(
            self._start_path
        )
        # Full cold reset between cases: reset_stream clears KV/crossattn but NOT the
        # VAE's streaming temporal cache, which otherwise makes a later case's first
        # block decode to 4n(=12) instead of the cold 4n-3(=9) and fails frame accounting.
        for obj in (
            getattr(self._pipe, "vae", None),
            getattr(getattr(self._pipe, "vae", None), "model", None),
            getattr(self._pipe, "encoder", None),
        ):
            if obj is not None and hasattr(obj, "clear_cache"):
                with contextlib.suppress(Exception):
                    obj.clear_cache()
        if hasattr(self._pipe, "_clear_generation_caches"):
            with contextlib.suppress(Exception):
                self._pipe._clear_generation_caches()
        self._pipe.set_prompts([prompt or "| unknown |"], device=self._device)
        self._pipe.reset_stream(1, dtype=self._dtype, device=self._device, initial_latent=None)
        self._torch.manual_seed(int(seed))
        self._first = True
        self._emission = 0

    def step(self, actions: list[str]):
        assert self._loaded, "step before load"
        import numpy as np

        keys = set()
        for tok in actions:
            keys.update(TOKEN_TO_KEYS.get(tok, []))
        action = {k: (k in keys) for k in KEY_ORDER}

        if self._first:
            self._pipe.set_first_frame_latent(
                self._start_path, height=STREAM_H, width=STREAM_W, device=self._device
            )
            self._first = False
        self._pipe.set_act(
            action, height=STREAM_H, width=STREAM_W, num_frames=self._num_fpb, device=self._device
        )
        noise = self._torch.randn(self._latent_shape, device=self._device, dtype=self._dtype)
        denoise_pred = self._pipe.generate_next_block(noise)
        frames = self._decode(self._pipe, denoise_pred) if denoise_pred is not None else []
        arr = np.ascontiguousarray(np.stack([np.asarray(f, dtype=np.uint8) for f in frames])) \
            if frames else np.zeros((0, STREAM_H, STREAM_W, 3), dtype=np.uint8)
        self._emission += 1
        yield (arr, int(arr.shape[0]))

    def run_script(self, script: Any):
        raise RuntimeError("abot-world-0 is interactive; the harness calls step()")

    def close(self) -> None:
        self._loaded = False
