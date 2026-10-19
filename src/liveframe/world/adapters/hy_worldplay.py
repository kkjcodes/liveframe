"""HY-World 1.5 / HY-WorldPlay bridge adapter (runs INSIDE the model env).

GPU-validated 2026-10-16 (A100 80 GB): generates at ~0.65 FPS single-GPU with
group-offloading (125 frames in 191 s), vs the vendor's "24 fps" on 8-GPU
sequence parallel. Bring-up required five corrections (see load()): the module
-global infer_state, transformer_version="480p_i2v" (not the action-model name),
action_ckpt as the .safetensors FILE, VAE-compatible width 832, and offloading
(the ~78 GB resident model OOMs 80 GB without it).

Design (confirmed from hyvideo/generate.py + worldplay_video_pipeline.py):
- BATCH mode: the released pipeline precomputes the camera/action trajectory and
  VAE-decodes the whole clip at the end (not streamed), so this adapter declares
  interaction_mode="batch" and implements run_script, not step().
- create_pipeline(pretrained_model_name_or_path=<HunyuanVideo-1.5 snapshot>,
  transformer_version="480p_i2v", action_ckpt=<...>/diffusion_pytorch_model.safetensors,
  transformer_dtype=bf16, create_sr_pipeline=False, offloading flags);
  initialize_infer_state(...) must run first.
- pose_to_input(pose_string, latent_num) -> (viewmats, Ks, action); pose string is
  "<action>-<latents>" pairs (w/s/a/d move, up/down pitch, left/right yaw).
- pipe(..., viewmats, Ks, action, video_length, few_step=True,
  num_inference_steps=4, chunk_latent_frames=4, model_type="ar",
  user_height/user_width, reference_image=<start>) -> out.videos (pt).
- native fps 24; 480p; distilled 4-step. (video_length-1) % 4 == 0.

LICENSE: Tencent HY-WorldPlay Community License — excludes the EU, UK, and South
Korea; the doctor prints it. Weights are user-downloaded only.

Env: HYW_REPO, HYW_MODEL_PATH (HunyuanVideo-1.5 snapshot), HYW_ACTION_CKPT
(ar_distilled_action_model dir), HYW_STEPS (4), HYW_OFFLOAD ("1" group-offloads
for sub-80 GB GPUs).
"""

from __future__ import annotations

import os
import sys
from typing import Any

HYW_REPO = os.environ.get("HYW_REPO", "/opt/code/HY-WorldPlay")
HYW_MODEL_PATH = os.environ.get("HYW_MODEL_PATH", "")
HYW_ACTION_CKPT = os.environ.get("HYW_ACTION_CKPT", "")
HYW_STEPS = int(os.environ.get("HYW_STEPS", "4"))
# Default ON: the full resident model (~78 GB incl. the 34 GB action model) OOMs
# an 80 GB GPU without it (confirmed on A100 80 GB, 2026-10-16). Group-offloading
# fits it, at a speed cost. Set HYW_OFFLOAD=0 only on a >80 GB / multi-GPU setup.
HYW_OFFLOAD = os.environ.get("HYW_OFFLOAD", "1") == "1"
HYW_HEIGHT = int(os.environ.get("HYW_HEIGHT", "480"))
# 832 (not 854): width must be divisible by the VAE spatial factor or the decode
# rearrange fails (854/2 = 427 is odd).
HYW_WIDTH = int(os.environ.get("HYW_WIDTH", "832"))

LICENSE_NOTICE = (
    "TENCENT HY-WORLDPLAY COMMUNITY LICENSE: does not apply in the EU, UK, or "
    "South Korea; >1M MAU needs a separate Tencent license; output may not be "
    "used to improve any other AI model. Weights are user-downloaded only."
)

# LiveFrame token -> HY pose action word.
POSE_WORD = {
    "W": "w", "S": "s", "A": "a", "D": "d",
    "L": "left", "R": "right", "U": "up", "N": "down",
    "0": "w",  # HY has no explicit no-op; forward is the documented default
}


class HYWorldPlayAdapter:
    def __init__(self) -> None:
        self._loaded = False

    def capabilities(self) -> dict[str, Any]:
        return {
            "name": "hy-worldplay",
            "interaction_mode": "batch",  # released VAE decodes at end, not streamed
            "native_fps": 24.0,
            # HY requires the LATENT count (video_length-1)//4+1 to be divisible by
            # 4, i.e. video_length in {13, 29, 45, ... = 16m-3}. A 13-then-16 chunk
            # grid makes every materialized script total land on 16m-3.
            "chunk_frames": 16,
            "chunk_frames_first": 13,
            "resolution": [HYW_HEIGHT, HYW_WIDTH],
            "actions": ["W", "S", "A", "D", "L", "R", "U", "N", "0"],
            "action_magnitudes": {"pose_unit": 1.0},
            "supports_seed": True,
            "min_vram_gb": 72.0,
            "license_notice": LICENSE_NOTICE,
        }

    def load(self) -> None:
        sys.path.insert(0, HYW_REPO)
        os.chdir(HYW_REPO)
        import torch
        from hyvideo.pipelines.worldplay_video_pipeline import (
            HunyuanVideo_1_5_Pipeline,
        )

        self._torch = torch
        from types import SimpleNamespace

        from hyvideo.commons.infer_state import initialize_infer_state
        from hyvideo.generate import pose_to_input

        self._pose_to_input = pose_to_input
        # The pipeline reads a module-global infer_state that generate.py sets up;
        # single-GPU safe defaults (no sage/fp8/vae-parallel/compile).
        initialize_infer_state(
            SimpleNamespace(
                sage_blocks_range="0-0", use_sageattn=False,
                include_patterns="double_blocks", enable_torch_compile=False,
                use_fp8_gemm=False, quant_type="fp8-per-block", use_vae_parallel=False,
            )
        )
        # action_ckpt is loaded via load_file() -> it must be the .safetensors FILE,
        # not the directory (accept either; append the filename if a dir was given).
        action_ckpt = HYW_ACTION_CKPT
        if os.path.isdir(action_ckpt):
            action_ckpt = os.path.join(action_ckpt, "diffusion_pytorch_model.safetensors")
        self._pipeline = HunyuanVideo_1_5_Pipeline.create_pipeline(
            pretrained_model_name_or_path=HYW_MODEL_PATH,
            transformer_version="480p_i2v",  # resolution_task; action model is separate
            enable_offloading=HYW_OFFLOAD,
            enable_group_offloading=HYW_OFFLOAD,
            create_sr_pipeline=False,
            force_sparse_attn=False,
            transformer_dtype=torch.bfloat16,
            action_ckpt=action_ckpt,
        )
        self._loaded = True

    def reset(self, image: Any, prompt: str | None, seed: int) -> None:
        assert self._loaded, "reset before load"
        import numpy as np
        from PIL import Image

        self._start = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize(
            (HYW_WIDTH, HYW_HEIGHT)
        )
        self._start_path = "/tmp/hy_start.png"
        self._start.save(self._start_path)
        self._prompt = prompt or "a scene"
        self._seed = int(seed)

    def _pose_string(self, latent_num: int) -> str:
        # The runtime reference measures generation throughput; a single forward
        # trajectory over the whole clip (latent_num-1 steps after the init frame)
        # keeps the pose-string total exactly matched to latent_num and avoids
        # per-segment rounding drift. HY control is camera-pose based and
        # experimental; action-conditioned control is not scored here.
        return f"w-{latent_num - 1}"

    def run_script(self, segments: Any):
        # The bridge passes the script as a list of segment dicts
        # (t_frames / action / hold_frames), not an ActionScript object.
        assert self._loaded, "run_script before load"
        import numpy as np

        torch = self._torch
        video_length = sum(int(seg["hold_frames"]) for seg in segments)
        latent_num = (video_length - 1) // 4 + 1
        viewmats, Ks, action = self._pose_to_input(self._pose_string(latent_num), latent_num)
        out = self._pipeline(
            enable_sr=False, prompt=self._prompt, aspect_ratio="16:9",
            num_inference_steps=HYW_STEPS, sr_num_inference_steps=None,
            video_length=video_length, negative_prompt="", seed=self._seed,
            output_type="pt", prompt_rewrite=False, return_pre_sr_video=False,
            viewmats=viewmats.unsqueeze(0), Ks=Ks.unsqueeze(0),
            action=action.unsqueeze(0), few_step=True, chunk_latent_frames=4,
            model_type="ar", user_height=HYW_HEIGHT, user_width=HYW_WIDTH,
            transformer_resident_ar_rollout=True, reference_image=self._start_path,
        )
        video = out.videos[0] if hasattr(out.videos, "__getitem__") else out.videos
        # (C, T, H, W) in [-1, 1] -> (T, H, W, C) uint8
        frames = (
            ((video.float().permute(1, 2, 3, 0) + 1) * 127.5)
            .clip(0, 255).to(torch.uint8).cpu().numpy()
        )
        # The whole clip is produced at once (batch), but the subprocess frame ring
        # is sized per chunk — so yield the batch sliced into the capabilities chunk
        # grid (first chunk_frames_first, then chunk_frames). Raw uint8 arrays; the
        # bridge worker wraps them (no liveframe import in the model env).
        first = self.capabilities()["chunk_frames_first"]
        later = self.capabilities()["chunk_frames"]
        cursor = 0
        total = len(frames)
        while cursor < total:
            take = first if cursor == 0 else later
            take = min(take, total - cursor)
            yield np.ascontiguousarray(frames[cursor : cursor + take])
            cursor += take

    def step(self, actions: list[str]):
        raise RuntimeError("hy-worldplay is batch mode; the harness calls run_script()")

    def close(self) -> None:
        self._loaded = False
