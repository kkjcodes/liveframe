"""Matrix-Game 2.0 bridge adapter (runs INSIDE the model environment).

Standalone by design: no liveframe imports; the bridge worker copies this file
onto PYTHONPATH in the model env. It imports the user's Matrix-Game checkout
(never vendored) and drives the streaming components block by block.

Upstream provenance (record per spec 6.5):
- repo: https://github.com/SkyworkAI/Matrix-Game, directory ``Matrix-Game-2``
  (cloned at depth 1; exact commit recorded by doctor/fetch at run time).
- decoded frames come from ``demo_utils.vae_block3.VAEDecoderWrapper`` called
  per 3-latent block with a rolling VAE cache, exactly as
  ``CausalInferenceStreamingPipeline.inference`` does (causal_inference.py,
  "Step 3.4").
- actions enter per block: upstream's loop calls the module-level
  ``get_current_action`` (a blocking ``input()``) and splices the result into
  full-length keyboard/mouse tensors via ``cond_current``. TRUE stepwise
  interaction is therefore supported; this adapter owns the identical block
  loop (denoise steps -> clean-context KV rerun -> per-block VAE decode) and
  writes each chunk's action span into the same tensors before denoising it,
  instead of reading stdin. Upstream's per-block ``process_video`` writes and
  ``input()`` prompts are omitted (output capture replaces them); numerics of
  the sampled path are unchanged.
- chunking: ``num_frame_per_block = 3`` latents; the causal VAE emits
  1 + 4*(3-1) = 9 pixel frames for the first block and 12 thereafter.
- frame basis: generated FPS counts decoded PIXEL frames in host memory. The
  causal VAE expands L latents to 4L-3 pixel frames: the first 3-latent block
  decodes to 1+4*2 = 9 frames, later blocks to 12 (causal_inference.py
  "Step 3.4" decode and the cond_current frame arithmetic, lines 76-94). No
  interpolation exists anywhere in the path. The declared native_fps of 12.0
  is upstream's own export rate (utils/visualize.py:165, ``fps = 12``). The
  vendor's "real-time [25fps]" claim states neither a frame basis nor
  hardware; claim-check verdicts for non-vendor hardware are reported as
  "config not stated", never as pass/fail.

Environment variables:
  MG2_REPO     path to the Matrix-Game checkout's Matrix-Game-2 directory
  MG2_WEIGHTS  path to the downloaded Skywork/Matrix-Game-2.0 directory
  MG2_LATENTS  latent-frame horizon (multiple of 3; pixel frames = 4*L-3)
  MG2_MODE     'universal' (default); templerun is out of scope for M2
"""

from __future__ import annotations

import os
import sys
from typing import Any

MG2_REPO = os.environ.get("MG2_REPO", "/opt/code/Matrix-Game/Matrix-Game-2")
MG2_WEIGHTS = os.environ.get("MG2_WEIGHTS", "/workspace/models/Matrix-Game-2.0")
MG2_LATENTS = int(os.environ.get("MG2_LATENTS", "30"))
MG2_MODE = os.environ.get("MG2_MODE", "universal")

CAM_VALUE = 0.1
KEYBOARD_COLUMNS = {"W": 0, "S": 1, "A": 2, "D": 3}
MOUSE_VALUES = {
    "L": (0.0, -CAM_VALUE),
    "R": (0.0, CAM_VALUE),
    "U": (CAM_VALUE, 0.0),
    "N": (-CAM_VALUE, 0.0),
}


class MatrixGame2Adapter:
    def __init__(self) -> None:
        if MG2_LATENTS % 3 != 0:
            raise ValueError("MG2_LATENTS must be a multiple of num_frame_per_block=3")
        self._latents = MG2_LATENTS
        self._pixel_frames = 4 * (self._latents - 1) + 1
        self._block = 0
        self._loaded = False

    def capabilities(self) -> dict[str, Any]:
        return {
            "name": "matrix-game-2",
            "interaction_mode": "interactive",
            "native_fps": 12.0,
            "chunk_frames": 12,
            "chunk_frames_first": 9,
            "resolution": [352, 640],
            "actions": ["W", "S", "A", "D", "L", "R", "U", "N", "0"],
            "action_magnitudes": {
                "keyboard_one_hot": 1.0,
                "mouse_pitch_yaw_value": CAM_VALUE,
            },
            "supports_seed": True,
            "min_vram_gb": 24.0,
            "license_notice": "MIT (SkyworkAI/Matrix-Game)",
        }

    # -- lifecycle -------------------------------------------------------------

    def load(self) -> None:
        sys.path.insert(0, MG2_REPO)
        os.chdir(MG2_REPO)  # upstream model_config paths are repo-relative
        import torch
        from demo_utils.vae_block3 import VAEDecoderWrapper
        from omegaconf import OmegaConf
        from pipeline import CausalInferenceStreamingPipeline
        from safetensors.torch import load_file
        from utils.wan_wrapper import WanDiffusionWrapper
        from wan.vae.wanx_vae import get_wanx_vae_wrapper

        self._torch = torch
        self._device = torch.device("cuda")
        self._dtype = torch.bfloat16
        config = OmegaConf.load(
            os.path.join(MG2_REPO, "configs/inference_yaml/inference_universal.yaml")
        )
        config.pop("mode", None)
        generator = WanDiffusionWrapper(
            **getattr(config, "model_kwargs", {}), is_causal=True
        )
        vae_decoder = VAEDecoderWrapper()
        vae_state = torch.load(
            os.path.join(MG2_WEIGHTS, "Wan2.1_VAE.pth"), map_location="cpu"
        )
        vae_decoder.load_state_dict(
            {
                key: value
                for key, value in vae_state.items()
                if "decoder." in key or "conv2" in key
            }
        )
        vae_decoder.to(self._device, torch.float16)
        vae_decoder.requires_grad_(False)
        vae_decoder.eval()
        pipeline = CausalInferenceStreamingPipeline(
            config, generator=generator, vae_decoder=vae_decoder
        )
        checkpoint = os.path.join(MG2_WEIGHTS, "base_distilled_model", "base_distill.safetensors")
        pipeline.generator.load_state_dict(load_file(checkpoint))
        self._pipeline = pipeline.to(device=self._device, dtype=self._dtype)
        self._pipeline.vae_decoder.to(torch.float16)
        vae = get_wanx_vae_wrapper(MG2_WEIGHTS, torch.float16)
        vae.requires_grad_(False)
        vae.eval()
        self._vae = vae.to(self._device, self._dtype)
        from torchvision.transforms import v2

        self._frame_process = v2.Compose(
            [
                v2.Resize(size=(352, 640), antialias=True),
                v2.ToTensor(),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self._loaded = True

    def reset(self, image: Any, prompt: str | None, seed: int) -> None:
        del prompt  # image-conditioned model; no text prompt path in upstream
        assert self._loaded, "reset before load"
        torch = self._torch
        from PIL import Image
        from pipeline.causal_inference import ZERO_VAE_CACHE
        from utils.misc import set_seed

        set_seed(int(seed))
        import copy

        import numpy as np

        array = np.asarray(image, dtype=np.uint8)
        pil = Image.fromarray(array)
        pil = self._resizecrop(pil, 352, 640)
        frame = (
            self._frame_process(pil)[None, :, None, :, :]
            .to(dtype=self._dtype, device=self._device)
        )
        padding = torch.zeros_like(frame).repeat(1, 1, 4 * (self._latents - 1), 1, 1)
        img_cond = torch.concat([frame, padding], dim=2)
        tiler = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
        img_cond = self._vae.encode(img_cond, device=self._device, **tiler).to(self._device)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        self._cond = {
            "cond_concat": torch.cat([mask_cond[:, :4], img_cond], dim=1).to(
                device=self._device, dtype=self._dtype
            ),
            "visual_context": self._vae.clip.encode_video(frame).to(
                device=self._device, dtype=self._dtype
            ),
            "keyboard_cond": torch.zeros(
                [1, self._pixel_frames, 4], device=self._device, dtype=self._dtype
            ),
            "mouse_cond": torch.zeros(
                [1, self._pixel_frames, 2], device=self._device, dtype=self._dtype
            ),
        }
        self._noise = torch.randn(
            [1, 16, self._latents, 44, 80], device=self._device, dtype=self._dtype
        )
        pipe = self._pipeline
        pipe._initialize_kv_cache(1, self._dtype, self._device)
        pipe._initialize_kv_cache_mouse_and_keyboard(1, self._dtype, self._device)
        pipe._initialize_crossattn_cache(1, self._dtype, self._device)
        self._vae_cache = copy.deepcopy(ZERO_VAE_CACHE)
        for index in range(len(self._vae_cache)):
            self._vae_cache[index] = None
        self._block = 0
        self._start_latent = 0

    @staticmethod
    def _resizecrop(image: Any, th: int, tw: int) -> Any:
        w, h = image.size
        if h / w > th / tw:
            new_w = int(w)
            new_h = int(new_w * th / tw)
        else:
            new_h = int(h)
            new_w = int(new_h * tw / th)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        return image.crop((left, top, left + new_w, top + new_h))

    # -- stepping ---------------------------------------------------------------

    def _write_action_span(self, tokens: list[str]) -> None:
        """Fill this block's frame span of the full-length condition tensors.

        Mirrors upstream ``cond_current(..., replace=...)`` arithmetic: block 0
        covers pixel frames [0, 1 + 4*(nfpb-1)), block b >= 1 covers the next
        4*nfpb frames, ending at 1 + 4*(start + nfpb - 1).
        """

        nfpb = self._pipeline.num_frame_per_block
        start_latent = self._start_latent
        final_frame = 1 + 4 * (start_latent + nfpb - 1)
        span = (1 + 4 * (nfpb - 1)) if start_latent == 0 else 4 * nfpb
        keyboard = [0.0, 0.0, 0.0, 0.0]
        pitch = yaw = 0.0
        for token in tokens:
            if token in KEYBOARD_COLUMNS:
                keyboard[KEYBOARD_COLUMNS[token]] = 1.0
            elif token in MOUSE_VALUES:
                pitch, yaw = MOUSE_VALUES[token]
            elif token != "0":
                raise ValueError(f"unsupported Matrix-Game-2 token: {token}")
        torch = self._torch
        self._cond["keyboard_cond"][:, final_frame - span : final_frame] = torch.tensor(
            keyboard, device=self._device, dtype=self._dtype
        )
        self._cond["mouse_cond"][:, final_frame - span : final_frame] = torch.tensor(
            [pitch, yaw], device=self._device, dtype=self._dtype
        )

    def step(self, actions: list[str]):
        assert self._loaded, "step before load"
        torch = self._torch
        from einops import rearrange
        from pipeline.causal_inference import cond_current

        pipe = self._pipeline
        nfpb = pipe.num_frame_per_block
        if self._start_latent + nfpb > self._latents:
            raise RuntimeError(
                f"horizon exhausted: MG2_LATENTS={self._latents} supports "
                f"{self._latents // nfpb} chunks"
            )
        self._write_action_span(list(actions))
        start = self._start_latent
        with torch.no_grad():
            noisy_input = self._noise[:, :, start : start + nfpb]
            new_cond = cond_current(self._cond, start, nfpb)
            for index, current_timestep in enumerate(pipe.denoising_step_list):
                timestep = (
                    torch.ones([1, nfpb], device=self._device, dtype=torch.int64)
                    * current_timestep
                )
                if index < len(pipe.denoising_step_list) - 1:
                    _, denoised = pipe.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_cond,
                        timestep=timestep,
                        kv_cache=pipe.kv_cache1,
                        kv_cache_mouse=pipe.kv_cache_mouse,
                        kv_cache_keyboard=pipe.kv_cache_keyboard,
                        crossattn_cache=pipe.crossattn_cache,
                        current_start=start * pipe.frame_seq_length,
                    )
                    next_timestep = pipe.denoising_step_list[index + 1]
                    noisy_input = pipe.scheduler.add_noise(
                        rearrange(denoised, "b c f h w -> (b f) c h w"),
                        torch.randn_like(
                            rearrange(denoised, "b c f h w -> (b f) c h w")
                        ),
                        next_timestep
                        * torch.ones([nfpb], device=self._device, dtype=torch.long),
                    )
                    noisy_input = rearrange(
                        noisy_input, "(b f) c h w -> b c f h w", b=denoised.shape[0]
                    )
                else:
                    _, denoised = pipe.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=new_cond,
                        timestep=timestep,
                        kv_cache=pipe.kv_cache1,
                        kv_cache_mouse=pipe.kv_cache_mouse,
                        kv_cache_keyboard=pipe.kv_cache_keyboard,
                        crossattn_cache=pipe.crossattn_cache,
                        current_start=start * pipe.frame_seq_length,
                    )
            context_timestep = (
                torch.ones([1, nfpb], device=self._device, dtype=torch.int64)
                * pipe.args.context_noise
            )
            pipe.generator(
                noisy_image_or_video=denoised,
                conditional_dict=new_cond,
                timestep=context_timestep,
                kv_cache=pipe.kv_cache1,
                kv_cache_mouse=pipe.kv_cache_mouse,
                kv_cache_keyboard=pipe.kv_cache_keyboard,
                crossattn_cache=pipe.crossattn_cache,
                current_start=start * pipe.frame_seq_length,
            )
            video, vae_cache = pipe.vae_decoder(
                denoised.transpose(1, 2).half(), *self._vae_cache
            )
            self._vae_cache = vae_cache
            torch.cuda.synchronize()
            frames = (
                ((video.float() + 1) * 127.5)
                .clip(0, 255)
                .to(torch.uint8)[0]
                .permute(0, 2, 3, 1)
                .contiguous()
                .cpu()
                .numpy()
            )
        self._start_latent += nfpb
        self._block += 1
        yield frames

    def close(self) -> None:
        self._loaded = False
