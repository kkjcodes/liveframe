"""Oasis-500M bridge adapter (runs INSIDE /opt/envs/oasis).

Drives etched-ai/open-oasis's DiT-S/2 diffusion-forcing model one autoregressive
frame per step(), mirroring generate.py exactly (ddim_steps=10 sub-steps, sliding
window <= max_frames=32), decoding each new frame as it is produced. 360x640,
native fps 20. We measure the OPEN 500M weights vanilla on an A100 — the 20 fps
claim is "Decart's inference engine", not these weights. Actions are neutral
(no-op): action content does not change per-frame compute, and we measure
throughput, not control, for this model.
"""
from __future__ import annotations

import os
import sys
from typing import Any

OASIS_REPO = os.environ.get("OASIS_REPO", "/opt/code/open-oasis")
WEIGHTS = os.environ.get("OASIS_WEIGHTS", "/opt/code/open-oasis/weights")
H, W = 360, 640
DDIM_STEPS = int(os.environ.get("OASIS_DDIM_STEPS", "10"))
ACTION_DIM = 25
MAX_NOISE = 1000
NOISE_ABS_MAX = 20
STAB = 15
SCALE = 0.07843137255


class OasisAdapter:
    def __init__(self) -> None:
        self._loaded = False

    def capabilities(self) -> dict[str, Any]:
        return {
            "name": "oasis-500m",
            "interaction_mode": "interactive",
            "native_fps": 20.0,
            "chunk_frames": 1,
            "resolution": [H, W],
            "actions": ["W", "S", "A", "D", "L", "R", "U", "N", "0"],
            "supports_seed": True,
            "license_notice": "open-oasis 500M; 20fps claim is Decart's engine, not these weights",
        }

    def load(self) -> None:
        os.chdir(OASIS_REPO)
        sys.path.insert(0, OASIS_REPO)
        import torch
        from dit import DiT_models
        from safetensors.torch import load_model
        from utils import sigmoid_beta_schedule
        from vae import VAE_models

        self._torch = torch
        self._device = "cuda:0"
        model = DiT_models["DiT-S/2"]()
        load_model(model, os.path.join(WEIGHTS, "oasis500m.safetensors"))
        self._model = model.to(self._device).eval()
        vae = VAE_models["vit-l-20-shallow-encoder"]()
        load_model(vae, os.path.join(WEIGHTS, "vit-l-20.safetensors"))
        self._vae = vae.to(self._device).eval()
        betas = sigmoid_beta_schedule(MAX_NOISE).float().to(self._device)
        alphas = 1.0 - betas
        from einops import rearrange
        self._rearrange = rearrange
        self._acp = rearrange(torch.cumprod(alphas, dim=0), "T -> T 1 1 1")
        self._noise_range = torch.linspace(-1, MAX_NOISE - 1, DDIM_STEPS + 1)
        self._loaded = True

    def reset(self, image: Any, prompt: str | None, seed: int) -> None:
        assert self._loaded, "reset before load"
        import numpy as np
        import torch
        from einops import rearrange
        from torchvision.transforms.functional import resize

        torch.manual_seed(int(seed)); torch.cuda.manual_seed(int(seed))
        img = torch.from_numpy(np.asarray(image, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        img = resize(img, (H, W)).unsqueeze(0).to(self._device)  # (1,3,H,W) in [0,1]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.half):
            enc = self._vae.encode(img * 2 - 1).mean * SCALE
        x = rearrange(enc, "(b t) (h w) c -> b t c h w", t=1,
                      h=H // self._vae.patch_size, w=W // self._vae.patch_size)
        self._x = x  # (B,1,c,h,w) latent buffer
        self._i = 1
        self._actions = torch.zeros(1, 4096, ACTION_DIM, device=self._device)  # neutral
        self._emission = 0

    def step(self, actions: list[str]):
        assert self._loaded, "step before load"
        import numpy as np
        import torch

        x = self._x; i = self._i; dev = self._device; rr = self._rearrange
        acp = self._acp; nr = self._noise_range; mf = self._model.max_frames
        chunk = torch.randn((x.shape[0], 1, *x.shape[-3:]), device=dev)
        chunk = torch.clamp(chunk, -NOISE_ABS_MAX, NOISE_ABS_MAX)
        x = torch.cat([x, chunk], dim=1)
        start = max(0, i + 1 - mf)
        for ni in reversed(range(1, DDIM_STEPS + 1)):
            t_ctx = torch.full((x.shape[0], i), STAB - 1, dtype=torch.long, device=dev)
            t = torch.full((x.shape[0], 1), nr[ni], dtype=torch.long, device=dev)
            tn = torch.full((x.shape[0], 1), nr[ni - 1], dtype=torch.long, device=dev)
            tn = torch.where(tn < 0, t, tn)
            t = torch.cat([t_ctx, t], dim=1); tn = torch.cat([t_ctx, tn], dim=1)
            xc = x.clone()[:, start:]; t = t[:, start:]; tn = tn[:, start:]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.half):
                v = self._model(xc, t, self._actions[:, start:i + 1])
            x_start = acp[t].sqrt() * xc - (1 - acp[t]).sqrt() * v
            x_noise = ((1 / acp[t]).sqrt() * xc - x_start) / (1 / acp[t] - 1).sqrt()
            an = acp[tn]; an[:, :-1] = torch.ones_like(an[:, :-1])
            if ni == 1:
                an[:, -1:] = torch.ones_like(an[:, -1:])
            x_pred = an.sqrt() * x_start + x_noise * (1 - an).sqrt()
            x[:, -1:] = x_pred[:, -1:]
        # decode just the new frame
        newlat = rr(x[:, -1:], "b t c h w -> (b t) (h w) c")
        with torch.no_grad():
            dec = (self._vae.decode(newlat / SCALE) + 1) / 2
        frame = (torch.clamp(dec, 0, 1) * 255).byte()  # (1,3,H,W)
        arr = frame[0].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)[None]  # (1,H,W,3)
        self._x = x; self._i += 1; self._emission += 1
        yield (np.ascontiguousarray(arr), 1)

    def run_script(self, script: Any):
        raise RuntimeError("oasis-500m is interactive; the harness calls step()")

    def close(self) -> None:
        self._loaded = False
