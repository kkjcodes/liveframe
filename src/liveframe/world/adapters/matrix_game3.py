"""Matrix-Game 3.0 bridge adapter (runs INSIDE the model environment).

Validated against the live model's source (Matrix-Game-3/) on 2026-10-15.
Drives the upstream interactive pipeline's internal rollout: generate() loops
num_iterations blocks, calling a module-level get_current_action() (blocking
stdin) per block and process_video() on each block's decoded pixels. This
adapter monkeypatches get_current_action (queue-fed by step()) and process_video
(captures each block's frames) and runs generate() on a worker thread, yielding
one block of native frames per step() call. 57 pixel frames on block 0, 40
after (causal temporal stride 4); native fps 17. No interpolation.

Env: MG3_REPO, MG3_WEIGHTS, MG3_BLOCKS (num_iterations / horizon), MG3_SIZE
("H*W"), MG3_INT8 ("1" default), MG3_STEPS, MG3_VAE.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
from types import SimpleNamespace
from typing import Any

MG3_REPO = os.environ.get("MG3_REPO", "/opt/code/Matrix-Game/Matrix-Game-3")
MG3_WEIGHTS = os.environ.get("MG3_WEIGHTS", "/workspace/models/Matrix-Game-3.0")
MG3_BLOCKS = int(os.environ.get("MG3_BLOCKS", "4"))
MG3_SIZE = os.environ.get("MG3_SIZE", "704*1280")
MG3_INT8 = os.environ.get("MG3_INT8", "1") == "1"
MG3_STEPS = int(os.environ.get("MG3_STEPS", "3"))
MG3_VAE = os.environ.get("MG3_VAE", "mg_lightvae_v2")

FIRST_BLOCK_FRAMES = 57
LATER_BLOCK_FRAMES = 40
CAM = 0.1
# LiveFrame token -> MG3 interactive keyboard 6-one-hot (w/s/a/d, q=idle).
KEYBOARD = {
    "W": [1, 0, 0, 0, 0, 0], "S": [0, 1, 0, 0, 0, 0],
    "A": [0, 0, 1, 0, 0, 0], "D": [0, 0, 0, 1, 0, 0],
}
# LiveFrame token -> MG3 mouse [pitch, yaw].
MOUSE = {"U": [CAM, 0.0], "N": [-CAM, 0.0], "L": [0.0, -CAM], "R": [0.0, CAM]}

_SENTINEL = object()


class MatrixGame3Adapter:
    def __init__(self) -> None:
        self._loaded = False
        self._height, self._width = (int(x) for x in MG3_SIZE.split("*"))

    def capabilities(self) -> dict[str, Any]:
        return {
            "name": "matrix-game-3",
            "interaction_mode": "interactive",
            "native_fps": 17.0,
            "chunk_frames": LATER_BLOCK_FRAMES,
            "chunk_frames_first": FIRST_BLOCK_FRAMES,
            "resolution": [self._height, self._width],
            "actions": ["W", "S", "A", "D", "L", "R", "U", "N", "0"],
            "action_magnitudes": {"keyboard_one_hot": 1.0, "mouse_pitch_yaw_value": CAM},
            "supports_seed": True,
            "min_vram_gb": None,
            "license_notice": "Apache-2.0 (Skywork/Matrix-Game-3.0)",
        }

    def load(self) -> None:
        sys.path.insert(0, MG3_REPO)
        os.chdir(MG3_REPO)
        import torch
        from pipeline import inference_interactive_pipeline as interactive
        from wan.configs import WAN_CONFIGS

        self._torch = torch
        self._module = interactive
        self._args = SimpleNamespace(
            size=MG3_SIZE, ckpt_dir=MG3_WEIGHTS, num_iterations=MG3_BLOCKS,
            num_inference_steps=MG3_STEPS, sample_shift=5.0, sample_guide_scale=5.0,
            save_name="lf", output_dir="/tmp/mg3_out", vae_type=MG3_VAE,
            lightvae_pruning_rate=None, use_async_vae=False, compile_vae=False,
            use_int8=MG3_INT8, verify_quant=False, fa_version="0",
            interactive=True, use_base_model=False, ulysses_size=1,
        )
        os.makedirs(self._args.output_dir, exist_ok=True)
        cfg = WAN_CONFIGS["matrix_game3"]
        self._pipeline = interactive.MatrixGame3Pipeline(
            cfg, MG3_WEIGHTS, device_id=0, rank=0, use_sp=False,
            args=self._args, fa_version="0", use_base_model=False,
        )
        self._loaded = True

    def reset(self, image: Any, prompt: str | None, seed: int) -> None:
        assert self._loaded, "reset before load"
        import numpy as np
        from PIL import Image

        self._pil = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize(
            (self._width, self._height)
        )
        self._prompt = prompt or ""
        self._seed = int(seed)
        self._action_q: queue.Queue = queue.Queue()
        self._frame_q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._blocks_emitted = 0

    def _patched_get_action(self) -> dict:
        tokens = self._action_q.get()
        if tokens is _SENTINEL:
            raise _StopGenerate()
        keyboard = [0, 0, 0, 0, 0, 0]
        mouse = [0.0, 0.0]
        for token in tokens:
            if token in KEYBOARD:
                keyboard = KEYBOARD[token]
            elif token in MOUSE:
                mouse = MOUSE[token]
        return {
            "keyboard": self._torch.tensor(keyboard),
            "mouse": self._torch.tensor(mouse),
        }

    def _patched_process_video(self, input_video, *a, **k) -> None:
        # input_video is (T, H, W, C) uint8 for this block. Queue it, then after
        # the last block the harness asked for, abort generate() BEFORE its
        # post-loop processing (which otherwise crashes the worker and breaks
        # re-entrancy across cases). The frames are already queued, so step()
        # still delivers them.
        self._frame_q.put(input_video)
        self._blocks_emitted += 1
        if self._blocks_emitted >= MG3_BLOCKS:
            raise _StopGenerate()

    def _run_generate(self) -> None:
        try:
            self._pipeline.generate(
                self._prompt, self._pil, max_area=self._height * self._width,
                shift=self._args.sample_shift,
                num_inference_steps=self._args.num_inference_steps,
                guide_scale=self._args.sample_guide_scale, seed=self._seed,
                use_base_model=False, args=self._args,
            )
        except _StopGenerate:
            pass
        except BaseException:  # surfaced to step()
            import traceback
            self._error = traceback.format_exc()
            self._frame_q.put(_SENTINEL)

    def step(self, actions: list[str]):
        assert self._loaded, "step before load"
        import numpy as np

        if self._thread is None:
            self._module.get_current_action = self._patched_get_action
            self._module.process_video = self._patched_process_video
            self._thread = threading.Thread(target=self._run_generate, daemon=True)
            self._thread.start()
        self._action_q.put(list(actions))
        frames = self._frame_q.get()
        if frames is _SENTINEL:
            raise RuntimeError(f"matrix-game-3 generate failed: {self._error}")
        yield np.ascontiguousarray(frames)

    def run_script(self, script: Any):
        raise RuntimeError("matrix-game-3 is interactive; the harness calls step()")

    def close(self) -> None:
        thread = getattr(self, "_thread", None)
        if thread is not None:
            self._action_q.put(_SENTINEL)
            while not self._frame_q.empty():
                try:
                    self._frame_q.get_nowait()
                except Exception:
                    break
            thread.join(timeout=30)
            self._thread = None
        self._loaded = False


class _StopGenerate(Exception):
    pass
