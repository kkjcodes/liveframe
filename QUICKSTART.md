# LiveFrame World — Quickstart

Point the harness at an action-conditioned world model and get a verified
runtime + stability report — one command, one GPU, minutes. Every step below
was run against the published `0.3.0` package in a clean virtualenv (nothing
from the source tree on the path), so it is exactly what a `pip install` user
experiences.

---

## 1. Install

LiveFrame's core is dependency-free; the world harness needs the `[world]`
extra (numpy, OpenCV, jsonschema).

```bash
python -m venv venv && source venv/bin/activate
pip install "liveframe[world]"
```

From a source checkout instead:

```bash
pip install -e ".[world]"
```

Confirm it landed:

```bash
liveframe version          # -> 0.3.0
liveframe world --help      # run | verify | fetch | doctor | suites | compare | export
```

---

## 2. Prove the loop on CPU first (ships in the package)

The `synthetic` backend is built in — no model, no GPU — so you can prove the
whole pipeline in seconds before touching real weights:

```bash
liveframe world run synthetic --suite smoke
```

It prints the paths it wrote. Every run produces three things in
`runs/<model>/<suite>/<timestamp>/`:

| File | What it is |
|---|---|
| `report.html` | Self-contained page — no scripts, no network. Open it directly. |
| `report.json` | Machine-readable headline + per-case metrics. |
| `ledger.json` | Digest-bound manifest; re-verifiable byte for byte (step 6). |

See the suites and their content-addressed digests:

```bash
liveframe world suites
# smoke(1 case) quick(8) standard(24) soak(1x300s) memory(8)
```

`smoke` and `quick` are deliberately short, so the real-time verdict reports
`null` — the real-time rule needs a continuous 60-second stream (`soak`).

---

## 3. Bring your own model

Any `FILE.py:ClassName` implementing the adapter protocol works, run
in-process. Save this as `my_world.py` — it is a complete, runnable stand-in
(it pans the start image per action). Replace the body with your model; keep
the five methods and the declared capabilities.

```python
# my_world.py
from __future__ import annotations

import time
from collections.abc import Iterator, Sequence

import numpy as np

from liveframe.engine import NativeFrame
from liveframe.world.backend import WorldBackendError, WorldCapabilities


class MyWorld:
    def __init__(self, *, height: int = 144, width: int = 256) -> None:
        self._h, self._w = height, width
        self._canvas = None
        self._pos = np.zeros(2)
        self._emission = 0
        self._loaded = False

    @property
    def capabilities(self) -> WorldCapabilities:
        return WorldCapabilities(
            name="my-world",
            interaction_mode="interactive",   # step() is called per chunk
            native_fps=16.0,                  # the rate it is DESIGNED to be viewed at
            chunk_frames=4,                   # pixel frames emitted per step()
            resolution=(self._h, self._w),    # exact; any other shape fails the case
            # Declare EVERY action the suites command, or the case fails with a
            # clear "action token 'L' is outside the declared capabilities" error:
            actions=frozenset({"W", "S", "A", "D", "L", "R", "U", "N", "0"}),
            action_magnitudes={"W": 3.0, "S": 3.0, "A": 3.0, "D": 3.0},
            supports_seed=True,
        )

    def load(self) -> None:
        # Load weights here. Cold start is reported as context, never blended
        # into a case's generation time.
        self._loaded = True

    def reset(self, image: np.ndarray, prompt: str | None, seed: int) -> None:
        if not self._loaded:
            raise WorldBackendError("reset() before load()")
        self._canvas = np.tile(np.asarray(image, dtype=np.uint8), (3, 3, 1))
        self._pos = np.array([float(self._h), float(self._w)])
        self._emission = 0
        self._rng = np.random.default_rng(seed)

    def step(self, actions: Sequence[str]) -> Iterator[NativeFrame]:
        if self._canvas is None:
            raise WorldBackendError("step() before reset()")
        delta = {"W": (-3.0, 0.0), "S": (3.0, 0.0),
                 "A": (0.0, -3.0), "D": (0.0, 3.0),
                 "L": (0.0, -1.0), "R": (0.0, 1.0),
                 "U": (-1.0, 0.0), "N": (1.0, 0.0), "0": (0.0, 0.0)}
        frames = []
        for _ in range(self.capabilities.chunk_frames):
            for token in actions:
                self._pos += delta[token]
            self._pos[0] = float(np.clip(self._pos[0], 0, self._h * 2))
            self._pos[1] = float(np.clip(self._pos[1], 0, self._w * 2))
            r, c = int(self._pos[0]), int(self._pos[1])
            frames.append(np.ascontiguousarray(self._canvas[r:r + self._h, c:c + self._w]))
        # Emit one NativeFrame per step — never buffer and dump at the end; the
        # harness measures the cadence your users would actually see.
        yield NativeFrame(
            index=self._emission,
            payload=np.stack(frames),               # (chunk_frames, H, W, 3) uint8
            native_frame_count=len(frames),
            metadata={"worker_ts": time.perf_counter()},
        )
        self._emission += 1

    def run_script(self, script: object) -> Iterator[NativeFrame]:
        # Batch-mode models implement this instead; interactive ones raise.
        raise WorldBackendError("my-world is interactive; the harness calls step()")

    def close(self) -> None:
        self._canvas = None
```

Run it:

```bash
liveframe world run my_world.py:MyWorld --suite smoke     # 1 case, fast sanity
liveframe world run my_world.py:MyWorld --suite quick      # 8 cases
```

The run directory is slugged from the **filename** (`my_world`), not the
declared `name`, so look in `runs/my_world/quick/<timestamp>/`.

> **Three rules the harness enforces per case (it fails the case, not the run):**
> 1. `capabilities` must tell the truth — `resolution` is exact, and every
>    commanded action must be declared.
> 2. `step` must emit real work as it happens — no buffering frames and
>    releasing them at the end.
> 3. Never interpolate or duplicate frames to look faster — only native frames
>    are counted, and the frame-accounting check fails the case if the count
>    doesn't match the script.

---

## 4. Read the report

Open `runs/my_world/quick/<timestamp>/report.html` in a browser (it is fully
self-contained). Or pull the headline from JSON:

```bash
python - <<'PY'
import json, glob
run = sorted(glob.glob("runs/my_world/quick/*/"))[-1]
h = json.load(open(run + "report.json"))["headline"]
print(f"{h['cases_ok']}/{h['cases_total']} ok, {h['cases_failed']} failed; "
      f"fps median {h['generated_fps_median']}")
PY
```

If a case failed, `report.json` → `cases[].error` says exactly why in plain
language (e.g. an undeclared action token, or a frame whose shape didn't match
the declared resolution).

---

## 5. Run inside your model's own environment

Your model usually needs its own torch/CUDA. Keep the harness in its own
process and run the SAME adapter file inside the model's interpreter — the
harness talks to it over a subprocess bridge:

```bash
liveframe world run my_world.py:MyWorld --suite quick \
    --python /path/to/model-env/bin/python
```

---

## 6. Verify the evidence (and prove it's tamper-evident)

Re-check every artifact digest in a finished run:

```bash
liveframe world verify runs/my_world/quick/<timestamp>/ledger.json
# -> "verified": true   (exit code 0)
```

The ledger is digest-bound: change a single character in it and verification
fails with a nonzero exit code (`world ledger self-digest mismatch`, exit 2).
That is what lets someone else trust a report they didn't produce.

---

## 7. Reference adapters (optional)

Three real models ship as named adapters. They need the vendor's weights and
checkout, so preflight first — every failed check prints its exact fix:

```bash
liveframe world doctor matrix-game-2 --python /opt/envs/mg2/bin/python
liveframe world fetch matrix-game-2            # wraps `hf download`
liveframe world run  matrix-game-2 --suite quick --python /opt/envs/mg2/bin/python
```

Also useful:

- `liveframe world compare A B` — side-by-side HTML; configuration differences
  are listed **before** any numbers, so a 25-fps-on-8-GPUs claim is never shown
  next to a 1-GPU number without the caveat.
- `liveframe world export --format worldmark|wbench|raw` — hand frames + action
  logs to an external quality scorer.

---

## What the numbers mean

Generated FPS counts decoded native frames only; interpolated or duplicated
frames never count. Cold start is reported as context, not blended into
generation time. Vendor claims get a neutral verdict — `different
configuration` when the claim names a setup (different GPU count, step count,
resolution), else `config not stated`. The harness never says "reproduced",
and it times frames as they are delivered, so it errs low rather than
inflating.
