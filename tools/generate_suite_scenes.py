"""Generate the procedural start-image scenes for the world suites.

Every scene is a deterministic function of its name (seeded NumPy only), so CI
can regenerate them and compare hashes against tests/fixtures/MANIFEST.json
(fictional-data enforcement). Scenes are stored as .npy (byte-deterministic
across platforms, unlike PNG encoders). No real images, no people.

Scene tags (spec section 9): first-person outdoor, indoor, stylized, and
third-person-like compositions, all procedural.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

HEIGHT, WIDTH = 352, 640

SCENES: dict[str, dict[str, object]] = {
    "outdoor-meadow": {"seed": 101, "tag": "outdoor", "horizon": 0.55, "boxes": 4},
    "outdoor-dunes": {"seed": 102, "tag": "outdoor", "horizon": 0.6, "boxes": 2},
    "indoor-hall": {"seed": 201, "tag": "indoor", "horizon": 0.4, "boxes": 7},
    "indoor-depot": {"seed": 202, "tag": "indoor", "horizon": 0.45, "boxes": 9},
    "stylized-blocks": {"seed": 301, "tag": "stylized", "horizon": 0.5, "boxes": 12},
    "stylized-night": {"seed": 302, "tag": "stylized", "horizon": 0.5, "boxes": 6},
    "thirdperson-court": {"seed": 401, "tag": "third-person", "horizon": 0.62, "boxes": 5},
    "thirdperson-yard": {"seed": 402, "tag": "third-person", "horizon": 0.58, "boxes": 8},
}


def render_scene(name: str) -> np.ndarray:
    spec = SCENES[name]
    rng = np.random.default_rng(int(spec["seed"]))
    horizon = int(HEIGHT * float(spec["horizon"]))
    ys = np.linspace(0.0, 1.0, HEIGHT)[:, None]
    xs = np.linspace(0.0, 1.0, WIDTH)[None, :]

    sky_top = rng.integers(60, 200, size=3)
    sky_bottom = rng.integers(120, 240, size=3)
    ground_a = rng.integers(40, 160, size=3)
    ground_b = rng.integers(60, 200, size=3)

    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    sky_mix = np.clip(ys / max(spec["horizon"], 1e-6), 0, 1)
    for channel in range(3):
        frame[:horizon, :, channel] = (
            sky_top[channel] * (1 - sky_mix[:horizon]) + sky_bottom[channel] * sky_mix[:horizon]
        )
    checker = ((np.floor(xs * 16) + np.floor((ys - float(spec["horizon"])) * 24)) % 2).astype(
        bool
    )
    for channel in range(3):
        ground = np.where(checker, ground_a[channel], ground_b[channel])
        frame[horizon:, :, channel] = ground[horizon:, :]

    for _ in range(int(spec["boxes"])):
        box_w = int(rng.integers(WIDTH // 16, WIDTH // 5))
        box_h = int(rng.integers(HEIGHT // 12, HEIGHT // 4))
        x0 = int(rng.integers(0, WIDTH - box_w))
        y0 = int(rng.integers(horizon - box_h // 2, HEIGHT - box_h))
        color = rng.integers(30, 250, size=3)
        shade = np.linspace(1.0, 0.72, box_h)[:, None]
        for channel in range(3):
            frame[y0 : y0 + box_h, x0 : x0 + box_w, channel] = color[channel] * shade
    noise = rng.integers(0, 14, size=frame.shape)
    return np.clip(frame + noise, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "src" / "liveframe" / "world" / "suites" / "scenes",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in SCENES:
        np.save(args.output_dir / f"{name}.npy", render_scene(name))
        print(f"wrote {name}.npy")


if __name__ == "__main__":
    main()
