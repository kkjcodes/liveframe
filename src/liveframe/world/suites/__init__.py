"""Suite manifests: scenes x seeds x action patterns, resolved per model.

A suite case stores its motion as a PATTERN (action tokens with fractional
weights) plus a duration in seconds; materialization converts it to a
frame-exact :class:`ActionScript` aligned to the target model's chunk grid
(first-chunk size included), so one suite definition serves models with
different native rates and chunking. The materialized script digest is part
of the case key, so resume detection is exact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from liveframe.gates import contract_digest
from liveframe.world.actions import ActionScript
from liveframe.world.backend import WorldCapabilities

SUITE_NAMES = ("smoke", "quick", "standard", "soak", "memory")

# Suite definitions (spec section 9). Patterns are (action, weight) rows.
_MIXED = (("W", 2), ("L", 1), ("0", 1))
_REVERSAL = (("W", 2), ("S", 2), ("L", 1), ("R", 1))
_HOLDS = (("0", 1), ("W", 1), ("0", 1), ("D", 1))
_LOOP_YAW = (("L", 4),)  # closed-loop intent; closure frame depends on magnitudes
_LOOP_RETURN = (("W", 1), ("S", 1))

# Canonical scene order: exports index scenes by THIS tuple so image indices
# stay aligned across models and runs (never by what happens to be present).
SCENE_NAMES = (
    "outdoor-meadow",
    "outdoor-dunes",
    "indoor-hall",
    "indoor-depot",
    "stylized-blocks",
    "stylized-night",
    "thirdperson-court",
    "thirdperson-yard",
)
_SCENES_ALL = SCENE_NAMES

SUITES: dict[str, dict[str, Any]] = {
    "smoke": {
        "duration_seconds": 10.0,
        "cases": [("outdoor-meadow", 0, _MIXED)],
    },
    "quick": {
        "duration_seconds": 20.0,
        "cases": [
            (scene, seed, _MIXED)
            for scene in _SCENES_ALL[:4]
            for seed in (0, 1)
        ],
    },
    "standard": {
        "duration_seconds": 60.0,
        "cases": [
            (scene, seed, pattern)
            for scene in _SCENES_ALL
            for seed, pattern in ((0, _MIXED), (1, _REVERSAL), (2, _HOLDS))
        ],
    },
    "soak": {
        "duration_seconds": 300.0,
        "cases": [("outdoor-meadow", 0, _MIXED)],
    },
    "memory": {
        "duration_seconds": 30.0,
        "cases": [
            (scene, seed, pattern)
            for scene in _SCENES_ALL[:4]
            for seed, pattern in ((0, _LOOP_YAW), (1, _LOOP_RETURN))
        ],
    },
}


def load_scene(name: str) -> np.ndarray:
    package = resources.files("liveframe.world.suites") / "scenes" / f"{name}.npy"
    with resources.as_file(package) as path:
        scene: np.ndarray = np.load(path)
    return scene


def fit_scene(scene: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    """Resize a canonical scene to the backend's native resolution.

    Nearest-neighbor via index maps: deterministic, dependency-free, and
    identical across platforms (unlike interpolating resizers).
    """

    height, width = resolution
    if scene.shape[:2] == (height, width):
        return scene
    rows = np.linspace(0, scene.shape[0] - 1, height).round().astype(int)
    cols = np.linspace(0, scene.shape[1] - 1, width).round().astype(int)
    fitted: np.ndarray = scene[rows][:, cols]
    return fitted


def pattern_script(
    pattern: tuple[tuple[str, int], ...],
    *,
    duration_seconds: float,
    capabilities: WorldCapabilities,
) -> ActionScript:
    """Materialize a fractional pattern into a chunk-grid-exact ActionScript."""

    first = capabilities.chunk_frames_first or capabilities.chunk_frames
    total = first
    while total < duration_seconds * capabilities.native_fps:
        total += capabilities.chunk_frames
    weights = sum(weight for _, weight in pattern)
    segments: list[tuple[int, str, int]] = []
    cursor = 0
    for index, (action, weight) in enumerate(pattern):
        hold = (
            total - cursor
            if index == len(pattern) - 1
            else max(1, round(total * weight / weights))
        )
        hold = min(hold, total - cursor)
        if hold <= 0:
            # A silently-dropped action would mean the case no longer tests
            # what its pattern declares. Unreachable for the shipped suites.
            raise ValueError(
                f"pattern action {action!r} received no frames: {total} total "
                f"frames cannot fit a {len(pattern)}-action pattern"
            )
        segments.append((cursor, action, hold))
        cursor += hold
    return ActionScript.from_segments(segments)


@dataclass(frozen=True, slots=True)
class SuiteCase:
    case_id: str
    scene: str
    seed: int
    script: ActionScript

    def key(self, *, suite: str, model: str) -> str:
        return contract_digest(
            {
                "suite": suite,
                "model": model,
                "case_id": self.case_id,
                "scene": self.scene,
                "seed": self.seed,
                "script_digest": self.script.digest(),
            }
        )


def materialize_suite(name: str, capabilities: WorldCapabilities) -> list[SuiteCase]:
    if name not in SUITES:
        raise ValueError(f"unknown suite: {name} (have {sorted(SUITES)})")
    definition = SUITES[name]
    cases = []
    for index, (scene, seed, pattern) in enumerate(definition["cases"]):
        script = pattern_script(
            tuple(pattern),
            duration_seconds=float(definition["duration_seconds"]),
            capabilities=capabilities,
        )
        cases.append(
            SuiteCase(
                case_id=f"{name}-{index:03d}-{scene}-s{seed}",
                scene=scene,
                seed=seed,
                script=script,
            )
        )
    return cases


def suite_digest(name: str) -> str:
    definition = SUITES[name]
    return contract_digest(
        {
            "name": name,
            "duration_seconds": definition["duration_seconds"],
            "cases": [
                {"scene": scene, "seed": seed, "pattern": list(map(list, pattern))}
                for scene, seed, pattern in definition["cases"]
            ],
        }
    )


def export_suite_manifests(directory: Path) -> list[Path]:
    """Write the suite definitions as JSON (hashed into run ledgers)."""

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, definition in SUITES.items():
        payload = {
            "schema": "liveframe.world.suite.v1",
            "name": name,
            "duration_seconds": definition["duration_seconds"],
            "suite_digest": suite_digest(name),
            "cases": [
                {"scene": scene, "seed": seed, "pattern": list(map(list, pattern))}
                for scene, seed, pattern in definition["cases"]
            ],
        }
        path = directory / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    return written
