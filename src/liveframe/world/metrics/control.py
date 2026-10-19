"""Control metrics: does the video's motion match the commanded action?

Default CPU tier (spec section 8.2): dense Farneback optical flow on
grayscale frames (captured at <= 256 px short side), statistics restricted to
TEXTURED pixels (Sobel gradient magnitude above a floor) because textureless
regions (sky, flat walls) report near-zero flow and would otherwise fake
parallax under pure rotation. Per-frame-pair classification:

- no-op: median textured-pixel flow magnitude below a calibrated threshold;
- forward/back: divergence (trace of an affine flow fit on textured points);
  expansion = forward;
- strafe vs yaw: translation parallel to the image plane produces
  depth-dependent flow (on a ground plane, |dx| grows toward lower, closer
  rows) while rotation produces depth-independent flow. The discriminator is
  the ROW-VARIATION RATIO |d(dx)/d(row)| * height / |mean_dx|: measured ~5.4
  for strafe and <= 1.2 for yaw across 0.5x-2x speeds on the synthetic
  ground truth (calibration_sweep in agent/world/metric-validation-m3.json),
  thresholded at 2.5.
  This replaces the spec's suggested homography-residual test, which did not
  separate the classes on synthetic ground truth (recorded in the M3 report);
- direction: sign of mean textured dx, interpreted under the strafe/yaw
  decision.

Sign conventions (calibrated on the synthetic world): camera yaw LEFT and
camera strafe LEFT both shift scene content RIGHT in the image (mean_dx > 0
-> L or A). Every function is deterministic for fixed frames.

Classes: W (forward), S (back), A/D (strafe), L/R (yaw), 0 (no-op). Pitch
(U/N) is capability-gated and outside the M3 gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# Calibrated on the synthetic world (calibration_sweep recorded in
# agent/world/metric-validation-m3.json via tools/validate_world_metrics.py).
# Flow units: pixels at <=256 px short side.
NOOP_MEDIAN_FLOW_THRESHOLD = 0.08
FORWARD_DIVERGENCE_THRESHOLD = 0.02
ROW_VARIATION_STRAFE_THRESHOLD = 2.5
TEXTURE_GRADIENT_FLOOR = 20.0
MIN_TEXTURED_PIXELS = 100


def to_grayscale(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("frames must be (n, h, w, 3) uint8")
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (frames.astype(np.float32) @ weights).astype(np.float32)


# OpenCV's own op-level threading fights our pair-level pool for the same cores
# and leaves each pair slow; one thread per Farneback call, parallelised across
# pairs, is both faster and deterministic. Set once at import.
cv2.setNumThreads(1)

# Pair-level flow dominates the metric pass. calcOpticalFlowFarneback is a
# GIL-releasing C++ call, so a thread pool gives real parallelism with zero
# pickling and byte-identical results (each pair is independent and order is
# preserved). This is what clears the >=40 frames/s metric-throughput gate on
# many-core hosts; a swap to DIS flow (faster still) is tracked separately and
# needs classifier recalibration, so it is not bundled here.
_FLOW_POOL_MIN_PAIRS = 8


def _pair_features_parallel(gray: np.ndarray) -> list[PairFeatures]:
    pair_count = len(gray) - 1
    if pair_count <= 0:
        return []
    if pair_count < _FLOW_POOL_MIN_PAIRS:
        return [pair_features(gray[i], gray[i + 1]) for i in range(pair_count)]
    import os
    from concurrent.futures import ThreadPoolExecutor

    workers = min(pair_count, (os.cpu_count() or 1))

    def _one(index: int) -> PairFeatures:
        return pair_features(gray[index], gray[index + 1])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, range(pair_count)))


def farneback_flow(gray_prev: np.ndarray, gray_next: np.ndarray) -> np.ndarray:
    flow: np.ndarray = cv2.calcOpticalFlowFarneback(  # type: ignore[call-overload]
        gray_prev,
        gray_next,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow


@dataclass(frozen=True, slots=True)
class PairFeatures:
    median_flow: float
    mean_dx: float
    mean_dy: float
    divergence: float
    row_variation_ratio: float
    textured_fraction: float


def pair_features(gray_prev: np.ndarray, gray_next: np.ndarray) -> PairFeatures:
    flow = farneback_flow(gray_prev, gray_next)
    height, width = flow.shape[:2]
    gradient_x = cv2.Sobel(gray_prev, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Sobel(gray_prev, cv2.CV_32F, 0, 1)
    textured = np.hypot(gradient_x, gradient_y) > TEXTURE_GRADIENT_FLOOR
    if int(textured.sum()) < MIN_TEXTURED_PIXELS:
        textured = np.ones_like(textured)

    rows, cols = np.nonzero(textured)
    dx = flow[rows, cols, 0].astype(np.float64)
    dy = flow[rows, cols, 1].astype(np.float64)
    magnitude = np.hypot(dx, dy)
    median_flow = float(np.median(magnitude))
    mean_dx = float(dx.mean())
    mean_dy = float(dy.mean())

    # Affine fit (flow ~ A x + b) on textured points; divergence = trace(A).
    centered = np.stack(
        [cols - width / 2.0, rows - height / 2.0, np.ones_like(cols)], axis=1
    ).astype(np.float64)
    displacements = np.stack([dx, dy], axis=1)
    coefficients, *_ = np.linalg.lstsq(centered, displacements, rcond=None)
    divergence = float(coefficients[0, 0] + coefficients[1, 1])

    # Row-variation ratio: depth-dependence of horizontal flow.
    row_design = np.stack([rows.astype(np.float64), np.ones_like(rows, np.float64)], axis=1)
    (row_slope, _), *_ = np.linalg.lstsq(row_design, dx, rcond=None)
    row_variation_ratio = abs(float(row_slope)) * height / max(abs(mean_dx), 1e-6)

    return PairFeatures(
        median_flow=median_flow,
        mean_dx=mean_dx,
        mean_dy=mean_dy,
        divergence=divergence,
        row_variation_ratio=row_variation_ratio,
        textured_fraction=float(textured.mean()),
    )


def classify_pair(features: PairFeatures) -> str:
    if features.median_flow < NOOP_MEDIAN_FLOW_THRESHOLD:
        return "0"
    if abs(features.divergence) > FORWARD_DIVERGENCE_THRESHOLD and abs(
        features.divergence
    ) * 50.0 > abs(features.mean_dx):
        return "W" if features.divergence > 0 else "S"
    if features.row_variation_ratio > ROW_VARIATION_STRAFE_THRESHOLD:
        return "A" if features.mean_dx > 0 else "D"
    return "L" if features.mean_dx > 0 else "R"


_CLASS_OF_TOKEN = {"W": "W", "S": "S", "A": "A", "D": "D", "L": "L", "R": "R", "0": "0"}


def commanded_class(action: str) -> str | None:
    """Primary gate class of a (possibly combined) action; None for pitch-only."""

    for token in action.split("+"):
        if token in _CLASS_OF_TOKEN:
            return _CLASS_OF_TOKEN[token]
    return None


def _motion_energy(features: PairFeatures) -> dict[str, float]:
    """Decompose a pair's motion into per-class energies for purity."""

    horizontal = abs(features.mean_dx)
    divergent = abs(features.divergence) * 50.0
    strafe_fraction = min(
        1.0, features.row_variation_ratio / (2.0 * ROW_VARIATION_STRAFE_THRESHOLD)
    )
    return {
        ("A" if features.mean_dx > 0 else "D"): horizontal * strafe_fraction,
        ("L" if features.mean_dx > 0 else "R"): horizontal * (1.0 - strafe_fraction),
        ("W" if features.divergence > 0 else "S"): divergent,
    }


@dataclass(frozen=True, slots=True)
class SegmentControl:
    action: str
    commanded: str | None
    start_pair: int
    pairs: int
    direction_accuracy: float | None
    direction_purity: float | None
    motion_onset_frames: int | None
    noop_drift: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "commanded_class": self.commanded,
            "start_pair": self.start_pair,
            "pairs": self.pairs,
            "direction_accuracy": self.direction_accuracy,
            "direction_purity": self.direction_purity,
            "motion_onset_frames": self.motion_onset_frames,
            "noop_drift": self.noop_drift,
        }


def analyze_segments(
    frames: np.ndarray,
    segments: Sequence[tuple[int, str, int]],
) -> tuple[list[SegmentControl], list[str], list[PairFeatures]]:
    """Per-segment control metrics over captured frames.

    Pair ``i`` spans frames (i, i+1). A segment starting at frame ``t`` owns
    pairs ``t-1 .. t+hold-2``: its first pair is the transition INTO it, which
    is where onset counting starts (frame t is the first generated under the
    new action).
    """

    gray = to_grayscale(frames)
    features = _pair_features_parallel(gray)
    classes = [classify_pair(f) for f in features]
    results: list[SegmentControl] = []
    for t_frames, action, hold in segments:
        start_pair = max(0, t_frames - 1)
        end_pair = min(len(features), t_frames + hold - 1)
        pair_count = max(0, end_pair - start_pair)
        if pair_count == 0:
            continue
        commanded = commanded_class(action)
        segment_classes = classes[start_pair:end_pair]
        segment_features = features[start_pair:end_pair]
        if commanded is None:
            results.append(
                SegmentControl(action, None, start_pair, pair_count, None, None, None, None)
            )
            continue
        if commanded == "0":
            drift = float(np.mean([f.median_flow for f in segment_features]))
            onset = next((i for i, c in enumerate(segment_classes) if c == "0"), None)
            accuracy = segment_classes.count("0") / pair_count
            results.append(
                SegmentControl(
                    action, "0", start_pair, pair_count, accuracy, None, onset, drift
                )
            )
            continue
        onset = next((i for i, c in enumerate(segment_classes) if c == commanded), None)
        after_onset = segment_classes[onset:] if onset is not None else []
        accuracy = after_onset.count(commanded) / len(after_onset) if after_onset else 0.0
        purities = []
        for f in segment_features[onset:] if onset is not None else []:
            energies = _motion_energy(f)
            total = sum(energies.values())
            purities.append(energies.get(commanded, 0.0) / total if total > 0 else 0.0)
        results.append(
            SegmentControl(
                action,
                commanded,
                start_pair,
                pair_count,
                accuracy,
                float(np.mean(purities)) if purities else None,
                onset,
                None,
            )
        )
    return results, classes, features


def simple_ssim(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Deterministic uniform-window SSIM on uint8 RGB images."""

    if image_a.shape != image_b.shape:
        raise ValueError("SSIM inputs must share a shape")
    a = to_grayscale(image_a[None])[0].astype(np.float64)
    b = to_grayscale(image_b[None])[0].astype(np.float64)
    kernel = (8, 8)
    mu_a = cv2.blur(a, kernel)
    mu_b = cv2.blur(b, kernel)
    sigma_a = cv2.blur(a * a, kernel) - mu_a * mu_a
    sigma_b = cv2.blur(b * b, kernel) - mu_b * mu_b
    sigma_ab = cv2.blur(a * b, kernel) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    )
    return float(ssim_map.mean())


def reversal_return(frames: np.ndarray, start: int, end: int) -> float:
    """Similarity between the frame before a reversal pair and after it."""

    return simple_ssim(frames[start], frames[min(end, len(frames) - 1)])
