"""Reusable intrinsic video quality metric aggregation."""

from __future__ import annotations

import math
from collections.abc import Sequence

from liveframe.perceptual_policy import summarize

QUALITY_METRIC_NAMES = frozenset(
    {
        "clip_prompt_mean",
        "clip_prompt_p10",
        "motion_clip_distance_mean",
        "motion_lpips_mean",
        "temporal_delta_cosine_mean",
        "luma_flicker_mean",
    }
)


def _flicker(means: Sequence[float]) -> list[float]:
    return [
        abs(means[index + 1] - 2 * means[index] + means[index - 1])
        for index in range(1, len(means) - 1)
    ]


def intrinsic_metrics(
    *,
    prompt_cosines: Sequence[float],
    motion_clip_distances: Sequence[float],
    motion_feature_norms: Sequence[float],
    motion_lpips: Sequence[float],
    temporal_delta_cosines: Sequence[float],
    luma_means: Sequence[float],
) -> dict[str, float]:
    """Validate complete-video series and reduce them to frozen v1 metrics."""

    if not prompt_cosines:
        return {}
    frame_count = len(prompt_cosines)
    if frame_count < 3:
        raise ValueError("intrinsic video metrics require at least three frames")
    expected_lengths = {
        "motion_clip_distances": (len(motion_clip_distances), frame_count - 1),
        "motion_feature_norms": (len(motion_feature_norms), frame_count - 1),
        "motion_lpips": (len(motion_lpips), frame_count - 1),
        "temporal_delta_cosines": (len(temporal_delta_cosines), frame_count - 2),
        "luma_means": (len(luma_means), frame_count),
    }
    for name, (actual, expected) in expected_lengths.items():
        if actual != expected:
            raise ValueError(
                f"intrinsic metric series {name} has {actual} values; expected {expected}"
            )
    series = {
        "prompt_cosines": prompt_cosines,
        "motion_clip_distances": motion_clip_distances,
        "motion_feature_norms": motion_feature_norms,
        "motion_lpips": motion_lpips,
        "temporal_delta_cosines": temporal_delta_cosines,
        "luma_means": luma_means,
    }
    for name, values in series.items():
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"intrinsic metric series {name} must be finite")
    valid_motion = sum(value > 1e-8 for value in motion_feature_norms)
    if not valid_motion:
        raise ValueError("intrinsic temporal metrics are undefined for zero feature motion")
    return {
        "clip_prompt_mean": float(summarize(list(prompt_cosines))["mean"]),
        "clip_prompt_p10": float(summarize(list(prompt_cosines))["p10"]),
        "motion_clip_distance_mean": float(
            summarize(list(motion_clip_distances))["mean"]
        ),
        "motion_valid_fraction": valid_motion / len(motion_feature_norms),
        "motion_lpips_mean": float(summarize(list(motion_lpips))["mean"]),
        "temporal_delta_cosine_mean": float(
            summarize(list(temporal_delta_cosines))["mean"]
        ),
        "luma_flicker_mean": float(summarize(_flicker(luma_means))["mean"]),
    }
