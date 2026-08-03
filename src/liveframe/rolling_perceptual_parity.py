"""Measure long-horizon pixel-space parity for a Rolling Forcing replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from liveframe.perceptual_policy import (
    LAYERED_POLICY_SCHEMA,
    evaluate_layered_policy,
    evaluate_policy,
    load_policy,
    summarize,
)
from liveframe.quality_metrics import intrinsic_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _torch_module_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _load_pixels(path: Path) -> tuple[np.memmap, dict[str, Any]]:
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "liveframe.decoded-pixels.v1":
        raise ValueError("unsupported decoded pixel schema")
    if metadata.get("dtype") != "uint8" or metadata.get("layout") != "T,H,W,C":
        raise ValueError("decoded pixels must be uint8 T,H,W,C")
    if _sha256(path) != metadata.get("sha256"):
        raise ValueError("decoded pixel checksum mismatch")
    shape = tuple(int(value) for value in metadata["shape"])
    if path.stat().st_size != int(np.prod(shape)):
        raise ValueError("decoded pixel byte size mismatch")
    return np.memmap(path, mode="r", dtype=np.uint8, shape=shape), metadata


def _torch_frames(array: np.ndarray, device: str) -> Any:
    import torch

    value = torch.from_numpy(np.asarray(array).copy()).permute(0, 3, 1, 2)
    return value.to(device=device, dtype=torch.float32).div_(127.5).sub_(1.0)


def _cosine_rows(left: Any, right: Any) -> list[float]:
    import torch.nn.functional as functional

    return functional.cosine_similarity(left, right, dim=-1).detach().cpu().tolist()


def _clip_feature_tensor(value: Any) -> Any:
    """Normalize Transformers CLIP API variants to projected image features."""

    pooled = getattr(value, "pooler_output", None)
    return pooled if pooled is not None else value


def _checkpoint(series: list[float], fps: int, seconds: float) -> float | None:
    index = round(seconds * fps) - 1
    return series[index] if 0 <= index < len(series) else None


def _flicker(means: list[float]) -> list[float]:
    """Absolute second difference of frame-average luma; one value per interior frame."""

    return [
        abs(means[index + 1] - 2 * means[index] + means[index - 1])
        for index in range(1, len(means) - 1)
    ]


def _intrinsic_metrics(
    *,
    prompt_cosines: list[float],
    motion_clip_distances: list[float],
    motion_feature_norms: list[float],
    motion_lpips: list[float],
    temporal_delta_cosines: list[float],
    luma_means: list[float],
) -> dict[str, float]:
    return intrinsic_metrics(
        prompt_cosines=prompt_cosines,
        motion_clip_distances=motion_clip_distances,
        motion_feature_norms=motion_feature_norms,
        motion_lpips=motion_lpips,
        temporal_delta_cosines=temporal_delta_cosines,
        luma_means=luma_means,
    )


def run_perceptual_parity(
    reference_path: Path,
    candidate_path: Path,
    *,
    clip_model_id: str,
    clip_revision: str,
    policy_path: Path | None,
    numerical_report_path: Path | None,
    prompt: str | None = None,
    batch_size: int = 8,
    device: str = "mps",
    fps: int = 16,
) -> dict[str, Any]:
    import lpips
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

    if batch_size < 1 or fps < 1:
        raise ValueError("batch_size and fps must be positive")
    reference, reference_metadata = _load_pixels(reference_path)
    candidate, candidate_metadata = _load_pixels(candidate_path)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate pixel shapes differ")
    if reference_metadata["source"]["fixture_id"] != candidate_metadata["source"]["fixture_id"]:
        raise ValueError("decoded pixels originate from different fixtures")
    if reference_metadata["decoder"] != candidate_metadata["decoder"]:
        raise ValueError("reference and candidate must use identical decoder provenance")

    policy, policy_digest = load_policy(policy_path)
    lpips_model = lpips.LPIPS(net="alex", version="0.1").to(device).eval()
    clip_model = CLIPModel.from_pretrained(
        clip_model_id, revision=clip_revision, local_files_only=False
    ).to(device).eval()
    clip_processor = CLIPImageProcessor.from_pretrained(
        clip_model_id, revision=clip_revision, local_files_only=False
    )
    prompt_feature = None
    if prompt is not None:
        if not prompt.strip():
            raise ValueError("prompt must not be blank")
        tokenizer = CLIPTokenizer.from_pretrained(
            clip_model_id, revision=clip_revision, local_files_only=False
        )
        tokens = tokenizer([prompt], padding=True, return_tensors="pt").to(device)
        prompt_feature = _clip_feature_tensor(clip_model.get_text_features(**tokens))

    lpips_values: list[float] = []
    clip_cosines: list[float] = []
    temporal_clip_cosines: list[float] = []
    temporal_lpips_delta: list[float] = []
    prior_reference_features = None
    prior_candidate_features = None
    prior_reference_frame = None
    prior_candidate_frame = None
    prior_reference_motion = None
    prior_candidate_motion = None
    reference_prompt_cosines: list[float] = []
    candidate_prompt_cosines: list[float] = []
    reference_motion_clip_distances: list[float] = []
    candidate_motion_clip_distances: list[float] = []
    reference_motion_feature_norms: list[float] = []
    candidate_motion_feature_norms: list[float] = []
    reference_motion_lpips_values: list[float] = []
    candidate_motion_lpips_values: list[float] = []
    reference_temporal_delta_cosines: list[float] = []
    candidate_temporal_delta_cosines: list[float] = []
    reference_luma_means: list[float] = []
    candidate_luma_means: list[float] = []

    with torch.inference_mode():
        for start in range(0, reference.shape[0], batch_size):
            end = min(start + batch_size, reference.shape[0])
            reference_batch = _torch_frames(reference[start:end], device)
            candidate_batch = _torch_frames(candidate[start:end], device)
            lpips_values.extend(
                lpips_model(reference_batch, candidate_batch)
                .reshape(-1)
                .detach()
                .cpu()
                .tolist()
            )

            reference_images = [Image.fromarray(frame) for frame in reference[start:end]]
            candidate_images = [Image.fromarray(frame) for frame in candidate[start:end]]
            combined = clip_processor(
                images=reference_images + candidate_images, return_tensors="pt"
            )["pixel_values"].to(device)
            features = _clip_feature_tensor(
                clip_model.get_image_features(pixel_values=combined)
            )
            reference_features, candidate_features = features.split(end - start)
            clip_cosines.extend(_cosine_rows(reference_features, candidate_features))
            if prompt_feature is not None:
                reference_prompt_cosines.extend(
                    _cosine_rows(reference_features, prompt_feature.expand_as(reference_features))
                )
                candidate_prompt_cosines.extend(
                    _cosine_rows(candidate_features, prompt_feature.expand_as(candidate_features))
                )

            if prior_reference_features is not None:
                reference_features = torch.cat(
                    [prior_reference_features, reference_features], dim=0
                )
                candidate_features = torch.cat(
                    [prior_candidate_features, candidate_features], dim=0
                )
                reference_motion = reference_features[1:] - reference_features[:-1]
                candidate_motion = candidate_features[1:] - candidate_features[:-1]
                temporal_clip_cosines.extend(_cosine_rows(reference_motion, candidate_motion))
            elif reference_features.shape[0] > 1:
                reference_motion = reference_features[1:] - reference_features[:-1]
                candidate_motion = candidate_features[1:] - candidate_features[:-1]
                temporal_clip_cosines.extend(_cosine_rows(reference_motion, candidate_motion))
            else:
                reference_motion = reference_features[:0]
                candidate_motion = candidate_features[:0]
            reference_motion_clip_distances.extend(
                1.0 - value
                for value in _cosine_rows(reference_features[:-1], reference_features[1:])
            )
            candidate_motion_clip_distances.extend(
                1.0 - value
                for value in _cosine_rows(candidate_features[:-1], candidate_features[1:])
            )
            reference_motion_feature_norms.extend(
                reference_motion.norm(dim=-1).detach().cpu().tolist()
            )
            candidate_motion_feature_norms.extend(
                candidate_motion.norm(dim=-1).detach().cpu().tolist()
            )
            if reference_motion.shape[0]:
                reference_for_delta = reference_motion
                candidate_for_delta = candidate_motion
                if prior_reference_motion is not None:
                    reference_for_delta = torch.cat(
                        [prior_reference_motion, reference_for_delta], dim=0
                    )
                    candidate_for_delta = torch.cat(
                        [prior_candidate_motion, candidate_for_delta], dim=0
                    )
                if reference_for_delta.shape[0] > 1:
                    reference_temporal_delta_cosines.extend(
                        _cosine_rows(reference_for_delta[:-1], reference_for_delta[1:])
                    )
                    candidate_temporal_delta_cosines.extend(
                        _cosine_rows(candidate_for_delta[:-1], candidate_for_delta[1:])
                    )
                prior_reference_motion = reference_motion[-1:]
                prior_candidate_motion = candidate_motion[-1:]
            prior_reference_features = reference_features[-1:]
            prior_candidate_features = candidate_features[-1:]

            if prior_reference_frame is not None:
                reference_batch = torch.cat([prior_reference_frame, reference_batch], dim=0)
                candidate_batch = torch.cat([prior_candidate_frame, candidate_batch], dim=0)
            reference_motion_lpips = lpips_model(reference_batch[:-1], reference_batch[1:])
            candidate_motion_lpips = lpips_model(candidate_batch[:-1], candidate_batch[1:])
            reference_motion_lpips_values.extend(
                reference_motion_lpips.reshape(-1).detach().cpu().tolist()
            )
            candidate_motion_lpips_values.extend(
                candidate_motion_lpips.reshape(-1).detach().cpu().tolist()
            )
            temporal_lpips_delta.extend(
                (reference_motion_lpips - candidate_motion_lpips)
                .abs()
                .reshape(-1)
                .detach()
                .cpu()
                .tolist()
            )
            prior_reference_frame = reference_batch[-1:]
            prior_candidate_frame = candidate_batch[-1:]
            reference_luma_means.extend(
                np.asarray(reference[start:end], dtype=np.float64).mean(axis=(1, 2, 3)).tolist()
            )
            candidate_luma_means.extend(
                np.asarray(candidate[start:end], dtype=np.float64).mean(axis=(1, 2, 3)).tolist()
            )
            print(f"measured pixel frames {start}:{end}/{reference.shape[0]}", flush=True)

    observed = {
        "lpips_mean": float(summarize(lpips_values)["mean"]),
        "lpips_p95": float(summarize(lpips_values)["p95"]),
        "clip_cosine_mean": float(summarize(clip_cosines)["mean"]),
        "clip_cosine_minimum": float(summarize(clip_cosines)["minimum"]),
        "temporal_clip_cosine_mean": float(summarize(temporal_clip_cosines)["mean"]),
        "temporal_clip_cosine_minimum": float(
            summarize(temporal_clip_cosines)["minimum"]
        ),
        "temporal_lpips_delta_mean": float(summarize(temporal_lpips_delta)["mean"]),
        "temporal_lpips_delta_p95": float(summarize(temporal_lpips_delta)["p95"]),
    }
    identity_observed = {
        "pairwise_lpips_mean": observed["lpips_mean"],
        "pairwise_clip_distance_mean": 1.0 - observed["clip_cosine_mean"],
        "pairwise_temporal_clip_distance_mean": 1.0
        - observed["temporal_clip_cosine_mean"],
    }
    reference_intrinsic = _intrinsic_metrics(
        prompt_cosines=reference_prompt_cosines,
        motion_clip_distances=reference_motion_clip_distances,
        motion_feature_norms=reference_motion_feature_norms,
        motion_lpips=reference_motion_lpips_values,
        temporal_delta_cosines=reference_temporal_delta_cosines,
        luma_means=reference_luma_means,
    )
    candidate_intrinsic = _intrinsic_metrics(
        prompt_cosines=candidate_prompt_cosines,
        motion_clip_distances=candidate_motion_clip_distances,
        motion_feature_norms=candidate_motion_feature_norms,
        motion_lpips=candidate_motion_lpips_values,
        temporal_delta_cosines=candidate_temporal_delta_cosines,
        luma_means=candidate_luma_means,
    )
    numerical_canary = None
    if numerical_report_path is not None:
        numerical = json.loads(numerical_report_path.read_text(encoding="utf-8"))
        numerical_canary = {
            "report": str(numerical_report_path),
            "report_sha256": _sha256(numerical_report_path),
            "passed": numerical["passed"],
            "failures": numerical["failures"],
            "output_latents": numerical["comparisons"]["output_latents"],
        }
    checkpoints = {
        label: {
            "lpips": _checkpoint(lpips_values, fps, seconds),
            "clip_cosine": _checkpoint(clip_cosines, fps, seconds),
            "temporal_clip_cosine": _checkpoint(
                temporal_clip_cosines, fps, seconds
            ),
            "temporal_lpips_delta": _checkpoint(
                temporal_lpips_delta, fps, seconds
            ),
        }
        for label, seconds in (("15s", 15.0), ("60s", 60.0))
        if reference.shape[0] / fps >= seconds
    }
    if policy is not None and policy.get("schema") == LAYERED_POLICY_SCHEMA:
        policy_verdict = evaluate_layered_policy(
            candidate_intrinsic, identity_observed, policy
        )
    else:
        policy_verdict = evaluate_policy(observed, policy)
    return {
        "schema": "liveframe.rolling-perceptual-parity-report.v2",
        "fixture_id": reference_metadata["source"]["fixture_id"],
        "pixel_shape": list(reference.shape),
        "fps": fps,
        "playback_seconds": reference.shape[0] / fps,
        "decoder": reference_metadata["decoder"],
        "metrics_runtime": {
            "device": device,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "lpips_version": importlib.metadata.version("lpips"),
            "transformers_version": importlib.metadata.version("transformers"),
            "lpips_network": "alex-v0.1",
            "lpips_state_sha256": _torch_module_sha256(lpips_model),
            "clip_model_id": clip_model_id,
            "clip_revision": clip_revision,
            "clip_state_sha256": _torch_module_sha256(clip_model),
        },
        "pixel_artifacts": {
            "reference_sha256": reference_metadata["sha256"],
            "candidate_sha256": candidate_metadata["sha256"],
        },
        "numerical_canary": numerical_canary,
        "perceptual_policy": {
            "path": str(policy_path) if policy_path else None,
            "digest": policy_digest,
            "verdict": policy_verdict,
        },
        "claim_layers": {
            "trajectory_canary": numerical_canary,
            "intrinsic_quality": {
                "prompt": prompt,
                "reference": reference_intrinsic,
                "candidate": candidate_intrinsic,
            },
            "pairwise_identity": identity_observed,
        },
        "observed": observed,
        "summaries": {
            "lpips": summarize(lpips_values),
            "clip_cosine": summarize(clip_cosines),
            "temporal_clip_cosine": summarize(temporal_clip_cosines),
            "temporal_lpips_delta": summarize(temporal_lpips_delta),
        },
        "checkpoints": checkpoints,
        "curves": {
            "lpips": lpips_values,
            "clip_cosine": clip_cosines,
            "temporal_clip_cosine": temporal_clip_cosines,
            "temporal_lpips_delta": temporal_lpips_delta,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--clip-model", required=True)
    parser.add_argument("--clip-revision", required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--numerical-report", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_perceptual_parity(
        args.reference,
        args.candidate,
        clip_model_id=args.clip_model,
        clip_revision=args.clip_revision,
        policy_path=args.policy,
        numerical_report_path=args.numerical_report,
        prompt=args.prompt,
        batch_size=args.batch_size,
        device=args.device,
        fps=args.fps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "perceptual_verdict": report["perceptual_policy"]["verdict"],
                "numerical_canary_passed": (
                    report["numerical_canary"]["passed"]
                    if report["numerical_canary"] is not None
                    else None
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()