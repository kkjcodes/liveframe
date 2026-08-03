"""Single-video CLIP and LPIPS measurement for frozen quality captures."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np

from liveframe.quality_capture import sha256_file
from liveframe.quality_metrics import QUALITY_METRIC_NAMES, intrinsic_metrics

PIXEL_SCHEMA = "liveframe.decoded-pixels.v1"
PAIRWISE_IDENTITY_METRIC_NAMES = frozenset(
    {
        "pairwise_lpips_mean",
        "pairwise_clip_distance_mean",
        "pairwise_temporal_clip_distance_mean",
    }
)


def torch_module_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def load_pixels(path: Path) -> tuple[np.memmap[Any, Any], dict[str, Any]]:
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema") != PIXEL_SCHEMA:
        raise ValueError("unsupported decoded pixel schema")
    if metadata.get("dtype") != "uint8" or metadata.get("layout") != "T,H,W,C":
        raise ValueError("decoded pixels must be uint8 T,H,W,C")
    if sha256_file(path) != metadata.get("sha256"):
        raise ValueError("decoded pixel checksum mismatch")
    shape = tuple(int(value) for value in metadata["shape"])
    if len(shape) != 4 or shape[-1] != 3 or path.stat().st_size != int(np.prod(shape)):
        raise ValueError("decoded pixel shape or byte size mismatch")
    return np.memmap(path, mode="r", dtype=np.uint8, shape=shape), metadata


def validate_metric_artifact(protocol: dict[str, Any], repository_root: Path) -> dict[str, Any]:
    provenance = protocol["metric_provenance"]
    path = repository_root / str(provenance["artifact"])
    if sha256_file(path) != provenance["artifact_sha256"]:
        raise ValueError("metric provenance artifact checksum mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("metric provenance artifact must be an object")
    return value


class PerceptualMetricRunner:
    """Keep frozen metric models resident while measuring complete videos."""

    def __init__(self, provenance: dict[str, Any], *, device: str, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("metric batch size must be positive")
        lpips = importlib.import_module("lpips")
        transformers = importlib.import_module("transformers")

        self.device = device
        self.batch_size = batch_size
        clip = provenance["clip"]
        self.clip_model = transformers.CLIPModel.from_pretrained(
            clip["repository"], revision=clip["revision"], local_files_only=False
        ).to(device).eval()
        self.clip_processor = transformers.CLIPImageProcessor.from_pretrained(
            clip["repository"], revision=clip["revision"], local_files_only=False
        )
        self.clip_tokenizer = transformers.CLIPTokenizer.from_pretrained(
            clip["repository"], revision=clip["revision"], local_files_only=False
        )
        lpips_spec = provenance["lpips"]
        self.lpips_model = lpips.LPIPS(
            net=lpips_spec["network"], version=lpips_spec["version"]
        ).to(device).eval()
        if importlib.metadata.version("lpips") != lpips_spec["package_version"]:
            raise ValueError("LPIPS package version does not match frozen provenance")
        if torch_module_sha256(self.clip_model) != clip["state_sha256"]:
            raise ValueError("CLIP state does not match frozen provenance")
        if torch_module_sha256(self.lpips_model) != lpips_spec["state_sha256"]:
            raise ValueError("LPIPS state does not match frozen provenance")

    def _frames(self, array: np.ndarray[Any, Any]) -> Any:
        import torch

        value = torch.from_numpy(np.asarray(array).copy()).permute(0, 3, 1, 2)
        return value.to(device=self.device, dtype=torch.float32).div_(127.5).sub_(1.0)

    @staticmethod
    def _cosine(left: Any, right: Any) -> list[float]:
        import torch.nn.functional as functional

        return functional.cosine_similarity(left, right, dim=-1).detach().cpu().tolist()

    @staticmethod
    def _features(value: Any) -> Any:
        pooled = getattr(value, "pooler_output", None)
        return pooled if pooled is not None else value

    def measure(self, pixel_path: Path, *, prompt: str, expected_frames: int) -> dict[str, float]:
        import torch
        from PIL import Image

        pixels, _ = load_pixels(pixel_path)
        if pixels.shape[0] != expected_frames:
            raise ValueError(
                f"decoded pixel frame count {pixels.shape[0]} does not match {expected_frames}"
            )
        tokens = self.clip_tokenizer([prompt], padding=True, return_tensors="pt").to(
            self.device
        )
        prompt_feature = self._features(self.clip_model.get_text_features(**tokens))
        prompt_cosines: list[float] = []
        motion_distances: list[float] = []
        motion_norms: list[float] = []
        motion_lpips: list[float] = []
        delta_cosines: list[float] = []
        luma_means: list[float] = []
        prior_features = None
        prior_motion = None
        prior_frame = None
        with torch.inference_mode():
            for start in range(0, pixels.shape[0], self.batch_size):
                end = min(start + self.batch_size, pixels.shape[0])
                frames = self._frames(pixels[start:end])
                images = [Image.fromarray(frame) for frame in pixels[start:end]]
                clip_pixels = self.clip_processor(images=images, return_tensors="pt")[
                    "pixel_values"
                ].to(self.device)
                features = self._features(
                    self.clip_model.get_image_features(pixel_values=clip_pixels)
                )
                prompt_cosines.extend(
                    self._cosine(features, prompt_feature.expand_as(features))
                )
                sequence = (
                    torch.cat([prior_features, features], dim=0)
                    if prior_features is not None
                    else features
                )
                motion = sequence[1:] - sequence[:-1]
                motion_distances.extend(
                    1.0 - value for value in self._cosine(sequence[:-1], sequence[1:])
                )
                motion_norms.extend(motion.norm(dim=-1).detach().cpu().tolist())
                delta_sequence = (
                    torch.cat([prior_motion, motion], dim=0)
                    if prior_motion is not None
                    else motion
                )
                if delta_sequence.shape[0] > 1:
                    delta_cosines.extend(
                        self._cosine(delta_sequence[:-1], delta_sequence[1:])
                    )
                if motion.shape[0]:
                    prior_motion = motion[-1:]
                prior_features = sequence[-1:]
                lpips_frames = (
                    torch.cat([prior_frame, frames], dim=0)
                    if prior_frame is not None
                    else frames
                )
                motion_lpips.extend(
                    self.lpips_model(lpips_frames[:-1], lpips_frames[1:])
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .tolist()
                )
                prior_frame = lpips_frames[-1:]
                luma_means.extend(
                    np.asarray(pixels[start:end], dtype=np.float64)
                    .mean(axis=(1, 2, 3))
                    .tolist()
                )
        measured = intrinsic_metrics(
            prompt_cosines=prompt_cosines,
            motion_clip_distances=motion_distances,
            motion_feature_norms=motion_norms,
            motion_lpips=motion_lpips,
            temporal_delta_cosines=delta_cosines,
            luma_means=luma_means,
        )
        return {name: measured[name] for name in QUALITY_METRIC_NAMES}

    def measure_pairwise(
        self, reference_path: Path, candidate_path: Path, *, expected_frames: int
    ) -> dict[str, float]:
        """Measure the frozen same-seed identity tripwires for one video pair."""

        import torch
        from PIL import Image

        reference, _ = load_pixels(reference_path)
        candidate, _ = load_pixels(candidate_path)
        if reference.shape != candidate.shape:
            raise ValueError("reference and candidate pixel shapes differ")
        if reference.shape[0] != expected_frames:
            raise ValueError(
                f"decoded pixel frame count {reference.shape[0]} does not match "
                f"{expected_frames}"
            )
        lpips_values: list[float] = []
        clip_distances: list[float] = []
        temporal_clip_distances: list[float] = []
        prior_reference_features = None
        prior_candidate_features = None
        with torch.inference_mode():
            for start in range(0, reference.shape[0], self.batch_size):
                end = min(start + self.batch_size, reference.shape[0])
                reference_frames = self._frames(reference[start:end])
                candidate_frames = self._frames(candidate[start:end])
                lpips_values.extend(
                    self.lpips_model(reference_frames, candidate_frames)
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .tolist()
                )
                images = [Image.fromarray(frame) for frame in reference[start:end]] + [
                    Image.fromarray(frame) for frame in candidate[start:end]
                ]
                clip_pixels = self.clip_processor(images=images, return_tensors="pt")[
                    "pixel_values"
                ].to(self.device)
                features = self._features(
                    self.clip_model.get_image_features(pixel_values=clip_pixels)
                )
                reference_features, candidate_features = features.split(end - start)
                clip_distances.extend(
                    1.0 - value
                    for value in self._cosine(reference_features, candidate_features)
                )
                reference_sequence = (
                    torch.cat([prior_reference_features, reference_features], dim=0)
                    if prior_reference_features is not None
                    else reference_features
                )
                candidate_sequence = (
                    torch.cat([prior_candidate_features, candidate_features], dim=0)
                    if prior_candidate_features is not None
                    else candidate_features
                )
                reference_motion = reference_sequence[1:] - reference_sequence[:-1]
                candidate_motion = candidate_sequence[1:] - candidate_sequence[:-1]
                temporal_clip_distances.extend(
                    1.0 - value
                    for value in self._cosine(reference_motion, candidate_motion)
                )
                prior_reference_features = reference_features[-1:]
                prior_candidate_features = candidate_features[-1:]
        if len(lpips_values) != expected_frames or len(clip_distances) != expected_frames:
            raise RuntimeError("pairwise frame metrics are incomplete")
        if len(temporal_clip_distances) != expected_frames - 1:
            raise RuntimeError("pairwise temporal metrics are incomplete")
        return {
            "pairwise_lpips_mean": float(sum(lpips_values) / len(lpips_values)),
            "pairwise_clip_distance_mean": float(
                sum(clip_distances) / len(clip_distances)
            ),
            "pairwise_temporal_clip_distance_mean": float(
                sum(temporal_clip_distances) / len(temporal_clip_distances)
            ),
        }
