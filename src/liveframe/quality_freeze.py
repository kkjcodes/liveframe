"""Literal external anchors for the frozen quality calibration campaign."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PROTOCOL_DIGEST = "sha256:aef6a7ad9426c97a9ae7f99c9a7a4f93799a574ca9e88d341da430c60ca75c52"
MANIFEST_DIGEST = "sha256:8967e70cbc3978544eea92428d2dda78044ebbcaae91e933271f7e1abd72fdc4"
CHECKPOINT_DIGEST = "sha256:1f123dce60e60424a9c6d949643176840fceffb947932997bcf1202d93656d98"
PROTOCOL_V2_DIGEST = "sha256:f1b39f36a0ed6f361aaabf38062868d5f3e12a953627c67b1a736c8d56148b93"
SANITY_V2_DIGEST = "sha256:e0308cdb235264112c049d46b27e6370b3d0e0c7abc5f92ccbd79886d62c5841"
POLICY_BUNDLE_V2_DIGEST = (
    "sha256:41a228ceac16a026c7771d2e2c496714e5476c8e9287d3da3d15a78e993a9870"
)
MLX_CANDIDATE_BATCH_DIGEST = (
    "sha256:5019c73470eb11f63ff8a525fa8081a7f02b4ea764283f071df0c52d51bb57e7"
)
H100_REFERENCE_BATCH_DIGEST = (
    "sha256:10af5568b5c42ac9d081621a0f63862ec4c96c8751d03c8b12e775aa6dab337a"
)


def require_frozen_identities(protocol_digest: str, manifest: Mapping[str, Any]) -> None:
    if protocol_digest != PROTOCOL_DIGEST:
        raise ValueError(
            f"quality protocol is not the frozen campaign artifact: {protocol_digest}"
        )
    if manifest.get("protocol_digest") != PROTOCOL_DIGEST:
        raise ValueError("quality manifest protocol digest is not the frozen campaign artifact")
    if manifest.get("manifest_digest") != MANIFEST_DIGEST:
        raise ValueError(
            "quality manifest is not the frozen campaign artifact: "
            f"{manifest.get('manifest_digest')}"
        )


def require_frozen_policy_v2(
    protocol_digest: str,
    sanity: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> None:
    if protocol_digest != PROTOCOL_V2_DIGEST:
        raise ValueError("holdout protocol is not the frozen v2 artifact")
    if sanity.get("protocol_digest") != PROTOCOL_V2_DIGEST:
        raise ValueError("holdout sanity report is bound to another protocol")
    from liveframe.perceptual_policy import canonical_digest

    if canonical_digest(sanity) != SANITY_V2_DIGEST:
        raise ValueError("holdout sanity report is not the frozen v2 artifact")
    if bundle.get("protocol_digest") != PROTOCOL_V2_DIGEST:
        raise ValueError("holdout policy bundle is bound to another protocol")
    if bundle.get("bundle_digest") != POLICY_BUNDLE_V2_DIGEST:
        raise ValueError("holdout policy bundle is not the frozen v2 artifact")
    unsigned_bundle = {
        key: value for key, value in bundle.items() if key != "bundle_digest"
    }
    if canonical_digest(unsigned_bundle) != POLICY_BUNDLE_V2_DIGEST:
        raise ValueError("holdout policy bundle payload digest mismatch")


def require_frozen_mlx_candidate_batch(batch: Mapping[str, Any]) -> None:
    from liveframe.perceptual_policy import canonical_digest

    if batch.get("batch_digest") != MLX_CANDIDATE_BATCH_DIGEST:
        raise ValueError("MLX candidate batch is not the frozen holdout artifact")
    unsigned = {key: value for key, value in batch.items() if key != "batch_digest"}
    if canonical_digest(unsigned) != MLX_CANDIDATE_BATCH_DIGEST:
        raise ValueError("MLX candidate batch payload digest mismatch")
    jobs = batch.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 128:
        raise ValueError("MLX candidate batch must contain exactly 128 jobs")
    seeds = {job.get("seed") for job in jobs if isinstance(job, dict)}
    if seeds != {81131, 90001, 101111, 120011}:
        raise ValueError("MLX candidate batch seed set mismatch")
    if any(
        job.get("split") != "holdout"
        or job.get("role") != "candidate"
        or job.get("backend") != "mlx"
        for job in jobs
    ):
        raise ValueError("MLX candidate batch contains a non-candidate job")


def require_frozen_h100_reference_batch(batch: Mapping[str, Any]) -> None:
    from liveframe.perceptual_policy import canonical_digest

    if batch.get("batch_digest") != H100_REFERENCE_BATCH_DIGEST:
        raise ValueError("H100 reference batch is not the frozen holdout artifact")
    unsigned = {key: value for key, value in batch.items() if key != "batch_digest"}
    if canonical_digest(unsigned) != H100_REFERENCE_BATCH_DIGEST:
        raise ValueError("H100 reference batch payload digest mismatch")
    jobs = batch.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 128:
        raise ValueError("H100 reference batch must contain exactly 128 jobs")
    if any(
        job.get("split") != "holdout"
        or job.get("role") != "reference"
        or job.get("backend") != "cuda"
        for job in jobs
    ):
        raise ValueError("H100 reference batch contains a non-reference job")
