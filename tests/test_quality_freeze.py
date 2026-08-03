from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from liveframe.quality_freeze import (
    MANIFEST_DIGEST,
    MLX_CANDIDATE_BATCH_DIGEST,
    POLICY_BUNDLE_V2_DIGEST,
    PROTOCOL_DIGEST,
    PROTOCOL_V2_DIGEST,
    require_frozen_identities,
    require_frozen_mlx_candidate_batch,
    require_frozen_policy_v2,
)

ROOT = Path(__file__).parents[1]


def test_frozen_campaign_identities_pass() -> None:
    require_frozen_identities(
        PROTOCOL_DIGEST,
        {"protocol_digest": PROTOCOL_DIGEST, "manifest_digest": MANIFEST_DIGEST},
    )


@pytest.mark.parametrize(
    ("protocol_digest", "manifest", "message"),
    [
        (
            "sha256:other",
            {"protocol_digest": "sha256:other", "manifest_digest": MANIFEST_DIGEST},
            "protocol is not",
        ),
        (
            PROTOCOL_DIGEST,
            {"protocol_digest": "sha256:other", "manifest_digest": MANIFEST_DIGEST},
            "manifest protocol digest",
        ),
        (
            PROTOCOL_DIGEST,
            {"protocol_digest": PROTOCOL_DIGEST, "manifest_digest": "sha256:other"},
            "manifest is not",
        ),
    ],
)
def test_frozen_campaign_identity_mismatch_fails(
    protocol_digest: str, manifest: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        require_frozen_identities(protocol_digest, manifest)


def test_frozen_v2_policy_identities_pass() -> None:
    sanity = json.loads(
        (ROOT / "artifacts" / "h100-calibration-sanity.v2.json").read_text()
    )
    bundle = json.loads(
        (ROOT / "artifacts" / "perceptual-policy-bundle.v2.json").read_text()
    )

    assert bundle["bundle_digest"] == POLICY_BUNDLE_V2_DIGEST
    require_frozen_policy_v2(PROTOCOL_V2_DIGEST, sanity, bundle)


def test_frozen_v2_policy_tampering_fails() -> None:
    sanity = json.loads(
        (ROOT / "artifacts" / "h100-calibration-sanity.v2.json").read_text()
    )
    bundle = json.loads(
        (ROOT / "artifacts" / "perceptual-policy-bundle.v2.json").read_text()
    )
    altered = copy.deepcopy(sanity)
    altered["warnings"] = []

    with pytest.raises(ValueError, match="sanity report is not"):
        require_frozen_policy_v2(PROTOCOL_V2_DIGEST, altered, bundle)


def test_frozen_v2_rejects_tampered_policy_bundle() -> None:
    sanity = json.loads(
        (ROOT / "artifacts" / "h100-calibration-sanity.v2.json").read_text()
    )
    bundle = json.loads(
        (ROOT / "artifacts" / "perceptual-policy-bundle.v2.json").read_text()
    )
    bundle["expected_holdout_seeds"][0] = "altered"

    with pytest.raises(ValueError, match="bundle payload digest mismatch"):
        require_frozen_policy_v2(PROTOCOL_V2_DIGEST, sanity, bundle)


def test_frozen_mlx_candidate_batch_passes_and_rejects_tampering() -> None:
    batch = json.loads(
        (ROOT / "artifacts" / "mlx-holdout-candidates.v1.json").read_text()
    )

    assert batch["batch_digest"] == MLX_CANDIDATE_BATCH_DIGEST
    require_frozen_mlx_candidate_batch(batch)
    altered = copy.deepcopy(batch)
    altered["jobs"] = altered["jobs"][:-1]
    with pytest.raises(ValueError, match="payload digest mismatch"):
        require_frozen_mlx_candidate_batch(altered)
