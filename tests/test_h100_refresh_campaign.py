from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_h100_refresh_campaign import (  # noqa: E402
    BATCH_DIGEST,
    MANIFEST_DIGEST,
    _drift,
    _selected_jobs,
    load_campaign,
    require_clean_outputs,
    require_execution_acknowledgment,
    validate_identity_anchor_report,
)

PROTOCOL = ROOT / "artifacts/perceptual-protocol.v1.json"
BUNDLE = ROOT / "artifacts/perceptual-policy-bundle.v2.json"
MANIFEST = ROOT / "artifacts/refresh-addressability-manifest.v2.json"
BATCH = ROOT / "artifacts/h100-refresh-proof-batch.v2.json"
OUTPUT = Path("artifacts/h100-refresh-proof-results-v2")


def test_runner_loads_only_the_final_frozen_campaign() -> None:
    _protocol, manifest, batch = load_campaign(
        protocol_path=PROTOCOL,
        bundle_path=BUNDLE,
        manifest_path=MANIFEST,
        batch_path=BATCH,
        output_root=OUTPUT,
    )

    assert manifest["manifest_digest"] == MANIFEST_DIGEST
    assert batch["batch_digest"] == BATCH_DIGEST
    assert len(batch["jobs"]) == 64


def test_runner_rejects_output_root_outside_frozen_batch() -> None:
    with pytest.raises(ValueError, match="output root differs"):
        load_campaign(
            protocol_path=PROTOCOL,
            bundle_path=BUNDLE,
            manifest_path=MANIFEST,
            batch_path=BATCH,
            output_root=Path("artifacts/not-the-frozen-output"),
        )


def test_runner_job_selection_is_ordered_and_fail_closed() -> None:
    _protocol, _manifest, batch = load_campaign(
        protocol_path=PROTOCOL,
        bundle_path=BUNDLE,
        manifest_path=MANIFEST,
        batch_path=BATCH,
        output_root=OUTPUT,
    )
    first = str(batch["jobs"][0]["run_id"])
    second = str(batch["jobs"][1]["run_id"])

    assert [row["run_id"] for row in _selected_jobs(batch, {first, second}, 1)] == [
        first
    ]
    with pytest.raises(ValueError, match="unknown run IDs"):
        _selected_jobs(batch, {"refresh-proof-unknown"}, None)
    with pytest.raises(ValueError, match="limit must be positive"):
        _selected_jobs(batch, None, 0)


def test_runner_requires_clean_outputs_and_exact_visibility_acknowledgment(
    tmp_path: Path,
) -> None:
    job = {"run_id": "refresh-proof-test"}
    require_clean_outputs([job], tmp_path)
    output = tmp_path / "refresh-proof-test"
    output.mkdir()
    (output / "partial.bin").write_bytes(b"partial")
    with pytest.raises(FileExistsError, match="not empty"):
        require_clean_outputs([job], tmp_path)

    identity = tmp_path / "identity.json"
    with pytest.raises(ValueError, match="exact manifest"):
        require_execution_acknowledgment("wrong", identity)
    with pytest.raises(ValueError, match="identity-anchor-report"):
        require_execution_acknowledgment(MANIFEST_DIGEST, None)
    assert require_execution_acknowledgment(MANIFEST_DIGEST, identity) == identity


def test_runner_accepts_the_frozen_nested_identity_anchor_report() -> None:
    report = ROOT / "artifacts/h100-cfpp-1step-denominator.v1.json"

    assert validate_identity_anchor_report(report).startswith("sha256:")


def test_runner_rejects_conflicting_identity_anchor_fields(tmp_path: Path) -> None:
    report = tmp_path / "identity.json"
    report.write_text(
        '{"passed": true, "identity_anchor": {"passed": false}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not passing"):
        validate_identity_anchor_report(report)


def test_allocation_smoke_is_proof_blind_by_construction() -> None:
    source = (ROOT / "tools/run_h100_refresh_campaign.py").read_text(encoding="utf-8")
    method = source.split("def run_allocation_smoke", 1)[1].split("    def run(", 1)[0]
    main = source.split("def main()", 1)[1]

    assert '"proof_inputs_loaded": False' in method
    assert '"proof_results_generated": False' in method
    assert "job[" not in method
    assert "manifest[" not in method
    assert "torch.arange(480 * 832 * 3" in method
    assert main.index("if args.allocation_smoke_output is not None:") < main.index(
        "protocol, manifest, batch = load_campaign("
    )


def test_drift_cosine_is_clamped_to_its_mathematical_range(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    import torch.nn.functional as functional

    monkeypatch.setattr(
        functional,
        "cosine_similarity",
        lambda *_args, **_kwargs: torch.tensor(1.0000001192092896),
    )

    assert _drift(torch.ones(2), torch.ones(2))["cosine"] == 1.0