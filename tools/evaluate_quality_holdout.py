"""Evaluate the sealed holdout only after complete stochastic byte identity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.quality_artifacts import atomic_json  # noqa: E402
from liveframe.quality_calibration import evaluate_holdout, load_protocol  # noqa: E402
from liveframe.quality_capture import load_results, load_run_manifest  # noqa: E402
from liveframe.quality_freeze import (  # noqa: E402
    require_frozen_identities,
    require_frozen_policy_v2,
)
from liveframe.quality_holdout import (  # noqa: E402
    PAIRWISE_REPORT_SCHEMA,
    build_holdout_input,
    pair_holdout_results,
)
from liveframe.quality_metric_runner import (  # noqa: E402
    PerceptualMetricRunner,
    validate_metric_artifact,
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _publish_immutable(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if _object(path) != value:
            raise FileExistsError(f"refusing to replace evaluator artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def _artifact_sha(result: Any, name: str) -> str:
    return str(next(row["sha256"] for row in result["artifacts"] if row["path"] == name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-v2", type=Path, required=True)
    parser.add_argument("--sanity-v2", type=Path, required=True)
    parser.add_argument("--policy-bundle-v2", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metric-device", default="mps")
    parser.add_argument("--metric-batch-size", type=int, default=8)
    args = parser.parse_args()

    protocol, protocol_digest = load_protocol(args.protocol)
    manifest = load_run_manifest(args.manifest, protocol_digest=protocol_digest)
    require_frozen_identities(protocol_digest, manifest)
    protocol_v2, protocol_v2_digest = load_protocol(args.protocol_v2)
    sanity_v2 = _object(args.sanity_v2)
    bundle_v2 = _object(args.policy_bundle_v2)
    require_frozen_policy_v2(protocol_v2_digest, sanity_v2, bundle_v2)
    if protocol_v2.get("source_protocol_digest") != protocol_digest:
        raise ValueError("v2 policy does not descend from the capture protocol")

    result_paths = [
        *args.reference_root.rglob("result.json"),
        *args.candidate_root.rglob("result.json"),
    ]
    pairs = pair_holdout_results(
        protocol,
        manifest,
        load_results(result_paths),
        require_complete=True,
    )

    provenance = validate_metric_artifact(protocol, REPOSITORY_ROOT)
    metrics = PerceptualMetricRunner(
        provenance,
        device=args.metric_device,
        batch_size=args.metric_batch_size,
    )
    reports: list[dict[str, Any]] = []
    for pair in pairs:
        candidate_run_id = str(pair.candidate_job["run_id"])
        report_path = args.output_root / "pairwise" / candidate_run_id / "report.json"
        if report_path.exists():
            report = _object(report_path)
        else:
            report = {
                "schema": PAIRWISE_REPORT_SCHEMA,
                "reference_run_id": pair.reference_job["run_id"],
                "candidate_run_id": candidate_run_id,
                "stochastic_payload_sha256": pair.stochastic_payload_sha256,
                "metric_provenance_digest": protocol["metric_provenance"][
                    "artifact_sha256"
                ],
                "pixel_artifacts": {
                    "reference_sha256": _artifact_sha(
                        pair.reference_result, "pixels.bin"
                    ),
                    "candidate_sha256": _artifact_sha(
                        pair.candidate_result, "pixels.bin"
                    ),
                },
                "identity_metrics": metrics.measure_pairwise(
                    pair.reference_pixels,
                    pair.candidate_pixels,
                    expected_frames=int(pair.candidate_job["pixel_frames"]),
                ),
            }
            _publish_immutable(report_path, report)
        reports.append(report)

    holdout_input = build_holdout_input(
        protocol, bundle_v2, manifest, pairs, reports
    )
    verdict = evaluate_holdout(bundle_v2, holdout_input)
    _publish_immutable(args.output_root / "holdout-input.json", holdout_input)
    _publish_immutable(args.output_root / "verdict.json", verdict)
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()