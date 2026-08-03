"""Execute a frozen MLX candidate batch for the sealed quality holdout."""

# mypy: ignore-errors

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from liveframe.mlx_quality_runner import MLXQualityRunner  # noqa: E402
from liveframe.quality_calibration import load_protocol  # noqa: E402
from liveframe.quality_capture import (  # noqa: E402
    load_holdout_candidate_batch_plan,
    load_run_manifest,
)
from liveframe.quality_executor import execute_batch  # noqa: E402
from liveframe.quality_freeze import (  # noqa: E402
    require_frozen_identities,
    require_frozen_mlx_candidate_batch,
    require_frozen_policy_v2,
)
from liveframe.quality_metric_runner import validate_metric_artifact  # noqa: E402


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-v2", type=Path, required=True)
    parser.add_argument("--sanity-v2", type=Path, required=True)
    parser.add_argument("--policy-bundle-v2", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--noise-root", type=Path, required=True)
    parser.add_argument("--metric-batch-size", type=int, default=8)
    parser.add_argument("--metric-device", default="mps")
    parser.add_argument(
        "--vae-execution-mode",
        choices=("eager", "compiled-steady"),
        default="eager",
    )
    args = parser.parse_args()

    protocol, protocol_digest = load_protocol(args.protocol)
    manifest = load_run_manifest(args.manifest, protocol_digest=protocol_digest)
    require_frozen_identities(protocol_digest, manifest)
    protocol_v2, protocol_v2_digest = load_protocol(args.protocol_v2)
    sanity_v2 = _object(args.sanity_v2)
    bundle_v2 = _object(args.policy_bundle_v2)
    require_frozen_policy_v2(protocol_v2_digest, sanity_v2, bundle_v2)
    if protocol_v2["source_protocol_digest"] != protocol_digest:
        raise ValueError("v2 policy does not descend from the frozen capture protocol")
    batch = load_holdout_candidate_batch_plan(args.batch, manifest)
    require_frozen_mlx_candidate_batch(batch)
    if Path(batch["output_root"]) != args.output_root:
        raise ValueError("output root does not match the frozen candidate batch")
    metric_provenance = validate_metric_artifact(protocol, REPOSITORY_ROOT)
    runner = MLXQualityRunner(
        model_dir=args.model_dir,
        bundle=bundle_v2,
        metric_provenance=metric_provenance,
        metric_batch_size=args.metric_batch_size,
        noise_root=args.noise_root,
        metric_device=args.metric_device,
        vae_execution_mode=args.vae_execution_mode,
    )
    report = execute_batch(protocol, manifest, batch, args.output_root, runner)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()