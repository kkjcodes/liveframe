"""LiveFrame command-line interface."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import liveframe
from liveframe.backends import BackendError, create_backend
from liveframe.claims import cmd_recompute, cmd_verify
from liveframe.engine import GenerationRequest
from liveframe.gates import GateError, load_bundled_gate, load_gate
from liveframe.host import collect_host_provenance, mobile_readiness
from liveframe.qualification import WORKLOAD_PROFILES, qualify
from liveframe.qualification_ledger import (
    build_qualification_ledger,
    load_qualification_ledger,
    write_qualification_ledger,
)

# Research-only modules (converters, evaluator, manifests, runs) are imported
# lazily inside their command handlers so the published wheel, which ships only
# the core, still provides a working CLI; invoking a research-only command from
# a wheel install reports that it needs the repository checkout.


class _ResearchCommandUnavailable(ValueError):
    def __init__(self, command: str) -> None:
        super().__init__(
            f"the '{command}' command requires the LiveFrame research checkout "
            "(github.com/kkjcodes/liveframe); it is not part of the published package"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="liveframe")
    parser.add_argument("--home", type=Path, help="runtime state directory (or LIVEFRAME_HOME)")
    commands = parser.add_subparsers(dest="command", required=True)

    gate = commands.add_parser("gate", help="inspect or verify immutable gate artifacts")
    gate_commands = gate.add_subparsers(dest="gate_command", required=True)
    gate_commands.add_parser("show")
    verify = gate_commands.add_parser("verify")
    verify.add_argument("path", nargs="?", type=Path)

    submit = commands.add_parser("submit", help="start a durable detached run")
    submit.add_argument("--idempotency-key")
    submit.add_argument("run_command", nargs=argparse.REMAINDER)

    status = commands.add_parser("status", help="show durable run state")
    status.add_argument("run_id")

    logs = commands.add_parser("logs", help="show the tail of a run log")
    logs.add_argument("run_id")
    logs.add_argument("--stream", choices=("stdout", "stderr"), default="stdout")
    logs.add_argument("--lines", type=int, default=100)

    cancel = commands.add_parser("cancel", help="request cooperative cancellation")
    cancel.add_argument("run_id")

    artifacts = commands.add_parser("artifacts", help="list run artifacts with checksums")
    artifacts.add_argument("run_id")

    evaluate = commands.add_parser("evaluate", help="evaluate gates.v1 proof evidence")
    evaluate.add_argument("evidence", type=Path)

    manifest = commands.add_parser("manifest", help="inspect upstream artifact pins")
    manifest.add_argument("path", type=Path)
    manifest.add_argument("--require-resolved", action="store_true")

    commands.add_parser("doctor", help="show host and secure mobile-control readiness")

    convert = commands.add_parser("convert-cfpp", help="convert a pinned CF++ student to MLX")
    convert.add_argument("--checkpoint", required=True, type=Path)
    convert.add_argument("--stock-mlx-dir", required=True, type=Path)
    convert.add_argument("--output-dir", required=True, type=Path)
    convert.add_argument("--expected-sha256", required=True)
    convert.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )

    qualify_command = commands.add_parser(
        "qualify", help="measure a backend against a deployment workload profile"
    )
    qualify_command.add_argument("--backend", default="synthetic")
    qualify_command.add_argument(
        "--profile", choices=tuple(WORKLOAD_PROFILES), default="interactive-480p"
    )
    qualify_command.add_argument("--prompt", default="LiveFrame synthetic qualification")
    qualify_command.add_argument("--duration", type=float)
    qualify_command.add_argument("--native-fps", type=float, default=16.0)
    qualify_command.add_argument("--width", type=int, default=832)
    qualify_command.add_argument("--height", type=int, default=480)
    qualify_command.add_argument("--seed", type=int, default=0)
    qualify_command.add_argument("--output", type=Path)

    qualification = commands.add_parser(
        "qualification", help="inspect or verify qualification evidence"
    )
    qualification_commands = qualification.add_subparsers(
        dest="qualification_command", required=True
    )
    qualification_verify = qualification_commands.add_parser(
        "verify", help="verify a qualification ledger"
    )
    qualification_verify.add_argument("ledger", type=Path)

    world = commands.add_parser("world", help="world-model runtime evaluation")
    world_commands = world.add_subparsers(dest="world_command", required=True)
    world_run = world_commands.add_parser("run", help="run a suite against a world model")
    world_run.add_argument(
        "model",
        help=(
            "synthetic, matrix-game-2, or your own adapter as FILE.py:ClassName / "
            "module:ClassName (see examples/byo_adapter/)"
        ),
    )
    world_run.add_argument(
        "--suite",
        default="smoke",
        choices=("smoke", "quick", "standard", "soak", "memory"),
    )
    world_run.add_argument("--python", type=Path, help="model-env interpreter (bridge)")
    world_run.add_argument(
        "--repo", type=Path, help="model repo checkout (recorded as provenance commit)"
    )
    world_run.add_argument("--run-root", type=Path, default=Path("runs"))
    world_run.add_argument(
        "--resume", type=Path, help="existing run dir: completed cases are skipped"
    )
    world_run.add_argument(
        "--shard",
        help=(
            "INDEX/TOTAL deterministic case subset; multi-GPU = one process per "
            "GPU (set CUDA_VISIBLE_DEVICES per process)"
        ),
    )
    world_run.add_argument(
        "--cold-start-storage",
        default="unknown",
        help="where weights load from (e.g. network-volume, local-nvme)",
    )
    world_run.add_argument(
        "--gpu-price-usd-per-hour",
        type=float,
        help="enables cost_per_generated_minute; omitted = field omitted (spec 7.3)",
    )
    world_run.add_argument(
        "--vendor-claim",
        help="vendor's published rate claim, quoted verbatim with its source",
    )
    world_verify = world_commands.add_parser("verify", help="verify a world run ledger")
    world_verify.add_argument("ledger", type=Path)
    world_fetch = world_commands.add_parser(
        "fetch", help="download a reference model's weights (wraps `hf download`)"
    )
    world_fetch.add_argument(
        "model", help="matrix-game-2 | matrix-game-3 | hy-worldplay"
    )
    world_fetch.add_argument("--local-dir", type=Path, required=True)
    world_fetch.add_argument(
        "--print-only",
        action="store_true",
        help="print the download command(s) and license notice without running",
    )
    world_doctor = world_commands.add_parser(
        "doctor", help="preflight a model environment; every failure prints its fix"
    )
    world_doctor.add_argument(
        "model",
        help="matrix-game-2 | matrix-game-3 | hy-worldplay | custom (--pins/--imports)",
    )
    world_doctor.add_argument(
        "--python", type=Path, required=True, help="model-env interpreter to probe"
    )
    world_doctor.add_argument("--repo", type=Path, help="model repo checkout")
    world_doctor.add_argument("--weights", type=Path, help="weights directory")
    world_doctor.add_argument(
        "--pins", type=Path, help="adapter pins.json lockfile (custom adapters)"
    )
    world_doctor.add_argument(
        "--imports",
        default="torch",
        help="comma-separated modules the adapter needs (custom adapters)",
    )
    world_suites = world_commands.add_parser(
        "suites", help="list suites with their cases and digests"
    )
    world_suites.add_argument(
        "--export-dir", type=Path, help="also write the suite manifests as JSON"
    )
    world_compare = world_commands.add_parser(
        "compare", help="side-by-side HTML; config differences listed before numbers"
    )
    world_compare.add_argument("report_a", type=Path, help="runA/report.json")
    world_compare.add_argument("report_b", type=Path, help="runB/report.json")
    world_compare.add_argument(
        "--output", type=Path, default=Path("comparison.html")
    )
    world_export = world_commands.add_parser(
        "export", help="export videos + action logs for external scorers"
    )
    world_export.add_argument("run_dir", type=Path)
    world_export.add_argument(
        "--format", default="raw", choices=("raw", "worldmark", "wbench")
    )
    world_export.add_argument("--output-dir", type=Path, required=True)
    world_export.add_argument("--model-tag", help="override the model tag in layouts")

    claims_verify = commands.add_parser(
        "verify", help="verify digest-bound artifact references in a claims ledger"
    )
    claims_verify.add_argument("ledger")
    claims_verify.add_argument(
        "--artifacts-root",
        default=None,
        help="root directory artifact paths are relative to (default: ledger grandparent)",
    )

    recompute = commands.add_parser(
        "recompute", help="recompute headline arithmetic from ledger values"
    )
    recompute.add_argument("ledger")

    commands.add_parser("version", help="print version")
    return parser


def _clean_remainder(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


# Reference-model weights. Each entry: HF repo id(s) and an optional license
# notice that MUST be shown before download. `extra` repos (backbones, encoders)
# are listed so the user knows the full footprint; HY also needs an HF token.
_REFERENCE_WEIGHTS: dict[str, dict[str, Any]] = {
    "matrix-game-2": {
        "repos": ["Skywork/Matrix-Game-2.0"],
        "approx_gb": 28,
        "license": "Apache-2.0 (Skywork/Matrix-Game-2.0)",
    },
    "matrix-game-3": {
        "repos": ["Skywork/Matrix-Game-3.0"],
        "approx_gb": 83,
        "license": "Apache-2.0 (Skywork/Matrix-Game-3.0)",
    },
    "hy-worldplay": {
        "repos": [
            "tencent/HY-WorldPlay",
            "tencent/HunyuanVideo-1.5",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "google/byt5-small",
            "black-forest-labs/FLUX.1-Redux-dev",
        ],
        "approx_gb": None,
        "needs_hf_token": True,
        "license": (
            "TENCENT HY-WORLDPLAY COMMUNITY LICENSE. This license DOES NOT APPLY "
            "in the European Union, United Kingdom and South Korea. Over 1M MAU "
            "requires a separate license from Tencent. Output may not be used to "
            "improve any other AI model. You must download weights yourself and "
            "accept the license on Hugging Face; LiveFrame never redistributes "
            "weights. FLUX.1-Redux-dev requires an HF access token."
        ),
    },
}


def _world_fetch(args: argparse.Namespace) -> int:
    import shutil
    import subprocess

    entry = _REFERENCE_WEIGHTS.get(args.model)
    if entry is None:
        raise ValueError(
            f"unknown reference model: {args.model} "
            f"(have {sorted(_REFERENCE_WEIGHTS)})"
        )
    if entry.get("license"):
        print(f"LICENSE — {args.model}:\n  {entry['license']}\n")
    commands = [
        ["hf", "download", repo, "--local-dir", str(args.local_dir / repo.split("/")[-1])]
        for repo in entry["repos"]
    ]
    if args.print_only or not shutil.which("hf"):
        if not args.print_only:
            print("`hf` CLI not found; printing commands only. Install: "
                  "pip install 'huggingface_hub[cli]'\n")
        for command in commands:
            print(" ".join(command))
        return 0
    for command in commands:
        print(f"$ {' '.join(command)}")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(f"fetch failed for {command[2]} (rc={result.returncode})")
            return 1
    return 0


def _capture_provenance(python: Path, repo: Path | None) -> dict[str, Any]:
    """Best-effort model-env torch/CUDA + repo commit for the run record."""
    import subprocess

    provenance: dict[str, Any] = {}
    try:
        out = subprocess.run(
            [str(python), "-c",
             "import torch;print(torch.__version__);print(torch.version.cuda)"],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode == 0:
            lines = out.stdout.strip().splitlines()
            provenance["torch"] = lines[0] if lines else None
            provenance["cuda"] = lines[1] if len(lines) > 1 else None
    except (OSError, subprocess.SubprocessError):
        pass
    if repo is not None:
        try:
            commit = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=30,
            )
            if commit.returncode == 0:
                provenance["repo_commit"] = commit.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return provenance


def _world_command(args: argparse.Namespace) -> int:
    # Lazy: the world harness needs the [world] extra (numpy).
    try:
        from liveframe.world.ledger import verify_world_ledger
        from liveframe.world.runner import run_suite
    except ModuleNotFoundError as exc:
        raise ValueError(
            "the world harness requires `pip install liveframe[world]`"
        ) from exc

    if args.world_command == "compare":
        from liveframe.world.report import build_comparison, render_comparison_html

        report_a = json.loads(args.report_a.read_text(encoding="utf-8"))
        report_b = json.loads(args.report_b.read_text(encoding="utf-8"))
        comparison = build_comparison(report_a, report_b)
        args.output.write_text(render_comparison_html(comparison), encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "comparable": comparison["comparable"],
                    "config_differences": comparison["config_differences"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.world_command == "export":
        from liveframe.world.export import export_run

        summary = export_run(
            args.run_dir,
            args.output_dir,
            export_format=args.format,
            model_tag=args.model_tag,
        )
        print(
            json.dumps(
                {key: value for key, value in summary.items() if key != "files"}
                | {"files_written": len(summary["files"])},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.world_command == "fetch":
        return _world_fetch(args)

    if args.world_command == "suites":
        from liveframe.world.suites import SUITES, export_suite_manifests, suite_digest

        listing = {
            name: {
                "cases": len(definition["cases"]),
                "seconds_per_case": definition["duration_seconds"],
                "suite_digest": suite_digest(name),
            }
            for name, definition in SUITES.items()
        }
        if args.export_dir is not None:
            export_suite_manifests(args.export_dir)
        print(json.dumps(listing, indent=2, sort_keys=True))
        return 0

    if args.world_command == "doctor":
        from liveframe.world import doctor

        if args.model == "matrix-game-2":
            results = doctor.run_matrix_game2_doctor(
                python=args.python,
                repo=args.repo or Path("/opt/code/Matrix-Game/Matrix-Game-2"),
                weights=args.weights or Path("/workspace/models/Matrix-Game-2.0"),
            )
        elif args.model == "matrix-game-3":
            results = doctor.run_matrix_game3_doctor(
                python=args.python,
                repo=args.repo or Path("/opt/code/Matrix-Game/Matrix-Game-3"),
                weights=args.weights or Path("/workspace/models/Matrix-Game-3.0"),
            )
        elif args.model == "hy-worldplay":
            results = doctor.run_hy_worldplay_doctor(
                python=args.python,
                repo=args.repo or Path("/opt/code/HY-WorldPlay"),
                weights=args.weights or Path("/workspace/models/HY-WorldPlay"),
            )
        elif args.model == "custom":
            modules = tuple(
                name.strip() for name in args.imports.split(",") if name.strip()
            )
            results = [doctor.check_hf_cli()]
            results.extend(doctor.check_model_env_imports(args.python, modules))
            if args.pins is not None:
                results.extend(doctor.check_adapter_pins(args.python, args.pins))
            if args.weights is not None:
                # No required-file list for custom adapters: check the
                # directory exists and is non-empty rather than passing a
                # vacuous zero-file check.
                weights_dir = Path(args.weights)
                has_files = weights_dir.is_dir() and any(weights_dir.iterdir())
                results.append(
                    {
                        "check": "weights-present",
                        "passed": has_files,
                        "detail": (
                            f"{weights_dir} exists and is non-empty"
                            if has_files
                            else f"{weights_dir} is missing or empty"
                        ),
                        "fix": None
                        if has_files
                        else f"hf download <repo> --local-dir {weights_dir}",
                    }
                )
        else:
            raise ValueError(
                f"unknown doctor target: {args.model} "
                "(matrix-game-2, matrix-game-3, hy-worldplay, or custom)"
            )
        failures = 0
        for result in results:
            mark = "ok " if result["passed"] else "FAIL"
            print(f"[{mark}] {result['check']}: {result['detail']}")
            if not result["passed"]:
                failures += 1
                if result.get("fix"):
                    print(f"       fix: {result['fix']}")
        print(f"{len(results) - failures}/{len(results)} checks passed")
        return 0 if failures == 0 else 1

    if args.world_command == "verify":
        ledger = verify_world_ledger(args.ledger)
        print(
            json.dumps(
                {
                    "schema": ledger["schema"],
                    "ledger_digest": ledger["ledger_digest"],
                    "files": len(ledger["files"]),
                    "verified": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0


    from liveframe.world.backend import WorldBackend

    # Parse --shard before any backend (a bridged worker process) is created,
    # so a malformed value can't leave a worker behind.
    shard: tuple[int, int] | None = None
    if args.shard:
        index_text, separator, total_text = args.shard.partition("/")
        if not separator or not index_text.isdigit() or not total_text.isdigit():
            raise ValueError(
                f"--shard must be INDEX/TOTAL with a 0-based index "
                f"(e.g. 0/4 .. 3/4), got {args.shard!r}"
            )
        shard = (int(index_text), int(total_text))

    backend: WorldBackend
    metadata: dict[str, Any]
    model_name = args.model
    if args.model == "synthetic":
        from liveframe.world.adapters.synthetic import SyntheticWorld

        backend = SyntheticWorld(busy=True, chunk_delay_seconds=0.004)
        metadata = {"backend": "synthetic-in-process"}
    elif args.model == "matrix-game-2":
        if args.python is None:
            raise ValueError("matrix-game-2 requires --python <model-env interpreter>")
        from liveframe.world.adapters import matrix_game2
        from liveframe.world.bridge.host import BridgedWorld

        bridged = BridgedWorld(
            python=args.python,
            adapter="matrix_game2:MatrixGame2Adapter",
            adapter_file=Path(matrix_game2.__file__),
            chunk_timeout_seconds=300.0,
        )
        backend = bridged
        metadata = {
            "backend": "bridged",
            "worker_python": bridged.worker_python,
            # M2.1 item 4: the frame basis as a machine-readable report field.
            "frame_basis": {
                "counted": "decoded pixel frames in host memory",
                "latent_to_pixel": "4L-3 causal expansion (first block 9, then 12)",
                "native_fps_source": "upstream utils/visualize.py:165 (fps = 12)",
                "interpolation": "none",
                "vendor_claim_basis": "not stated by vendor",
            },
        }
    elif args.model in ("matrix-game-3", "hy-worldplay"):
        if args.python is None:
            raise ValueError(f"{args.model} requires --python <model-env interpreter>")
        from liveframe.world.adapters import hy_worldplay, matrix_game3
        from liveframe.world.bridge.host import BridgedWorld

        ref_adapter, adapter_spec = (
            (matrix_game3, "matrix_game3:MatrixGame3Adapter")
            if args.model == "matrix-game-3"
            else (hy_worldplay, "hy_worldplay:HYWorldPlayAdapter")
        )
        ref_adapter_file = ref_adapter.__file__
        assert ref_adapter_file is not None
        # Batch adapters (HY) generate the WHOLE clip before the first chunk
        # replies, so the first-chunk wait is the full generation wall — a slow
        # offloaded batch run needs a long timeout; interactive (MG3) does not.
        bridged_timeout = 3600.0 if args.model == "hy-worldplay" else 600.0
        bridged = BridgedWorld(
            python=args.python,
            adapter=adapter_spec,
            adapter_file=Path(ref_adapter_file),
            chunk_timeout_seconds=bridged_timeout,
        )
        backend = bridged
        metadata = {
            "backend": "bridged",
            "worker_python": bridged.worker_python,
            # These reference adapters are authored from upstream source and are
            # validated against the live model in the M5 GPU session; a run here
            # before that validation fails closed in the adapter.
            "adapter_status": "authored-not-yet-validated (M5)",
        }
    elif ":" in args.model:
        # BYO adapter: FILE.py:ClassName in-process, + --python to run it
        # bridged inside the model's own environment.
        if args.python is not None:
            target, _, class_name = args.model.rpartition(":")
            if not target.endswith(".py"):
                raise ValueError(
                    "bridged custom adapters need a FILE.py:ClassName spec "
                    "(the file is sent to the model-env worker)"
                )
            from liveframe.world.bridge.host import BridgedWorld

            adapter_file = Path(target).resolve()
            bridged_custom = BridgedWorld(
                python=args.python,
                adapter=f"{adapter_file.stem}:{class_name}",
                adapter_file=adapter_file,
                chunk_timeout_seconds=300.0,
            )
            backend = bridged_custom
            metadata = {
                "backend": "bridged",
                "adapter_spec": args.model,
                "worker_python": bridged_custom.worker_python,
            }
        else:
            from liveframe.world.loader import load_backend

            backend = load_backend(args.model)
            metadata = {"backend": "custom-in-process", "adapter_spec": args.model}
        target = args.model.rpartition(":")[0]
        # File specs name the run dir by file stem; dotted module specs keep
        # the full dotted path (Path.stem would truncate pkg.module to "pkg").
        model_name = Path(target).stem if target.endswith(".py") else target
    else:
        raise ValueError(
            f"unknown world model: {args.model} (synthetic, matrix-game-2, or "
            "FILE.py:ClassName — see examples/byo_adapter/)"
        )

    if shard is not None:
        metadata["shard"] = {"index": shard[0], "total": shard[1]}

    # Provenance (Gate M5): record the model-env torch/CUDA and the repo commit,
    # so each bundle carries them, not just the GPU/driver capture. Best-effort.
    if args.python is not None:
        metadata["provenance"] = _capture_provenance(
            args.python, getattr(args, "repo", None)
        )

    try:
        _run_dir, summary = run_suite(
            backend,
            suite=args.suite,
            run_root=args.run_root,
            model_name=model_name,
            metadata=metadata,
            cold_start_storage=args.cold_start_storage,
            gpu_price_usd_per_hour=args.gpu_price_usd_per_hour,
            vendor_claim=args.vendor_claim,
            resume_dir=args.resume,
            shard=shard,
        )
    finally:
        backend.close()
    print(json.dumps(summary, indent=2, sort_keys=True))
    # Human-friendly pointer to the actual report file (not just the dir).
    if "report_html" in summary:
        print(f"\nreport: {summary['report_html']}", file=sys.stderr)
    return 0 if summary["failures"] == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "world":
            return _world_command(args)

        if args.command == "verify":
            return cmd_verify(args)

        if args.command == "recompute":
            return cmd_recompute(args)

        if args.command == "version":
            print(liveframe.__version__)
            return 0

        if args.command == "gate":
            artifact = (
                load_gate(args.path)
                if args.gate_command == "verify" and args.path is not None
                else load_bundled_gate()
            )
            print(
                json.dumps(
                    {
                        "gate_id": artifact.gate_id,
                        "digest": artifact.digest,
                        "verified": True,
                        "source": str(artifact.source),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "evaluate":
            try:
                from liveframe.evaluator import EvaluationError, evaluate_native_proof
            except ModuleNotFoundError as exc:
                raise _ResearchCommandUnavailable("evaluate") from exc
            raw = json.loads(args.evidence.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise EvaluationError("proof evidence must be a JSON object")
            try:
                proof = evaluate_native_proof(raw)
            except EvaluationError as exc:
                raise ValueError(str(exc)) from exc
            print(json.dumps(proof.as_dict(), indent=2, sort_keys=True))
            return 0

        if args.command == "manifest":
            try:
                from liveframe.manifests import ManifestError, load_upstream_manifest
            except ModuleNotFoundError as exc:
                raise _ResearchCommandUnavailable("manifest") from exc
            try:
                upstream = load_upstream_manifest(args.path)
                if args.require_resolved:
                    upstream.require_resolved()
            except ManifestError as exc:
                raise ValueError(str(exc)) from exc
            print(
                json.dumps(
                    {
                        "schema": upstream.schema,
                        "status": upstream.status,
                        "unresolved_ids": upstream.unresolved_ids,
                        "pins_resolved": upstream.status in {"pinned", "frozen"}
                        and not upstream.unresolved_ids,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "doctor":
            readiness = mobile_readiness()
            print(
                json.dumps(
                    {
                        "mobile_ready": readiness.ready,
                        "host": collect_host_provenance(),
                        "required_manual_steps": [
                            name
                            for name, present in {
                                "install_tailscale_and_authenticate": readiness.tailscale,
                                "install_tmux": readiness.tmux,
                            }.items()
                            if not present
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "convert-cfpp":
            try:
                from liveframe.converters import execute_cfpp_conversion
                from liveframe.converters.cfpp_to_mlx import ConversionError
            except ModuleNotFoundError as exc:
                raise _ResearchCommandUnavailable("convert-cfpp") from exc
            try:
                metadata = execute_cfpp_conversion(
                    args.checkpoint,
                    args.stock_mlx_dir,
                    args.output_dir,
                    expected_source_sha256=args.expected_sha256,
                    target_dtype=args.dtype,
                )
            except ConversionError as exc:
                raise ValueError(str(exc)) from exc
            print(metadata)
            return 0

        if args.command == "qualify":
            profile = WORKLOAD_PROFILES[args.profile]
            duration = args.duration if args.duration is not None else profile.min_playback_seconds
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("duration must be finite and positive")
            request = GenerationRequest(
                prompt=args.prompt,
                native_frames=math.ceil(duration * args.native_fps),
                native_fps=args.native_fps,
                width=args.width,
                height=args.height,
                seed=args.seed,
            )
            backend = create_backend(args.backend)
            run = qualify(backend, profile, request)
            ledger = build_qualification_ledger(backend.capabilities, run)
            if args.output is not None:
                write_qualification_ledger(args.output, ledger)
            print(json.dumps(ledger, indent=2, sort_keys=True))
            return 0

        if args.command == "qualification":
            ledger = load_qualification_ledger(args.ledger)
            print(
                json.dumps(
                    {
                        "schema": ledger["schema"],
                        "ledger_digest": ledger["ledger_digest"],
                        "verified": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        try:
            from liveframe.runs import RunError, RunStore
        except ModuleNotFoundError as exc:
            raise _ResearchCommandUnavailable(args.command) from exc
        try:
            store = RunStore(root=args.home)
            if args.command == "submit":
                command = _clean_remainder(args.run_command)
                result = store.submit(command, idempotency_key=args.idempotency_key)
            elif args.command == "status":
                result = store.status(args.run_id)
            elif args.command == "logs":
                print(store.logs(args.run_id, stream=args.stream, lines=args.lines))
                return 0
            elif args.command == "cancel":
                result = store.cancel(args.run_id)
            elif args.command == "artifacts":
                result = {"run_id": args.run_id, "artifacts": store.artifacts(args.run_id)}
            else:
                raise AssertionError(f"unhandled command: {args.command}")
        except RunError as exc:
            raise ValueError(str(exc)) from exc
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        BackendError,
        GateError,
        OSError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(f"liveframe: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
