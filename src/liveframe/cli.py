"""LiveFrame command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from liveframe.converters import execute_cfpp_conversion
from liveframe.converters.cfpp_to_mlx import ConversionError
from liveframe.evaluator import EvaluationError, evaluate_native_proof
from liveframe.gates import GateError, load_bundled_gate, load_gate
from liveframe.host import collect_host_provenance, mobile_readiness
from liveframe.manifests import ManifestError, load_upstream_manifest
from liveframe.runs import RunError, RunStore


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
    return parser


def _clean_remainder(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
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
            raw = json.loads(args.evidence.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise EvaluationError("proof evidence must be a JSON object")
            print(json.dumps(evaluate_native_proof(raw).as_dict(), indent=2, sort_keys=True))
            return 0

        if args.command == "manifest":
            upstream = load_upstream_manifest(args.path)
            if args.require_resolved:
                upstream.require_resolved()
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
            metadata = execute_cfpp_conversion(
                args.checkpoint,
                args.stock_mlx_dir,
                args.output_dir,
                expected_source_sha256=args.expected_sha256,
                target_dtype=args.dtype,
            )
            print(metadata)
            return 0

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
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        EvaluationError,
        ConversionError,
        GateError,
        ManifestError,
        RunError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"liveframe: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
