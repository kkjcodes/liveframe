from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from liveframe.cli import main


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        artifact = self.root / "artifacts" / "sample.json"
        artifact.parent.mkdir()
        artifact.write_text('{"passed": true}\n', encoding="utf-8")
        digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
        self.ledger = self.root / "artifacts" / "claims.json"
        self.payload = {
            "schema": "liveframe.publication-claims.v1",
            "engineering_speedups": [
                {
                    "id": "speedup",
                    "baseline": {
                        "artifact": "artifacts/sample.json",
                        "sha256": digest,
                        "wall_seconds": 10.0,
                        "vae_seconds": 8.0,
                    },
                    "candidate": {
                        "artifact": "artifacts/sample.json",
                        "sha256": digest,
                        "wall_seconds": 8.0,
                        "vae_seconds": 6.0,
                    },
                    "wall_speedup": 1.25,
                    "wall_reduction_fraction": 0.2,
                    "vae_reduction_fraction": 0.25,
                }
            ],
            "negative_results": [
                {
                    "id": "target-gap",
                    "evidence": {
                        "generated_fps": 2.0,
                        "target_fps": 16.0,
                        "target_gap_factor": 8.0,
                    },
                }
            ],
        }
        self._write_ledger()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_ledger(self) -> None:
        self.ledger.write_text(json.dumps(self.payload), encoding="utf-8")

    def _run(self, *arguments: str) -> tuple[int, str]:
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = main(list(arguments))
        return result, output.getvalue()

    def test_verify_deduplicates_repeated_artifact(self) -> None:
        result, output = self._run(
            "verify", str(self.ledger), "--artifacts-root", str(self.root)
        )
        self.assertEqual(result, 0)
        self.assertIn("all 1 unique digest-bound artifacts verified", output)

    def test_verify_fails_on_digest_mismatch(self) -> None:
        self.payload["engineering_speedups"][0]["baseline"]["sha256"] = (
            "sha256:" + "0" * 64
        )
        self.payload["engineering_speedups"][0]["candidate"]["sha256"] = (
            "sha256:" + "0" * 64
        )
        self._write_ledger()
        result, output = self._run(
            "verify", str(self.ledger), "--artifacts-root", str(self.root)
        )
        self.assertEqual(result, 1)
        self.assertIn("DIGEST MISMATCH", output)

    def test_verify_rejects_path_outside_root(self) -> None:
        claim = self.payload["engineering_speedups"][0]
        claim["baseline"]["artifact"] = "../sample.json"
        claim["candidate"]["artifact"] = "../sample.json"
        self._write_ledger()
        result, output = self._run(
            "verify", str(self.ledger), "--artifacts-root", str(self.root)
        )
        self.assertEqual(result, 1)
        self.assertIn("artifact path must remain below root", output)

    def test_verify_rejects_incomplete_artifact_reference(self) -> None:
        del self.payload["engineering_speedups"][0]["baseline"]["sha256"]
        self._write_ledger()
        result, output = self._run("verify", str(self.ledger))
        self.assertEqual(result, 2)
        self.assertIn("artifact references require", output)

    def test_recompute_checks_all_supported_arithmetic(self) -> None:
        result, output = self._run("recompute", str(self.ledger))
        self.assertEqual(result, 0)
        self.assertEqual(output.count(" ok"), 4)
        self.assertIn("vae_reduction_fraction", output)

    def test_recompute_rejects_empty_ledger(self) -> None:
        self.payload["engineering_speedups"] = []
        self.payload["negative_results"] = []
        self._write_ledger()
        result, output = self._run("recompute", str(self.ledger))
        self.assertEqual(result, 2)
        self.assertIn("no supported arithmetic claims", output)

    def test_rejects_unsupported_schema_without_traceback(self) -> None:
        self.payload["schema"] = "unknown"
        self._write_ledger()
        result, output = self._run("verify", str(self.ledger))
        self.assertEqual(result, 2)
        self.assertIn("unsupported claims ledger schema", output)
        self.assertNotIn("Traceback", output)


if __name__ == "__main__":
    unittest.main()
