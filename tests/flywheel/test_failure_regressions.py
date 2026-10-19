"""Regression reruns for every failure-library entry (addendum A section 4).

CPU-checkable regressions run here in CI. GPU-only checks (probing the real
model environment) are marked ``gpu`` and run at the start of every GPU
session before any measurement; locally they skip unless MG2_PYTHON points at
a model-env interpreter.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from liveframe.world import doctor

FAILURES_DIR = (
    Path(__file__).resolve().parents[2] / "agent" / "world" / "flywheel" / "failures"
)


def test_every_failure_entry_names_a_regression_check() -> None:
    entries = sorted(FAILURES_DIR.glob("F-*.md"))
    assert len(entries) >= 5
    for entry in entries:
        text = entry.read_text(encoding="utf-8")
        assert "Regression check:" in text, entry.name
        assert "Detection:" in text, entry.name
        assert "Recurrences:" in text, entry.name


def test_f0001_hf_cli_rename(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    # Neither CLI present: the fix must teach the `hf` rename.
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    result = doctor.check_hf_cli()
    assert result["passed"] is False
    assert "hf download" in result["fix"]
    assert "not `huggingface-cli`" in result["fix"]

    # New-style CLI present.
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/hf" if name == "hf" else None)
    assert doctor.check_hf_cli()["passed"] is True


def test_f0002_chmod_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ok = doctor.check_path_supports_chmod(tmp_path)
    assert ok["passed"] is True

    def refuse(_path: object, _mode: int) -> None:
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr(os, "chmod", refuse)
    refused = doctor.check_path_supports_chmod(tmp_path)
    assert refused["passed"] is False
    assert "container disk" in refused["fix"]


def _fake_probe_failure(message: str):
    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["python"], returncode=1, stdout="", stderr=message + "\n"
        )

    return fake_run


def test_f0003_diffusers_skew(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_probe_failure(
            "AttributeError: module 'torch.utils._pytree' has no attribute "
            "'register_constant'"
        ),
    )
    result = doctor.check_diffusers_torch_compat("python")
    assert result["passed"] is False
    assert "diffusers==0.32.2" in result["fix"]
    assert "transformers==4.49.0" in result["fix"]


def test_f0004_torchao_fix_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_probe_failure("AttributeError: ... register_constant (via torchao)"),
    )
    result = doctor.check_diffusers_torch_compat("python")
    assert result["passed"] is False
    assert "pip uninstall -y torchao" in result["fix"]


def test_f0005_flash_attn_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_probe_failure("ModuleNotFoundError: No module named 'flash_attn'"),
    )
    results = doctor.check_model_env_imports("python", modules=("flash_attn",))
    assert results[0]["passed"] is False
    assert "flash-attn==2.7.4.post1" in results[0]["fix"]
    assert "action_module.py" in results[0]["fix"]


def test_u0001_adapter_pins_detect_drift_and_forbidden_packages(
    tmp_path: Path,
) -> None:
    """U-0001: pin drift and must-be-absent packages print exact fix commands."""

    import json
    import sys

    import pytest as pytest_module

    pins_file = tmp_path / "pins.json"
    pins_file.write_text(
        json.dumps(
            {
                "schema": "liveframe.world.adapter-pins.v1",
                "adapter": "test",
                "pins": {
                    "pytest": pytest_module.__version__,  # matches
                    "numpy": "0.0.1",  # drifts
                    "not-installed-pkg": "1.0.0",  # missing
                },
                "forbidden": {"pytest": "pretend it breaks imports"},
            }
        ),
        encoding="utf-8",
    )
    results = {
        result["check"]: result
        for result in doctor.check_adapter_pins(sys.executable, pins_file)
    }
    assert results["pin-pytest"]["passed"] is True
    assert results["pin-numpy"]["passed"] is False
    # PEP 503: a pins key with hyphens must match dist-info that says
    # underscores (the flash-attn/flash_attn case).
    import importlib.metadata

    te_version = importlib.metadata.version("typing_extensions")
    pins_file.write_text(
        json.dumps({"pins": {"typing-extensions": te_version}}), encoding="utf-8"
    )
    (canonical,) = doctor.check_adapter_pins(sys.executable, pins_file)
    assert canonical["passed"] is True, canonical
    assert results["pin-numpy"]["fix"] == "pip install numpy==0.0.1"
    assert results["pin-not-installed-pkg"]["passed"] is False
    assert "missing" in results["pin-not-installed-pkg"]["detail"]
    assert results["absent-pytest"]["passed"] is False
    assert results["absent-pytest"]["fix"] == "pip uninstall -y pytest"


def test_u0001_matrix_game2_lockfile_is_shipped_and_wired() -> None:
    import json

    pins_file = (
        Path(doctor.__file__).parent / "adapters" / "matrix_game2.pins.json"
    )
    payload = json.loads(pins_file.read_text(encoding="utf-8"))
    assert payload["schema"] == "liveframe.world.adapter-pins.v1"
    # The exact versions verified during M2 bring-up (gate-m2.md).
    assert payload["pins"]["diffusers"] == "0.32.2"
    assert payload["pins"]["transformers"] == "4.49.0"
    assert payload["pins"]["flash-attn"] == "2.7.4.post1"
    assert "torchao" in payload["forbidden"]


@pytest.mark.gpu
def test_real_model_env_doctor() -> None:
    """Run before any GPU measurement: the real model env must pass every check."""

    python = os.environ.get("MG2_PYTHON")
    if not python:
        pytest.skip("MG2_PYTHON not set; GPU-session-only check")
    results = doctor.run_matrix_game2_doctor(
        python=python,
        repo=Path(os.environ.get("MG2_REPO", "/opt/code/Matrix-Game/Matrix-Game-2")),
        weights=Path(os.environ.get("MG2_WEIGHTS", "/workspace/models/Matrix-Game-2.0")),
    )
    failures = [result for result in results if not result["passed"]]
    assert not failures, failures
