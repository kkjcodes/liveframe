"""CPU-verifiable contract tests for the M5 reference adapters.

The generation internals need a GPU + weights and are validated in the M5
session; these tests pin the parts that must be correct WITHOUT the model:
capability declarations, the territorial license notice, pins lockfiles, and
the CLI/doctor/fetch wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from liveframe.world.adapters.hy_worldplay import LICENSE_NOTICE, HYWorldPlayAdapter
from liveframe.world.adapters.matrix_game3 import MatrixGame3Adapter

ADAPTERS = Path(__file__).resolve().parents[1] / "src/liveframe/world/adapters"


def test_matrix_game3_capabilities_match_upstream_facts() -> None:
    caps = MatrixGame3Adapter().capabilities()
    assert caps["name"] == "matrix-game-3"
    assert caps["interaction_mode"] == "interactive"
    assert caps["native_fps"] == 17.0  # upstream export rate, not MG2's 12
    assert caps["chunk_frames_first"] == 57
    assert caps["chunk_frames"] == 40
    assert "Apache-2.0" in caps["license_notice"]


def test_matrix_game3_step_requires_load() -> None:
    # MG3 is validated (step() drives the real model, GPU-validated 2026-10-15);
    # the CPU-testable contract is that it refuses to step before load().
    adapter = MatrixGame3Adapter()
    with pytest.raises(AssertionError, match="step before load"):
        next(adapter.step(["W"]))


def test_hy_worldplay_is_batch_mode_with_territorial_license() -> None:
    caps = HYWorldPlayAdapter().capabilities()
    assert caps["name"] == "hy-worldplay"
    # Released VAE decode is batch-at-end; declared honestly, not faked as live.
    assert caps["interaction_mode"] == "batch"
    assert caps["native_fps"] == 24.0
    assert caps["min_vram_gb"] == 72.0
    for region in ("EU", "UK", "South Korea"):
        assert region in LICENSE_NOTICE
    assert caps["license_notice"] == LICENSE_NOTICE


def test_hy_worldplay_run_script_requires_load() -> None:
    # HY is batch mode; run_script is implemented (validated vs source), but
    # must refuse to run before load() — the CPU-testable contract.
    adapter = HYWorldPlayAdapter()
    with pytest.raises(AssertionError, match="run_script before load"):
        next(adapter.run_script(object()))


@pytest.mark.parametrize(
    "name,validated",
    [("matrix_game3", True), ("hy_worldplay", True)],
)
def test_pins_lockfiles_parse(name: str, validated: bool) -> None:
    payload = json.loads((ADAPTERS / f"{name}.pins.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "liveframe.world.adapter-pins.v1"
    assert payload["pins"]  # non-empty
    # MG3 is GPU-validated (verified date set); HY is still authored-not-validated.
    assert (payload["verified"] is not None) == validated


def test_cli_fetch_prints_hy_license_before_download(capsys: pytest.CaptureFixture) -> None:
    from liveframe.cli import main

    rc = main(
        ["world", "fetch", "hy-worldplay", "--local-dir", "/tmp/x", "--print-only"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "DOES NOT APPLY" in out or "does not apply" in out.lower()
    assert "tencent/HY-WorldPlay" in out


def test_cli_fetch_rejects_unknown_model(capsys: pytest.CaptureFixture) -> None:
    from liveframe.cli import main

    rc = main(["world", "fetch", "nope", "--local-dir", "/tmp/x", "--print-only"])
    assert rc != 0
    assert "unknown reference model" in capsys.readouterr().err


def test_doctor_hy_worldplay_surfaces_license(capsys: pytest.CaptureFixture) -> None:
    # Probe with this interpreter (imports will mostly fail — that's fine; we
    # only assert the license notice is the first surfaced check).
    import sys

    from liveframe.cli import main

    main(
        [
            "world",
            "doctor",
            "hy-worldplay",
            "--python",
            sys.executable,
            "--weights",
            "/nonexistent",
        ]
    )
    out = capsys.readouterr().out
    assert "license-notice" in out
    assert "South Korea" in out
