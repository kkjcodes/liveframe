from __future__ import annotations

from pathlib import Path

import pytest

from liveframe.world.loader import load_adapter_class, load_backend, parse_adapter_spec
from liveframe.world.runner import run_suite

REPO = Path(__file__).resolve().parents[1]
DRIFT_SPEC = str(REPO / "examples" / "byo_adapter" / "drift_world.py") + ":DriftWorld"


def test_parse_adapter_spec_accepts_files_and_modules() -> None:
    assert parse_adapter_spec("my_adapter.py:MyWorld") == ("my_adapter.py", "MyWorld")
    assert parse_adapter_spec("pkg.module:Cls") == ("pkg.module", "Cls")
    # Windows drive letters survive because only the LAST colon splits.
    assert parse_adapter_spec("C:/models/a.py:World") == ("C:/models/a.py", "World")
    for bad in ("no-colon", ":Cls", "file.py:", "file.py:not an identifier"):
        with pytest.raises(ValueError, match="adapter spec"):
            parse_adapter_spec(bad)


def test_loader_errors_name_the_problem(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="adapter file not found"):
        load_adapter_class(str(tmp_path / "missing.py") + ":X")

    module = tmp_path / "half.py"
    module.write_text(
        "class Half:\n"
        "    def load(self): ...\n"
        "    def close(self): ...\n",
        encoding="utf-8",
    )
    with pytest.raises(ImportError, match="no class named 'Nope'"):
        load_adapter_class(str(module) + ":Nope")
    with pytest.raises(TypeError, match="missing capabilities, reset, step, run_script"):
        load_adapter_class(str(module) + ":Half")


def test_module_spec_loads_the_builtin_synthetic_world() -> None:
    backend = load_backend(
        "liveframe.world.adapters.synthetic:SyntheticWorld", chunk_delay_seconds=0.0
    )
    backend.load()
    assert backend.capabilities.name == "synthetic"


def test_example_adapter_passes_a_full_smoke_suite(tmp_path: Path) -> None:
    """The shipped BYO example must stay runnable end to end, forever."""

    backend = load_backend(DRIFT_SPEC)
    run_dir, summary = run_suite(
        backend,
        suite="smoke",
        run_root=tmp_path,
        model_name="drift-world",
        cold_start_storage="local",
    )
    assert summary["failures"] == 0
    assert (run_dir / "report.html").exists()
