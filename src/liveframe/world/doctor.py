"""Environment doctor checks for world-model adapters.

Each check returns a structured result (``check``, ``passed``, ``detail``,
``fix``) and never raises: a doctor run must always complete and print every
failure with its exact fix command (spec section 11). Checks that probe a
model environment take the interpreter path explicitly so they work across
the bridge boundary.

The check set is seeded from the M2 bring-up failure library
(agent/world/flywheel/failures/): every environment failure hit during
Matrix-Game 2.0 bring-up has a corresponding check here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

CheckResult = dict[str, Any]


def _result(check: str, passed: bool, detail: str, fix: str | None = None) -> CheckResult:
    return {"check": check, "passed": passed, "detail": detail, "fix": fix}


def _probe(python: str | Path, code: str, timeout: float = 120.0) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    if completed.returncode == 0:
        return True, completed.stdout.strip()
    tail = (completed.stderr or completed.stdout).strip().splitlines()
    return False, tail[-1] if tail else f"exit code {completed.returncode}"


def check_hf_cli() -> CheckResult:
    """F-0001: huggingface_hub >= 2.0 renamed ``huggingface-cli`` to ``hf``."""

    if shutil.which("hf"):
        return _result("hf-cli", True, "`hf` CLI found")
    if shutil.which("huggingface-cli"):
        return _result(
            "hf-cli",
            True,
            "legacy `huggingface-cli` found (pre-2.0 huggingface_hub)",
        )
    return _result(
        "hf-cli",
        False,
        "no Hugging Face CLI on PATH",
        fix="pip install 'huggingface_hub[cli]' and use `hf download <repo>` "
        "(the command is `hf`, not `huggingface-cli`, since huggingface_hub 2.0)",
    )


def check_path_supports_chmod(path: Path) -> CheckResult:
    """F-0002: network volumes may forbid chmod, breaking git clone and venvs."""

    probe_dir = Path(path)
    if not probe_dir.is_dir():
        return _result(
            "chmod-support",
            False,
            f"{probe_dir} is not a directory",
            fix="create the directory or point at an existing one",
        )
    try:
        with tempfile.NamedTemporaryFile(dir=probe_dir, delete=True) as handle:
            os.chmod(handle.name, 0o644)
    except PermissionError as error:
        return _result(
            "chmod-support",
            False,
            f"chmod refused under {probe_dir}: {error}",
            fix="put git clones and virtualenvs on the container disk (e.g. /opt), "
            "keep only weights/outputs on the network volume",
        )
    except OSError as error:
        return _result("chmod-support", False, f"cannot probe {probe_dir}: {error}")
    return _result("chmod-support", True, f"chmod works under {probe_dir}")


def check_model_env_imports(
    python: str | Path, modules: tuple[str, ...] = ("torch", "flash_attn")
) -> list[CheckResult]:
    """F-0005 (and friends): hard imports the adapter path needs must resolve."""

    fixes = {
        "flash_attn": "pip install flash-attn==2.7.4.post1 --no-build-isolation "
        "(wan/modules/action_module.py imports it unguarded)",
        "torch": "install the model env per the adapter docstring",
    }
    results = []
    for module in modules:
        ok, detail = _probe(python, f"import {module}; print({module!r}+' ok')")
        results.append(
            _result(
                f"import-{module}",
                ok,
                detail,
                None if ok else fixes.get(module, f"pip install {module}"),
            )
        )
    return results


def check_diffusers_torch_compat(python: str | Path) -> CheckResult:
    """F-0003/F-0004: too-new diffusers/transformers/torchao vs the pinned torch.

    Probes the exact import surface that broke during M2 bring-up
    (``diffusers.loaders`` -> pytree ``register_constant`` on torch < 2.11).
    """

    ok, detail = _probe(
        python,
        "from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin; "
        "print('diffusers loaders ok')",
    )
    if ok:
        return _result("diffusers-torch-compat", True, detail)
    fix = (
        "pin vendor-era versions for the installed torch: for torch 2.6 use "
        "`pip install diffusers==0.32.2 transformers==4.49.0` and "
        "`pip uninstall -y torchao` (latest torchao/diffusers need torch>=2.11 "
        "pytree APIs)"
    )
    return _result("diffusers-torch-compat", False, detail, fix)


def check_adapter_pins(python: str | Path, pins_file: Path) -> list[CheckResult]:
    """U-0001 (F-0003/F-0004): model-env versions must match the adapter lockfile.

    Each adapter ships a ``<name>.pins.json`` with the exact versions verified
    during bring-up plus packages that must be ABSENT. Drift prints the exact
    install/uninstall command instead of failing later inside the model.
    """

    import re

    def canonical(name: str) -> str:
        # PEP 503: dist-info may say flash_attn or flash-attn; treat as equal.
        return re.sub(r"[-_.]+", "-", name).lower()

    payload = json.loads(Path(pins_file).read_text(encoding="utf-8"))
    pins: dict[str, str] = payload.get("pins", {})
    forbidden: dict[str, str] = payload.get("forbidden", {})
    ok, detail = _probe(
        python,
        # Broken dist-info can yield a None Name; skip those, don't crash.
        "import json, importlib.metadata as m; "
        "print(json.dumps({d.metadata['Name']: d.version "
        "for d in m.distributions() if d.metadata['Name']}))",
    )
    if not ok:
        return [_result("adapter-pins", False, f"could not list model env: {detail}")]
    installed = {canonical(name): version for name, version in json.loads(detail).items()}
    results: list[CheckResult] = []
    for package, wanted in sorted(pins.items()):
        actual = installed.get(canonical(package))
        if actual == wanted:
            results.append(_result(f"pin-{package}", True, f"{package}=={actual}"))
        else:
            results.append(
                _result(
                    f"pin-{package}",
                    False,
                    f"{package}: installed {actual or 'missing'}, lockfile {wanted}",
                    fix=f"pip install {package}=={wanted}",
                )
            )
    for package, reason in sorted(forbidden.items()):
        actual = installed.get(canonical(package))
        if actual is None:
            results.append(_result(f"absent-{package}", True, f"{package} not installed"))
        else:
            results.append(
                _result(
                    f"absent-{package}",
                    False,
                    f"{package} {actual} installed but must be absent: {reason}",
                    fix=f"pip uninstall -y {package}",
                )
            )
    return results


def check_weights_present(weights_dir: Path, required: tuple[str, ...]) -> CheckResult:
    missing = [name for name in required if not (Path(weights_dir) / name).exists()]
    if missing:
        return _result(
            "weights-present",
            False,
            f"missing under {weights_dir}: {missing}",
            fix=f"hf download <repo> --local-dir {weights_dir}",
        )
    return _result("weights-present", True, f"all {len(required)} required files present")


def run_matrix_game2_doctor(
    *,
    python: str | Path,
    repo: Path,
    weights: Path,
) -> list[CheckResult]:
    """The full Matrix-Game 2.0 preflight (every M2 bring-up failure covered)."""

    results = [check_hf_cli()]
    results.append(check_path_supports_chmod(Path(repo).parent))
    results.append(
        check_weights_present(
            weights,
            ("Wan2.1_VAE.pth", "base_distilled_model/base_distill.safetensors"),
        )
    )
    # cv2/easydict/ftfy: upstream pipeline imports discovered in the M4
    # bring-up — absent from the vendor's obvious requirements, crash at load.
    results.extend(
        check_model_env_imports(
            python, modules=("torch", "flash_attn", "cv2", "easydict", "ftfy")
        )
    )
    results.append(check_diffusers_torch_compat(python))
    results.extend(
        check_adapter_pins(
            python, Path(__file__).parent / "adapters" / "matrix_game2.pins.json"
        )
    )
    return results


def run_matrix_game3_doctor(
    *, python: str | Path, repo: Path, weights: Path
) -> list[CheckResult]:
    """Matrix-Game 3.0 preflight (adapter authored from source; pins UNVERIFIED)."""

    results = [check_hf_cli()]
    results.append(check_path_supports_chmod(Path(repo).parent))
    results.append(
        check_weights_present(
            weights,
            ("base_distilled_model/diffusion_pytorch_model.safetensors",),
        )
    )
    results.extend(
        check_model_env_imports(
            python, modules=("torch", "cv2", "easydict", "ftfy", "decord")
        )
    )
    results.extend(
        check_adapter_pins(
            python, Path(__file__).parent / "adapters" / "matrix_game3.pins.json"
        )
    )
    return results


def run_hy_worldplay_doctor(
    *, python: str | Path, repo: Path, weights: Path
) -> list[CheckResult]:
    """HY-WorldPlay preflight. ALWAYS surfaces the territorial license notice."""

    from liveframe.world.adapters.hy_worldplay import LICENSE_NOTICE

    results = [
        _result("license-notice", True, LICENSE_NOTICE),
        check_hf_cli(),
        check_path_supports_chmod(Path(repo).parent),
        check_weights_present(
            weights, ("ar_distilled_action_model/diffusion_pytorch_model.safetensors",)
        ),
    ]
    results.extend(
        check_model_env_imports(
            python, modules=("torch", "diffusers", "transformers", "qwen_vl_utils")
        )
    )
    results.extend(
        check_adapter_pins(
            python, Path(__file__).parent / "adapters" / "hy_worldplay.pins.json"
        )
    )
    return results
