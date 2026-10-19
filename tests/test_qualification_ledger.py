from __future__ import annotations

from copy import deepcopy

import pytest

from liveframe.backends.synthetic import SyntheticBackend
from liveframe.engine import GenerationRequest
from liveframe.gates import contract_digest
from liveframe.qualification import INTERACTIVE_480P, qualify
from liveframe.qualification_ledger import (
    build_qualification_ledger,
    verify_qualification_ledger,
)


class StepClock:
    def __init__(self, step: float = 0.25) -> None:
        self.value = -step
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


def _ledger() -> dict[str, object]:
    backend = SyntheticBackend(chunk_size=480)
    run = qualify(
        backend,
        INTERACTIVE_480P,
        GenerationRequest("prompt", native_frames=960, seed=7),
        clock=StepClock(),
    )
    return build_qualification_ledger(
        backend.capabilities,
        run,
        created_at="2026-10-06T00:00:00.000000Z",
        host_provenance={"system": "test"},
    )


def test_qualification_ledger_verifies_digest_and_arithmetic() -> None:
    verify_qualification_ledger(_ledger())


def test_qualification_ledger_rejects_tampering() -> None:
    ledger = deepcopy(_ledger())
    metrics = ledger["metrics"]
    assert isinstance(metrics, dict)
    metrics["generated_fps"] = 999.0

    with pytest.raises(ValueError, match="digest mismatch"):
        verify_qualification_ledger(ledger)


def test_qualification_ledger_recomputes_gate_verdicts() -> None:
    ledger = deepcopy(_ledger())
    qualification = ledger["qualification"]
    assert isinstance(qualification, dict)
    qualification["verdict"] = "pass"
    unsigned = {key: value for key, value in ledger.items() if key != "ledger_digest"}
    ledger["ledger_digest"] = contract_digest(unsigned)

    with pytest.raises(ValueError, match="gate recomputation mismatch"):
        verify_qualification_ledger(ledger)