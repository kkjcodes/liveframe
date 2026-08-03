from __future__ import annotations

import pytest

from liveframe.memory import MemoryAllocation, MemoryBudget, MemoryMode

GIB = 1024**3


def allocation(
    name: str, peak_gib: int, *, anchor: bool = False, policy: str = "retain"
) -> MemoryAllocation:
    size = peak_gib * GIB
    return MemoryAllocation(name, size, size, size, "run", policy, anchor)


def test_36gb_budget_uses_measured_process_ceiling() -> None:
    budget = MemoryBudget(
        capacity_bytes=36 * GIB,
        os_reserve_bytes=6 * GIB,
        mode=MemoryMode.RESIDENT_TEXT,
        allocations=(allocation("weights", 12), allocation("kv", 8), allocation("vae", 4)),
    )

    budget.assert_safe()
    assert budget.process_ceiling_bytes == 30 * GIB
    assert budget.fits


def test_budget_rejects_peak_over_ceiling() -> None:
    budget = MemoryBudget(
        capacity_bytes=36 * GIB,
        os_reserve_bytes=6 * GIB,
        mode=MemoryMode.RESIDENT_TEXT,
        allocations=(allocation("weights", 20), allocation("activations", 12)),
    )

    with pytest.raises(MemoryError, match="exceeds"):
        budget.assert_safe()


def test_semantic_anchor_cannot_be_silently_evicted() -> None:
    budget = MemoryBudget(
        capacity_bytes=36 * GIB,
        os_reserve_bytes=6 * GIB,
        mode=MemoryMode.EVICTED_TEXT,
        allocations=(allocation("sink anchors", 2, anchor=True, policy="evict under pressure"),),
    )

    with pytest.raises(MemoryError, match="semantic anchors"):
        budget.assert_safe()
