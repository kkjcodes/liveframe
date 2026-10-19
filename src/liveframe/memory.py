"""Measured unified-memory budget model with explicit residency policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemoryMode(str, Enum):
    RESIDENT_TEXT = "resident-text"
    EVICTED_TEXT = "evicted-text"
    DUAL_STATE_REFRESH = "dual-state-refresh"


@dataclass(frozen=True, slots=True)
class MemoryAllocation:
    name: str
    theoretical_bytes: int
    resident_bytes: int
    peak_bytes: int
    lifetime: str
    eviction_or_recompute: str
    semantic_anchor: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.lifetime or not self.eviction_or_recompute:
            raise ValueError("memory allocation metadata must not be empty")
        if min(self.theoretical_bytes, self.resident_bytes, self.peak_bytes) < 0:
            raise ValueError("memory allocation sizes must be non-negative")
        if self.peak_bytes < self.resident_bytes:
            raise ValueError("peak bytes cannot be below resident bytes")


@dataclass(frozen=True, slots=True)
class MemoryBudget:
    capacity_bytes: int
    os_reserve_bytes: int
    mode: MemoryMode
    allocations: tuple[MemoryAllocation, ...]

    def __post_init__(self) -> None:
        if self.capacity_bytes <= 0 or self.os_reserve_bytes < 0:
            raise ValueError("memory capacity must be positive and reserve non-negative")
        if self.os_reserve_bytes >= self.capacity_bytes:
            raise ValueError("OS reserve must be below total capacity")

    @property
    def process_ceiling_bytes(self) -> int:
        return self.capacity_bytes - self.os_reserve_bytes

    @property
    def peak_bytes(self) -> int:
        return sum(item.peak_bytes for item in self.allocations)

    @property
    def fits(self) -> bool:
        return self.peak_bytes <= self.process_ceiling_bytes

    def assert_safe(self) -> None:
        if not self.fits:
            excess = self.peak_bytes - self.process_ceiling_bytes
            raise MemoryError(f"memory budget exceeds process ceiling by {excess} bytes")
        anchors = [item.name for item in self.allocations if item.semantic_anchor]
        evicted_anchor = any(
            "evict" in item.eviction_or_recompute.lower()
            for item in self.allocations
            if item.semantic_anchor
        )
        if evicted_anchor:
            names = ", ".join(anchors)
            raise MemoryError(f"semantic anchors may not be silently evicted: {names}")
