"""Runtime-neutral planning for the CF++ causal self-attention KV cache."""

from __future__ import annotations

from dataclasses import dataclass


class CachePlanError(ValueError):
    """Raised when a causal cache update would violate sequence accounting."""


@dataclass(frozen=True, slots=True)
class CacheUpdatePlan:
    """Indices for one append, same-position overwrite, or rolling update."""

    current_start: int
    current_end: int
    local_start: int
    local_end: int
    global_end: int
    evicted_tokens: int
    rolled_tokens: int
    sink_tokens: int
    attention_start: int

    @property
    def is_overwrite(self) -> bool:
        return self.current_end == self.global_end and self.current_start < self.global_end


def plan_cache_update(
    *,
    current_start: int,
    token_count: int,
    previous_global_end: int,
    previous_local_end: int,
    capacity: int,
    max_attention_size: int,
    local_attention: bool = False,
    sink_tokens: int = 0,
) -> CacheUpdatePlan:
    """Mirror upstream CF++ cache indices without allocating runtime tensors.

    Repeated denoising calls at the same global position overwrite the current
    block. A clean-context refresh therefore replaces the speculative K/V at
    that position rather than extending the cache.
    """

    values = {
        "current_start": current_start,
        "token_count": token_count,
        "previous_global_end": previous_global_end,
        "previous_local_end": previous_local_end,
        "capacity": capacity,
        "max_attention_size": max_attention_size,
        "sink_tokens": sink_tokens,
    }
    if any(value < 0 for value in values.values()):
        raise CachePlanError("cache indices and sizes must be non-negative")
    if token_count == 0:
        raise CachePlanError("token_count must be positive")
    if capacity == 0 or max_attention_size == 0:
        raise CachePlanError("cache and attention capacities must be positive")
    if previous_local_end > capacity:
        raise CachePlanError("previous local cache end exceeds capacity")
    if sink_tokens > capacity:
        raise CachePlanError("sink tokens exceed cache capacity")
    if current_start > previous_global_end:
        raise CachePlanError("causal cache updates cannot leave a global token gap")

    current_end = current_start + token_count
    if current_end < previous_global_end:
        raise CachePlanError("cache update cannot target an expired historical suffix")

    advances = current_end > previous_global_end
    evicted_tokens = 0
    rolled_tokens = 0
    if local_attention and advances and token_count + previous_local_end > capacity:
        evicted_tokens = token_count + previous_local_end - capacity
        rolled_tokens = previous_local_end - evicted_tokens - sink_tokens
        if rolled_tokens < 0:
            raise CachePlanError("cache capacity cannot preserve the configured sink")

    local_end = (
        previous_local_end
        + current_end
        - previous_global_end
        - evicted_tokens
    )
    local_start = local_end - token_count
    if local_start < 0 or local_end > capacity:
        raise CachePlanError("cache update exceeds allocated local storage")

    global_end = current_end
    attention_start = max(0, local_end - max_attention_size)
    return CacheUpdatePlan(
        current_start=current_start,
        current_end=current_end,
        local_start=local_start,
        local_end=local_end,
        global_end=global_end,
        evicted_tokens=evicted_tokens,
        rolled_tokens=rolled_tokens,
        sink_tokens=sink_tokens,
        attention_start=attention_start,
    )