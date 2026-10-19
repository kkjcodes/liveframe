"""Exact activation probe specification for TAEW2.1 port parity."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ANCHOR = re.compile(r"^decoder\.(\d+)\.calls\[([0-9,]+)\]$")
EXPECTED_LAYER_CALLS = {
    0: 21,
    5: 21,
    7: 21,
    11: 21,
    13: 21,
    17: 42,
    19: 42,
    22: 84,
}


@dataclass(frozen=True, slots=True)
class ActivationProbe:
    layer: int
    calls: tuple[int, ...]


def parse_activation_probes(values: list[str]) -> tuple[ActivationProbe, ...]:
    probes: list[ActivationProbe] = []
    layers: set[int] = set()
    for value in values:
        match = _ANCHOR.fullmatch(value)
        if match is None:
            raise ValueError(f"invalid TAEW2.1 activation anchor: {value}")
        layer = int(match.group(1))
        calls = tuple(int(item) for item in match.group(2).split(","))
        if layer not in EXPECTED_LAYER_CALLS:
            raise ValueError(f"unsupported TAEW2.1 activation layer: {layer}")
        if layer in layers or not calls or tuple(sorted(set(calls))) != calls:
            raise ValueError(f"duplicate layer or calls in TAEW2.1 anchor: {value}")
        if calls[-1] >= EXPECTED_LAYER_CALLS[layer]:
            raise ValueError(f"TAEW2.1 anchor call exceeds layer schedule: {value}")
        layers.add(layer)
        probes.append(ActivationProbe(layer, calls))
    if layers != set(EXPECTED_LAYER_CALLS):
        raise ValueError("TAEW2.1 activation anchors do not cover the frozen layer set")
    return tuple(probes)