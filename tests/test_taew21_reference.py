from __future__ import annotations

import pytest

from liveframe.taew21_reference import (
    EXPECTED_LAYER_CALLS,
    parse_activation_probes,
)

FROZEN_ANCHORS = [
    "decoder.0.calls[0,10,20]",
    "decoder.5.calls[0,10,20]",
    "decoder.7.calls[0,10,20]",
    "decoder.11.calls[0,10,20]",
    "decoder.13.calls[0,10,20]",
    "decoder.17.calls[0,20,41]",
    "decoder.19.calls[0,20,41]",
    "decoder.22.calls[0,41,83]",
]


def test_frozen_taew21_probes_cover_exact_causal_schedule() -> None:
    probes = parse_activation_probes(FROZEN_ANCHORS)

    assert {probe.layer for probe in probes} == set(EXPECTED_LAYER_CALLS)
    assert sum(len(probe.calls) for probe in probes) == 24


def test_taew21_probes_reject_out_of_schedule_or_missing_layer() -> None:
    invalid = list(FROZEN_ANCHORS)
    invalid[-1] = "decoder.22.calls[0,41,84]"
    with pytest.raises(ValueError, match="exceeds layer schedule"):
        parse_activation_probes(invalid)

    with pytest.raises(ValueError, match="do not cover"):
        parse_activation_probes(FROZEN_ANCHORS[:-1])