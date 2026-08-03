from __future__ import annotations

import pytest

from liveframe.prompting import PromptCoordinator, PromptQueueError


def test_prompt_activates_only_at_boundary() -> None:
    coordinator = PromptCoordinator()
    coordinator.submit("first scene", 0.0)
    encoding = coordinator.start_encoding(0.1)
    completed = coordinator.finish_encoding(encoding, 0.5)

    assert coordinator.active is None
    activated = coordinator.activate_at_boundary(1.0)
    assert activated is not None
    assert activated.version == completed.version
    assert activated.activated_at == 1.0


def test_newest_completed_prompt_wins_boundary() -> None:
    coordinator = PromptCoordinator(embedding_capacity=2)
    coordinator.submit("old", 0.0)
    coordinator.submit("new", 0.1)
    old = coordinator.start_encoding(0.2)
    coordinator.finish_encoding(old, 0.3)
    new = coordinator.start_encoding(0.4)
    coordinator.finish_encoding(new, 0.5)

    activated = coordinator.activate_at_boundary(1.0)

    assert activated is not None
    assert activated.prompt == "new"
    assert [item.prompt for item in coordinator.superseded] == ["old"]
    assert coordinator.superseded[0].superseded


def test_prompt_queue_overflow_is_explicit() -> None:
    coordinator = PromptCoordinator(request_capacity=1)
    coordinator.submit("first", 0.0)

    with pytest.raises(PromptQueueError, match="overflow"):
        coordinator.submit("second", 0.1)
