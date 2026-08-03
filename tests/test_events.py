from __future__ import annotations

import json
from pathlib import Path

from liveframe.events import EVENT_SCHEMA, Event, EventWriter


def test_event_writer_appends_versioned_ndjson(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path)
    writer.append(Event.create("frame.generated", "run-1", pixel_frame=1))
    writer.append(Event.create("frame.displayed", "run-1", pixel_frame=1))

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["frame.generated", "frame.displayed"]
    assert all(event["schema"] == EVENT_SCHEMA for event in events)
