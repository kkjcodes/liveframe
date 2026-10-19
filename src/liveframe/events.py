"""Versioned append-only events used by runs and reports."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVENT_SCHEMA = "liveframe.event.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class Event:
    event: str
    run_id: str
    timestamp: str
    data: Mapping[str, Any]
    schema: str = EVENT_SCHEMA

    @classmethod
    def create(cls, event: str, run_id: str, **data: Any) -> Event:
        return cls(event=event, run_id=run_id, timestamp=utc_now(), data=data)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class EventWriter:
    """Thread-safe NDJSON writer with a durability flush per control event."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Event) -> None:
        line = event.to_json() + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
