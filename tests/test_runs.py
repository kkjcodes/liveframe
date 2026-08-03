from __future__ import annotations

import os
import sys
from pathlib import Path

from liveframe.runs import RunStore, wait_for_state


def test_detached_run_survives_submitter_and_records_output(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    submitted = store.submit(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; print('mobile-ok'); Path('result.txt').write_text('done')",
        ],
        idempotency_key="same-phone-request",
    )
    run_id = str(submitted["run_id"])
    duplicate = store.submit(["unused"], idempotency_key="same-phone-request")
    assert duplicate["run_id"] == run_id

    finished = wait_for_state(store, run_id, {"completed"})
    assert finished["gate_id"] == "gates.v1"
    assert store.logs(run_id) == "mobile-ok"
    assert store.artifacts(run_id) == [
        {
            "path": "result.txt",
            "bytes": 4,
            "sha256": "sha256:a4c3ed04a95a3da14a9d235c83d868bed7c0f45cf7f3faa751ee8f50598d2211",
        }
    ]
    assert any(event["event"] == "run.completed" for event in store.iter_events(run_id))
    assert any(event["event"] == "control.logs_read" for event in store.iter_events(run_id))
    assert any(event["event"] == "control.artifacts_listed" for event in store.iter_events(run_id))


def test_cooperative_cancel_is_durable(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    submitted = store.submit(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(60)"]
    )
    run_id = str(submitted["run_id"])
    running = wait_for_state(store, run_id, {"running"})
    assert os.path.exists(f"/proc/{running['worker_pid']}") or sys.platform != "linux"

    store.cancel(run_id)
    finished = wait_for_state(store, run_id, {"cancelled"})
    assert finished["cancel_requested_at"]
    events = list(store.iter_events(run_id))
    assert any(event["event"] == "run.cancel_requested" for event in events)
    assert any(event["event"] == "run.cancelled" for event in events)
