from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from liveframe.world.actions import ActionScript
from liveframe.world.bridge.host import BridgedWorld
from liveframe.world.session import WorldCase, run_case

ADAPTER_FILE = Path(__file__).parent / "data" / "bridge_test_adapter.py"
HEIGHT, WIDTH = 48, 64
START_IMAGE = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def _bridged(config: dict | None = None) -> BridgedWorld:
    return BridgedWorld(
        python=sys.executable,
        adapter="bridge_test_adapter:GradientWorldAdapter",
        adapter_file=ADAPTER_FILE,
        chunk_timeout_seconds=30.0,
        env={"BRIDGE_TEST_CONFIG": json.dumps(config or {})},
    )


def _expected_frame(frame_index: int, seed: int) -> np.ndarray:
    base = (frame_index * 7 + seed) % 251
    row = (np.arange(WIDTH, dtype=np.uint16) + base) % 256
    frame = np.broadcast_to(row, (HEIGHT, WIDTH)).astype(np.uint8)
    return np.stack([frame, frame[::-1], np.full_like(frame, base)], axis=-1)


def test_bridge_round_trips_capabilities_and_pixels() -> None:
    world = _bridged()
    try:
        capabilities = world.capabilities
        assert capabilities.name == "bridge-test-gradient"
        assert capabilities.resolution == (HEIGHT, WIDTH)
        world.load()
        world.reset(START_IMAGE, None, 11)
        frames = list(world.step(["W"]))
        assert len(frames) == 1
        assert frames[0].index == 0
        assert frames[0].native_frame_count == 4
        assert frames[0].metadata is not None
        assert frames[0].metadata["worker_ts"] > 0
        for offset in range(4):
            assert np.array_equal(frames[0].payload[offset], _expected_frame(offset, 11))
        more = list(world.step(["L"]))
        assert more[0].index == 1
        assert np.array_equal(more[0].payload[0], _expected_frame(4, 11))
    finally:
        world.close()


def test_bridge_runs_a_full_session_case() -> None:
    world = _bridged()
    try:
        world.load()
        case = WorldCase(
            case_id="bridge-session",
            start_image=START_IMAGE,
            script=ActionScript.from_segments([(0, "W", 16), (16, "L", 16)]),
            seed=3,
        )
        result = run_case(world, case)
        assert result.status == "ok", result.error
        assert result.metrics is not None
        assert result.metrics["native_frames"] == 32
        assert result.metrics["control_latency"]["samples"] == 1
    finally:
        world.close()


def test_bridge_reports_adapter_crash_with_stderr_tail() -> None:
    world = _bridged({"crash_at_chunk": 2})
    try:
        world.load()
        case = WorldCase(
            case_id="bridge-crash",
            start_image=START_IMAGE,
            script=ActionScript.from_segments([(0, "W", 32)]),
            seed=3,
        )
        result = run_case(world, case)
        assert result.status == "failed"
        assert result.error_type == "WorldBackendError"
        assert "injected adapter crash" in (result.error or "")
    finally:
        world.close()


def test_bridge_overhead_is_within_two_percent_gate() -> None:
    """M2 gate: synthetic workload over the bridge vs in-process, <= 2%."""

    chunk_count = 120
    busy = 0.02  # 20 ms per chunk, the scale of a fast real model's chunk
    script = ActionScript.from_segments([(0, "W", chunk_count * 4)])
    case = WorldCase(
        case_id="bridge-overhead", start_image=START_IMAGE, script=script, seed=5
    )

    # In-process equivalent of the same adapter workload.
    sys.path.insert(0, str(ADAPTER_FILE.parent))
    try:
        import os

        os.environ["BRIDGE_TEST_CONFIG"] = json.dumps({"busy_seconds": busy})
        from bridge_test_adapter import GradientWorldAdapter

        class InProcess:
            def __init__(self) -> None:
                self._adapter = GradientWorldAdapter()

            @property
            def capabilities(self):
                from liveframe.world.backend import WorldCapabilities

                raw = self._adapter.capabilities()
                return WorldCapabilities(
                    name=raw["name"],
                    interaction_mode=raw["interaction_mode"],
                    native_fps=raw["native_fps"],
                    chunk_frames=raw["chunk_frames"],
                    resolution=tuple(raw["resolution"]),
                    actions=frozenset(raw["actions"]),
                )

            def load(self) -> None:
                self._adapter.load()

            def reset(self, image, prompt, seed) -> None:
                self._adapter.reset(image, prompt, seed)

            def step(self, actions):
                from liveframe.engine import NativeFrame

                for payload in self._adapter.step(list(actions)):
                    yield NativeFrame(
                        index=self._adapter._chunk - 1,
                        payload=payload,
                        native_frame_count=int(payload.shape[0]),
                    )

            def run_script(self, script):
                raise NotImplementedError

            def close(self) -> None:
                self._adapter.close()

        # Best-of-3 on both sides removes OS scheduler noise (F-0009).
        local_walls = []
        for _ in range(3):
            local = InProcess()
            local.load()
            started = time.perf_counter()
            local_result = run_case(local, case)
            local_walls.append(time.perf_counter() - started)
            assert local_result.status == "ok", local_result.error
        local_wall = min(local_walls)
    finally:
        sys.path.remove(str(ADAPTER_FILE.parent))

    bridged_walls = []
    for _ in range(3):
        world = _bridged({"busy_seconds": busy})
        try:
            world.load()
            started = time.perf_counter()
            bridged_result = run_case(world, case)
            bridged_walls.append(time.perf_counter() - started)
            assert bridged_result.status == "ok", bridged_result.error
        finally:
            world.close()
    bridged_wall = min(bridged_walls)

    overhead = (bridged_wall - local_wall) / local_wall
    assert overhead <= 0.02, (
        f"bridge overhead {overhead:.4%} exceeds 2% "
        f"(in-process {local_wall:.3f}s, bridged {bridged_wall:.3f}s)"
    )


@pytest.mark.parametrize("bad_proto", [True])
def test_worker_rejects_protocol_mismatch(bad_proto: bool) -> None:
    import subprocess

    from liveframe.world.bridge import worker as worker_module

    process = subprocess.run(
        [
            sys.executable,
            worker_module.__file__,
            "--adapter",
            "bridge_test_adapter:GradientWorldAdapter",
        ],
        input=json.dumps({"type": "hello", "proto": "wrong.v0"}) + "\n",
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(ADAPTER_FILE.parent),
        },
    )
    assert process.returncode == 1
    assert "protocol mismatch" in process.stdout
