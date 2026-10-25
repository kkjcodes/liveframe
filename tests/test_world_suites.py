from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from liveframe.world.adapters.synthetic import SyntheticWorld
from liveframe.world.ledger import verify_world_ledger
from liveframe.world.report import build_report, render_html
from liveframe.world.runner import run_suite
from liveframe.world.suites import (
    SUITE_NAMES,
    SUITES,
    SuiteCase,
    export_suite_manifests,
    fit_scene,
    load_scene,
    materialize_suite,
    pattern_script,
    suite_digest,
)


def _world() -> SyntheticWorld:
    return SyntheticWorld(chunk_delay_seconds=0.0)


# ---------------------------------------------------------------------------
# Suite definitions
# ---------------------------------------------------------------------------


def test_every_declared_suite_exists_and_digests_are_stable() -> None:
    assert set(SUITE_NAMES) == set(SUITES)
    digests = {name: suite_digest(name) for name in SUITE_NAMES}
    assert len(set(digests.values())) == len(digests)  # all distinct
    assert digests == {name: suite_digest(name) for name in SUITE_NAMES}  # repeatable


def test_pattern_script_lands_exactly_on_the_chunk_grid() -> None:
    world = _world()
    world.load()
    capabilities = world.capabilities
    script = pattern_script(
        (("W", 2), ("L", 1), ("0", 1)),
        duration_seconds=10.0,
        capabilities=capabilities,
    )
    first = capabilities.chunk_frames_first or capabilities.chunk_frames
    assert (script.total_frames - first) % capabilities.chunk_frames == 0
    assert script.total_frames >= 10.0 * capabilities.native_fps


def test_materialize_suite_covers_cases_with_distinct_resume_keys() -> None:
    world = _world()
    world.load()
    cases = materialize_suite("quick", world.capabilities)
    assert len(cases) == len(SUITES["quick"]["cases"])
    keys = {case.key(suite="quick", model="synthetic") for case in cases}
    assert len(keys) == len(cases)
    # Key changes with the model name (different chunk grid = different runs).
    other = cases[0].key(suite="quick", model="another-model")
    assert other not in keys


def test_materialize_suite_rejects_unknown_names() -> None:
    world = _world()
    world.load()
    with pytest.raises(ValueError, match="unknown suite"):
        materialize_suite("nope", world.capabilities)


def test_scenes_load_as_uint8_rgb_and_fit_to_any_resolution() -> None:
    scene = load_scene("outdoor-meadow")
    assert scene.dtype == np.uint8
    assert scene.shape == (352, 640, 3)
    fitted = fit_scene(scene, (144, 256))
    assert fitted.shape == (144, 256, 3)
    assert fitted.dtype == np.uint8
    # Identity when already at the target resolution.
    assert fit_scene(scene, (352, 640)) is scene
    # Deterministic.
    assert np.array_equal(fitted, fit_scene(scene, (144, 256)))


def test_export_suite_manifests_round_trips_digests(tmp_path: Path) -> None:
    written = export_suite_manifests(tmp_path)
    assert {path.stem for path in written} == set(SUITE_NAMES)
    payload = json.loads((tmp_path / "quick.json").read_text(encoding="utf-8"))
    assert payload["suite_digest"] == suite_digest("quick")


# ---------------------------------------------------------------------------
# Report builder + renderer
# ---------------------------------------------------------------------------


def _report(cases, **overrides):
    world = _world()
    world.load()
    defaults = dict(
        suite="smoke",
        suite_digest=suite_digest("smoke"),
        model_name="synthetic",
        capabilities=world.capabilities,
        cases=cases,
        cold_start_seconds=1.5,
        cold_start_storage="local-ssd",
        environment={"platform": "test", "gpu": None},
        gpu_price_usd_per_hour=None,
        vendor_claim=None,
        extra={},
        time_to_first_report_seconds=None,
    )
    defaults.update(overrides)
    return build_report(**defaults)


_OK_CASE = {
    "case_id": "smoke-000",
    "scene": "outdoor-meadow",
    "seed": 0,
    "status": "ok",
    "error": None,
    "metrics": {
        "generated_fps": 20.0,
        "sustained_fps": {"min": 18.0, "windows": 2},
        "time_to_first_frame_seconds": 0.4,
        "wall_seconds": 8.0,
        "native_frames": 160,
        "stalls": {"count": 0},
        "emission_intervals": {"count": 39, "p50": 0.05, "p95": 0.06, "max": 0.07},
    },
    "metric_throughput_fps": 55.0,
}


def test_control_section_is_withheld_until_validated_on_real_output() -> None:
    report = _report([_OK_CASE])
    assert report["control_validated_on_real_output"] is False
    page = render_html(report)
    # Experimental framing: off the headline, points to worldmark export.
    assert "experimental" in page.lower()
    assert "public reporting" in page
    assert "worldmark" in page
    assert "Commanded actions are not ground truth" in page
    # The gating is also stated in the not-measured evidence list.
    assert any("control metrics" in note for note in report["not_measured"])


def test_claim_check_is_neutral_and_survives_empty_runs() -> None:
    failed = {**_OK_CASE, "status": "error", "metrics": None}
    # Claim names a config (H100) -> "different configuration", never a
    # pass/fail word; and the builder never says "reproduced".
    report = _report([failed], vendor_claim="25 FPS on H100 (vendor blog)")
    assert report["claim_check"]["verdict"] == "different configuration"
    assert "reproduced" not in report["claim_check"]["verdict"]
    assert report["headline"]["generated_fps_median"] is None
    page = render_html(report)  # must not raise on the None median
    assert "25 FPS on H100" in page
    # Claim with no config stated -> "config not stated".
    bare = _report([failed], vendor_claim="25 fps (press release)")
    assert bare["claim_check"]["verdict"] == "config not stated"
    # Consumer GPU names count as a stated config (regression: "RTX 5090"
    # claims were mislabeled "config not stated").
    consumer = _report([failed], vendor_claim="16 fps on a single RTX 5090")
    assert consumer["claim_check"]["verdict"] == "different configuration"


def test_render_html_is_self_contained_and_keeps_rows_for_skipped_cases() -> None:
    skipped = {
        "case_id": "smoke-001",
        "scene": "outdoor-dunes",
        "seed": 1,
        "status": "skipped-complete",
        "resume_key": "abc",
    }
    page = render_html(_report([_OK_CASE, skipped]))
    assert "smoke-000" in page and "smoke-001" in page
    assert "skipped-complete" in page
    for forbidden in ("<script", "http://", "https://", "url("):
        assert forbidden not in page
    # Cold start is context, not a headline stat card.
    headline_block = page.split("<h2>", 1)[0]
    assert "Cold start" not in headline_block


def test_headline_aggregates_only_ok_cases() -> None:
    failed = {
        "case_id": "smoke-001",
        "scene": "outdoor-dunes",
        "seed": 1,
        "status": "error",
        "error": "boom",
        "metrics": None,
    }
    report = _report([_OK_CASE, failed])
    headline = report["headline"]
    assert headline["cases_ok"] == 1
    assert headline["cases_total"] == 2
    assert headline["cases_failed"] == 1
    assert headline["generated_fps_median"] == pytest.approx(20.0)
    assert headline["sustained_fps_min"] == pytest.approx(18.0)
    assert headline["metric_throughput_fps_median"] == pytest.approx(55.0)


def test_priced_report_carries_cost_per_generated_minute() -> None:
    report = _report([_OK_CASE], gpu_price_usd_per_hour=0.36)
    # 8 s wall for 10 s of video at 0.36 USD/h -> 0.0008 USD/s * 8 s / (10/60 min)
    expected = (8.0 * 0.36 / 3600.0) / (160 / 16.0 / 60.0)
    assert report["headline"]["cost_per_generated_minute_usd_median"] == pytest.approx(
        expected
    )
    assert report["headline"]["gpu_price_usd_per_hour"] == 0.36


# ---------------------------------------------------------------------------
# End-to-end run_suite on the synthetic backend
# ---------------------------------------------------------------------------


def test_run_suite_smoke_end_to_end_with_resume(tmp_path: Path) -> None:
    run_dir, summary = run_suite(
        _world(),
        suite="smoke",
        run_root=tmp_path,
        model_name="synthetic",
        cold_start_storage="local-ssd",
    )
    assert summary["failures"] == 0
    assert summary["cases"] == 1
    assert summary["time_to_first_report_seconds"] > 0

    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["schema"] == "liveframe.world.report.v1"
    assert report["suite_digest"] == suite_digest("smoke")
    assert report["control_validated_on_real_output"] is False
    assert report["headline"]["generated_fps_median"] is not None
    (case,) = report["cases"]
    assert case["status"] == "ok"
    assert case["metric_throughput_fps"] > 0
    assert (run_dir / "report.html").exists()
    assert verify_world_ledger(run_dir / "ledger.json")

    # Resume: the completed case is skipped, artifacts stay verifiable, and the
    # headline is REBUILT FROM STORED RESULTS, not emptied.
    _, resumed = run_suite(
        _world(),
        suite="smoke",
        run_root=tmp_path,
        model_name="synthetic",
        cold_start_storage="local-ssd",
        resume_dir=run_dir,
    )
    assert resumed["skipped"] == 1
    assert resumed["failures"] == 0
    resumed_report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    (resumed_case,) = resumed_report["cases"]
    assert resumed_case["status"] == "skipped-complete"
    assert resumed_case["metrics"]["generated_fps"] == pytest.approx(
        report["cases"][0]["metrics"]["generated_fps"]
    )
    assert resumed_case["script"] == report["cases"][0]["script"]
    headline = resumed_report["headline"]
    assert headline["cases_ok"] == 1
    assert headline["generated_fps_median"] == pytest.approx(
        report["headline"]["generated_fps_median"]
    )
    assert verify_world_ledger(run_dir / "ledger.json")


def test_resume_reruns_cases_whose_definition_changed(tmp_path: Path) -> None:
    """A resume key mismatch (different model/suite/script) must re-run, not skip."""

    run_dir, _ = run_suite(
        _world(),
        suite="smoke",
        run_root=tmp_path,
        model_name="synthetic",
        cold_start_storage="local-ssd",
    )
    # Same artifacts, different model name -> different resume key -> re-run.
    _, resumed = run_suite(
        _world(),
        suite="smoke",
        run_root=tmp_path,
        model_name="another-model",
        cold_start_storage="local-ssd",
        resume_dir=run_dir,
    )
    assert resumed["skipped"] == 0
    assert resumed["failures"] == 0


def test_suite_case_is_frozen() -> None:
    world = _world()
    world.load()
    (case,) = materialize_suite("smoke", world.capabilities)
    assert isinstance(case, SuiteCase)
    with pytest.raises(AttributeError):
        case.seed = 5  # type: ignore[misc]
