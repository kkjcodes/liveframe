from __future__ import annotations

import json
from pathlib import Path

import pytest

from liveframe.world.adapters.synthetic import SyntheticWorld
from liveframe.world.export import export_run
from liveframe.world.report import build_comparison, render_comparison_html
from liveframe.world.runner import run_suite


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir, summary = run_suite(
        SyntheticWorld(chunk_delay_seconds=0.0),
        suite="smoke",
        run_root=tmp_path_factory.mktemp("runs"),
        model_name="synthetic",
        cold_start_storage="local-ssd",
    )
    assert summary["failures"] == 0
    return run_dir


def test_raw_export_layout(smoke_run: Path, tmp_path: Path) -> None:
    summary = export_run(smoke_run, tmp_path, export_format="raw")
    assert summary["cases"] == 1
    video = tmp_path / "videos" / "smoke-000-outdoor-meadow-s0.mp4"
    actions = tmp_path / "actions" / "smoke-000-outdoor-meadow-s0.jsonl"
    assert video.exists() and video.stat().st_size > 0
    rows = [
        json.loads(line)
        for line in actions.read_text(encoding="utf-8").splitlines()
    ]
    assert rows and all("action" in row and "start_frame" in row for row in rows)
    cases = json.loads((tmp_path / "cases.json").read_text(encoding="utf-8"))
    assert cases[0]["capture_tier"] == "short side 256"
    assert (tmp_path / "report.json").exists()


def test_worldmark_export_layout(smoke_run: Path, tmp_path: Path) -> None:
    export_run(smoke_run, tmp_path, export_format="worldmark", model_tag="demo")
    # {domain}/{view}/{MODEL}/{image:03d}_{action:03d}.mp4, MODEL upper-cased.
    video = tmp_path / "synthetic" / "first" / "DEMO" / "000_000.mp4"
    assert video.exists() and video.stat().st_size > 0
    mapping = json.loads(
        (tmp_path / "worldmark_mapping.json").read_text(encoding="utf-8")
    )
    assert "arena_inputs" in mapping["caveat"]
    assert mapping["mapping"][0]["case_id"] == "smoke-000-outdoor-meadow-s0"
    assert mapping["mapping"][0]["actions"]  # action log travels with the video


def test_wbench_export_layout(smoke_run: Path, tmp_path: Path) -> None:
    export_run(smoke_run, tmp_path, export_format="wbench", model_tag="demo")
    # work_dirs/<model>/videos/case_{id}_combined.mp4 + data/cases/case_{id}.json
    video = tmp_path / "work_dirs" / "demo" / "videos" / "case_1_combined.mp4"
    assert video.exists() and video.stat().st_size > 0
    case = json.loads(
        (tmp_path / "data" / "cases" / "case_1.json").read_text(encoding="utf-8")
    )
    assert case["case_id"] == 1
    assert case["turns"][0]["tokens"]
    assert case["source_case_id"] == "smoke-000-outdoor-meadow-s0"


def test_export_rejects_unknown_format(smoke_run: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown export format"):
        export_run(smoke_run, tmp_path, export_format="leaderboard")


def test_compare_flags_config_differences_before_numbers(smoke_run: Path) -> None:
    report = json.loads((smoke_run / "report.json").read_text(encoding="utf-8"))
    other = json.loads(json.dumps(report))  # deep copy
    other["environment"]["gpu"] = "NVIDIA H100, 80 GB"
    other["headline"]["generated_fps_median"] = 999.0

    comparison = build_comparison(report, other)
    assert comparison["comparable"] is False
    assert [row["key"] for row in comparison["config_differences"]] == ["gpu"]
    page = render_comparison_html(comparison)
    assert "Configurations differ" in page
    # Config diff section renders before the metric table.
    assert page.index("Config differences") < page.index("Headline metrics")
    for forbidden in ("<script", "http://", "https://"):
        assert forbidden not in page

    same = build_comparison(report, report)
    assert same["comparable"] is True
    assert "Configurations differ" not in render_comparison_html(same)
