"""World run reports: machine-readable JSON + a single-file HTML rendering.

Report schema ``liveframe.world.report.v1`` (spec section 10.2). Rendering
rules that encode project policy:

- CONTROL METRICS ARE NOT PUBLIC until classifier-vs-human agreement on real
  model output exists (Kumar's M4 correction 2): the report JSON always
  carries them, but ``control_validated_on_real_output`` is false until the
  labeling study lands, and the HTML renderer shows a validation-pending
  notice instead of control numbers while it is false.
- Claim checks are neutral: the measured value is shown next to a LINK to the
  vendor's claim with verdict labels limited to ``reproduced``,
  ``not reproduced``, and ``config not stated`` — never pass/fail language —
  and only when the configuration matches is any comparison implied
  (section 15 answer 4; vendors receive reports before publication).
- Cold start is context, never headline (M2.1 item 6).

The HTML is fully self-contained: inline CSS, inline SVG, no scripts, no
network access.
"""

from __future__ import annotations

import html
from typing import Any

import numpy as np

from liveframe.world.backend import WorldCapabilities

REPORT_SCHEMA = "liveframe.world.report.v1"
CONTROL_VALIDATED_ON_REAL_OUTPUT = False  # flips when the labeling study lands


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def build_report(
    *,
    suite: str,
    suite_digest: str,
    model_name: str,
    capabilities: WorldCapabilities,
    cases: list[dict[str, Any]],
    cold_start_seconds: float,
    cold_start_storage: str,
    environment: dict[str, Any],
    gpu_price_usd_per_hour: float | None,
    vendor_claim: str | None,
    extra: dict[str, Any],
    time_to_first_report_seconds: float | None,
) -> dict[str, Any]:
    # Completed = run this invocation OR resumed from verified artifacts; a
    # fully-resumed run must rebuild the same headline, not an empty one.
    ok_cases = [
        case
        for case in cases
        if case["status"] in ("ok", "skipped-complete") and case.get("metrics")
    ]
    fps_values = [case["metrics"]["generated_fps"] for case in ok_cases]
    sustained_mins = [
        case["metrics"]["sustained_fps"]["min"]
        for case in ok_cases
        if case["metrics"]["sustained_fps"]["min"] is not None
    ]
    ttff_values = [case["metrics"]["time_to_first_frame_seconds"] for case in ok_cases]
    throughput = [
        case["metric_throughput_fps"] for case in ok_cases if "metric_throughput_fps" in case
    ]
    realtime_rows = [
        case["metrics"]["realtime"]
        for case in ok_cases
        if (case["metrics"].get("realtime") or {}).get("applicable")
    ]
    latency_p50 = [
        case["metrics"]["control_latency"]["p50"]
        for case in ok_cases
        if (case["metrics"].get("control_latency") or {}).get("p50") is not None
    ]
    latency_p95 = [
        case["metrics"]["control_latency"]["p95"]
        for case in ok_cases
        if (case["metrics"].get("control_latency") or {}).get("p95") is not None
    ]
    headline: dict[str, Any] = {
        "model": model_name,
        "suite": suite,
        "resolution": list(capabilities.resolution),
        "interaction_mode": capabilities.interaction_mode,
        "generated_fps_median": _median(fps_values),
        "sustained_fps_min": min(sustained_mins) if sustained_mins else None,
        "time_to_first_frame_p50": _median(ttff_values),
        "cases_ok": len(ok_cases),
        "cases_total": len(cases),
        "cases_failed": sum(
            1 for case in cases if case["status"] not in ("ok", "skipped-complete")
        ),
        "metric_throughput_fps_median": _median(throughput),
        # None = no interactive case made the realtime check applicable.
        "realtime_pass": (
            all(row["passed"] for row in realtime_rows) if realtime_rows else None
        ),
        "control_latency_p50_median": _median(latency_p50),
        "control_latency_p95_max": max(latency_p95) if latency_p95 else None,
        # A "median" over a single ok case is a point estimate, not a
        # distribution — the project requires warmup + >=3 trials for a true
        # reference, so flag it explicitly (reviewer finding, M5).
        "single_trial": len(ok_cases) <= 1,
        "trials": len(ok_cases),
    }
    if gpu_price_usd_per_hour is not None and ok_cases:
        from liveframe.world.metrics.runner import cost_per_generated_minute

        costs = [
            cost_per_generated_minute(
                wall_seconds=case["metrics"]["wall_seconds"],
                native_frames=case["metrics"]["native_frames"],
                native_fps=capabilities.native_fps,
                gpu_price_usd_per_hour=gpu_price_usd_per_hour,
            )
            for case in ok_cases
        ]
        headline["cost_per_generated_minute_usd_median"] = _median(costs)
        headline["gpu_price_usd_per_hour"] = gpu_price_usd_per_hour

    claim_check: dict[str, Any] | None = None
    if vendor_claim:
        # Verdicts: reproduced / not-reproduced (both require a config-matched
        # comparison, which the builder cannot assert on its own), config-not-
        # stated (the claim names no config), or different-configuration (the
        # claim names a config but it differs from what we measured — the honest
        # label for a single-GPU run vs a multi-GPU vendor claim). The builder
        # defaults to different-configuration when the claim text names a config
        # (contains "GPU"/"@"/"parallel"/steps), else config-not-stated. A human
        # upgrades to reproduced / not-reproduced only after matching the config.
        names_config = any(
            marker in vendor_claim.lower()
            for marker in ("gpu", "@", "parallel", "step", "h100", "a100", "sp")
        )
        claim_check = {
            "claim": vendor_claim,
            "measured_generated_fps_median": headline["generated_fps_median"],
            "verdict": "different configuration" if names_config else "config not stated",
            "note": (
                "measured single-GPU; the vendor claim's configuration (see the "
                "claim text) differs from this run's hardware/config in the "
                "environment section. 'reproduced'/'not reproduced' require a "
                "configuration-matched comparison by a human."
            ),
        }

    # Interpretation-critical caveats must live IN the machine-readable report,
    # not only a sibling README (reviewer finding, M5). Assemble from the run
    # metadata (extra) + the declared interaction mode.
    caveats = []
    if capabilities.interaction_mode == "batch":
        caveats.append(
            "batch mode: all frames decoded at the end; the FPS is batch "
            "throughput, not an interactive/streaming rate"
        )
    if extra.get("offloaded"):
        caveats.append("ran with CPU offloading (weights streamed) — slower than resident")
    if extra.get("upstream_patched"):
        caveats.append(f"upstream source patched: {extra['upstream_patched']}")
    if extra.get("attention") and extra["attention"] != "flash":
        caveats.append(f"attention backend: {extra['attention']} (not flash-attn)")
    if headline["single_trial"]:
        caveats.append(
            "single-trial point estimate (1 ok case); not a warmup+>=3-trial reference"
        )

    return {
        "schema": REPORT_SCHEMA,
        "headline": headline,
        "claim_check": claim_check,
        "caveats": caveats,
        "capabilities": capabilities.as_dict(),
        "suite_digest": suite_digest,
        "cold_start_seconds": cold_start_seconds,
        "cold_start_storage": cold_start_storage,
        "control_validated_on_real_output": CONTROL_VALIDATED_ON_REAL_OUTPUT,
        "cases": cases,
        "environment": environment,
        "extra": extra,
        "time_to_first_report_seconds": time_to_first_report_seconds,
        "not_measured": _not_measured(capabilities),
    }


def _not_measured(capabilities: WorldCapabilities) -> list[str]:
    notes = [
        "peak_vram_gb / gpu_util / power: NVML sampling not yet wired (lands with the "
        "next GPU-session milestone)",
    ]
    if capabilities.interaction_mode == "batch":
        notes.append("real-time and action-latency claims: not applicable in batch mode")
    if not CONTROL_VALIDATED_ON_REAL_OUTPUT:
        notes.append(
            "control metrics: EXPERIMENTAL — computed and stored, but excluded from "
            "the headline and public reporting until the classifier reaches the "
            "promotion bar (per-class >=85% blind two-labeler agreement on fresh "
            "clips). Use `world export --format worldmark` for control scoring"
        )
    return notes


# ---------------------------------------------------------------------------
# HTML rendering (single file, inline CSS/SVG, no scripts, no network)
# ---------------------------------------------------------------------------

_CSS = """
:root{
  --bg:#fafaf8; --panel:#ffffff; --ink:#1c1c1a; --muted:#6b6b66;
  --line:#e4e2dc; --accent:#3d6bb3; --good:#1b7a2d; --bad:#b00020;
  --good-bg:#e9f5ec; --bad-bg:#fbeaec; --notice-bg:#fdf6e3; --notice-line:#d9c37a;
  --bar:#7596c8;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#16161a; --panel:#1f1f24; --ink:#e8e8e4; --muted:#9a9a94;
    --line:#34343b; --accent:#7ba3dd; --good:#6fce84; --bad:#ef8a97;
    --good-bg:#1d2b21; --bad-bg:#322126; --notice-bg:#2b2618; --notice-line:#6b5d2e;
    --bar:#5a7cb0;
  }
}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
     margin:0;background:var(--bg);color:var(--ink);line-height:1.5}
main{max-width:1020px;margin:0 auto;padding:2.5rem 1.25rem 4rem}
h1{font-size:1.5rem;letter-spacing:-0.01em;margin:0 0 0.25rem}
.sub{color:var(--muted);margin:0 0 1.75rem;font-size:0.95rem}
h2{font-size:0.8rem;text-transform:uppercase;letter-spacing:0.09em;color:var(--muted);
   margin:2.5rem 0 0.75rem;padding-bottom:0.4rem;border-bottom:1px solid var(--line)}
table{border-collapse:collapse;width:100%;font-size:0.88rem;background:var(--panel);
      border:1px solid var(--line);border-radius:10px;overflow:hidden}
th{font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--muted);
   font-weight:600}
td,th{border-bottom:1px solid var(--line);padding:0.5rem 0.75rem;text-align:left;
      vertical-align:middle}
tr:last-child td{border-bottom:none}
.headline{display:grid;grid-template-columns:repeat(auto-fit,minmax(10.5rem,1fr));
          gap:0.75rem}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;
      padding:0.8rem 1rem}
.stat b{display:block;font-size:1.45rem;font-weight:650;letter-spacing:-0.01em;
        font-variant-numeric:tabular-nums}
.stat b small{font-size:0.8rem;font-weight:500;color:var(--muted)}
.stat span{font-size:0.78rem;color:var(--muted)}
.chip{display:inline-block;border-radius:999px;padding:0.1rem 0.65rem;
      font-size:0.78rem;font-weight:600}
.chip.ok{color:var(--good);background:var(--good-bg)}
.chip.failed{color:var(--bad);background:var(--bad-bg)}
.chip.neutral{color:var(--muted);background:var(--bg);border:1px solid var(--line)}
.notice{background:var(--notice-bg);border:1px solid var(--notice-line);
        padding:0.75rem 1rem;border-radius:10px;font-size:0.9rem}
.failed{color:var(--bad)} .ok{color:var(--good)}
code{background:var(--panel);border:1px solid var(--line);border-radius:4px;
     padding:0 0.3rem;font-size:0.85em;word-break:break-all}
svg{max-width:100%;display:block}
ul{padding-left:1.2rem} li{margin:0.3rem 0}
p,li{font-size:0.92rem}
footer{margin-top:3.5rem;font-size:0.8rem;color:var(--muted);
       border-top:1px solid var(--line);padding-top:1rem}
.num{font-variant-numeric:tabular-nums;text-align:right}
"""

# Action timeline colors (consistent across every report).
_ACTION_COLORS = {
    "W": "#4c9e5f", "S": "#c2574f", "A": "#4c7fb8", "D": "#7a5fb8",
    "L": "#b88a3c", "R": "#3ca6a6", "0": "#9a9a94",
}


def _timeline_svg(script: list[list[Any]] | None) -> str:
    """One compact strip: the commanded action over the case's native frames."""

    if not script:
        return ""
    total = sum(int(hold) for _, _, hold in script) or 1
    parts = ['<svg viewBox="0 0 360 26" role="img" aria-label="action timeline">']
    cursor = 0.0
    for _t_frames, action, hold in script:
        width = 360.0 * int(hold) / total
        base = str(action).split("+")[0]
        color = _ACTION_COLORS.get(base, "#888888")
        parts.append(
            f'<rect x="{cursor:.1f}" y="6" width="{max(width, 1.0):.1f}" height="14" '
            f'rx="2" fill="{color}"></rect>'
        )
        if width >= 18:
            parts.append(
                f'<text x="{cursor + width / 2:.1f}" y="17" font-size="10" '
                f'fill="#ffffff" text-anchor="middle">{html.escape(str(action))}</text>'
            )
        cursor += width
    parts.append("</svg>")
    return "".join(parts)


def _stat(label: str, value: Any, unit: str = "") -> str:
    shown = "n/a" if value is None else (f"{value:.2f}" if isinstance(value, float) else value)
    return (
        f'<div class="stat"><b>{html.escape(str(shown))}'
        f'<small>{html.escape(unit)}</small></b>'
        f"<span>{html.escape(label)}</span></div>"
    )


def _sustained_svg(case: dict[str, Any]) -> str:
    metrics = case.get("metrics") or {}
    intervals = metrics.get("emission_intervals") or {}
    fps = metrics.get("generated_fps")
    if fps is None or not intervals.get("count"):
        return ""
    # Compact p50/p95/max emission-interval bars: jitter shape at a glance.
    rows = [("p50", intervals["p50"]), ("p95", intervals["p95"]), ("max", intervals["max"])]
    peak = max(value for _, value in rows) or 1.0
    bars = []
    for index, (name, value) in enumerate(rows):
        width = 150 * value / peak
        y = 4 + index * 15
        bars.append(
            f'<text x="0" y="{y + 9}" font-size="9" fill="var(--muted)">{name}</text>'
            f'<rect x="28" y="{y}" width="{max(width, 1.0):.1f}" height="11" rx="2" '
            f'fill="var(--bar)"></rect>'
            f'<text x="{34 + width:.1f}" y="{y + 9}" font-size="9" '
            f'fill="var(--muted)">{value:.3f}s</text>'
        )
    return (
        f'<svg viewBox="0 0 240 52" width="240" role="img" '
        f'aria-label="emission intervals">{"".join(bars)}</svg>'
    )


def render_html(report: dict[str, Any]) -> str:
    headline = report["headline"]
    environment = report["environment"]
    resolution = "x".join(str(side) for side in headline["resolution"])
    chips = []
    if headline["cases_failed"] == 0 and headline["cases_ok"] > 0:
        chips.append('<span class="chip ok">all cases ok</span>')
    elif headline["cases_failed"]:
        chips.append(
            f'<span class="chip failed">{headline["cases_failed"]} case(s) failed</span>'
        )
    if headline.get("realtime_pass") is True:
        chips.append('<span class="chip ok">real-time at native rate</span>')
    elif headline.get("realtime_pass") is False:
        chips.append('<span class="chip failed">below native rate</span>')
    else:
        chips.append('<span class="chip neutral">real-time check not applicable</span>')
    gpu = environment.get("gpu") or "no GPU detected"
    parts: list[str] = [
        "<!-- rendered by liveframe world; self-contained, no scripts, no network -->",
        f"<style>{_CSS}</style>",
        "<main>",
        f"<h1>{html.escape(headline['model'])} "
        f"<span style='font-weight:400;color:var(--muted)'>· "
        f"{html.escape(headline['suite'])} suite</span></h1>",
        f'<p class="sub">{html.escape(str(gpu))} · {html.escape(resolution)} · '
        f"{html.escape(headline['interaction_mode'])} · "
        f"native {report['capabilities']['native_fps']:g} fps "
        f"&nbsp; {' '.join(chips)}</p>",
        '<div class="headline">',
        _stat("generated FPS, median", headline["generated_fps_median"]),
        _stat("sustained FPS, worst window", headline["sustained_fps_min"]),
        _stat("time to first frame, p50", headline["time_to_first_frame_p50"], " s"),
        _stat(
            "action latency p50 / p95",
            f"{headline['control_latency_p50_median']:.2f} / "
            f"{headline['control_latency_p95_max']:.2f}"
            if headline.get("control_latency_p50_median") is not None
            and headline.get("control_latency_p95_max") is not None
            else None,
            " s",
        ),
        _stat("cases ok", f"{headline['cases_ok']}/{headline['cases_total']}"),
    ]
    if "cost_per_generated_minute_usd_median" in headline:
        parts.append(
            _stat(
                "cost per generated minute",
                headline["cost_per_generated_minute_usd_median"],
                " USD",
            )
        )
    parts.append("</div>")

    claim = report.get("claim_check")
    if claim:
        measured = claim["measured_generated_fps_median"]
        measured_text = f"{measured:.2f}" if measured is not None else "n/a"
        parts.append("<h2>Claim check</h2>")
        parts.append(
            f"<p>Vendor claim: <em>{html.escape(claim['claim'])}</em> — measured "
            f"generated FPS median: <b>{measured_text}</b>. "
            f"Verdict: <b>{html.escape(claim['verdict'])}</b>. "
            f"{html.escape(claim['note'])}</p>"
        )

    parts.append("<h2>Runtime, per case</h2>")
    parts.append(
        "<table><tr><th>case</th><th>actions</th><th>status</th>"
        '<th class="num">FPS</th><th class="num">first frame</th>'
        '<th class="num">stalls</th><th>emission intervals</th></tr>'
    )
    for case in report["cases"]:
        metrics = case.get("metrics") or {}
        status_class = "ok" if case["status"] in ("ok", "skipped-complete") else "failed"
        fps = metrics.get("generated_fps")
        ttff = metrics.get("time_to_first_frame_seconds")
        stalls = (metrics.get("stalls") or {}).get("count")
        fps_cell = (
            f'<td class="num">{fps:.2f}</td>' if fps is not None else '<td class="num">n/a</td>'
        )
        ttff_cell = (
            f'<td class="num">{ttff:.3f}s</td>' if ttff is not None else '<td class="num">n/a</td>'
        )
        label = f"{case['case_id']}"
        scene = case.get("scene")
        scene_note = (
            f"<br><span style='color:var(--muted);font-size:0.8em'>"
            f"{html.escape(str(scene))} · seed {case.get('seed')}</span>"
            if scene
            else ""
        )
        parts.append(
            f"<tr><td>{html.escape(label)}{scene_note}</td>"
            f"<td>{_timeline_svg(case.get('script'))}</td>"
            f'<td><span class="chip {status_class}">{html.escape(case["status"])}'
            f"</span></td>"
            f"{fps_cell}{ttff_cell}"
            f'<td class="num">{stalls if stalls is not None else "n/a"}</td>'
            f"<td>{_sustained_svg(case)}</td></tr>"
        )
        if case.get("error"):
            parts.append(
                f'<tr><td colspan="7" class="failed" style="font-size:0.85em">'
                f"{html.escape(str(case['error']))}</td></tr>"
            )
    parts.append("</table>")
    parts.append(
        f"<p>Cold start: {report['cold_start_seconds']:.1f} s from "
        f"{html.escape(report['cold_start_storage'])} storage — reported as context, "
        "never blended into generation numbers.</p>"
    )

    parts.append("<h2>Control <span class=\"chip neutral\">experimental</span></h2>")
    if report["control_validated_on_real_output"]:
        parts.append("<p>(control metrics table would render here)</p>")
    else:
        parts.append(
            '<div class="notice"><b>Control metrics are experimental and excluded '
            "from the headline and public reporting by default.</b> They are "
            "computed and stored in this run's evidence, but the optical-flow "
            "classifier is not yet validated to the promotion bar (per-class "
            "≥ 85% blind two-labeler agreement on fresh clips). Commanded "
            "actions are not ground truth. For control scoring now, export the "
            "run for a dedicated benchmark: "
            "<code>liveframe world export &lt;run&gt; --format worldmark</code>.</div>"
        )

    parts.append("<h2>What was not measured</h2><ul>")
    for note in report["not_measured"]:
        parts.append(f"<li>{html.escape(note)}</li>")
    parts.append("</ul>")

    parts.append("<h2>Evidence</h2><ul>")
    parts.append(
        f"<li>Suite digest: <code>{html.escape(report['suite_digest'])}</code></li>"
    )
    parts.append(
        "<li>Re-verify every artifact: <code>liveframe world verify "
        "&lt;run_dir&gt;/ledger.json</code></li>"
    )
    parts.append(
        f"<li>Host: {html.escape(str(environment.get('platform')))} · "
        f"Python {html.escape(str(environment.get('python')))} · "
        f"{environment.get('cpu_count')} CPUs</li>"
    )
    if report.get("time_to_first_report_seconds") is not None:
        parts.append(
            f"<li>Time from command to this report: "
            f"{report['time_to_first_report_seconds']:.1f} s</li>"
        )
    parts.append("</ul>")
    parts.append(
        "<footer>liveframe world report v1 — self-contained page, no scripts, no "
        "network. Every number traces to the digest-bound run ledger next to this "
        "file.</footer>"
    )
    parts.append("</main>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Compare (spec 10.3): config differences first, numbers second, never silent.
# ---------------------------------------------------------------------------

_CONFIG_KEYS = (
    ("model", lambda report: report["headline"]["model"]),
    ("suite", lambda report: report["headline"]["suite"]),
    ("suite_digest", lambda report: report["suite_digest"]),
    ("resolution", lambda report: "x".join(map(str, report["headline"]["resolution"]))),
    ("interaction_mode", lambda report: report["headline"]["interaction_mode"]),
    ("native_fps", lambda report: report["capabilities"]["native_fps"]),
    ("gpu", lambda report: report["environment"].get("gpu")),
    ("platform", lambda report: report["environment"].get("platform")),
)

_COMPARE_METRICS = (
    "generated_fps_median",
    "sustained_fps_min",
    "time_to_first_frame_p50",
    "metric_throughput_fps_median",
    "cost_per_generated_minute_usd_median",
    "cases_ok",
    "cases_failed",
)


def build_comparison(report_a: dict[str, Any], report_b: dict[str, Any]) -> dict[str, Any]:
    differences = []
    for key, getter in _CONFIG_KEYS:
        value_a, value_b = getter(report_a), getter(report_b)
        if value_a != value_b:
            differences.append({"key": key, "a": value_a, "b": value_b})
    metrics = {
        name: {
            "a": report_a["headline"].get(name),
            "b": report_b["headline"].get(name),
        }
        for name in _COMPARE_METRICS
    }
    return {
        "schema": "liveframe.world.comparison.v1",
        "a": {"model": report_a["headline"]["model"], "suite": report_a["headline"]["suite"]},
        "b": {"model": report_b["headline"]["model"], "suite": report_b["headline"]["suite"]},
        "config_differences": differences,
        "comparable": not differences,
        "metrics": metrics,
    }


def render_comparison_html(comparison: dict[str, Any]) -> str:
    label_a = f"{comparison['a']['model']} ({comparison['a']['suite']})"
    label_b = f"{comparison['b']['model']} ({comparison['b']['suite']})"
    parts = [
        "<!-- rendered by liveframe world; self-contained, no scripts -->",
        f"<style>{_CSS}</style>",
        "<main>",
        f"<h1>{html.escape(label_a)} <span style='font-weight:400;"
        f"color:var(--muted)'>vs</span> {html.escape(label_b)}</h1>",
        '<p class="sub">liveframe world compare</p>',
    ]
    differences = comparison["config_differences"]
    if differences:
        parts.append(
            '<div class="notice"><b>Configurations differ.</b> The numbers below '
            "are shown with their configs and must not be read as a same-config "
            "comparison.</div>"
        )
        parts.append("<h2>Config differences</h2>")
        parts.append("<table><tr><th>key</th><th>A</th><th>B</th></tr>")
        for row in differences:
            parts.append(
                f"<tr><td>{html.escape(str(row['key']))}</td>"
                f"<td>{html.escape(str(row['a']))}</td>"
                f"<td>{html.escape(str(row['b']))}</td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append("<p>No config differences detected across the tracked keys.</p>")
    parts.append("<h2>Headline metrics</h2>")
    parts.append(
        f"<table><tr><th>metric</th><th>{html.escape(label_a)}</th>"
        f"<th>{html.escape(label_b)}</th></tr>"
    )
    for name, pair in comparison["metrics"].items():
        def _format(value: Any) -> str:
            if value is None:
                return "n/a"
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value)

        parts.append(
            f"<tr><td>{html.escape(name)}</td><td>{html.escape(_format(pair['a']))}</td>"
            f"<td>{html.escape(_format(pair['b']))}</td></tr>"
        )
    parts.append("</table>")
    parts.append(
        "<footer>liveframe world comparison v1 — magnitudes are not normalized "
        "across models; compare direction and timing, not distance.</footer>"
    )
    parts.append("</main>")
    return "\n".join(parts)
