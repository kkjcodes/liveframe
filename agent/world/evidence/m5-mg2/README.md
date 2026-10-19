# M5 — Matrix-Game 2.0 reference (RTX 4090, 2026-10-15)

- `runs/matrix-game-2/standard/<id>/` — reference bundle: 24 cases (8 scenes x 3
  patterns, 60s each), 24/24 ok, 0 failures. report.json/html + ledger.json;
  `liveframe world verify <id>/ledger.json` passes.
- Generated FPS median 11.24; sustained-min 9.6; metric throughput 55.7 fps.
- `metric-validation-m5.json` — throughput gate RE-MEASURED on the pod:
  55.3 fps, metric_throughput_ge_40fps_on_pod = TRUE (closes the M4 open).
- Vendor claim-check: "25fps" shown as config-not-stated (single 4090, 352x640).
- Pod cost: 49 min @ $0.89/h = $0.73.
