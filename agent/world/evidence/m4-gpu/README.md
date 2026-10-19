# M4 GPU-session evidence (RTX 4090, 2026-10-14)

- `matrix-game-2/quick/<id>/` — the citable reference bundle: full capture,
  report.json + report.html + ledger.json. Verify with
  `liveframe world verify <id>/ledger.json` (passes; 43 files).
- `smoke-trial-{1..4}.json` — the warmup + 3-trial reproducibility summaries
  (generated FPS 10.74 / 10.91 / 10.66 / 10.80). These are CLI summary outputs,
  not published verifiable bundles; the per-trial frame captures were not
  pulled home (bandwidth) so their run dirs are intentionally absent.
- `metric-validation-m4.json` — live metric-throughput gate (35.5 fps on the
  pod CPU; see gate-m4.md for the open condition and fix).
- `quick-run.json` — quick-suite summary (time-to-first-report 502.8 s = 8.38 min).
