# M5 — HY-World 1.5 / HY-WorldPlay reference (A100 80GB, 2026-10-16)

- `runs/hy-worldplay/smoke/<id>/` — reference bundle: 1/1 case ok, batch mode,
  480x832. report + ledger; `liveframe world verify` passes.
- Generated FPS 0.60 (single A100 80GB, group-offloaded, 4-step distilled).
  Direct pipeline test: 0.65 fps (125 frames / 191s) — harness number (0.60)
  matches the direct number, validating the harness measurement.
- Claim-check: vendor "24 fps 480p" uses 8-GPU SEQUENCE PARALLEL; our single-GPU
  offloaded number is ~0.6 fps — the largest claim/measured gap of the three models.
- Weights (~286GB, incl. gated FLUX.1-Redux-dev) downloaded with Kumar's HF token;
  pod terminated (token copy destroyed). Volume liveframe-hy-weights pending cleanup.
