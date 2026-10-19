# M5 — Matrix-Game 3.0 reference (A100 80GB, 2026-10-15)

- `runs/matrix-game-3/smoke/<id>/` — reference bundle: 1/1 case ok. report +
  ledger; `liveframe world verify` passes.
- Generated FPS 1.89 (single A100, int8, SDPA attention), TTFF ~24.8s, 704x1280.
- Claim-check: vendor "40fps @720p" needs 8 DiT + 1 VAE GPUs + flash-attn;
  our single-GPU SDPA number is 1.89 fps. The gap is the honest story.
- NOTE: quick (8-case) multi-run failed 7/8 — MG3's generate() is a one-shot
  full-video generator, not cleanly re-entrant across cases in one process
  (worker dies after the first case's post-loop). A per-case pipeline-state
  reset is future adapter work; the single-case smoke bundle is the reference.
- Pod cost: 61 min @ $1.79/h = $1.83.
