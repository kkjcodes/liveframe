# Progress Review — 2026-08-20 (overnight autonomous run, taew2_1 tier complete)

This review records the full post-restart execution of the taew2_1 gate chain,
run autonomously overnight 2026-08-18 → 2026-08-20. Every runner was frozen in
git BEFORE its result was visible; result artifacts were committed in batch at
the end of the run per the user's standing instruction. Nothing was pushed.

## Verdict in one paragraph

The taew2_1 FP32-im2col tier is a **performance GO and a quality-screen STOP**.
The sealed smoke gate passed post-restart, the frozen timing screen passed with
a decoder median of 3.5145 s (ceiling 13.7731 s), and the formal
one-warmup/five-trial benchmark passed both frozen rules: complete wall
57.1265 s → **21.9021 s median (2.6083× point speedup; one-sided 95% bootstrap
lower confidence bound 2.5818 against the 1.5× floor)**, decoder-only
41.3194 s → 7.4825 s (5.5222×), pinned latent quality anchor passing on every
trial and identical tiny pixel digests across all five trials. The 32-job
paired campaign then completed cleanly (50 GB of digest-bound paired captures),
and the frozen metric screen returned an authoritative **FAIL**
(`artifacts/taew21-metric-verdict.v1.json`): the release tier stops and the
human study is not permitted under this protocol iteration. The paired speed
result remains publishable with the quality gap honestly characterized.

## Gate-by-gate record

1. **Smoke PASS** (`outputs/taew21-fp32-im2col-smoke.v1.json`, commit
   `7f384fc`): 81 contiguous finite uint8 frames at 480×832, zero start swap
   and zero delta, pixel digest bound.
2. **Screen PASS** (`outputs/taew21-fp32-im2col-screen.v1.json`, `4c0aadb`;
   runner frozen at `bfdd40c`): timed walls 3.549/3.507/3.514 s, median
   3.5145 s vs the 13.7731 s ceiling, every trial reproducing the smoke pixel
   digest with zero swap.
3. **Formal PASS** (`outputs/taew21-cfpp-formal-benchmark.v1.json`, `67b8189`;
   runner frozen at `0c1df3f`): trials 21.743/21.902/21.860/22.045/22.178 s;
   both the 38.0843 s point ceiling and the bootstrap confidence rule
   (10000 resamples, seed 20260809) passed; claim-eligible.
4. **Campaign authorization PASS**
   (`outputs/taew21-campaign-authorization.v1.json`): protocol + seed-exclusion
   validation plus the digest-bound gate chain.
5. **32-job paired campaign COMPLETE**
   (`outputs/taew21-campaign/campaign-summary.v1.json`; runner frozen at
   `e538279`, amended `7382ac2` pre-results): per job one latent generation
   from an immutable noise plan, decoded by BOTH the full Wan VAE and the tiny
   decoder in deterministic-randomized order; 15 s jobs ≈ 4 min, 60 s jobs
   ≈ 18 min; full decode ≈ 94/380 s vs tiny ≈ 10/40 s (15 s/60 s horizons).
6. **Metric screen FAIL** (`artifacts/taew21-metric-verdict.v1.json`;
   evaluator frozen at `035e136`): frozen decision rule
   `all-intrinsic-bands-and-all-identity-tripwires-must-pass` → FAIL.

## What failed, precisely

- **Passing**: pairwise CLIP distance 32/32 (max 0.035 vs 0.05); pairwise
  LPIPS 27/32, all five failures marginal (0.1020–0.1158 vs 0.10) and all in
  `high-motion-dense-scenes`.
- **Failing**: the temporal-CLIP tripwire in 28/32 pairs (0.28–0.62 vs 0.35),
  across ALL strata including near-static ones; intrinsic motion/temporal
  bands (`motion_clip_distance_mean` 14, `temporal_delta_cosine_mean` 17,
  `motion_lpips_mean` 10, `luma_flicker_mean` 7 tiny-video failures) in all
  eight cells.
- **Reading**: the tiny decoder reproduces content and composition (LPIPS/CLIP
  identity clean) but its different high-frequency texture systematically
  perturbs frame-to-frame feature differences, which every motion/temporal
  instrument amplifies. Two observations are candidates for a FUTURE
  prospective addendum (they are diagnostics, not excuses, and no retuning is
  permitted after visibility): (a) cosine distance between consecutive-frame
  CLIP feature deltas is noise-dominated for near-static content — the 0.35
  "loose" bound is empirically not loose for this feature; (b) the intrinsic
  bands used n=4 full-decoder videos per cell, so MAD scales are narrow.

## Consequences (per the frozen protocol)

- Release tier: **STOPPED**. taew2_1 does not ship as a quality-validated tier.
- Human study: **NOT permitted**; the pre-registered study assets remain frozen
  and unused (power simulation `artifacts/taew21-study-power.v1.json`: 30
  participants × 18 trials, 486 usable-judgment target; bundle renderer
  `tools/build_taew21_study_bundle.py` gated on a metric PASS).
- Publication: the paired result IS publishable — a 2.61× complete-wall
  speedup (1.418 → 3.70 generated FPS on M5) with measured quality deltas.
  This also strengthens the honest-measurement thesis: the pipeline caught a
  real quality gap that a demo-driven evaluation would have shipped.
- Any retuned metric screen or fp16 track requires a new prospective addendum.

## Operational notes

- The first campaign launch was stopped pre-results: per-job T5-XXL encoder
  loads swapped 8.5 GiB on this host (~36 GB RAM). Amendment `7382ac2` encodes
  all 16 prompts once (single T5 residency, 10.2 s) and demotes swap to
  recorded telemetry for the campaign only (campaign timing carries no claim
  authority; pixels are deterministic under paging).
- The power simulator was amended twice pre-response (`809b116`, `cea903f`):
  the protocol's own 400–600 planning envelope caps pass probability at true
  chance near 0.75–0.79 even for independent judgments, and additive-logit
  heterogeneity at chance simulates partially-distinguishable videos (a
  different hypothesis); the frozen model now scales heterogeneity with
  discriminability.
- Validation state: 219 tests pass, Ruff clean, disk 1.1 TiB free after 50 GB
  of campaign artifacts.

## Next decisions (user)

1. Approve the batched result commits (made at end of run) and decide on push.
2. Decide whether to draft a prospective metric-screen addendum v2 (temporal
   instrument redesign + larger calibration set) or publish the tier as
   speed-GO/quality-screen-STOP as-is.
3. arXiv: fold the 2.61× result, the MLX conv2d drift finding, and the metric
   FAIL into the paper (v2), and consider the upstream ml-explore issue.
