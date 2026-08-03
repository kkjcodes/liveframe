# LiveFrame — Project State & Restart Handoff

**Last updated:** 2026-08-02
**Status:** Foundation + measurement complete. Correctness verdict is an authoritative FAIL. Fast engine not built. Next concrete milestone: **clean 1-step CF++ M5 baseline.**
**Validation as of this snapshot:** 140 tests pass, Ruff clean, strict mypy clean.

> Read this first on restart. It captures where the project stands, what was decided, the hard-won lessons, and the exact next steps. Companion docs: `liveframe_plan.md` (vision), `liveframe_research_track.md` (Delta Scheduling proposal), `liveframe_execution_plan.md` (ladder/gates), plus the M0/M1 reports below.

---

## 1. One-paragraph situation

LiveFrame set out to demonstrate *real-time streaming video diffusion on Apple Silicon, honestly measured*. We built a world-class measurement + correctness apparatus and ran it. Two results are now in, both sobering: (a) the MLX/Metal port is **very slow** (contaminated baseline ~0.185 FPS on the heavy Rolling Forcing model), and (b) it **fails cross-platform output parity** — the sealed holdout returned an authoritative FAIL. Neither result blocks the actual next step. The generic *acceleration engine* has not been built, and the core acceleration hypothesis (Delta Scheduling has useful **wall-clock** headroom) has **never been tested on a single model**. The immediate next move is a clean 1-step CF++ baseline on M5 — one measurement that tells us whether we're chasing 3×, 10×, or an unrealistic 50×.

---

## 2. The three-piece mental model

1. **Speedometer (BUILT)** — honest throughput/latency/memory harness; separates DiT denoise / cache refresh / VAE / metrics / overhead; detects thermal & memory problems.
2. **Safety test (BUILT)** — sealed cross-platform holdout: CUDA vs MLX on byte-identical noise, three-layer verdict (trajectory canary / intrinsic-quality non-inferiority / pairwise identity). **The current MLX implementation FAILED this test.**
3. **Fast engine (NOT BUILT)** — no generic acceleration engine yet. Rolling Forcing is heavy/slow; 1-step CF++ converted but not benchmarked/validated on M5; Delta Scheduling is still only a proposal with no oracle run.

"Generic fast diffusion" ≠ one magic trick. The realistic generic layer is a **backend-neutral engine** that runs different distilled models, schedules across frames/chunks, reuses compute only when *measured* safe, quantizes per hardware, supports CUDA+MLX, and rejects optimizations when quality drifts.

---

## 3. What is BUILT and validated

- **Causal MLX execution primitives** — absolute-offset RoPE, per-layer KV caches, cached cross-attn, flow→x0, Rolling Forcing anchor cache (24-frame physical / 21-frame attention / 3-frame re-RoPE'd anchor) with eviction/sink. LiveFrame-owned; does not patch site-packages.
- **Scheduler accounting** — honest forward-pass counts: 2-step = 44 denoise + 21 refresh = **65** total; 1-step = 24 + 21 = **45**. (The refresh pass is a full transformer forward per block and does NOT shrink with fewer denoise steps — see Lessons.)
- **H100 numerical parity** (earlier phase): single-block cos 0.99992; full 2-step 21-block cos 0.99975; rolling 60.56s (84 windows, eviction active, anchor re-RoPE) cos **0.998944** — which **misses the frozen 0.999 numerical canary**. Cache cursors exact throughout.
- **Frozen quality protocol** (pre-registered, digest-bound, fail-closed) — 16 prompts × 4 strata × 2 horizons (15s/60s); 8 calibration seeds + 4 disjoint holdout seeds; three-layer verdict; cross-seed intrinsic floors; per-horizon floor derivation; identity tripwires as pre-registered same-seed bounds (NOT cross-seed floored).
- **Calibration** — 256 H100 teacher videos, checksum-valid. v1 sanity **failed closed** on one cell (`dense_motion_03/60s/motion_clip_distance_mean`, robust scale 0.000916 < 0.001). Resolved *principledly* via **v2 amendment** (epsilon-dominated cells → diagnostic, uniform; `motion_lpips` recognized as high-motion motion discriminator; **no threshold moved**, coverage preserved, holdout still sealed). v2 sanity passes all 32 cells.
- **Shared serialized noise** (`shared-rolling-noise.v1`) — both CUDA reference and MLX candidate *consume the same bytes*; byte-identity enforced at eval, not regenerated. This is the fix for the earlier "regenerate noise per side" hazard.
- **Sealed holdout captured both sides** — 128 H100 references + 128 MLX candidates, all checksum-valid, all stochastic payloads match frozen source plans.
- **Reports written:** `m0_correctness_report.md` (portability/correctness), `m1_sealed_holdout_report.md` (the FAIL), calibration/pilot/reference/candidate reports.

---

## 4. THE KEY RESULT — sealed holdout: authoritative FAIL

Ran once, published atomically, **will not be retuned or rerun** (per `retuning_policy: "no changes after any holdout result is visible"`). The verdict is **clean** — swap contamination affects timing only, not output bytes, so the quality/identity numbers are trustworthy.

**Verdict:** `passed=false`, 32/32 prompt-horizon cells failed. Bundle `sha256:41a228ce…`.

| Layer | Result |
|---|---|
| **Pairwise identity** (MLX == CUDA trajectory, same noise) | **0/128 videos, 0/32 cells** — decisive, not borderline |
| ↳ pairwise CLIP | 29/128 pass |
| ↳ pairwise LPIPS | 2/128 pass |
| ↳ temporal CLIP | 0/128 pass, nearest miss **0.0998** beyond bound |
| **Intrinsic quality** (candidate vs teacher floors) | 90/128 videos pass, 38 fail; **19/32 cells pass, 13 fail** |
| ↳ by horizon | 15s 49/64, 60s 41/64 (degrades with horizon) |
| ↳ failure spread | diffuse across all strata/metrics — **no clean "viable M5 lane"** |

**Interpretation:** Cross-platform AR trajectory reproduction is **not achievable** with BF16 on Metal — at *any* horizon, even 15s (small per-step float differences compound through the AR loop into different videos; temporal motion diverges worst). MLX output is *often independently plausible* (70% intrinsic pass) but is neither CUDA-equivalent nor cleanly teacher-quality. The three-layer design earned its keep: it separated "wrong trajectory" (identity) from "still-plausible video" (intrinsic).

**Strategic consequence:** the strong **portable/reproducible-across-backends** claim is dead and not fixable by tuning or shorter horizons. This is a genuine, novel, publishable **negative result** (nobody has quantified cross-platform AR video-diffusion divergence). It does **NOT** block the speed thread — see §6.

---

## 5. Speed status (still open, no clean number yet)

- **M5 contaminated baseline** (Rolling Forcing, wrong+heavy model, under Docker memory contention + swap): 15s median **0.287 FPS**, 60s median **0.185 FPS** (~87 min/clip, ~1% of real-time). **Super-linear** 15s→60s scaling (6.35× time for 4.09× frames) = swap signature. Thermal-drift diagnostic (final-16 17.8% slower) **invalid** (confounded by swap/restart). Confirmed genuine M5 Metal GPU (`mlx-metal-m5-max`, `applegpu_g17s`), not CPU/Docker fallback.
- **H100 (works):** 1-step CF++ ~19.3 FPS, 2-step ~15.3 FPS — but only a 3.4s native-window **burst**, not sustained 60s.
- **Do NOT anchor targets on 0.185** — contaminated AND wrong model. The clean 1-step baseline replaces it.
- Peak-memory capture and a fail-closed M5-device assert were recommended for the candidate runner but **not yet confirmed added** — add before the clean run.

---

## 6. Agreed next steps (with corrections locked in)

Direction agreed: **do NOT train a custom model first.** Build a model-agnostic, training-free acceleration algorithm against *fixed existing models*; a co-designed model is a separate later track (Track B). Corrections adopted over the last strategy pass:

1. **Clean 1-step CF++ M5 baseline** — no swap (instrument peak memory; confirm it fits without swapping), single session, on the *1-step CF++* model (not Rolling Forcing). Report 15s and 60s separately. **This is the gating measurement**; it says whether we need 3×, 10×, or ~50×.
2. **Quantization early** — int4/int8 on M5 neural accelerators is the reliable, highest-probability lever and the user's home turf. Bank it as a separately-attributed accelerator *before* betting everything on the DS research bet (insurance if DS dies).
3. **DS oracle on CF++ FIRST** — offline measurement on the *unchanged* model. Needs almost none of the "generic engine" architecture. Gate on **WALL-CLOCK reclaimable, not FLOP-reclaimable** (ragged/masked skipping on Metal often doesn't convert to time; FLOP redundancy looks encouraging on distilled models but wall-clock is at risk). Require headroom on the **shippable model class**, not "any two classes."
4. **Widen only if it clears** — second architecture + start the generic engine/adapter layer.
5. **Custom co-designed model last** (Track B) — as an upper-bound control, never as the first move.

### Critical framing correction: which parity matters
- **Cross-platform parity (MLX == CUDA)** — DEAD (holdout), and **NOT needed** for the speed engine.
- **Within-backend optimization parity (optimized-MLX vs full-precision-MLX)** — this is what the fast engine's quality gate uses. Each optimization must preserve *that backend's own* baseline. You never need MLX to match CUDA. Absolute quality is still checked vs teacher floors; cross-platform divergence becomes a *diagnostic*, not a gate.

---

## 7. Reality-check on the 50× ambition (stretch, not expectation)

- **Model switch (4-step RF → 1-step CF++) ≈ 1.7–2×, not 3–6×** — denoise shrinks ~4× but **refresh does not shrink** (refresh ≈ 43% of DiT in the profile; 1 forward/block regardless of step count). The 3–6× only appears if you *also* cut refresh, which is a separate, quality-risky lever — don't double-count.
- **Levers cannibalize** (bottleneck-shifting), so realistic *same-output* stacked gain ≈ **5–15×**, not the naïve product. 50× almost certainly needs **output-changing** levers (lower resolution/fps or model family).
- **Resolution drop is a product tier, not a free speedup** — changes the output and needs floors recalibrated (current floors at 480×832).
- **Fast-clip mode does NOT dodge the throughput wall** — a 5s clip in <5s ≈ 16 FPS generation, same as streaming (a bit easier only via batching). Genuine throughput reliefs: lower resolution, lower output fps (8fps anime halves frames), or a longer render budget.
- Decision thresholds from the clean baseline: <0.5 FPS → 50× unrealistic without changing resolution/model; ~1 FPS → 8 FPS needs 8× (hard); ~3 FPS → 2.7× (encouraging); ~6 FPS → 16 FPS becomes a serious target.

---

## 8. Hard-won lessons / gotchas (do not relearn these)

- **Verify canonical digests freshly on restart.** Digests are canonical (sorted compact JSON), not raw-file SHAs. Local/pod sync drifted mid-session once (a stale `b317…` protocol read; the true canonical was `aef6…`). Always recompute; don't trust an in-context digest.
- **The retuning line is holdout visibility.** Calibration-side fixes are allowed *before* any holdout is opened (that's how the v2 sanity amendment was legitimate). After a holdout verdict is visible: nothing changes. The digest chain (protocol → calibration → bundle → holdout) enforces this mechanically.
- **A sanity/adequacy floor ≠ a quality pass/fail bar.** The v2 amendment fixed the *test design* (epsilon-dominated cells can't be authoritative gates) without moving any number — that's principled; lowering 0.001→0.0009 would have been cheating.
- **Serialize noise once, consume on both sides; verify byte-identity at eval.** Never regenerate per-backend (that's the forbidden cross-runtime seed equivalence). Both runners now read `shared-rolling-noise.v1`; the evaluator byte-compares before any metric.
- **Pin the instrument.** Metric models (CLIP/LPIPS) are verified by state-dict SHA against `metric_provenance`, so a different checkout can't silently shift a verdict.
- **MLX-Metal cannot run inside a Linux Docker container on Mac** (no Metal passthrough). The contaminated run was native but had a Docker container competing for the 36 GB unified memory → swap. Ensure a clean, no-swap environment for real benchmarks.
- **Swap contaminates timing, not output bytes.** So the FAIL verdict is trustworthy even though the FPS number isn't.
- **DS gate must be wall-clock.** FLOP-reclaimable overstates payoff on distilled models; ragged-skip overhead is a bigger fraction of a cheap 1-step forward.
- **This project's failure mode is building scaffolding before testing the core hypothesis.** Resist. The DS oracle is a cheap offline measurement — run it on the product model before building any generic engine.

---

## 9. Key artifacts & digests (current canonical)

| Artifact | Digest / path |
|---|---|
| Protocol v1 (canonical) | `sha256:aef6a7ad9426c97a9ae7f99c9a7a4f93799a574ca9e88d341da430c60ca75c52` |
| Run manifest | `sha256:8967e70cbc3978544eea92428d2dda78044ebbcaae91e933271f7e1abd72fdc4` |
| Protocol v2 | `sha256:f1b39f36a0ed6f361aaabf38062868d5f3e12a953627c67b1a736c8d56148b93` |
| Policy bundle v2 (verdict binds here) | `sha256:41a228ceac16a026c7771d2e2c496714e5476c8e9287d3da3d15a78e993a9870` |
| Sanity v2 (passes) | `sha256:e0308cdb235264112c049d46b27e6370b3d0e0c7abc5f92ccbd79886d62c5841` |
| Shared noise index | `sha256:3fbfbf2896e566f663a1b781e20872c393b25e0d6333d33268b3c069e4b72a17` |
| MLX candidate batch | `sha256:5019c73470eb11f63ff8a525fa8081a7f02b4ea764283f071df0c52d51bb57e7` |
| H100 reference batch | `sha256:10af5568b5c42ac9d081621a0f63862ec4c96c8751d03c8b12e775aa6dab337a` |
| Holdout input | `sha256:27ab87ae15fd08a631b2da685fda395fdfacb57f61da2cfb947ee2f2698fc1d3` |
| **Holdout verdict (FAIL)** | `sha256:146cc868534f1a0955f7d8e49f899df0da63137c4ecb83acd68c1487993f3bcd` |
| Rolling Forcing checkpoint (PyTorch) | `sha256:1f123dce60e60424a9c6d949643176840fceffb947932997bcf1202d93656d98` |
| Rolling Forcing MLX model | `sha256:1f15d166a8a60bed70c2a8ab2923f40bd2b6b53ed2e0be87905bf45d0666d971` |
| CLIP revision (pinned) | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |

**Raw corpus:** 128 H100 references + 128 MLX candidates preserved locally (`artifacts/h100-holdout-results`, `artifacts/mlx-holdout-results`); full raw ~208 GB on the RunPod persistent volume. **Keep the persistent volume; the compute pod can be stopped.**

**Key source modules:** `backends/mlx_causal.py`, `causal_cache.py`, `scheduler.py`, `rolling_forcing.py`, `quality_stochastic.py`, `mlx_quality_runner.py`, `perceptual_policy.py`, `quality_calibration.py`, `quality_capture.py`, `quality_holdout.py`, `mlx_rolling_parity.py`, `rolling_perceptual_parity.py`, `tools/evaluate_quality_holdout.py`. Converter: `converters/cfpp_to_mlx.py`.

---

## 10. First actions on restart

1. `cd /Users/kumarjha/Code/liveframe && .venv/bin/python -m pytest -q` — confirm green (expect ~140).
2. Recompute the canonical digests in §9 and confirm they still match (sync-drift guard).
3. Confirm the RunPod compute pod is **stopped** and the persistent volume retained.
4. Finalize `m1_sealed_holdout_report.md` as a **standalone negative-result deliverable** — it's the tangible artifact that exists now.
5. Begin the **clean 1-step CF++ M5 baseline** (§6.1): add peak-memory capture + M5-device assert, ensure no swap, single session, 15s and 60s separately, within-backend quality gate. This number gates the entire fast-engine decision.

---

## 11. Open decision awaiting the baseline

Whether to invest in M5 optimization at all vs. pivot to **"DS acceleration on H100 (where the substrate works) + honest M5-characterization as the Apple-Silicon result."** The clean 1-step baseline is the cheapest, most decisive input to that decision. If it lands far from viable and DS shows no wall-clock headroom, the honest, publishable outcome is: the methodology + the two negative results (cross-platform AR divergence; the Apple-Silicon throughput wall) + working H100 numbers. That is still a real contribution.
