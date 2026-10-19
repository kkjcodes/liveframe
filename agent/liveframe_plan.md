# LiveFrame — Real-Time Video Generation: Project Plan (v2)

> **Superseded on 2026-10-06.** The active end goal and delivery order are defined in
> `liveframe_product_mission.md`. This document remains as historical planning context; publication
> and an undifferentiated real-time demonstration are no longer project success criteria.

**Project name:** **LiveFrame** · `pip install liveframe` · repo: `kkjcodes/liveframe` · tagline: *"Real-time video generation, honestly measured."*

**Author basis:** kkjcodes / m5-flux-block-cache-benchmark · dev.to post "A 1.125x FLUX speedup was real..."
**Goal:** **Real-time streaming video generation — proven, honestly measured, and publicly demonstrated.** Cloud GPU first (where real-time is reachable fast), then the unclaimed frontier: real-time streaming video diffusion on Apple Silicon. Ships as a distributable package.
**Supersedes:** v1 (sub-5s clip plan). Clip-mode sub-5s remains as a byproduct milestone, not the headline.
**Date:** 2026-07-19

---

## 1. What "real-time" means here (the proof standard)

Every public real-time claim gets to be vague. Ours doesn't. A configuration counts as real-time only if, measured by the harness:

- **Throughput:** sustained generation FPS ≥ output playback FPS over a **≥ 60-second continuous stream** (not a 3-second burst).
- **Latency:** time-to-first-frame ≤ 1s, and p95 per-chunk latency below the chunk's playback duration (no mid-stream stalls).
- **Thermal honesty:** sustained-state measurement, not cold-start — the FLUX harness's thermal/interleaved discipline applies directly. A MacBook that does 20 FPS for 30 seconds and 9 FPS after throttling is a 9 FPS machine.
- **Quality:** per-frame LPIPS/CLIP vs the many-step teacher, temporal-consistency metrics, and **long-horizon drift** curves (AR streaming's known failure mode), all reported against cross-seed floors, stratified by content class including high-motion.

This proof standard *is* the differentiator. Real-time demos exist; real-time demos with pre-registered gates, sustained-thermal numbers, and calibrated quality floors do not.

## 2. Why this is provable, and where the novelty is

The bidirectional→autoregressive distillation lineage has already crossed real-time on NVIDIA:

| System | Base | Throughput | Hardware | Availability |
| --- | --- | --- | --- | --- |
| CausVid | Wan-class AR distill | ~9.4 FPS | 1×H100 | code released |
| Self-Forcing | Wan2.1-1.3B | ~17 FPS | 1×H100 | open |
| **Causal Forcing / ++** (ICML 2026) | Wan2.1-1.3B | 17 FPS, 0.69s latency | 1×H100 | code + weights (thu-ml) |
| **StreamDiffusionV2** (MLSys 2026 oral) | Wan2.1/CausVid 1.3B & 14B | **64.5 FPS @ 512², 1-step; 58 FPS for 14B**; TTFF 0.37–0.47s | 1×H100 | open release stated |
| MirageLSD (Decart), Seaweed APT2 | proprietary | live-stream real-time | clusters | closed |

Three consequences:

1. **Cloud real-time is a reachable early milestone, not a moonshot.** Rent H100s, stand on Causal Forcing + StreamDiffusionV2, and the work is *verification and demonstration at our proof standard* — deliverable within weeks. This is the "PROVE IT" moment, front-loaded.
2. **The open frontier is Apple Silicon.** Nobody has demonstrated real-time (or even fast) streaming video diffusion on a Mac. FP8 fails on Metal, official pipelines NaN on MPS, and the M5's GPU neural accelerators are uncharacterized for video DiTs. First real-time-class streaming video diffusion on Apple Silicon is a genuinely novel, ownable result.
3. **The system techniques are training-free** (rolling KV cache, sink tokens + RoPE refresh, motion-aware noise scheduling, SLO-aware chunking) — squarely in your wheelhouse. The models are Apache-2.0. **We do not train; we build the streaming engine and the proof.**

## 3. Budget math (hypotheses → Phase 0 measures them)

Reference point: 64.5 FPS @ 512×512, 1-step, Wan2.1-1.3B on H100 (~1 PFLOPS effective). M5 Max effective throughput with neural accelerators is unknown publicly — call it 25–60 TFLOPS until measured, i.e. **~15–40× less than H100**.

Naive M5 Max port of the same config: **~2–4 FPS**. The multiplicative ladder to real-time-class on-device:

| Lever | Multiplier (est.) | Notes |
| --- | --- | --- |
| 1-step AR student (given) | — | baseline already assumes it |
| int4/int8 quantized matmuls on M5 neural accelerators | 1.5–3× | your home turf; FP8 impossible on Metal, int-quant is the route |
| Resolution 512²→~360p | ~2× | fewer tokens, quadratic attention relief |
| Sliding-window/sparse attention Metal kernels | 1.5–2.5× | attention dominates at streaming sequence lengths |
| Frame interpolation (generate 8–12, present 16–24) | ~2× | cheap RIFE-class interpolator; disclosed honestly as a tier |
| Overlap VAE decode with next-chunk DiT | 1.1–1.3× | unified memory makes this natural |

Stacked midpoints put **8–16 FPS presented at ~360p on M5 Max** in credible range, with ~16–24 FPS as the stretch. That is demonstrable real-time-class streaming on a laptop. Every number above is a Phase 0/2 measurement target, not a promise.

Cloud cost envelope: H100 at ~$2–3/hr; Phase 1 needs ~40–80 hrs (~$150–250); ongoing verification ~20 hrs/mo. Cheap relative to the result.

## 4. Goal ladder (pre-registered, kill-ruled)

- **RT-0 (cloud, weeks 2–4):** Reproduce ≥17 FPS (Causal Forcing) and ≥30 FPS (StreamDiffusionV2 config) sustained 60s on 1×H100, at the §1 proof standard, with quality floors. Public artifact: report + live demo recording.
- **RT-1 (M5, weeks 4–7):** Streaming AR generation *works at all* on MLX/M5 Max — correct output, any FPS. (The correctness landmines — Metal numerics, VAE NaNs — live here.)
- **RT-2 (M5, weeks 7–10):** ≥ 8 FPS sustained @ ~360p on M5 Max (generated, pre-interpolation), TTFF ≤ 2s.
- **RT-3 (M5, weeks 10–14):** ≥ 16 FPS presented, 60s sustained, thermals disclosed → **claim: real-time-class streaming video diffusion on Apple Silicon, first public demonstration.**
- **Byproduct:** clip mode (5s @ 480p) falls out of the same engine: sub-5s on H100 immediately; M5 Max clip-mode number published with the same honesty.

Kill/pivot rules: if RT-2 misses after the full lever stack, pivot the on-device claim to "real-time preview stream + fast clip finalize" (still novel, still demonstrable) and publish the negative result with the budget table — the FLUX playbook.

## 5. Phases

### Phase 0 — Baselines and budget table (weeks 1–2)
1. Rent 1×H100; run Causal Forcing and StreamDiffusionV2 reference stacks; capture sustained-FPS/latency/quality baselines with harness discipline (port `stability.py`/`timing.py`/thermal probes; add streaming metrics: per-chunk latency distribution, TTFF, drift-over-time).
2. On M5 Max: stand up `mlx-video` (Wan2.1-1.3B, Wan2.2+Lightning); convert Causal Forcing / Self-Forcing student weights to MLX; dense, unoptimized chunk-AR run for the first on-device row.
3. Deliverable: measured budget table replacing every estimate in §3; pick primary student checkpoint.

### Phase 1 — Cloud real-time, proven and demonstrated (weeks 2–4) → RT-0
1. Wrap the winning cloud stack in the package API (`stream(prompt)` generator + `clip(prompt)`).
2. Run the full proof standard; publish interactive demo (prompt switching mid-stream) as a recorded + live artifact.
3. Writeup #1: "Real-time video generation, measured honestly" — reproduction numbers vs papers' claims, quality floors, what survives sustained load.

### Phase 2 — The M5 streaming engine (weeks 4–10) → RT-1, RT-2
1. MLX port of the causal student: rolling KV cache, sink tokens + RoPE refresh, chunked scheduler (SLO-aware, adapted from StreamDiffusionV2's training-free system design).
2. Quantization (int8→int4) gated by perceptual metrics; neural-accelerator shape tuning; `mx.compile`/fused kernels.
3. Sliding-window / block-sparse attention Metal kernels — biggest single lever; pre-registered gate: ≥1.5× step time at LPIPS within floor margin.
4. Streaming VAE decode overlapped with DiT compute; motion-aware noise control; optional interpolation tier (always disclosed).
5. Per-op parity tests vs PyTorch reference throughout (correctness harness, FLUX-style discipline).

### Phase 3 — Streaming quality harness (parallel, weeks 2–10)
Cross-seed floors for per-frame LPIPS/CLIP + temporal-consistency; long-horizon drift curves (60s+); content-stratified suite with high-motion and dense-scene classes; curated side-by-side and drift galleries in reports. Every FPS claim ships with its quality row.

### Phase 4 — Package, demo, publications (weeks 10–14) → RT-3
1. Package (`pip install liveframe`): MLX backend (Apple) + CUDA backend (NVIDIA); `stream`/`clip` APIs; `liveframe benchmark` subcommand that prints the user's own sustained budget table; live demo app (local web UI, type-to-steer).
2. The RT-3 demonstration: 60-second sustained live generation on the M5 Max MacBook, on camera, thermals on screen. That's the demo that travels.
3. Writeups: (#2) M5 neural accelerators vs video DiTs — first public characterization; (#3) real-time on Apple Silicon result (or the honest miss + budget table); credit Causal Forcing, StreamDiffusionV2, FastVideo, Wan, mlx-video throughout.

## 6. Top risks
1. **MLX/Metal numerics** (NaN-class failures) — parity tests per op; budgeted in RT-1, not discovered in RT-2.
2. **M5 throughput lands low-end** (~25 TFLOPS effective) — interpolation tier + 320p floor keep a demonstrable claim; pivot rule in §4 absorbs the worst case.
3. **AR drift over long horizons** — sink tokens/RoPE refresh are training-free mitigations; drift curves make it visible instead of hidden.
4. **Closed-source surprises** (StreamDiffusionV2 release slips) — Causal Forcing + Self-Forcing are already open; SDv2's techniques are reimplementable from the paper.
5. **Scope creep** — milestone lock per phase; the Grok-chat reconciliation stands: warmup done, adaptive gate killed, step-internal caching retired; all new caching effort targets streaming-shaped redundancy only.

## 7. This week
1. New repo skeleton; port timing/thermal/stability core from `m5_flux_benchmark`; define streaming metric schema.
2. Book ~20 H100-hours; reproduce Causal Forcing's 17 FPS as-is (no harness yet) to de-risk the stack.
3. `mlx-video` running on the M5 Max; first Wan2.1-1.3B on-device row.
4. Weight conversion spike: Causal Forcing student checkpoint → MLX safetensors; dense single-chunk forward parity vs PyTorch.

---

## Appendix: carried over from v1 (still true)

- Step-internal block caching is retired: a 1-step AR student has no step-to-step redundancy to cache. Your caching/scheduling expertise redeploys to streaming-shaped redundancy (KV/chunk reuse, attention sparsity, compute scheduling across chunks — the "trajectory-aware compute scheduling" your post pointed at).
- The measurement infrastructure is the moat. Port it, don't rebuild it.
- Human tolerance for divergence remains the license: gate on perceptual + temporal metrics vs calibrated floors, never pixel identity.
- Grok-chat status: warmup ✓ shipped · adaptive residual gate ✗ killed (pre-registered) · perceptual loop ✓ shipped · step reduction → superseded by adopting distilled AR students · Apple-Silicon-plus-rigor positioning → now the core of the real-time claim.
