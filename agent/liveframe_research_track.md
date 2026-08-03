# LiveFrame Research Track — "Delta Scheduling": Motion-Proportional Compute for Real-Time Video Diffusion

**Status:** research proposal (pre-registration style) · companion to the LiveFrame project plan v2
**Scope:** a new, training-free algorithm tested first on the one-step CF++ streaming model. A
generic video-DiT claim is attempted only after the CF++ wall-clock oracle passes and the same
policy transfers to a second architecture.
**Author basis:** builds on kkjcodes' block-residual caching formulation, residual instrumentation, and cross-seed-floor evaluation methodology from `m5-flux-block-cache-benchmark`.
**Date:** 2026-07-19

---

## 1. The observation the algorithm is built on

In streaming video diffusion, **compute per output frame is constant, but information change per frame is not.** A talking-head clip spends 90% of its pixels on a static background; an animation spends whole seconds on held cels, static backgrounds, and repeated motion cycles. Yet every chunk, every token, every block gets the same denoising work.

Your FLUX project already proved the two halves of the license needed to exploit this:

1. **Block residuals are reusable** when the underlying content isn't changing (1.125x on images, where *nothing* is changing between steps except the trajectory itself).
2. **Humans tolerate divergence** — the quality gate that matters is perceptual + temporal, calibrated against floors, not pixel identity.

Video hands us what FLUX never had: a *ground-truth change signal*. Between consecutive latent frames/chunks, the delta is measurable, cheap, and semantically meaningful (it is literally the motion). The algorithm is therefore:

> **Delta Scheduling: measure where the video is changing, and spend the compute budget only there — reusing block residuals and KV state for static regions, allocating denoising steps per chunk in proportion to motion, under a hard real-time budget.**

Think foveated rendering, but the fovea is motion instead of gaze.

## 2. Why this survives distillation (and step-caching didn't)

Step-redundancy methods (DeepCache, TeaCache, PAB, your block cache) die at 1–4 steps: there are no redundant steps left to skip. **Spatiotemporal redundancy is orthogonal to step count and survives distillation untouched** — a 1-step AR student still recomputes the entire static background of every chunk from scratch. That is the compute Delta Scheduling reclaims. This is the structural reason the idea is not "another cache": it targets the redundancy axis that the field's own step-deletion success has left as the dominant remaining waste.

## 3. The algorithm (three mechanisms, one controller)

Integration point: block-level hooks on the video DiT — the same architecture as your mflux harness (`_BlockCacheController` generalizes almost directly).

**M1 — Static-region cross-chunk reuse.** Partition the latent into spatiotemporal tiles (block-granular, not token-granular — ragged per-token branching is a Metal/CUDA performance trap). For tiles whose latent delta vs the previous chunk falls below a calibrated threshold: reuse cached per-block residuals and KV entries from the previous chunk instead of recomputing. High-delta tiles compute normally. This is your block-residual formulation transposed from the step axis to the time axis.

**M2 — Per-chunk step allocation.** The AR student supports 1–4 steps. Allocate steps per chunk by measured chunk-level motion energy: high-motion chunks get 3–4 steps, near-static chunks get 1 (or even a partial pass). Composition-critical chunks (first chunk, scene changes detected by delta spikes) always get full compute — the direct descendant of your warmup insight: *protect composition-forming compute, harvest the rest.*

**M3 — SLO budget controller.** Real-time gives a hard budget: each chunk must complete within its playback duration. The controller holds a per-chunk FLOP budget and spends it greedily by predicted benefit (delta magnitude × block sensitivity), degrading gracefully: drop steps on static chunks first, then widen reuse thresholds, then reduce sparsity refresh rate. This converts "best effort" acceleration into a *deadline-guaranteed* system — which is what makes it a real-time algorithm rather than a speedup trick. (StreamDiffusionV2's SLO batching schedules *work*; this schedules *the model's own computation*. Different layer.)

**Signals (all cheap, all training-free):** latent frame deltas (free — the latents exist), per-block residual norms (your existing instrumentation), KV drift norms, and optionally a low-res decoded luminance diff as a sanity signal. No learned components in v1.

## 4. The lesson from the residual-gate kill: oracle before policy

The FLUX adaptive gate failed because the signal didn't discriminate where it needed to. That mistake does not get repeated. **Phase R0 is an oracle study, run before any policy is built:**

- Generate with full compute; log per-tile latent deltas, per-block residuals, KV drift at every chunk.
- **Oracle reuse experiment:** replay generation, forcing reuse on exactly the tiles where post-hoc ground truth says reuse was safe (output difference below floor-calibrated threshold). This measures the *ceiling*: what fraction of block compute is reclaimable at acceptable quality, per content class.
- **Signal validation:** does the cheap delta signal predict the oracle's safe-reuse set? (Precision/recall, per content class.) If the signal can't find the redundancy, the policy is dead before it's built — kill or find a better signal, at diagnostic cost instead of full-sweep cost.

Pre-registered viability gate for continuing past R0: on unchanged one-step CF++, the oracle
shows **≥30% reclaimable block compute at quality within floor margin** on both
limited-animation/held-frame and dialogue/minimal-motion content; the delta signal achieves
**≥0.8 precision and ≥0.6 recall** against the oracle set; and a realizable dense/compacted path
reduces measured end-to-end wall time by **≥17%** on at least one supported backend. Wall time
includes delta calculation, selection, compaction, cache movement, synchronization, and the
interaction with the existing AR KV cache. A FLOP-only saving or a saving confined to a heavy
multi-step model is a STOP. Below any threshold, the honest writeup is the CF++ spatiotemporal
reuse ceiling and the research branch stops before an online policy or generic engine is built.

## 5. Pre-registered hypotheses

- **H1 (ceiling):** In one-step CF++ streaming generation, ≥30% of block-level compute is
	oracle-reclaimable on static-heavy content and the realizable oracle path reduces end-to-end
	wall time by ≥17%, at quality within the frozen margin.
- **H2 (policy):** The training-free delta-signal policy captures ≥60% of the oracle ceiling, yielding **≥1.4x sustained FPS on animation-class content and ≥1.15x on general content**, with temporal-consistency and drift metrics within floor margins, measured interleaved on both M5 Max and H100.
- **H3 (generality):** The same hooks + policy, untuned beyond thresholds, transfer to a second model family (Wan2.2-5B first; stretch: LTX class) with ≥half the relative gain — establishing "general-purpose" honestly rather than by assertion.

Kill criteria mirror the FLUX playbook: H1 fails → publish the ceiling study, stop. H2 misses timing or quality gates after two policy iterations → publish the negative result with the oracle-vs-policy gap analysis. No unbounded sweeps.

## 6. Why this ties directly to the animation project

Animation is the **best-case content class** for Delta Scheduling, not an afterthought: held frames, static painted backgrounds, flat-shaded regions with tiny deltas, cyclic motion. And the confirmed workload is an exact fit:

**Animation project workload (confirmed):** Wan text-to-video, predominantly 2D anime style, clips of **5–15 seconds**, stitched into ~**1-minute** sequences.

That workload shapes the research track in four concrete ways:

1. **Eval suite:** the first-class animation content class is 2D/anime-style T2V prompts — limited-animation style with holds, flat-shaded characters over painted backgrounds, dialogue scenes (mouth/eye motion only), and cyclic motion (walk loops, hair/cloth sway). These are the highest-redundancy scenarios DS exists for; flat-shaded regions also stress the seam-artifact metric (banding shows on flat color long before it shows on texture).
2. **Clip length is the forcing function for AR streaming:** 5–15s exceeds Wan's native ~5s/81-frame window, so long-clip generation *requires* the chunked AR mode LiveFrame is built on — the animation project isn't just a beneficiary of the streaming engine, it's structurally dependent on it. Drift curves must be measured to 15s per clip, not just 5s.
3. **Stitching adds a cross-clip consistency requirement:** a 1-minute sequence stitched from 4–10 clips lives or dies on style stability across clip boundaries. Add stitch-boundary metrics (color-palette drift, CLIP-style-embedding drift across boundaries, last-frame→first-frame perceptual distance) and evaluate conditioning strategies for continuation (final-chunk latent or last-frame image conditioning as the next clip's anchor). DS interacts favorably here: reused static-region residuals are, by construction, *more* stable across a stitch than freshly sampled ones — worth measuring as a possible free consistency win (pre-registered as an exploratory question, not a promised result).
4. **The headline demo writes itself:** a 1-minute anime sequence generated clip-by-clip in real time on the M5 Max — algorithm paper, LiveFrame package, and the animation project's pipeline in a single artifact.

## 7. Relation to prior art (what's actually new)

| Prior work | What it exploits | Why Delta Scheduling is different |
| --- | --- | --- |
| DeepCache / TeaCache / PAB / your block cache | redundancy **across denoising steps** | dead at 1–4 steps; DS targets **spatiotemporal** redundancy, which survives distillation |
| RAS (region-adaptive sampling, images) | spatial redundancy within one image | DS operates on *temporal* deltas in streaming AR, with reuse (not just step allocation) and a deadline controller |
| TokenFlow | cross-frame feature propagation for **editing consistency** | DS reuses for **compute reduction** under a budget; generation, not editing |
| VSA / sparse attention | attention-pattern sparsity (trained) | orthogonal; DS composes with sparse attention (M1 reduces work *before* attention sparsity applies) |
| StreamDiffusionV2 | system-level SLO batching, motion-aware **noise** | DS schedules the **model's compute itself** (blocks, steps, reuse); composes with their system layer |
| Self-Forcing / Causal Forcing | step deletion via distillation | DS runs *on top of* their students; it is the post-distillation acceleration layer |

Honest framing for the paper: individual ingredients have precedents; the contribution is (a) the streaming/AR few-step setting where step-caching is dead, (b) reuse + step allocation + deadline controller unified under one measured budget, (c) the oracle-first methodology and calibrated floors — and (d) demonstrated on both H100 and Apple Silicon.

## 8. Experimental plan

- **R0 (2 wks):** after one clean CF++ baseline, add only the instrumentation needed for the
	offline CF++ oracle, realizable wall-clock ceiling, and signal validation. Run quantization as
	a separate, independently measured track in parallel. Gate: §4.
- **R1 (2–3 wks):** M1 static-tile reuse alone, interleaved timing + full quality battery. Gate: ≥1.2x on animation class, floors held.
- **R2 (2 wks):** + M2 step allocation; then M3 budget controller; ablate each (the harness's policy-set machinery does this naturally).
- **R3 (2–3 wks):** transfer to Wan2.2-TI2V-5B (H3); M5 Max end-to-end integration into LiveFrame tiers (`--delta-sched` flag); 15s-clip drift curves + stitch-boundary consistency evaluation for the 1-minute pipeline.
- **R4:** writeup — target venue-grade (the oracle methodology + negative-result rigor is what reviewers rarely see), plus the dev.to narrative post and the animation real-time demo.

Sequencing: clean one-step CF++ baseline first; int8/int4 quantization and the offline CF++ oracle
then proceed in parallel. R0 uses the existing model loop plus narrow instrumentation only. It
does not wait for, or authorize building, a generic engine, adapter framework, broad model matrix,
or custom model. A second architecture is tested only after the CF++ compute, quality, signal,
and realized wall-clock gates all pass.

## 9. Risks and failure modes

1. **Seam/ghosting artifacts at reuse boundaries** — static tiles adjacent to moving regions may shimmer. Mitigations: tile overlap + feathered blending in latent space; conservative dilation of the motion mask. Measured by a dedicated boundary-artifact metric, not just global LPIPS.
2. **Ragged compute doesn't pay on real hardware** — masked/partial-block execution may save FLOPs but not milliseconds (kernel launch and memory traffic dominate). Mitigation: tile-granular batching (compact the compute set into dense batches); pre-registered *wall-clock* gates, never FLOP gates — a lesson your timing discipline already encodes.
3. **Delta signal fails on slow global motion** (pans: everything changes a little). Fallback: chunk-level M2 still applies even when tile-level M1 yields nothing; pans get full compute, which is correct behavior.
4. **AR error accumulation interacts with reuse** — stale residuals may compound drift. The drift-curve methodology (LiveFrame Phase 3) is the instrument that catches this; refresh cadence (force full compute every N chunks) is the mitigation, and its cost is in the budget model.
5. **Overhead of the signal itself** — delta computation and controller logic must stay <2% of chunk budget; measured in R0.

## 10. What "success" buys, concretely

- LiveFrame gains a differentiating algorithm, not just a port: `liveframe --delta-sched` as the thing nobody else ships.
- The M5 real-time ladder gets its missing multiplier (§3 of the plan needs ~1.5x beyond engineering levers; H2's animation-class ≥1.4x is exactly that, on exactly the content the animation project needs).
- A genuine research artifact: general-purpose, training-free, post-distillation acceleration for streaming video DiTs, evaluated with the most honest methodology in the space — the FLUX project's real legacy, scaled up.
