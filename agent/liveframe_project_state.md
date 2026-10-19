# LiveFrame — Project State & Restart Handoff

**Last updated:** 2026-10-09
**Governing direction:** `agent/liveframe_product_mission.md` (adopted 2026-10-06): make
streaming video generation measurable, reproducible, and progressively deployable — executable
deployment qualifications instead of ambiguous FPS claims. Read it first on restart.
`agent/liveframe_plan.md` (2026-07-19 real-time/publication plan) is superseded, retained as
historical context.

---

## 1. One-paragraph situation

LiveFrame built a pre-registered, digest-bound measurement apparatus for streaming video
diffusion and ran it to completion on one full acceleration campaign. The headline evidence:
a **2.6083× complete-wall speedup** (57.127 s → 21.902 s median, 95% bootstrap LCB 2.5818,
1.418 → 3.70 generated FPS) on M5 via a TAEW2.1 tiny-VAE MLX decoder in fp32-im2col mode,
**claim-eligible** under the frozen protocol — alongside three publishable negative/characterization
results. The paper was rejected by arXiv moderation (2026-09); the project pivoted to the
qualification-and-optimization product mission. The first product slice (fail-closed
qualification evaluator, synthetic backend, self-verifying ledgers, `liveframe qualify`/`verify`
CLI) is implemented and committed. Next engineering step: wrap the existing M5 CF++ path as the
first real adapter and produce the first genuine qualification ledger from evidence that
already exists.

## 2. Authoritative evidence on record (all local commits; NOTHING PUSHED without approval)

| Result | Verdict | Key artifact / commit |
| --- | --- | --- |
| M5 CF++ 1-step baseline (frozen denominator) | 57.127 s wall, 1.418 FPS, VAE 72% | `artifacts/m5-cfpp-1step-cache4-baseline.v1.json` |
| H100 CF++ denominator | 5.9609 s, 13.59 FPS | `artifacts/h100-cfpp-1step-denominator.v1.json` |
| Refresh KV-reuse mechanism | authoritative **NO-GO** (64-job H100 campaign, 0/64) | `artifacts/h100-refresh-numerical-decision.v1.json` |
| Cross-platform CUDA↔MLX identity | authoritative **FAIL** (0/128) — publishable finding | sealed holdout reports |
| MLX `conv2d` fp32 accumulation drift | isolated by bisection: 0.0854 native vs 0.00143 im2col | `outputs/taew21-block11-call000-bisection.v1.json`; upstream issue draft `agent/mlx_conv2d_issue_draft.md` |
| TAEW2.1 parity (fp16) | NO-GO v1–v3 | `outputs/taew21-port-parity.v1..v3.json` |
| TAEW2.1 parity (fp32-im2col) | **GO** (max err 0.00744, cos 0.9999996) | `outputs/taew21-port-parity.v5.json` |
| TAEW2.1 smoke / screen / formal | PASS / PASS (3.51 s median) / **PASS 2.6083×, LCB 2.5818** | `outputs/taew21-cfpp-formal-benchmark.v1.json` (67b8189) |
| 32-job paired campaign | complete, 50 GB digest-bound captures | `outputs/taew21-campaign/campaign-summary.v1.json` |
| Metric screen v1 | authoritative **FAIL** (temporal instruments) | `artifacts/taew21-metric-verdict.v1.json` |
| H.264 null control | v1 temporal tripwire **invalidated by demonstration** | `artifacts/taew21-null-control.v1.json` |
| Metric screen v2 (scale-referenced) | PASS 32/32 — **routing authority only** | `artifacts/taew21-metric-screen.v2.json` |
| Human study | **CLOSED UNLAUNCHED** (zero responses; machinery frozen, launch-ready) | `artifacts/taew21-study-closure.v1.json` |

**TAEW2.1 tier status:** speed claim stands (always cite as "2.61× complete-wall with 5.52×
decoder-only"); quality is **NOT ESTABLISHED** (v1 metric FAIL stands; v2 screen routed only;
the sole quality authority — the human study — was never exercised). Under the product schema
this is a configuration passing performance gates with quality evidence `not_evaluated`.

Study machinery (if ever reopened): amendments v6→v6.1→v6.2→v6.3 digest chain; bundle at
`outputs/taew21-study-bundle/` (manifest-bound to v6.3, hash-leak-proofed, 102 unique video
digests); power v1 (operative 486 target) + v2 (empirical: 71.9% pass at true chance);
blinded QC + analyzer with terminal-state enforcement; runbook
`agent/taew21_study_launch_runbook.md`. Six adversarial review rounds each converted an attack
into a regression test (18 dedicated tests).

## 3. Product track state (post-pivot)

Committed (2026-10-09, commits 8b3b922..3dfbfd7 + mission adoption):
- Engine: seeded `GenerationRequest`, streaming frame sinks inside the timing boundary,
  bounded-memory `clip(retain_events=False)`.
- `liveframe.qualification`: three v0 profiles (`interactive-480p`, `creative-preview`,
  `offline-streaming`), fail-closed evaluator — performance can pass but the verdict stays
  `INCOMPLETE` without quality/stability/memory/thermal evidence.
- Deterministic CPU `synthetic` backend through the real registry.
- `liveframe.qualification-run.v1` ledger: self-digest, profile binding, arithmetic checks,
  independent gate recomputation, tamper rejection, atomic writes.
- CLI: `liveframe qualify`, `liveframe qualification verify`; README quickstart.

Known gaps (from the 2026-10 review): user demand is unvalidated (mission needs an adoption
test, not just build milestones); `interactive-480p` is unpassable by any current on-device
system — lead with bottleneck diagnosis and the lower tiers; the quality gate is a bare boolean
and will need evidence-artifact bindings; v0 profiles are unfrozen by design.

**Next engineering steps (mission delivery order):** (4) wrap the M5 CF++ path as the first
real adapter — all weights/fixtures present, zero-swap preflights apply; (5) reduced M5
qualification campaign + bottleneck report; (6) CUDA adapter; (7) one measured optimization.
An M5 MLX qualification run requires the usual clean-host discipline (see §5).

**ACTIVE IMPLEMENTATION PRIORITY (user directive 2026-10-09):**
`liveframe-world-handoff.md` — extend LiveFrame into a runtime-truth evaluation harness for
action-conditioned interactive world models (`liveframe world run <model> --suite quick`,
target 0.3.0 `[world]` extra). The user chose to implement this FIRST, ahead of the mission's
M5-adapter step; it is one approach under the same qualification mission, not a replacement.
Follow the spec's milestone gates in order.

## 4. Dissemination (softened clause: demoted, not banned)

arXiv v2 submission (2026-09-17 source archive, committed) was rejected by moderation; no
further venue submissions. Remaining cheap plan: (a) post the same source to Zenodo for a DOI
timestamp; (b) file the ml-explore `conv2d` issue (draft + runnable repro in `agent/` and
`tools/`); (c) optional engineering blog post. None may consume meaningful engineering time.

## 5. Operational lessons (hard-won; do not relearn)

- **Host:** M5 Max, ~36 GB RAM, Metal working set 30.15 GB. T5-XXL loads swap ~8.5 GiB;
  zero-swap gates are only achievable right after a restart. Frozen benchmarks fail closed on
  nonzero start swap — this is correct, don't bypass it.
- Small per-step numerical diffs compound through AR loops; measure at full horizons.
- Pre-register gates before results are visible; one evidence unit per commit; amendments for
  any post-freeze change, with post-hoc introductions disclosed.
- Metric instruments need null controls: the v1 temporal tripwire failed a perceptually-null
  H.264 round-trip. Calibration bands from n=4 videos are too narrow.
- Native MLX `conv2d` is not numerically interchangeable with PyTorch fp32 conv; the im2col
  fallback in `src/liveframe/decoders/taew21_mlx.py` (`convolution_mode="fp32-im2col"`) is the
  accurate path and also ~11.8× faster than the full VAE despite materializing patches.
- Current wall breakdown after the tiny-VAE swap: denoise ≈7.1 s, refresh ≈6.6 s, VAE ≈7.5 s,
  of 21.9 s — the bottleneck has moved to the transformer; refresh-reuse is a dead end
  (NO-GO), so further gains need quantization/attention/resolution levers or training-based
  approaches (explicitly out of scope).

## 6. Validation state

As of 2026-10-09: 247 tests pass locally with Metal available, Ruff clean. Strict mypy is
clean on the qualification/engine/CLI modules; pre-existing MLX-typing errors remain in older
decoder/profiling modules. Standing rules: never push without explicit approval;
`.vscode/settings.json` stays uncommitted by design; `liveframe-0.1.0-package*` artifacts are
untracked release scratch.
