# H100 Periodic Refresh-Reuse Negative Result

**Status:** frozen numerical STOP; authoritative timing prohibited
**Decision artifact:** `artifacts/h100-refresh-numerical-decision.v1.json`

## Hypothesis

Framewise CF++ performs one mandatory clean-cache refresh per generated latent frame. The tested
mechanism retained a full-refresh anchor every third frame, captured raw normalized keys and
values at all 30 transformer layers, and substituted them into the following two frames after
reapplying temporal RoPE to the keys. The candidate skipped destination K projection,
K normalization, and V projection only during those substituted refreshes.

The prospective contract required independent reference and candidate mutable state, identical
serialized stochastic inputs, exact substitution ancestry, 21-frame rolling context, and
horizon-indexed latent, per-layer K/V, decoded-pixel, intrinsic-quality, and identity evidence.

## Campaign

- Hardware: NVIDIA H100 80GB HBM3.
- Workload: two high-redundancy strata, four prompts per stratum, four seeds, and 15-second and
  60-second horizons.
- Result count: 64/64 complete and numerically eligible.
- Aggregation: at least three of four videos must pass in every prompt/horizon cell; all 16 cells
  must pass.
- Timing separation: the instrumented campaign was numerical-only. Authoritative timing required
  a later uninstrumented paired replay after numerical GO.

## Outcome

The mechanism failed all 16 prompt/horizon cells at 0/4 passing videos. Failure was broad:

| Frozen rule | Failing videos |
| --- | ---: |
| Pairwise temporal CLIP distance | 64/64 |
| Pairwise LPIPS | 59/64 |
| Intrinsic motion LPIPS | 56/64 |
| Intrinsic motion CLIP distance | 34/64 |
| Pairwise CLIP distance | 32/64 |
| Intrinsic temporal-delta cosine | 15/64 |

Representative failures combine suppressed candidate motion with large same-seed temporal
trajectory changes. This is not a borderline aggregate miss and cannot be rescued by timing.

## Decision

The frozen periodic raw-K/V re-RoPE substitution is not quality-safe for the tested framewise
CF++ refresh path. The pre-registered decision is STOP. No authoritative timing replay was run,
because numerical GO was a prerequisite.

Even before overhead, eliminating two of every three refresh forwards can save at most 13.04% of
the frozen complete wall, below the separate 17% wall gate. A successful broader acceleration
claim would therefore also require denoise-side savings or a different mechanism.

This result does not reject all Delta Scheduling designs. Adaptive refresh, partial-layer
refresh, error-corrected reuse, spatially selective denoise reuse, and new model families remain
untested hypotheses. They must begin as new experiments and cannot tune against these visible
proof results.

## Broader DS disposition

The denoise-side oracle was not run. The project stopped after this refresh gate by choice and
archived the research track before collecting oracle instrumentation. The frozen H100 denominator
put denoising at 24.1592% of complete wall; reaching the separate 17% wall gate through denoising
alone required at least 70.3667% denoise reclaim before overhead. That is an untested feasibility
bar, not a failed-oracle measurement. This report rejects the periodic refresh mechanism only.

## Evidence identity

- Campaign validation SHA-256:
  `sha256:d9e8c75e3233a0af4acc5bafee926ae01bdc5a94f31fceed5bd4b9454c572737`
- Numerical decision file SHA-256:
  `sha256:c0fe75da36e5a5e968c288021b8ca8fde8b9a1d0e71bb04ba792cc793f9bd4b4`
- Numerical decision canonical digest:
  `sha256:51b1146027c6213336d9e9eaf2c9bbd404e3b517ab78aad80370510f4e8624bb`
