# LiveFrame Publication Package

**Status:** fallback package active; experimental capture complete for the three negative results
**Evidence ledger:** `artifacts/liveframe-publication-claims.v1.json`

## Package thesis

LiveFrame is an evidence-first study of real-time diffusion video rather than a claim that every
pre-registered target succeeded. Its publishable contribution has three parts:

1. A reproducible measurement methodology that separates numerical trajectory, intrinsic
   quality, same-seed identity, and complete-wall performance.
2. A claim-eligible systems optimization on Apple M5 Max: a bounded MLX allocator policy reduces
   one-step CF++ complete wall time by 17.41% (`1.2108x`) with exact raw RGB output and zero swap.
3. Three quantified negative results that close specific hypotheses without overgeneralizing
   beyond the tested model, runtime, mechanism, and workload.

## Claim matrix

| Result | Quantified outcome | Supported claim | Boundary |
| --- | --- | --- | --- |
| M5 VAE buffer policy | 69.167s to 57.127s; 1.171 to 1.418 FPS | 17.41% complete-wall reduction with exact RGB and zero swap | 81-frame, 480x832 one-step CF++, M5 Max, MLX 0.32.0 |
| Cross-runtime AR divergence | identity 0/128 videos and 0/32 cells; intrinsic 19/32 cells | The frozen CUDA/MLX Rolling Forcing pair fails the pre-registered portability contract | Not an impossibility result for all CUDA/MLX implementations |
| M5 native-FPS gap | best claim-eligible row 1.418 FPS; 11.28x below 16 FPS | The tested 480x832 configuration is not real-time | Not a 60-second sustained result or a hardware impossibility claim |
| Periodic refresh reuse | 0/16 cells; temporal identity fails 64/64; LPIPS identity fails 59/64 | Two-of-three raw-K/V re-RoPE refresh substitution is not quality-safe | Does not reject adaptive, partial-layer, corrected, or denoise-side reuse |

The old 0.185 FPS 60-second M5 campaign remains useful as a long-horizon operational diagnostic,
but memory contention and swap make it ineligible as the clean headline throughput row. The clean
1.418 FPS fixture is the publication performance denominator. It is explicitly not sustained
60-second evidence.

## Measurement contribution

The methodology is itself a primary result:

- Prospective prompt, seed, stratum, horizon, threshold, and aggregation contracts.
- Serialized stochastic tensors with literal cross-runtime byte-identity checks.
- Digest-bound checkpoints, source revisions, metric weights, runtime identities, and outputs.
- Independent trajectory, intrinsic-quality, and same-seed identity verdicts.
- Fifteen-second and sixty-second drift evaluation rather than native-window extrapolation.
- Component timing for denoise, mandatory refresh, VAE, materialization, encoding, and wall time.
- Separate numerical and timing eligibility, including exact GPU driver requirements.
- Atomic publication, complete-campaign validation, and proof-blind memory preflight.
- Pre-declared STOP rules that prevent threshold movement after result visibility.

## Proposed paper structure

1. **Problem and claim discipline:** why short-burst FPS and uncalibrated perceptual scores are
   insufficient for autoregressive diffusion video.
2. **LiveFrame methodology:** stochastic identity, layered quality contracts, long-horizon drift,
   component timing, provenance, and fail-closed publication.
3. **Systems characterization:** H100 and M5 component budgets, including the exact-output M5 VAE
   allocator speedup.
4. **Negative result I:** quantified CUDA/MLX autoregressive divergence under byte-identical input.
5. **Negative result II:** measured M5 target gap and decoder-dominated bottleneck decomposition.
6. **Negative result III:** prospective long-horizon rejection of periodic refresh K/V reuse.
7. **Lessons for acceleration research:** quality-safe timing gates, refresh-versus-denoise
   accounting, and why numerical eligibility must precede timing claims.
8. **Limitations and future work:** no universal backend impossibility, no 60-second clean M5
    streaming row, no general Delta Scheduling rejection, no denoise-side oracle result, no second
    model family, and a diagnostic-only compiled VAE candidate requiring a new prospective gate.

## Closed diagnostics and unrun gates

- **Compiled VAE:** measured at 35.985s VAE-only with zero swap, but it changes pixels and its
   artifact is `claim_eligible: false`. No acceptance threshold was frozen before output
   visibility. It is future work, not a bounded publication claim.
- **Denoise-side DS oracle:** never run. The track stopped after the failed refresh-substitution
   gate by choice. A denoise-only path would have needed at least 70.3667% denoise reclaim before
   overhead to reach the frozen 17% complete-wall gate; that is a feasibility bar, not an observed
   failure.

## Artifact release tiers

**Tier A - repository release:** source, tests, schemas, manifests, compact reports, claim ledger,
environment identities, and reproduction commands.

**Tier B - curated evidence:** representative videos selected by a pre-declared rule, drift plots,
timing tables, failure distributions, and compact per-video metric records.

**Tier C - archived corpus:** the large raw H100/MLX and refresh campaign trees retained on the
network volume with checksummed indexes. Do not require downloading the full corpus to validate
the compact claims.

## Repository visibility decision

Keep the repository non-public until Tier A and Tier B have durable public locations, the curated
videos complete content review, and the published files verify against
`artifacts/liveframe-publication-release-tiers.v1.json`. Tier C may remain in checksummed cold
storage and is not a publication blocker. After those gates pass, make the repository public at
the frozen release revision and publish the evidence URLs in the same release operation so no
manifest points to an unavailable tier.

This decision does not require another H100 run. The inaccessible refresh volume still requires
one low-cost mount to copy and verify its raw corpus before that volume can be terminated; that is
a storage closeout task, not a condition for publishing the compact claims.

## Immediate work order

1. Freeze a repository snapshot and record its revision without modifying result artifacts.
2. Use the three completed standalone reports: `m1_sealed_holdout_report.md`,
   `m5_native_fps_target_report.md`, and `h100_refresh_negative_result_report.md`.
3. Generate tables and plots only from the claim ledger and immutable source artifacts.
4. Add a compact artifact inventory with retention and public-release decisions for the large
   corpora.
5. Run the full local test, Ruff, and strict mypy gates; publish any unavailable hardware tests
   as explicit reproduction requirements.
6. Draft the manuscript from the claim matrix before adding optional demos or product surface.

No new H100 allocation is required for this package. New acceleration ideas must start as new,
versioned experiments and cannot replace these frozen outcomes.
