# LiveFrame Product Mission

**Adopted:** 2026-10-06  
**Status:** Active governing direction

## End Goal

Make streaming video generation measurable, reproducible, and progressively deployable on
real hardware.

LiveFrame exists to help engineers determine whether a streaming video system works for a real
workload, understand why it fails, and verify that an optimization improves complete-wall
performance without silently weakening output semantics, quality, stability, or hardware
eligibility. Publication is demoted, not banned: it is no longer a primary objective or a
success criterion, but low-cost dissemination — upstream issue reports, DOI-timestamped
preprint archives, and engineering blog posts — is encouraged when it timestamps findings or
drives adoption, and must not consume meaningful engineering time. Venue submissions and
reviewer cycles are out of scope.

## Real-World Problem

Streaming video systems are difficult to compare because implementations use incompatible
definitions of frames, duration, latency, timing boundaries, quality, and hardware state. A
reported FPS value may count interpolation, omit decoding, cover only a short burst, or hide
memory and thermal failure. Developers consequently spend time and compute integrating systems
that cannot satisfy their actual application requirements.

LiveFrame will replace ambiguous performance claims with executable deployment qualifications.
For a selected model, runtime, hardware target, and workload profile, it will answer:

- Can the system sustain the required native generated FPS and duration?
- How long until the first displayable frame, and how variable are later chunks?
- Does quality or temporal stability degrade over the requested horizon?
- Does the run remain within memory, swap, and thermal eligibility limits?
- Which measured component is the primary bottleneck?
- Can another engineer reproduce the result from digest-bound inputs and provenance?

## Users

1. Application developers deciding whether a model can support an interactive camera, game,
   creative tool, live avatar, or virtual-production workflow.
2. Model researchers testing whether a change improves complete-wall behavior without violating
   quality or long-horizon stability.
3. Runtime and hardware engineers locating bottlenecks in denoising, attention, cache behavior,
   decoding, synchronization, memory, or encoding.
4. Operators deciding which hardware and model configuration justifies its deployment cost.

## Product Loop

LiveFrame has two connected tracks.

### Qualification

- Frozen workload profiles with explicit native-frame, duration, latency, quality, memory, and
  thermal requirements.
- Runtime adapters that preserve upstream semantics and declare capabilities and provenance.
- Synchronized complete-wall measurement with native and presentation frames kept separate.
- Machine-readable, digest-bound evidence and fail-closed gate evaluation.
- Human-readable reports that explain failed gates and measured bottlenecks.

### Optimization

- Profile systems that fail a useful workload profile.
- Select one measured bottleneck and one falsifiable optimization hypothesis.
- Compare baseline and candidate under identical requests, stochastic inputs, timing boundaries,
  and quality policies.
- Keep improvements only when complete-wall performance improves and all required gates remain
  eligible.
- Upstream reusable fixes and preserve scoped negative results so others do not repeat failed
  experiments.

The qualification harness establishes truth. Profiling turns it into understanding. Optimization
turns that understanding into deployable progress.

## Initial Workload Profiles

- `interactive-480p`: at least 60 seconds, 16 native generated FPS, TTFF at most 2 seconds,
  p95 chunk latency at most 250 ms, plus quality, long-horizon, memory, and thermal gates.
- `creative-preview`: at least 15 seconds, 4 native generated FPS, TTFF at most 4 seconds,
  p95 chunk latency at most 1 second, plus the same evidence classes.
- `offline-streaming`: at least 60 seconds with no real-time speed claim, while still requiring
  quality, long-horizon, memory, and thermal eligibility.

These are version-zero product profiles. Before real model results are called authoritative, each
profile must be frozen in a digest-bound protocol with exact prompt, stochastic-input, timing,
quality-policy, trial-order, and failure semantics.

## First 90-Day Outcome

LiveFrame can qualify two streaming video systems on Apple Silicon and NVIDIA hardware, identify
their complete-wall bottlenecks, reproduce every reported result, and demonstrate one verified
performance improvement without violating frozen quality limits.

Required deliverables:

1. One stable `liveframe qualify` workflow.
2. Synthetic contract and campaign tests that run without model dependencies.
3. One MLX CF++ adapter built from the existing M5 benchmark path.
4. One CUDA adapter built from an existing Causal or Rolling Forcing path.
5. Two complete 60-second qualification reports with machine-readable evidence.
6. One component-level bottleneck report.
7. One validated optimization or a precise negative result.
8. Reproduction instructions that another engineer can execute.

## Delivery Order

1. Workload profiles and fail-closed qualification results.
2. CPU-only synthetic adapter and end-to-end contract tests.
3. Streaming artifact sink, timing boundaries, synchronization, and benchmark ledger schema.
4. Existing M5 CF++ path packaged as the first real adapter.
5. Reduced qualification campaign and bottleneck analysis on M5.
6. First CUDA adapter and comparable hardware-specific qualification.
7. One measured optimization experiment.
8. Static comparison views only after two trustworthy real-system ledgers exist.

Broad adapter coverage, badges, third-party automation, and remote-service benchmarking are later
work. They must not delay a useful two-system qualification and optimization loop.

## GPU Policy

CPU-only development covers schemas, profiles, synthetic adapters, campaign orchestration,
verification, report generation, and CI. Apple MLX validation requires a native Apple Silicon GPU
session. CUDA adapter validation, 60-second campaigns, metric execution, and CUDA optimization
require rented NVIDIA GPU time.

GPU jobs begin only after their protocol and stop conditions pass CPU validation. Every allocation
must have a stated question, time cap, expected artifact, and fail-closed eligibility checks. Start
with smoke and qualification runs; authorize full campaigns only after the adapter produces valid
evidence.

## Success Measures

- Independently reproducible system qualifications.
- Deployment decisions made from complete evidence rather than nominal FPS.
- Reduced engineering and GPU time spent on unsuitable configurations.
- Faster identification of complete-wall bottlenecks.
- Verified speedups at unchanged workload and quality requirements.
- Reusable improvements accepted by upstream projects.
- Real applications enabled or ruled out with precise evidence.

Paper acceptance, citation count, leaderboard size, and adapter count are not success measures.