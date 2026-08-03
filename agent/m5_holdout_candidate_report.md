# M5 Max holdout candidate capture report

**Date:** 2026-08-01  
**Status:** 128/128 complete and validated; sealed evaluation completed with authoritative FAIL

## Frozen workload

- Candidate batch: `sha256:5019c73470eb11f63ff8a525fa8081a7f02b4ea764283f071df0c52d51bb57e7`
- Shared-noise index: `sha256:3fbfbf2896e566f663a1b781e20872c393b25e0d6333d33268b3c069e4b72a17`
- Device: Apple M5 Max, `applegpu_g17s`, 36 GiB unified memory
- Runtime: MLX 0.32.0 and MLX Metal 0.32.0
- Videos: 64 at 15 seconds and 64 at 60 seconds
- Artifact tree: approximately 104 GB

## Validation

All 128 result records pass the frozen manifest, artifact checksum, frame-count, metric-set,
finite-value, and runtime-identity checks. A post-capture audit rehashed every declared artifact,
validated every republished stochastic payload and per-tensor checksum, and required each
payload digest and tensor table to equal the corresponding frozen shared plan. The audit passed
128/128 in 49.7 seconds. There are no temporary files or incomplete result directories.

The initial process completed 14 jobs before it was stopped under severe unified-memory pressure;
a concurrent Docker workload was consuming approximately 15 GB. The restart strictly validated
and skipped those 14 results, then completed the remaining 114 jobs in 339,290.6 seconds. No
completed result was regenerated or replaced.

## Resident-model batch throughput

Values are per-video p10 / median / p90 across 64 independent jobs at each horizon. Total DiT is
diffusion plus clean-cache refresh plus scheduler time. Generation is Total DiT plus VAE time.
Native FPS excludes prompt loading/encoding, metrics, artifact hashing/publication, and executor
overhead.

| Horizon | Total DiT (s) | VAE (s) | Generation (s) | Native FPS | Real-time factor |
|---|---:|---:|---:|---:|---:|
| 15s | 456.7 / 719.6 / 833.6 | 102.6 / 106.0 / 110.5 | 559.5 / 825.5 / 941.9 | 0.252 / 0.287 / 0.425 | 0.0157 / 0.0179 / 0.0266 |
| 60s | 3592.1 / 4835.6 / 5480.9 | 399.4 / 404.0 / 425.3 | 3994.2 / 5241.3 / 5898.2 | 0.164 / 0.185 / 0.243 | 0.0103 / 0.0116 / 0.0152 |

| Horizon | Text encode/load (s) | Metric (s) | Executor wall (s) |
|---|---:|---:|---:|
| 15s | 8.0 / 11.8 / 14.5 | 7.0 / 7.5 / 7.9 | 576.1 / 846.8 / 964.3 |
| 60s | 8.9 / 11.5 / 14.1 | 12.7 / 13.4 / 14.0 | 4018.1 / 5270.5 / 5929.1 |

TTFF was not measured. T5 is loaded and encoded per job, so text timing is not a one-time prompt
encoding measurement. This campaign is not sustained-streaming evidence: latent generation and
VAE decoding are sequential, and each row is a complete batch job.

## Operational drift diagnostic

For 60-second jobs in publication order, median native FPS was 0.201 for the first 16 and 0.165
for the final 16, a 17.8% decrease. The campaign reported no macOS thermal or performance
warnings, but this comparison is not a controlled thermal conclusion. The run included the
Docker memory-contention incident, a process restart, and sustained heavy swap use. A future
sustained benchmark must begin from a clean memory state, record package power where available,
and use interleaved or explicitly controlled thermal conditions.

## Sealed evaluation outcome

The evaluator validated and literally compared all 128 stochastic payload pairs before loading
the pinned CLIP/LPIPS instruments. It then published an authoritative FAIL: intrinsic quality
failed 38/128 videos and pairwise identity failed 128/128. The candidate tree, policy bundle,
pairwise reports, holdout input, and verdict remain immutable. See
`agent/m1_sealed_holdout_report.md`.