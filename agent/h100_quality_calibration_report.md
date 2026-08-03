# H100 quality calibration report

**Date:** 2026-07-26  
**Status:** capture and ingestion complete; frozen sanity gate failed; policy derivation and holdout blocked

## Frozen identities

- Protocol: `sha256:aef6a7ad9426c97a9ae7f99c9a7a4f93799a574ca9e88d341da430c60ca75c52`
- Manifest: `sha256:8967e70cbc3978544eea92428d2dda78044ebbcaae91e933271f7e1abd72fdc4`
- Rolling checkpoint: `sha256:1f123dce60e60424a9c6d949643176840fceffb947932997bcf1202d93656d98`
- Full calibration batch: `sha256:272ffba93a86fbdf032aa0322f20311a107b226b64a9ce3d48a00c378f7d37eb`
- First-seed checkpoint report: `sha256:8e3ab670d1247591786e287f8c395e09d7340db0b069d550f8b4c816e2b3f790`
- Calibration input: `sha256:437081b72edcb531b4a18733a524a4ffba2a0438ce3ad9cd358d6ac2901c984a`
- Sanity report: `sha256:320953225c1e4775ec10285d27ec38f48a504f0cba4fd89161ea828f11e3f6e3`

## Completion

Strict restart validation reports zero pending jobs. All 256 teacher videos were ingested from
16 prompts, 8 calibration seeds, and both exact horizons. Every published result passed artifact
checksum, identity, frame-count, metric-set, and finite-value validation.

Measured executor time was 4.2316 H100 hours. Declared raw artifacts total 208.3185 GiB; the
result tree occupies 209 GiB on disk. Raw pixels, BF16 latents, and stochastic tensors remain on
the expanded RunPod volume and must be retained through review.

The first-seed checkpoint passed 32/32 jobs and projected 4.2043 H100 hours and 208.3185 GiB,
below its fail-closed limits of 5.2875 hours and 263.75 GiB.

## Sanity outcome

All 192 prompt/horizon/metric cells satisfy physical bounds. The frozen report emits 39
`epsilon_dominated` warnings and one failure:

`high-motion teacher spread is degenerate: high-motion-dense-scenes/dense_motion_03/60s/motion_clip_distance_mean`

The eight seed values are:

| Seed | Value |
|---:|---:|
| 11003 | 0.0162424028 |
| 17389 | 0.0162385012 |
| 25111 | 0.0143191280 |
| 33941 | 0.0166715663 |
| 42073 | 0.0157007196 |
| 50821 | 0.0144179292 |
| 60101 | 0.0165029681 |
| 70937 | 0.0142746344 |

The median is 0.0159696104. The robust cross-seed scale is 0.0009157380, below the frozen
0.001 scale epsilon by 0.0000842620 (8.43%). Because this is a high-motion cell and metric, the
protocol classifies the spread as degenerate and fails closed.

## Decision

Cross-seed non-degeneracy did not pass the frozen calibration rule. Do not derive or freeze a
quality policy bundle from this calibration, launch holdout jobs, or inspect holdout output.
Do not weaken or reinterpret the frozen threshold after observing this result. Any follow-up
must be a new, explicitly versioned protocol and campaign while preserving this failed outcome.
