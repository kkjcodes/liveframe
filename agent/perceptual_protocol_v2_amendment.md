# Perceptual protocol v2 amendment

**Date:** 2026-07-26  
**Status:** frozen before holdout visibility

## Ancestry

Protocol v1 and its failed sanity report remain immutable. Protocol v2 is a
calibration-informed policy-derivation amendment; it does not claim that the existing 256
teacher videos were captured under a new protocol.

- Source protocol: `sha256:aef6a7ad9426c97a9ae7f99c9a7a4f93799a574ca9e88d341da430c60ca75c52`
- Source calibration: `sha256:437081b72edcb531b4a18733a524a4ffba2a0438ce3ad9cd358d6ac2901c984a`
- Source failed sanity report: `sha256:320953225c1e4775ec10285d27ec38f48a504f0cba4fd89161ea828f11e3f6e3`
- Protocol v2: `sha256:f1b39f36a0ed6f361aaabf38062868d5f3e12a953627c67b1a736c8d56148b93`
- Sanity v2: `sha256:e0308cdb235264112c049d46b27e6370b3d0e0c7abc5f92ccbd79886d62c5841`
- Policy bundle v2: `sha256:41a228ceac16a026c7771d2e2c496714e5476c8e9287d3da3d15a78e993a9870`

## Amendment

A metric cell whose raw cross-seed MAD scale is less than or equal to its unchanged
`scale_epsilon` is diagnostic and cannot define an authoritative candidate bound. All centers,
raw scales, epsilon values, and diagnostic rules remain in the bundle. Identity tripwires remain
authoritative and cannot rescue an intrinsic failure.

Every prompt and horizon must retain at least one authoritative metric in each declared category:

- Semantic fidelity: `clip_prompt_mean` or `clip_prompt_p10`.
- Frame dynamics: `motion_clip_distance_mean` or `motion_lpips_mean`.
- Temporal behavior: `temporal_delta_cosine_mean`, `motion_lpips_mean`, or
  `luma_flicker_mean`.

`motion_lpips_mean` is eligible for both frame-dynamics and temporal-behavior coverage because
it measures perceptual change between consecutive decoded frames. This dual role is declared
before holdout visibility rather than inferred from holdout results.

## Result

The v2 sanity report passes all 32 prompt/horizon policies. It demotes exactly 39 of 192 metric
cells, including the v1 failure, and retains complete declared coverage in every policy. All
three identity tripwires remain authoritative.

No holdout result files were present when the v2 protocol, sanity report, and policy bundle were
frozen. No threshold, seed, prompt, horizon, metric implementation, or identity limit changed.

## Holdout readiness

The holdout remains sealed. The prompt-conditioned MLX runner now executes only frozen
candidate jobs, verifies the Rolling Forcing checkpoint ancestry and v2 bundle, serializes the
Torch CPU stochastic stream, and publishes CUDA-compatible BF16 latent and uint8 pixel schemas.
Its exact 128-job batch is pinned before result visibility.

The remaining execution gap is evaluator orchestration. It must load all captures and publish
the final input and verdict
   atomically. Its checksum-bound pairing, three-metric resident measurement path, and
   bundle-bound input assembly are implemented and fail on missing or mismatched pairs.

The H100 runner now rejects non-CUDA jobs. This prevents accidental execution of an MLX manifest
job on CUDA while the candidate runner is incomplete.
