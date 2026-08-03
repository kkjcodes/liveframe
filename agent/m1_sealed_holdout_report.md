# M1 sealed holdout report

**Date:** 2026-08-01  
**Status:** authoritative FAIL; frozen evaluation completed once

## Frozen evidence

- Policy bundle: `sha256:41a228ceac16a026c7771d2e2c496714e5476c8e9287d3da3d15a78e993a9870`
- H100 references: 128/128 validated
- M5 candidates: 128/128 validated
- CUDA/MLX stochastic byte-identity gate: 128/128 passed before metric initialization
- Pairwise reports: 128/128 atomically published
- Holdout input: `sha256:27ab87ae15fd08a631b2da685fda395fdfacb57f61da2cfb947ee2f2698fc1d3`
- Verdict: `sha256:146cc868534f1a0955f7d8e49f899df0da63137c4ecb83acd68c1487993f3bcd`

The evaluator loaded the digest-bound metric artifact and asserted the actual CLIP and LPIPS
state dictionaries before computing any metric. CLIP used the frozen repository revision. The
policy and instrument were not modified after holdout visibility.

## Authoritative outcome

The frozen holdout verdict is **FAIL**. This is the final result of the sealed campaign and will
not be retuned or rerun.

The two independent layers both fail:

| Layer | Individual videos | Frozen three-of-four prompt/horizon cells |
|---|---:|---:|
| Intrinsic quality | 90 pass / 38 fail | 19 pass / 13 fail |
| Pairwise identity | 0 pass / 128 fail | 0 pass / 32 fail |

Intrinsic quality failures are more frequent at 60 seconds: 23/64 videos fail there versus
15/64 at 15 seconds. By content stratum, cyclic motion, dialogue/minimal motion, and high-motion
dense scenes each pass 22/32 videos; limited-animation/held-frame content passes 24/32.

## Identity tripwire

Identity failure is broad rather than borderline:

| Metric | Bound | Passing videos | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|---:|
| Pairwise CLIP distance mean | <= 0.05 | 29/128 | 0.0043 | 0.0884 | 0.2621 |
| Pairwise LPIPS mean | <= 0.10 | 2/128 | 0.0429 | 0.4692 | 0.6804 |
| Pairwise temporal CLIP distance mean | <= 0.35 | 0/128 | 0.4498 | 0.9361 | 0.9912 |

Every video fails at least one identity tripwire. Temporal CLIP misses its bound by at least
0.0998, and LPIPS fails 126/128 videos with the nearest failure 0.0442 above its bound. A CPU
metric reread was not performed because the decisive identity outcome is not a floating-point
hairline case.

## Intrinsic quality failures

The 38 failing videos contain the following frozen-floor violations; one video can fail multiple
metrics:

| Metric | Failures |
|---|---:|
| Motion LPIPS mean | 16 |
| CLIP prompt mean | 13 |
| CLIP prompt p10 | 12 |
| Motion CLIP distance mean | 12 |
| Temporal delta cosine mean | 7 |
| Luma flicker mean | 1 |

Some intrinsic misses are close to their individual bounds, but they do not alter the sealed
verdict: intrinsic quality independently fails 13 prompt-horizon cells, while the separately
required identity layer fails every prompt-horizon cell by large margins.

## Interpretation

The portability contract succeeded at the input level: CUDA and MLX consumed exactly the same
serialized BF16 stochastic tensors, in the same declared order, and republished byte-identical
streams. Output behavior did not remain within the frozen quality and same-trajectory identity
bands. The publishable finding is therefore not an input or provenance ambiguity; it is an
authoritative cross-runtime output divergence for the tested Rolling Forcing checkpoint and
implementation, strongest in long-horizon temporal identity.

M1 remains a completed result. Subsequent engineering may diagnose or improve MLX parity, but it
must use a new versioned campaign and cannot replace this verdict. The separate one-step CF++
speed baseline remains outside these sealed inputs and should begin only after this result is
archived.