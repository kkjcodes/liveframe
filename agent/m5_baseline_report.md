# M5 Max Rolling Forcing baseline

**Date:** 2026-07-26  
**Status:** first frozen steady-state component baseline; no acceleration applied

## Workload

- Device: Apple M5 Max, 36 GiB unified memory
- Runtime: MLX / MLX Metal 0.32.0
- Rolling checkpoint conversion: `sha256:1f15d166a8a60bed70c2a8ab2923f40bd2b6b53ed2e0be87905bf45d0666d971`
- VAE: `sha256:977530e453dbfabbab31e2972e1577d8d7e2840ba7410c81aa3fd421c0cd7414`
- Frozen fixture: `rolling-forcing-native-60latent-local21-anchor3-h100`
- Fixture manifest: `sha256:784de22c95596b2ada818de20dd96f4c3b31f80136723900769d8891ac644fed`
- Configuration: 21 latent frames, 7 three-frame blocks, 10 diagonal windows, 20 transformer forwards
- Output: 81 native pixel frames at 16 FPS, representing 5.0625 seconds
- Benchmark artifact canonical digest: `sha256:8b03977c21f98052a1e8fd9f34296f06d4719aa4902f96b0220d2fe152643bf6`

The benchmark uses the frozen fixture's precomputed prompt embedding, initial noise prefix, and
scheduler-noise tensors. Model loading and prompt encoding are excluded as steady-state setup.
One warmup preceded three measured trials. Generation, causal VAE decode, pixel materialization,
and VideoToolbox H.264 encoding are included in end-to-end wall time.

## Median result

| Component | Seconds | Share of wall |
|---|---:|---:|
| DiT denoise | 46.279 | 43.9% |
| Clean-cache refresh | 11.329 | 10.7% |
| Scheduler | 0.065 | 0.1% |
| Causal VAE | 45.681 | 43.3% |
| Pixel materialization | 0.245 | 0.2% |
| H.264 encoding | 0.726 | 0.7% |
| End-to-end wall | 105.445 | 100% |

Native generated throughput is **0.768 FPS**. A 5.0625-second clip takes **105.445 seconds**,
20.83 times playback duration and 21.09 times the sub-5-second target. Peak MLX memory is
20.766 GiB; active memory after each trial is 3.015 GiB.

Measured trial wall times were 103.292, 105.445, and 151.839 seconds. The third trial is a
material outlier, so this is a first baseline rather than a final sustained distribution.
`pmset -g therm` reported no thermal or performance warning before or after each trial, but
package power was not measured because `powermetrics` requires privileged access.

## Interpretation

Transformer work, including clean-cache refresh, totals 57.607 seconds (54.6%). Causal VAE
decode costs 45.681 seconds (43.3%). Encoding, scheduling, and pixel materialization are not
first-order bottlenecks. The first optimization work should therefore profile transformer
forward kernels and cache refresh, while VAE overlap or decoder optimization is the second
major lever.

This row is not TTFF or sustained-streaming evidence: decoding starts after latent generation,
prompt encoding is excluded, and only three measured clips were run. It is the first honest
M5 component baseline and reports no speedup.
