# M5 Native-FPS Target Report

**Status:** quantified target miss with one banked exact-output optimization
**Primary artifacts:** `artifacts/m5-cfpp-1step-baseline.v1.json` and
`artifacts/m5-cfpp-1step-cache4-baseline.v1.json`

## Question

Can the frozen 480x832 one-step CF++ configuration approach the declared 16 native generated FPS
target on Apple M5 Max without changing model output?

## Clean baseline

The claim-eligible 1 GiB allocator baseline used one warmup and three measured trials on Apple
M5 Max with MLX and MLX Metal 0.32.0. It began and ended with zero swap, passed the pinned latent
anchor, and measured:

| Quantity | Median |
| --- | ---: |
| Complete wall | 69.167s |
| Generated throughput | 1.171 FPS |
| TTFF | 1.489s |
| Denoising | 7.218s |
| Mandatory refresh | 6.709s |
| Causal VAE | 54.431s |
| Peak MLX memory | 26.669 GiB |

The decoder consumed 78.7% of complete wall, making it the measured first optimization target.

## Banked optimization

Increasing the bounded MLX free-buffer cache from 1 GiB to 4 GiB before model loading reduced
median VAE wall to 41.319s and complete wall to 57.127s. Throughput increased to 1.418 FPS. The
result is a `1.2108x` complete-wall speedup, a 17.41% wall reduction, and a 24.09% VAE-wall
reduction. All trials retained the exact 97,044,480-byte raw RGB digest, passed the latent anchor,
and added zero swap.

This is the claim-eligible systems speedup. The earlier Amdahl projection and output-changing
compiled decoder are not substituted for it.

## Target decision

The optimized configuration remains `16 / 1.4179056 = 11.2842x` below the 16 native-FPS target.
It therefore does not establish real-time video generation on Apple Silicon. The result is a
measured target gap for this configuration, not a hardware impossibility result.

The prior 60-second Rolling Forcing campaign measured a 0.185 FPS median, but concurrent memory
pressure and sustained swap contaminate that timing. It remains an operational and long-horizon
diagnostic, not the headline clean denominator. Conversely, the clean 81-frame row is not a
60-second sustained-streaming claim. Both limitations must remain visible.

## What remains open

Quantization, lower-resolution product tiers, different model families, custom Metal kernels,
decoder redesign, overlap, and disclosed interpolation were not exhausted. Any future result
must use a new versioned experiment and may build on the 4 GiB exact-output baseline without
replacing this target-gap finding.

## Evidence identity

- 1 GiB baseline SHA-256:
  `sha256:99b18d143dfff1323cdf850b367efc84b8051df676abd5fe54348b6695f0ec3d`
- 4 GiB baseline SHA-256:
  `sha256:5bdc9fdbae59842158f278163d738656d1e4c05996ccf7495cf4d074009e3ce1`
- Exact raw RGB SHA-256:
  `sha256:4b55f063d0164fd1c44358186af6cd04ae2dba1356dedd5b86403dfd101e8f61`
