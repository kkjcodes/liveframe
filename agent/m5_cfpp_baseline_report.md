# M5 one-step CF++ clean baseline

**Status:** authoritative baseline complete on 2026-08-02.

## Configuration

- Apple M5 Max, 36 GiB unified memory, MLX/MLX Metal 0.32.0.
- Frozen one-step CF++ BF16 model and 21-latent/81-pixel-frame fixture.
- One warmup followed by three measured trials.
- Model loading and prompt encoding excluded; generation, incremental causal VAE decode, pixel
  materialization, and VideoToolbox H.264 encoding included.
- MLX limited to Metal's 28.080 GiB recommended working set with a 1 GiB free-buffer cache.
- Startup swap was 0 MiB. Warmup and all measured trials created 0 MiB swap.

## Median result

| Component | Seconds | Share of wall |
|---|---:|---:|
| CF++ denoising | 7.218 | 10.4% |
| Clean-cache refresh | 6.709 | 9.7% |
| Incremental causal VAE | 54.431 | 78.7% |
| Pixel materialization | 0.237 | 0.3% |
| H.264 encoding | 0.301 | 0.4% |
| End-to-end wall | 69.167 | 100% |

Native generated throughput is **1.171 FPS**. The 5.0625-second clip takes **69.167 seconds**,
13.66 times playback duration. Median TTFF is **1.489 seconds**. Measured wall times were 69.650,
69.167, and 67.938 seconds; measured throughput was 1.163, 1.171, and 1.192 FPS.

Peak MLX memory was **26.669 GiB**; active memory after a trial was 15.147 GiB and the free MLX
cache was 0.521 GiB. System free-memory readings remained 60-79% around measured trials, post-run
swap remained 0 MiB, and `pmset -g therm` reported no thermal or performance warning.

All three trials passed the pinned H100 latent quality anchor: cosine 0.999490, RMSE 0.031555,
and max absolute error 0.220703. The retained H.264 artifact is 81 frames at 832x480 and 16 FPS.

## Interpretation

The clean one-step CF++ baseline is **1.171 FPS**, not the earlier contaminated 0.185 FPS
observation. VAE decode is the first-order bottleneck at 78.7% of wall time. Quantizing or
accelerating the transformer alone can affect only about 20.1% of this measured path unless VAE
decode is overlapped or optimized. Quantization remains independently useful, but its expected
end-to-end ceiling must be reported against this component budget.

This row is eligible for clean baseline claims. It is not sustained 60-second evidence and does
not include prompt encoding or model load time.

## Artifacts

- Report: `artifacts/m5-cfpp-1step-baseline.v1.json`
- Report SHA-256: `sha256:99b18d143dfff1323cdf850b367efc84b8051df676abd5fe54348b6695f0ec3d`
- Video: `outputs/m5-cfpp-1step-baseline.mp4`
- Video SHA-256: `sha256:6b3ba4b09fcf25807bc3d54d4561e3a6c5345a7a4ab76a2e09aef208b1091866`
- Log: `outputs/m5-cfpp-1step-baseline.log`