# M5 one-step CF++ VAE profiling report

**Status:** 4 GiB VAE policy clean-confirmed and integrated into the authoritative
post-VAE baseline on 2026-08-03.

## Evidence boundary

The original authoritative clean baseline was 69.167 seconds end to end at 1.171 generated FPS,
with 54.431 seconds in incremental causal VAE decode. The initial overnight experiments began
with 1,366.25 MiB of pre-existing system swap, so those artifacts remain diagnostic and have
`claim_eligible: false`. The later restart-gated confirmation, integrated matrix, and
authoritative post-VAE baseline all began and ended with 0 MiB swap.

The full-pipeline control always retained the baseline allocator policy: a 28.080 GiB MLX
memory limit and 1 GiB free-buffer cache. DiT, context, and KV state were then released before
applying a separate VAE-only cache policy. This separation is required: applying the larger
cache limit after the full pipeline is already resident can retain old buffers and cause swap.
Configuring 4 GiB before model loading is separately validated below and does coexist safely.

## Main result: bounded allocator reuse

Interleaved eager VAE-only repeats isolate the free-buffer cache policy while preserving exact
raw RGB output:

| Repeat | Cache | VAE wall | Added swap | Raw RGB identity |
|---|---:|---:|---:|---:|
| R1 | 1 GiB | 52.402 s | 0 MiB | exact |
| R2 | 4 GiB | 40.383 s | 0 MiB | exact |
| R3 | 1 GiB | 52.027 s | 0 MiB | exact |
| R4 | 4 GiB | 40.278 s | 0 MiB | exact |
| Median | 1 GiB | 52.214 s | 0 MiB | exact |
| Median | 4 GiB | 40.330 s | 0 MiB | exact |

The median VAE-only speedup is **1.295x**, or a **22.76% VAE wall reduction**. Both 4 GiB
repeats are faster than both 1 GiB repeats. The 4 GiB composed eager run reproduced at
40.317 seconds with 18.294 GiB peak MLX memory and the pinned raw RGB digest
`sha256:4b55f063d0164fd1c44358186af6cd04ae2dba1356dedd5b86403dfd101e8f61`.

Holding every non-VAE baseline component fixed, the repeated median projects to **55.067
seconds end to end, 1.471 generated FPS, and a 20.39% wall reduction**. This is an Amdahl-style
projection, retained here as the pre-integration estimate. The authoritative integrated result
below supersedes it for performance claims.

## Integrated coexistence

A clean, single-trial 1/2/4 GiB matrix tested the interleaved streaming pipeline with DiT,
context, KV state, and causal VAE resident together:

| Cache | VAE wall | End-to-end wall | Generated FPS | Added swap | Raw RGB |
|---:|---:|---:|---:|---:|---:|
| 1 GiB | 52.074 s | 66.730 s | 1.214 | 0 MiB | exact |
| 2 GiB | 51.142 s | 66.013 s | 1.227 | 0 MiB | exact |
| 4 GiB | 42.074 s | 56.887 s | 1.424 | 0 MiB | exact |

All rows passed the pinned latent-quality anchor and produced the same 97,044,480-byte raw RGB
digest. The 2 GiB point recovers only 1.8% of VAE wall and does not cross the allocator-reuse
threshold. The 4 GiB point coexists successfully when configured before model loading, retaining
3.586 GiB in the MLX free-buffer cache without increasing the 26.661 GiB peak or creating swap.

The claim-eligible confirmation used one warmup and three measured 4 GiB trials. Median
end-to-end wall is **57.127 seconds**, native generated throughput is **1.418 FPS**, and VAE wall
is **41.319 seconds**. Relative to the authoritative 1 GiB baseline, this is a **1.211x
end-to-end speedup**, **17.41% wall reduction**, **1.317x VAE speedup**, and **24.09% VAE wall
reduction**. All three trials passed raw RGB identity, latent quality, zero-swap, and thermal
gates. This measured row replaces the Amdahl projection for integrated streaming claims.

## Execution-mode A/Bs

| VAE-only mode | Cache | Wall | Relative to paired eager | Raw RGB identity |
|---|---:|---:|---:|---:|
| eager | 1 GiB | 52.653 s | control | exact |
| async/deferred | 1 GiB | 52.460 s | 1.004x | exact |
| eager | 4 GiB | 40.317 s | control | exact |
| async/deferred | 4 GiB | 40.336 s | 1.000x | exact |
| eager | 1 GiB | 52.103 s | control | exact |
| compiled steady | 1 GiB | 48.221 s | 1.081x | changed |
| eager | 4 GiB | 40.349 s | control | exact |
| compiled steady | 4 GiB | 35.985 s | 1.121x | changed |

Deferred materialization does not provide a useful gain and does not compose with the larger
cache. This rejects synchronization/materialization as the cause of the 4 GiB improvement.
The repeatable exact-output effect is allocator reuse.

Compilation does compose with the cache, reaching 35.985 seconds VAE-only, but changes the raw
RGB output. Against eager RGB over 97,044,480 values, compiled output has cosine 0.999997700,
RMSE 0.277515, mean absolute error 0.076857, and max absolute error 5. It is an output-changing
quality candidate, not an exact optimization, and cannot be banked without the frozen quality
gates.

## Stage profile

With the 4 GiB cache, the top staged evaluation shares are:

| Decoder stage | Eval share |
|---|---:|
| `upsamples.12.ResidualBlock` | 19.86% |
| `upsamples.13.ResidualBlock` | 16.37% |
| `upsamples.14.ResidualBlock` | 14.95% |
| `upsamples.08.ResidualBlock` | 9.20% |
| `upsamples.09.ResidualBlock` | 7.22% |
| `upsamples.11.Resample` | 6.31% |
| `upsamples.10.ResidualBlock` | 6.07% |
| `upsamples.07.Resample` | 3.56% |

The final three high-resolution residual blocks consume **51.19%** of staged evaluation. The
4 GiB policy changes memory reuse and total wall time but does not change the optimization
target: dense high-resolution residual convolutions remain dominant. Stage instrumentation
adds substantial synchronization overhead, so its 47.816-second wall time must not be compared
directly with eager as an implementation candidate.

## Decision

1. Bank the 4 GiB free-buffer cache as the authoritative exact-output post-VAE baseline when
   configured before model loading.
2. Close async/deferred materialization as a useful acceleration path for this decoder.
3. Retain compiled steady state as an output-changing candidate pending frozen quality tests.
4. Target the final high-resolution residual convolutions for kernel or quantization work.
5. Do not test an 8 GiB cache without a separate zero-swap preflight and fail-closed memory gate;
   4 GiB already retains 3.586 GiB while the integrated pipeline peaks at 26.669 GiB.

## Clean confirmation

The restart-gated clean confirmation completed with 0 MiB startup swap, 92% free memory, and no
swap created by either the full-pipeline control or isolated VAE phase. The isolated 4 GiB eager
decode completed in **39.484 seconds**, reproduced the pinned raw RGB digest exactly, and peaked
at 18.294 GiB MLX memory. The retained MLX free-buffer cache was 3.892 GiB after decode.

This confirms the 4 GiB policy as a byte-exact isolated-decoder acceleration. The subsequent
integrated matrix and authoritative run also confirm coexistence with resident DiT, context,
and KV state.

The completed command was:

```bash
PYTHONUNBUFFERED=1 caffeinate -dimsu .venv/bin/python \
  tools/profile_mlx_cfpp_vae.py \
  --model-dir weights/CFPP-framewise-1step-MLX \
  --fixture outputs/golden/cfpp-1step-native-21block-h100 \
  --baseline-report artifacts/m5-cfpp-1step-baseline.v1.json \
  --output artifacts/m5-cfpp-vae-profile-cache4-clean.v1.json \
  --modes eager \
  --cache-limit-gib 4 \
  --max-start-swap-mib 0
```

## Artifacts

- Authoritative post-VAE baseline: `artifacts/m5-cfpp-1step-cache4-baseline.v1.json`
- Authoritative report SHA-256: `sha256:5bdc9fdbae59842158f278163d738656d1e4c05996ccf7495cf4d074009e3ce1`
- Authoritative video: `outputs/m5-cfpp-1step-cache4-baseline.mp4`
- Authoritative video SHA-256: `sha256:245b41d6077948e23006af753178ae943e84e0702fa423f291b66a77bfbae151`
- Integrated matrix: `artifacts/m5-cfpp-integrated-cache1-clean-diagnostic.v1.json`,
  `artifacts/m5-cfpp-integrated-cache2-clean-diagnostic.v1.json`, and
  `artifacts/m5-cfpp-integrated-cache4-clean-diagnostic.v1.json`
- Clean exact-output profile: `artifacts/m5-cfpp-vae-profile-cache4-clean.v1.json`
- Clean profile SHA-256: `sha256:a3ea464ae2f6a83826d876d61e5158e42a52dc94a6ed378a376c0c5b16b35e91`
- Composed exact-output profile: `artifacts/m5-cfpp-vae-profile-cache4-composed-diagnostic.v1.json`
- Composed profile SHA-256: `sha256:c93167641f45ad8f87f0b15328c3f99e776e73ec8cf20aa62294c0d3ee68a672`
- Compiled 4 GiB profile: `artifacts/m5-cfpp-vae-profile-cache4-compiled-diagnostic.v1.json`
- Compiled profile SHA-256: `sha256:0fee7903de6d0b8e707e1684b466971629f30ff0372532d4cad209685203b801`
- Repeat matrix: `artifacts/m5-cfpp-vae-cache-repeat-r1-1gib.v1.json` through
  `artifacts/m5-cfpp-vae-cache-repeat-r4-4gib.v1.json`