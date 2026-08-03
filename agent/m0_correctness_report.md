# M0 Draft: Reproducible CUDA-to-Metal Rolling Forcing

**Status:** draft technical report; correctness milestone, not a performance or intrinsic-quality claim  
**Protocol freeze:** 2026-07-25  
**Frozen quality protocol digest:** `sha256:aef6a7ad9426c97a9ae7f99c9a7a4f93799a574ca9e88d341da430c60ca75c52`
**Capture manifest digest:** `sha256:8967e70cbc3978544eea92428d2dda78044ebbcaae91e933271f7e1abd72fdc4`

## Abstract

LiveFrame reproduces a specialized Rolling Forcing video-diffusion trajectory from an NVIDIA
H100 CUDA reference on Apple Metal through MLX. The experiment binds model revisions, serialized
stochastic tensors, cache behavior, output latents, decoder provenance, and metric states by
SHA-256. The 60.56-second replay completes 84 rolling windows with exact cache cursors, active
eviction, a permanent three-frame anchor, and anchor re-RoPE at each sliding position.

The result is bounded cross-platform BF16 drift, not bit identity. The 14.81-second run passes
the frozen 0.999 final-cosine canary at 0.999393. The 60.56-second run reaches 0.998944 and
therefore fails that unchanged trajectory canary. This miss is reported as a trajectory-only
diagnostic. It is not evidence of intrinsic quality failure, and existing perceptual
measurements cannot supply an intrinsic pass because they were visible before the quality
protocol was frozen.

## Claims

M0 supports only these claims:

1. CUDA and MLX execute the same declared Rolling Forcing schedule with serialized stochastic
   inputs and matching structural cache behavior.
2. The 60.56-second MLX replay remains numerically bounded while eviction and anchor re-RoPE
   are active.
3. Same-decoder pixel-space measurements show close pairwise agreement for this historical
   seed, but remain measurement-only.
4. The experiment is correctness evidence. It is not sustained M5 throughput evidence, native
   cross-runtime VAE parity, or an authoritative intrinsic-quality verdict.

## Three independent verdict layers

| Layer | Question | M0 status |
| --- | --- | --- |
| Trajectory canary | Did CUDA and MLX remain inside the frozen numerical band? | Pass at 14.81s; fail at 60.56s |
| Intrinsic non-inferiority | Is candidate video quality within prompt- and horizon-specific teacher floors? | Pending calibration and blind holdout |
| Pairwise identity | Is the same-seed candidate still the same video trajectory? | Historical measurement only; authoritative tripwire pending blind holdout |

No layer may rescue a failure in another layer. In particular, perceptual agreement does not
convert the 60.56-second numerical canary failure into a pass.

## Reproduction contract

### Source and candidate

| Field | Value |
| --- | --- |
| Reference runtime | PyTorch CUDA BF16 |
| Reference device | NVIDIA H100 80GB HBM3 |
| CUDA | 12.8 |
| Driver | 550.163.01 |
| Candidate runtime | MLX Metal BF16 |
| Candidate device | Apple M5 Max, `Device(gpu, 0)` |
| MLX / MLX Metal | 0.32.0 / 0.32.0 |
| Candidate architecture | arm64 |
| Rolling checkpoint SHA-256 | `sha256:1f123dce60e60424a9c6d949643176840fceffb947932997bcf1202d93656d98` |
| MLX model SHA-256 | `sha256:1f15d166a8a60bed70c2a8ab2923f40bd2b6b53ed2e0be87905bf45d0666d971` |

### Rolling semantics

- Three latent frames per generated block.
- 24-frame physical cache and 21-frame visible window.
- Permanent three-frame anchor stored un-RoPE'd and re-RoPE'd at read position.
- Separate denoise and clean-context refresh cache paths.
- Explicit `context_noise=0`.
- Pixel-frame semantics: `P = 4 * (L - 1) + 1`.

### Decoder and metrics

The historical pixel comparison decodes both CUDA reference latents and MLX candidate latents
through the same MLX causal VAE. This isolates DiT trajectory parity and does not establish
H100-VAE-to-MLX-VAE parity.

| Component | Pin |
| --- | --- |
| Causal VAE SHA-256 | `sha256:977530e453dbfabbab31e2972e1577d8d7e2840ba7410c81aa3fd421c0cd7414` |
| VAE emission protocol | First emission 1 frame, steady emission 4 frames |
| LPIPS | Alex v0.1 |
| LPIPS state SHA-256 | `sha256:973be7f57af4d36c6aeecee198cb98ae315192054653634e3ba5d1a421a5023b` |
| CLIP | `openai/clip-vit-base-patch32` |
| CLIP revision | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| CLIP state SHA-256 | `sha256:57aa824e35fc53de9c7aa7ee56749bc5467fae8c7837130924d193d399578415` |

## Numerical results

| Horizon | Latents | Pixel frames | Windows | Final cosine | RMSE | Max abs | Frozen 0.999 canary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 14.8125s | 60 | 237 | 23 | 0.99939294 | 0.02045736 | 0.15234375 | Pass |
| 60.5625s | 243 | 969 | 84 | 0.99894401 | 0.02675502 | 0.19531250 | **Fail** |

The long result shows slow numerical drift across heterogeneous BF16 runtimes. Structural
invariants remain exact, but the final cosine is below the pre-existing threshold. M0 preserves
that distinction rather than weakening the threshold or describing the whole port as passed.

## Historical pairwise pixel measurements

| Horizon | LPIPS mean | CLIP cosine mean | Temporal CLIP-delta cosine mean | Temporal LPIPS-motion delta mean |
| --- | ---: | ---: | ---: | ---: |
| 14.8125s | 0.013149 | 0.999052 | 0.947309 | 0.002965 |
| 60.5625s | 0.014297 | 0.998749 | 0.933763 | 0.003902 |

At the 60-second checkpoint, the point measurements are LPIPS 0.015535, CLIP cosine 0.997498,
temporal CLIP-delta cosine 0.868728, and temporal LPIPS-motion delta 0.005719. The temporal
change is visible and motivates horizon-specific floors. These values remain historical
measurements because the seed and outputs were visible before protocol freeze.

## Artifact index

| Artifact | Purpose |
| --- | --- |
| `artifacts/h100-parity.v1.json` | H100 fixture, scheduler, VAE, and short-burst benchmark provenance |
| `artifacts/rolling-parity.v1.json` | Durable 14.81s and 60.56s parity result summary |
| `artifacts/perceptual-metrics.v1.json` | Historical LPIPS/CLIP implementation provenance |
| `artifacts/perceptual-protocol.v1.json` | Frozen horizon-indexed quality and identity protocol |
| `artifacts/perceptual-protocol.schema.json` | Machine-readable protocol schema |
| `artifacts/perceptual-run-manifest.v1.json` | Pre-registered calibration and holdout workload manifest |

## Reproduction commands

Repository-owned validation:

```text
python -m pytest
ruff check src/liveframe tests tools
mypy
```

Protocol and planned-run verification:

```text
python -m liveframe.quality_manifest \
  --protocol artifacts/perceptual-protocol.v1.json \
  --output artifacts/perceptual-run-manifest.v1.json
```

The H100 fixture capture and MLX replay commands will be inserted here after their stable CLI
entry points replace environment-specific working commands. Until then, this section is not
publication-complete.

## Limitations and pending evidence

- The 60.56-second trajectory canary fails at 0.998944; it has not been reinterpreted.
- Intrinsic teacher calibration and blind holdout have not been run.
- Existing perceptual results are ineligible for authoritative promotion.
- Cross-runtime VAE pixel parity remains unvalidated.
- The H100 19.3 generated-FPS result is a short native-window burst, not a sustained result.
- Sustained M5 FPS, TTFF, queueing, memory, power, and thermal behavior remain unmeasured.
- No Delta Scheduling speed or quality claim is made.

## M0 publication checklist

- [x] Immutable trajectory canary retained.
- [x] Structural cache semantics reported.
- [x] CUDA and Metal provenance reported.
- [x] Historical same-decoder metrics labeled measurement-only.
- [x] Frozen future quality protocol and capture manifest linked.
- [ ] Stable end-to-end fixture capture and replay commands documented.
- [ ] Curated drift plots and cache-window diagrams added.
- [ ] Artifact availability and licensing review completed.
- [ ] Independent reproduction run completed from a clean environment.
