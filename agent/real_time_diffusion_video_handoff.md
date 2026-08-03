# Real-Time Diffusion Video Project Handoff

> **2026-07-27 continuation note:** this handoff predates the active LiveFrame implementation.
> Do not create another repository or treat this workspace as a completed prior benchmark.
> LiveFrame now lives in this repository; `agent/liveframe_execution_plan.md` is the scheduling
> authority. H100 calibration is complete, frozen protocol v2 and its policy bundle exist, and
> all 128 sealed H100 holdout references have been captured and validated against shared
> stochastic plans. The immediate next phase is the exact 128-job MLX candidate capture on
> Apple M5 Max, followed by the fail-closed byte-identity evaluator. This document remains only
> as an origin-story lookback for the project boundary and measurement principles below.

This handoff is for a new agent starting a separate project for real-time or near-real-time diffusion video generation. The current repo should be treated as a completed benchmark milestone, not as the main implementation home for the video work.

## Recommended Project Boundary

Create a new codebase for the video project.

Reason: this repository has a clean identity: benchmarking training-free block/residual caching for 4-bit FLUX.1-dev on Apple Silicon. Real-time diffusion video has a different shape: streaming generation, frame scheduling, temporal consistency, preview UX, media encoding, and different success metrics.

Use this repo as prior art and a measurement reference, not as the production base.

## What To Carry Forward

Useful habits and components from this benchmark:

- Interleaved timing for baseline-vs-candidate comparisons.
- Explicit warmup and thermal-state awareness on Apple Silicon.
- Quality reports that include visual samples, not only scalar metrics.
- Prompt or scene stratification instead of one headline example.
- Cross-seed or cross-run floors before interpreting reference metrics.
- Clear separation between trajectory preservation and perceived output quality.
- Negative-result logging when an idea fails narrowly.

Useful implementation ideas to inspect before copying anything:

- `src/m5_flux_benchmark/pipelines/mflux_clean.py`: custom mflux generation loop and timing discipline.
- `src/m5_flux_benchmark/cache/base.py`: cache strategy abstractions, cache stats, and metadata pattern.
- `src/m5_flux_benchmark/reports.py`: report rendering and bootstrap timing CI style.
- `src/m5_flux_benchmark/quality_rescore.py`: perceptual rescore pattern with LPIPS, CLIP, and SSIM.
- `README.md`: public framing for claims, limitations, and artifacts.

Copy small ideas, not the whole architecture.

## What Not To Inherit Blindly

- Do not assume block-residual caching is the right acceleration surface for video.
- Do not optimize for same-seed PSNR as the primary quality metric.
- Do not let one easy prompt or scene become the headline.
- Do not benchmark sequential baseline/candidate runs without interleaving.
- Do not claim transfer beyond the exact hardware, model, quantization, and runtime tested.
- Do not make this repo absorb large video artifacts or streaming application code.

## Prior Result To Remember

The strongest closed result from this repo:

- Model/runtime: 4-bit FLUX.1-dev through `mflux` on Apple M5 Max, 36GB unified memory.
- Best cache policy: `joint_12_18_start8_single_28_37_start6_i2_residual`.
- Timing: about `1.125x` speedup with interleaved validation.
- Median same-seed PSNR: `29.06`.
- Median windowed SSIM: `0.94575`.
- Median LPIPS Alex: `0.05519`.
- Median CLIP delta: `0.00181`.
- Limitation: dense interiors remained weak at about `22.62` PSNR.
- Residual-volatility gate did not fix dense interiors: speed dropped from about `1.121x` to `1.093x`, while dense PSNR moved only about `0.2 dB`.

Interpretation: warmup-delayed residual reuse found a real but bounded image-generation speedup. The next video project should ask whether trajectory-aware scheduling or another video-native compute allocation strategy is a better surface than block reuse.

## Initial Product Question

Frame the new work as an exploration, not a commitment:

> Can diffusion video generation feel real-time on Apple Silicon by spending compute more intelligently across time, frames, and denoising phases?

This is intentionally broader than "make FLUX faster." It leaves room for model choice, frame interpolation, distillation, caching, scheduler changes, streaming previews, and UX tricks.

## Details To Collect From The User

Ask these before implementation starts:

- Target experience: live preview, text-to-video render, image-to-video, webcam stylization, or animation tool?
- Definition of real-time: 24 fps, 12 fps, 4 fps interactive preview, or fast final render?
- Target hardware: same Apple M5 Max 36GB, or broader Apple Silicon support?
- Model preference: FLUX-derived image frames, video diffusion model, diffusion plus interpolation, or open to alternatives?
- Output constraints: resolution, duration, aspect ratio, prompt update latency, seed control, and export format.
- Quality constraints: temporal coherence, prompt alignment, identity preservation, motion smoothness, or visual fidelity.
- UX surface: CLI benchmark first, local web UI, native macOS app, or notebook prototype?
- Publication goal: research benchmark, demo app, blog series, or reusable toolkit?

## Suggested First Milestone

Build a minimal benchmark-first prototype before a polished UI:

1. Generate a short sequence with a chosen baseline method.
2. Measure end-to-end latency, per-frame latency, peak memory, and warmup cost.
3. Save frames/video plus a compact report.
4. Add one acceleration idea at a time.
5. Compare against baseline with interleaved or matched thermal-state runs where possible.

The first milestone should answer: what is the true baseline, where is the time spent, and which part of the pipeline is worth attacking?

## Candidate Technical Directions

These are hypotheses, not commitments:

- Trajectory-aware compute scheduling: spend fewer expensive evaluations when the latent trajectory is stable.
- Frame-to-frame reuse: use previous frame latents or features carefully, with temporal-drift checks.
- Low-fps diffusion plus interpolation: generate keyframes and use a separate interpolation method for perceived smoothness.
- Progressive preview: show lower-step or lower-resolution previews while final frames refine.
- Prompt-aware scheduling: treat dense scenes, text, faces, and simple product-like scenes differently.
- Apple Silicon profiling: identify whether bottlenecks are transformer compute, memory movement, decoding, encoding, or synchronization.

## Suggested Metrics

Use multiple metric families from the start:

- Performance: first-frame latency, steady-state frame time, throughput, peak memory, thermal drift.
- Image quality: LPIPS, SSIM, CLIP or prompt-alignment proxy, and visual grids.
- Video quality: temporal consistency, flicker, motion smoothness, identity/object persistence.
- User experience: time-to-first-preview, time-to-acceptable-preview, final export time.

Avoid letting PSNR or one perceptual score define success by itself.

## Artifact Discipline

Keep generated videos and large frame dumps out of git by default. Track only curated reports, small preview images, and maybe short compressed demo clips if publication requires them.

Recommended new-project output layout:

```text
outputs/
  runs/<timestamp>/metrics.jsonl
  runs/<timestamp>/report.md
  runs/<timestamp>/previews/
  curated/
docs/
  methodology.md
  claims.md
```

## Public-Framing Guardrails

- Say "real-time" only after defining the threshold.
- Separate preview speed from final render speed.
- Separate temporal coherence from single-frame quality.
- Report hardware, memory, model, quantization, resolution, frame count, and scheduler.
- Treat failures as findings when they identify the next constraint.

## Starting Prompt For The New Agent

Use something like this when starting the new workspace:

> We are starting a new benchmark-first project for real-time or near-real-time diffusion video generation on Apple Silicon. Use the completed FLUX block-cache benchmark only as measurement prior art. Do not extend that repo unless explicitly asked. First, help define the minimum baseline experiment, artifact layout, metrics, and model/runtime choice. Keep claims conservative and design the harness so speed and quality can be tested together.

## Current User Note

The user will provide more details before implementation. Treat this document as a starting context packet, not a final project spec.