# H100 holdout reference capture report

**Date:** 2026-07-27  
**Status:** complete and validated; candidate capture and holdout verdict pending

## Frozen identities

- Capture protocol: `sha256:aef6a7ad9426c97a9ae7f99c9a7a4f93799a574ca9e88d341da430c60ca75c52`
- Run manifest: `sha256:8967e70cbc3978544eea92428d2dda78044ebbcaae91e933271f7e1abd72fdc4`
- Rolling checkpoint: `sha256:1f123dce60e60424a9c6d949643176840fceffb947932997bcf1202d93656d98`
- H100 reference batch: `sha256:10af5568b5c42ac9d081621a0f63862ec4c96c8751d03c8b12e775aa6dab337a`
- Shared-noise index: `sha256:3fbfbf2896e566f663a1b781e20872c393b25e0d6333d33268b3c069e4b72a17`
- Shared-noise contract: `liveframe.shared-rolling-noise.v1`

## Completion

The sealed CUDA reference phase produced 128/128 manifest-owned videos: 64 at the exact
15-second horizon and 64 at the exact 60-second horizon. The output tree occupies approximately
105 GB. The final validation read every declared artifact and required metric, validated every
stochastic tensor table and per-tensor checksum, and required each captured payload digest to
equal the corresponding frozen source-plan digest.

The final executor resume report contains 128 jobs, 112 completed run IDs, 16 validated and
checksum-skipped run IDs, and 6,551.20 seconds of wall time. No candidate/reference pairwise
metric and no holdout verdict has been computed.

## Fail-closed incidents

### Runtime float identifier mismatch

The first launch stopped before publishing a result at the third rolling window. Torch 2.8
reported the warped 500-step value as `833.3333740234375`; the plan created under local Torch
recorded the frozen semantic name with `833.3333129882812`. Tensor bytes were not mismatched.
The repair made CUDA request transition names from configured integer schedule steps through the
same `transition_name()` helper used by the serializer and MLX. Scheduler math and frozen plan
bytes did not change. Focused tests passed before redeployment, and the H100 replay then advanced
beyond the failed transition under full GPU load.

### Network-volume quota

RunPod's FUSE mount exposed the shared backing filesystem through `df`, which showed roughly
100 TB free, while the account volume had a separate 300 GB quota. The capture stopped after 16
published results while atomically writing job 17. There was no Python traceback or kernel OOM.
All 16 results and their captured stochastic streams were validated before recovery. Only the
three-file incomplete directory was removed. The volume was expanded to 500 GB and the frozen
batch resumed, checksum-skipping the 16 complete jobs.

Operational rule: use the configured RunPod volume quota and `du -s --block-size=1 /workspace`
for campaign capacity planning; do not use the FUSE mount's `df` availability as quota evidence.

## Next phase

Run `artifacts/mlx-holdout-candidates.v1.json` on the Apple M5 Max with the same eight shared
plans. Preserve each result's timing breakdown and runtime provenance. Candidate timing is a
first-order resident-model batch-throughput characterization, not TTFF or a thermally controlled
sustained-streaming claim. Compute per-video native FPS as:

`pixel_frames / (diffusion_seconds + clean_cache_refresh_seconds + scheduler_seconds + vae_seconds)`

After all 128 candidates validate, retrieve the H100 references and invoke the evaluator once.
The evaluator must complete literal stochastic byte identity for all 128 pairs before loading
CLIP/LPIPS or computing any pairwise metric.