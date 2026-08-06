# H100 Framewise Refresh Campaign Runbook

This runbook controls the frozen v2 refresh-addressability campaign. Run commands from the
Liveframe repository root. The default runner mode is source/checkpoint/campaign preflight only
and cannot generate proof results.

## Frozen identities

- Manifest: `artifacts/refresh-addressability-manifest.v2.json`
- Manifest digest: `sha256:c97a6fc1f042a7f56eab08273f0f6bd2f343b938498907c267a7d7ab3334d59b`
- Batch: `artifacts/h100-refresh-proof-batch.v2.json`
- Batch digest: `sha256:d9541d8a7e77f0495bf80ed8b8536bed4b56ea0974e48ed1b8732d045d3e4934`
- Source revision: `8db419e341e5fc52542c0b2c4542728420ddfb4a`
- One-step checkpoint: `sha256:bdb1b475fc88d528f510158a0990cd457f02c661b527ceb11ca9e4728533e2d0`
- Output root: `artifacts/h100-refresh-proof-results-v2`

V1 artifacts are superseded and must not be executed.

## Local preflight

```bash
.venv/bin/python tools/run_h100_refresh_campaign.py \
  --protocol artifacts/perceptual-protocol.v1.json \
  --policy-bundle artifacts/perceptual-policy-bundle.v2.json \
  --manifest artifacts/refresh-addressability-manifest.v2.json \
  --batch artifacts/h100-refresh-proof-batch.v2.json \
  --output-root artifacts/h100-refresh-proof-results-v2 \
  --causal-root third_party/Causal-Forcing \
  --checkpoint weights/Causal-Forcing/causal-forcing++/framewise-1step.pt \
  --preflight-output artifacts/h100-refresh-runner-preflight.v2.json
```

The report must state `selected_jobs: 64` and `proof_results_generated: false`. Re-running the
command validates byte equality and refuses to replace a changed report.

## Proof-blind allocation smoke

Run this before authorizing any numerical proof job. It uses a fixed synthetic prompt, zero model
inputs, and deterministic moving pixels for metric allocation; it does not load campaign prompts
or seeds and cannot produce proof results. It exercises
one maximum-context generator forward while both trajectory cache sets are resident, all 243
sequential VAE decode steps, and one frozen metric batch. The three model families are CUDA-resident
in separate phases.

```bash
.venv/bin/python tools/run_h100_refresh_campaign.py \
  --protocol artifacts/perceptual-protocol.v1.json \
  --policy-bundle artifacts/perceptual-policy-bundle.v2.json \
  --manifest artifacts/refresh-addressability-manifest.v2.json \
  --batch artifacts/h100-refresh-proof-batch.v2.json \
  --output-root artifacts/h100-refresh-proof-results-v2 \
  --causal-root third_party/Causal-Forcing \
  --checkpoint weights/Causal-Forcing/causal-forcing++/framewise-1step.pt \
  --identity-anchor-report artifacts/h100-cfpp-1step-denominator.v1.json \
  --allocation-smoke-output artifacts/h100-refresh-allocation-smoke.v1.json
```

The report must state `proof_inputs_loaded: false`, `proof_results_generated: false`,
`pixel_frames: 969`, and contain peak allocated/reserved bytes for maximum-context generation,
full-sequence VAE cache growth, and the metric batch. Do not proceed to proof execution if the
smoke raises CUDA OOM or does not publish all three checkpoints.

## Visibility boundary

Do not add `--execute` until proof visibility is explicitly authorized. Execution additionally
requires the exact manifest digest as an acknowledgment and a passing identity-anchor report.
The instrumented runner produces numerical drift, quality, and identity evidence only. It sets
`timing_protocol: paired-interleaved-instrumented-numerical-only`, so its timing is structurally
ineligible for the 17% wall gate even on driver `580.159.04`.

The candidate and reference trajectories have independent KV caches, cross-attention caches,
prompt tensors, and stochastic tensor objects. They share immutable model weights only. Frames
`0,3,6,...` execute candidate full-refresh anchors; the following two frames substitute from the
latest candidate anchor. Reference K/V is observed for drift and ancestry but is never a candidate
source.

## Authorized numerical execution

This command opens proof results. Do not use it for a trial run.

```bash
.venv/bin/python tools/run_h100_refresh_campaign.py \
  --protocol artifacts/perceptual-protocol.v1.json \
  --policy-bundle artifacts/perceptual-policy-bundle.v2.json \
  --manifest artifacts/refresh-addressability-manifest.v2.json \
  --batch artifacts/h100-refresh-proof-batch.v2.json \
  --output-root artifacts/h100-refresh-proof-results-v2 \
  --causal-root third_party/Causal-Forcing \
  --checkpoint weights/Causal-Forcing/causal-forcing++/framewise-1step.pt \
  --identity-anchor-report artifacts/h100-cfpp-1step-denominator.v1.json \
  --execute \
  --acknowledge-proof-visibility sha256:c97a6fc1f042a7f56eab08273f0f6bd2f343b938498907c267a7d7ab3334d59b
```

The runner refuses unknown jobs, altered campaign artifacts, dirty selected output directories,
source/checkpoint drift, non-H100 GPUs, non-passing identity reports, and output replacement. A
partial run can be resumed only by selecting untouched jobs with repeated `--run-id`; completed
output directories remain immutable.

## Timing follow-up

Do not use instrumented numerical timing for the wall gate. If and only if numerical evidence
passes, add a separate uninstrumented paired replay that:

1. Uses driver `580.159.04`, PyTorch `2.8.0+cu128`, and CUDA `12.8`.
2. Preserves the same frozen jobs, stochastic bytes, three-frame anchor cadence, and outputs.
3. Verifies output identity against the numerical run before accepting timing.
4. Measures paired interleaved complete wall including re-RoPE, selection, cache movement,
   synchronization, decode, and required production overhead.
5. Binds the frozen denominator digest and publishes `paired-interleaved-same-runtime` only after
   those checks pass.

The periodic cadence caps refresh-forward elimination at two thirds, or approximately 13.0351%
of frozen complete wall before overhead. Refresh-only timing cannot satisfy the 17% wall gate;
denoise-side reclaim remains required.
