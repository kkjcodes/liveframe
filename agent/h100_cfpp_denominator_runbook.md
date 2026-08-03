# H100 one-step CF++ denominator runbook

**Status:** locally validated harness; authoritative H100 execution pending.

## Purpose

Capture the claim-eligible H100 component denominator required before any Delta Scheduling
instrumentation, oracle replay, or DS-on timing. This row does not replace the frozen M5
denominator. It determines whether the pre-registered 17% end-to-end DS gate is mathematically
attainable on H100.

## Frozen workload

- One-step CF++, 21 latent blocks, 81 generated pixel frames at 480x832 and 16 FPS.
- Full-attention fixture semantics: `local_attn_frames=-1`, `sink_frames=0`.
- Source revision: `8db419e341e5fc52542c0b2c4542728420ddfb4a`.
- Checkpoint SHA-256: `sha256:bdb1b475fc88d528f510158a0990cd457f02c661b527ceb11ca9e4728533e2d0`.
- VAE SHA-256: `sha256:38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`.
- Output-latent anchor: `sha256:966be57aebc4369ba9c197168dddc5c587cd9c891ba73ce613e5956d4e598512`.
- Streamed-pixel FP16 anchor: `sha256:d47d7b5cb1b93aeed3b004d2a16f90bef389319ef3f5dd0c62f4efff5b87124c`.
- TF32 disabled. One complete warmup followed by three measured trials.
- End-to-end wall includes denoising, mandatory clean-cache refresh, incremental causal VAE,
  pixel transfer/materialization, and H.264 encoding.

## Environment

Use the same H100 80 GB image, driver, CUDA, PyTorch, pinned Causal-Forcing checkout, checkpoint,
and Wan model root used for the existing H100 fixture. The private LiveFrame repository must be
checked out at the commit under evaluation. `ffmpeg` with `libx264` support is required.

Record before execution:

```bash
git -C "$LIVEFRAME_ROOT" rev-parse HEAD
git -C "$CAUSAL_FORCING_ROOT" rev-parse HEAD
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -V
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
ffmpeg -hide_banner -encoders 2>/dev/null | grep libx264
```

## Authoritative command

Run from the pinned Causal-Forcing checkout so its `utils.wan_wrapper` and `wan_models` paths
resolve exactly:

```bash
cd "$CAUSAL_FORCING_ROOT"
PYTHONUNBUFFERED=1 python "$LIVEFRAME_ROOT/tools/benchmark_h100_pipeline.py" \
  --checkpoint "$CFPP_1STEP_CHECKPOINT" \
  --student-steps 1 \
  --latent-blocks 21 \
  --local-attn-frames -1 \
  --sink-frames 0 \
  --repeats 3 \
  --telemetry-interval 1 \
  --output "$LIVEFRAME_ROOT/artifacts/h100-cfpp-1step-denominator.v1.json"
```

The runner fails before publication on checkpoint/VAE mismatch or output identity drift. Report
publication is atomic.

## Acceptance and feasibility

Accept the row only when:

1. `claim_eligible` and `identity_anchor.passed` are both true.
2. GPU is NVIDIA H100 80 GB, TF32 is false, and runtime/source digests match this runbook.
3. All three samples reproduce both output anchors and contain complete component and telemetry
   rows.
4. No CUDA OOM, thermal throttle, competing GPU process, or failed H.264 encode occurred.
5. The retained JSON SHA-256 is recorded in `agent/liveframe_execution_plan.md` before any DS
   instrumentation or oracle result is produced.

Compute the conservative DS-addressable share as:

```text
denoising_seconds_median / wall_seconds_median
```

Do not count mandatory clean-cache refresh unless a separately specified DS mechanism can skip
it while preserving cache semantics. H100 is eligible as the primary M4 wall-clock backend only
if the measured addressable share exceeds 17% by enough margin to include selection, compaction,
cache movement, synchronization, and signal overhead. Otherwise record a pre-measurement
feasibility STOP.