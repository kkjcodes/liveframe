# H100 quality pilot runbook

Status: locally validated; not yet executed on H100.

## Frozen inputs

- Protocol: `artifacts/perceptual-protocol.v1.json`
- Protocol digest: `sha256:aef6a7ad9426c97a9ae7f99c9a7a4f93799a574ca9e88d341da430c60ca75c52`
- Manifest: `artifacts/perceptual-run-manifest.v1.json`
- Manifest digest: `sha256:8967e70cbc3978544eea92428d2dda78044ebbcaae91e933271f7e1abd72fdc4`
- Pilot batch: `artifacts/h100-quality-pilot.v1.json`
- Pilot batch digest: `sha256:c9e8e2882e7a85059473e8d8d3fa7a896bbb8295f75db8dc5c53aefd00130525`
- Checkpoint digest: `sha256:1f123dce60e60424a9c6d949643176840fceffb947932997bcf1202d93656d98`

The pilot contains four teacher jobs: `limited_hold_01` and `dense_motion_01`, each at
60 and 243 latent frames. These decode to exactly 237 and 969 pixel frames.

## Pod prerequisites

1. NVIDIA H100 80GB, driver `550.163.01` or the recorded replacement, and CUDA 12.8.
2. The pinned Causal-Forcing checkout and `chunkwise/longvideo.pt` checkpoint.
3. Python 3.11+ with this project, Causal-Forcing dependencies, PyTorch CUDA, LPIPS 0.1.4,
   Transformers, Pillow, NumPy, and OmegaConf installed.
4. Enough local disk for at least 5 GiB for the pilot. Full calibration is estimated below
   250 GiB and remains under the frozen 1000 GiB ceiling.
5. Hugging Face model snapshots available for the pinned Wan, UMT5, and CLIP revisions.

## Preflight

From the repository root:

```bash
python tools/run_h100_quality_batch.py --help
python -m liveframe.quality_capture prepare \
  --protocol artifacts/perceptual-protocol.v1.json \
  --manifest artifacts/perceptual-run-manifest.v1.json \
  --output-root artifacts/h100-quality-results \
  --output /tmp/rebuilt-pilot.json \
  --run-id quality-8f0c2b4cef7d21d863f6 \
  --run-id quality-cd2f6f745bfe759e2b36 \
  --run-id quality-46026182f137efa0bfae \
  --run-id quality-1399d2d968b8c20ee468
```

The rebuilt batch must contain four jobs and report the frozen pilot batch digest above. Do not
substitute the rebuilt file for the committed pilot unless its canonical batch digest matches.

## Pilot command

Run only after the explicit H100 start notification:

```bash
python tools/run_h100_quality_batch.py \
  --protocol artifacts/perceptual-protocol.v1.json \
  --manifest artifacts/perceptual-run-manifest.v1.json \
  --batch artifacts/h100-quality-pilot.v1.json \
  --output-root artifacts/h100-quality-results \
  --checkpoint "$LONGVIDEO_CHECKPOINT" \
  --metric-batch-size 8
```

The executor loads the generator, text encoder, VAE, CLIP, and LPIPS once. Every job uses its
manifest seed, serializes initial and scheduler noise, saves BF16 latents and lossless uint8
pixels, checks exact frame counts, measures the saved pixels, and atomically writes `result.json`
only after all checks pass.

## Stop conditions

Stop the pilot without launching calibration if any condition occurs:

- checkpoint, protocol, manifest, batch, metric artifact, CLIP state, or LPIPS state mismatch;
- CUDA out-of-memory or peak storage projection above the frozen ceiling;
- output latent or pixel frame count mismatch;
- non-finite or incomplete intrinsic metrics;
- elapsed time indicates the four-job pilot will exceed the approved 90-minute window;
- restart validation reports altered result identity or artifact checksums.

After success, rerun the preflight `prepare` command against the same output root. It must
checksum-validate all four results and emit a batch containing zero jobs. Do not ingest an
incomplete calibration, derive a policy from the pilot, launch holdout jobs, or inspect holdout
output.

## Authorized calibration sequence

The runner is externally pinned to protocol `aef6...75c52`, manifest `8967...2fdc4`, and
checkpoint `1f123d...6d98`. Internally consistent alternative artifacts are rejected before
model initialization.

1. Confirm strict resume reports 252 pending jobs; the four pilot jobs must be credited.
2. Run `artifacts/h100-calibration-first-seed.v1.json` (32 declared jobs). The executor must
  checksum-skip the four pilot jobs and complete the remaining 28.
3. Run `tools/check_h100_calibration_phase.py`. It requires all 16 prompts at both horizons for
  seed 11003, exact pixels, finite metrics, valid checksums, and projected burn at most 5.2875
  H100 hours and 263.75 GiB.
4. Only after that report passes, run `artifacts/h100-calibration-full.v1.json`. The executor
  checksum-skips all 32 first-seed jobs and completes the remaining 224.
5. Ingest all 256 results and run the complete calibration sanity report. Cross-seed
  non-degeneracy remains pending until this step passes.

Retain raw pixels, latents, and stochastic artifacts through complete ingestion, sanity
verification, policy derivation, and archival review. Do not launch or inspect holdout jobs.
