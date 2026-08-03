# H100 quality pilot report

**Date:** 2026-07-25  
**Status:** operational pilot passed; quality calibration and holdout remain pending

## Frozen identities

- Protocol: `sha256:aef6a7ad9426c97a9ae7f99c9a7a4f93799a574ca9e88d341da430c60ca75c52`
- Manifest: `sha256:8967e70cbc3978544eea92428d2dda78044ebbcaae91e933271f7e1abd72fdc4`
- Pilot batch: `sha256:c9e8e2882e7a85059473e8d8d3fa7a896bbb8295f75db8dc5c53aefd00130525`
- Rolling checkpoint: `sha256:1f123dce60e60424a9c6d949643176840fceffb947932997bcf1202d93656d98`

## Runtime

- GPU: NVIDIA H100 80GB HBM3
- Driver: 550.144.03
- PyTorch: 2.8.0+cu128
- CUDA runtime: 12.8
- Python: 3.12.3
- TF32: disabled

The migrated pod driver differs from the previously recorded 550.163.01 and is retained as
runtime provenance. It does not change any frozen workload identity.

## Outcome

All four prompt-conditioned jobs completed and strict restart validation returned zero pending
jobs. Artifact SHA-256 values, exact frame counts, finite metric sets, and result identities were
recomputed from disk after execution.

| Prompt | Horizon | Latent / pixel frames | Diffusion | VAE | Metrics | Executor wall |
|---|---:|---:|---:|---:|---:|---:|
| limited_hold_01 | 15s | 60 / 237 | 12.923s | 5.571s | 2.959s | 23.668s |
| limited_hold_01 | 60s | 243 / 969 | 53.380s | 22.598s | 11.702s | 94.372s |
| dense_motion_01 | 15s | 60 / 237 | 12.733s | 5.443s | 2.947s | 22.869s |
| dense_motion_01 | 60s | 243 / 969 | 53.437s | 22.854s | 11.662s | 94.314s |

Total pilot wall time was 237.799 seconds. The four result directories occupy 3.3 GiB. Linear
projection for 256 calibration captures is approximately 4.23 H100 hours and 211 GiB, below the
frozen 80-hour and 1000-GiB ceilings. This projection excludes setup and final ingestion overhead.

The stochastic manifests contain 61 tensors for each 60-latent job and 244 tensors for each
243-latent job: one initial-noise tensor plus one scheduler-noise tensor per declared transition.

## Visual sanity

Twelve-frame contact sheets from each checksummed pixel artifact show no blank output, decode
corruption, abrupt tensor discontinuity, or catastrophic long-horizon collapse. The observations
are not a quality-policy verdict:

- The held-animation prompt drifts into lateral camera motion and eventually loses the subject,
  especially at the 60-second horizon.
- The dense-motion prompt remains a coherent market chase but changes runners, storefronts, and
  signage over time.

These behaviors reinforce the need for complete calibration and frozen intrinsic thresholds.
They must not be converted into ad hoc pilot gates or used to retune the protocol.
Because every pilot job used calibration seed 11003, the pilot cannot establish cross-seed
non-degeneracy. That conclusion remains pending until all eight calibration seeds are ingested.

## Decision

The execution machinery is ready for the 256-job teacher calibration. Holdout remains sealed.
Full calibration must use the same protocol, manifest, checkpoint, metric states, exact horizons,
and checksum-valid restart path. No policy may be derived until all 256 calibration results pass
complete ingestion and sanity checks.
