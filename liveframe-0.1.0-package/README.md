# LiveFrame

**Real-time video generation, honestly measured.**

LiveFrame is an evidence-first measurement framework for real-time autoregressive
diffusion video, spanning NVIDIA CUDA and Apple MLX. It separates four claim layers —
numerical trajectory, intrinsic quality, same-seed identity, and complete-wall
performance — with pre-registered thresholds and digest-bound provenance.

Paper: *LiveFrame: Evidence-First Measurement of Real-Time Autoregressive Diffusion
Video Across CUDA and Apple MLX* (2026).

## This release (0.1.0)

This first release ships the **evidence verifier** for LiveFrame claim ledgers:

```bash
pip install liveframe

# re-hash every unique digest-bound artifact referenced by a ledger (fail-closed)
liveframe verify path/to/liveframe-publication-claims.v1.json --artifacts-root path/to/repo

# independently recompute headline arithmetic from values recorded in the ledger
liveframe recompute path/to/liveframe-publication-claims.v1.json
```

`verify` requires the separately distributed evidence artifacts referenced by the
ledger; model weights and evidence files are not bundled in this Python package.
It rejects missing files, digest mismatches, unsupported schemas, conflicting
references, and artifact paths outside the selected root.

The streaming engine APIs (`stream`, `clip`, `benchmark`) are planned for later
releases. See the repository for the full evidence bundle, claim protocol, and
reports.

## License

Apache-2.0
