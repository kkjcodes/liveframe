# LiveFrame

LiveFrame defines **real-time** as at least **16 native generated frames per second for a continuous 60-second stream**, with time-to-first-frame, p95 chunk latency, thermal state, memory, intrinsic quality, same-seed identity, and long-horizon drift disclosed. Interpolated or duplicated presentation frames never count as generated FPS.

The strongest clean Apple result in this repository is **1.418 native generated FPS**, measured on an Apple M5 Max with MLX/MLX Metal 0.32.0, Wan2.1-T2V-1.3B Causal-Forcing++ one-step generation, and 480x832 output over an 81-frame fixture. It is **11.28x short of 16 FPS** and is not a 60-second sustained-streaming result.

That miss is the headline, not a footnote. LiveFrame's contribution is an evidence-first methodology, one exact-output systems speedup, and three scoped negative results that prevented broader false claims.

## Results

| Result | Measurement | Scope |
| --- | --- | --- |
| M5 allocator optimization | 69.167s to 57.127s complete wall; 1.171 to 1.418 FPS; `1.2108x` speedup | Apple M5 Max, MLX 0.32.0, one-step CF++, 480x832, 81 frames |
| Cross-runtime AR divergence | pairwise identity passed 0/128 videos and 0/32 prompt-horizon cells | Frozen Rolling Forcing checkpoint, byte-identical CUDA/MLX stochastic inputs, 15s and 60s |
| M5 target gap | 1.418 FPS, 11.28x below the 16 FPS threshold | Clean short fixture, not a hardware impossibility or sustained claim |
| Periodic refresh reuse | 0/16 quality cells; temporal identity failed 64/64 videos | H100 numerical campaign, two high-redundancy strata, 15s and 60s |

The M5 speedup came from increasing the bounded MLX free-buffer cache from 1 GiB to 4 GiB before model loading. All measured trials reported the same pre-encode raw RGB digest and zero swap. The original 97 MB raw RGB payload was not retained; the immutable benchmark artifact records the repeated digest, while the shipped H.264 file and its decoded RGB have separate hashes. See [the ledger audit](agent/liveframe_ledger_audit.md).

The 60-second M5 holdout campaign measured 0.185 FPS median, but concurrent memory pressure and sustained swap make that timing ineligible as the clean headline denominator. Its quality and identity evidence remains valid because output bytes were unaffected by timing contamination.

## What Failed, Precisely

### 1. Cross-runtime autoregressive reproduction

CUDA and MLX consumed byte-identical serialized BF16 stochastic tensors under the frozen Rolling Forcing schedule. The MLX outputs often remained independently plausible, but they did not preserve the same trajectory:

- Intrinsic quality: 90/128 videos and 19/32 prompt-horizon cells passed.
- Same-seed identity: 0/128 videos and 0/32 cells passed.
- Pairwise temporal CLIP identity: 0/128 videos passed.

This rejects the tested checkpoint/runtime pair under the frozen portability contract. It does not establish that every CUDA and MLX video implementation must diverge.

### 2. The tested M5 configuration was not real-time

The clean optimized row is 1.418 native FPS at 480x832. Quantization, lower-resolution tiers, different models, custom Metal kernels, decoder redesign, and disclosed interpolation were not exhausted, so this is a configuration result rather than an Apple Silicon impossibility claim.

### 3. Periodic refresh K/V reuse was not quality-safe

The tested H100 mechanism reused raw normalized K/V from a full-refresh anchor for the next two framewise refreshes after reapplying temporal RoPE. It failed all 16 frozen prompt-horizon cells. No authoritative timing replay was run because numerical quality GO was a prerequisite.

The broader denoise-side Delta Scheduling oracle was **never run**. The research track stopped after the refresh gate by choice. On the frozen H100 denominator, a denoise-only path needed at least 70.3667% denoise reclaim before overhead to reach the 17% complete-wall gate. That is an untested feasibility bar, not a failed oracle.

## Measurement Method

LiveFrame separates four claims that are often conflated:

1. **Numerical trajectory:** cache invariants and latent drift under identical inputs.
2. **Intrinsic quality:** prompt alignment, motion, flicker, and temporal behavior against prompt-specific teacher distributions.
3. **Same-seed identity:** whether reference and candidate remain recognizably the same generated video.
4. **Complete-wall performance:** end-to-end timing including denoise, mandatory refresh, VAE, materialization, synchronization, and encoding.

The evidence system includes:

- Pre-registered prompts, seeds, content strata, 15-second and 60-second horizons, thresholds, and aggregation.
- Serialized stochastic tensors with literal cross-runtime byte checks.
- Digest-bound checkpoints, source revisions, metric weights, runtimes, and outputs.
- Independent numerical and timing eligibility.
- Component-resolved timing and zero-swap checks.
- Atomic publication and fail-closed complete-campaign validation.
- Proof-blind H100 allocation testing before prospective result visibility.

The machine-readable claim surface is [artifacts/liveframe-publication-claims.v1.json](artifacts/liveframe-publication-claims.v1.json). The publication plan and limitations are in [agent/liveframe_publication_package.md](agent/liveframe_publication_package.md).

## Evidence Release

Evidence is split by cost and reproducibility:

- **Compact release:** schemas, manifests, metric JSON, decision artifacts, and reports.
- **Representative previews:** 64 CUDA/MLX pairs selected using the protocol's pre-registered seed ranks 0 and 2 at both horizons.
- **Cold archive:** complete raw pixel, latent, stochastic, and refresh campaign corpora.

The SHA-bound extraction manifest is [artifacts/liveframe-publication-release-tiers.v1.json](artifacts/liveframe-publication-release-tiers.v1.json). Generated release payloads are staged under ignored `release/` storage rather than committed to Git.

## Reproduction Boundary

LiveFrame source is Apache-2.0. Upstream Causal-Forcing source, Causal-Forcing checkpoints, Wan2.1 weights, and tokenizer weights are not redistributed in this repository. Users acquire pinned assets from their publishers and verify them against [artifacts/upstream.v1.json](artifacts/upstream.v1.json). See [THIRD_PARTY.md](THIRD_PARTY.md) before distributing a reproduction bundle.

The frozen evidence revision is tagged `evidence-freeze-2026-08-06`. That tag resolves to commit `ea742ee2ea438b6d9ff2d1bca0b4053f9eeaeb44` and contains the campaign evidence and initial publication ledger.

## Development

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check src/liveframe tests tools
mypy
```

Metal-dependent tests require a native macOS session with an accessible Metal device. The VS Code sandbox can run the CPU suite but cannot initialize MLX Metal.

Useful verification commands:

```bash
liveframe gate verify
liveframe doctor
python tools/prepare_publication_release.py --help
```

## Repository Layout

- `src/liveframe/`: runtime-neutral contracts, MLX backend, quality policy, evidence capture, and refresh proof code.
- `tools/`: pinned benchmark, H100 campaign, and publication extraction tools.
- `artifacts/`: compact immutable protocols, manifests, measurements, and decisions.
- `agent/`: standalone reports, runbooks, project state, and publication package.
- `tests/`: contract, evidence, policy, backend, and publication-ledger tests.

## License

LiveFrame is licensed under Apache-2.0. See [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md).
