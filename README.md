# LiveFrame

[![PyPI](https://img.shields.io/pypi/v/liveframe.svg)](https://pypi.org/project/liveframe/)
[![Python](https://img.shields.io/pypi/pyversions/liveframe.svg)](https://pypi.org/project/liveframe/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://github.com/kkjcodes/liveframe/actions/workflows/ci.yml/badge.svg)](https://github.com/kkjcodes/liveframe/actions/workflows/ci.yml)

LiveFrame helps engineers determine whether a streaming video-generation system works for a real
workload, understand why it fails, and verify that an optimization improves complete-wall
performance without silently weakening quality or deployment eligibility. The active product
mission is documented in [agent/liveframe_product_mission.md](agent/liveframe_product_mission.md).

## LiveFrame World: runtime truth for interactive world models

Point the harness at an action-conditioned world model and get a verified
runtime + control + stability report - one command, one GPU, minutes:

```bash
pip install "liveframe[world]"
# (or from a source checkout: pip install -e ".[world]")

# Prove the loop end to end on CPU first (~seconds):
liveframe world run synthetic --suite smoke

# Bring your own model: any FILE.py:ClassName implementing the adapter
# protocol, in-process. pip users write their own file (see QUICKSTART.md for a
# complete paste-and-run adapter); source checkouts can start from the template
# in examples/byo_adapter/drift_world.py. Declare every action the suites
# command (W S A D L R U N 0) or the case fails with a clear contract error.
liveframe world run my_world.py:MyWorld --suite quick

# Same adapter file, running inside your model's own environment:
liveframe world run my_model.py:MyWorld --suite quick \
    --python /path/to/model-env/bin/python

# Matrix-Game 2.0 reference adapter (weights + checkout required; run
# `liveframe world doctor matrix-game-2 --python ...` first - every failed
# check prints its exact fix):
liveframe world run matrix-game-2 --suite quick --python /opt/envs/mg2/bin/python
```

Each run writes a self-contained `report.html` (no scripts, no network), a
machine-readable `report.json`, and a digest-bound ledger; the command prints
the `report.html` path when it finishes, and
`liveframe world verify <run>/ledger.json` re-checks every artifact byte.
(`smoke` and `quick` are deliberately short, so `realtime_pass` reports `null`
- the real-time rule needs a continuous 60-second stream; use `soak` for it.)
Generated FPS counts decoded native frames only - interpolated or duplicated
frames never count, cold start is reported as context rather than blended into
generation numbers, and vendor claims are compared with neutral verdicts
(`different configuration` when the claim names a setup, else `config not
stated`) - the builder never says "reproduced". Suites: `smoke`,
`quick`, `standard`, `soak`, `memory` (`liveframe world suites`). Also:
`world compare A B` (config differences listed before any numbers) and
`world export --format worldmark|wbench|raw` for interop with quality-focused
benchmarks. Start at [QUICKSTART.md](QUICKSTART.md) for a step-by-step guide -
install, a paste-and-run adapter, reading the report, and verifying the ledger -
or the template in [examples/byo_adapter/](examples/byo_adapter/).

### Measured references (with receipts)

We ran four published interactive world models through the harness, each in a
stated configuration, every number backed by a re-verifiable ledger. The headline
isn't "these models are slow" - it's that published frame rates are hard to
reproduce from what's shipped. When a claim is fully specified and we run that
exact config (Matrix-Game 2.0), it reproduces; otherwise we show which condition
differs (a different GPU, a proprietary engine vs the open weights, or a streaming
mode that isn't the released code).

| Model | Claim (as published) | Measured | Why they differ (stated conditions) |
|---|---|---|---|
| Matrix-Game 2.0 | 25 fps (1× H100) | **19.7 fps** | same H100, same steps - **reproduces** |
| ABot-World-0 | 16 fps (720p, 1× RTX 5090) | **5.85 fps** | same RTX 5090; remaining gap = their optional fp8/sage stack off (not pip-installable) + the interactive delivery path. (A100 reference: 3.70) |
| Oasis-500M | 20 fps (Decart's proprietary engine) | **1.06 fps** | we measured the open 500M weights vanilla (1× A100 PCIe) - a different artifact than the engine behind the claim |
| HY-World 1.5 | 24 fps (streaming; GPU/steps unstated) | **1.36 fps** | released pipeline runs batch, not streaming - not directly comparable |

**None of these results contradict a vendor's claim.** The one claim that is fully
specified and matched (Matrix-Game 2.0) reproduces; the other three can't be tested
as published from what's shipped - a different GPU class, a proprietary serving
engine, or a mode not in the released code. That reproducibility gap, stated
precisely, is the finding.

It times frames as they are delivered (so it errs low, never inflates) and counts
native decoded frames only. We run the released pipeline with conservative settings
(optional fp8/attention/compile off - for ABot this includes stubbing its optional
fp8 kernel, unused at our settings), so these are floors, not each model's best.
The Oasis adapter decodes every frame as it is generated (the interactive cost);
the upstream script batch-decodes at the end. Every number reproduces from a
digest-bound ledger (`liveframe world verify`). Matrix-Game 3.0's claim is 9 GPUs
(8 DiT + 1 VAE) - more than a single node - so its cluster figure is pending; its
single-GPU number is 2.04 fps.

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
liveframe qualify --backend synthetic --profile interactive-480p \
	--output artifacts/synthetic-qualification.v1.json
liveframe qualification verify artifacts/synthetic-qualification.v1.json
python tools/prepare_publication_release.py --help
```

The synthetic qualification exercises request construction, native-frame accounting, timing, and
profile gates without a GPU. Its overall verdict is intentionally `incomplete` until required
quality, long-horizon, memory, and thermal evidence is attached; passing FPS alone never qualifies
a deployment.

## Repository Layout

- `src/liveframe/`: runtime-neutral contracts, MLX backend, quality policy, evidence capture, and refresh proof code.
- `tools/`: pinned benchmark, H100 campaign, and publication extraction tools.
- `artifacts/`: compact immutable protocols, manifests, measurements, and decisions.
- `agent/`: standalone reports, runbooks, project state, and publication package.
- `tests/`: contract, evidence, policy, backend, and publication-ledger tests.

## License

LiveFrame is licensed under Apache-2.0. See [LICENSE](LICENSE) and [THIRD_PARTY.md](THIRD_PARTY.md).
