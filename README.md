# LiveFrame

Real-time diffusion video generation, honestly measured.

LiveFrame currently includes the immutable proof contract, durable experiment control,
causal Wan/CF++ inference on Apple MLX, and digest-bound H100 golden references. Full
native-rate streaming and performance proof remain in progress; no real-time performance
claim is made yet.

The strongest correctness result now replays the released Rolling Forcing long-video
checkpoint from CUDA BF16 through Metal BF16 for 243 latent frames (969 pixel-frame
semantics, 60.56 seconds). Across all 84 rolling windows, cache cursors remain exact,
eviction is active after the 24-frame physical cache fills, the three-frame anchor is
re-RoPE'd at each sliding position, and drift remains bounded rather than compounding
catastrophically. The final latent cosine is 0.998944, so the frozen 0.999 numerical canary
remains visibly failed; it has not been weakened or reinterpreted.

This is MLX/Metal correctness evidence, not an M5 throughput result. The published H100
19.3 FPS row is only a 3.4-second native-window burst. Sustained 60-second M5 FPS, TTFF,
queue, memory, and thermal proof has not yet been measured.

The same-decoder 60.56-second pixel measurement currently reports LPIPS 0.01430 mean
(0.01962 p95), CLIP image cosine 0.99875 mean, temporal CLIP-delta cosine 0.93376 mean,
and temporal LPIPS-motion delta 0.00390 mean. At 60 seconds the point measurements are
0.01554 LPIPS, 0.99750 CLIP cosine, 0.86873 temporal CLIP cosine, and 0.00572 temporal
LPIPS delta. These are measurement results, not an authoritative perceptual pass: no
cross-seed floors were pre-registered before this capture. The harness therefore records
`measurement-only` and cannot convert the failed numerical canary into a pass.

Quality evidence is now deliberately split into three non-substitutable layers:

1. **Trajectory canary — same numbers.** Structural cache checks and the immutable 0.999
	latent cosine canary test numerical reproduction. Quality metrics cannot override it.
2. **Intrinsic non-inferiority — equally good video.** Prompt alignment, temporal stability,
	motion preservation, and flicker are measured per complete video and compared with
	prompt-specific teacher cross-seed center and spread. Every prompt and content stratum
	must pass its frozen one-sided or two-sided margin.
3. **Pairwise identity — still the same video.** Same-seed LPIPS and CLIP trajectory distances
	remain load-bearing identity tripwires. Their deliberately loose limits are separately
	pre-registered on the same-seed identity scale, never inferred from cross-seed content
	distance and never treated as intrinsic quality.

The independent sample is always a complete video; frame count cannot inflate sample size.
Calibration inputs are bound to a frozen protocol digest, derived policy bundles are bound to
their calibration digest, and blind holdouts are bound to the resulting bundle digest. The
already-observed 60.56-second result is therefore ineligible for retroactive promotion.

## Implemented now

- Immutable, canonicalized, SHA-256-bound `gates.v1` contract.
- Versioned append-only NDJSON events.
- Durable detached run submission with stable run IDs.
- Idempotent mobile-friendly submit, status, logs, cancel, and artifact inspection.
- Per-run gate provenance, action events, output capture, and artifact checksums.
- CPU tests for gate tampering, lifecycle durability, idempotency, cancellation, and event encoding.
- Runtime-neutral backend contracts, portable serialized-noise fixtures, protected-first-block
	scheduler transitions, and fail-closed CF++ conversion planning.
- Pinned `mlx-video` installation plus a verified stock Wan2.1 1.3B bfloat16 conversion and
	finite native-resolution smoke output.
- LiveFrame-owned temporal-offset RoPE, causal K/V append/overwrite behavior, and complete
	30-layer native-resolution causal MLX forwards.
- Exact-input H100-to-MLX parity across the complete 21-latent-block 2-step schedule:
	44 denoising forwards, 21 clean-context refreshes, and 23 serialized scheduler-noise draws.
- H100 batch and incremental VAE references proving an exact 1-frame first emission followed
	by 20 four-frame emissions, with zero numerical difference between batch and cached decode.
	These are correctness and bring-up results, not a real-time performance claim.
- Specialized Rolling Forcing parity over 84 windows and 60.56 seconds with exact cache
	cursors, permanent anchor/working-cache probes, active eviction, and explicit Metal device,
	MLX version, macOS, architecture, and Python provenance in v2 reports.
- Digest-bound MLX candidate-latent artifacts plus a same-decoder MLX VAE path for H100
	reference and MLX candidate latents.
- LPIPS, pinned CLIP image-feature, and temporal feature/motion parity measurement. The
	harness fails closed to measurement-only until cross-seed perceptual floors are frozen;
	the 0.999 numerical canary remains a separate verdict.
- Prompt-conditioned intrinsic video metrics, robust teacher center/spread calibration,
	prompt-specific direction-aware non-inferiority bounds, independent-video holdout
	aggregation, and separate pairwise identity tripwires. Synthetic policy tests are complete;
	the real calibration and blinded holdout captures have not yet been run.

## Development

Python 3.11 or newer is required.

```text
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check .
mypy
```

Verify the frozen contract:

```text
liveframe gate verify
liveframe doctor
```

Evaluate a JSON proof-evidence document against the digest-bound native-rate contract:

```text
liveframe evaluate proof-evidence.json
```

Inspect external pins, or fail closed until every required revision is frozen:

```text
liveframe manifest artifacts/upstream.v1.json
liveframe manifest artifacts/upstream.v1.json --require-resolved
```

The checked-in Phase 0 manifest pins full Git and Hugging Face revisions. Large checkpoint
files also carry Hugging Face LFS SHA-256 values and must match after download. PyTorch pickle
checkpoints are treated as untrusted input and may only be loaded through the isolated,
weights-only conversion path. Model execution must not start from mutable `main` references.

Submit a detached command and retain the returned `run_id`:

```text
liveframe submit --idempotency-key first-smoke -- your-command --arg value
liveframe status RUN_ID
liveframe logs RUN_ID
liveframe artifacts RUN_ID
liveframe cancel RUN_ID
```

Set `LIVEFRAME_HOME` or pass `--home` to choose where run manifests, events, logs, and artifacts are stored. Generated media, weights, and the default output directory are excluded from Git.

## Secure control from a phone

The v1 control plane is deliberately not an internet-facing web service:

1. Install Tailscale on the Mac/H100 host and phone.
2. Enable key-only SSH on the host and disable password authentication.
3. Connect from a phone SSH client over the Tailscale address.
4. Use the commands above. Runs are detached, so closing the phone or losing connectivity does not stop them.
5. Use a persistent `tmux` session for interactive setup, but submit actual experiments through `liveframe` so lifecycle and provenance remain durable.

Do not port-forward a LiveFrame runner or expose SSH directly to the public internet. Revoke stale Tailscale devices and SSH keys. On the Mac, prevent sleep for the duration of a declared benchmark and record charger and macOS power mode in run metadata. A read-only mobile dashboard is planned; browser mutation is intentionally deferred until it has short-lived authentication, authorization, CSRF protection, and complete auditing.

## Repository layout

- `src/liveframe/data/gates/`: frozen gate contract and schema.
- `src/liveframe/gates.py`: canonicalization and digest verification.
- `src/liveframe/events.py`: event envelope and durable NDJSON writer.
- `src/liveframe/runs.py`: run store and mobile control operations.
- `src/liveframe/backends/mlx_causal.py`: causal MLX attention, caches, and Wan forwards.
- `src/liveframe/mlx_parity.py`: single-block H100-to-MLX numerical parity.
- `src/liveframe/mlx_pipeline_parity.py`: complete scheduler/cache pipeline parity.
- `src/liveframe/mlx_rolling_parity.py`: eviction-active specialized long-horizon parity.
- `src/liveframe/mlx_vae_decode.py`: same-decoder reference/candidate pixel artifacts.
- `src/liveframe/rolling_perceptual_parity.py`: LPIPS, CLIP, and temporal parity metrics.
- `src/liveframe/perceptual_policy.py`: fail-closed flat and layered policy evaluators.
- `src/liveframe/quality_calibration.py`: digest-bound teacher calibration and blind holdout
	aggregation with one independent row per video.
- `tools/`: pinned H100 capture harnesses.
- `agent/`: research plans and handoff documents.
- `tests/`: CPU-only contract and run-control tests.

## Gate policy

`gates.v1` is append-only and immutable. Every run embeds its gate ID and digest. Any semantic change requires a new artifact such as `gates.v2` and a compatibility changelog; existing results are never silently reinterpreted.
