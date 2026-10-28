# LiveFrame 1.0 local data: what was deleted and how to rebuild it

Written 2026-10-28 before clearing about 335 GB of local-only LiveFrame 1.0 data (the MLX
real-time diffusion video study). Every tracked receipt (JSON verdicts, protocols, pins) stays
in git. Only gitignored payloads are removed. The 1.0 code in `src/liveframe` and `tools/` is
untouched, so every step below still runs.

## 1. Upstream weights (`weights/`, about 64 GB): cheap, download only

All sources are pinned with revisions and SHA-256 digests in `artifacts/upstream.v1.json`.

```bash
# PyTorch sources (needs ~/.hf_token.env or `hf auth login`)
hf download Wan-AI/Wan2.1-T2V-1.3B --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a \
  --local-dir weights/Wan2.1-T2V-1.3B
hf download zhuhz22/Causal-Forcing --revision 2f8eb8bb6eeb1238da9d13e5420d342a74d634a6 \
  causal-forcing++/framewise-1step.pt causal-forcing++/framewise-2step.pt chunkwise/longvideo.pt \
  --local-dir weights/Causal-Forcing
```

Then check each file's SHA-256 against `artifacts/upstream.v1.json`.

**Stock Wan MLX** (`weights/Wan2.1-T2V-1.3B-MLX`): converted with Blaizzy/mlx-video at revision
`87db56a5`, module `mlx_video.models.wan_2.convert`, MLX 0.32.0. Full record and expected
digests in `artifacts/mlx-stock-wan.v1.json`.

**CF++ and Rolling Forcing MLX** (`weights/*-MLX`): rebuild with the repo's own converter,
`liveframe convert-cfpp`, from the three `.pt` files above and the stock Wan MLX dir. Expected
output digests are in `artifacts/weights-conversion-receipts/`.

**Upstream source** (`third_party/Causal-Forcing`):
`git clone https://github.com/thu-ml/Causal-Forcing.git third_party/Causal-Forcing && git -C third_party/Causal-Forcing checkout 8db419e341e5fc52542c0b2c4542728420ddfb4a`

## 2. Noise plans (`artifacts/*-noise.v1`, about 4.5 GB): free, deterministic

`tools/build_taew21_campaign_noise.py` materializes them as pure functions of
(seed, latent_frames). The tracked index `artifacts/taew21-campaign-noise-index.v1.json` binds
every payload digest, so a rebuild is verified byte for byte.

## 3. Holdout corpora and campaign outputs (about 270 GB): expensive, needs GPU time

- `artifacts/h100-holdout-results` (CUDA side): `tools/prepare_h100_holdout_reference_batch.py`
  then `tools/run_h100_quality_batch.py` on a rented H100.
- `artifacts/mlx-holdout-results` (MLX side): `tools/run_mlx_quality_batch.py` on an M5 Max.
- `outputs/taew21-campaign`, `outputs/taew21-study-bundle`, `outputs/golden`,
  `outputs/perceptual`, `outputs/parity`: the `tools/*taew21*`, `tools/capture_h100_*` and
  `tools/benchmark_*` scripts, per the runbooks in `agent/`.
- `release/liveframe-v1/previews`: `tools/prepare_publication_release.py` from the two holdout
  corpora above.

The verdicts these produced are already in git (`artifacts/quality-holdout-evaluation/`,
`artifacts/liveframe-publication-claims.v1.json`, `artifacts/taew21-*`), so the published
conclusions survive without the payloads.

**Decision (Kumar, 2026-10-28):** no plan to resubmit the 1.0 paper, so these were deleted
without an archive copy. If the numbers are ever needed again, regenerate them in this order:

1. Rebuild weights (section 1) and noise plans (section 2). Confirm digests match.
2. Rent one H100 (Kumar creates the pod). Run the CUDA reference batch with
   `tools/prepare_h100_holdout_reference_batch.py` then `tools/run_h100_quality_batch.py`.
   Stop the pod as soon as it finishes.
3. On the M5 Max, run `tools/run_mlx_quality_batch.py` for the MLX candidate side.
4. Re-run `tools/evaluate_quality_holdout.py`. The new verdict should match the committed
   `artifacts/quality-holdout-evaluation/verdict.json`. A mismatch is itself a finding.
5. Only if previews are needed again, run `tools/prepare_publication_release.py` and compare
   its metric digests with `artifacts/liveframe-publication-release-tiers.v1.json`.

Same-seed byte identity is part of the protocol, so regenerated outputs should hash the same
on the same hardware and runtime versions. Cross-runtime results were never identical; that
was one of the published negative results.

## Kept on purpose

- `agent/world/evidence/**` frame payloads (about 6.3 GB, gitignored). These are LiveFrame
  World receipts for the launch numbers, and the committed ledgers list them, so
  `liveframe world verify` fails without them. Rebuilding means re-renting the GPUs.
- `release/liveframe-v1/metrics` (about 1 MB): the compact per-video metrics whose digests are
  recorded in `artifacts/liveframe-publication-release-tiers.v1.json`.
- `.venv`, `.hf_token.env`, `publishing/`, `paper/`.
- `liveframe-0.1.0-package/`: 9 of its files are committed to git despite the `.gitignore`
  entry, so deleting it locally would show up as a deletion in the next commit.
