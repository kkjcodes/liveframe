# M5 one-step CF++ clean baseline runbook

Status: authoritative timing completed on 2026-08-02. See `agent/m5_cfpp_baseline_report.md`.

## Frozen inputs

- Model: `weights/CFPP-framewise-1step-MLX`
- Fixture: `outputs/golden/cfpp-1step-native-21block-h100`
- Fixture manifest: `sha256:d621037dc84c416dd7182991fc8e3f10a0245894cf5fd74002a517dc65862eb9`
- Converted model: `sha256:40735e0eb76c4a710163d337a5167cee2d7b1ed27dea9b1d8357714bebc1115b`
- Source checkpoint: `sha256:bdb1b475fc88d528f510158a0990cd457f02c661b527ceb11ca9e4728533e2d0`
- VAE: `sha256:977530e453dbfabbab31e2972e1577d8d7e2840ba7410c81aa3fd421c0cd7414`

## Restart boundary

Restart after closing Docker, Claude, Chrome, and other GPU or memory-heavy applications. After
login, open only VS Code and one terminal in this workspace. Do not start Docker or another model
process. The preflight permits at most 128 MiB startup swap and requires at least 25% free memory.
The runner caps MLX at Metal's 28.08 GiB recommended working set and caps reusable free buffers at
1 GiB. Do not raise either allocator limit.

An explicitly claim-ineligible diagnostic on the capped path completed on 2026-08-02 with a
26.661 GiB MLX peak, 0.520 GiB free cache, 15.139 GiB active memory after the trial, and a -32 MiB
swap delta. It passed the pinned latent quality anchor and completed one 81-frame trial in 66.996
seconds at 1.209 generated FPS.

## Post-restart preflight

```zsh
cd /Users/kumarjha/Code/liveframe
.venv/bin/python tools/benchmark_mlx_cfpp.py \
  --model-dir weights/CFPP-framewise-1step-MLX \
  --fixture outputs/golden/cfpp-1step-native-21block-h100 \
  --output artifacts/m5-cfpp-1step-baseline.v1.json \
  --video-output outputs/m5-cfpp-1step-baseline.mp4 \
  --preflight-only
```

Do not relax either threshold if preflight fails. Close the named processes and restart again.

## Authoritative baseline

```zsh
cd /Users/kumarjha/Code/liveframe
set -o pipefail
PYTHONUNBUFFERED=1 caffeinate -dimsu .venv/bin/python tools/benchmark_mlx_cfpp.py \
  --model-dir weights/CFPP-framewise-1step-MLX \
  --fixture outputs/golden/cfpp-1step-native-21block-h100 \
  --output artifacts/m5-cfpp-1step-baseline.v1.json \
  --video-output outputs/m5-cfpp-1step-baseline.mp4 \
  --warmups 1 \
  --trials 3 \
  2>&1 | tee outputs/m5-cfpp-1step-baseline.log
```

The run fails closed if startup state is dirty, any warmup or trial creates more than 128 MiB of
swap, asset ancestry differs, any trial misses the pinned latent quality anchor, or publication is
incomplete. Reserve 8-10 uninterrupted minutes; the capped diagnostic measured 67 seconds per
trial, and the authoritative run performs one warmup plus three measured trials.