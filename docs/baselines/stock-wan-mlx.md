# Stock Wan2.1 MLX bring-up

Date: 2026-07-20

The pinned `mlx-video` runtime was installed and the exact Wan2.1-T2V-1.3B snapshot was downloaded and converted to MLX bfloat16. Large source and converted weights remain outside Git. Their input and output hashes are recorded in the local-conversion manifest.

## Smoke result

A deliberately minimal native-resolution request exercised tokenizer loading, UMT5 encoding, one dense DiT denoising step, VAE decoding, and H.264 output:

- Resolution: 832×480.
- Requested frames: 1.
- Encoded frames observed by `ffprobe`: 4 at 16 FPS.
- T5 encoding: 10.3 seconds.
- Denoising: 1.9 seconds.
- VAE decode: 4.4 seconds.
- End to end: 18.2 seconds.
- CFG disabled with guidance 1.0.
- Output duration: 0.25 seconds.

This row is **not performance-claim eligible**. It uses stock Wan, one step only as a pipeline smoke test, cold text encoding, and a one-frame request rather than the native 81-frame horizon.

## Findings

1. The stock MLX pipeline loads and produces finite media on this M5 Max.
2. The generator hardcodes `google/umt5-xxl` for tokenizer lookup. The resolved cache revision was independently confirmed as the pinned `66cb9e7e85526fe440a945569e42c72fb6cbc0ad`; proof runs must force offline resolution or use the exact local tokenizer directory.
3. A one-latent-frame request encoded four pixel frames, not the asymmetric one-frame first emission suggested by $4(L-1)+1$. This may be a `mlx-video` VAE padding/trim behavior. Streaming accounting must measure the actual first emission and parity-check it against upstream before implementation treats four frames as valid.
4. CF++ conversion remains separate from stock Wan conversion. Unknown CF++ tensor names fail closed in LiveFrame's metadata conversion planner.
