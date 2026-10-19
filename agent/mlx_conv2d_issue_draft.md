# Draft: ml-explore/mlx issue — fp32 `conv2d` silently runs at TF32 precision on M5 (NAX)

**Status:** draft, ready to post to `github.com/ml-explore/mlx/issues` (user action).
**Cross-link when posting:** ml-explore/mlx-swift#492 (same mechanism reported for conv3d).

---

**Title:** `[Metal] fp32 conv2d on M5 (NAX) silently accumulates at TF32: single 3x3 conv is ~700x further from an fp64 reference unless MLX_ENABLE_TF32=0`

## Describe the bug

On an M5-class GPU, `mx.conv2d` with float32 inputs and weights produces results far outside
float32 accumulation error, and the deviation disappears with `MLX_ENABLE_TF32=0`. A single
3×3 convolution (256→256 channels, 60×104 spatial, He-scaled random weights, output scale ≈ 1)
measured against a float64 CPU ground truth:

| path | max abs err (default) | max abs err (`MLX_ENABLE_TF32=0`) |
| --- | --- | --- |
| PyTorch CPU fp32 (control) | 1.4e-05 | 1.4e-05 |
| `mx.conv2d` fp32 | **9.7e-02** | 1.4e-04 |
| explicit im2col + `@` (same math) | 5.3e-03 | 1.5e-05 |

Two observations beyond mlx-swift#492 (which reports this for conv3d ≥ 0.32.2):

1. `conv2d` is affected as well, and roughly ~20× harder than a plain fp32 matmul of the same
   arithmetic — consistent with the Winograd path paying the TF32 rounding in two
   transform-domain matmuls rather than one.
2. The error compounds badly through real networks: in a 23-layer fp32 video-VAE decoder we
   measured 0.085 max final-output error via native `conv2d` vs 0.0014 via an explicit
   im2col+matmul port of the same weights (which is how we first noticed, in August, before
   identifying TF32 as the mechanism). A 12-layer synthetic conv+ReLU chain in the repro below
   shows the same growth.

## To Reproduce

Self-contained script (mlx + torch + numpy only), run twice:

```bash
python repro_mlx_conv2d_drift.py                     # default: drift
MLX_ENABLE_TF32=0 python repro_mlx_conv2d_drift.py   # fp32-faithful
```

[attach `tools/repro_mlx_conv2d_drift.py` from the LiveFrame repository]

## Expected behavior

Either fp32 means fp32 by default (TF32 as explicit opt-in), or the default is documented
loudly enough that numerical-parity work can find it: we spent a multi-day bisection isolating
this to conv accumulation order before connecting it to TF32, and shipped an im2col fallback as
a workaround. If the default stays, a per-op or per-stream API (like
`torch.backends.cuda.matmul.allow_tf32`) plus a note in the `conv2d` docstring would prevent
others repeating that.

## Desktop

- Apple M5 Max (NAX), macOS 25.6.0
- mlx 0.32.0 (python), torch 2.13.0 (CPU reference)
- Also reproduced at decoder level on mlx pinned in 2026-08 (see numbers above)

---

## Posting notes (not part of the issue)

- Verify against the latest mlx before posting; if a fix landed, convert into a comment on
  mlx-swift#492 confirming conv2d-on-python impact instead of a new issue.
- Project consequences, recorded here for our own narrative: the TAEW2.1 "conv2d accumulation
  drift" standalone finding now has an identified mechanism (TF32-on-NAX default). The
  fp32-im2col decoder path remains the shipping configuration (its parity GO and the 2.61×
  formal benchmark are unaffected — im2col under default TF32 still passed the frozen policy
  limits, and is ~11.8× faster than the full VAE). Worth testing later: native conv2d under
  `MLX_ENABLE_TF32=0` might pass parity too and could be faster than im2col; that would be a
  new candidate under a new prospective addendum, not a retroactive change.
