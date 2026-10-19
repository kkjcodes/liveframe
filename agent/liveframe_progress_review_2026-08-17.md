# Progress Review — 2026-08-17 (pre-restart checkpoint)

Reviewer verification of the taew2_1 FP32 diagnostic arc and the blocked smoke gate.
Written immediately before a host restart to clear swap. All claims below were
independently re-verified against the repository this session.

## Where things stand (one paragraph)

The taew2_1 MLX port is now **parity-GO at FP32** (result v5, all tensors pass,
final float max error 0.00744087, cosine 0.99999962) after a bisection localized
the v4 failure to **native MLX `conv2d` accumulation/order drift** and fix
iteration 1 (explicit padded im2col+matmul for 3×3 convolutions, FP32 diagnostic
configuration only) repaired it within the two-iteration cap. The precision
policy is frozen ahead of any smoke/timing visibility. The Branch A smoke gate is
implemented and sealed, but the one recorded attempt is formally **invalid: the
host began with 19,412.44 MiB swap** (the runner now fails closed before even
importing MLX). Swap remained above 18 GiB during the final preflight checks, so
the restart is genuinely required. Nothing downstream (timing, campaign, metrics,
human study) has run. The sealed diagnostic arc ends at `113412a`; this handoff
is the only intended follow-up commit. Everything remains local and unpushed.

## Verified narrative since 2026-08-09

1. **Amendment v4 + v4.1 provenance correction** (`3e2750c`, `52b786e`): the
   pinned upstream weights and the v1 golden are float16, so a valid diagnostic
   required a new PyTorch FP32 reference (same weight values upcast, same
   anchors/fixture) rather than comparing FP32 MLX to an fp16 golden. Correct
   catch — the originally drafted diagnostic could not have separated the
   hypotheses.
2. **FP32-vs-FP32 diagnostic FAILED** (`3e0e7ff`, `outputs/taew21-port-parity.v4.json`,
   `passed: false`) → per the pre-registered rules this confirmed a genuine
   op-semantics/numerics bug and authorized Branch B only.
3. **Bisection** (`4684a8a`, `outputs/taew21-block11-call000-bisection.v1.json`):
   first 0.05 crossing at memory block 9; layers 6–8 clean; cause isolated to
   convolution accumulation/ordering — native MLX `conv2d` final-block max error
   0.0854714 vs 0.00143135 with explicit padded im2col+matmul and fixed
   kernel/channel ordering. Padding, upsampling, scheduling, and activation
   semantics were explicitly ruled out.
4. **Fix iteration 1** (`5d14f84` pre-registration → `d81f983` implementation →
   `cdeb6dc` execution binding → `13add0a` result): parity v5 **GO** — zero
   failing tensors under unchanged policy-v2 limits.
5. **Precision policy frozen** (`270f016`,
   `artifacts/taew21-precision-policy.v1.json`,
   status `frozen-before-fp32-im2col-smoke-or-timing-visibility`).
6. **Smoke gate sealed, attempt blocked** (`aaa0bc5` gate; `994e6c1` moves the
   zero-swap preflight ahead of MLX import/fixture load/decoder construction;
   `113412a` records `outputs/taew21-fp32-im2col-smoke-attempt.v1.json`:
   `blocked-by-nonzero-start-swap`, decision fields correctly state the candidate
   is *not falsified* and the attempt carries *no formal smoke authority*).
7. Validation re-run this session: 19 focused tests pass, Ruff clean.

## Assessment

- **Evidence-chain integrity: excellent.** Every repair was pre-registered before
  its candidate ran; the failed v4 result and the invalid smoke attempt were
  preserved rather than discarded; the fp16 NO-GO (v1–v3) remains the recorded
  verdict for the fp16 configuration.
- **The bisection finding has standalone value.** A reproducible, quantified
  MLX-`conv2d`-vs-PyTorch fp32 accumulation discrepancy (0.085 vs 0.0014 on the
  same weights) is exactly the kind of numerical-parity datapoint the paper's
  divergence result predicts. Consider filing a minimal reproduction upstream
  (ml-explore/mlx) and citing it in any v2 of the paper — it strengthens the
  "backends are not interchangeable" thesis with a named mechanism.
- **Open risk — im2col performance.** The im2col+matmul repair buys correctness
  but its speed is unmeasured; im2col materializes large intermediates and may be
   slower (or faster) than native conv on Metal. The frozen Branch A screen is one
   warmup plus three measured trials, zero start swap and zero swap delta for every
   trial, with decoder median at or below `13.773148263333496 s` (3x versus the
   `41.31944479000049 s` full-VAE comparator). If it fails, stop the FP32-im2col
   shipping track with no timing claim. FP16 remains deferred and requires a new
   prospective addendum and quality authority; do not route to it implicitly.
- **Open risk — unpushed commits.** The entire diagnostic arc through `113412a`
   and this handoff exist only on this machine. A push requires explicit user
   approval per the standing rule.

## Post-restart execution order

1. Run `sysctl vm.swapusage`. It must report exactly `used = 0.00M`; otherwise
   stop without importing MLX or starting `caffeinate`.
2. Confirm `git log -2 --oneline` shows the handoff commit followed immediately
   by diagnostic anchor `113412a`, and
   `outputs/taew21-fp32-im2col-smoke.v1.json` does not exist. Leave the unrelated
   `.vscode/settings.json` and release-package files untouched.
3. From `/Users/kumarjha/Code/liveframe`, run the sealed smoke unchanged:

   ```bash
   caffeinate -dimsu .venv/bin/python tools/smoke_taew21_fp32_im2col.py \
     --policy artifacts/taew21-precision-policy.v1.json \
     --parity-result outputs/taew21-port-parity.v5.json \
     --weights artifacts/taew21-mlx-weights.v4-fp32/taew2_1-mlx.safetensors \
     --weights-sha256 sha256:45d7a2585169e24903ea826a6c76605a2b2df54b2e252b2224551c6889038461 \
     --fixture outputs/golden/cfpp-1step-native-21block-h100/output_latents.bin \
     --fixture-sha256 sha256:966be57aebc4369ba9c197168dddc5c587cd9c891ba73ce613e5956d4e598512 \
     --output outputs/taew21-fp32-im2col-smoke.v1.json
   ```

4. The smoke must emit exactly 81 contiguous finite `uint8` THWC frames at
   480x832, bind the pixel digest, and observe zero start swap and zero delta.
   Commit that immutable smoke result as one evidence unit. Do not push without
   explicit user approval.
5. Only on smoke PASS, implement and freeze the screening runner before seeing
   its timing. Run one warmup and three measured decoder trials. Every run
   requires zero start swap and zero delta; median decoder time must be no more
   than `13.773148263333496 s`. Screening has routing authority only, not claim
   authority.
6. Only on screening PASS, run the original one-warmup/five-trial formal decoder
   and complete-wall benchmark. Keep its evidence separate from screening.
7. On formal timing PASS, proceed to the frozen 32-job campaign checks (seed-exclusion
   validation → paired captures → metric screening → pre-sized human study).
   None of these may start before their predecessor gate.
8. On smoke or screening FAIL, preserve the bounded NO-GO as its own evidence
   unit and stop downstream work. Any FP16 work needs a new prospective addendum.

## Non-negotiable constraints after restart

- Do not modify the denoiser, scheduler, KV state, latent fixture, native decoder
  default, or existing FP16 evidence.
- Use `TAEW21MLXDecoder(..., convolution_mode="fp32-im2col")` with the bound FP32
  weights above.
- One evidence unit per commit. Do not rewrite or overwrite prior evidence.
- Do not push without explicit user approval.
- Do not begin work outside the branch or gate that prospectively authorizes it.

## Standing state (unchanged)

- arXiv package: built and evidence-linked; submission status not verifiable from
  the repo this session.
- Refresh mechanism: authoritative NO-GO; in-loop work requires the
  training/adaptation track (Causal-rCM lineage).
- DS denominators (M5 57.127 s / H100 5.9609 s) remain frozen and untouched.
- Unrelated `.vscode/settings.json` modification remains uncommitted by design.
