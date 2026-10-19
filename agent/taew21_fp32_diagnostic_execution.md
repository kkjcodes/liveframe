# TAEW2.1 Full-FP32 Parity Diagnostic — Execution Brief

**Status:** authorized next step after the v3 FP32-accumulation NO-GO; nothing below this
gate (smoke, timing, 32-job campaign, metrics, human study) is unblocked by this brief.
**Purpose:** separate the two surviving hypotheses for the five failing activation probes:
(H1) the MLX port has an op-semantics bug, versus (H2) the port is correct and the residual
error comes from framework-specific fp16 arithmetic. The pinned source weights and existing
PyTorch golden are both float16; merely upcasting MLX while comparing against that fp16 golden
cannot distinguish these hypotheses. The valid diagnostic upcasts the same pinned weight
values and latent fixture to float32 in both PyTorch and MLX, then compares them under the
unchanged policy-v2 limits.

The executing agent updates this document with results as gates complete.

## Step 0 — Pre-register amendment v4 BEFORE any fp32 candidate execution

Create `artifacts/taew21-protocol-amendment.v4.json` (extend the existing builder-tool
pattern; canonical self-digest; fail-closed validation). Required content:

- `schema`: `liveframe.taew21-protocol-amendment.v4`
- `status`: `frozen-before-full-fp32-diagnostic-execution`
- `purpose`: full-FP32 **diagnostic** candidate — explicitly NOT a shipping configuration;
  its result cannot authorize smoke/timing/campaign/human on its own; the fp16 v3 NO-GO
  remains the recorded verdict for the fp16 configuration.
- `allowed_implementation_change`:
  - convert weights at `float32` into a NEW directory `artifacts/taew21-mlx-weights.v4-fp32/`
    (never overwrite v1/v2), manifest binding every tensor and recording target dtype;
  - decoder compute dtype float32 end-to-end (inputs, weights, all blocks);
  - `op_order`: unchanged; anchors unchanged; checker unchanged; policy v2 limits unchanged.
- Digest bindings (computed 2026-08-09 from the current tree — recompute and confirm before
  freezing; if any differs, STOP and reconcile):
  - `base_protocol_digest`: `sha256:ad0c4b7eed88bff8f6cf66cc33eb2aa1eadea61a97ddebb68eead782c59626f7`
  - `policy_v2_sha256`: `sha256:92219705c499983758e06fe28b3a8a659c8ab619ddc5b1b4f400e26b0edc197e`
  - `previous_amendment_sha256` (v3): `sha256:6a17338795d9b2f43ec809726b3feb8f92a5785d061e24dd3f2dbb77046c5c72`
  - `previous_result_sha256` (parity v3): `sha256:a558c27208c3a78d28a7166ef24499d881c3fcb7ea61505122ff4b5a8295791c`
  - `pre_diagnostic_decoder_sha256`: `sha256:a213f6f7203d66184d853fc35a28db2b6859bf37e82956e4e2dc16b562032052`
  - `pre_diagnostic_checker_sha256`: `sha256:39ef19505b1434835f95b96f30df4529ac036b087b585c1ba968812641435814`
  - `pre_diagnostic_converter_sha256`: `sha256:fa7880e4442f7890f49f0cb626b7c1155af8522c2ba67a26c96115266df191e9`
- Pre-registered decision rules (immutable once frozen):
  - **ALL probes + final outputs pass at fp32** → the port is semantically correct. The fp16
    parity NO-GO stands for the fp16 configuration. Proceed to the Branch A precision-policy
    fork below. No probe limit may be relaxed for fp16 afterward.
  - **ANY probe fails at fp32** → op-semantics bug confirmed. Authorize Branch B bisection
    only. No threshold changes.

Commit the amendment alone ("Pre-register TAEW2.1 full-FP32 parity diagnostic") BEFORE
running any fp32 decode.

### Step 0.1 — provenance correction before implementation

Post-freeze review established that the pinned safetensors decoder tensors are all float16
and the v1 golden manifest records `execution_dtype: float16`. Preserve amendment v4, but
freeze and commit amendment v4.1 before implementation. It must:

- record this provenance correction and bind amendment v4;
- authorize a new diagnostic-only PyTorch FP32 reference at
  `outputs/golden/taew21-pytorch-fp32-reference.v1/`, using the same pinned source weight
  values, implementation, activation anchors, and latent fixture, all upcast to float32;
- require a post-implementation execution binding that binds both the new FP32 reference
  and FP32 MLX candidate before either candidate metric is evaluated;
- preserve the existing FP16 golden and v1-v3 results unchanged;
- apply the Step 3 decision rules only to the same-precision FP32-vs-FP32 comparison.

## Step 1 — Implement

1. `tools/convert_taew21_to_mlx.py`: add an explicit `--target-dtype {float16,float32}`
   (fail-closed on anything else); manifest must record the dtype; produce
   `taew21-mlx-weights.v4-fp32/` with digests.
2. `src/liveframe/decoders/taew21_mlx.py`: support full-fp32 execution (load fp32 weights,
   fp32 inputs; the existing `FP32_MEMORY_LAYERS` accumulation path is a no-op when the
   whole graph is fp32 — do not remove it, it belongs to the frozen v3 configuration).
3. Keep `tools/check_taew21_port_parity.py` byte-identical to
  `pre_diagnostic_checker_sha256`; it is cryptographically locked to the v2 fp16 weights
  by amendment v2 and therefore cannot directly authorize the new fp32 artifact. Add a
  separate diagnostic runner that imports its unchanged `_metrics`, `_verdict`, and call-
  count logic, validates amendment v4 plus the fp32 conversion manifest, and emits the
  same tensor inventory and per-domain policy-v2 verdicts. Do not duplicate or alter the
  comparison math.
4. After implementation and tests, but still before any fp32 candidate execution, freeze
  and commit `artifacts/taew21-fp32-execution-binding.v1.json`. It must bind amendment v4,
  the exact diagnostic-runner, decoder, converter, fp32 weights, conversion manifest,
  golden, policy v2, and immutable v4 output path. Recompute every digest and stop on a
  mismatch. This binding may authorize execution only; it may not change limits, anchors,
  comparison math, or the Step 3 decision rules.
5. Tests: converter fp32 round-trip + dtype plumbing added to the focused suite; full
   focused suite, Ruff, and `git diff --check` green before execution.
6. Extend the reference-capture tool with an explicit fail-closed execution dtype and
  capture the FP32 diagnostic reference before any MLX FP32 execution. This is a new
  diagnostic artifact, never a replacement for the frozen FP16 golden.

## Step 2 — Execute the diagnostic

- Run the bound diagnostic runner, using the unchanged checker comparison engine, with the
  fp32 weights/config against the same pinned H100 latent fixture. Output:
  `outputs/taew21-port-parity.v4.json` (immutable; refuse overwrite; embed policy digest,
  execution-binding digest, and weight-manifest digest).
- No timing claims from this run; caffeinate optional; swap state does not gate a
  correctness-only run but record it anyway.
- Record the verdict in `taew21_acceleration_implementation_plan.md` and commit the
  evidence separately ("Record TAEW2.1 full-FP32 parity diagnostic result").

## Step 3 — Branch on the pre-registered rules

### Branch A — fp32 passes everything (port correct; precision is policy)

1. Freeze a short precision-policy addendum BEFORE any timing visibility, declaring the
   candidate shipping configurations in priority order:
   a. **fp32 tiny decoder** — if its decoder-only wall on the smoke fixture is still a
      large multiple faster than the 40.3 s full-VAE row, ship fp32 and defer fp16
      entirely (simplest honest path; the tiny decoder is ~100× smaller than the full VAE,
      so fp32 is likely still fast).
   b. **fp16 as a declared Track A precision variant** — only via a new prospective
      addendum; its quality authority is the campaign metric + human gates, not probe
      parity (the plan's Track A already lists "decoder precision" as its own
      output-changing approximation).
2. Then, and only then, resume the frozen A2→A5 sequence for the chosen configuration:
   smoke (exactly 81 frames, 480×832, finite, no swap) → decoder-only + integrated
   complete-wall timing (zero-swap protocol, one warmup + three trials) → 32-job paired
   campaign → metric screening → pre-sized human study.

### Branch B — any probe still fails at fp32 (semantics bug)

1. Bisect block 11, call 000 only (worst probe, 0.187): instrument BOTH implementations to
   dump intra-block intermediates on that call — conv input/output, nonlinearity output,
   memory-state before/after, upsample output — and compare op-by-op to find the FIRST
   divergent op.
2. Ranked suspects: (1) memory-state initialization / first-frame path (call 000 is 3×
   worse than later calls), (2) conv padding semantics, (3) upsample/resample mode
   (nearest vs bilinear, corner alignment), (4) activation-function approximation.
3. Fix the identified op; rerun the UNCHANGED checker; publish `taew21-port-parity.v5.json`.
   At most two fix iterations (mirror the P5 policy-iteration cap); if parity still fails,
   record the port as failed, preserve all evidence, and stop the taew track with a
   written finding.

## Standing constraints

- Do not touch the denoiser, scheduler, KV cache, or latent generation.
- Do not run smoke/timing/campaign/human outside the branch that authorizes them.
- One evidence unit per commit; no push without explicit user approval.
- Leave `.vscode/settings.json` and the release archives untouched.
- Probe counts for reference: v2 = six probes over limit (block-11 ×3, block-17 ×3);
  v3 = five (block-11 ×3, block-17 calls 0 and 41). The plan's execution record already
  states these correctly; keep any new summaries consistent with it.

## Results log (executing agent appends below)

- Amendment v4 was frozen in `3e2750c`. Pre-execution review then established that
  both the pinned source weights and existing PyTorch golden are float16, so amendment
  v4.1 was frozen in `52b786e` to require same-precision PyTorch-FP32 versus MLX-FP32.
- The exact FP32 diagnostic path was committed in `eda6759`, the new diagnostic-only
  PyTorch FP32 reference manifest in `6bb9880`, and the final execution binding in
  `f1f748d`. The 1.5 GB reference tensor payload remains local and is bound by SHA-256
  `2fa57e329437c23c0416e8094311cf5c108349aa0226fd49bd45ffe63733755c`.
- `outputs/taew21-port-parity.v4.json` is **Branch B**. Four probes exceed the unchanged
  0.05 maximum-error limit: block-11 calls 000 (`0.0785387`), 010 (`0.0547631`), and
  020 (`0.0578114`), plus block-17 call 000 (`0.0731414`). All means and cosines pass.
  Final float passes (`max=0.0200548`, `mean=0.000498922`, `cosine=0.999999165`) and
  RGB passes (`max=5`, `mean=0.127241`, `cosine=0.999996461`). Block-17 calls 020 and
  041, which failed in earlier configurations, pass in same-precision FP32.
- Branch A remains blocked. Authorized next action: block-11 call-000 op-by-op bisection
  only, beginning with memory initialization, then padding, upsample semantics, and
  activation approximation. At most two fix iterations remain authorized.
- Block-11 call-000 bisection v1 found that its input already differs by max `0.0698964`;
  zero memory is exact, so memory initialization is not the first defect. Bisection v2
  localized the first 0.05 crossing to memory block 9: layers 6, 7, and 8 remain at
  `0.0052883`, `0.0021322`, and `0.0095122`, while layer 9 reaches `0.0647915`.
- Shared-input layer-9 isolation reproduced the error with identical FP32 input and exact
  weights, identifying convolution accumulation/order rather than scheduling, padding
  shape, upsampling, or activation semantics. Native MLX `conv2d` produced final block
  max error `0.0854714`; explicit padded im2col+matmul with fixed kernel/channel ordering
  reduced it to `0.00143135`. This authorizes fix iteration 1: explicit im2col+matmul for
  3x3 convolutions in the FP32 diagnostic configuration only, frozen before v5 execution.
- Fix iteration 1 was frozen by amendment v5 and its exact execution binding.
  `outputs/taew21-port-parity.v5.json` is a **semantic parity GO**: all 24 activation
  probes plus float and RGB final outputs pass the unchanged policy. Block-11 maxima are
  `0.00618576`, `0.0100189`, and `0.00971261`; block-17 call 000 is `0.0153069`.
  Final float passes with max `0.00744087`, mean `0.000551782`, and cosine `0.999999620`;
  RGB passes with max 2, mean `0.140753`, and cosine `0.999996581`.
- The port is semantically correct in the full-FP32 explicit-im2col configuration. Enter
  Branch A: freeze the precision-policy addendum before smoke or timing visibility. No
  performance claim is authorized by the diagnostic runs.
