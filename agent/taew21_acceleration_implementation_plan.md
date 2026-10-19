# LiveFrame taew2_1 Acceleration Implementation Plan

**Status:** prospective plan; no taew2_1 result has been observed in LiveFrame.
**Primary objective:** reduce complete M5 generation wall time by replacing the
out-of-loop causal VAE decoder while measuring both machine metrics and blinded
human detectability.
**Controlling baseline:** `artifacts/m5-cfpp-1step-cache4-baseline.v1.json`.

## 1. Decision and scope

This plan merges perceptual-budget optimization into the existing acceleration
sequence. It does not replace the sequence and does not reopen the rejected
training-free refresh mechanism.

The immediate track is an output-changing decoder substitution:

1. Hold the generated latent trajectory fixed.
2. Decode the same latents with the pinned full Wan VAE and taew2_1.
3. Measure decoder-only and integrated complete-wall speed.
4. Apply the frozen metric battery as a screening gate.
5. Run one pre-registered blinded study on the same paired videos.

The current M5 baseline is 57.127 seconds complete wall, of which 41.319 seconds
(72.33%) is causal VAE decode. Decoder elimination would leave 15.808 seconds, so
the zero-overhead end-to-end ceiling is 3.614x. This is only a feasibility bound;
the taew2_1 gain must be measured and must include conversion, synchronization,
materialization, and encoding.

## 2. Approximation boundary

### Track A: out-of-loop approximations

These may use a noticeability budget because their error does not alter the
autoregressive latent state:

- full VAE to taew2_1 decoder substitution;
- decoder precision and compilation;
- output resolution tiers;
- post-hoc interpolation and presentation FPS.

Each tier still requires complete-video temporal checks. "Out of loop" means the
error does not feed the generator, not that temporal artifacts are impossible.

### Track B: in-loop approximations

KV reuse, refresh skipping, denoiser layer skipping, and approximate recurrent
state updates are not authorized by a training-free perceptual budget. The frozen
refresh result and CUDA/MLX divergence show that small local perturbations can
become trajectory-defining over an autoregressive horizon.

Further in-loop work therefore requires a training or adaptation mechanism that
exposes the model to the approximation, followed by horizon-indexed drift and
quality gates. Causal-rCM-style robustness work belongs on this separate track.
It does not block Track A.

## 3. Falsifiable hypothesis and first check

**Hypothesis:** a taew2_1 MLX decoder can consume the exact frozen Wan latent
contract and emit the exact requested native-frame count and resolution, with a
decoder wall-time reduction that survives complete-wall integration and a
bounded human detectability rate.

**First disconfirming check:** decode one frozen 21-latent fixture with upstream
PyTorch taew2_1 and freeze its output and selected intermediate activations as the
port-parity reference. The MLX port must then remain within a prospectively frozen
PyTorch-to-MLX error band and decode exactly 81 RGB frames at 480x832 with finite
values, stable chunk accounting, and no swap. Failure in weight mapping, latent
scaling, temporal expansion, output range, frame count, port parity, or memory
stops the port before campaign infrastructure or participant spending. This gate
distinguishes a taew2_1 quality limitation from an incorrect MLX implementation.

## 4. Implementation slices

### A0. Freeze provenance and protocol before quality visibility

Create `artifacts/taew21-protocol.v1.json` containing:

- taew2_1 repository revision, license, weight source, and SHA-256;
- converter source digest and converted-weight digest;
- full-VAE digest and controlling M5 baseline digest;
- PyTorch version, taew2_1 upstream implementation digest, parity-policy digest,
  parity-fixture digest, selected activation anchors, and frozen
  MLX-versus-PyTorch error limits;
- latent schema, scaling convention, tensor layout, frame expansion, output
  range, color conversion, and expected resolution;
- 16 prompts, two frozen horizons, seeds, and content strata;
- warmup/trial policy, allocator limit, swap ceiling, thermal capture, and timing
  boundaries;
- a minimum worthwhile complete-wall speedup of 1.5x and its prospective
   confidence rule;
- frozen metric provenance and screening rules;
- human-study design, exclusion rules, sample-size calculation, primary test,
  epsilon, and stopping rule.

Protocol validation must reject unknown keys, missing digests, duplicate jobs,
post-freeze mutation, and output paths outside the campaign root. Generation
seeds must be drawn prospectively and be disjoint from every calibration,
development, opened-holdout, refresh, and publication seed previously observed
in this project. Materialize the union of those burned seeds as an immutable
`artifacts/taew21-seed-exclusions.v1.json`; protocol validation must reject any
intersection. The 16 prompts must be selected before candidate videos or
candidate metrics are opened. Reuse existing frozen prompts where they provide
the required motion/content coverage; add prompts only prospectively.

Before capturing the PyTorch golden output, freeze
`artifacts/taew21-port-parity-policy.v1.json` with the source revision, weight and
latent digests, selected activation anchors, comparison metrics, and error limits.
Establish those limits from declared dtype/primitive tolerances and synthetic
per-operation calibration; the golden values may not tune them. Capture and hash
the PyTorch golden under that policy, then freeze `taew21-protocol.v1.json` so it
binds both policy and golden digests. The complete A0 protocol must be frozen
before any taew2_1 MLX output, quality metric, or timing is visible.

### A1. Introduce a narrow decoder contract

Add `src/liveframe/decoders/base.py` with a decoder protocol whose input is one
declared latent tensor and whose output is a sequence of native RGB chunks with
explicit frame counts. Implement:

- `FullWanMLXDecoder`: an adapter around the current
  `liveframe.mlx_vae_decode` path;
- `TAEW21MLXDecoder`: the converted tiny decoder;
- shared validation for layout, dtype, finite values, frame count, dimensions,
  range, and model provenance.

Do not change the denoiser, scheduler, KV cache, or generated latents. Do not add
a generic model-adapter framework. The adapter exists only to make the paired
decoder experiment impossible to confound.

Nearest tests:

- `tests/test_decoder_contract.py`: synthetic shape, frame-count, nonfinite,
  range, and provenance failures;
- `tests/test_taew21_converter.py`: key mapping and round-trip manifest checks;
- extend `tests/test_mlx_vae_profile.py` for timing-accounting compatibility.

### A2. Port and verify taew2_1 on MLX

Add:

- `src/liveframe/decoders/taew21_mlx.py`;
- `tools/convert_taew21_to_mlx.py`;
- `tools/capture_taew21_pytorch_reference.py`;
- `tools/check_taew21_port_parity.py`;
- `tools/smoke_taew21_mlx.py`.

The converter must fail closed on missing, unexpected, or shape-mismatched
weights and write a manifest binding every source tensor to its converted tensor.
The PyTorch capture must consume the same immutable latent payload as MLX and
write float decoder output, final RGB, selected intermediate activations, runtime
provenance, and digests. The parity checker must compare the declared anchors and
final output under the frozen limits before the smoke result is eligible. It must
report every failed tensor rather than collapsing failures into one aggregate.

The smoke command runs the A1 first check only after port parity passes and
records per-stage wall time, peak MLX memory, allocator state, swap delta, output
shape, and pixel digest.

No quality or speed claim is made from the smoke fixture.

### A3. Build paired decoder captures

Add `tools/run_taew21_campaign.py`. For every frozen job it must:

1. Load one immutable latent payload once and verify its digest.
2. Decode it with both adapters in randomized interleaved order.
3. Write lossless RGB artifacts and viewable videos with source-latent and
   decoder provenance.
4. Verify exact pair equality for frame count, dimensions, playback FPS, and
   encoding policy.
5. Record decoder-only wall, materialization, encoding, peak memory, swap,
   thermals, and complete integrated wall separately.
6. Refuse partial, duplicate, or overwritten result artifacts.

The paired campaign must not regenerate denoiser outputs. Full and tiny videos
must differ only in decoder choice.

### A4. Reuse the metric stack as a screening gate

Reuse `PerceptualMetricRunner` from
`src/liveframe/quality_metric_runner.py` to compute:

- intrinsic semantic, motion, temporal, and flicker metrics for each output;
- pairwise LPIPS, CLIP distance, and temporal-CLIP distance;
- horizon- and content-stratified summaries;
- frame-indexed error curves to expose local temporal failures hidden by means.

Create `tools/evaluate_taew21_campaign.py` and an immutable
`artifacts/taew21-metric-verdict.v1.json`. Existing frozen floors are screening
instruments, not validated human-noticeability thresholds. A metric pass permits
the human study; it does not establish perceptual equivalence. A severe metric
failure stops the release tier but the paired result remains publishable.

### A5. Run one blinded discrimination study

Use the same 16-prompt by two-horizon pairs. The primary task should measure
discrimination rather than preference:

- show a full-decoder reference followed by randomized A/B clips containing one
  full and one tiny decode of the same latents;
- ask which A/B clip matches the reference;
- randomize left/right order and decoder labels per judgment;
- use a separate secondary question for quality preference;
- balance jobs across participants and include pre-registered attention and
  repeat-consistency checks.

Define chance as 50% and define the prospective noticeability budget as no more
than five percentage points above chance. The primary acceptance rule is that a
pre-registered one-sided 95% upper confidence bound for discrimination accuracy
is at most 55%, with participant and video clustering handled in the analysis.
Determine participant count by simulation before launch; do not choose it after
observing responses. Report raw accuracy, uncertainty, exclusions, completion
time, device/display restrictions, and per-stratum estimates.

The unclustered best-case arithmetic already requires approximately 271
independent judgments at true chance performance:

$$
1.645\sqrt{0.25/N} \leq 0.05.
$$

Participant and video clustering increase that requirement. The initial planning
envelope is therefore 400--600 usable judgments in a balanced incomplete-block
design, approximately 30 participants completing 16--20 trials each, with the
final count set by the pre-registered simulation. Budget recruitment, fair pay,
platform fees, exclusions, and replacement participants before launch; expect a
few hundred US dollars rather than treating the study as a negligible expense.
The simulation must also show the sensitivity of the rule: if true accuracy is
around 53%, this design may correctly fail to establish the 55% upper bound.

Before recruitment, complete the applicable ethics/consent review, publish the
anonymized protocol, and confirm that compensation meets the platform's fair-pay
guidance. Store no direct participant identifiers in the repository.

Metric-versus-human correlation is a pre-registered secondary analysis. With 32
video pairs it is calibration evidence for this mechanism and corpus, not a
universal perceptual model. Report bootstrap uncertainty and leave-one-prompt-out
stability rather than selecting the best-correlating metric after visibility.

## 5. Decision gates

The taew2_1 tier ships only if all applicable gates pass:

1. **Contract:** exact frame count/dimensions, finite output, frozen latent
   identity, valid provenance, and no swap contamination.
2. **Performance:** at least 1.5x paired complete-wall speedup under the
   confidence rule frozen in A0. Report both decoder-only and end-to-end speedup;
   do not extrapolate from decoder timing. On the controlling component times,
   1.5x end-to-end requires approximately 1.855x effective decoder acceleration
   before any additional integration overhead. A0 must size the interleaved
   timing repeats so the chosen confidence rule can resolve this floor.
3. **Metrics:** all pre-registered authoritative screening cells pass. Diagnostic
   metrics remain diagnostic.
4. **Human:** the primary discrimination rule passes without changing epsilon,
   exclusions, sample size, or strata after response visibility.

Outcomes:

- **GO:** ship `taew2_1` as an explicitly output-changing fast decoder tier.
- **METRIC GO / HUMAN NO-GO:** publish the calibration miss; do not call the tier
  perceptually equivalent.
- **QUALITY GO / SPEED NO-GO:** retain the port as a diagnostic; do not ship it as
  acceleration.
- **EARLY CONTRACT NO-GO:** stop before campaign or recruitment.

The compiled full VAE remains diagnostic-only under the old evidence because no
threshold was frozen before its output was observed. It may enter a later new
prospective protocol, but it cannot borrow authority from this taew2_1 gate.

### Execution record: 9 August 2026

A0 through the port-parity portion of A2 executed under protocol digest
`sha256:ad0c4b7eed88bff8f6cf66cc33eb2aa1eadea61a97ddebb68eead782c59626f7`.
The PyTorch golden contains 24 selected causal activation probes plus float and
RGB final outputs for the controlling 21-latent/81-frame fixture.

The first MLX check failed because a non-contiguous NumPy OIHW-to-OHWI transpose
was serialized with incorrect element order. Preserve that NO-GO result as
`outputs/taew21-port-parity.v1.json`. Amendment v2 froze a converter-only repair
before repaired-candidate metrics were opened; all 64 repaired tensors were then
verified byte-exact against their intended source transforms.

The repaired result, `outputs/taew21-port-parity.v2.json`, remains an **EARLY
CONTRACT NO-GO** under the unchanged policy. Final float output passes with max
absolute error `0.0302734375`, mean absolute error `0.0005723141`, and cosine
similarity `0.9999987744`. Six deep-memory activation probes exceed the frozen
`0.05` maximum-error limit, and RGB output differs by up to 8 integer levels.
Do not reinterpret the passing final-float metrics as overall parity, retune the
limits after visibility, or proceed to smoke, timing, quality campaign, or human
recruitment under this protocol.

Before another candidate was executed, policy v2 prospectively separated the
unchanged float limits from uint8 RGB units. Its RGB limits are the mechanical
round-to-nearest consequences of the float limits: maximum absolute error 13,
mean absolute error 2.275, and minimum cosine similarity 0.999. Amendment v3
then froze FP32 accumulation in memory blocks 11 and 17, with unchanged operation
ordering and a cast back to the incoming dtype at each selected block boundary.

The resulting immutable report, `outputs/taew21-port-parity.v3.json`, is still an
**EARLY CONTRACT NO-GO**. FP32 accumulation repaired `decoder.17.call.020`, but
all three selected block-11 probes and block-17 calls 0 and 41 still exceed the
unchanged 0.05 maximum-error limit. Their mean-error and cosine limits pass.
Final float output passes with max absolute error `0.03076171875`, mean absolute
error `0.0005677876`, and cosine similarity `0.9999987938`; RGB also passes its
prospective v2 limits with max error 8, mean error `0.1447771372`, and cosine
similarity `0.9999959017`. Stop before smoke, timing, the 32-job campaign,
metric screening, or human recruitment. Any further numerical repair requires a
new amendment and candidate version; v1 through v3 remain immutable evidence.

The authorized next step is the pre-registered full-FP32 diagnostic specified in
`agent/taew21_fp32_diagnostic_execution.md`. Both the converted source weights
and the existing PyTorch golden are float16, so a valid diagnostic must upcast
the same pinned values and fixture on both sides and compare PyTorch FP32 with
MLX FP32. Amendment v4 was preserved, and the provenance correction requires a
v4.1 amendment plus a new diagnostic-only FP32 reference before implementation
or candidate execution. This separates port-semantics failure from
framework-specific fp16 arithmetic before any further repair attempt.

The same-precision full-FP32 diagnostic was executed under amendments v4 and
v4.1 and the exact execution binding. `outputs/taew21-port-parity.v4.json` is a
**Branch B NO-GO**: block-11 calls 000, 010, and 020 have maximum errors
`0.0785387`, `0.0547631`, and `0.0578114`, and block-17 call 000 has maximum
error `0.0731414`. All corresponding means and cosines pass. Final float and RGB
outputs also pass, with maximum errors `0.0200548` and 5 respectively. Because
any FP32 probe failure pre-authorized only Branch B, smoke, timing, campaign,
metrics, and human recruitment remain blocked. The next action is block-11
call-000 op-by-op bisection; at most two fix iterations are authorized.

The bisection localized the first threshold crossing to memory block 9 and then
isolated native MLX 3x3 convolution accumulation order using a shared identical
input. Explicit padded im2col plus matmul reduced the isolated block's output max
error from `0.0854714` to `0.00143135`. Amendment v5 froze that repair for the
full-FP32 diagnostic configuration. `outputs/taew21-port-parity.v5.json` is a
**semantic parity GO**: all 24 activation probes and both final outputs pass.
Final float max error is `0.00744087`; RGB max error is 2. Branch A is now
authorized, beginning with a precision-policy addendum frozen before any smoke
or timing visibility. Diagnostic runtime has no performance authority.

## 6. Execution order

### First 48 hours

1. Verify taew2_1 license, revision, weight availability, architecture, latent
   scaling, temporal behavior, and output range against upstream source.
2. Build the burned-seed exclusion artifact and freeze the port-parity policy,
   speed floor, and fail-closed result schemas.
3. Capture the upstream PyTorch golden fixture and selected activation anchors,
   then freeze the complete protocol binding the policy and golden digests.
4. Implement the converter and synthetic decoder-contract tests.
5. Run MLX port parity, then the one-fixture M5 smoke check only if parity passes.

### Week 1

1. Make full and tiny adapters consume one immutable latent artifact.
2. Validate PyTorch-to-MLX parity and exact 81-frame output, then capture
   decoder-only timing/memory.
3. Integrate the tiny adapter into the existing complete-wall CF++ benchmark.
4. Validate that fresh campaign seeds are disjoint from the immutable exclusion
   set, then freeze campaign jobs, metric rules, and study analysis.

### Week 2

1. Run the 32 paired jobs with interleaved decoder order.
2. Evaluate frozen metrics and inspect only the pre-declared diagnostics.
3. If the screening gate passes, render the blinded study bundle and validate its
   randomization manifest without opening assignment labels.

### Week 3

1. Launch the pre-sized study.
2. Freeze raw responses, apply pre-registered exclusions, and publish the verdict.
3. Integrate the tier only on a complete GO and update `liveframe benchmark` with
   decoder choice, generated FPS, decoder wall, complete wall, and quality status.

## 7. Definition of done

This slice is complete when another supported M5 machine can verify model and
latent digests, reproduce full/tiny paired outputs, recompute metrics, audit the
blinded assignment and analysis without participant identifiers, and reproduce
the paired complete-wall result. The public claim must name the exact decoder,
resolution, horizon, corpus, hardware, metric gate, human epsilon, and whether
the reported FPS is native generated or presented.

After this gate, the next out-of-loop candidates are decoder quantization,
resolution, and interpolation, each under a new prospective protocol. In-loop
acceleration resumes only on the training/adaptation track.