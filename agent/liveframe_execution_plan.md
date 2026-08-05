# LiveFrame Accelerated Execution Plan

**Status:** active execution plan; M1 sealed holdout completed with authoritative FAIL  
**Date:** 2026-08-01  
**Supersedes for scheduling:** the phase ordering in `liveframe_plan.md`; the research
details in `liveframe_research_track.md` remain applicable.

## 1. End goal and success boundary

LiveFrame will deliver a portable, reproducible diffusion-video engine with honest
performance and quality measurement. It has two independently useful output modes:

- **Streaming mode:** native generated frames approaching or reaching 16 FPS, with no
  interpolation counted as generated FPS.
- **Fast-clip mode:** 5-15 second clips through the same engine. Sub-5-second generation is
  a real target for a declared 5-second configuration on M5 Max. H100 results remain useful
  baselines, but a 5-second/81-frame H100 run near 4.2 seconds does not by itself satisfy a
  novel fast-clip milestone. A 15-second clip under 5 seconds on H100 is a separate stretch.

The engine and measurement system are sufficient for project success without a novel
acceleration algorithm. The core result is publishable when it provides:

1. Backend-neutral scheduling, stochastic inputs, cache semantics, and artifact schemas.
2. CUDA and MLX implementations whose outputs stay within frozen numerical, intrinsic
   quality, and same-seed identity bands. Portable correctness means **within the frozen
   parity band**, not bit-identical output.
3. Reproducible environment, model, decoder, metric-state, and output provenance.
4. A measured budget table for CUDA and Apple Silicon covering throughput, latency, memory,
   thermal behavior, quality, and long-horizon drift.
5. A usable `stream()` and `clip()` interface with honest generated-versus-presented frame
   accounting.

**Delta Scheduling (DS) is a gated stretch contribution.** Its first and load-bearing oracle
target is the one-step CF++ model intended for the streaming product. It proceeds beyond the
oracle study only if, on that model and on both limited-animation/held-frame and
dialogue/minimal-motion content:

- the oracle finds at least **30% reclaimable block compute** while intrinsic quality remains
  within its frozen floor; and
- the cheap delta signal reaches at least **0.80 precision and 0.60 recall** against the
   oracle-safe reuse set.

The oracle must also demonstrate at least **17% end-to-end wall-clock reduction on one-step
CF++** under dense, realizable execution on at least one supported backend. The measured path
must include delta calculation, selection, compaction, cache movement, synchronization, and
the interaction with the existing AR KV cache. A FLOP-only saving, or a saving confined to a
heavy multi-step model, cannot authorize DS implementation. The 17% reduction is the minimum
ceiling consistent with P5's immediate 1.2x wall-clock target.

If any condition fails, DS stops. The oracle study and negative result are published, and
the portable engine plus measured budget table stand as the successful project result.

## 2. Proof ladder

Each rung includes its own publication checkpoint. A later rung is not required to publish an
earlier one.

### P0 - Portable correctness foundation

**Exit criteria**

- CUDA and MLX execute the same declared Rolling Forcing schedule and serialized stochastic
  tensors.
- Cache cursors, eviction, anchor storage, and anchor re-RoPE invariants pass.
- Numerical outputs are reported against the immutable frozen parity band and `0.999` canary;
  failures remain failures.
- Same-decoder pixel artifacts and metric implementations are digest-bound.
- The three claim layers are separate: trajectory, intrinsic quality, and pairwise identity.

**Current state:** substantially reached. The 60.56-second replay is structurally correct;
the final cosine of 0.998944 visibly misses the unchanged 0.999 canary. Intrinsic calibration
capture is complete. The frozen v1 sanity gate failed one epsilon-dominated high-motion cell;
the preserved v2 derivation amendment demotes all epsilon-dominated cells uniformly while
requiring semantic, frame-dynamics, and temporal-behavior coverage. Blind holdout remains
sealed.

**Publishable checkpoint M0:** reproducibility/correctness technical report with fixtures,
Metal provenance, cache probes, parity curves, the canary miss, and exact reproduction steps.
The report must frame the canary as a trajectory-only diagnostic: the observed BF16
cross-platform drift is bounded but misses the frozen threshold, while the intrinsic
non-inferiority verdict remains pending until M1. It must not describe the canary miss as a
quality failure or claim an intrinsic pass early. This can ship before performance
optimization.

### P1 - Frozen quality protocol and baseline budget table

**Implementation steps**

1. Freeze a canonical protocol artifact containing prompts, content strata, calibration and
   holdout seeds, 15-second and 60-second horizons, metric directions, margins, sample minima,
   identity tripwires, visual-selection rules, and cost/storage ceilings.
2. Complete intrinsic metric checks: batch-boundary tests, synthetic flicker and motion cases,
   degenerate-motion rejection, finite-value checks, and metric model/state provenance.
3. Capture independent CUDA teacher calibration videos, one record per complete video.
4. Derive and freeze **horizon-indexed** prompt-specific teacher centers, robust spreads, and
   floors separately for the declared 15-second and 60-second horizons. A native-window floor
   cannot be reused as a 60-second floor.
5. Size seed count per prompt and prompt count per stratum separately. Mean-gated metrics use
   at least 8 calibration seeds and 4 blind holdout seeds per prompt. Quantile gates require
   a pre-declared estimator with enough independent videos for that quantile; p05/p95 remain
   diagnostic unless there are at least 20 independent videos per prompt. Frames never count
   as independent samples.
6. Freeze the resulting policy bundle before opening holdout results.
7. Run baseline CUDA and MLX workloads with no optimization sweep. Record DiT, VAE, transfer,
   synchronization, and encoding time separately.

**Publishable checkpoint M1:** a baseline paper/report titled along the lines of “Portable
Diffusion Video, Measured Honestly,” containing the frozen protocol, cross-runtime correctness,
blind quality verdict, and measured hardware budget table. This is an independently sufficient
successful outcome.

**Current state:** the H100 teacher calibration is complete at 256/256 videos. Protocol v2 and
its policy bundle were frozen before holdout visibility. The H100 holdout-reference capture and
M5 MLX candidate capture are both complete at 128/128 videos: 64 at 15 seconds and 64 at 60
seconds per backend. All 128 CUDA/MLX stochastic payload pairs passed literal byte identity.
The one-shot frozen evaluator has now run and published an authoritative **FAIL**. Intrinsic
quality passed 90/128 videos but failed 38/128 and clears only 19/32 frozen prompt-horizon cells
at the required three-of-four rate. Pairwise identity failed 128/128 videos and clears 0/32
prompt-horizon cells. The policy remains unchanged and the sealed evaluation will not be rerun.
See `agent/m1_sealed_holdout_report.md`.

The first M5 steady-state component row measures 0.768 native FPS for an 81-frame clip, with
transformer work at 54.6% and causal VAE at 43.3% of median wall time. The completed candidate
campaign adds 64 independent timing rows at each frozen horizon. Median native FPS is 0.287 for
15-second jobs and 0.185 for 60-second jobs. Treat those rows as resident-model batch throughput,
not TTFF or thermally controlled sustained-streaming evidence. Native
generation FPS is `pixel_frames / (diffusion_seconds + clean_cache_refresh_seconds +
scheduler_seconds + vae_seconds)`; prompt loading/encoding, metrics, and artifact publication
remain separately reported and excluded from that FPS. See `agent/m5_holdout_candidate_report.md`.

### P1 stage lookback - H100 holdout references

- Frozen reference batch: `sha256:10af5568b5c42ac9d081621a0f63862ec4c96c8751d03c8b12e775aa6dab337a`.
- Shared-noise index: `sha256:3fbfbf2896e566f663a1b781e20872c393b25e0d6333d33268b3c069e4b72a17`;
   eight plans, 968,048,640 payload bytes.
- First launch failed before publishing a result because CUDA formatted a warped timestep as
   `833.3333740234375` while the serialized semantic identifier used `833.3333129882812`.
   CUDA transition names now derive from configured integer schedule steps, as MLX and the plan
   serializer already did; numerical scheduler values and stochastic bytes were unchanged.
- The restarted run reached 16 results before the 300 GB RunPod volume quota stopped the process
   during atomic publication of job 17. The 16 results were checksum- and byte-validated, the
   sole incomplete directory was removed, the volume was expanded to 500 GB, and execution
   resumed through the same restartable batch contract.
- Final state: 128/128 results, no partial directory, 105 GB reference tree, all artifact and
   stochastic checks green. Resume completed 112 jobs in 6,551 seconds; 16 were validated and
   checksum-skipped. See `agent/h100_holdout_reference_report.md`.

### P2 - Portable product surface and fast-clip baseline

**Implementation steps**

1. Establish a clean one-step CF++ baseline on M5 Max: no competing container workload, clean
   swap state, fixed workload, within-backend quality check, peak MLX memory, system memory
   pressure, swap delta, and thermals.
2. Quantize that unchanged model as an independently attributed acceleration track. Measure
   int8 before int4 and bank only quality-passing wall-clock gains. This work proceeds in
   parallel with the DS oracle and does not depend on a DS GO.
3. Add `liveframe benchmark` output for generated FPS, presented FPS, TTFF, p50/p95 chunk
   latency, end-to-end clip time, peak memory, thermals, and all three quality layers.
4. Add only the minimum benchmark/checkpoint artifacts needed to repeat the baseline and
   quantization measurements.
5. Defer the generic `stream()`/`clip()` engine, broad adapter framework, and multi-model matrix
   until the CF++ DS oracle reaches a GO. If DS stops, build only the smallest product surface
   justified by the quantized baseline.
6. Add checkpoint/resume and artifact manifests so another supported machine can repeat the
   exact workload.
7. Validate the shipped MLX VAE against the pinned H100 VAE using serialized latent inputs,
   frozen pixel-space parity bands, identical emission semantics, and decoder provenance. The
   same-decoder DiT experiment does not substitute for cross-runtime VAE validation.
8. Measure M5 fast-clip mode first at one frozen 5-second configuration. Then optimize toward
   **under 5 seconds on M5 Max**, with every optimization rerun through the quality policy.

**Completed CF++ baseline evidence (2026-08-02):** one warmup and three measured trials completed
with 0 MiB startup and run swap, all latent quality anchors passing, and no thermal warning. Median
end-to-end time is 69.167 seconds for 81 frames (**1.171 generated FPS**), TTFF is 1.489 seconds,
and peak MLX memory is 26.669 GiB. Incremental causal VAE decode consumes 54.431 seconds (78.7%);
CF++ denoising plus cache refresh consumes 13.927 seconds (20.1%). See
`agent/m5_cfpp_baseline_report.md`. This supersedes the contaminated 0.185 FPS observation as the
one-step CF++ speed baseline.

**VAE optimization state (2026-08-03):** interleaved isolated-decoder repeats show that increasing
the VAE-only free-buffer cache from 1 GiB to 4 GiB reduces median VAE wall from 52.214 to 40.330
seconds (**1.295x**) with exact raw RGB output and no added swap. Async/deferred evaluation adds
no gain. Compiled steady state reaches 35.985 seconds at 4 GiB but changes RGB output and remains
quality-gated. A restart-gated clean run confirmed the isolated result at 39.484 seconds. A clean
integrated 1/2/4 GiB matrix then established that 4 GiB coexists with resident DiT, context, and
KV state: the claim-eligible one-warmup/three-trial baseline measures 57.127 seconds end to end,
1.418 generated FPS, and 41.319 seconds of VAE wall. This is 17.41% less end-to-end wall and
24.09% less VAE wall than the authoritative 1 GiB baseline, with exact raw RGB, all latent
quality anchors passing, and zero swap. See `agent/m5_cfpp_vae_profile_report.md`.

“Runs on any machine” means portable algorithm semantics and reproducible behavior on declared
supported backends. It does not mean every machine reaches the same latency. Initial support is
CUDA/NVIDIA and MLX/Apple Silicon; additional backends require their own parity and budget rows.

**Publishable checkpoint M2:** installable package, reproducible benchmark command, CUDA/M5
budget table, and a useful clip workflow. If M5 sub-5 seconds is missed, publish the measured
bottleneck table and fastest quality-passing configuration rather than hiding the miss.

### P3 - Sustained streaming result

**Implementation steps**

1. Profile and optimize the largest measured bottleneck only: quantized matmuls, attention,
   cache movement, VAE overlap, or synchronization.
2. Require each optimization to pass the frozen numerical/quality/identity policy before it
   enters the engine.
3. Run 60-second interleaved sustained trials under declared thermal and power conditions.
4. Report native generated FPS separately from optional interpolation/presented FPS.

**Exit criteria:** the selected configuration maintains both its FPS target **and intrinsic
quality within the frozen floor for the full 60 seconds**. It must also pass the pairwise
identity tripwire and publish drift curves; a fast first 15 seconds followed by quality decay
does not pass.

**Publishable checkpoint M3:** sustained CUDA and/or Apple Silicon streaming demonstration with
full thermal, latency, quality, identity, and drift evidence. A quality-passing 8 FPS M5 result
is publishable even if 16 FPS is not reached; 16 native FPS remains the headline target.

### P4 - Delta Scheduling oracle gate

P4 starts immediately after the clean one-step CF++ baseline. It runs in parallel with
quantization and requires no generic engine, adapter framework, or second model.

**Pre-registered wall-clock denominator (2026-08-02, before oracle visibility):** P4 has two
staged decisions. Oracle-reclaimable compute, quality, and cheap-signal precision/recall may be
measured on the current clean CF++ pipeline while VAE work proceeds independently. The 17%
realized wall-clock decision is evaluated later as DS-on versus DS-off on one named
post-VAE-optimization baseline artifact. That artifact, configuration, allocator policy, quality
result, and digest must be frozen before realizable DS wall-clock measurements begin. DS-on and
DS-off must use identical model/VAE weights, quantization, execution schedule, inputs, runtime,
and measurement policy; the only permitted difference is DS and its required overhead. The
baseline may not be selected or changed after any DS wall-clock result is visible. Earlier
compute/quality/signal results cannot authorize P5 without this paired 17% wall-clock pass.

**Frozen post-VAE DS denominator (2026-08-03, before DS wall-clock visibility):** the named
DS-off baseline is `artifacts/m5-cfpp-1step-cache4-baseline.v1.json`, SHA-256
`sha256:5bdc9fdbae59842158f278163d738656d1e4c05996ccf7495cf4d074009e3ce1`. It uses
the frozen one-step CF++ BF16 model and 21-block fixture, MLX/MLX Metal 0.32.0, the Metal
28.080 GiB recommended working-set limit, and a 4 GiB free-buffer cache configured before model
loading. One warmup and three measured trials yield a 57.127-second median wall denominator,
1.418 generated FPS, and 41.319 seconds of VAE wall. All trials created 0 MiB swap, passed the
pinned H100 latent anchor, and reproduced raw RGB digest
`sha256:4b55f063d0164fd1c44358186af6cd04ae2dba1356dedd5b86403dfd101e8f61`.
The retained video is `outputs/m5-cfpp-1step-cache4-baseline.mp4`, SHA-256
`sha256:245b41d6077948e23006af753178ae943e84e0702fa423f291b66a77bfbae151`. This
denominator is immutable for the staged DS realized-wall decision; later non-DS optimizations
require a separately named experiment and cannot retroactively replace it.

**Backend-feasibility amendment (2026-08-03, before oracle or DS wall-clock visibility):** the
frozen M5 denominator is retained as an immutable secondary control, but it cannot be the sole
M4 realized-wall backend. Its 57.127-second median contains 7.928 seconds of denoising (13.88%
of wall), 6.874 seconds of mandatory clean-cache refresh (12.03%), and 41.319 seconds of VAE
decode (72.33%). The current DS design reuses spatial block residuals and KV state during
denoising; it does not establish that the clean-cache refresh can be skipped. Its defensible M5
end-to-end ceiling is therefore 13.88% before DS overhead, below the 17% gate. Even counting all
transformer work would require reclaiming at least 65.6% of transformer wall before overhead.

Before any DS instrumentation, oracle replay, or DS-on timing begins, capture and freeze a
claim-eligible one-step CF++ H100 denominator with the same 21-block fixture and complete
component timing for denoising, clean-cache refresh, scheduler, causal VAE, transfers,
synchronization, encoding, and end-to-end wall. It must publish model/VAE/fixture/runtime
digests, quality and identity results, trial policy, and its report digest. Pre-register H100 as
the primary backend for the 17% paired DS wall-clock decision only if its measured DS-addressable
wall share makes 17% mathematically attainable after overhead. Otherwise M4 records a
pre-measurement feasibility STOP rather than running a foreordained timing test. The M5 baseline
remains valid for cross-backend overhead and negative-control reporting and is not replaced by
the H100 row. Execute this prerequisite through `agent/h100_cfpp_denominator_runbook.md`.

**Frozen H100 DS denominator (2026-08-04, before DS instrumentation or oracle visibility):**
the claim-eligible primary-backend report is
`artifacts/h100-cfpp-1step-denominator.v1.json`, SHA-256
`sha256:82394c2bad77718668c8fbe7b42dd568de5a607679737312a8004321e740f766`.
It binds harness SHA-256
`sha256:93f886e6e397fb4fc49888ed5fca1534c89e8876189655a787231f09e41589d4`,
Causal-Forcing revision `8db419e341e5fc52542c0b2c4542728420ddfb4a`, the frozen one-step
checkpoint/VAE/output anchors, full attention, TF32 disabled, and NVIDIA H100 80GB HBM3 driver
580.159.04. One warmup and three measured trials all reproduced exact latent and FP16 pixel
anchors. The median complete wall is 5.960916 seconds (13.588516 generated FPS), including
1.440107 seconds denoising, 1.165517 seconds mandatory refresh, 1.851257 seconds causal VAE,
0.545644 seconds materialization, and 0.363311 seconds encoding. The denoising-only addressable
share is 24.1592%, so H100 is mathematically eligible as the primary paired wall-clock backend,
but the oracle must reclaim at least 70.3667% of denoising before overhead to make 17% complete
wall attainable. The generic 30% reclaim floor alone implies only about 7.2477% complete-wall
reduction. Even perfect denoising elimination permits at most about 0.427 seconds total DS
overhead. These measured constraints are immutable for M4; the realized path must still pass the
paired 17% gate.

**Refresh-addressability amendment (2026-08-04, after denominator visibility and before any P4
oracle capture or result visibility):** P4 must settle whether mandatory clean-cache refresh is
DS-addressable before broad oracle instrumentation. The first specified mechanism is cached
refresh-KV substitution with temporal re-RoPE for genuinely static content. It is motivated by
the existing Rolling Forcing three-frame anchor path: `RollingForcingKVCache` stores the anchor's
raw key, re-applies temporal RoPE when its visible position changes, and reuses its value. That
implementation is evidence that temporal key repositioning machinery exists, not evidence that
full-attention CF++ refresh can already be skipped. The existing anchor operates under a bounded
21-frame local-attention policy, while this denominator uses full attention. Moreover, values
are not position-rotated, and deeper-layer refresh K/V depend on contextual transformer
activations rather than only the input tile and rotary offset. Re-RoPE refresh reuse therefore
remains a falsifiable mechanism hypothesis until the following proof gate passes.

1. Specify the substituted state per layer and tile: raw key, re-RoPE'd key, value, cache indices,
   and the source and destination temporal positions. Directly relabeling already-RoPE'd keys or
   reusing cache entries with unresolved positional ancestry is prohibited.
2. Compare substituted refresh state against full refresh at every affected layer, then continue
   generation so the test measures future latent, K/V, and decoded-pixel drift rather than only
   immediate cache similarity. Full-attention propagation is part of the measured error.
3. Publish horizon-indexed drift curves through both frozen 15-second and 60-second horizons for
   the declared prompts, independent seeds, and content strata. A three-to-five-block spot check
   cannot authorize refresh accounting. No shorter-horizon result may be extrapolated unless an
   extrapolation rule is separately frozen before its source observations are visible.
4. Require the same frozen intrinsic quality floors and identity tripwires used by P4. Cache
   cursor correctness, finite tensors, and exact source/destination position accounting are
   structural prerequisites and cannot rescue a quality or trajectory failure.
5. Demonstrate a realizable dense-skipping or compacted execution path. Refresh FLOPs count as
   reclaimed only when the measured implementation avoids them after including re-RoPE,
   selection, cache movement, compaction, and synchronization overhead.
6. Report four non-substitutable quantities: denoise-only reclaim, refresh-only reclaim, combined
   reclaim weighted by the frozen component times, and paired complete-wall reduction including
   all overhead. If this proof gate fails, refresh remains mandatory and denoise-only accounting
   controls M4.

On the frozen H100 denominator, refresh is 19.5527% of complete wall and denoise plus refresh is
43.7118%. Thirty percent combined reclaim implies only 13.1135% complete-wall reduction before
overhead and is not a GO. The zero-overhead lower bound for the 17% wall gate is 38.8911%
combined reclaim; any realizable path requires more. Therefore M4 passes only when both the
pre-registered 30% oracle-compute floor and the paired 17% complete-wall gate pass. A result such
as 31% combined reclaim is explicitly a wall-gate failure, not authorization to proceed.

The prospective proof contract is frozen as
`artifacts/refresh-addressability-manifest.v1.json`, manifest digest
`sha256:4433ab656e55e9e337c27d6b453b43991594e9fc2fddf1cb814eaeb72f975d48`.
It contains 64 paired full-refresh/re-RoPE jobs: all eight prompts in the two DS gate strata,
both 15-second and 60-second horizons, and four fresh seeds (`130003`, `140009`, `150011`,
`160001`) disjoint from calibration and opened holdout. The mechanism, ancestry schema, drift
curves, quality binding, and four-way accounting are immutable after the first proof result is
visible. Synthetic development data may test the implementation but may not alter this campaign.

**Implementation steps**

1. Use the frozen H100 component denominator above; do not replace or retune it after DS
   instrumentation or oracle visibility.
2. Run the refresh-addressability proof gate above and freeze its result before deciding whether
   P4 instrumentation may count refresh. Do not tune its mechanism against P4 oracle captures.
3. Sub-profile and optimize the causal VAE first, banking only quality-passing changes. Freeze the
   resulting named post-VAE baseline and digest before realizable DS wall-clock measurement.
4. Instrument unchanged one-step CF++ full-compute generation to record per-tile latent deltas,
   block residual norms, KV drift, wall-clock block cost, prompt identifier, horizon, seed, and
   content stratum. Preserve the two M4 GO strata as explicit fields rather than inferring them
   after capture.
5. Build an offline oracle replay that selectively reuses a tile only where post-hoc evidence
   says reuse remains inside the frozen quality floor.
6. Measure oracle-reclaimable block compute and actual wall-clock ceiling separately.
   Wall-clock measurement must use a realizable dense/compacted execution path and include
   selection, compaction, cache movement, and synchronization overhead.
7. Evaluate cheap delta-signal precision and recall against the oracle-safe set. Report the
   fraction of the oracle ceiling captured, not precision in isolation.
8. Freeze the R0 report before implementing a learned or hand-tuned reuse policy.

**Decision checkpoint M4**

- **GO:** on one-step CF++, for both limited-animation/held-frame and dialogue/minimal-motion
   classes, at least
   30% oracle-reclaimable block compute at quality within the horizon-indexed floor, with the
   cheap signal reaching at least 0.80 precision and 0.60 recall. In the separately staged paired
   test against the pre-registered post-VAE baseline, the realizable oracle path must also reduce
   end-to-end wall-clock time by at least 17% on at least one supported backend, including all
   selection, compaction, cache, synchronization, and signal overhead. Proceed to P5 and test a
   second architecture only if every condition passes.
- **STOP:** any compute, quality, precision, recall, or wall-clock threshold fails. Publish the
   CF++ oracle ceiling/negative result, archive DS as a completed research branch, and continue
   independently with quantization and measured product optimization.

This checkpoint is publishable in either direction because it measures a previously uncertain
spatiotemporal reuse ceiling.

### P5 - Delta Scheduling implementation, only after GO

**Implementation steps**

1. Implement M1 static-tile reuse with dense tile compaction and periodic full refresh.
2. Require at least 1.2x wall-clock speedup on animation content with all quality gates passing.
3. Add M2 motion-proportional step allocation, then M3 deadline-budget control, one at a time.
4. Permit at most two policy iterations. If neither clears quality and speed gates, stop and
   publish the oracle-versus-policy gap.
5. Port the exact policy semantics from CUDA to MLX; backend kernels may differ.
6. Only after both backends pass, test unchanged policy logic on a second model family.

**Publishable checkpoint M5:** DS algorithm report if the policy captures at least 60% of the
oracle ceiling and yields at least 1.4x sustained FPS on animation or 1.15x on general content,
with 60-second quality/drift gates intact. Cross-family transfer is a later strengthening result,
not a requirement for the portable-engine paper.

## 3. Accelerated critical path

Work is ordered to produce evidence quickly and prevent speculative implementation.

### Next 72 hours

1. Write and validate the frozen protocol JSON/schema using the implemented
   `liveframe.perceptual-policy.v2` policy schema. Keep this name distinct from
   `liveframe.perceptual-metrics.v1` and rolling parity report schema versions.
2. Select the smallest defensible suite with four explicit quality strata:
   limited-animation/held frames, dialogue/minimal motion, cyclic motion, and high-motion
   dense scenes. Use at least four prompts per stratum, 8 calibration seeds per prompt, and
   4 untouched holdout seeds per prompt at fixed 15-second and 60-second horizons. The quality
   suite covers all four strata; the initial DS GO gate is intentionally limited to the first
   two high-redundancy classes named in M4.
3. Add deterministic intrinsic-metric tests and degenerate-motion rejection.
4. Add a calibration manifest generator that emits expected run IDs before any capture begins.
5. Produce a single-machine baseline profile split by DiT/VAE/synchronization/encoding.
6. Draft the M0 report directly from existing 60.56-second artifacts; do not wait for DS.

### Following 7 days

1. Allocate H100 time to teacher calibration first because M1 and every later quality gate
   depend on it. Run calibration in bounded batches and derive the frozen policy bundle.
2. Run the untouched holdout once; no threshold changes after results are visible.
3. Capture the first complete M5 and H100 budget rows using identical workload manifests.
4. Use only remaining H100 allocation for P4 full-compute oracle capture; P4 may not preempt
   calibration, holdout, or baseline budget rows.
5. Implement the minimal backend-neutral `clip()` contract and benchmark command.
6. Begin DS logging instrumentation, but no reuse policy implementation yet.

### Following 2-3 weeks

1. Publish M0/M1 as soon as the blind holdout and budget rows are complete.
2. Optimize M5 fast-clip mode against the measured dominant cost.
3. Run the DS oracle experiment and make the M4 GO/STOP decision.
4. If GO, begin M1 tile reuse; if STOP, redirect all acceleration effort to portable engine
   bottlenecks and sustained streaming.

## 4. Scope controls

- Do not start a second model family before M2 and the DS GO decision.
- Do not build a polished UI before `clip()`, `stream()`, and `benchmark` are stable.
- Do not tune against holdout seeds or reinterpret historical measurement-only runs.
- Do not count interpolation toward native generated FPS.
- Do not weaken `gates.v1`, the 0.999 canary, or a frozen quality policy.
- Do not ship a cross-platform pixel portability claim until MLX-versus-H100 VAE parity has a
   frozen result; if P2 cannot establish it, scope the claim explicitly to same-decoder DiT
   parity and report native decoder outputs separately.
- Do not optimize unprofiled components.
- Keep visual galleries required for publication but off the implementation critical path.
- Mobile control and broad hardware support remain deferred until the package surface is stable.

## 5. Immediate definition of done

The next implementation cycle is complete when all of the following exist:

- A reviewed, canonical, digest-bound protocol artifact.
- Synthetic tests proving intrinsic metrics and fail-closed behavior.
- A generated calibration run manifest with fixed prompts and seeds.
- One baseline component-cost profile on M5 Max.
- An M0 report outline populated with current verified results and explicit failures.

At that point expensive captures can begin without compromising blindness, and a publishable
milestone is on a short, concrete path independent of Delta Scheduling.