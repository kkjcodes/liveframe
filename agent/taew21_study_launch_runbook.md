# TAEW2.1 Human Study — Launch Runbook

Everything below is pre-registered; do not modify design parameters after any
response exists. Sources of authority: `artifacts/taew21-protocol.v1.json`
(`human_study`), `artifacts/taew21-study-power.v1.json` (selected design and
operative 486 target), `artifacts/taew21-protocol-amendment.v6.json` (v2
routing + human study as sole quality authority),
`artifacts/taew21-protocol-amendment.v6.1.json` (target-based stopping,
replacement enforcement), `artifacts/taew21-protocol-amendment.v6.2.json`
(primary-cohort completeness, `collection_complete` routing, execution
binding), `artifacts/taew21-protocol-amendment.v6.3.json` (terminal-state
rule for response freezing, exhaustion clarification — its digest is
embedded in the bundle manifest and verified by both tools), `artifacts/taew21-study-power.v2.json` (empirical error rates of the
executed collection state machine), and the frozen tools referenced here.

## Preconditions (automated; verify before recruiting)

1. `artifacts/taew21-metric-screen.v2.json` exists with
   `"verdict": "route-to-human-study"`.
2. The bundle exists at `outputs/taew21-study-bundle/` with
   `bundle-manifest.json`, `participant-schedules.json`, 32 study items,
   2 attention items, and `assignments.sealed.json` (DO NOT OPEN — it is
   opened only by the analysis tool after the response freeze).

## Design (fixed)

- 30 primary participants (P00–P29); each completes 22 trials: 18 study + 2
  attention + 2 repeat-consistency, in the per-participant randomized order
  given by `participant-schedules.json`.
- 6 pre-registered replacement participants (R30–R35), each with their OWN
  schedule row. **Target-based stopping (frozen in amendments v6.1–v6.3):**
  every primary slot P00–P29 must end accounted-for — a submitted session, or
  a withdrawn/duplicate entry (slots are schedule slots, not people: a
  recruit who never starts is re-slotted to the same ID). Only once all
  primaries are accounted AND usable judgments are below 486 do you activate
  replacements — exactly one at a time, strictly in R-ID order, re-running
  the QC tool between activations. Never activate a replacement while the
  target is already met (the analyzer hard-errors on this), and never re-use
  an excluded participant's ID. If all six replacements are exhausted below
  target, the study returns no-authority.
- Trial: watch `reference.mp4`, then `A.mp4` and `B.mp4`; answer
  "Which clip (A or B) matches the reference?" plus optional secondary
  "Which clip did you prefer? (A/B/none)".
- Stopping: collect until the blinded QC tool reports
  **`collection_complete: true`** — the fail-closed field requiring all
  primaries accounted AND zero policy violations AND (target met (486) OR
  valid replacement exhaustion, per amendment v6.3 — exhaustion authorizes
  freezing solely for the no-authority verdict). Never route on raw
  `target_met`. The analyzer independently enforces the same terminal-state
  rule and hard-errors on a premature freeze. QC is outcome-blind — it counts
  inclusions/exclusions only and never reads sealed assignments or computes
  study accuracy, so running it mid-collection is not peeking:

  ```bash
  .venv/bin/python tools/qc_taew21_study_responses.py \
    --bundle-root outputs/taew21-study-bundle \
    --responses outputs/taew21-study-responses.v1.json \
    --power artifacts/taew21-study-power.v1.json \
    --amendment artifacts/taew21-protocol-amendment.v6.3.json
  ```

**Distribution rule (blinding-critical):** participants receive item
directories (`item-*/`, `attention-*/`) and their own trial order ONLY.
`bundle-manifest.json`, `participant-schedules.json`, `attention-key.json`,
and `assignments.sealed.json` are never participant-accessible. Every bundle
video is a distinct encode (builder-verified unique digests), so hash
comparison cannot recover the answer key.

## Platform steps (Prolific or equivalent)

1. Host the bundle videos on a static page or survey tool (Qualtrics /
   Gorilla / custom page). Each participant gets their `participant_id`
   schedule. Videos must play at native 832×480 or larger, sound off.
2. Screener: normal or corrected-to-normal vision; desktop/laptop only
   (no phones); one submission per person.
3. Consent page: state the task (video comparison), time (~20-25 min),
   pay, data anonymity (no identifiers stored in the repo), right to
   withdraw. Meet the platform's fair-pay guidance (US federal-minimum
   floor at minimum; Prolific recommends >= $12/hr — pay ~$5-6 for ~25 min).
4. Budget envelope: 30 × ~$5.50 + ~33% platform fee + ~4 replacement
   participants ≈ **$250-300**.
5. Export responses to `outputs/taew21-study-responses.v1.json` in the
   format documented at the top of `tools/analyze_taew21_study_responses.py`
   (participant_id, item_id, role, answer, optional preference; plus
   withdrawn/duplicate participant lists).

## Response freeze and analysis (automated once you have the file)

```bash
.venv/bin/python tools/analyze_taew21_study_responses.py \
  --bundle-root outputs/taew21-study-bundle \
  --responses outputs/taew21-study-responses.v1.json \
  --v1-verdict artifacts/taew21-metric-verdict.v1.json \
  --power artifacts/taew21-study-power.v1.json \
  --amendment artifacts/taew21-protocol-amendment.v6.3.json \
  --responses-frozen \
  --output artifacts/taew21-study-analysis.v1.json
```

- `--responses-frozen` is the response-freeze assertion: pass it only when
  data collection is complete. It is what authorizes opening
  `assignments.sealed.json`.
- Verdict: `quality_go: true` iff BOTH the usable-judgment target (486) is
  met AND the one-sided 95% two-way clustered upper confidence bound on
  accuracy is <= 0.55. An under-target sample yields
  `no-authority-insufficient-usable-judgments` regardless of its bound.

## After the verdict

- **GO**: integrate the tier (decoder choice + generated FPS + decoder wall
  + complete wall + quality status into `liveframe benchmark`), update the
  paper, push (with explicit approval). The banked speed claim is a **2.61×
  complete-wall speedup with a 5.52× decoder-only speedup** — always cite
  both numbers with their denominators.
- **Not established**: a non-GO does NOT prove people can see the
  difference; it means the 55% bound was not shown (the design carries a
  28.1% inconclusive risk even at true chance). Publish as speed-GO /
  quality-not-established with the study estimate and uncertainty; the
  exercise still closes with an honestly-measured answer.

## Known risks (pre-registered; empirical values from
## `artifacts/taew21-study-power.v2.json`, the executed-design simulation)

- Pass probability at TRUE chance (accuracy 0.50) is **71.9%** — a 28.1%
  risk of an inconclusive fail even if the decoders are perfectly
  indistinguishable (the protocol's own 400-600 envelope caps this;
  accepting it was frozen).
- Pass probability is 14.25% at true accuracy 0.53 (the study's power for
  an acceptable-but-near-boundary effect — it will usually, and correctly,
  fail to bound 0.53 under 0.55), 1.85% at 0.55, and 0% at 0.58. "False
  pass" applies only to values outside the acceptable region (e.g. 0.58),
  where the rate is 0% in simulation.
