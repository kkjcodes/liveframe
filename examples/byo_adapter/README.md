# Bring your own world model

`drift_world.py` is a complete, working adapter in ~120 lines, most of them
comments. Copy it, replace the toy generation with calls into your model, and
the whole harness — suites, timed + metric passes, resume, reports, compare,
export — works against your model unchanged.

## The 10-minute path

```bash
pip install "liveframe[world]"

# 1. Prove the loop works with the toy adapter (CPU, ~seconds):
liveframe world run examples/byo_adapter/drift_world.py:DriftWorld --suite smoke

# 2. The command prints a `report:` line with the report.html path — open it.
#    (It's <run_dir>/report.html, where run_dir is in the JSON it printed.)

# 3. Copy drift_world.py, point step() at your model, declare honest
#    capabilities, and run again.
```

## Running inside your model's environment

Your model's dependencies (torch version, CUDA stack) never need to be
compatible with the harness. Pass your model env's interpreter and the same
adapter file runs **inside that environment**, talking to the harness over a
subprocess bridge:

```bash
liveframe world run my_model.py:MyWorld --suite quick \
    --python /opt/envs/my-model/bin/python
```

Preflight that environment first — every failed check prints its exact fix:

```bash
liveframe world doctor custom --python /opt/envs/my-model/bin/python \
    --imports torch,diffusers --pins my_model.pins.json
```

`pins.json` records the exact package versions you verified your adapter
against (see `src/liveframe/world/adapters/matrix_game2.pins.json` for the
shape). Resolver drift is the #1 bring-up failure class; the lockfile turns it
into a one-line fix instead of an afternoon.

## What the harness checks so you don't have to

- **Frame accounting** — every scripted frame arrives exactly once, in order.
- **Payload contract** — shape, dtype, and value range match your declared
  capabilities; violations fail the case, not the run.
- **Honest timing** — chunks are timed at receipt on the harness side;
  validation and capture never inflate your numbers. Cold start is reported
  as context, never blended into generation time.
- **Evidence** — every run directory carries a digest-bound ledger;
  `liveframe world verify <run>/ledger.json` re-checks every artifact.

## Rules worth repeating

1. `native_fps` is the rate your model is designed to be viewed at, not the
   rate it achieves. The harness measures achieved rates itself.
2. Emit frames as they are decoded. Buffering until the end hides stalls and
   misrepresents the cadence a user would experience.
3. Never duplicate or interpolate frames; native frames only.
