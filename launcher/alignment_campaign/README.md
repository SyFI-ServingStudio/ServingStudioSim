# `launcher.alignment_campaign` — the alignment matrix engine

One alignment run validates one workload. Validating a *model* means running a
matrix of them, because a passing aggregate over one operating point says nothing
about the others. This package drives that matrix.

It is a daily tool first and a regression guard second. The regression half is
one flag (`compare --record`) on top of the daily half.

**It generates no evidence.** Every phase it runs is one of the five
`launcher alignment` stage commands, invoked unchanged. What it removes is
retyping the same command fifteen times, hand-transcribing numbers out of JSON,
and re-deriving a tolerance from memory.

## Layering

```
  pack (data, one per model × precision × GPU)      engine (code, model-agnostic)
  ────────────────────────────────────────────      ───────────────────────────────
  campaign.yaml     cases × axes, variants     ──▶  render   ─▶ run dirs: phase configs + traces
  hosts/*.yaml      checkpoint/nsys/venv/GPUs  ──▶  run      ─▶ one phase, all ready cases
  label_rules/      order-independent rules    ──▶  label    ─▶ apply to fixpoint, then check
  acceptance.yaml   tolerances + rationale     ──▶  extract  ─▶ metrics.json (never hand-typed)
  traces/invariants.json  the workloads' sha256 ──▶  compare  ─▶ vs tolerance, vs golden
  expert_popularity.json  measured routing          check    ─▶ pack self-check (pure CPU)
                                                                   ▲
  tests/golden/alignment_<pack>/   recorded  ◀── compare --record ─┘
```

The split is the point. Formulas depend on the Analyzer report schema and not on
the model, so they live in `metrics.py`; copying them into each pack would create
N copies to keep in sync. Tolerances are a judgement about one matrix, so they
live in the pack. Machine facts live in the host profile, so moving a matrix to
another box is one file.

## Verbs

```bash
python -m launcher alignment-campaign check   --pack <pack_dir>
python -m launcher alignment-campaign render  --pack <> --host <> --out-root <> [--case 01]
python -m launcher alignment-campaign run     --pack <> --out-root <> --phase <P> \
                                              [--case ...] [--dry-run] [--refresh] [--resume]
python -m launcher alignment-campaign label   --pack <> --run-dir <>
python -m launcher alignment-campaign extract [--pack <>] --runs <dir[:dir]> --out metrics.json
python -m launcher alignment-campaign compare [--pack <>] --measured metrics.json \
                                              [--json|--markdown] [--record]
```

`--pack` is **optional for `extract` and `compare`** — the read side. Point
`--runs` at a single one-off run directory and you get its numbers by the same
formula table, judged against the engine defaults, with no golden involved. That
makes the tool useful before a pack exists, which is exactly when a new model is
being aligned. `--pack` adds the tolerance overrides, the recorded baseline, and
`--record`.

`render`, `run`, `label` and `check` always need a pack: a case that was never
declared cannot be rendered.

### `run --phase P`

**The batch axis is one phase across N cases, never one case across N phases.**
There is no `--all` and never will be. `alignment/README.md` states "no phase
implicitly launches the next one"; at matrix scale that invariant keeps the
human inspection point between phases, while fanning one phase out across cases
only removes keystrokes. `skills/operate-run-alignment` asks for exactly this
shape: shared prerequisites once, then the cases in parallel with their exit
codes collected separately.

Execution reuses the launcher's existing machinery rather than reinventing it:
`ResourceScheduler` for in-process budgets, `LauncherLeases` for cross-process
exclusion, `RunJournal` for per-case stage state and exit codes, and the shared
`.complete` marker from `launcher/process/markers.py`.

Each case×phase carries its own marker, so a failure in labeling or analysis
never costs a repeat of the capture before it. A completed case is skipped;
re-running it takes an explicit `--refresh`. `--dry-run` prints the planned
commands plus, for every excluded case, *why* — already complete, or missing a
named input. Planning is a pure function (`execute.plan_phase`), which is what
makes it testable on CPU.

For `--phase simulation`, worker settings come from `simulation.yaml`, including
`gpu_time_multiplier` (default 1.0). Kernel analysis is an independent phase, not
a prerequisite for simulation. To adopt a reported multiplier, set it explicitly
in the experiment preset and record the source and approximation in experiment
notes. The campaign does not read calibration reports or inject worker overrides.

### `label`

`initialize`, then apply the manifest's rules repeatedly until the label
**state** stops changing, then `check`. The termination test is state equality,
not "no rule fired": `apply_rules` counts an overwrite as applied even when the
new label is byte-identical, so a fired-count loop would run to its cap every
time. Sweeps are needed at all because a rule may read its neighbour's
operation, which an earlier sweep is what writes.

The rules are a **set**, not a sequence. Every file the manifest names is loaded
and the union is applied, so no label may depend on which rule is tried first.
Two checks hold that: `subsumptions` proves an order-dependent pair from the
rule text alone and runs in `check`; `disagreements` observes one against a real
capture at the fixpoint and runs here. A manifest that still declares
`ordered_rule_files` is rejected rather than silently reordered.

`unfired` is a drift signal worth reading — a rule that matches nothing across
every case in the pack is dead weight left over from an older inventory.

## Pack layout

Packs live in `presets/alignment/<pack>/`, not `tests/fixtures/`: they are run
inputs, used to drive real GPUs.

```
presets/alignment/
  hosts/
    example.yaml            template for a new machine — the only tracked one
    *.yaml                  gitignored: a real profile is nothing but absolute paths
  <pack>/
    campaign.yaml           the matrix: variants (topology) × cases (operating points)
                            plus `recorded_on`, the recording machine as facts
    acceptance.yaml         tolerances + a written rationale per exception
    expert_popularity.json  measured routing demand, if the arch consumes it
    label_rules/            manifest.json + the rule files it names
    traces/invariants.json  each trace's sha256; the rows themselves are not
                            committed — `shapes` x `repeats` in campaign.yaml is
                            the trace, and `render` generates it
```

### `campaign.yaml`

A **variant** is a topology: `deployment`, `gpu` (the canonical name from
`gpu/spec.json`), `checkpoint` (a key into the host profile), `engine`,
`backend`, `server`, `arch`, `worker`, `input_builder`, `label_rules`, and
`profile_passes`.

`arch` is an **open dict** validated against the tag's parameters in the
Rust-exported `deployment_schema.json` via `Registry.arch_params`. It is not a
field union baked into this engine — that is what lets a new model land as a new
pack with no engine change. It is also written **once**: `launcher/alignment.py`
already projects `simulation.yaml`'s arch/gpu/backends into the timing-predict
config, so the pack renders the simulation preset and nothing copies arch fields
to a second place.

`profile_passes` is a **list** of `{kind, name, trace}`, not a fixed pair.
GLM-5.2 runs two (`nsys` + `workload_metrics`); the archived Llama3-8B alignment
ran one. `name` is simultaneously the config stem, the artifact directory, and
the `--phase` argument, so adding a pass touches no enum.

Each pass can also set `warmup: true` (default false). This renders
`workload.warmup: true` for that pass, using req-frontend's bounded warmup,
drain, prefix-cache reset and measurement boundary. vLLM variants must enable
server load tracking. Warmup does not alter either materialized trace.

A **case** is one operating point: `max_model_len`, `max_concurrency`, the two
arrival modes, `gpu_memory_utilization`, `capture_seconds`, `device_role`, its
`workload_trace` (and `kernel_trace` where the NSYS capture is bounded
separately), plus `purpose` and `coverage` so the matrix explains itself.

Optional per-case `chunk_size` sets both server `max_num_batched_tokens` and
worker `max_batch_tokens`. CUDA graph capture size remains a separate variant
setting. Optional calibrated `speculative_acceptance` supplies one conditional
probability per draft position. Rendering preserves the real-engine trace and
writes a separate `trace_speculative.csv` for the simulator with its required
`speculative` input tag. Architecture depth, worker depth and vector length must
agree; borrowed calibration remains marked `provisional`.

Two case fields carry more weight than their size suggests:

- **`attn_gpu_memory_gb` and `rate` are `Calibrated`**, not plain numbers. Each
  records `value`, `derived_from`, `evidence`, and `status: measured |
  provisional`. These are *measured* prerequisites — KV capacity is read out of a
  workload-only server log, the open-loop rate comes from a saturation run — and
  the generator scripts this replaced shipped a literal
  `# provisional; replace after saturation calibration` next to a hard-coded
  value. Separating the two means re-rendering on a new machine cannot silently
  inherit a stale calibration. `check` warns on any `provisional`; `compare
  --record` refuses to write a golden that depends on one unless
  `--accept-provisional` is passed, and then names the provisional fields in the
  provenance sidecar.
- **`analyze_iterations`** overrides the profiler's whole-capture default with a
  stable numbered excerpt. Use it when the accepted evidence population is a
  particular iteration interval that must remain fixed even if capture duration
  later changes.

`raw_overrides` is the escape hatch for a field this schema has not learned yet.
`check` warns when a pack uses it — it is a place to record a gap, not a place
to live.

### `acceptance.yaml`

`analyzer_schema` pins the report versions the tolerances were written against;
`compare` refuses to record against a run that disagrees. `tolerances.default`
covers every metric in the formula table; `tolerances.per_case` overrides, and
**every override must carry a `rationale`** — the precedent is
`alignment/load_generator/req-frontend/src/bin/selfcheck/check.rs`, which makes
`tolerance_rationale` a mandatory field rather than an optional comment. A
tolerance without a reason is indistinguishable from one widened to make a red
number go away.

`warn_fraction` is a single knob: a value consuming more than that fraction of
its tolerance is WARN rather than OK. One knob rather than a second per-metric
threshold, because "close to the declared edge" is the actionable signal and
nobody has the data to set fifteen separate warn bands.

## Metrics

`metrics.py` holds the formula table, grouped tight → loose in the order
`skills/top-align-with-framework` prescribes: kernel timing, then duty cycle and
workload structure, then end-to-end. `ratio(sim, meas) = (sim/meas - 1) * 100`.
Each metric's formula string is printed with its result and written into the
golden's provenance sidecar, so the derivation stays visible without living in
every pack.

The table **pins the report `schema_version` it reads** rather than adapting to
whichever one it is handed: `iteration` 2, `e2e` 1, `workload` 1 — the versions
the analyzer emits today (`ALIGNMENT_ITERATION_SCHEMA_VERSION`, `io::SCHEMA_VERSION`).
Anything else lands in a case's `issues` and clears `available`, so it becomes an
unavailable case rather than a wrong number.

Adapting was the alternative, and it was tried: `iteration` 1 carried no
`comparison` block, so its kernel error was an unweighted per-iteration mean
where 2 is duration-weighted over all iterations. Supporting both meant either
two metric names to keep aligned forever, or one golden key whose statistic
turned over with the schema. Since nothing emits 1 any more, refusing it costs
nothing and removes that whole class of question.

`extract` reads only the three small reports (iteration, e2e, workload). It never
opens `alignment_kernel_inventory.jsonl`, whose largest instance is 338 MB.

## Comparison and goldens

`compare` performs two independent judgements and does not blend them:

- **Acceptance** against `acceptance.yaml`: OK / WARN / FAIL, non-zero exit on
  FAIL. This is a declared threshold, following `conservation/workload.rs`.
- **Drift** against the recorded golden: warn only, per `skills/dev-run-tests`'
  "goldens are monitors, not thresholds".

Goldens live in `tests/golden/alignment_<pack>/<gpu_name>.json` with exactly the
shape of the existing store — flat `{key: float}`, sorted keys, trailing newline.
Keys are `<variant>/<case>@<metric>`, so a new topology is a new key prefix and
nothing migrates. `launcher/golden.py` holds the one storage implementation that
both `compare --record` and `pytest --update-golden` use.

That a production command writes under `tests/` is a real wrinkle. It is the
lesser evil: a second golden store would have to be kept consistent with the
first, and the alignment numbers are exactly the kind of thing the existing store
exists to hold.

## Files

| File | Contents |
|---|---|
| `pack.py` | Pack/variant/case dataclasses, `Calibrated`, YAML loading, host profiles |
| `render.py` | Phase configs and traces from a `(case, variant, host)` triple |
| `check.py` | Pure-CPU pack self-check; trace invariants; cross-phase consistency |
| `execute.py` | Phase planning (pure) and execution (scheduler + leases + journal) |
| `label.py` | Applying the rule set to a fixpoint, then `check` |
| `metrics.py` | The formula table, pinned to one report schema per report |
| `extract.py` | Run directories → `metrics.json` |
| `compare.py` | Acceptance, drift, `--record`, text/markdown/JSON rendering |
| `cli.py` | Argument parsing and verb dispatch |

`tests/test_alignment_campaign.py` is parameterized over **every** pack under
`presets/alignment/*/`, so a new pack inherits the gate without touching the
test.
