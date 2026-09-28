---
name: operate-run-simulation
description: Use when the user wants to run a ServingStudio Sim simulation from a preset (YAML preferred, JSON also accepted). This skill locates the repo logs root, names the experiment as date_index_short_name, prefers expressing requested variations as internal sweeps inside one copied preset, always does a dry run to inspect the expanded run plan before launching, and runs through the ServingStudio Sim launcher.
---

# Run ServingStudio Sim Simulation

Prepare and run a ServingStudio Sim simulation from an existing preset. This is the
ServingStudio Sim analogue of the global `run-moesim-experiment` skill — same shape (copy a
preset into a dated experiment dir, encode variations as sweeps, launch), but
the launcher and conventions below are ServingStudio Sim-specific. When in doubt about the
preset/sweep philosophy, defer to `run-moesim-experiment`.

**Preset format is the nested config tree** (`deployment` / `workload` / `io` /
`pools`, plus control blocks), authored in **YAML** (JSON also parses). The
authoritative, worked reference for the format — the tree skeleton, every sweep
mechanism, and the validation rules — is `launcher/README.md`. This skill is the
run *workflow*; defer to that README for the preset *grammar*.

For the *offline* cost predictor (explicit batch shapes, no discrete-event sim /
workload / pools) use `operate-run-timing-predict` instead — this skill is only
for a real deployment run driven by a workload trace.

Resolve the repository root from the active checkout with
`git rev-parse --show-toplevel`. Its `logs/` directory is the canonical logs
root for that checkout.

Launcher (run from the repo root):
- `uv run python -m launcher <preset>.yaml [--dry-run] [--override path=value ...]`

`uv run` is mandatory — it pins the PyO3-embedded interpreter to the 3.12 venv;
a bare `python` / `cargo` links the system 3.9 and crashes (see memory
`vibesim_pyo3_build_python_pin`). The launcher builds the binary + schema itself,
so you never set `LD_LIBRARY_PATH`/`PYTHONPATH` by hand.

## Trigger

Use when the user wants to run a named ServingStudio Sim simulation, gives a preset to
run, or wants a preset copied into a dedicated dated experiment log directory
before launch.

## Experiment naming

Format: `YYYYMMDD_N_<short-name>`

- **Date first**: `YYYYMMDD` for today.
- **`N`**: a per-day index starting at `0`, incrementing for each additional run
  the same day. Pick `N` by scanning the logs root for existing entries that
  share today's date prefix and using the next free index
  (`20260523_0_...`, `20260523_1_...`, ...).
- **`<short-name>`**: 5 words or fewer, underscore-separated. Strip filler words
  ("sweep", "experiment", "test", "run", "baseline") unless the user insists.
  Example: `20260523_0_llama3_tp_rate`.

If the user did not give a name, ask for one before proceeding.

## Workflow

1. Identify the preset to run (under `presets/`, or one the user names).
2. Decide the experiment name per the rules above (ask if not given).
3. Decide whether the requested variations fit one preset as internal sweeps —
   default to "yes" whenever the schema supports it (see Sweeps below). Only
   create multiple experiment folders / multiple presets when one preset cannot
   express the runs cleanly, or the user explicitly asks.
4. Compute the experiment dir: `logs/<experiment-name>`.
5. If the dir already exists, stop and ask whether to reuse it or rename. Do not
   silently overwrite an existing copied preset.
6. Copy the chosen preset to `logs/<experiment-name>/preset.yaml` (keep `.json`
   only if the source is JSON). Do not modify the original preset under
   `presets/` unless the user asks.
7. Edit the copied preset's internal `io.log_dir` to land under
   `logs/<experiment-name>`, preserving any template tail / placeholders (see
   log_dir Rewrite). Encode the requested variations in the `sweep` /
   `compound` / `derived` control blocks (see Sweeps below). Unless the user
   explicitly requests an analyzer subset, remove any inherited
   `analyze_subjects` key from the copied preset or variants manifest so the
   launcher runs every applicable analyzer subject. Do not add `--no-analyze`
   to the launch command by default. `--no-plot` is also opt-in; see
   "Plots and `--no-plot`" below.
   For MoE models, select the routing source using the MoE rule below before
   the dry run.
8. **Dry-run first** — validate + expand without launching, and inspect the run
   plan and per-run `log_dir`s:
   ```bash
   cd "$(git rev-parse --show-toplevel)"
   uv run python -m launcher logs/<experiment-name>/preset.yaml --dry-run
   ```
   Confirm the `[plan] N run(s)` count and that each expanded `log_dir` is
   distinct and lands under `logs/<experiment-name>/`. Fix the preset and
   re-dry-run until the plan is right.
9. Launch (drop `--dry-run`):
   ```bash
   uv run python -m launcher logs/<experiment-name>/preset.yaml
   ```

## MoE routing source

Before running any MoE model, read and apply
[the shared routing-source selection rule](references/moe-routing.md).

## Sweeps

Variations are **named** in a control block and **referenced** in the config
tree by a `${name}` placeholder — not by making a tree leaf list-typed. A
`${name}` may only stand for a *value* leaf; it can never replace structure (the
`deployment`, a provider `type` tag, or a whole `arch`/`worker`/pool block —
those select which schema applies and must be literal). Expanded by the launcher
(no launcher-side sweep flags). The four mechanisms:

- **`sweep`** → independent dims, crossed as a cartesian product. A **list**
  value is an unlabeled dim (`tensor_parallel: [1, 2, 4]`); a **dict** value
  `{label: value}` gives readable `log_dir` labels (`fp8: {on: true, off: false}`).
- **`compound`** → a named zip group whose members move **together** (correlated
  columns, no cartesian blow-up): `{group: {label: {member: value, ...}}}`. The
  group name is one aggregation axis. Use this instead of the old `sweep_groups`.
- **`derived`** → one name computed by a *formula* over sweep/compound inputs
  (e.g. `decode_replicas: "16 // decode_tp"`).
- **`constraints`** → a list of boolean expressions that prune invalid combos.

Every sweep dim / compound member must be **config-effective** (reach a real
config leaf via `${name}`), and names should spell out their meaning
(`prefill_tp`, not `ptp`) since they surface in `log_dir`, the aggregation axis,
and error messages. For *structural* comparison (different `arch.type`, pool
shapes, or deployments) use a **`variants`** manifest of separate presets, not a
placeholder — see `launcher/README.md`.

Prefer one preset with internal sweeps over many sibling experiment folders.

## Per-kernel backend selection (optional)

Every kernel picks its backend from a candidate set; best-of-N keeps the fastest
at eval. The default set is the arch's const-default. To tailor it per kernel:

1. **Emit the skeleton** — enumerate the distinct kernels this preset touches
   (structural walk, NO GPU / profiling) and print a `backends:` skeleton, one
   entry per kernel role, pre-filled with the current default and annotated
   `kind | dtype | options | shape`:
   ```bash
   cd "$(git rev-parse --show-toplevel)"
   uv run python -m launcher logs/<experiment-name>/preset.yaml \
     --emit-backends logs/<experiment-name>/backends.yaml
   ```
   (omit the FILE arg to print to stdout instead.)
   `options` is already dtype- AND GPU-filtered (from each kernel's declared
   `BackendSupport`), so it lists only the backends legal for this run (e.g. `trt`
   is dropped off a non-Blackwell GPU).

2. **Edit `backends.yaml`** — set each role to a subset (`[fa3]` forces one,
   `[fa2, fa3]` is best-of-N), keep the default, or point it at a `${var}` declared
   under the preset's `sweep:` — a backend candidate list is a normal sweep value,
   so backends cross-product with tp/ep like any other axis.

3. **Point the preset at it** — add `backends_file: backends.yaml` (resolved
   relative to the preset), or inline a `backends:` block. Keys are pool-prefixed
   role names (e.g. `attn/afd.attn.prefill`), stable across a tp/ep sweep.

The launcher re-enumerates + validates every concrete run: an unknown/stale role
key, a backend the kernel can't run at its dtype/GPU, or a role left unassigned
(strict coverage) is a hard error. If a sweep yields runs with DIFFERENT kernel
role sets (e.g. `tp=1` has no `tp_allreduce`), `--emit-backends` hard-rejects —
split those into separate presets/files. Shape-only differences across the sweep
are fine (the skeleton marks the moved shape `(varies)`; one map still spans it).

## log_dir Rewrite

`io.log_dir` uses `{name}` string templating where each `{name}` is a **bare
identifier** — no `{x.attr}` / `{x[0]}` / `{x!r}` / `{x:spec}`. A `{name}`
resolves to a dict-sweep / compound label if it has one, else the swept value
stringified. When rewriting, keep the existing path shape and replace only the
experiment-name segment:

- `logs/<old-name>` → `logs/<experiment-name>`
- `logs/<old-name>/tp{tensor_parallel}` → `logs/<experiment-name>/tp{tensor_parallel}`
- missing `log_dir` → `logs/<experiment-name>`

For a sweep, extend the path with a `{name}` per swept axis so each expanded run
gets a distinct subdir. An unknown `{name}` stays literal and warns — usually a
misspelled sweep dimension. The launcher rejects a sweep whose runs collide on
`log_dir`, so cover every swept axis in the template.

Example sweep preset (YAML, nested tree — a TP sweep on llama3-8b; note the TP
arch tag `llama3_dense_tp` is what exposes `tp_size`):
```yaml
deployment: unified

workload:
  trace_files:
    - trace/aime_long.csv
  input_file_format: text-generation-independent
  arrival_mode: trace_timed
  session_dependency: independent
  request_rate: 150.0
  duration_ms: 20000000.0
  run_to_end: false

io:
  log_dir: logs/20260526_0_llama3_tp_throughput/tp{tensor_parallel}

pools:
  main:
    groups:
      - gpu: "NVIDIA H200"
        replicas: 1
        arch:
          type: llama3_dense_tp        # the TP arch; exposes tp_size
          model_config: model/config/llama3_8b.json
          tp_size: ${tensor_parallel}  # typed-int injection from the sweep below
        worker:
          type: barebone

sweep:
  tensor_parallel: [1, 2, 4, 8]
```

## Cache build output

The launcher prebuilds each unique kernel cache key sequentially before any
parallel launch. That output now lands under
`logs/<experiment-name>/.cache_build/<run-folder-label>/stdout.log` (inside the
experiment dir, named after the run folder — not a scattered top-level hash
dir). Nothing to configure; just expect that `.cache_build/` subdir.

**Do not hand-enumerate the missing profile.db rows.** The cache build *is* the
mechanism for a cold cache: `build-cache-only` walks the compiled cost tree and
JIT-profiles exactly the specs the run will look up. Do not read the preset,
guess which kernel shapes are missing, and pre-fill them with
`kernel-profile count-missing` / `kernel-profile run` batches — a hand-derived
list will not match the tree's real lookups, and it duplicates work the launcher
does for free. Just launch; if you want the coverage *before* spending GPU time,
run `uv run python -m launcher <preset> --cache-report` (reports missing specs
per kernel per cache key, builds nothing). Reach for
`operate-profile-existing-kernel` only for a question about one specific
kernel's rows, never as a pre-step for a simulation.

## GPU selection

**Do not restrict how many GPUs the run may use unless there is a concrete
reason to.** L1 owns idle-device discovery and arrangement: `find_idle_gpus`
(`profiling/exec/local.py`) reads `nvidia-smi` and hands only genuinely idle
devices to the GPU pool, which distributes the cache-build / JIT-profiling work
across them itself. Setting `CUDA_VISIBLE_DEVICES=<one idle gpu>` (or
`VIBESIM_PROFILE_GPUS`) only *shrinks* the set L1 gets to choose from — it
serializes a fill that could have run in parallel, and it hard-fails if that one
card turns out to be busy. Launch the launcher bare. Pin devices only when the
user asks, or when the fill must stay off specific cards; say why when you do.
(Note this is about the *profiling* devices, not the simulated cluster — the
modeled GPU count comes from the preset's pools, and is never inferred from the
host.)

### No GPU: `--no-gpu`

`--no-gpu` (or `SERVINGSTUDIO_NO_GPU=1`) guarantees the launcher uses no GPU in
any mode. Pass it when the user asks, or when the run must stay off the host's
GPUs. A run over a warm `profile.db` is unchanged. With missing rows, the cache
prebuild fails before profiling and names the missing count; check coverage
first with `--cache-report`. Report that failure; do not drop `--no-gpu`
without asking.

## Analysis output

Each finished run is auto-analyzed (best-effort — a failure never fails the run)
into `reports/` (numbers JSON), `payloads/` (plot JSON), and `plots/` (PNG) under
its `log_dir`. The full list of analysis **subjects** a run produces and what each
one reads and emits is the canonical Subjects catalog in
`analyzer/README.md` — point there instead of guessing metric names. Selection is
the optional preset key `analyze_subjects`.

### Plots and `--no-plot`

By default the launcher also renders PNGs: each run's `plots/`, and a sweep's
aggregate figures in the experiment root's `plots/`. `--no-plot` runs the
same analysis and writes the same `reports/`, `payloads/` and trace, but
renders no PNGs at all. The Analyzer UI and `operate-use-analyzer` read
reports and payloads, not PNGs, so nothing downstream breaks.

Rendering is the most expensive post-run step. On a 640-run Llama-3-8B sweep
it took ~22 CPU-s per run against ~3 CPU-s for the simulation, and
`--no-plot` cut the sweep from 200 s to 92 s. Keep the default for a single
run or a small sweep. Pass `--no-plot` when the user asks for speed or says
they do not need figures. For a large sweep (hundreds of runs), suggest
`--no-plot` to the user before launching, rather than choosing it for them.
With either flag, you can later render one run's figures with
`uv run python analyzer/python render <run log_dir>`.

`--no-plot` is not `--no-analyze`: `--no-analyze` skips the analyzer, so no
reports, payloads, trace or PNGs are written.

**Default invariant: run all applicable subjects.** Omit `analyze_subjects`
(preferred; an empty list has the same launcher meaning) and do not pass
`--no-analyze`. “All” means every registered subject whose applicability gate
accepts that run; it does not mean forcing deployment-incompatible subjects.
Do not inherit a narrow `analyze_subjects` list from the source preset merely
because it was present there. Keep or create a subset only when the user
explicitly requests specific subjects. If one subject fails, report that
best-effort analysis failure rather than silently rerunning with a narrower
list.

## Output

After setup and launch, report:
- experiment name
- rewritten `log_dir`
- copied preset path
- the `--dry-run` plan summary (run count)
- exact launcher command used
- MoE routing source: the selected corpus or popularity file, or the reason for uniform fallback
- analysis selection (`all applicable` by default, or the user-requested subset),
  and whether PNGs were rendered (`--no-plot`)
- any analyzer subjects that failed best-effort post-run analysis

When the result reaches `ready`, switch to
`skills/operate-use-analyzer/SKILL.md` before interpreting or reporting result
values. Use the stable Analyzer experiment ID returned by this managed run; do
not rediscover a different similarly named sweep unless this ID is unavailable.
Read the exact Analyzer resource in the current turn and copy its value-adjacent
citation tokens into the answer. Launcher output and artifact paths establish
execution provenance, but they are not substitutes for Analyzer result values.
