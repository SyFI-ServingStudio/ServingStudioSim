---
name: operate-run-simulation
description: Use when the user wants to run an MLSim simulation from a preset (YAML preferred, JSON also accepted). This skill locates the repo logs root, names the experiment as date_index_short_name, prefers expressing requested variations as internal sweeps inside one copied preset, always does a dry run to inspect the expanded run plan before launching, and runs through the MLSim launcher.
---

# Run MLSim Simulation

Prepare and run an MLSim simulation from an existing preset. This is the
MLSim analogue of the global `run-moesim-experiment` skill — same shape (copy a
preset into a dated experiment dir, encode variations as sweeps, launch), but
the launcher and conventions below are MLSim-specific. When in doubt about the
preset/sweep philosophy, defer to `run-moesim-experiment`.

**Preset format is the nested config tree** (`deployment` / `workload` / `io` /
`pools`, plus control blocks), authored in **YAML** (JSON also parses). The
authoritative, worked reference for the format — the tree skeleton, every sweep
mechanism, and the validation rules — is `launcher/README.md`. This skill is the
run *workflow*; defer to that README for the preset *grammar*.

Repo root:
- `/m-coriander/coriander/kanzhu/MLSim_workspace/main`

Canonical logs root:
- `/m-coriander/coriander/kanzhu/MLSim_workspace/main/logs`

Launcher (run from the repo root):
- `uv run python -m launcher <preset>.yaml [--dry-run] [--override path=value ...]`

`uv run` is mandatory — it pins the PyO3-embedded interpreter to the 3.12 venv;
a bare `python` / `cargo` links the system 3.9 and crashes (see memory
`mlsim_pyo3_build_python_pin`). The launcher builds the binary + schema itself,
so you never set `LD_LIBRARY_PATH`/`PYTHONPATH` by hand.

## Trigger

Use when the user wants to run a named MLSim simulation, gives a preset to
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
   `compound` / `derived` control blocks (see Sweeps below).
8. **Dry-run first** — validate + expand without launching, and inspect the run
   plan and per-run `log_dir`s:
   ```bash
   cd /m-coriander/coriander/kanzhu/MLSim_workspace/main
   uv run python -m launcher logs/<experiment-name>/preset.yaml --dry-run
   ```
   Confirm the `[plan] N run(s)` count and that each expanded `log_dir` is
   distinct and lands under `logs/<experiment-name>/`. Fix the preset and
   re-dry-run until the plan is right.
9. Launch (drop `--dry-run`):
   ```bash
   uv run python -m launcher logs/<experiment-name>/preset.yaml
   ```

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
   cd /m-coriander/coriander/kanzhu/MLSim_workspace/main
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

## Output

After setup and launch, report:
- experiment name
- rewritten `log_dir`
- copied preset path
- the `--dry-run` plan summary (run count)
- exact launcher command used
