---
name: run-simulation
description: Use when the user wants to run an MLSim simulation from a preset JSON. This skill locates the repo `logs/` root, names the experiment as `YYYYMMDD_N_<short-name>` (date, then a per-day index, then a name of 5 words or fewer), prefers expressing requested variations as internal sweeps inside one copied preset, always does a `--dry-run` to inspect the expanded run plan before launching, and runs through the MLSim launcher with `uv run python -m launcher`.
---

# Run MLSim Simulation

Prepare and run an MLSim simulation from an existing preset JSON. This is the
MLSim analogue of the global `run-moesim-experiment` skill — same shape (copy a
preset into a dated experiment dir, encode variations as sweeps, launch), but
the launcher and conventions below are MLSim-specific. When in doubt about the
preset/sweep philosophy, defer to `run-moesim-experiment`.

Repo root:
- `/m-coriander/coriander/kanzhu/MLSim_workspace/main`

Canonical logs root:
- `/m-coriander/coriander/kanzhu/MLSim_workspace/main/logs`

Launcher (run from the repo root):
- `uv run python -m launcher <preset>.json [--dry-run] [--override k=v ...]`

`uv run` is mandatory — it pins the PyO3-embedded interpreter to the 3.12 venv;
a bare `python` / `cargo` links the system 3.9 and crashes (see memory
`mlsim_pyo3_build_python_pin`). The launcher builds the binary + schema itself,
so you never set `LD_LIBRARY_PATH`/`PYTHONPATH` by hand.

## Trigger

Use when the user wants to run a named MLSim simulation, gives a preset JSON to
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

1. Identify the preset JSON to run (under `presets/`, or one the user names).
2. Decide the experiment name per the rules above (ask if not given).
3. Decide whether the requested variations fit one preset as internal sweeps —
   default to "yes" whenever the schema supports it (see Sweeps below). Only
   create multiple experiment folders / multiple presets when one preset cannot
   express the runs cleanly, or the user explicitly asks.
4. Compute the experiment dir: `logs/<experiment-name>`.
5. If the dir already exists, stop and ask whether to reuse it or rename. Do not
   silently overwrite an existing copied preset.
6. Copy the chosen preset to `logs/<experiment-name>/preset.json`. Do not modify
   the original preset under `presets/` unless the user asks.
7. Edit the copied preset's internal `log_dir` to `logs/<experiment-name>`,
   preserving any template tail / placeholders (see log_dir Rewrite). Encode the
   requested variations as sweep lists/dicts/`sweep_groups`.
8. **Dry-run first** — validate + expand without launching, and inspect the run
   plan and per-run `log_dir`s:
   ```bash
   cd /m-coriander/coriander/kanzhu/MLSim_workspace/main
   uv run python -m launcher logs/<experiment-name>/preset.json --dry-run
   ```
   Confirm the `[plan] N run(s)` count and that each expanded `log_dir` is
   distinct and lands under `logs/<experiment-name>/`. Fix the preset and
   re-dry-run until the plan is right.
9. Launch (drop `--dry-run`):
   ```bash
   uv run python -m launcher logs/<experiment-name>/preset.json
   ```

## Sweeps

Variations are expressed inside the preset, expanded by the launcher (no
launcher-side sweep flags). Driven by the Rust schema param type, not syntax:

- **scalar param + list value** → swept (`"tp_size": [1, 2, 4]`).
- **scalar param + dict value** → labeled sweep; the keys become readable
  `log_dir` labels (`"cost_tree": {"fold": false, "tree": true}`).
- **`*_list` param + list value** → that list is the *value*, never a sweep
  (e.g. `trace_files`).
- **`sweep_groups`** → zip several fields that must move together into one
  composite dim (e.g. `tp_size` + `head_parallel`).
- **`derived`** / **`constraints`** → compute dependent params and prune invalid
  combinations.

Prefer one preset with internal sweeps over many sibling experiment folders.

## log_dir Rewrite

When rewriting `log_dir`, keep the existing path shape; replace only the
experiment-name segment.

- `logs/<old-name>` → `logs/<experiment-name>`
- `logs/<old-name>/{model}_{tp}` → `logs/<experiment-name>/{model}_{tp}`
- missing `log_dir` → `logs/<experiment-name>`

For a sweep, extend the path with placeholders so each expanded run gets a
distinct subdir. Available placeholders: any param value, sweep labels (the dict
keys / group labels above), and aliases `{model}` (model_config stem), `{tp}`,
`{ep}`, `{rate}`. Unknown placeholders stay literal and warn — usually a
misspelled sweep dimension. The launcher rejects a sweep whose runs collide on
`log_dir`, so make the template cover every swept axis.

Example sweep preset:
```json
{
  "deployment": "unified",
  "model_config": "model/config/llama3_8b.json",
  "trace_files": ["trace/aime_long.csv"],
  "tp_size": [1, 2, 4],
  "request_rate": [50.0, 150.0],
  "log_dir": "logs/20260523_0_llama3_tp_rate/tp{tp}/rate{rate}"
}
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
