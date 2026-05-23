# MLSim Launcher (L7-α)

Python orchestration for sim runs: turn a **preset JSON** into one or more
reproducible simulator invocations. This is the practical preset-authoring
reference. The authoritative spec is `docs/detailed_design/L7/design.md` §1; if
this file disagrees with the code, the code (and design.md) win — open an issue.

## CLI

```
python -m launcher <preset.json> [<preset2.json> ...] [--override k=v ...] [--dry-run] [--refresh] [--build-type debug|release]
python -m launcher list-params [--human]
```

- Multiple presets run as one batch. Single vs sweep is decided by how many
  param sets the preset(s) expand to.
- `--dry-run` validates + expands + prints the resolved param sets; spawns nothing.
- `--refresh` re-runs every run. By default the launcher **resumes**: a run that
  finished writes a `.complete` marker into its `log_dir`, and re-invoking the
  same preset skips any run already marked complete (a crashed run leaves no
  marker, so it is retried). Aggregation still covers the whole sweep.
- `list-params` prints the deployment schema (Rust-authoritative). `--human` for a table.

The set of legal params per deployment is **not defined here** — it comes from
the Rust binary via `simulator list-params` (`target/<build_type>/deployment_schema.json`).
Run `python -m launcher list-params --human` to see every param, type, default,
and description for each deployment.

## Preset structure

A preset is a flat JSON object: `deployment` + concrete param values, plus three
optional sweep control blocks.

```jsonc
{
  "deployment": "unified",                 // required; picks which params are legal
  "model_config": "model/config/llama3_8b.json",
  "tp_size": 4,                            // concrete scalar
  "request_rate": 10.0,
  "log_dir": "logs/{model}_tp{tp}",        // template (see below)

  "sweep_groups": { ... },                 // optional — zipped multi-field dims
  "derived":      { ... },                 // optional — computed assignments
  "constraints":  [ ... ]                  // optional — boolean filters
}
```

## Sweep dimensions

Whether a value means "a sweep" or "a literal value" is decided by the param's
**schema type**, never by syntax:

| Preset form | Param's schema type | Meaning | Example |
|---|---|---|---|
| scalar | any scalar | one value | `"tp_size": 4` |
| **list** | scalar (`int`/`float`/`string`/`bool`/`path`) | **sweep dim** (unlabeled) | `"tp_size": [1, 2, 4]` |
| **dict** | scalar | **labeled sweep dim** (keys = labels for `log_dir`) | `"trace_files": {"low": "u_10.csv", "high": "u_100.csv"}` |
| list | collection (`int_list`/`path_list`/...) | the list **is the value**, NOT a sweep | `"trace_files": ["a.csv", "b.csv"]` |

Independent sweep dims multiply (cartesian product).

### `sweep_groups` — zipped multi-field dims

Each group is one dimension; each entry sets several fields **together** (zip,
not product). Use when params must move in lockstep.

```jsonc
"sweep_groups": {
  "parallelism": [
    {"tp_size": 1, "head_parallel": 1},
    {"tp_size": 2, "head_parallel": 2},
    {"tp_size": 4, "head_parallel": 4}
  ]
}
```
→ 3 runs (not 9). A group dim still multiplies against other independent dims.

A group's entries may instead be a **dict**, whose keys become the dimension's
labels (the list form labels by index: `0`, `1`, …). Use this when you want
readable per-run output dirs, e.g. for an A/B over flags that span several params:

```jsonc
"log_dir": "bench/rate_{rate}",
"sweep_groups": {
  "rate": {
    "low":  {"request_rate": 50.0},
    "high": {"request_rate": 300.0}
  }
}
```
→ 2 runs into `bench/rate_low`, `bench/rate_high`. An empty entry (`{}`) is the
all-defaults baseline. The zipped fields apply identically to the list form; only
the label (and thus the `{rate}` log_dir placeholder) differs.

### `derived` — computed assignments

Assign a param from an expression over other params. Evaluated **per candidate**,
in declaration order, so a later `derived` may reference an earlier one.

```jsonc
"derived": { "head_parallel": "tp_size", "nvl_num_gpu": "head_parallel * 2" }
```

- LHS must be a **declared schema param** (no auxiliary names).
- LHS must not also be a sweep dim / `sweep_groups` field.
- `derived` is the **only** way to introduce a computed value; it keeps the
  search space the size of the actual sweep dims (unlike adding a `==` constraint).

### `constraints` — boolean filters

Drop any candidate where any constraint is false (silent — not an error).

```jsonc
"constraints": ["tp_size * ep_size <= 32", "ep_size % tp_size == 0"]
```

Constraints only **filter**; they never introduce a new name.

## Expression grammar (`derived` RHS + `constraints`)

Expressions are evaluated in a sandbox (`asteval`, never `eval`). The validator
rejects anything the evaluator can't run, so a preset never passes validation
then silently produces `None`.

**Allowed:**
- **Names** — any param available at that point: base values, defaulted params
  (e.g. `ep_size` even if unset), sweep-dim values, `sweep_groups` fields, and
  earlier `derived` results.
- **Numbers / strings / booleans** literals.
- **Arithmetic:** `+ - * / // % **`, unary `-`/`+`.
- **Comparison:** `< <= > >= == !=` (chained allowed).
- **Boolean:** `and` `or` `not`.
- **Ternary:** `a if cond else b`.
- **Function calls — allow-list only:** `min`, `max`, `abs`.

**Not allowed** (rejected at validation): any other function call (`open`,
`range`, `len`, ...), attribute access (`x.foo`), subscript (`x[0]`), list/dict
literals, comprehensions, lambdas, walrus, f-strings.

A runtime failure inside an allowed expression (e.g. division by zero) raises a
clear error rather than yielding `None`.

> `derived x = y` vs `constraint x == y`: use `derived` to *link* params (keeps
> the sweep small); use `==` in a constraint only to *filter* over values that
> are already swept.

## `log_dir` templating

`log_dir` may contain `{...}` placeholders, resolved (in priority order) from
sweep labels, then param values, then these aliases:

| Placeholder | Resolves to |
|---|---|
| `{model}` | `model_config` filename stem |
| `{tp}` | `tp_size` |
| `{ep}` | `ep_size` |
| `{rate}` | `request_rate` |
| `{<any_param>}` | that param's value |
| `{<dict_sweep_label>}` | the chosen label of a labeled sweep dim |

Unknown placeholders are left untouched and emit a warning, because they often
mean a misspelled sweep dim in the output path. Example:
`"logs/{model}_tp{tp}_r{rate}"` → `logs/llama3_8b_tp4_r10.0`.

## `--override key=value`

Patches a preset from the CLI. Values are parsed as JSON when possible
(`tp_size=4`→int, `request_rate=2.5`→float, `fp8=true`→bool, `null`→None,
`[1,2]`→list); anything not valid JSON stays a bare string (`cp_plan=ring`,
`model_config=model/x.json`).

## Expansion pipeline & validation order

`validate_params` (fails fast, lists all errors) → `expand_sweep_params`
(5 steps: identify dims → cartesian product → apply `derived` → apply
`constraints` → stash labels) → `normalize_params` (type-coerce + fill defaults)
→ `_format_log_dir`.

Static checks run before expansion (design §1.2.1.1 V1–V6):

| # | Check |
|---|---|
| V1 | `derived` LHS is a declared schema param |
| V2 | a `derived` LHS is not also a sweep dim / `sweep_groups` field |
| V3 | `derived` RHS names resolve in topological order |
| V4 | every required param has a value after `derived` |
| V5 | `constraints` free vars are all defined |
| V6 | expression grammar is within the allowed set (above) |

Rust-owned ParamDef `choices` are also checked here, including scalar values,
list sweeps, and labeled dict sweeps.

## Output layout

Single run → `<log_dir>/` with shared invocation metadata (`preset.json`,
`git_snapshot/`) plus per-run metadata (`raw/params.json`, `manifest.json`,
`raw/command.txt`), stable artifact buckets (`raw/`, `plots/`, `reports/`,
`payloads/`, `traces/`), `stdout.log`, a `.complete` marker (written only on a
zero exit; drives `--refresh`/resume), and the rust binary's outputs. Sweep →
an experiment root holding the shared `preset.json`, `git_snapshot/`,
`sweep_summary.csv` / `plots/` (aggregator), and one subdir per run. Sweep run
dirs contain only run-specific metadata and outputs, not repeated git/preset
copies. Expanded sweep runs must have unique `log_dir` values; duplicates abort
the plan before dry-run output, resume filtering, cache prebuild, or launch.

## Worked example — every mechanism at once

This preset exercises all of: `sweep_groups` (zipped, dict-labeled), an
independent `list_sweep`, a labeled `dict_sweep`, a collection-typed list-value
(NOT swept), `derived` (function call + ternary), a `constraint` filter, and a
templated `log_dir`.

```jsonc
{
  "deployment": "unified",
  "model_config": "model/config/llama3_8b.json",
  "trace_files": ["a.csv", "b.csv"],            // path_list → value, NOT a sweep
  "ep_size": [4, 8],                            // list_sweep (independent dim)
  "request_rate": {"lo": 1.0, "hi": 100.0},     // dict_sweep (labeled dim)
  "sweep_groups": {
    "par": {                                    // zipped dim, dict-labeled: tp & hp move together
      "tp2hp2": {"tp_size": 2, "head_parallel": 2},
      "tp4hp1": {"tp_size": 4, "head_parallel": 1}
    }
  },
  "derived": {
    "nvl_num_gpu": "max(tp_size, 2)",                       // allow-listed call
    "attn_gpu_memory_gb": "80.0 if ep_size <= 4 else 160.0" // ternary over a swept param
  },
  "constraints": ["tp_size * ep_size <= 16"],   // filter
  "log_dir": "logs/{model}/tp{tp}_ep{ep}_{request_rate}"
}
```

**Dimensions** → cartesian product `par(2) × ep_size(2) × request_rate(2) = 8`
candidates (`trace_files` is a value, not a dim). The `constraint` drops the two
`tp=4, ep=8` candidates (`32 > 16`), leaving **6 runs**:

```
 tp  hp  ep   rate  nvl  attn_gb  labels                              log_dir
  2   2   4    1.0    2     80.0   {request_rate: lo, par: tp2hp2}  logs/llama3_8b/tp2_ep4_lo
  4   1   4    1.0    4     80.0   {request_rate: lo, par: tp4hp1}  logs/llama3_8b/tp4_ep4_lo
  2   2   4  100.0    2     80.0   {request_rate: hi, par: tp2hp2}  logs/llama3_8b/tp2_ep4_hi
  4   1   4  100.0    4     80.0   {request_rate: hi, par: tp4hp1}  logs/llama3_8b/tp4_ep4_hi
  2   2   8    1.0    2    160.0   {request_rate: lo, par: tp2hp2}  logs/llama3_8b/tp2_ep8_lo
  2   2   8  100.0    2    160.0   {request_rate: hi, par: tp2hp2}  logs/llama3_8b/tp2_ep8_hi
```

How each column arises:
- **`hp` tracks `tp`** (2↔2, 4↔1) because both come from the same `sweep_groups`
  entry — zipped, not multiplied against `ep`/`rate`.
- **`nvl` = `max(tp, 2)`** → 2 when tp=2, 4 when tp=4 (`derived`, function call).
- **`attn_gb`** = 80 when ep≤4 else 160 (`derived`, ternary over the swept `ep_size`).
- **No `tp4_ep8` row** — the constraint removed those two candidates.
- **`{request_rate}` in `log_dir` resolves to the label `lo`/`hi`**, not `1.0`/`100.0`,
  because `_format_log_dir` checks sweep labels first. `{ep}` has no label (it's a
  `list_sweep`), so it falls back to the value `4`/`8`. Use the dict form when you
  want a readable label in the path.

**Cache builds:** `request_rate` is not in the cache key, so the 6 runs collapse
to **3** `build-cache-only` passes — one per distinct `(tp, ep)`: `(2,4)`, `(4,4)`,
`(2,8)` — each reused by its `lo` and `hi` run.

## Handoff to the aggregator

After a sweep, the launcher calls `aggregate_sweep(run_infos, base_dir, groups=groups)`.
The sweep mechanisms above are **compiled away** by expansion — the aggregator
never sees `sweep_groups` / `derived` / `constraints`; it gets a flat product.
Each run is one row of the **axis-coordinate → output-folder** map:

```python
run_infos[i] = {
    "log_dir": "<absolute path>",        # this run's folder (absolute, == base_dir's root)
    "sweep":   {axis: real_value, ...},  # the run's coordinate
    "labels":  {dim: label, ...},        # readable ticks (dict_sweep key, group entry index)
}
groups = {"par": ["head_parallel", "tp_size"]}   # zip-bound columns → one composite axis
```

Concrete example after expansion:

```python
run_infos = [
    {
        "log_dir": "/abs/logs/ep32/par_0/rr_lo",
        "sweep": {
            "ep_size": 32,
            "tp_size": 1,
            "head_parallel": 1,
            "request_rate": 1.0,
        },
        "labels": {"par": "0", "request_rate": "lo"},
    },
    {
        "log_dir": "/abs/logs/ep32/par_1/rr_hi",
        "sweep": {
            "ep_size": 32,
            "tp_size": 2,
            "head_parallel": 2,
            "request_rate": 100.0,
        },
        "labels": {"par": "1", "request_rate": "hi"},
    },
]
groups = {"par": ["head_parallel", "tp_size"]}
```

- **What counts as an axis** is mechanism-agnostic: a param is an axis iff it
  takes more than one distinct value across the runs. So `list`/`dict` sweeps,
  `sweep_groups` fields, and varying `derived` results all show up as axes;
  constant params and `log_dir` are excluded.
- The full `run_infos` list **is** the axis→folder mapping — the aggregator reads
  metrics from each `log_dir` and pivots on `sweep`; it never reconstructs a path
  from a coordinate. `log_dir` is absolute so it joins unambiguously under the
  (also resolved) `base_dir`.
- `labels` carry the readable tick for labeled dims (e.g. `request_rate`'s
  `lo`/`hi`), since `sweep` holds the numeric value (`1.0`/`100.0`).
- `groups` lets the aggregator treat zip-bound columns as a single composite axis
  rather than a sparse `tp × hp` grid. It is a required kwarg in the launcher ↔
  aggregator contract; there is no fallback to an older two-arg aggregator.
- The aggregator is analyzer-owned and optional here: if it is not importable the
  sweep still completes and the summary step is skipped.
