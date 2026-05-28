# MLSim Launcher (L7)

The **run interface**: turn a **preset** (a JSON/YAML config) into one or more
reproducible invocations of the Rust `simulator` binary. The launcher owns
everything *around* a run — building the binary, validating the config against
the binary's own schema, expanding sweeps, prebuilding the kernel cache, writing
repro metadata, spawning runs in parallel, and kicking off post-run analysis. It
contains **no simulation logic**; the Rust binary does the actual work.

This is the practical, code-matching reference. The authoritative spec is
`docs/detailed_design/L7/design.md` (and `docs/new-interface-design.md` for the
config-tree shape); if this file disagrees with the code, the code (and the
design docs) win — open an issue.

## What it exposes / what it requires

**Exposed upward** — two equivalent surfaces:

- **CLI** (`python -m launcher`, see bottom) for humans, skills, and tests.
- **Importable API** (`launcher/__init__.py`) for callers that construct runs in
  process: `load_schema`, `validate_params`, `expand_sweep_params`,
  `normalize_params`, `build_cli_command`, `run_single`, `run_sweep`. Per design
  INV-1, *everyone* goes through `launcher.schema` rather than hand-rolling argv —
  this is the single place a preset becomes a command line.

**Required from below:**

- A buildable **`simulator` Rust crate**. The launcher shells `cargo build`, and
  on success runs `simulator list-params` to discover the **param schema** — the
  launcher never hardcodes param data (it is Rust-authoritative).
- The PyO3 env the binary needs to embed Python (it calls L1 `profiling.perf_api`
  for kernel times). `exec._build_subprocess_env` wires `LD_LIBRARY_PATH` /
  `PYTHONHOME` / `PYTHONPATH` so callers never set them by hand.
- For real runs: GPUs + a warm/warmable `profile.db` (the launcher prebuilds it).
- Optionally the standalone **`analyzer`** crate + its Python renderer (post-run
  analysis is best-effort; a missing analyzer never fails a run).

## Directory map

```
__main__.py    CLI entry. Thin dispatch only: parse argv, apply --override,
               build once, then validate → expand → (dry-run | cache-report |
               run_single | run_sweep). No sim logic.
__init__.py    The importable public API (re-exports from schema/ + sweep).

schema/        The single source of param truth on the Python side. Per-param
  __init__.py    DATA is Rust-authoritative; this package is the LOGIC over it.
  loader.py      Read target/<build>/deployment_schema.json into a `Registry`;
                 walk a concrete config tree against it (`iter_slots`,
                 `unknown_keys`). Knows the fixed tree skeleton.
  validate.py    Static preset validation → list of human-readable errors.
  expand.py      sweep → cartesian product → derived → constraints → ${name}
                 substitution → normalize (coerce + defaults) → log_dir template.
  expr.py        Sandboxed `derived`/`constraint` evaluation (asteval + an ast
                 grammar gate that stays in lock-step with it).
  argv.py        Concrete config tree → config file on disk + [binary, sub, path].

exec.py        cargo_build (+ schema discovery, INV-8), the PyO3 subprocess env,
               async `SimulationRunner` (one run subprocess), and `run_analysis`.
sweep.py       Top-level flow: `run_single` / `run_sweep`. Resume markers,
               parallel launch under a semaphore, sweep-axis aggregation contract.
cache_build.py Prebuild profile.db per unique cache key, sequentially (INV-3),
               so parallel runs don't race the GPU / SQLite. `--cache-report`.
metadata.py    Repro metadata written BEFORE spawn (INV-2): preset.json,
               git_snapshot/, raw/params.json, command.txt, manifest.json.
```

## The pipeline (one batch, start to finish)

`__main__.main` is the spine; everything below it is a stage:

1. **Build once** (`exec.cargo_build`). Compiles `simulator`, then runs
   `list-params` → `target/<build>/deployment_schema.json`. Schema discovery is
   *part of* the build (INV-8), so the schema always matches the just-built binary.
   Even `--dry-run` builds, because it validates against this schema.
2. **Load schema** (`schema.loader.load_schema`) → a `Registry`. Absent file =
   `SchemaNotFound` with build instructions, never a Python fallback.
3. **Per preset:** apply `--override`, pop launcher-only keys (e.g.
   `analyze_subjects`), then **validate** (`validate_params`) → **expand**
   (`expand_sweep_params`) → **normalize** (`normalize_params`) → **template the
   log_dir** (`_format_log_dir`). Output: a flat list of concrete config trees.
4. **Guard:** reject duplicate `log_dir`s across the whole plan
   (`validate_unique_log_dirs`).
5. **Branch:** `--dry-run` prints the expanded configs and stops; `--cache-report`
   probes coverage and stops; otherwise hand off to `sweep`.
6. **Run** (`sweep.run_single` / `run_sweep`): prebuild caches → write metadata →
   spawn the binary → mark `.complete` → best-effort analyze. A sweep does this
   across many configs in parallel under a semaphore.

## Preset shape (matches the code, not the old flat format)

A preset is a **nested config tree** plus up to three launcher control blocks.
The tree skeleton is fixed launcher knowledge; the *fields* inside each node come
from the Rust schema.

```yaml
deployment: unified              # required; selects which pool roles exist
workload:                        # run-global workload params
  trace_files: ["trace/smoke.csv"]
io:                              # run-global output + logging
  log_dir: "logs/tp{tensor_parallel}"   # {name} drops a sweep value into the path
pools:
  main:                          # role name (fixed by the deployment)
    groups:
      - gpu: "NVIDIA H200"
        replicas: 1
        arch:                    # provider tag + its own params
          type: llama3_dense_tp  # the TP arch — it is what exposes tp_size
          model_config: model/config/llama3_8b.json
          tp_size: ${tensor_parallel}   # injected (typed int) by the sweep below
        worker:                  # sibling provider tag + its own params
          type: barebone

sweep:                           # optional: independent dims → cartesian product
  tensor_parallel: [1, 2, 4]
# derived: and constraints: are also available — see the worked example below.
```

- **Where params are legal** is decided by `deployment` → pool roles → each
  group's `arch.type` / `worker.type` (the "contract" + provider tags). Run
  `python -m launcher list-params --human` to see every legal field.
- **Unknown keys anywhere are rejected** (`unknown_keys`): a Rust tagged enum
  silently ignores unknown fields, so arch/worker typos must be caught here.

### Sweeps: `${name}` placeholders, not list-typed leaves

Unlike the old flat format, a sweep is **named** in the `sweep` block and
**referenced** by placeholder:

- **`${}` varies a VALUE, never structure (params-only).** A `${name}` may only
  stand for a value leaf (scalar or list). It may **not** replace structure — the
  `deployment`, a provider `type` tag, a whole `arch`/`worker` block, or a pool —
  because the tag/shape selects which schema applies and must be statically known.
  Structural variation across runs is expressed by **separate presets** (a
  `variants` manifest, below), not by a placeholder.
- **Name sweeps/derived/compound for what they mean** (`prefill_tp`, not `ptp`):
  the name is user-facing — it surfaces in `log_dir`, as a sweep aggregation axis,
  and in validation error messages, so a meaningful name pays off everywhere.
- `${name}` as a whole leaf anywhere in the tree → typed substitution.
- `{name}` inside `io.log_dir` → string templating; a field must be a **bare
  identifier** (no `{x.attr}` / `{x[0]}` / `{x!r}` / `{x:spec}`). Dict-sweep /
  compound labels win, then the resolved value; unknown placeholders warn + stay
  literal.
- `sweep` values: a **list** is an unlabeled dim; a **dict** `{label: value}`
  gives readable per-run log dirs. Independent dims form a cartesian product.
- **Correlated columns** have two forms: `derived` (one name moving by a *formula*
  over its inputs) and **`compound`** (an explicit *tabular* zip — see below).
- `derived` / `constraints` are evaluated in a sandbox (`asteval`): names,
  numbers/strings/bools, `+ - * / // % **`, comparisons, `and/or/not`, ternary,
  and only `min`/`max`/`abs` calls. The static grammar gate rejects anything the
  evaluator can't run, so a preset never validates then silently yields `None`.

### `compound`: named zip groups (correlated columns, no cartesian blow-up)

A `compound` group zips several members so they move **together** — not as a
cartesian product. The **group name is one aggregation axis**; each labeled row
binds all its members at once:

```yaml
compound:
  tp_rate:                          # group name = one axis (ticks: fast / slow)
    fast: { tp: 4, rate: 10 }
    slow: { tp: 8, rate: 20 }
# tp_size: ${tp}, request_rate: ${rate}   →  (4,10) and (8,20), NOT a 2×2 grid
```

- Each row is `{label: {member: value}}` — same `{label: …}` shape as a dict-sweep,
  extended to multiple members. The label is the axis tick + `{tp_rate}` in log_dir.
- Members fill params via `${member}`; a group is **one cartesian factor** (crossed
  with `sweep` dims and other groups).
- Rules: every row of a group declares the **same member set**; group + member
  names are **globally unique**; every member must be **config-effective** (reach a
  config leaf, like a sweep dim). `_sweep_axes` folds members into the group axis,
  so the correlated pair is one axis, never a fake `tp × rate` grid.

### `variants`: a single-axis manifest for *structural* comparison

Different `arch.type` / pool shapes / deployments are **structure**, so you compare
them across **separate presets**, indexed by a manifest (a file with no
`deployment:`, exactly **one** named file axis):

```yaml
# arch_compare.yaml — run as `python -m launcher arch_compare.yaml`
variants:
  arch:                                  # axis name = aggregation axis (ticks dense/moe)
    dense: presets/unified_dense.yaml    # each label → a complete standalone preset
    moe:   presets/unified_moe.yaml
```

Each referenced preset keeps its own params-only `sweep`/`compound`; the launcher
runs them as one batch, tags every run's `_env` with `{arch: <label>}` (a named
axis crossed with each file's internal axes), and prefixes each run's `log_dir`
with the label so cross-file runs never collide. **Strictly one axis** — a
multi-axis manifest (which would need fragment merging) is rejected.

### Worked example: every technique in one preset

A `pd` (prefill/decode) sweep that exercises a dict-sweep, a list-sweep, a
**`compound` group** (correlated `max_batch_tokens` + `batch_policy`), a `derived`
column, a `constraint`, typed `${name}` injection across **both** pools, a
templated `log_dir`, two different worker tags, and the launcher-only
`analyze_subjects` key (YAML form):

```yaml
deployment: pd

workload:
  trace_files:
    - trace/aime_long.csv
  duration_ms: 20000.0
  run_to_end: true
  request_rate: 12.0

io:
  log_dir: "logs/pd_{prefill_tp}_d{decode_tp}tp_r{decode_replicas}_{batch}"   # {prefill_tp}/{batch} = labels; the rest stringify their values
  log_level: info
  quiet: false
  force_cache_build: false

pools:
  prefill:
    placement: least-queued
    groups:
      - gpu: "NVIDIA H200"
        replicas: 2
        arch:
          type: llama3_dense_tp
          model_config: model/config/llama3_8b.json
          fp8: true
          tp_size: ${prefill_tp}
        worker:
          type: chunked_prefill
          attn_gpu_memory_gb: 80.0
          max_batch_tokens: ${max_batch_tokens}   # from the `batch` compound group
          batch_policy: ${batch_policy}            # …moves together with it
  decode:
    placement: round-robin
    groups:
      - gpu: "NVIDIA H200"
        replicas: ${decode_replicas}
        arch:
          type: llama3_dense_tp
          model_config: model/config/llama3_8b.json
          fp8: true
          tp_size: ${decode_tp}
        worker:
          type: barebone
          attn_gpu_memory_gb: 80.0

sweep:
  prefill_tp:            # dict-sweep → readable labels p8 / p4
    p8: 8
    p4: 4
  decode_tp: [2, 4, 8]   # list-sweep → unlabeled, value used directly
compound:
  batch:                 # correlated knobs move together (one axis, ticks big/small)
    big:   { max_batch_tokens: 16384, batch_policy: separate-prefill-priority }
    small: { max_batch_tokens: 8192,  batch_policy: mix }
derived:
  decode_replicas: "16 // decode_tp"   # decode replicas track a ~16-GPU budget (replicas * tp)
constraints:
  - "prefill_tp >= decode_tp"          # prefill must shard at least as fine as decode
analyze_subjects:                      # launcher-only; popped before validation
  - throughput
  - latency
```

> **Name sweeps/derived for what they mean** (`prefill_tp`, not `ptp`). That name
> is not private bookkeeping — it surfaces verbatim in `log_dir` templates, as a
> **sweep aggregation axis**, and in validation error messages. A cryptic `ptp`
> makes every one of those harder to read; spell it out.

This expands to **10 runs** — the 2×3×2 cartesian product (prefill_tp × decode_tp ×
`batch`) minus the two combos `prefill_tp >= decode_tp` rejects (`prefill_tp=4,
decode_tp=8`, for both `batch` rows):

| prefill_tp | decode_tp | decode_replicas | batch | log_dir |
|:----------:|:---------:|:---------------:|:-----:|------------------------------|
|     8      |     2     |        8        | big   | `logs/pd_p8_d2tp_r8_big`     |
|     8      |     2     |        8        | small | `logs/pd_p8_d2tp_r8_small`   |
|     8      |     4     |        4        | big   | `logs/pd_p8_d4tp_r4_big`     |
|     8      |     4     |        4        | small | `logs/pd_p8_d4tp_r4_small`   |
|     8      |     8     |        2        | big   | `logs/pd_p8_d8tp_r2_big`     |
|     8      |     8     |        2        | small | `logs/pd_p8_d8tp_r2_small`   |
|     4      |     2     |        8        | big   | `logs/pd_p4_d2tp_r8_big`     |
|     4      |     2     |        8        | small | `logs/pd_p4_d2tp_r8_small`   |
|     4      |     4     |        4        | big   | `logs/pd_p4_d4tp_r4_big`     |
|     4      |     4     |        4        | small | `logs/pd_p4_d4tp_r4_small`   |

`${prefill_tp}` / `${decode_tp}` land as **ints** in each pool's `arch.tp_size`;
`${decode_replicas}` (a `derived` int) lands in `decode`'s `replicas`; the `batch`
group's `${max_batch_tokens}` + `${batch_policy}` land together in the prefill
worker (they co-vary, so `big`/`small` is one axis, not a 2-way cross). In
`log_dir`, `{prefill_tp}` / `{batch}` resolve to their labels (`p8` / `big`), while
`{decode_tp}` / `{decode_replicas}` stringify their values — note `{batch}` is
needed to keep the two batch rows' dirs distinct. Because only `fp8` + `tp_size`
are `affects_cache` (the `batch` params are runtime, not kernel-shaping), the
prebuild collapses these 10 runs to just the 5 distinct (prefill-tp, decode-tp)
cache keys (§Cache prebuild).

Patch any leaf from the CLI without editing the file — dotted path, JSON-typed
value:

```bash
python -m launcher pd_sweep.yaml \
  --override workload.request_rate=20 \
  --override pools.decode.groups.0.worker.attn_gpu_memory_gb=140
```

> `pd` *execution* is not wired yet (`build` bails); schema validation, sweep
> expansion, cache grouping, and metadata all work — this shows the config
> interface's full expressiveness, not a runnable deployment.

## Validation: what's accepted, what's rejected

Validation behaves like a **compiler front-end**: anything provably illegal,
suspicious, or no-op is rejected up front with a precise message — never left to
crash in `normalize`/`expand`, and never allowed to run as a wrong or duplicate
simulation. It is **two phase-bound judgments over one shared schema walk**
(`validate._walk_schema`: deployment → unknown keys → structure → per-leaf
type/choices/required), plus a set of cross-candidate plan guards. The two
judgments differ on exactly one axis — whether a whole-leaf `${name}` placeholder
defers its check or is checked concretely.

### Parse gate (load time, `__main__`)

Before validation, the loader is strict: a preset file's **root must be a mapping**
(an empty file / `[]` is rejected, not a traceback), **duplicate keys are rejected**
(`json`/PyYAML default to last-wins — that silent drop is an error here), and the
launcher-only `analyze_subjects` must be a **list of strings**.

### Raw, pre-expansion (`validate.validate_params`)

The shared walk with placeholders **deferred**, plus the control-block checks. A
whole-leaf `${name}` is "supplied later", so its type / choice / provider-tag is
postponed to the concrete phase; only required-presence is meaningful now.

| Class | Rejected when |
|---|---|
| Tag / key | unknown `deployment`; any unknown key in the tree (an arch/worker payload typo — the Rust tagged enum can't catch it); `arch.type` / `worker.type` not advertised for the pool's contract |
| Structural | a required pool role is missing; a pool has no non-empty `groups`; a group / `arch` / `worker` is not a mapping |
| Type | a concrete leaf value is not its ParamDef type — `tp_size: "x"`, `trace_files: "a.csv"` (scalar for a list), a `bool` given an int |
| Closed set | a concrete `choices` leaf holds a value outside the set |
| Required | a no-default, non-`optional` leaf is absent (even if its whole parent block was dropped) |
| Structure is literal (params-only) | a `${...}` placeholder on `deployment`, an `arch.type` / `worker.type` tag, or a whole `arch` / `worker` block — structure selects the schema and must be literal; sweep the tag's params, or split variants into separate presets |
| Control-block shape | `sweep` / `derived` / a `compound` group is not a mapping, `constraints` is not a list of strings, or a `compound` row is not a `{member: value}` mapping (a wrong shape must not silently degrade to a no-op — `sweep: []` is falsy — or crash downstream) |
| Expression | `derived` / `constraints` use a construct outside {arithmetic, comparison, boolean, ternary, `min`/`max`/`abs`}, or a free variable not from sweep / compound / an earlier derived |
| Sweep hygiene | partial `${...}` inside a larger string (R1); a sweep dim / compound member that is not **config-effective** — never reaching a run config value via `${name}` or a config-effective `derived` (R2; appearing only in `io.log_dir` / a constraint is a no-op); a `derived` name never used (R3); an empty sweep dim (R4); a `derived` that is **constant** — derives from no sweep dim (R7); a `${name}` / `io.log_dir` `{name}` that resolves to no declared name (P) |
| Block shape | `workload` / `io` present but not a mapping (`io: []` / `io: "x"`) — a non-dict would be silently rebuilt as the default block |
| Symbol names | a `sweep` dim, `derived` name, `compound` group, or `compound` member name that is not a string **identifier** (it becomes an `${name}` / `_env` key); a `sweep` dict-sweep label or `compound` row label that is not a non-empty path-safe string (it reaches `io.log_dir` as a path segment) |
| Compound group | rows of a group declare different member sets; a group/member name collides with a sweep dim, derived name, another group, or another group's member |
| log_dir template | an `io.log_dir` `{field}` that is not a bare `{name}` — attribute (`{x.__class__}`), index (`{x[0]}`), conversion (`{x!r}`), format spec (`{x:04d}`), or a malformed template |
| Manifest | a `variants` manifest with an unknown top-level key, not exactly one axis, a non-identifier axis name, an empty axis, a non-path-safe label, a non-string / missing preset path, or an axis name that collides with an inner sweep/compound/derived name of a referenced preset |

### Concrete, post-expansion (`validate.validate_expanded`)

The **complete** schema judgment and the single gate that says "this config is
safe to hand to the Rust binary" — the *same* shared walk, now with placeholders
**no longer deferred**, run on each expanded candidate **before `normalize`** (so
a coercion like `bool("false") == True` cannot hide an error). A candidate that
passes carries zero `${` anywhere, every provider tag resolved, every required
leaf present, every value the right type / an allowed choice.

| Class | Rejected when |
|---|---|
| Provider tag | a swept/derived `arch.type` / `worker.type` resolves to a tag not advertised for the contract — `type: ${k}` with `k="not_a_real_arch"` |
| Tag-specific required | with the tag concrete, one of *its* required params is absent — `worker.type` swept to `chunked_prefill` without `max_batch_tokens` |
| Type / choices | a swept/derived value is the wrong type or outside a closed set — `log_level: ${x}` with `x="verbose"`, a `"false"` string into a `bool` param |
| Unknown key | an unknown key in the concrete tree (re-run here so the gate is self-contained) |
| Placeholder residue | any `${...}` left in the tree after substitution — a whole-leaf one, or a partial `${...}` inside a larger string |

### Plan guards (cross-candidate, in `__main__`)

These compare runs across the whole expanded plan, so they live at the plan layer
rather than in either per-config judgment:

| Class | Rejected when |
|---|---|
| Distinct configs (R5) | two runs have identical config trees (differing only in `log_dir`) — a swept dim that changes no config value |
| Unique log_dirs | two runs would write to the same `log_dir` |
| Non-empty plan (R6) | expansion yields zero runs (empty dim, or constraints reject every combination) |

The raw and concrete judgments are duals over the same walk: raw catches what's
provable from the preset alone; concrete catches what only the fully-substituted
run reveals. An invariant test asserts the post-condition — any candidate the
concrete gate accepts has no placeholder residue.

**Not yet checked:** numeric range / sign (e.g. `replicas > 0`, `tp_size > 0`) —
this needs `min` / `max` on the Rust-owned ParamDef; not currently modeled.

## Cache prebuild (the contention fix)

A cache-miss run JIT-profiles a kernel and writes `profile.db`; N parallel runs
sharing a missing key would race the GPU (wrong timings) and the SQLite writer
(lock errors). So before any parallel launch, `cache_build.prebuild_caches` runs
`build-cache-only` **once per unique cache key, sequentially** (INV-3).

**Which params form the key is Rust-authoritative**: `cache_key` walks the tree
and collects every leaf whose ParamDef is tagged `affects_cache`. Runs differing
only in non-kernel params (request rate, replicas, `log_dir`, …) collapse to one
prebuild. `--cache-report` runs the binary's `dry-run` per key to show coverage
without building anything.

## Resume & output layout

- **Resume (default):** a zero-exit run writes a `.complete` marker into its
  `log_dir`; re-invoking the same preset skips runs already marked complete (a
  crash leaves no marker → retried). `--refresh` re-runs everything. Aggregation
  still spans the whole sweep.
- **Metadata before spawn (INV-2):** so a crashed run still has triage info.
  Single run → `<log_dir>/` with `preset.json`, `git_snapshot/`, `manifest.json`,
  `raw/{params.json,command.txt,run_config.yaml}`, the buckets `raw/ plots/
  reports/ payloads/ traces/`, `stdout.log`, and the `.complete` marker. Sweep →
  an experiment root holding the shared `preset.json` / `git_snapshot/` and one
  subdir per run (run subdirs carry only run-specific files, no repeated copies).

## Sweep aggregation handoff

After a sweep, `sweep._aggregate` calls the analyzer-owned `aggregate_sweep`.
**Sweep axes are mechanism-agnostic**: any `sweep`/`derived` name that takes more
than one distinct value across runs (read from each run's stashed `_env`) is an
axis; constants are excluded. Each run is one row of an axis-coordinate →
output-folder map:

```python
run_infos[i] = {
    "log_dir": "<absolute path>",     # this run's folder
    "sweep":   {axis: value, ...},    # its coordinate (from _env)
    "labels":  {dim: label, ...},     # readable ticks (dict-sweep keys)
}
aggregate_sweep(run_infos, base_dir, groups={})
```

The aggregator is optional: if it is not importable the sweep still completes and
the summary step is skipped.

## CLI

```
python -m launcher <preset.yaml> [<preset2> ...] [--override path=value ...]
                   [--dry-run] [--cache-report] [--refresh]
                   [--build-type <cargo-profile>] [--profile [--profile-freq HZ]]
                   [--no-analyze]
python -m launcher list-params [--human] [--build-type ...]
```

- Multiple presets run as one batch; single vs. sweep is decided by how many
  configs the preset(s) expand to.
- `--override path=value` patches the **nested** tree by dotted path
  (`io.log_dir`, `pools.main.groups.0.arch.tp_size`); the value is parsed as JSON
  when possible, else kept as a bare string.
- `--dry-run` validates + expands + prints the resolved configs; spawns nothing.
- `--cache-report` reports profile.db coverage per unique cache key, then exits.
- `--profile` wraps a single run with `perf record` (skill `profile-sim-speed`).
- `list-params` dumps the Rust-authoritative schema (`--human` for a table).

See skill `run-simulation` for preset/sweep authoring conventions and dated log
dirs.
