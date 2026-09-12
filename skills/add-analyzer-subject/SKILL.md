---
name: add-analyzer-subject
description: Use when adding a new VibeSim analyzer subject (a metric) end to end — the Rust compute side (a report + payload JSON built from a run's parquet via DataFusion) and the Python render side (a matplotlib PNG drawn from the payload). Covers the flat registry, the report/payload contract, the Applies gate, keeping a large run within the speed budget, and — emphatically — reusing the shared session / io / cdf infra and the Python common style / figure / layout helpers instead of re-implementing them. NOT for running an existing analyzer (that is run-moesim-analyzer) or adding an L1 kernel (that is top-add-kernel).
---

# Add Analyzer Subject

Add one analyzer **subject** — one metric → one `(report, payload)` pair → one or
more PNGs — across both halves of the boundary:

- **Rust** (`analyzer/rust/src/`) scans the run's parquet with DataFusion, does
  the math, and emits `reports/<name>_report.json` (numbers, for an LLM) plus
  `payloads/<name>_<shape>.json` (arrays, for the plot). It has **no `simulator`
  dependency**.
- **Python** (`analyzer/python/`) draws matplotlib over the payload JSON only. It
  **never opens parquet**.

The JSON pair is the only boundary. A subject lives in a **category** folder
(`request/`, `throughput/`, `utilization/`, …) — a category is a tag plus a source
family (`doc/analyzer.md`, *Categories: grain plus source*), not a registration unit; the registry stays
flat. You are the only agent editing the tree; when several subjects are added in
parallel the orchestrator isolates each in a git worktree
(`dev-orchestrate-parallel-subagents`), but you still run every step here normally.

## Reuse, do not reinvent

This codebase already has the scaffolding. A new subject writes **only its
data-specific math (Rust) and its data-specific drawing (Python)** — everything
else is a call into shared infra. If you find yourself parsing a path, formatting
a number, applying rcParams, building a DataFusion session, or hand-rolling a CDF,
stop: it already exists. Adding a new shared helper, color, or rcParam is a *stop
and ask* — shared look-and-feel is the whole point of `common/`.

Rust — reuse from `session.rs` / `io.rs` / `cdf.rs`, never re-implement:

| Need | Use | Notes |
|---|---|---|
| register a parquet table | `session::register_if_exists(ctx,name,path)` | register-once-reuse; `false` ⇒ emit `unavailable` |
| fail loud on schema drift | `session::require_columns(ctx,table,&[...])` | the column contract / drift guard |
| run SQL → batches | `session::collect(ctx, sql)` | DataFusion does the heavy scan/aggregate |
| read a column / a cell | `session::col`, `value_f64`, `value_f32_list` | numeric coercion already handled |
| artifact paths | `io::resolve_artifact_path` / `report_path` / `payload_path` | searches root + `raw/`/`reports/`/… |
| schema stamp | `io::SCHEMA_VERSION` | put on every report + payload |
| deployment / GPU facts | `io::read_deployment`, `read_run_meta`, `read_worker_pools` | bare-JSON sidecars, no `simulator` dep |
| CDF / percentiles / stats | `cdf::{stats, percentile_sorted, clean_nonnegative_sorted, cdf_series}` + `CdfSeries`/`MetricStats` | for any distribution subject |

Python — reuse from `common/`, never re-implement:

| Need | Use |
|---|---|
| load payload / output path | `common.layout.load_payload`, `plot_output_path` (also `resolve_artifact`) |
| figure + axes | `common.figure.new_axes(figsize=...)` |
| number formatting | `common.figure.fmt_value(v, unit)` (`ms`/`ms/token`/`%`/`tok/s`/…), `fmt_ms`, `fmt_count` — add a unit branch here, never a local formatter |
| corner annotation | `common.figure.corner_box(ax, lines, loc=...)` |
| legend | `common.figure.add_legend(ax)` — frameless, `loc="best"`; never a raw `ax.legend(...)` |
| title + labels + grid + save | `common.figure.finalize(fig, ax, out, title=, xlabel=, ylabel=, run_label=)` |
| colors / rcParams | `common.style.{CURVE, ACCENT, GRID, MARKER}` — importing `common.*` applies the shared rcParams; do NOT set your own |
| generic CDF curve | `common.cdf_plot` (don't re-draw a CDF) |

`render(log_dir)` returns a **list of picklable jobs** (`partial(...)`), run in
parallel by `__main__`; one payload may drive several figures.

**Before writing a new renderer, look for a plot of the same shape.** A CDF →
`common/cdf_plot.py`; a stairs / time-series → `throughput/segment_plot.py` or
`utilization/util_plot.py`. If your drawing would duplicate logic already in a
sibling renderer, lift the shared part into `common/` instead of copying it — but
a new `common/` helper ripples across every plot, so that lift is a *stop and
ask*.

## Keep it fast

`analyze run` scans a run's parquet, and a large real run holds millions of
`cost_log` rows / hundreds of millions of slots. The runner times each subject
and prints its own wall (subjects run concurrently over a shared read-only
`SessionContext`, so a slow subject is a real regression — it does not hide behind
the others). **On a large run, keep your subject well under 10 s; treat 30 s as a
hard ceiling you must never cross.** Get there by pushing work down, in rough order
of leverage — the point is the direction, not any specific number:

- **Do the reduction in SQL, not in Rust.** `session::collect` runs a DataFusion
  scan with predicate/projection pushdown and parallel hash-aggregation. Push the
  `WHERE` / `GROUP BY` / aggregation into the query and pull back only the small
  reduced result — never `SELECT` millions of rows into Rust and loop. See
  `utilization/series.rs` collapsing tens of millions of rows with one `GROUP BY`.
- **Downsample distributions in the payload.** Emit a CDF as a bounded,
  evenly-spaced curve, not a per-sample array — it is visually identical and tiny.
  `cdf::cdf_series` already does this; use it rather than dumping raw samples.
- **Stride-sample slot-scale data when exact is infeasible.** When even the
  aggregation is over hundreds of millions of slots, sample a subset of iterations
  (a temporal stride) and compute the distribution exactly over that sample; a
  location's rate is near-constant across iterations, so the sample is
  representative. See `batch/kernel_throughput.rs`. Always log what was sampled;
  never silently drop data.
- **If you must loop in Rust, keep the hot loop allocation-free** — pre-intern keys
  to integer ids (array index, no per-row hashing / string clone), as in the slot
  interning in `batch/kernel_throughput.rs`.

If none of these gets you under budget, that is a *stop and ask*: a new payload
shape or a sim-side pre-aggregation may be the real fix.

## Required reading

Read these first and cite the sections you use:

- `doc/analyzer.md` — the analyzer design contract: the report/payload envelope,
  the flat registry, categories = grain + source (find your metric's row or justify
  a new category), and the applicability/scope gate.
- `analyzer/README.md` — the code-matching reference and the **canonical Subjects
  catalog** (every registered subject + what it reads and emits); confirm your
  metric is not a duplicate and note where it will be listed.
- Reference pairs — open both halves and mirror the one whose shape matches:
  - *distribution / CDF* → `analyzer/rust/src/request/slo.rs` +
    `analyzer/python/request/slo_plot.py` (uses `cdf.rs` + `common/cdf_plot.py`);
  - *time-series* → `analyzer/rust/src/throughput/segment.rs` +
    `analyzer/python/throughput/segment_plot.py`, or the simpler
    `analyzer/rust/src/utilization/series.rs` +
    `analyzer/python/utilization/util_plot.py`.
- The `cost_log` schema (`simulator/src/log/README.md`) if your source is `cost_log`.

## Decide before coding

Write these down before touching code:

1. **Name** — the snake_case CLI token (`analyze run <dir> <name>`) and the Python
   `RENDERERS` key. Same string on both sides.
2. **Category** — an existing one (reuse a sibling's folder + extraction) or a new
   one (`doc/analyzer.md` grain + source test). A new category is one
   `Category` variant + a `src/<category>/` folder; the registry stays flat.
3. **Source parquet + columns** — the `require_columns` contract.
4. **Applies** — `Applies::All` (Tier-1, deployment-agnostic) or
   `Applies::Deployments(&[...])` (deployment-shaped). Never branch on deployment
   inside the metric — that is what the gate is for.
5. **Scope** — `Scope::Run` for an ordinary post-run subject (almost always this);
   `Scope::Alignment` only for a subject that reads the alignment manifest instead
   of a plain run.
6. **Shape** — distribution (`metrics` + CDF payload, reuse `cdf.rs`) or time-series
   (`totals` / `segments` + array payload). Match a reference pair.

## Build it

**Rust — compute:**

1. `src/<cat>/mod.rs`: `pub mod <name>;` (create the category dir if new).
2. `src/<cat>/<name>.rs`: `run_<name>(ctx, dir) -> Result<(Value, Value)>`.
   `register_if_exists(...)`; on a missing table return
   `unavailable` / `unavailable_payload` (copy the fallbacks from `slo.rs` /
   `segment.rs` verbatim, change only the `reason`). Then `require_columns(...)`,
   `collect(...)` the SQL, and reuse `cdf.rs` for a distribution. The report is
   `{schema_version: SCHEMA_VERSION, meta, available, metrics|totals|segments,
   definitions}`; the payload is `{schema_version, meta, ...arrays..., definitions}`.
   Build both with `serde_json::json!(...)` — no typed structs yet.
3. `main.rs`: `mod <cat>;` if the category is new.
4. `registry.rs`: for a new category add a `Category` variant + its `label()`; add
   one `SUBJECTS` row (`name`, `category`, `description`, `report_name`,
   `payload_name`, `applies`, **and `scope` — `Scope::Run` for an ordinary
   subject**); add one `run_subject` arm → `<cat>::<name>::run_<name>`; add
   `use crate::<cat>;` if new.
5. `cargo build` (analyzer) clean.

**Python — render:**

1. `<cat>/__init__.py` (empty) if the category dir is new.
2. `<cat>/<name>_plot.py`: `render(log_dir) -> list[Callable[[], Path]]` returning
   `partial(...)` jobs; draw with `common.figure` / `style` / `layout` only; bail
   cleanly (`return []`) when `payload.meta.available` is false or the arrays are
   empty.
3. `__main__.py`: `from <cat> import <name>_plot` + `RENDERERS["<name>"] = ...`.

**Docs — list it in the catalog:**

1. `analyzer/README.md`: add the subject to the **Subjects** table (name,
   category, what it reads, the report / payload it emits). If the category is new,
   also add its row to the `doc/analyzer.md` category table and the README
   directory map. The canonical catalog must list every subject — a subject is not
   done until it appears there.

**Verify:**

1. `analyze list` shows the subject.
2. `analyze run logs/unified_smoke <name>` → report + payload; cross-check a total
   against an independent duckdb one-liner (`uv run --with duckdb python -c ...`).
3. `uv run python analyzer/python render logs/unified_smoke <name>` → PNG; eyeball it.
4. A **larger run that actually carries your subject's inputs** — sanity +
   cross-check, and confirm the subject's printed wall stays within the speed
   budget above. Pick the run by checking for the sidecars/columns you read, not
   by name: `logs/` holds runs written by older launchers, and a subject that
   reads a sidecar block those runs predate will correctly report `unavailable`
   there. (`logs/unified_aime` is one such run — its `params.json` is the legacy
   flat layout with no nested `workload` block.) An `unavailable` on an old run
   is a passing graceful-degrade, not a verification.
5. No regression: `analyze run logs/unified_smoke` (all subjects) still ok; existing
   reports unchanged; `(cd analyzer/rust && cargo test)` passes; a full render has no
   import error.

Build the analyzer binary at `target/(debug|release)/analyze`. Run Python via
`uv run` (py3.12 venv; system py3.9 fails on `StrEnum`). For cross-checks duckdb is
a dev dep: `uv run --with duckdb python -c "..."`.

## Skeleton anchors

Copy these, do not invent them:

- **Unavailable fallbacks** (`unavailable` + `unavailable_payload`) — copy from
  `slo.rs` / `segment.rs` and change only the `reason`. Every subject must degrade
  gracefully when its parquet is absent; `analyze` is best-effort per subject.
- **`definitions()`** — a small `json!` of one-line metric definitions, embedded in
  BOTH report and payload (self-describing artifacts).
- **`meta`** — always `log_dir`; add `gpu_name` / `num_gpus` (from `read_run_meta`)
  when the metric normalizes per-GPU; add a `reason` on the unavailable path.
- **Renderer** — mirror `segment_plot.py` / `util_plot.py`: `_edges_s` for stairs,
  `corner_box` for summary numbers, `finalize` for title/labels/save. Colors come
  from `common.style`; percent / rate labels from `fmt_value`.

## Report at the end

State: subject / category / source / shape and Applies; files created or edited
(mark `[~]` any verified-pre-existing on a resume); a reuse audit (which
`session` / `io` / `cdf` fns and which `common/` helpers you used, and confirm you
added no new formatter / color / rcParam, or justify each); the verification
results including the duckdb cross-check number vs the report total and the
large-run wall vs the speed budget; and skill friction (be candid — unclear,
missing, or out-of-order steps; where you read source instead of the skill; the one
change that would have helped most). Note whether the `analyzer/README.md`
Subjects catalog — and, for a new category, the `doc/analyzer.md` category table —
was updated.

## Stop and ask

Pause (or note loudly and pick the nearest sane option) when:

- the metric needs a **new shared helper** — a `common.figure` formatter, a
  `common.style` color / rcParam, a new `session` / `cdf` primitive, or a payload
  *shape* not covered by the two reference pairs; shared infra ripples across every
  plot;
- the source parquet / columns don't exist, or the metric needs data the sim does
  not log yet (a sim-side change comes first);
- you cannot get a large run under the speed budget with the patterns above;
- the metric is genuinely deployment-shaped (Tier-2: `cost_log` per-rank / phase via
  the cost-tree manifest `node_labels`) — confirm the `Applies` set and whether the
  manifest exposes what you need;
- a new category's grain + source doesn't cleanly fit the `doc/analyzer.md`
  category model.
