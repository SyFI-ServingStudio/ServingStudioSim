---
name: add-analyzer-subject
description: Use when adding a new VibeSim analyzer subject (a metric) end to end — the Rust compute side (a report + payload JSON built from a run's parquet via DataFusion) and the Python render side (a matplotlib PNG drawn from the payload). Covers the flat registry, the report/payload contract, the Applies gate, and — emphatically — reusing the shared session / io / cdf infra and the Python common style / figure / layout helpers instead of re-implementing them. NOT for running an existing analyzer (that is run-moesim-analyzer) or adding an L1 kernel (that is top-add-kernel).
---

# Add Analyzer Subject (Rust compute + Python render)

Adds one analyzer **subject** (one metric → one `(report, payload)` pair → one or
more PNGs) across both halves of the boundary:

- **Rust** (`analyzer/rust/src/`) — scans the run's parquet with DataFusion, does
  the math, emits `reports/<name>_report.json` (numbers, for an LLM) +
  `payloads/<name>_<shape>.json` (arrays, for the plot). **No `simulator` dep.**
- **Python** (`analyzer/python/`) — matplotlib over the payload JSON only. **Never
  opens parquet.**

The JSON pair is the ONLY boundary. A subject lives in a **category** folder
(`request/`, `throughput/`, `utilization/`, …) — the category is a tag + a source
family (`old-doc/analyzer.md` §5), not a registration unit.

> **Single-agent skill.** You are the only agent editing the tree. If several
> subjects are added in parallel, the orchestrator isolates each in a git worktree
> (`dev-orchestrate-parallel-subagents`) — you still run every step here normally.

---

## 0. The #1 rule: REUSE, do not reinvent

This codebase already has the scaffolding. A new subject writes **only its
data-specific math (Rust) and its data-specific drawing (Python)** — everything
else is a call into shared infra. If you find yourself parsing a path, formatting a
number, applying rcParams, building a DataFusion session, or hand-rolling a CDF,
**stop — it already exists.** Adding a new shared helper, color, or rcParam is a
*stop and ask* (§6): shared look-and-feel is the whole point of `common/`.

**Rust — reuse from `session.rs` / `io.rs` / `cdf.rs` (never re-implement):**

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

**Python — reuse from `common/` (never re-implement):**

| Need | Use |
|---|---|
| load payload / output path | `common.layout.load_payload`, `plot_output_path` (also `resolve_artifact`) |
| figure + axes | `common.figure.new_axes(figsize=...)` |
| number formatting | `common.figure.fmt_value(v, unit)` (`ms`/`ms/token`/`%`/`tok/s`/…), `fmt_ms`, `fmt_count` — **add a unit branch here, never a local formatter** |
| corner annotation | `common.figure.corner_box(ax, lines, loc=...)` |
| legend | `common.figure.add_legend(ax)` — frameless, `loc="best"` (auto lowest-overlap); never a raw `ax.legend(...)` |
| title + labels + grid + save | `common.figure.finalize(fig, ax, out, title=, xlabel=, ylabel=, run_label=)` |
| colors / rcParams | `common.style.{CURVE, ACCENT, GRID, MARKER}` — importing `common.*` applies the shared rcParams; do NOT set your own |
| generic CDF curve | `common.cdf_plot` (don't re-draw a CDF) |

`render(log_dir)` returns a **list of picklable jobs** (`partial(...)`), run in
parallel by `__main__`. One payload may drive several figures.

---

## 1. Required reading (read first; cite the anchors you use)

- `old-doc/analyzer.md` — the design contract. Especially **§2** (report/payload
  shapes), **§4** (flat registry), **§5** (categories: grain + source — find your
  metric's row, or justify a new category), **§7** (Applies gate), **§9** (How to
  extend — this skill operationalizes it).
- **Reference pairs — open both halves and mirror the one whose shape matches:**
  - *Distribution / CDF* subject → `analyzer/rust/src/request/slo.rs` +
    `analyzer/python/request/slo_plot.py` (uses `cdf.rs` + `common/cdf_plot.py`).
  - *Time-series* subject → `analyzer/rust/src/throughput/segment.rs` +
    `analyzer/python/throughput/segment_plot.py`, or the simpler
    `analyzer/rust/src/utilization/series.rs` + `analyzer/python/utilization/util_plot.py`.
- The `cost_log` schema (`old-doc/logging.md`) if your source is `cost_log`.

---

## 2. Decide before coding (write these down)

1. **Name** — the snake_case CLI token (`analyze run <dir> <name>`) AND the Python
   `RENDERERS` key. Same string both sides.
2. **Category** — an existing one (sibling subject, reuse its folder + extraction)
   or new (`old-doc/analyzer.md` §5 grain+source test). New category = one `Category`
   variant + a `src/<category>/` folder; the registry STAYS flat.
3. **Source parquet + columns** — the `require_columns` contract.
4. **Applies** — `Applies::All` (Tier-1, deployment-agnostic) or
   `Applies::Deployments(&[...])` (deployment-shaped). Never branch on deployment
   inside the metric — that's what the gate is for.
5. **Shape** — distribution (`metrics` + CDF payload, reuse `cdf.rs`) or
   time-series (`totals`/`segments` + array payload). Match a reference pair.

---

## 3. Checklist (copy into your notes, check off as you go)

```
Subject: <name>   Category: <cat>   Source: <parquet>   Shape: <cdf|timeseries>

Rust — compute
[ ] R1 src/<cat>/mod.rs: `pub mod <name>;`  (new category dir if needed)
[ ] R2 src/<cat>/<name>.rs: `run_<name>(ctx, dir) -> Result<(Value, Value)>`.
       - register_if_exists(...) → missing ⇒ unavailable/unavailable_payload (copy
         the fallbacks from slo.rs/segment.rs verbatim).
       - require_columns(...); collect(...) the SQL; reuse cdf.rs if distribution.
       - report: {schema_version: SCHEMA_VERSION, meta, available, metrics|totals|
         segments, definitions}.  payload: {schema_version, meta, ...arrays...,
         definitions}.  Build as serde_json::json!(...) (no typed structs yet).
[ ] R3 main.rs: `mod <cat>;` if the category is new.
[ ] R4 registry.rs: (new cat) add Category variant + label(); add ONE SUBJECTS row
       (name/category/description/report_name/payload_name/applies); add ONE
       run_subject arm → <cat>::<name>::run_<name>; `use crate::<cat>;` if new.
[ ] R5 cargo build (analyzer) clean.

Python — render
[ ] P1 <cat>/__init__.py (empty) if the category dir is new.
[ ] P2 <cat>/<name>_plot.py: `render(log_dir) -> list[Callable[[],Path]]` returning
       partial(...) jobs; draw with common.figure/style/layout ONLY; bail cleanly
       (return []) when payload.meta.available is false / arrays empty.
[ ] P3 __main__.py: `from <cat> import <name>_plot` + RENDERERS["<name>"] = ....

Verify (§5)
[ ] V1 analyze list shows the subject.
[ ] V2 analyze run logs/unified_smoke <name> → report+payload; cross-check a total
       against an independent duckdb one-liner (uv run --with duckdb python -c ...).
[ ] V3 uv run python analyzer/python render logs/unified_smoke <name> → PNG; eyeball.
[ ] V4 Larger run (logs/unified_aime_large) sanity + cross-check.
[ ] V5 No regression: analyze run logs/unified_smoke (all subjects) still ok;
       existing reports unchanged; (cd analyzer/rust && cargo test) passes; full
       render has no import error.
```

> Build the analyzer binary at `target/(debug|release)/analyze`. Run Python via
> `uv run` (py3.12 venv; system py3.9 fails on StrEnum). For cross-checks duckdb is
> a dev dep: `uv run --with duckdb python -c "..."`.

---

## 4. Skeleton anchors (copy, don't invent)

- **Unavailable fallbacks** (`unavailable` + `unavailable_payload`): copy from
  `slo.rs` / `segment.rs` and only change the `reason`. Every subject must degrade
  gracefully when its parquet is absent — `analyze` is best-effort per subject.
- **`definitions()`**: a small `json!` of one-line metric definitions, embedded in
  BOTH report and payload (self-describing artifacts).
- **`meta`**: always `log_dir`; add `gpu_name`/`num_gpus` (from `read_run_meta`)
  when the metric normalizes per-GPU; add a `reason` on the unavailable path.
- **Renderer**: mirror `segment_plot.py` / `util_plot.py` — `_edges_s` for stairs,
  `corner_box` for summary numbers, `finalize` for title/labels/save. Colors come
  from `common.style`; percent/rate labels from `fmt_value`.

---

## 5. Output report (state at the end)
- Subject / category / source / shape; Applies.
- Files created/edited (mark `[~]` any verified-pre-existing on a resume).
- Reuse audit: which `session`/`io`/`cdf` fns and which `common/` helpers you used
  (and confirm you added NO new formatter/color/rcParam, or justify each).
- Verification: V1–V5 results, including the duckdb cross-check number vs the
  report's total.
- Skill friction (be candid): unclear/missing/out-of-order steps; where you read
  source instead of the skill; the one change that would have helped most.
- `old-doc/analyzer.md` is a symlink into a separate repo — note whether the §5 table
  row was updated or deferred.

---

## 6. Stop and ask
Pause (or note loudly and pick the nearest sane option) when:
- The metric needs a **new shared helper** — a `common.figure` formatter, a
  `common.style` color/rcParam, a new `session`/`cdf` primitive, or a new payload
  *shape* not covered by the two reference pairs. Shared infra changes ripple
  across every plot; confirm before adding.
- The source parquet/columns don't exist or the metric needs data the sim doesn't
  log yet (that's a sim-side change first).
- The metric is genuinely deployment-shaped (Tier-2: `cost_log` per-rank/phase via
  the cost-tree manifest `node_labels`) — confirm the `Applies` set and whether the
  manifest exposes what you need.
- A new category's grain+source doesn't cleanly fit `old-doc/analyzer.md` §5.
