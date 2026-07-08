# VibeSim Analyzer

Post-sim analysis: turn a run's parquet logs into **numbers** (for a human / LLM)
and **plots** (for the user). It is a **standalone crate** — the binary is
`analyze`, it has **no `simulator` dependency** (so it never drags in PyO3), and
it reads sim parquet purely **by column name** with a drift guard that fails loud
if the log schema moves.

This is the practical, code-matching reference; if this file disagrees with the code, the code wins —
open an issue.

## The split: Rust computes, Python renders

The module is two programs joined by one JSON contract — that is the **only**
boundary between the halves:

```
sim run ──writes→ <log_dir>/raw/{request_*.parquet,cost_log/worker_*.parquet,cost_manifest/worker_*.json}
                      │
  target/<build>/analyze run <log_dir> [subjects...]        (Rust, this::rust)
                      ├─→ <log_dir>/reports/<subject>_report.json    numbers, for an LLM
                      └─→ <log_dir>/payloads/<subject>_*.json        arrays, for the plot
                      │
  target/<build>/analyze trace <log_dir>                    (Rust, this::rust)
                      └─→ <log_dir>/traces/<prefix>.pftrace.gz
                      │
  python analyzer/python render <log_dir> [subjects...]     (Python, this::python)
                      └─→ <log_dir>/plots/<subject>_*.png
```

- **Rust does all compute** — scans parquet via DataFusion (parallel vectorized
  scan, predicate/projection pushdown, parallel hash-agg), does the math, emits
  JSON. This is where million-row grouped aggregations stay fast.
- **Python only renders** — matplotlib over the Rust payload JSON. It never opens
  a parquet file, so you can re-render without recomputing, and an LLM can read
  the report JSON with no plotting stack installed.

## What it exposes / what it requires

**Exposed upward** — two CLIs (no importable library API; callers shell out):

| Command | Side | Effect |
|---|---|---|
| `analyze run <log_dir> [subjects...]` | Rust | Compute subjects → `reports/` + `payloads/`, plus a subject-less `reports/analyzer_timing.json` run-meta sidecar. No subjects = all applicable. |
| `analyze trace <log_dir>` | Rust | Export a Perfetto per-kernel timeline from `cost_log/` + `cost_manifest/` → `traces/<prefix>.pftrace.gz`. |
| `analyze list` | Rust | Print the subject catalog. |
| `python analyzer/python render <log_dir> [subjects...]` | Python | Payloads → PNGs in `plots/`. No subjects = all renderers. |

The **launcher** is the primary caller: `launcher.exec.run_analysis` runs the
Rust `analyze run` then the Python `render` after each successful sim run. It is
**best-effort** — a missing analyzer binary or a failed subject never fails the
run (and `analyze` itself exits 0 even when individual subjects fail).

**Required from below** — a run directory written by the sim/L7, containing:

- `raw/request_slo.parquet` and `raw/request_state.parquet` — subject inputs.
- `raw/cost_log/worker_<pool_tag>_<worker_id>.parquet` plus matching
  `raw/cost_manifest/worker_<pool_tag>_<worker_id>.json` — trace inputs.
- `raw/params.json` — read only for the bare `deployment` string (drives the
  applicability gate). Absent → only deployment-agnostic subjects run.
- `raw/run_meta.json` — sim-written sidecar (`num_gpus`, `gpu_name`); the
  throughput subject reads it to normalize per-GPU. Absent → treated as 1 GPU.

All three are read as bare JSON / parquet by name — no `simulator` types crossed.

## Directory map

```
rust/                The `analyze` binary (DataFusion compute side).
  src/main.rs          CLI (`run` / `list`); the best-effort per-subject dispatch
                       loop; writes each (report, payload) + a run-timing report.
  src/registry.rs      THE SUBJECT CATALOG. `SUBJECTS` table + `run_subject`
                       dispatch + `select` (applicability gate). Adding a metric
                       touches only this file + a module under src/<category>/.
  src/io.rs            Artifact paths, `SCHEMA_VERSION`, read_deployment/run_meta.
  src/session.rs       DataFusion session, parquet registration, Arrow→Vec
                       extraction, and `require_columns` (the schema drift guard).
  src/cdf.rs           Shared numeric kernels: percentile, CDF downsample, and the
                       serde output shapes (`MetricStats`, `CdfSeries`).
  src/request/         Category = per-request/session metrics (slo).
  src/throughput/      Category = serving-rate-over-time metrics (throughput).
  src/trace/           Perfetto trace export from per-worker cost logs.

python/              The render side (matplotlib over payload JSON).
  __main__.py          `render <log_dir> [subjects]`; maps subject → renderer,
                       runs figure jobs in parallel (fork processes; matplotlib
                       is thread-hostile).
  request/, throughput/  One renderer module per subject; returns figure "jobs".
  common/              Shared plotting: payload loader + run-dir layout, figure
                       scaffolding, CDF plot, style.
```

## Subjects (the unit of analysis)

A **subject** is one analysis that emits a `(report, payload)` pair. A
**category** groups subjects sharing an analytical grain + parquet source; it is
just a tag + a `src/<category>/` folder, *not* a registration boundary — the
catalog (`registry::SUBJECTS`) is deliberately **flat**, so adding a metric is
one table row + one `run_subject` arm + the subject's module, never a new
per-category dispatch.

Current catalog:

| Subject | Category | Reads | Emits (report / payload) |
|---|---|---|---|
| `slo-general` | request | `request_slo.parquet` scalar columns | TTFT/TPOT/E2E stats / per-metric CDF series |
| `slo-detailed` | request | `request_slo.parquet` `output_token_times` column | ITL stats / CDF series when per-token logging is enabled |
| `throughput` | throughput | `request_state.parquet` (+ `run_meta.json`) | per-GPU prefill/decode/total TPS totals / fine `segments` + coarse `binned_segments` series |

**Deployment knowledge enters in exactly one place**: each subject's `Applies`
gate. Tier-1 (uniform-envelope) metrics are `Applies::All` and stay
deployment-blind; a deployment-shaped metric names the deployments it understands
and `select` drops it (with a note) for runs it doesn't apply to. So "point the
analyzer at any run" just works.

## The report/payload contract

Both halves agree on the run-dir layout (`raw/` · `reports/` · `payloads/` ·
`plots/`) and on a two-shape JSON contract, both stamped with `schema_version`
(`io::SCHEMA_VERSION`; bump on any breaking envelope change so the renderer can
refuse a payload it doesn't understand):

- **report** `<subject>_report.json` — `{schema_version, meta, available,
  metrics|totals|segments, definitions}`. Distribution subjects use
  `metrics: {<name>: {n, mean, p50, p90, p99, max}}` (`null` = no samples);
  time-series subjects use `totals` + a `segments` array.
- **payload** `<subject>_<shape>.json` — `{schema_version, meta, …arrays…}`,
  e.g. the SLO payload's `series: [{key, label, unit, n, x, y_pct, markers}]`.
  The throughput payload carries **two** views: the fine per-tick `segments` and
  a coarse `binned_segments` (≤`MAX_BINS`=10 equal-width bins, with `avg_per_gpu`
  run-average reference lines); the renderer draws both.
  An `available: false` / empty-`series` payload means "nothing to draw"; the
  renderer prints the reason and skips rather than erroring.

One report has **no subject**: `reports/analyzer_timing.json` (`main.rs`) is a
run-meta sidecar (which subjects ran, status, wall time) — not a
`<subject>_report.json`, and the renderer ignores it.

The Rust side never depends on `simulator`; the Python side never depends on
parquet. New metric work edits one category folder on each side plus the catalog
row — a flat registry, not per-deployment analyzers, is what keeps it from
re-fragmenting.
