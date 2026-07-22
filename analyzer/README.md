# VibeSim Analyzer

Post-run analysis: turn simulator parquet logs into **numbers** (for a human / LLM)
and **plots** (for the user), including paired measured alignment profiles. It
is a **standalone crate** — the binary is
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
                      └─→ <log_dir>/plots/  (PNG overview and breakdowns)

alignment profile ──writes→ <profile_log_dir>/{parsed.json,profile_result.json,...}
alignment timing-predict ──writes→ <predict_log_dir>/{raw/,cases,config,input_manifest}
alignment analyze ──reads completed roots + mapping
                  └─writes→ <analysis_log_dir>/alignment_manifest.json
                                           │
  target/<build>/analyze alignment <analysis_log_dir> [subjects...] (Rust)
                                           ├─→ reports/alignment_{iteration,e2e,workload}_report.json
                                           └─→ payloads/alignment_{iteration,e2e,workload}_series.json
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
| `analyze run <log_dir> [--lock-batch-size] [subjects...]` | Rust | Compute subjects → `reports/` + `payloads/`, plus a subject-less `reports/analyzer_timing.json` run-meta sidecar. Unlocked optimality keeps the standard names; `--lock-batch-size` writes `optimality_batch_locked_report.json` / `optimality_batch_locked_waterfall.json`, makes `R3=R2`, and classifies each current leaf for R5. No subjects = all applicable. |
| `analyze alignment <analysis_log_dir> [subjects...]` | Rust | Read the alignment manifest and compute iteration/E2E/workload subjects into this analysis root. No subjects = all alignment subjects. |
| `analyze trace <log_dir>` | Rust | Export a Perfetto per-kernel timeline from `cost_log/` + `cost_manifest/` → `traces/<prefix>.pftrace.gz`. |
| `analyze serve --logs-root <dir>` | Rust | Serve the read-only viz-ui catalog, bounded worker operation windows, and exact `(worker, iter, batch, operation)` CostTrees reconstructed lazily from `cost_log` + manifest. |
| `analyze list` | Rust | Print the subject catalog. |
| `python analyzer/python render <log_dir> [subjects...]` | Python | Payloads → PNG plots, including sampled alignment breakdowns. No subjects = all renderers. |

The **launcher** is the primary caller: `launcher.exec.run_analysis` runs the
Rust `analyze run` then the Python `render` after each successful sim run;
`run_alignment_analysis` computes and renders the subjects enabled by the
separate analyze phase config. Both execution paths are
**best-effort** — a missing analyzer binary, failed handoff, or failed subject
never fails the completed run.

When optimality is in launcher intent, `run_analysis` computes locked optimality
first and the normal unlocked subject set second. Both JSON pairs coexist;
Python renders the primary unlocked payload, while the UI descriptor publishes
locked optimality as `variants.batch_locked` for interactive switching.

**Required from below** — `analyze run` consumes a run directory written by the
sim/L7, containing:

- `raw/request_slo.parquet` and `raw/request_state.parquet` — subject inputs.
- `raw/cost_log/worker_<pool_tag>_<worker_id>.parquet` plus matching
  `raw/cost_manifest/worker_<pool_tag>_<worker_id>.json` — trace inputs.
- `raw/params.json` — normal analysis reads the bare `deployment` string (drives
  the applicability gate). Absent → only deployment-agnostic normal subjects run.
  Alignment iteration analysis needs no completed simulation: it derives the
  GPU-cycle multiplier from measured quantities alone (`Σ measured_gpu_cycle_ms /
  Σ measured_ms`) and applies it to the timing-predict totals itself.
- `raw/run_meta.json` — sim-written sidecar (`num_gpus`, `gpu_name`); the
  throughput subject reads it to normalize per-GPU. Absent → treated as 1 GPU.

All three are read as bare JSON / parquet by name — no `simulator` types crossed.
Worker detail remains outside the run descriptor body: the descriptor advertises
the `workers` capability, and the only selectable entity is one raw operation.
`workers/{pool}/{worker}/operations?offset=...&limit=...` reads any contiguous
global-ordinal range (at most 384 summaries); the UI uses 64-operation ranges
for drag navigation. Each summary is one `worker_cost` row with stable identity
`(iter_id,batch_id,operation_id)`, section/layer, and its exact bounded interval.
The worker index sorts globally by `(start_ms,iter_id,batch_id,operation_id)`;
the local decimal `operation_id` is assigned by
`(wall_start_ms,section,layer,total_time_ms)`. Duplicate local keys or
non-monotonic global end times fail loud, preserving deterministic O(log N +
hits) half-open seek without rescanning millions of rows.

`workers/{pool}/{worker}/operations/seek?at_ms=...` returns every operation
covering the cursor, an anchor (or nearest operation when there is no hit), a
suggested 64-operation viewport, and one surrounding 192-operation buffer.
`worker_kind` is `afd_attn`, `afd_ffn`, or `iterwise`; `batch_role` states
whether `batch_id` represents a `slot` or a `batch`. Exact CostTrees use
`operations/{iter_id}/{batch_id}/{operation_id}/cost-tree` and reconstruct only
that one raw row. A streaming scan builds a compact per-worker index once; a
512 MiB byte-budgeted LRU bounds retained indexes, and range JSON is not cached.
When the optimality subject is ready,
`workers/{pool}/{worker}/iterations/{iter_id}/optimality-kernel-ladder` folds all
rows for that exact worker iteration into the same R0-R5 per-kernel ladder used
by the run payload. Iterations have no scheduler holding-span boundary, so
R0=R1 and idle is zero; R1-R2 remains aggregate imbalance. This high-cardinality
detail is requested on selection and is not embedded for every iteration.
The alignment path reads `<analysis_log_dir>/alignment_manifest.json`, which
points to normalized NSYS JSON in the profile root, timing-predict cost
parquet/manifest, the profile's `replay_result` TraceLab JSONL, its optional
engine-core `request_timings_result` JSONL, sim request-SLO parquet, and exact
sequence-row-to-operation labels. An operation may own one
or more simulated leaf slots; its measured kernel durations are counted once
and its folded slot workloads are summed. Reports, payloads, and plots stay in
the analysis root; the input roots are never used as output directories.

## Directory map

```
rust/                The `analyze` binary (DataFusion compute side).
  src/main.rs          CLI (`run` / `alignment` / `list`); the best-effort
                       per-subject dispatch loop shared by both source scopes.
  src/registry.rs      THE SUBJECT CATALOG. `SUBJECTS` table + `run_subject`
                       dispatch + `select` (applicability gate). Adding a metric
                       touches only this file + a module under src/<category>/.
  src/io.rs            Artifact paths, `SCHEMA_VERSION`, read_deployment/run_meta.
  src/session.rs       DataFusion session, parquet registration, Arrow→Vec
                       extraction, and `require_columns` (the schema drift guard).
  src/cdf.rs           Shared numeric kernels: percentile, CDF downsample, and the
                       serde output shapes (`MetricStats`, `CdfSeries`).
  src/pca.rs           Shared numeric kernel: standardize + top-2 principal
                       components (the kernel-input-distribution feature projection).
  src/request/         Category = per-request/session metrics (slo).
  src/throughput/      Category = serving-rate-over-time metrics (throughput).
  src/utilization/     Per-worker GPU busy-fraction with per-pool averages over time.
  src/batch/           Batch composition + per-location achieved kernel throughput.
  src/backend/         Backend selection over a kernel position's input feature space.
  src/breakdown/       CostTree replay + run-wide leaf-position composition.
  src/optimality/      Sub-optimality waterfall (R0..R5 lower-bound ladder) in GPU·s;
                       grid_peaks.rs sidecar (sim `kernel-query peak`) + spec.rs hardware ceilings.
  src/kernel_query.rs  Shared transport to the sim `kernel-query` subcommand (grid/eval/peak).
  src/conservation/    Run-wide work-accounting checks (actual vs expected).
  src/concurrency/     In-flight concurrency and hierarchical request-stage populations over simulated wall-clock time.
  src/kv/              Per-pool KV-cache occupancy over time.
  src/alignment_iteration/  Per-iteration total/operation/kernel comparison.
  src/alignment_e2e/        Paired request latency + completion throughput.
  src/alignment_workload/   Measured-vs-sim scheduler batch shapes by iteration id and elapsed time.
  src/alignment_input.rs    Shared alignment manifest/path contract.
  src/trace/           Perfetto trace export from per-worker cost logs.

python/              The render side (matplotlib over payload JSON).
  __main__.py          `render <log_dir> [subjects]`; maps subject → renderer,
                       runs figure jobs in parallel (fork processes; matplotlib
                       is thread-hostile).
  request/, throughput/, utilization/, batch/, backend/, breakdown/, optimality/, conservation/, concurrency/, kv/  One renderer module per subject; returns figure "jobs".
  alignment_iteration/, alignment_e2e/, alignment_workload/  Alignment payload renderers.
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
| `utilization` | utilization | `cost_log` slot times (+ `run_meta.json`) | per-worker GPU compute utilization plus per-pool averages over time / `utilization_series` |
| `batch` | batch | `request_state.parquet` | per-batch composition (batch / prefill / decode token counts) over time + stats / `batch_scatter` series |
| `kernel-throughput` | batch | 1/50-sampled `cost_log` slots + matching CostTree manifests | achieved TFLOP/s (compute) and GB/s (memory BW) per cost-tree location / per-location `kernel_throughput_locations` stats |
| `kernel-input-distribution` | backend | sampled `cost_log` `slot_input` + `slot_backend` + matching CostTree manifest `backends` lists | per-position selected-backend counts/ratios + PCA/feature projection / one scatter per position (`kernel_input_distribution_scatter`), rendered to `plots/kernel_input_dist/<position>.png`; unavailable on runs without per-slot backend + input logging |
| `kernel-time-share` | breakdown | `cost_log` slot times + matching CostTree manifests | root kernel-time share by leaf position at overall / pool / worker levels; exact on small runs and bounded worker-stratified sampling on large runs |
| `optimality` | optimality | `cost_log` (exact R0/R1 + 1/stride-sampled fold) + CostTree manifests + `run_meta` `gpu_ids` counts + `gpu/spec.json` hardware ceilings + `raw/kernel_grid_peaks.json` sidecar + unlocked-only `model.work` necessary-work labeler | sub-optimality waterfall in GPU·s — telescoping buckets at cluster / pool / worker / iteration / per-kernel levels, plus per-worker R0→R5 stacked-kernel ladders with aggregate idle/imbalance chunks; unlocked level bars may split R5 into global necessary-work bands / `optimality_waterfall` |
| `concurrency` | concurrency | `request_slo.parquet` arrival + terminal timestamps | exact request count/peak/mean / <=512-bin time-weighted active-request series |
| `request-state` | concurrency | `request_slo.parquet` stage-transition lists + `run_meta.json` stage vocab/worker roster | exact category/pending peaks and means / 200-bin cluster and request-owner-worker open-category stacks plus owner-pool aggregate/average/worker pending series; execution-only pools (AFD FFN) are omitted; unavailable when stage logging is off |
| `workload-conservation` | conservation | `cost_log` actuals + `request_slo.parquet` per-request expected | run-wide prefill/decode/FFN/KV work accounting, pass/fail / `workload_conservation_checks` |
| `kv-occupancy` | kv | `kv_snapshot` stream + `run_meta.json` capacity | per-pool KV occupancy (active / projected-peak / promised tokens, and as a fraction of capacity) over time / `kv_occupancy_series` |
| `alignment-iteration` | alignment-iteration | normalized NSYS exact sequence rows + predict cost log/manifest + user mapping (no simulation) | kernel/mapping error stats + self-derived `recommended_gpu_time_multiplier` (`Σ measured_gpu_cycle_ms / Σ measured_ms`) / separate kernel-busy and measured first-kernel-to-next-first-kernel GPU-cycle overviews + per-iteration mapped stacks |
| `alignment-e2e` | alignment-e2e | TraceLab replay JSONL + parsed NSYS GPU timeline + optional vLLM engine-core request timing JSONL + sim `request_slo.parquet` | independent client-TTFT/sim, optional server-TTFT/sim, client-TPOT/sim, optional server-TPOT/sim, E2E stats, client-completion throughput, and server GPU-span throughput / available raw latency CDF overlays + client/sim completion series annotated with all aggregate rates |
| `alignment-workload` | alignment-workload | normalized NSYS iteration metrics + sim `cost_log.groups`/`wall_start_ms` | per-side workload summaries / fine prefill-token, decode-batch-size, scheduled-KV-workload, and actual iteration-cycle series by iteration id, plus decode batch size by elapsed time |

The alignment-iteration payload and report retain every paired iteration, but
the Python renderer evenly samples at most 128 per-iteration breakdown PNGs.
It keeps the overview and other subject-level plots directly under `plots/`,
then groups breakdowns in batches of 32 under raw-id ranges such as
`plots/iter_6_to_305/iter_6_breakdown.png`. Sampling therefore bounds rendering
cost without changing any computed statistic or discarding payload rows.
Breakdowns use a pre-sized 18-inch, 150-DPI canvas (2700 pixels wide), skip the
tight-bounding-box redraw, and use lossless PNG compression level 1. The smaller
set of overview/CDF figures keeps the shared 300-DPI PNG default. A breakdown
PNG newer than both its payload and renderer source is skipped, so an interrupted render resumes missing/stale figures
instead of regenerating all 128. Stale generated rows and replaced PNG
or JPG breakdowns are removed without touching subject-level plots.

The flat registry also carries a `Scope` (`run` or `alignment`) so default
selection never points a normal-run subject at an alignment bundle or vice
versa. **Deployment knowledge enters in exactly one place**: each subject's `Applies`
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
