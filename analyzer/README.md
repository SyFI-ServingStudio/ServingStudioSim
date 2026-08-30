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
| `analyze run <log_dir> [--lock-batch-size] [subjects...]` | Rust | Compute subjects → `reports/` + `payloads/`, plus one complete `reports/analyzer_timing.json` run-meta sidecar. Whenever optimality is selected, the normal command emits both unlocked primary and batch-locked variant atomically. `--lock-batch-size` is the low-level locked-only recomputation path. No subjects = all applicable. |
| `analyze alignment <analysis_log_dir> [subjects...]` | Rust | Read the alignment manifest and compute iteration/E2E/workload subjects into this analysis root. No subjects = all alignment subjects. |
| `analyze sweep <experiment_dir>` | Rust | Collect existing per-run SLO, throughput, utilization, and completion scalars listed by `sweep_manifest.json`; emit sweep report/payload JSON without rescanning parquet. |
| `analyze trace <log_dir>` | Rust | Export a Perfetto per-kernel timeline from `cost_log/` + `cost_manifest/` → `traces/<prefix>.pftrace.gz`. |
| `analyze serve --logs-root <dir>` | Rust | Serve the read-only viz-ui catalog, bounded worker operation windows, exact `(worker, iter, batch, operation)` CostTrees reconstructed lazily from `cost_log` + manifest, first-class `prediction`, `kernel_profile`, `kernel_measurement`, and `hardware/gpus` resources, and cross-run sweep aggregates. |
| `analyze list` | Rust | Print the subject catalog. |
| `python analyzer/python render <log_dir> [subjects...]` | Python | Payloads → PNG plots, including sampled alignment breakdowns. No subjects = all renderers. |

The **launcher** is the primary caller: `launcher.exec.run_analysis` runs the
Rust `analyze run` then the Python `render` after each successful sim run;
`run_alignment_analysis` computes and renders the subjects enabled by the
separate analyze phase config. Both execution paths are
**best-effort** — a missing analyzer binary, failed handoff, or failed subject
never fails the completed run.

When optimality is selected, one normal `analyze run` atomically computes both
the unlocked primary and locked variant and records both in the same timing
sidecar. Python renders the primary unlocked payload, while the UI descriptor
publishes locked optimality as `variants.batch_locked` for interactive switching.

**Required from below** — `analyze run` consumes a run directory written by the
sim/L7, containing:

- `artifact.meta.json` — explicit first-class root identity. UI discovery accepts
  `simulation_run` here and does not infer the resource type from any file below.
- `raw/request_slo.parquet` and `raw/request_state.parquet` — subject inputs.
- `raw/cost_log/worker_<pool_tag>_<worker_id>.parquet` plus matching
  `raw/cost_manifest/worker_<pool_tag>_<worker_id>.json` — trace inputs.
- `raw/params.json` — normal analysis reads the bare `deployment` string (drives
  the applicability gate). Absent → only deployment-agnostic normal subjects run.
  Alignment iteration analysis needs no completed simulation: it derives the
  GPU-cycle multiplier from measured quantities alone (`Σ measured_gpu_cycle_ms /
  Σ measured_busy_union_ms`) and applies it to the timing-predict totals itself.
- `raw/run_meta.json` — simulation-only sidecar (`num_gpus`, `gpu_name`); the
  throughput subject reads it to normalize per-GPU. Its presence or absence never
  selects the artifact kind. Timing predictions instead carry their physical L4
  extent in `prediction.meta.json.gpu_count` and do not create this sidecar.

The UI model resource preserves the raw config and enriches it with optional
`model.work` parameter counts (`total`, model-card-style `active`, and
`active_layers`). Unsupported architectures degrade only this enrichment to
`null`; the config resource remains readable.

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
When the optimality subject is ready, exact iteration analysis uses two distinct
high-cardinality resources. `workers/{pool}/{worker}/iterations/{iter_id}/optimality-waterfall`
returns the full one-row waterfall for that worker iteration, while the sibling
`optimality-kernel-ladder` resource returns the per-kernel ladder.
Both fold every matching row and use the explicit `mode` query. Iterations have
no scheduler holding-span boundary, so R0=R1 and idle is zero; R1-R2 remains
aggregate imbalance. Both modes may split R5 with segmented and fully fused
necessary-work floors. Batch-locked mode labels the exact observed batch; unlocked
mode labels 10,000 independent copies of its batch entries and normalizes back to one
iteration, amortizing weights without changing sequence length. When a strict
versioned semantic-location map covers the manifest, either mode's kernel ladder
also appends R6 and per-location necessary/redundant/under-accounted work; otherwise
it remains R0-R5. Run-level output uses the same distinction: locked analysis adds
per-iteration R6/R7 through worker, pool, and cluster; unlocked analysis composes
additive work under its saturated-batch policy. The detail meta records the mode and
replication factor.
The alignment path reads `<analysis_log_dir>/alignment_manifest.json`, which
points to normalized NSYS JSON in the profile root, timing-predict cost
parquet/manifest, the profile's `replay_result` req-frontend JSONL, its optional
engine-core `request_timings_result` JSONL, sim request-SLO parquet, and exact
sequence-row-to-operation labels. An operation may own one
or more simulated leaf slots; its measured kernel durations are counted once
and its folded slot workloads are summed. Reports, payloads, and plots stay in
the analysis root; the input roots are never used as output directories.
When schema-v4 folding assigns different sequences to disjoint rank subsets,
mapped rows with the same phase, semantic operation, physical category, and
per-device category-local ordinal are one logical occurrence and are joined
before cross-rank reduction. Category-local ordinals keep a rank-specific
auxiliary kernel from shifting later work onto the wrong occurrence. The original
rows remain the authority for mapping and unmapped-work audits. Uneven rank-local
shapes may select different tuned physical kernel names for that occurrence;
the joined row records every physical name and source row ID while still
requiring one `cross_rank` semantic.

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
  src/optimality/      Sub-optimality waterfall (R0..R7 lower-bound ladder) in GPU·s;
                       grid_peaks.rs sidecar (sim `kernel-query peak`) + spec.rs hardware ceilings.
  src/kernel_query.rs  Shared transport to the sim `kernel-query` subcommand (grid/eval/peak).
  src/conservation/    Run-wide work-accounting checks (actual vs expected).
  src/concurrency/     In-flight concurrency and hierarchical request-stage populations over simulated wall-clock time.
  src/kv/              Per-pool KV-cache occupancy over time.
  src/alignment_iteration/  Per-iteration total/operation/kernel comparison.
                       mod.rs owns the measured reduction and the `alignment-iteration`
                       report; timeline.rs reuses that reduction to emit
                       `alignment-timeline` with the timestamps kept; host.rs anchors
                       the optional CPU-side sidecar onto the same axis.
  src/alignment_e2e/        Paired request latency + completion throughput.
  src/alignment_workload/   Measured-vs-sim scheduler batch shapes by iteration id and elapsed time.
  src/alignment_input.rs    Shared alignment manifest/path contract.
  src/trace/           Perfetto trace export from per-worker cost logs.
  src/ui_service/       Read-only HTTP resources; beyond runs/sweeps/predictions it
                       owns hardware.rs (gpu/spec.json canonical name/aliases +
                       dense peaks + HBM + one-way/bidir interconnect), kernel_profile.rs
                       (discovery, descriptor, enriched curve), and kernel_measurement.rs
                       (discovery, descriptor, summary, declared-plot serving).

python/              The render side (matplotlib over payload JSON).
  __main__.py          `render <log_dir> [subjects]`; maps subject → renderer,
                       runs figure jobs in parallel (fork processes; matplotlib
                       is thread-hostile).
  request/, throughput/, utilization/, batch/, backend/, breakdown/, optimality/, conservation/, concurrency/, kv/  One renderer module per subject; returns figure "jobs".
  alignment_iteration/, alignment_e2e/, alignment_workload/  Alignment payload renderers.
                       (`alignment-timeline` has no matplotlib renderer: its payload exists to be
                        panned and zoomed, and a static PNG of it would only restate the sibling's
                        stacks. Its consumers are `viz-ui/smoke/align/` and the app's alignment view.)
  sweep/               Cross-run 1-D line, 2-D heatmap, and N-D faceted sweep plots.
  common/              Shared plotting: payload loader + run-dir layout, figure
                       scaffolding, CDF plot, style.
```

## First-class non-run resources

Beyond runs, the read-only service publishes `prediction`, `kernel_profile`,
`kernel_measurement`, and `hardware/gpus` resources (see `doc/analyzer.md` for
the full protocol). Kernel resources are produced by the Python profiling CLI
(`python -m profiling run ... --output-dir`, `... measure ...`) and are
discoverable without any conversation backend:

- `GET /api/v1/kernel-profiles`, `.../kernel-profiles/{id}/descriptor`, `.../curve`;
- `GET /api/v1/kernel-measurements`, `.../kernel-measurements/{id}/descriptor`,
  `.../summary`, `.../plots/{plot}`;
- `GET /api/v1/hardware/gpus?name=<gpu_name>`.

Guards: every discovery is workspace-aware, ignores `old-logs`, and rejects
duplicate/invalid ids; plot paths accept only declared basenames; curve
enrichment resolves the catalog by exact case-insensitive name/aliases and never
fabricates a line for a missing dtype/GPU. Legacy snapshots are discovered as
`kp_legacy_<hash>` / `km_legacy_<hash>` with no hardware limits.

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
| `slo-goodput` | request | `request_slo.parquet` completed/output-token/TPOT scalar columns + `params.json` configured duration/rate | hard-cutoff all-request accounting, strict arithmetic-mean TPOT pass, and fixed-window output goodput / TPOT CDF series |
| `throughput` | throughput | `request_state.parquet` (+ `run_meta.json`) | per-GPU prefill/decode/total TPS totals / fine `segments` + coarse `binned_segments` series |
| `utilization` | utilization | `cost_log` slot times (+ `run_meta.json`) | per-worker GPU compute utilization plus per-pool averages over time / `utilization_series` |
| `batch` | batch | `request_state.parquet` | per-batch composition (batch / prefill / decode token counts) over time + stats / `batch_scatter` series |
| `kernel-throughput` | batch | 1/50-sampled `cost_log` slots + matching CostTree manifests | achieved TFLOP/s (compute) and GB/s (memory BW) per cost-tree location / per-location `kernel_throughput_locations` stats |
| `kernel-input-distribution` | backend | sampled `cost_log` `slot_input` + `slot_backend` + matching CostTree manifest `backends` lists | per-position selected-backend counts/ratios + PCA/feature projection / one scatter per position (`kernel_input_distribution_scatter`), rendered to `plots/kernel_input_dist/<position>.png`; unavailable on runs without per-slot backend + input logging |
| `kernel-time-share` | breakdown | `cost_log` slot times + matching CostTree manifests | root kernel-time share by leaf position at overall / pool / worker levels; exact on small runs and bounded worker-stratified sampling on large runs |
| `optimality` | optimality | `cost_log` (exact R0/R1 + 1/stride-sampled fold) + CostTree manifests + `run_meta` `gpu_ids` counts + `gpu/spec.json` hardware ceilings + `raw/kernel_grid_peaks.json` sidecar + `model.work` necessary-work labeler | sub-optimality waterfall in GPU·s — telescoping buckets at cluster / pool / worker / iteration / per-kernel levels; analyzer-owned worker/pool/cluster kernel ladders extend R0→R5 with location-attributed segmented R6 and scope-fused R7, using saturated-work recomputation when unlocked and fixed-iteration addition when locked / `optimality_waterfall` |
| `concurrency` | concurrency | `request_slo.parquet` arrival + terminal timestamps | exact request count/peak/mean / <=512-bin time-weighted active-request series |
| `request-state` | concurrency | `request_slo.parquet` stage-transition lists + `run_meta.json` stage vocab/worker roster | exact category/pending peaks and means / 200-bin cluster and request-owner-worker open-category stacks plus owner-pool aggregate/average/worker pending series; execution-only pools (AFD FFN) are omitted; unavailable when stage logging is off |
| `workload-conservation` | conservation | `cost_log` actuals + `request_slo.parquet` immutable fresh/declared input and runtime hit/computed/output observations | run-wide prefix token balance, cache-aware causal-prefill/cold-equivalent/decode-KV, and prefill/decode/FFN accounting, pass/fail / `workload_conservation_checks` |
| `kv-occupancy` | kv | `kv_snapshot` stream + `run_meta.json` capacity | per-pool KV occupancy (active total / retained-prefix component / projected-peak / promised tokens, and as a fraction of capacity) over time; old streams remain readable with the missing prefix breakdown marked unavailable / `kv_occupancy_series` |
| `alignment-timeline` | alignment-iteration | the same inputs as `alignment-iteration`, but keeping every rank's per-kernel `(start_ns, end_ns)`, plus the optional host sidecar (`host_timeline` in the alignment manifest) | EVERY iteration, as an index plus a byte-range-addressed `alignment_timeline_iterations.jsonl`: raw measured intervals, per-slot UNIT sim times, the cost manifest verbatim, and — when the sidecar is present — per-thread NVTX and CUDA-runtime host lanes, so a client can draw the GPU, the sim and the CPU on ONE time axis. At most 32 iterations carry a distinguishing `selected_as`; the report writes up those. / reference-rank per-phase span/busy/idle + largest gaps named by the operations either side |
| `alignment-iteration` | alignment-iteration | normalized NSYS exact sequence rows + predict cost log/manifest + user mapping (no simulation) | per-iteration distributions, duration-weighted stage/all signed and absolute error, semantic operations ranked by absolute-error impact, mapping audit, and self-derived `recommended_gpu_time_multiplier` (`Σ measured_gpu_cycle_ms / Σ measured_busy_union_ms`) / separate kernel-busy and GPU-cycle overviews + byte-range-queryable per-iteration critical-device stream stacks |
| `alignment-e2e` | alignment-e2e | full-run req-frontend replay JSONL + optional engine-core request timing JSONL + sim `request_slo.parquet` | independent client-TTFT/sim, optional server-TTFT/sim, client-TPOT/sim, optional server-TPOT/sim, E2E stats, and client-completion throughput / available raw latency CDF overlays + client/sim completion series annotated with aggregate rates |
| `alignment-workload` | alignment-workload | full-run EngineCore iteration metrics + recorded replay monotonic window + sim `cost_log.groups`/`wall_start_ms` | per-side workload summaries / fine prefill-token, decode-batch-size, scheduled-KV-workload, and actual iteration-cycle series by iteration id, plus decode batch size by elapsed time |

`alignment-timeline` runs with `alignment-iteration` in the kernel-align phase; it
reads exactly what its sibling reads, so it needs no new artifact.

Both of them shard. A real 2,040-iteration capture holds hundreds of megabytes of
per-kernel detail — past what a browser can parse as one string, for views that
draw one iteration at a time. So each payload is an INDEX (one summary row per
iteration, plus what every iteration shares) naming a sibling `.jsonl` and the
byte range of each iteration inside it. A reader seeks; it never loads the file.
On the reference capture that is 544 MB -> 2.96 MB for
`alignment_iteration_series.json` and 2.0 MB of index beside a 249 MB seekable
shard for `alignment_timeline.json`.

The two per-iteration shards are published as immutable, content-addressed
generations (the SHA-256 is part of the filename), and the payload names the
exact generation whose byte ranges it indexes. This keeps a cached payload
readable while another analyzer rerun is in progress and makes identical reruns
reuse the same shard. Distinct and orphaned generations are deliberately not
deleted in the publication path: a retention tool must first exclude every
generation referenced by a current payload and allow for the UI cache grace
period. Long-lived run directories therefore need an explicit retention pass;
eagerly deleting the previous generation is unsafe.

Alignment-iteration schema v2 also shards the two run-wide inventories that can
grow with sequence diversity. `alignment_kernel_inventory.jsonl` retains every
per-position audit row without putting them in the report, and
`alignment_sequence_programs.jsonl` retains every folded per-stream program.
The series carries only sequence identity, occurrences, track summaries, and
whole-capture cost plus byte ranges into the program shard. The UI therefore
loads one selected program at a time; it never downloads every rank-local
multi-stream variant to draw one mapping board. Schema-v1 artifacts remain
readable through their embedded bare `program` or explicit `tracks` shape.

The alignment-iteration payload and report retain every paired iteration, but
the Python renderer evenly samples at most 128 per-iteration breakdown PNGs.
It keeps the overview and other subject-level plots directly under `plots/`,
then groups breakdowns in batches of 32 under raw-id ranges such as
`plots/iter_6_to_305/iter_6_breakdown.png`. Sampling therefore bounds rendering
cost without changing any computed statistic or discarding payload rows.
Each breakdown selects the same complete critical device as its headline, shows
its material CUDA streams separately, and folds the small tail into one explicit
`other streams` row. It reads only the sampled timeline byte ranges. Breakdowns
use a pre-sized 150-DPI canvas, skip the tight-bounding-box redraw, and use
lossless PNG compression level 1. The smaller
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
