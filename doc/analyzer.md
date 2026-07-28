# Analyzer

Post-run analysis turns a run's parquet logs into **numbers** (for a human or LLM)
and **plots** (for the user), including paired measured-vs-simulated alignment
profiles. It is the design contract for the analyzer: follow it so new metrics stay
uniform instead of re-fragmenting into per-deployment analyzers.

The analyzer is a **standalone crate** (`analyzer/rust`, binary `analyze`) with
**no `simulator` dependency**, so it never drags in PyO3. It reads sim parquet
**by column name** behind a presence check that fails loud when the log schema
drifts. This document is the current-state contract; for the file-by-file,
code-matching reference — and the live per-subject catalog — see
[`analyzer/README.md`](../analyzer/README.md), and for the workflow of adding a
metric use the `add-analyzer-subject` skill.

## The split: Rust computes, Python renders

The analyzer is two programs joined by one JSON contract, which is the **only**
boundary between the halves:

- **Rust does all compute.** It scans the run's parquet with DataFusion (parallel
  vectorized scan, predicate/projection pushdown, parallel hash-aggregation), does
  the math, and writes `reports/<subject>_report.json` (numbers) plus
  `payloads/<subject>_<shape>.json` (arrays).
- **Python only renders.** It draws matplotlib over the payload JSON and **never
  opens a parquet file**, so a run can be re-rendered without recomputing and an
  LLM can read a report with no plotting stack installed.

Neither half depends on the other's world: the Rust side never links `simulator`,
the Python side never touches parquet.

## The report/payload contract

Both halves agree on the per-run layout — `raw/` (sim parquet + sidecars) ·
`reports/` (numbers JSON) · `payloads/` (plot JSON) · `plots/` (PNG) — and on a
two-shape JSON envelope, each stamped with `schema_version` (`io::SCHEMA_VERSION`;
bump on any breaking envelope change so the renderer can refuse a payload it does
not understand):

- **report** `<subject>_report.json` — `{schema_version, meta, available,
  metrics | totals | segments, definitions}`. A distribution subject uses
  `metrics: {<name>: {n, mean, p50, p90, p99, max}}` (`null` = no samples); a
  time-series subject uses `totals` plus a `segments` array.
- **payload** `<subject>_<shape>.json` — `{schema_version, meta, …arrays…,
  definitions}`, arrays only. An `available: false` / empty payload means "nothing
  to draw": the renderer prints the reason and skips rather than erroring.

Both shapes are built inline with `serde_json::json!` values. `definitions` — a
small map of one-line metric definitions — is embedded in both the report and the
payload so each artifact is self-describing.

A subject may publish named **variants** when the same schema represents explicit
counterfactual policies. The primary report/payload names remain the backward-
compatible default; each variant gets its own durable files and descriptor hrefs.
Variants are not separate subjects and inherit the subject's schema version.
Optimality publishes primary unlocked artifacts plus a `batch_locked` variant.

## The flat registry

`registry.rs` holds **one** `SUBJECTS` catalog and **one** `run_subject` dispatch
for every category. A **subject** is one analysis emitting one `(report, payload)`
pair; a **category** groups subjects that share an analytical grain and parquet
source. The category is a tag and a source folder (`src/<category>/`), **not** a
registration unit — the registry is deliberately flat, so adding a metric is one
`SUBJECTS` row + one `run_subject` arm + the subject's module, never a new
per-category dispatch. The live per-subject list lives in
[`analyzer/README.md`](../analyzer/README.md) (the Subjects catalog).

## Categories: grain plus source

A category is defined by its **analytical grain** (the unit of analysis) and its
**input source family** — the shared grain + source is exactly what lets its
subjects reuse the same extraction code and cohabit a folder, and it is the test
for "new subject in an existing category" vs. "new category". The current
categories:

| Category | Grain | Source |
|---|---|---|
| `request` | one request (session = rollup) | `request_slo` |
| `throughput` | system over a time segment | `request_state` snapshots |
| `utilization` | a resource (GPU / pool) over time | `cost_log` |
| `batch` | one scheduled batch / one cost-tree location over time | `request_state`, `cost_log` |
| `backend` | a cost-tree position's selected backend over its input feature space | `cost_log` `slot_input` + `slot_backend` + manifest `backends` |
| `breakdown` | a cost-tree subtree by leaf position | `cost_log` + manifest labels |
| `optimality` | distance from optimal GPU·s as a lower-bound ladder | `cost_log` + manifest + `run_meta` GPU counts + `gpu/spec.json` + unlocked-only `model.work` necessary-work labeler |
| `conservation` | run-wide work accounting | `cost_log` vs `request_slo` |
| `concurrency` | system, pool, or worker request populations over lifecycle time | `request_slo` arrival/terminal events and optional stage-transition timelines |
| `kv` | a KV pool over time | `kv_snapshot` + `run_meta` capacity |
| `alignment-iteration` | one measured iteration joined to one predict case | normalized NSYS + predict `cost_log`/manifest + mapping |
| `alignment-e2e` | one measured/simulated latency distribution | TraceLab replay JSONL + optional vLLM EngineCore request-timing JSONL + sim `request_slo` |
| `alignment-workload` | one scheduler iteration by recorded iteration id | normalized NSYS iteration metrics + sim `cost_log` |

For a symmetric tensor-parallel alignment, one measured iteration contains one
range set per rank. The analyzer reduces each measured occurrence across its
ranks into one replica critical-path contribution, then sums those — so the
per-kernel segments and per-operation rows add up to the headline `measured_ms`,
which is compared to the simulator's Sum/Max cost tree. The reduction depends on
the mapping table's per-kernel `cross_rank` class, never on a category or name:

- **independent** (compute, point-to-point): the occurrence costs `max` over
  ranks of `(end - start)`. Cross-rank compute imbalance stays real work on the
  path — exactly what the single-rank sim cost under-models if it assumes balance.
- **synchronizing** (all-reduce / all-gather / all-to-all / fused all-reduce+norm):
  a barrier whose per-rank kernel duration includes waiting for the slowest rank
  to arrive. The occurrence costs `max(end) - max(start)` — from "last rank
  arrived" to "collective done" — which drops the arrival wait while keeping a
  self-imbalanced collective's bottleneck rank. For a symmetric all-reduce this
  equals `min(duration)`; for an imbalanced all-to-all it does not.

The dropped arrival wait is not attributed to any kernel; it surfaces only in the
wall-clock `measured_gpu_cycle_ms`. The kernel-align pass's derived duty-cycle
multiplier `recommended_gpu_time_multiplier = Σ measured_gpu_cycle_ms / Σ
measured_ms` spans exactly this gap, so its denominator shares the per-occurrence
`measured_ms` reduction above. GPU durations from different ranks are never
summed. `measured_busy_union_ms`
retains the old cross-rank interval union for audit. The analyzer keeps per-device
populations so rank skew is auditable, and distinguishes raw `rank_launches` from
`replica_calls` (symmetric launches divided by the captured device count). A shared
folded label program is valid only when the parser has proved the exact ordered
kernel sequence is identical across all participating devices.

The alignment trio is separate not by deployment but by **source scope** (see
below): it reads an alignment manifest instead of a plain run directory.

## Applicability, scope, and intent

Three independent decisions, never collapsed:

- **Applicability** (`Applies` on each subject) is intrinsic to
  `(metric × deployment)` and is the **only** place deployment knowledge enters the
  analyzer. The analyzer reads the deployment as a bare string from `params.json`
  and self-selects: a Tier-1 uniform-envelope metric is `Applies::All` and stays
  deployment-blind; a deployment-shaped metric names the deployments it understands
  and `select` drops it (with a note) elsewhere. Never branch on deployment inside
  a shared metric — move the concern to its own gated subject instead. Because
  applicability lives with the metric, `analyze run <dir>` on any run "just works".
- **Scope** (`Scope::{Run, Alignment}`) gates the **source envelope**, not
  deployment: `analyze run` selects only `Run` subjects, `analyze alignment` only
  `Alignment` subjects. There is no second alignment registry.
- **Intent** (which applicable subjects to actually run) lives at the
  launcher/preset and can only *narrow within* what is applicable — it never
  overrides applicability into running a nonsensical metric.

## Performance budget

`analyze run` scans a run's parquet, and a large real run holds millions of
`cost_log` rows / hundreds of millions of slots. Subjects run **concurrently** over
a shared read-only DataFusion session, and each subject's own wall time is recorded
(`reports/analyzer_timing.json`), so a slow subject is a visible regression rather
than one hidden behind the others. **A large run stays well under 10 s; treat 30 s
as a hard ceiling.** Keep it there by pushing work down — the direction matters, not
any specific constant:

- **Do the reduction in SQL.** Push `WHERE` / `GROUP BY` / aggregation into the
  DataFusion query and pull back only the small reduced result; never `SELECT`
  millions of rows into Rust to loop.
- **Downsample distributions.** Emit a CDF as a bounded, evenly-spaced curve, not a
  per-sample array — visually identical, tiny JSON.
- **Stride-sample slot-scale data.** When even the aggregation spans hundreds of
  millions of slots, sample a subset of iterations (a temporal stride) and compute
  the distribution exactly over that sample; log what was sampled, never silently
  drop data.

If none of these gets a large run under budget, that is a signal to change the
payload shape or pre-aggregate on the sim side, not to ship a slow subject.

## Launcher integration

Analysis is **best-effort** end to end — a missing analyzer binary, a failed
handoff, or a failed subject never fails a completed run. The launcher builds the
analyzer crate explicitly (a failed build warns, does not block), and after each
successful run it runs the Rust `analyze run` then the Python `render`. A run's
subject selection is a durable preset key (`analyze_subjects`, omitted = all
applicable); `--no-analyze` is the transient "skip it this time" switch.
When optimality is selected, launcher first computes its `batch_locked` variant
and then performs the normal unlocked analysis pass, so both policies are
available to viz-ui without rerunning simulation. The final
`analyzer_timing.json` belongs to the normal requested-subject pass.

## Cross-run sweep analysis

A launcher sweep is one explicit experiment envelope above ordinary run
directories. The launcher owns its coordinates because only expansion knows
which bindings formed the sweep; the analyzer owns collection of the already
computed per-run scalar reports. Neither side discovers membership by scanning
nearby directories.

The launcher writes `<experiment_dir>/sweep_manifest.json` before aggregation:

```json
{
  "schema_version": 1,
  "axes": ["request_rate", "tp_size"],
  "runs": [
    {
      "path": "rate40_tp2/simulation",
      "coordinates": {"request_rate": 40.0, "tp_size": 2},
      "labels": {}
    }
  ]
}
```

`axes` preserves launcher-DSL declaration order. Every run path is relative to
the experiment directory and membership is exactly the listed set; an analyzer
must reject absolute paths, traversal, duplicate coordinates, missing
coordinates, and paths that resolve outside the experiment. Different
experiment directories are never merged automatically.

A discovered run not claimed by any valid `sweep_manifest.json` is published as
its own **singleton aggregate**. This is a one-member envelope with `kind:
"singleton"`, zero axes, empty domains/coordinates, and the same scalar metric
descriptors as a sweep. It is never grouped with sibling runs or promoted into
a synthetic sweep. Runs already claimed by a manifest appear only through that
manifest-defined experiment, not again as singleton entries.

An `old-logs` directory is an archive boundary and is excluded from active run,
sweep, and singleton discovery. Moving an experiment there removes it from UI
catalogs without deleting its artifacts.

Aggregate catalog entries also publish bounded selection metadata derived from
their claimed runs' `raw/params.json`: the normalized experiment date from a
leading `YYYYMMDD` log-directory component, unique deployment tags, and unique
trace basenames. Manifest aggregates union metadata only across their declared
members; singleton aggregates use only their one run. Missing or malformed
params yield empty metadata instead of failing catalog discovery. The catalog
sorts by experiment date descending, then artifact update time descending;
entries without a dated experiment name sort after dated entries.

Repeated launcher invocations may extend the manifest only when they target the
same experiment directory and declare the same ordered axis list. The launcher
then upserts members by relative run path. A different axis list is a different
experiment envelope and must use another directory; silently combining the two
would make coordinate uniqueness and plot geometry ambiguous.

`analyze sweep <experiment_dir>` reads only `summary.json` plus the existing
`slo-general`, `throughput`, and `utilization` reports from each member. It does
not rescan parquet or invent a second definition of those metrics. Missing or
unfinished member artifacts become explicit lifecycle states and null metric
values rather than failing the whole sweep. Outputs use the normal analyzer
layout at the experiment root:

- `reports/sweep_summary_report.json` — one scalar row per member run;
- `payloads/sweep_metrics_grid.json` — ordered axes, coordinate domains, rows,
  and metric display metadata for the renderer.

The renderer is dimension-generic. One axis produces line plots; two axes
produce scalar heatmaps; with three or more axes, the first two remain the
heatmap axes and every coordinate combination of the remaining axes is a facet.
Facet panels use a near-square row/column layout. Missing grid cells remain
blank and are never interpolated. Constraints, verdicts, inquiry text, and
agent state are not sweep-analysis inputs.

The read-only UI service publishes sweep analysis as a separate protocol-v1
resource family:

- `GET /api/v1/sweeps` discovers experiment envelopes containing
  `sweep_manifest.json` plus unclaimed singleton runs below the configured logs
  roots. Each catalog entry has an opaque `sweep_id`, `kind` (`sweep` or
  `singleton`), display name, ordered axes, member count, aggregate status,
  experiment date, deployment/trace filter values, and payload href.
- `GET /api/v1/sweeps/{sweep_id}/payload` projects the existing
  `sweep_metrics_grid.json` for a manifest sweep. For a singleton it projects
  the same bounded scalars directly from that run's existing summary and
  analyzer reports; neither path rescans parquet.

The HTTP projection never exposes `meta.experiment_dir` or member filesystem
paths. When a manifest member resolves to a discovered run, its payload row
contains that run's opaque `run_id`; missing members retain `run_id: null`.
Clients may use this id for drill-down but must not infer run identity from
labels, coordinates, or hrefs. A manifest without an aggregate payload remains
visible as `pending`, so incomplete analysis is distinguishable from an empty
logs root.

The long-lived UI service may replace static `--logs-root` flags with
`--workspace-registry <registry.json>`. That registry is generated by the
conversation backend and contains stable workspace ids plus active/archived
logs roots. The service reloads it for discovery, scans only active workspaces,
and publishes `workspace_id` on every run/sweep catalog entry and payload.
Opaque `run_id` and `sweep_id` are therefore interpreted only together with
their workspace id; clients must not resolve an id against another workspace.

A managed Launcher run writes `experiment.meta.json` at the experiment root:

```json
{
  "schema_version": 1,
  "experiment_id": "e_...",
  "origin": {
    "kind": "managed",
    "workspace_id": "w_...",
    "conversation_id": "...",
    "turn_id": "...",
    "job_id": "j_...",
    "role": "implementer"
  }
}
```

When this file is present and valid, its `experiment_id` is the sweep catalog
identity. Legacy/direct experiments without it keep the deterministic
path-derived id. The Analyzer remains read-only: capability checks, allowed
logs roots, job lifecycle, and conversation-to-experiment relationships belong
to the shared backend and Launcher bridge.

The workload overview resolves `workload.trace_files` against the configured
logs root. Launcher output may be either logs-root-relative (`trace/<file>`,
`<experiment>/trace/<file>`) or workspace-relative with the logs-root basename
as its prefix (`logs/<experiment>/trace/<file>` when `--logs-root logs`). The
resolver tries the direct relative path first and only then strips that exact
basename prefix. Every component must remain normal (no absolute path or
traversal), the path must pass through a directory named `trace`, and the
canonical regular file must remain below the configured logs root. The UI
service must not assume that every launcher trace is under one root-level
`trace/` directory.

## Relationship to other docs

This document supersedes the archived `old-doc/analyzer.md` (kept only as design
history in a separate repo; it is not maintained). For the code-matching module
reference and the canonical per-subject catalog, see
[`analyzer/README.md`](../analyzer/README.md); for where the analyzer sits in the
layer stack, see [architecture.md](architecture.md); for the step-by-step of adding
a metric, use the `add-analyzer-subject` skill.
