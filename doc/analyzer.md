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
| `optimality` | distance from optimal GPU·s as a lower-bound ladder | `cost_log` + manifest + `run_meta` GPU counts + `gpu/spec.json` |
| `conservation` | run-wide work accounting | `cost_log` vs `request_slo` |
| `concurrency` | system, pool, or worker request populations over lifecycle time | `request_slo` arrival/terminal events and optional stage-transition timelines |
| `kv` | a KV pool over time | `kv_snapshot` + `run_meta` capacity |
| `alignment-iteration` | one measured iteration joined to one predict case | normalized NSYS + predict `cost_log`/manifest + mapping |
| `alignment-e2e` | one measured/simulated latency distribution | TraceLab replay JSONL + optional vLLM EngineCore request-timing JSONL + sim `request_slo` |
| `alignment-workload` | one scheduler iteration by recorded iteration id | normalized NSYS iteration metrics + sim `cost_log` |

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

## Relationship to other docs

This document supersedes the archived `old-doc/analyzer.md` (kept only as design
history in a separate repo; it is not maintained). For the code-matching module
reference and the canonical per-subject catalog, see
[`analyzer/README.md`](../analyzer/README.md); for where the analyzer sits in the
layer stack, see [architecture.md](architecture.md); for the step-by-step of adding
a metric, use the `add-analyzer-subject` skill.
