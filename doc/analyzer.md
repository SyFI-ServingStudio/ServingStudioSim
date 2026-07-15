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
| `conservation` | run-wide work accounting | `cost_log` vs `request_slo` |
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

## Launcher integration and publication lifecycle

Analysis is **best-effort with respect to simulation** — a missing analyzer
binary, a failed handoff, or a failed subject never changes a completed
simulation into a failed simulation. Publication to readers is stricter. After
each successful run the launcher executes one ordered generation:

1. Rust `analyze run` computes report/payload JSON;
2. Python `render` creates plots;
3. Rust `analyze trace` creates the bounded overview Perfetto trace.

Before compute starts, and after every transition, the launcher atomically
replaces `reports/analyzer_pipeline_state.json`. Its version-1 envelope contains
a fresh opaque `generation_id`, a stable `artifact_revision`, requested subject
tokens, exact trace artifact path, real producer version/revision/binary digest,
timestamps, the pipeline status (`pending`, `complete`, or `failed`), and explicit
compute/render/trace stage states. One per-run lease serializes generations so an
older producer cannot overwrite a newer attempt. `complete` is published only
after the trace stage reaches a terminal state. A compute failure makes the
pipeline `failed`; optional render or trace failure leaves the orchestration
`complete` but keeps that stage/resource explicitly failed (in particular,
`trace_failed` never becomes a ready trace). A launcher crash leaves a durable
`pending` generation. State is bounded JSON; readers never scrape stdout.

Before current-generation compute completes, old report/payload/trace files are
never published as `ready`. After compute completes, only requested subjects
listed by a timing artifact carrying the same `generation_id` may become ready,
even while render/trace is still pending; an unrequested or mismatched old pair
stays hidden. A trace becomes ready only when its stage is complete and its exact
state-recorded path passes containment and size checks. `artifact_revision` is
the stable `analysis.revision` for that generation. Runs predating this sidecar
remain readable through an explicit legacy path: valid bounded artifact pairs
may be ready, their revision is derived from immutable artifact contents, and
their producer is `legacy-unknown`, never the serving binary.

All analyzer JSON and trace outputs use a durable same-directory temporary file,
file sync, and atomic replace. Thus readers see the old or new complete file,
never a partially encoded artifact. A run's subject selection is a durable
preset key (`analyze_subjects`, omitted = all applicable); `--no-analyze` is the
transient "skip it this time" switch.

## Read-only UI service

The analyzer also owns the read boundary between completed run artifacts and
the browser. `analyze serve --logs-root <dir>` recursively discovers simulation
runs below one or more explicitly configured roots and exposes protocol-v1,
bounded resources under `/api/v1/`. This remains a read-only analyzer concern:
the service does not run subjects, render plots, mutate a run, or expose parquet.

The public resources are:

- `GET /api/v1/runs` — a catalog whose `run_id` values are stable opaque ids;
  a basename is only display text because nested sweeps commonly repeat names
  such as `simulation` and `tp4`;
- `GET /api/v1/runs/{run_id}/descriptor` — deployment, lifecycle, workers,
  analyzer-registry subject states, and relative artifact links;
- descriptor-linked summary, topology, report, payload, and Perfetto resources.

`SUBJECTS` remains the only source of subject tokens and report/payload names.
The HTTP layer must not copy that catalog or translate tokens to UI-specific
names. A subject is independently `pending`, `ready`, `unavailable`,
`not_generated`, or `failed`; an optional subject failure never invalidates the
descriptor or another subject. Only the bounded summary and topology are core
page resources. The topology envelope has its own
`TOPOLOGY_SCHEMA_VERSION`; it is not coupled to the report/payload envelope
version.

Every resource link is selected from a server-built allowlist for a resolved
run. Request parameters are never joined directly to filesystem paths. Roots
and runs are canonicalized, symlink escapes are rejected, and the service never
serves `raw/*.parquet`, `raw/gpu_cluster/**`, temporary files, or arbitrary
paths. Catalog, descriptor, and immutable artifacts use `ETag` and conditional
`304` responses so a UI can poll pending analysis without repeatedly decoding
unchanged JSON. Errors use `application/problem+json` with a stable `code` in
addition to human-readable detail.

Ready report, payload, and trace links are relative paths scoped by the
descriptor's `analysis.revision`. A request carrying a revision other than the
current readable generation fails closed with `artifact_generation_changed`;
the service never resolves that old link to a fixed filename from a newer
generation. Descriptor assembly and linked-resource reads use a bounded
seqlock-style check: capture pipeline/timing state, open or read every required
artifact, then capture state again and retry the whole read if it changed. Trace
reads keep the opened file descriptor across the final check, so a subsequent
atomic replacement cannot change the bytes being streamed. Legacy runs use the
content-derived legacy revision as the same read fence.

The default Host allowlist is limited to loopback spellings (`localhost`,
`127.0.0.1`, and `[::1]`) as a DNS-rebinding boundary. A same-origin development
proxy should preserve or rewrite `Host` to the loopback analyzer target. A
non-loopback reverse-proxy hostname is accepted only when the operator repeats
`--allow-host <hostname>` explicitly; changing `--bind` alone never expands the
allowlist. Method rejection includes `Allow: GET`.

Catalog discovery is cached for 30 seconds, refreshed by one single-flight scan
on a blocking worker, and never runs a multi-gigabyte tree walk on an async
request worker. Descriptor JSON has a bounded metadata-stamped cache. Artifact
reads on Linux use `openat2(RESOLVE_BENEATH|NO_SYMLINKS)` from a stable
configured-root descriptor over the combined root-relative run and artifact
path, then enforce the maximum and serve from that same descriptor. Linux
kernels without usable `openat2` support fail closed. Non-Linux builds retain
canonical containment and therefore treat configured logs roots as a
trusted-writer boundary. Large traces stream without a trace-sized service
buffer; their weak ETag contains device, inode, length, mtime seconds, and mtime
nanoseconds from the atomically published file, so conditional requests do not
hash or scan a 512 MB file first.

The full wire schema, href restrictions, cache-key rules, and analyzer-token to
UI-domain mapping live in the consumer protocol contract at
`../viz-ui/docs/data-protocol.md`. Both transports — checked-in artifact export
and HTTP — must reuse the same subject-specific decoders.

## Relationship to other docs

This document supersedes the archived `old-doc/analyzer.md` (kept only as design
history in a separate repo; it is not maintained). For the code-matching module
reference and the canonical per-subject catalog, see
[`analyzer/README.md`](../analyzer/README.md); for where the analyzer sits in the
layer stack, see [architecture.md](architecture.md); for the step-by-step of adding
a metric, use the `add-analyzer-subject` skill.
