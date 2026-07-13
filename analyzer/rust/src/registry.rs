//! The analyzer's subject catalog — one flat, root-level table for **every**
//! category. A *subject* is one analysis that emits a `(report, payload)` pair;
//! a *category* groups subjects sharing an analytical grain + input source. The
//! registry is intentionally NOT per-category: keeping
//! it flat means adding a metric in any category is one [`SUBJECTS`] row + one
//! [`run_subject`] arm, never a new per-category table/dispatch to repeat.

use std::path::Path;

use anyhow::{bail, Result};
use datafusion::prelude::SessionContext;
use serde_json::Value;

use crate::alignment_e2e;
use crate::alignment_iteration;
use crate::alignment_workload;
use crate::batch;
use crate::conservation;
use crate::kv;
use crate::request;
use crate::throughput;
use crate::utilization;

/// Analytical grain + source family a subject belongs to. A category is a *tag*
/// (and a source folder) here, not a registration boundary — that's what lets
/// the catalog stay flat. New category = a new variant + a `src/<cat>/` folder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    /// Per-request (session = rollup) over the fixed-schema `request_slo` /
    /// `request_state` parquet. Tier-1, deployment-agnostic.
    Request,
    /// System serving rate over time/iter windows. Tier-1, deployment-agnostic.
    Throughput,
    /// A compute resource (pool of workers / GPUs) busy fraction over time, from
    /// `cost_log`. Tier-1, deployment-agnostic.
    Utilization,
    /// Per-batch (per-iteration) composition over time, from `cost_log`. Tier-1,
    /// deployment-agnostic.
    Batch,
    /// Run-wide work-accounting invariants — `cost_log` actuals vs `request_slo`
    /// per-request expected. Tier-1, deployment-agnostic.
    Conservation,
    /// KV-cache pool occupancy over time, from the `kv_snapshot` stream. Tier-1,
    /// deployment-agnostic (a run without KV logging degrades to `unavailable`).
    Kv,
    /// One measured vLLM model iteration joined to one offline predict case.
    AlignmentIteration,
    /// Per-iteration scheduler workload over each run's recorded iteration ids.
    AlignmentWorkload,
    /// One request and run-level completion timeline joined across real/sim runs.
    AlignmentE2e,
}

impl Category {
    fn label(self) -> &'static str {
        match self {
            Category::Request => "request",
            Category::Throughput => "throughput",
            Category::Utilization => "utilization",
            Category::Batch => "batch",
            Category::Conservation => "conservation",
            Category::Kv => "kv",
            Category::AlignmentIteration => "alignment-iteration",
            Category::AlignmentWorkload => "alignment-workload",
            Category::AlignmentE2e => "alignment-e2e",
        }
    }
}

/// Which CLI artifact root a subject consumes. This is orthogonal to deployment
/// applicability: a normal simulation run and a paired alignment bundle are
/// different source envelopes, but both remain rows in the one flat registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Run,
    Alignment,
}

impl Scope {
    fn label(self) -> &'static str {
        match self {
            Scope::Run => "run",
            Scope::Alignment => "alignment",
        }
    }
}

/// Which deployments a subject is meaningful for — the ONLY place deployment
/// knowledge enters the analyzer, so shared metrics stay deployment-blind.
/// Tier-1 (uniform-envelope) subjects are [`Applies::All`]; a deployment-shaped
/// metric (e.g. EP imbalance) names the deployments it understands.
#[derive(Debug, Clone, Copy)]
pub enum Applies {
    All,
    #[allow(dead_code)] // first user lands with the first deployment-shaped metric
    Deployments(&'static [&'static str]),
}

impl Applies {
    fn matches(&self, deployment: Option<&str>) -> bool {
        match self {
            Applies::All => true,
            Applies::Deployments(list) => deployment.is_some_and(|d| list.contains(&d)),
        }
    }

    fn label(&self) -> String {
        match self {
            Applies::All => "all deployments".to_string(),
            Applies::Deployments(list) => list.join(", "),
        }
    }
}

/// One analysis subject. `name` is the CLI token (`analyze run <dir> <name>`) and
/// the Python renderer key; `description` is the one-line help shown by
/// `analyze list`; `report_name`/`payload_name` are the artifact files it writes
/// under `reports/` / `payloads/`.
pub struct Subject {
    pub name: &'static str,
    pub category: Category,
    pub description: &'static str,
    pub report_name: &'static str,
    pub payload_name: &'static str,
    pub applies: Applies,
    pub scope: Scope,
}

/// The whole catalog. Adding a metric = one row here + one arm in [`run_subject`]
/// + the subject's module under `src/<category>/`.
pub const SUBJECTS: &[Subject] = &[
    Subject {
        name: "slo-general",
        category: Category::Request,
        description: "Request/session latency SLOs from scalar columns: TTFT / TPOT / E2E + session E2E CDFs (always available).",
        report_name: "slo_general_report.json",
        payload_name: "slo_general_cdf.json",
        applies: Applies::All,
        scope: Scope::Run,
    },
    Subject {
        name: "slo-detailed",
        category: Category::Request,
        description: "Per-token ITL CDF from `output_token_times` (only when io.log_output_token_times was on; otherwise unavailable).",
        report_name: "slo_detailed_report.json",
        payload_name: "slo_detailed_cdf.json",
        applies: Applies::All,
        scope: Scope::Run,
    },
    Subject {
        name: "throughput",
        category: Category::Throughput,
        description: "Per-GPU prefill / decode / total tokens-per-second, per time segment.",
        report_name: "throughput_report.json",
        payload_name: "throughput_segments.json",
        applies: Applies::All,
        scope: Scope::Run,
    },
    Subject {
        name: "utilization",
        category: Category::Utilization,
        description: "Per-pool GPU compute utilization (fraction of workers busy) over time.",
        report_name: "utilization_report.json",
        payload_name: "utilization_series.json",
        applies: Applies::All,
        scope: Scope::Run,
    },
    Subject {
        name: "batch",
        category: Category::Batch,
        description: "Per-batch composition (batch / prefill / decode token counts) over time + stats.",
        report_name: "batch_report.json",
        payload_name: "batch_scatter.json",
        applies: Applies::All,
        scope: Scope::Run,
    },
    Subject {
        name: "kernel-throughput",
        category: Category::Batch,
        description: "Achieved throughput per cost-tree location (leaf name; Max siblings pooled): \
                      TFLOP/s (compute) and GB/s (memory BW) over 1/50-sampled cost_log slots.",
        report_name: "kernel_throughput_report.json",
        payload_name: "kernel_throughput_locations.json",
        applies: Applies::All,
        scope: Scope::Run,
    },
    Subject {
        name: "workload-conservation",
        category: Category::Conservation,
        description: "Run-wide work accounting: cost_log prefill/decode/FFN/KV actuals vs \
                      request_slo per-request expected (pass/fail).",
        report_name: "workload_conservation_report.json",
        payload_name: "workload_conservation_checks.json",
        applies: Applies::All,
        scope: Scope::Run,
    },
    Subject {
        name: "kv-occupancy",
        category: Category::Kv,
        description: "Per-pool KV-cache occupancy over time (active / projected-peak / promised tokens, \
                      and as a fraction of run_meta capacity) from the kv_snapshot stream.",
        report_name: "kv_occupancy_report.json",
        payload_name: "kv_occupancy_series.json",
        applies: Applies::All,
        scope: Scope::Run,
    },
    Subject {
        name: "alignment-iteration",
        category: Category::AlignmentIteration,
        description: "Per-vLLM-iteration measured vs timing-predict totals, per-kernel stacked breakdowns, mapping coverage, and kernel inventory.",
        report_name: "alignment_iteration_report.json",
        payload_name: "alignment_iteration_series.json",
        applies: Applies::All,
        scope: Scope::Alignment,
    },
    Subject {
        name: "alignment-workload",
        category: Category::AlignmentWorkload,
        description: "Measured-vs-sim scheduler workload by iteration id: prefill tokens, decode batch size, and scheduled KV tokens.",
        report_name: "alignment_workload_report.json",
        payload_name: "alignment_workload_series.json",
        applies: Applies::All,
        scope: Scope::Alignment,
    },
    Subject {
        name: "alignment-e2e",
        category: Category::AlignmentE2e,
        description: "Measured-vs-simulated raw TTFT/TPOT/E2E distributions and completion throughput; request ids audit completeness only.",
        report_name: "alignment_e2e_report.json",
        payload_name: "alignment_e2e_series.json",
        applies: Applies::All,
        scope: Scope::Alignment,
    },
];

/// Human-readable catalog for `analyze list` — one aligned line per subject:
/// `name  [category]  (applicability)  description`.
pub fn help() -> String {
    let name_w = SUBJECTS.iter().map(|s| s.name.len()).max().unwrap_or(0);
    let cat_w = SUBJECTS
        .iter()
        .map(|s| s.category.label().len())
        .max()
        .unwrap_or(0);
    let mut out = String::from(
        "analyzer subjects — run with `analyze {run|alignment} <log_dir> \
         [subjects...]` (no subjects = all applicable in that scope):\n",
    );
    for s in SUBJECTS {
        out.push_str(&format!(
            "  {:<name_w$}  [{:<cat_w$}]  ({}, {})  {}\n",
            s.name,
            s.category.label(),
            s.scope.label(),
            s.applies.label(),
            s.description,
        ));
    }
    out
}

/// Name → runner. The one place that grows an arm per subject (mapping a CLI
/// string to a typed async fn is irreducible) — but it's a single root function,
/// not per-category boilerplate.
pub async fn run_subject(name: &str, ctx: &SessionContext, dir: &Path) -> Result<(Value, Value)> {
    match name {
        "slo-general" => request::slo::run_slo_general(ctx, dir).await,
        "slo-detailed" => request::slo::run_slo_detailed(ctx, dir).await,
        "throughput" => throughput::segment::run_throughput(ctx, dir).await,
        "utilization" => utilization::series::run_utilization(ctx, dir).await,
        "batch" => batch::composition::run_batch(ctx, dir).await,
        "kernel-throughput" => batch::kernel_throughput::run_kernel_throughput(ctx, dir).await,
        "workload-conservation" => conservation::workload::run_workload(ctx, dir).await,
        "kv-occupancy" => kv::occupancy::run_kv_occupancy(ctx, dir).await,
        "alignment-iteration" => alignment_iteration::run(ctx, dir).await,
        "alignment-workload" => alignment_workload::run(ctx, dir).await,
        "alignment-e2e" => alignment_e2e::run(ctx, dir).await,
        other => bail!("unknown analyzer subject {other:?}"),
    }
}

/// Resolve the subjects to run. Empty `requested` = all (the `run <dir>` default);
/// otherwise the named ones. Either way, subjects whose [`Applies`] gate rejects
/// this run's `deployment` are dropped with a note (best-effort: pointing the
/// analyzer at any run "just works" — agnostic metrics always run, deployment-
/// shaped ones self-select). A requested name that matches no subject is a typo,
/// so it warns loudly with the valid names rather than silently running nothing.
pub fn select(
    requested: &[String],
    deployment: Option<&str>,
    scope: Scope,
) -> Vec<&'static Subject> {
    for name in requested {
        if !SUBJECTS.iter().any(|s| s.name == name) {
            let known: Vec<&str> = SUBJECTS.iter().map(|s| s.name).collect();
            eprintln!(
                "[analyze] unknown subject `{name}`; known: {} (see `analyze list`)",
                known.join(", ")
            );
        }
    }
    SUBJECTS
        .iter()
        .filter(|s| {
            if s.scope != scope {
                return false;
            }
            let wanted = requested.is_empty() || requested.iter().any(|r| r == s.name);
            if !wanted {
                return false;
            }
            if !s.applies.matches(deployment) {
                eprintln!(
                    "[analyze] subject `{}` not applicable to deployment {:?}; skipping",
                    s.name,
                    deployment.unwrap_or("<unknown>")
                );
                return false;
            }
            true
        })
        .collect()
}
