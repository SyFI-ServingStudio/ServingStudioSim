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
use crate::backend;
use crate::batch;
use crate::breakdown;
use crate::concurrency;
use crate::conservation;
use crate::kv;
use crate::optimality;
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
    /// Best-of-N backend selection over a kernel position's input feature space,
    /// from `cost_log` `slot_input` + `slot_backend` and the manifest candidate
    /// lists. Tier-1, deployment-agnostic (a run without the columns is unavailable).
    Backend,
    /// Run-wide CostTree kernel-time composition by semantic leaf position,
    /// reconstructed from `cost_log` slot lists + matching manifests.
    Breakdown,
    /// Distance from optimal GPU usage as a ladder of idealized lower bounds
    /// (idle / imbalance / batching / communication / hardware), from `cost_log`
    /// + manifests + `run_meta` GPU counts + `gpu/spec.json`. Tier-1.
    Optimality,
    /// Run-wide work-accounting invariants — `cost_log` actuals vs `request_slo`
    /// per-request expected. Tier-1, deployment-agnostic.
    Conservation,
    /// Run-level in-flight request concurrency over time, reconstructed from
    /// request arrival and terminal events in `request_slo`.
    Concurrency,
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
            Category::Backend => "backend",
            Category::Breakdown => "breakdown",
            Category::Optimality => "optimality",
            Category::Conservation => "conservation",
            Category::Concurrency => "concurrency",
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
        description: "Per-worker GPU compute utilization with per-pool averages over time.",
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
        name: "kernel-input-distribution",
        category: Category::Backend,
        description: "Per cost-tree position, the input feature-space distribution colored by which \
                      backend best-of-N selected (raw axes / PCA); one scatter per position. \
                      Unavailable on runs without per-slot backend + input logging.",
        report_name: "kernel_input_distribution_report.json",
        payload_name: "kernel_input_distribution_scatter.json",
        applies: Applies::All,
        scope: Scope::Run,
    },
    Subject {
        name: "kernel-time-share",
        category: Category::Breakdown,
        description: "Kernel-time composition by cost-tree leaf position at overall, per-pool, and per-worker levels (exact for small runs; bounded regular sampling for large runs).",
        report_name: "kernel_time_share_report.json",
        payload_name: "kernel_time_share_composition.json",
        applies: Applies::All,
        scope: Scope::Run,
    },
    Subject {
        name: "optimality",
        category: Category::Optimality,
        description: "Distance from optimal GPU usage as a sub-optimality waterfall (GPU·s): \
                      idle / imbalance / batching / communication / hardware-gap / hardware-optimal, \
                      at cluster / pool / worker / iteration / per-kernel levels; normal runs emit \
                      both batch modes, while `--lock-batch-size` recomputes only the locked variant.",
        report_name: "optimality_report.json",
        payload_name: "optimality_waterfall.json",
        applies: Applies::All,
        scope: Scope::Run,
    },
    Subject {
        name: "concurrency",
        category: Category::Concurrency,
        description: "Run-level in-flight request concurrency over time from request_slo arrival and terminal events.",
        report_name: "concurrency_report.json",
        payload_name: "concurrency_series.json",
        applies: Applies::All,
        scope: Scope::Run,
    },
    Subject {
        name: "request-state",
        category: Category::Concurrency,
        description: "Request-stage populations over time: every category in a conserved cluster stack, plus pending queues at pool and worker levels.",
        report_name: "request_state_report.json",
        payload_name: "request_state_series.json",
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
#[derive(Clone, Copy, Debug, Default)]
pub struct RunOptions {
    pub lock_batch_size: bool,
}

/// Resolve the output pair for one subject invocation. Optimality is the only
/// subject with two durable variants: keeping the original names for unlocked
/// preserves old consumers, while the locked names let launcher publish both
/// counterfactuals from one simulation without either overwriting the other.
pub fn artifact_names(subject: &Subject, options: RunOptions) -> (&'static str, &'static str) {
    if subject.name == "optimality" && options.lock_batch_size {
        (
            "optimality_batch_locked_report.json",
            "optimality_batch_locked_waterfall.json",
        )
    } else {
        (subject.report_name, subject.payload_name)
    }
}

pub async fn run_subject(
    name: &str,
    ctx: &SessionContext,
    dir: &Path,
    options: RunOptions,
) -> Result<(Value, Value)> {
    match name {
        "slo-general" => request::slo::run_slo_general(ctx, dir).await,
        "slo-detailed" => request::slo::run_slo_detailed(ctx, dir).await,
        "throughput" => throughput::segment::run_throughput(ctx, dir).await,
        "utilization" => utilization::series::run_utilization(ctx, dir).await,
        "batch" => batch::composition::run_batch(ctx, dir).await,
        "kernel-throughput" => batch::kernel_throughput::run_kernel_throughput(ctx, dir).await,
        "kernel-input-distribution" => backend::kernel_input_distribution::run(ctx, dir).await,
        "kernel-time-share" => breakdown::kernel_time_share::run(ctx, dir).await,
        "optimality" => optimality::run_optimality(ctx, dir, options.lock_batch_size).await,
        "concurrency" => concurrency::series::run_concurrency(ctx, dir).await,
        "request-state" => concurrency::request_state::run_request_state(ctx, dir).await,
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

#[cfg(test)]
mod tests {
    use super::{artifact_names, RunOptions, SUBJECTS};

    #[test]
    fn optimality_artifact_names_keep_both_batch_modes() {
        let subject = SUBJECTS
            .iter()
            .find(|subject| subject.name == "optimality")
            .expect("optimality subject");
        assert_eq!(
            artifact_names(subject, RunOptions::default()),
            ("optimality_report.json", "optimality_waterfall.json")
        );
        assert_eq!(
            artifact_names(
                subject,
                RunOptions {
                    lock_batch_size: true,
                },
            ),
            (
                "optimality_batch_locked_report.json",
                "optimality_batch_locked_waterfall.json",
            )
        );
    }
}
