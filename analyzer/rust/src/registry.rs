//! The analyzer's subject catalog — one flat, root-level table for **every**
//! category. A *subject* is one analysis that emits a `(report, payload)` pair;
//! a *category* groups subjects sharing an analytical grain + input source (see
//! `docs/analyzer.md`). The registry is intentionally NOT per-category: keeping
//! it flat means adding a metric in any category is one [`SUBJECTS`] row + one
//! [`run_subject`] arm, never a new per-category table/dispatch to repeat.

use std::path::Path;

use anyhow::{bail, Result};
use datafusion::prelude::SessionContext;
use serde_json::Value;

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
}

impl Category {
    fn label(self) -> &'static str {
        match self {
            Category::Request => "request",
            Category::Throughput => "throughput",
            Category::Utilization => "utilization",
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
}

/// The whole catalog. Adding a metric = one row here + one arm in [`run_subject`]
/// + the subject's module under `src/<category>/`.
pub const SUBJECTS: &[Subject] = &[
    Subject {
        name: "slo",
        category: Category::Request,
        description: "Request/session latency SLOs: TTFT / TPOT / ITL / E2E + session E2E CDFs.",
        report_name: "slo_report.json",
        payload_name: "slo_cdf.json",
        applies: Applies::All,
    },
    Subject {
        name: "throughput",
        category: Category::Throughput,
        description: "Per-GPU prefill / decode / total tokens-per-second, per time segment.",
        report_name: "throughput_report.json",
        payload_name: "throughput_segments.json",
        applies: Applies::All,
    },
    Subject {
        name: "utilization",
        category: Category::Utilization,
        description: "Per-pool GPU compute utilization (fraction of workers busy) over time.",
        report_name: "utilization_report.json",
        payload_name: "utilization_series.json",
        applies: Applies::All,
    },
];

/// Human-readable catalog for `analyze list` — one aligned line per subject:
/// `name  [category]  (applicability)  description`.
pub fn help() -> String {
    let name_w = SUBJECTS.iter().map(|s| s.name.len()).max().unwrap_or(0);
    let cat_w = SUBJECTS.iter().map(|s| s.category.label().len()).max().unwrap_or(0);
    let mut out = String::from(
        "analyzer subjects — run with `analyze run <log_dir> [subjects...]` \
         (no subjects = all applicable):\n",
    );
    for s in SUBJECTS {
        out.push_str(&format!(
            "  {:<name_w$}  [{:<cat_w$}]  ({})  {}\n",
            s.name,
            s.category.label(),
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
        "slo" => request::slo::run_slo(ctx, dir).await,
        "throughput" => throughput::segment::run_throughput(ctx, dir).await,
        "utilization" => utilization::series::run_utilization(ctx, dir).await,
        other => bail!("unknown analyzer subject {other:?}"),
    }
}

/// Resolve the subjects to run. Empty `requested` = all (the `run <dir>` default);
/// otherwise the named ones. Either way, subjects whose [`Applies`] gate rejects
/// this run's `deployment` are dropped with a note (best-effort: pointing the
/// analyzer at any run "just works" — agnostic metrics always run, deployment-
/// shaped ones self-select). A requested name that matches no subject is a typo,
/// so it warns loudly with the valid names rather than silently running nothing.
pub fn select(requested: &[String], deployment: Option<&str>) -> Vec<&'static Subject> {
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
