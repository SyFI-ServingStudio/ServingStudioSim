//! `analyze` — VibeSim post-run artifact analyzer binary.
//!
//! Reads a run's `raw/*.parquet` via DataFusion, computes metrics, and writes a
//! *report* JSON (numbers, into `reports/`) + a *payload* JSON (plot arrays for
//! the Python plotter, into `payloads/`). The Python side (`analyzer/python`)
//! renders PNGs from the payloads — it never touches parquet.
//!
//! `analyze run <log_dir> [subjects...]` computes simulator subjects;
//! `analyze alignment <analysis_log_dir> [subjects...]` computes paired measured
//! subjects from an `alignment_manifest.json`. Both use the one flat catalog in
//! [`registry`], separated by its source [`registry::Scope`] gate.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use serde_json::json;

mod alignment_e2e;
mod alignment_input;
mod alignment_iteration;
mod alignment_workload;
mod backend;
mod batch;
mod breakdown;
mod cdf;
mod conservation;
mod io;
mod kv;
mod pca;
mod perfetto;
mod registry;
mod request;
mod session;
mod throughput;
mod trace;
mod ui_service;
mod utilization;

use io::{payload_path, read_deployment, report_path, write_json, SCHEMA_VERSION};
use session::build_session;

#[derive(Parser, Debug)]
#[command(name = "analyze", about = "VibeSim post-run artifact analyzer")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run analyzer subjects over a run dir. With no subjects, runs all that are
    /// applicable to the run's deployment.
    Run {
        /// Run directory (holds `raw/*.parquet`); outputs land in its `reports/`
        /// and `payloads/` subdirs.
        log_dir: PathBuf,
        /// Subject names to run (e.g. `slo`); empty = all applicable.
        subjects: Vec<String>,
    },
    /// Accept the dedicated alignment-analysis artifact directory.
    Alignment {
        /// Analysis directory containing `alignment_manifest.json`; outputs
        /// land in this directory's report/payload dirs.
        analysis_log_dir: PathBuf,
        /// Alignment subject names; empty = all alignment subjects.
        subjects: Vec<String>,
    },
    /// List the available analyzer subjects and what each produces.
    List,
    /// Serve the read-only protocol-v1 run catalog for viz-ui.
    Serve {
        /// Logs root to scan recursively. Repeat for additional roots.
        #[arg(long = "logs-root", required = true)]
        logs_roots: Vec<PathBuf>,
        /// Loopback listener used by the viz-ui development proxy.
        #[arg(long, default_value = "127.0.0.1:8787")]
        bind: SocketAddr,
    },
    /// Export a Perfetto per-kernel timeline (`traces/<prefix>.pftrace.gz`).
    /// Samples `regions` evenly-spaced contiguous windows of `region_ms` each
    /// across the run and concatenates them; open in ui.perfetto.dev. A separate
    /// verb (not a subject) — the `analyze run` catalog is untouched.
    Trace {
        /// Run directory (holds `raw/cost_log/` + `raw/cost_manifest/`).
        log_dir: PathBuf,
        /// Number of evenly-spaced sample windows.
        #[arg(long, default_value_t = 16)]
        regions: usize,
        /// Duration (ms) of each sample window.
        #[arg(long, default_value_t = 200.0)]
        region_ms: f64,
        /// Cap on total slice pairs (enforced at iteration granularity).
        #[arg(long, default_value_t = 200_000)]
        max_slices: usize,
        /// Expand `Max` (parallel) branches, each on its own child-track lane
        /// (the legacy view). Default (off) collapses each `Max` to its critical
        /// (bottleneck) branch inline, keeping compute on a single lane.
        #[arg(long)]
        expanded: bool,
    },
    /// Render a human-readable cost tree (time + percentage) per iteration to
    /// `reports/iter_breakdown.ans` from the standard `cost_log` + `cost_manifest`.
    /// Arch input header + clean tree (no kernel config/shapes); the critical path
    /// is flagged with a `▸` gutter. ANSI-colored by default (yellow title bar, timing
    /// cell tinted by node type — non-leaf blue, leaf white — bold critical path) —
    /// `cat` it in a terminal; pass `--no-color` for a clean editor view. A verb (not
    /// a subject): its output is text, not JSON.
    GenIterBreakdown {
        /// Run directory (holds `raw/cost_log/` + `raw/cost_manifest/`).
        log_dir: PathBuf,
        /// Render only this iteration (by `iter_id`); default = every row.
        #[arg(long)]
        iter: Option<u64>,
        /// Cap on iterations rendered (large real runs have thousands of rows).
        #[arg(long, default_value_t = 32)]
        max_iters: usize,
        /// Emit plain text (no ANSI) so the file reads cleanly in an editor; default
        /// is colored (yellow title bar, timing non-leaf blue / leaf white, bold crit).
        #[arg(long)]
        no_color: bool,
    },
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    match Cli::parse().command {
        Command::List => {
            print!("{}", registry::help());
            Ok(())
        }
        Command::Serve { logs_roots, bind } => ui_service::serve(bind, logs_roots).await,
        Command::Run { log_dir, subjects } => run(log_dir, subjects).await,
        Command::Alignment {
            analysis_log_dir,
            subjects,
        } => alignment(analysis_log_dir, subjects).await,
        Command::Trace {
            log_dir,
            regions,
            region_ms,
            max_slices,
            expanded,
        } => {
            let ctx = build_session();
            // Trace build wall-time (read → place → write pftrace). Printed here,
            // not inside `trace::run`, to keep timing at the CLI boundary like the
            // `analyze run` subject timings. The windowed SQL scan keeps this small
            // even on multi-million-row cost_logs.
            let started = Instant::now();
            let res = trace::run(&ctx, &log_dir, regions, region_ms, max_slices, expanded).await;
            eprintln!(
                "[analyze] trace built in {:.1} ms",
                started.elapsed().as_secs_f64() * 1e3
            );
            res
        }
        Command::GenIterBreakdown {
            log_dir,
            iter,
            max_iters,
            no_color,
        } => {
            let ctx = build_session();
            breakdown::run(&ctx, &log_dir, iter, max_iters, !no_color).await
        }
    }
}

async fn alignment(analysis_log_dir: PathBuf, subjects: Vec<String>) -> Result<()> {
    let manifest = analysis_log_dir.join("alignment_manifest.json");
    if !manifest.is_file() {
        bail!(
            "alignment manifest not found: {}; run alignment timing-predict first",
            manifest.display()
        );
    }
    let ctx = build_session();
    run_subjects(
        ctx,
        analysis_log_dir,
        subjects,
        None,
        registry::Scope::Alignment,
    )
    .await
}

async fn run(log_dir: PathBuf, subjects: Vec<String>) -> Result<()> {
    let ctx = build_session();
    let deployment = read_deployment(&log_dir);

    // Pre-register the big shared table once so the concurrent subjects below don't
    // each re-open cost_log's per-worker parquet set. (Registration is an idempotent
    // replace in DataFusion, so a subject re-registering is harmless — this just
    // avoids redundant metadata opens.)
    let _ = session::register_cost_log(&ctx, &log_dir).await;

    run_subjects(ctx, log_dir, subjects, deployment, registry::Scope::Run).await
}

async fn run_subjects(
    ctx: datafusion::prelude::SessionContext,
    log_dir: PathBuf,
    subjects: Vec<String>,
    deployment: Option<String>,
    scope: registry::Scope,
) -> Result<()> {
    // Best-effort AND concurrent: each subject is an independent read over the shared
    // read-only ctx (SessionContext is Send+Sync+Clone) that writes its own files, so
    // on a many-core box overlapping them hides the long poles (batch / workload /
    // kernel-throughput) behind each other instead of summing. One subject failing or
    // panicking must not sink the rest. Tasks finish out of order but are reported in
    // catalog order for a stable log; per-subject timing is still the subject's own
    // wall (now overlapping), and `total_elapsed_ms` is the concurrent wall.
    let run_start = Instant::now();
    let selected = registry::select(&subjects, deployment.as_deref(), scope);
    let mut set = tokio::task::JoinSet::new();
    for (idx, subject) in selected.iter().enumerate() {
        let ctx = ctx.clone();
        let log_dir = log_dir.clone();
        let name = subject.name;
        set.spawn(async move {
            let started = Instant::now();
            let res = registry::run_subject(name, &ctx, &log_dir).await;
            (idx, res, started.elapsed().as_secs_f64() * 1e3)
        });
    }
    let mut results: Vec<Option<(Result<(serde_json::Value, serde_json::Value)>, f64)>> =
        (0..selected.len()).map(|_| None).collect();
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((idx, res, elapsed_ms)) => results[idx] = Some((res, elapsed_ms)),
            Err(e) => eprintln!("[analyze] a subject task panicked: {e}"),
        }
    }

    let mut subject_runs = Vec::new();
    for (idx, subject) in selected.iter().enumerate() {
        let (res, elapsed_ms) = match results[idx].take() {
            Some(r) => r,
            None => continue, // task panicked (already logged)
        };
        let status = match res {
            Ok((report, payload)) => {
                write_json(&report_path(&log_dir, subject.report_name), &report)?;
                write_json(&payload_path(&log_dir, subject.payload_name), &payload)?;
                "ok"
            }
            Err(e) => {
                eprintln!("[analyze] subject `{}` failed: {e:#}", subject.name);
                "failed"
            }
        };
        eprintln!("[analyze] {} {status} in {elapsed_ms:.1} ms", subject.name);
        subject_runs.push(json!({
            "name": subject.name,
            "status": status,
            "elapsed_ms": elapsed_ms,
        }));
    }

    // Run-level summary: which subjects ran, their status + wall time, the total.
    // A run-meta artifact (not a metric report), so its non-determinism is
    // isolated here. The Python renderer ignores it (no payload to draw).
    let run_report = json!({
        "schema_version": SCHEMA_VERSION,
        "log_dir": log_dir.display().to_string(),
        "deployment": deployment,
        "subjects": subject_runs,
        "total_elapsed_ms": run_start.elapsed().as_secs_f64() * 1e3,
    });
    write_json(&report_path(&log_dir, "analyzer_timing.json"), &run_report)?;
    Ok(())
}
