//! `analyze` — MLSim post-sim analyzer binary.
//!
//! Reads a run's `raw/*.parquet` via DataFusion, computes metrics, and writes a
//! *report* JSON (numbers, into `reports/`) + a *payload* JSON (plot arrays for
//! the Python plotter, into `payloads/`). The Python side (`analyzer/python`)
//! renders PNGs from the payloads — it never touches parquet.
//!
//! One verb: `analyze run <log_dir> [subjects...]`. With no subjects it runs
//! every subject applicable to the run's deployment (mirrors the Python renderer
//! defaulting to all). The subject catalog lives in [`registry`].

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, Subcommand};
use serde_json::json;

mod batch;
mod breakdown;
mod cdf;
mod io;
mod perfetto;
mod registry;
mod request;
mod session;
mod throughput;
mod trace;
mod utilization;

use io::{payload_path, read_deployment, report_path, write_json, SCHEMA_VERSION};
use session::build_session;

#[derive(Parser, Debug)]
#[command(name = "analyze", about = "MLSim post-sim parquet analyzer")]
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
    /// List the available analyzer subjects and what each produces.
    List,
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
        Command::Run { log_dir, subjects } => run(log_dir, subjects).await,
        Command::Trace {
            log_dir,
            regions,
            region_ms,
            max_slices,
        } => {
            let ctx = build_session();
            trace::run(&ctx, &log_dir, regions, region_ms, max_slices).await
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

async fn run(log_dir: PathBuf, subjects: Vec<String>) -> Result<()> {
    let ctx = build_session();
    let deployment = read_deployment(&log_dir);

    // Best-effort per subject: one subject failing (e.g. missing parquet) must
    // not sink the others, so an `analyze run <dir>` over a whole catalog still
    // emits everything it can. The process still exits 0 — the launcher hook
    // treats analysis as best-effort.
    //
    // Timing lives ONLY here (and in the run report below), never in a subject's
    // own report — those stay deterministic so two runs diff cleanly. The single
    // dispatch loop means timing every subject is one place, not per-subject.
    let run_start = Instant::now();
    let mut subject_runs = Vec::new();
    for subject in registry::select(&subjects, deployment.as_deref()) {
        let started = Instant::now();
        let status = match registry::run_subject(subject.name, &ctx, &log_dir).await {
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
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
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
