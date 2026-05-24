//! `simulator` binary entry — clap top-level CLI and subcommand dispatch.
//!
//! Per L7 design.md §1.8.2 the surface is three subcommands:
//!   - `run <deployment> <flags>`            — run one sim
//!   - `build-cache-only <deployment> <flags>` — prebuild profile.db, no sim
//!   - `list-params`                          — emit schema JSON (§1.2.7)
//!
//! Deployment selection is a nested clap subcommand (`DeploymentSel`), so flags
//! are statically routed: `run unified --model-config ...`. clap rejects flags
//! that the chosen deployment does not declare.
//!
//! `run` / `build-cache-only` parse fully (proving routing + validation) but
//! exit non-zero with a "pending L7-β" message: the L6 `Flow` / L7-β tick
//! driver that actually execute a sim are not implemented yet. `list-params`
//! is fully functional today and is what the launcher depends on.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::Context;
use clap::{Args, Parser, Subcommand};

use simulator::common::{RequestStore, SharedRequests};
use simulator::deployment::unified::{UnifiedDeployment, UnifiedParams};
use simulator::deployment::Deployment;
use simulator::log::LoggerSession;
use simulator::schema::list_params;
use simulator::sim::{run_sim, TickCfg, TraceFrontend};
use simulator::timing::PerfApiBridge;

// Heap profiling (opt-in, `--features dhat-heap`): dhat's allocator only
// intercepts Rust's `GlobalAlloc`, so Python/torch C-side allocations bypass it
// — the dump is dominated by the sim's own Rust allocations, which is exactly
// what we want to attribute. Writes `dhat-heap.json` when `_dhat` drops at exit.
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[derive(Parser)]
#[command(
    name = "simulator",
    version,
    about = "MLSim — ML serving + training simulator"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run one simulation.
    Run(RunArgs),
    /// Prebuild the profile.db kernel cache, then exit without simulating.
    BuildCacheOnly(RunArgs),
    /// Report how many kernel specs are missing from profile.db (the JIT work a
    /// cache build would do), per kernel, then exit without building or running.
    DryRun(RunArgs),
    /// Print the deployment schema JSON consumed by the launcher (§1.2.7).
    ListParams,
}

/// Shared payload for `run` / `build-cache-only`: pick a deployment, then its
/// flags. Adding a deployment = one `DeploymentSel` variant + one dispatch arm.
#[derive(Args)]
struct RunArgs {
    #[command(subcommand)]
    deployment: DeploymentSel,
}

#[derive(Subcommand)]
enum DeploymentSel {
    /// Co-located worker running the whole model per iteration.
    Unified(UnifiedParams),
}

fn main() -> anyhow::Result<()> {
    // Heap profiler guard (opt-in): starts recording allocations now, dumps
    // `dhat-heap.json` when it drops at the end of main.
    #[cfg(feature = "dhat-heap")]
    let _dhat = dhat::Profiler::new_heap();

    // Timed/leveled logging (heartbeats, cache-build progress, run summary).
    // `RUST_LOG` overrides the default `info` level; e.g. `RUST_LOG=debug`.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run(args) => cmd_run(args.deployment),
        Cmd::BuildCacheOnly(args) => cmd_build_cache(args.deployment),
        Cmd::DryRun(args) => cmd_dry_run(args.deployment),
        Cmd::ListParams => {
            // serde_json::Value serializes infallibly; pretty for `list-params`.
            println!(
                "{}",
                serde_json::to_string_pretty(&list_params()).expect("list_params JSON")
            );
            Ok(())
        }
    }
}

/// `run` — strict bridge (JIT off → fail-fast on missing `profile.db` rows),
/// build the deployment Flow, load the trace, drive the tick loop, log parquet.
fn cmd_run(sel: DeploymentSel) -> anyhow::Result<()> {
    let bridge = PerfApiBridge::new().context("starting the PyO3 perf_api bridge")?;
    let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
    match sel {
        DeploymentSel::Unified(p) => {
            let mut flow = UnifiedDeployment::build(&p, &bridge, Rc::clone(&store))?;
            let mut frontend = TraceFrontend::load(&p.workload.trace_files, p.workload.request_rate)?;
            let mut logger = LoggerSession::open(&p.io.log_dir)?;
            let cfg = TickCfg::new(p.workload.duration_ms, p.workload.run_to_end);
            let cause = run_sim(flow.as_mut(), &store, &mut frontend, &mut logger, &cfg)?;
            // run_sim emits the stats summary (completed/throughput/wall); here we
            // just point at where the parquet logs landed.
            tracing::info!(
                deployment = "unified",
                cause = ?cause,
                log_dir = %p.io.log_dir.display(),
                "run complete"
            );
        }
    }
    Ok(())
}

/// `build-cache-only` — enable JIT profiling, run the L4 cascade so missing
/// kernels are profiled into `profile.db`, then exit before the tick loop.
fn cmd_build_cache(sel: DeploymentSel) -> anyhow::Result<()> {
    let bridge = PerfApiBridge::new().context("starting the PyO3 perf_api bridge")?;
    bridge
        .enable_jit_profiling()
        .context("enabling JIT profiling for cache build")?;
    let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
    match sel {
        DeploymentSel::Unified(p) => {
            let _flow = UnifiedDeployment::build(&p, &bridge, store)?;
        }
    }
    tracing::info!("cache build complete: profile.db populated via JIT");
    Ok(())
}

/// `dry-run` — put the bridge in dry-run mode, run the same build cascade (which
/// then only `count_missing`s, never fits), and print one line per kernel showing
/// how many of its specs are absent from `profile.db` (the JIT work a real cache
/// build would do). Exits before the tick loop. JIT stays off so nothing is
/// profiled — this is a read-only coverage probe.
fn cmd_dry_run(sel: DeploymentSel) -> anyhow::Result<()> {
    let bridge = PerfApiBridge::new().context("starting the PyO3 perf_api bridge")?;
    bridge.enable_dry_run();
    let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
    match sel {
        DeploymentSel::Unified(p) => {
            let _flow = UnifiedDeployment::build(&p, &bridge, store)?;
        }
    }

    let report = bridge.take_dry_run_report();
    let total_missing: usize = report.iter().map(|k| k.missing).sum();
    let total_specs: usize = report.iter().map(|k| k.total).sum();
    println!("dry run: {} kernels", report.len());
    for k in &report {
        println!(
            "  {:<40} ({:<16}) {:>8} / {:<8} missing",
            k.name, k.kind, k.missing, k.total
        );
    }
    println!(
        "total: {total_missing} / {total_specs} specs missing across {} kernels to JIT",
        report.len()
    );
    Ok(())
}
