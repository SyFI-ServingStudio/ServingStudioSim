//! `simulator` binary entry — clap top-level CLI and subcommand dispatch.
//!
//! Per the L7 design, each run-like subcommand takes a path
//! to ONE structured config file (YAML/JSON), which the launcher writes:
//!   - `run <config>`              — run one sim
//!   - `build-cache-only <config>` — prebuild profile.db, no sim
//!   - `dry-run <config>`          — report missing profile.db rows, no sim
//!   - `emit-backends <config>`    — enumerate distinct kernels (JSON), no sim
//!   - `list-params`               — emit the param-schema registry JSON
//!
//! All three run-like subcommands share one parse (`load_config`) → `RunConfig`
//! (a serde enum tagged by `deployment`) → `deployment::build_flow` dispatch.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};

use simulator::common::{RequestStore, SharedRequests};
use simulator::deployment::{build_flow, RunConfig};
use simulator::log::LoggerSession;
use simulator::schema::list_params;
use simulator::sim::{
    run_sim, ArrivalSchedule, CapacityLimit, InputFileFormat, InputFileSchema, LoadedTrace,
    SessionDependency, TickCfg, TraceTag,
};
use simulator::timing::PerfApiBridge;
use simulator::timing_predict::PredictMode;

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
    about = "ServingStudio Sim — ML serving + training simulator"
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
    /// Enumerate the distinct kernels this config touches — one JSON record per
    /// `Kernel::build` (`{pool, name, kind, config, backends}`), with NO GPU /
    /// profiling (builds the cost-tree structure only). The launcher renders the
    /// `backends` skeleton + validates a backend map from this. Writes JSON on
    /// stdout (logs go to stderr).
    EmitBackends(RunArgs),
    /// Probe a built kernel's cost cache (cost-model introspection for the
    /// cache-fidelity harness). Reads a JSON request on stdin describing one
    /// kernel by `{kind, config, query_points}`; writes the interpolated
    /// best-of-N metrics per point + the fitted grid axes on stdout.
    KernelQuery,
    /// Predict per-building-block timing offline for a batch of explicit batch
    /// shapes — NO sim/scheduler/trace. Reads a minimal config (one arch selector
    /// `{iter|attn|ffn}` + gpu + a cases_file) and writes the standard
    /// `raw/cost_log` + `cost_manifest` artifacts (one row per case/section, tagged
    /// with `section`/`layer`), which `analyze trace` / `analyze run` consume.
    ///
    /// `--dry-run` builds the model and validates every case without costing any:
    /// it reports the `profile.db` specs a real run would JIT, writes nothing, and
    /// needs no GPU.
    TimingPredict(PredictArgs),
}

#[derive(Args)]
struct PredictArgs {
    /// Path to the predict config (`.yaml` / `.yml` / `.json`).
    config: PathBuf,
    /// Validate and report missing `profile.db` specs; cost and write nothing.
    #[arg(long)]
    dry_run: bool,
}

/// Shared payload for `run` / `build-cache-only` / `dry-run`: a path to one
/// structured config file. The `deployment` tag inside it picks the topology.
#[derive(Args)]
struct RunArgs {
    /// Path to the structured run config (`.yaml` / `.yml` / `.json`).
    config: PathBuf,
}

/// Parse a structured config file. YAML is a JSON superset, so `.json` uses the
/// JSON parser (clearer errors) and everything else uses the YAML parser.
fn load_config(path: &Path) -> Result<RunConfig> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading run config {}", path.display()))?;
    let cfg = if path.extension().and_then(|e| e.to_str()) == Some("json") {
        serde_json::from_str(&text)
            .with_context(|| format!("parsing JSON config {}", path.display()))?
    } else {
        serde_yaml::from_str(&text)
            .with_context(|| format!("parsing YAML config {}", path.display()))?
    };
    Ok(cfg)
}

/// Compact wall-clock log timestamp: `[MM:SS.mmm]` (UTC minute-of-hour). Drops
/// the date/hour/微秒 noise from the default RFC3339 stamp so log lines read as
/// `[17:06.300] INFO …`. Uses `SystemTime` directly to avoid a chrono/time dep.
struct CompactTime;

impl tracing_subscriber::fmt::time::FormatTime for CompactTime {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> std::fmt::Result {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = now.as_secs();
        write!(
            w,
            "[{:02}:{:02}.{:03}]",
            (secs / 60) % 60,
            secs % 60,
            now.subsec_millis()
        )
    }
}

fn main() -> anyhow::Result<()> {
    // Heap profiler guard (opt-in): starts recording allocations now, dumps
    // `dhat-heap.json` when it drops at the end of main.
    #[cfg(feature = "dhat-heap")]
    let _dhat = dhat::Profiler::new_heap();

    // Timed/leveled logging (heartbeats, cache-build progress, run summary).
    // `RUST_LOG` overrides the default `info` level; e.g. `RUST_LOG=debug`.
    // Logs go to stderr so stdout stays pure data for the JSON-emitting
    // subcommands (`kernel-query`, `list-params`).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        // Compact `[MM:SS.mmm]` stamp + no ANSI: the launcher pipes stderr into
        // `stdout.log` (a non-TTY), where the default RFC3339 stamp and color
        // escapes (`\e[2m…\e[0m`) become unreadable noise.
        .with_timer(CompactTime)
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run(args) => cmd_run(&args.config),
        Cmd::BuildCacheOnly(args) => cmd_build_cache(&args.config),
        Cmd::DryRun(args) => cmd_dry_run(&args.config),
        Cmd::EmitBackends(args) => cmd_emit_backends(&args.config),
        Cmd::ListParams => {
            // serde_json::Value serializes infallibly; pretty for `list-params`.
            println!(
                "{}",
                serde_json::to_string_pretty(&list_params()).expect("list_params JSON")
            );
            Ok(())
        }
        Cmd::KernelQuery => simulator::introspect::run_kernel_query(),
        Cmd::TimingPredict(args) => simulator::timing_predict::run_timing_predict(
            &args.config,
            if args.dry_run {
                PredictMode::DryRun
            } else {
                PredictMode::Run
            },
        ),
    }
}

/// `run` — load and type-check the trace, then start the strict bridge (JIT off),
/// build the deployment Flow, drive the tick loop, and log parquet.
fn cmd_run(config: &Path) -> anyhow::Result<()> {
    let cfg = load_config(config)?;

    // Parse into a concrete request family and narrow to the currently
    // supported text path before starting the bridge or building L4.
    let workload = cfg.workload();
    let input_file_format = InputFileFormat::parse(&workload.input_file_format)?;
    let input_file_tags = workload
        .input_file_tags
        .iter()
        .map(|tag| TraceTag::parse(tag))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let input_file_schema = InputFileSchema::new(input_file_format, input_file_tags)?;
    let arrival = ArrivalSchedule::parse(&workload.arrival_mode, workload.request_rate)?;
    let capacity = CapacityLimit::parse(workload.max_concurrency.map(|n| n as usize))?;
    let session_dependency = SessionDependency::parse(&workload.session_dependency)?;
    let loaded_trace = LoadedTrace::load(
        &workload.trace_files,
        &input_file_schema,
        arrival,
        capacity,
        session_dependency,
    )?;
    let mut frontend = loaded_trace.into_current_text_frontend()?;
    let tick_cfg = TickCfg::new(
        workload.duration_ms,
        workload.run_to_end,
        workload.tick_dt_us,
    );
    let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
    // Reserve empty id slots; no fake request exists before replay releases it.
    store.borrow_mut().reserve_slots(frontend.expected_count());

    let bridge = PerfApiBridge::new().context("starting the PyO3 perf_api bridge")?;
    let mut flow = build_flow(&cfg, &bridge, Rc::clone(&store))?;

    let log_dir = &cfg.io().log_dir;
    let mut logger = LoggerSession::open(
        log_dir,
        cfg.io().log_output_token_times,
        cfg.io().log_stage_transitions,
    )?;
    // Run-level GPU facts sidecar (L7): written before the tick loop so the
    // analyzer can normalize per-GPU even if the run later fails.
    simulator::log::write_run_meta(log_dir, &flow.cluster().borrow(), cfg.stage_vocab())?;
    let summary = run_sim(flow.as_mut(), &store, &mut frontend, &mut logger, &tick_cfg)?;
    // run_sim emits the stats summary (completed/throughput/wall) to the log;
    // persist the structured form as `<log_dir>/summary.json` for regression
    // tests + the aggregator, then point at the parquet logs.
    summary.write_json(log_dir)?;
    tracing::info!(
        cause = ?summary.cause,
        log_dir = %log_dir.display(),
        "run complete"
    );
    Ok(())
}

/// `build-cache-only` — collect every miss in one walk, then measure them all in
/// one pass, then exit before the tick loop.
///
/// This used to enable JIT and let the L4 cascade profile on demand, node by
/// node. An instrumented tp4/ep4/spec5 fill showed the cost of that: 23
/// `(kind, backend)` units became 55 separate profiling calls (`elementwise` 14
/// times, `single_gemm` 11), each fanned across four GPUs and joined, for 208
/// worker launches and 208 cold JIT/autotune caches. Nothing overlapped across
/// calls, so all four cards idled together 20% of the run. Measured end to end,
/// collecting first and running each unit whole on one card does the same 6308
/// rows in 2595 GPU-seconds instead of 6256.
///
/// The collect walk is the bridge's dry-run mode, which calls only
/// `count_missing` and never fits, so no kernel time is needed to discover what
/// is missing. JIT stays off throughout: nothing is profiled on demand, and
/// `issue_collected` does all the measuring.
fn cmd_build_cache(config: &Path) -> anyhow::Result<()> {
    let cfg = load_config(config)?;
    let bridge = PerfApiBridge::new().context("starting the PyO3 perf_api bridge")?;
    bridge.enable_dry_run();
    bridge
        .begin_collect()
        .context("starting the profiling collect pass")?;
    let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
    let flow = build_flow(&cfg, &bridge, store)?;
    drop(flow);

    // The plan is worth printing before spending GPU time on it: this is the
    // point at which a grid that quietly grew by an order of magnitude is still
    // cheap to notice.
    let report = bridge.take_dry_run_report();
    let total_missing: usize = report.iter().map(|k| k.missing).sum();
    let total_specs: usize = report.iter().map(|k| k.total).sum();
    println!(
        "cache build plan: {} kernels, {total_missing} / {total_specs} specs missing",
        report.len()
    );
    for k in report.iter().filter(|k| k.missing > 0) {
        println!(
            "  {:<40} ({:<16}) {:>8} / {:<8} missing",
            k.name, k.kind, k.missing, k.total
        );
    }
    // Called even at zero: it clears the collector and reports the no-op, and
    // an early return here would leave the collector installed.
    bridge
        .issue_collected(total_missing)
        .context("measuring the collected profiling work")?;
    tracing::info!("cache build complete: profile.db populated");
    Ok(())
}

/// `dry-run` — put the bridge in dry-run mode, run the same build cascade (which
/// then only `count_missing`s, never fits), and print one line per kernel showing
/// how many of its specs are absent from `profile.db` (the JIT work a real cache
/// build would do). Exits before the tick loop. JIT stays off so nothing is
/// profiled — this is a read-only coverage probe.
fn cmd_dry_run(config: &Path) -> anyhow::Result<()> {
    let cfg = load_config(config)?;
    let bridge = PerfApiBridge::new().context("starting the PyO3 perf_api bridge")?;
    bridge.enable_dry_run();
    let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
    let _flow = build_flow(&cfg, &bridge, store)?;

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

/// `emit-backends` — put the bridge in enumerate mode, run the same build cascade
/// (which records one `KernelEnum` per `Kernel::build` and skips all profiling /
/// GPU work), and print the raw records as JSON on stdout. The launcher consumes
/// this to render the per-kernel `backends` skeleton and to validate a backend
/// map (dedup by `(pool, name)`, dtype parsed from `config`, options from the
/// Python capability table). Structural only — no `profile.db`, no CUDA device.
fn cmd_emit_backends(config: &Path) -> anyhow::Result<()> {
    let cfg = load_config(config)?;
    let bridge = PerfApiBridge::new().context("starting the PyO3 perf_api bridge")?;
    bridge.enable_enumerate();
    let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
    let _flow = build_flow(&cfg, &bridge, store)?;

    let report = bridge.take_enum_report();
    // Pure data on stdout (logs are on stderr) so the launcher can parse it.
    println!(
        "{}",
        serde_json::to_string(&report).expect("KernelEnum records serialize")
    );
    Ok(())
}
