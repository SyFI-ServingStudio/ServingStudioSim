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

use clap::{Args, Parser, Subcommand};

use simulator::deployment::unified::UnifiedParams;
use simulator::schema::list_params;

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

impl DeploymentSel {
    fn name(&self) -> &'static str {
        match self {
            DeploymentSel::Unified(_) => "unified",
        }
    }
}

/// Exit code for "parsed fine, but the executing layer isn't built yet".
const EXIT_PENDING_BETA: i32 = 2;

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run(args) => pending_beta("run", args.deployment.name()),
        Cmd::BuildCacheOnly(args) => pending_beta("build-cache-only", args.deployment.name()),
        Cmd::ListParams => {
            // serde_json::Value serializes infallibly; pretty for `list-params`.
            println!(
                "{}",
                serde_json::to_string_pretty(&list_params()).expect("list_params JSON")
            );
        }
    }
}

/// CLI parsed and routed to a real deployment, but the L7-β tick driver / L6
/// Flow that would execute it do not exist yet. Report and exit non-zero so the
/// launcher records a clear failure rather than a silent no-op.
fn pending_beta(cmd: &str, deployment: &str) -> ! {
    eprintln!(
        "simulator {cmd} {deployment}: parsed OK, but the L7-β tick driver is not implemented yet \
         (Flow / tick loop pending). Only `list-params` is functional today."
    );
    std::process::exit(EXIT_PENDING_BETA);
}
