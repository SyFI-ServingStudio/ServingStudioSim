use clap::{Parser, Subcommand};

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
    Run,
    BuildCacheOnly,
    ListParams,
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run => println!("run: not yet implemented"),
        Cmd::BuildCacheOnly => println!("build-cache-only: not yet implemented"),
        Cmd::ListParams => println!("list-params: not yet implemented"),
    }
}
