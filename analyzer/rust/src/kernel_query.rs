//! Shared transport to the simulator's `kernel-query` introspection subcommand.
//!
//! Two analyzer consumers shell out to the same protocol: the viz-ui
//! `kernel-throughput` drill-down (`ui_service::kernel_throughput_analysis`,
//! `grid` + `eval`) and the `optimality` subject's per-config peak sidecar
//! (`optimality::grid_peaks`, the `peak` op). Both need the exact same
//! launcher-owned PyO3 environment and simulator-binary discovery, so that lives
//! here once rather than duplicated per caller. Kernel semantics stay in the
//! simulator; this module only locates the binary + launcher project and forwards
//! JSON through the repository's `uv`-managed Python environment.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};
use serde_json::Value;

/// The VibeSim repo root the launcher env + built binaries live under. The
/// analyzer is launched from the repo (directly or via `python -m launcher`), so
/// the current working directory is the honest default; `run_kernel_query`
/// resolves the launcher module against that project with `uv run`.
pub(crate) fn repo_root() -> Result<PathBuf> {
    std::env::current_dir().context("resolve analyzer repository root (current dir)")
}

/// Locate a built `simulator` binary carrying the same PyO3 ABI as the launcher's
/// `uv` environment.
/// Prefers the launcher-built release, then a sibling of the running `analyze`
/// binary, then a debug build.
pub(crate) fn simulator_binary(repo_root: &Path) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    // Release is the launcher-built production binary and therefore carries
    // the same PyO3 ABI as `.venv`; prefer it over an incidental debug build.
    candidates.push(repo_root.join("target/release/simulator"));
    if let Ok(current) = std::env::current_exe() {
        if let Some(directory) = current.parent() {
            candidates.push(directory.join("simulator"));
        }
    }
    candidates.push(repo_root.join("target/debug/simulator"));
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .context("could not find a built simulator beside analyze or under target/{release,debug}")
}

/// Send one `kernel-query` request (`{op, kind, config, ...}`) through the
/// launcher transport (`python -m launcher.kernel_query`) and decode its stdout
/// JSON. The launcher module applies the same PYTHONHOME/PYTHONPATH/
/// LD_LIBRARY_PATH as every other launcher-owned simulator subprocess.
pub(crate) fn run_kernel_query(
    repo_root: &Path,
    simulator: &Path,
    request: Value,
) -> Result<Value> {
    let mut child = Command::new("uv")
        .args(["run", "--project"])
        .arg(repo_root)
        .args(["python", "-m", "launcher.kernel_query", "--simulator"])
        .arg(simulator)
        .current_dir(repo_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| {
            format!(
                "start launcher kernel-query via uv run --project {}",
                repo_root.display()
            )
        })?;
    serde_json::to_writer(
        child
            .stdin
            .as_mut()
            .context("kernel-query stdin was not piped")?,
        &request,
    )
    .context("write kernel-query request")?;
    child
        .stdin
        .take()
        .context("kernel-query stdin disappeared")?
        .flush()
        .context("flush kernel-query request")?;
    let output = child.wait_with_output().context("wait for kernel-query")?;
    if !output.status.success() {
        bail!(
            "kernel-query failed with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    serde_json::from_slice(&output.stdout).context("decode kernel-query stdout")
}
