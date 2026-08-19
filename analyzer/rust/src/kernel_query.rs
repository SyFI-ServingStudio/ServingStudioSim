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

/// The file that marks a directory as a VibeSim checkout. It is the labeler
/// module the optimality floors spawn, so a directory holding it also holds the
/// `model/config`, `model/work/location_maps`, `gpu/spec.json`, and `uv` project
/// that every checkout-owned asset below is read from.
const CHECKOUT_MARKER: &str = "model/work/floors.py";

/// The checkout that owns `log_dir` — the working tree whose `model/` and `gpu/`
/// describe *that* result, which is not necessarily the one the analyzer runs
/// from.
///
/// `analyze serve` resolves results through a workspace registry spanning many
/// checkouts (`main/`, every `wt-*/` worktree, every managed `w_*/repo`), while
/// [`repo_root`] is just the service's own cwd. Reading a foreign run's assets
/// from the service cwd degrades silently: a run whose arch exists only on a
/// worktree branch loses its R6/R7 necessary-work floors behind a caveat,
/// because the labeler subprocess cannot open the `model/config/*.json` named
/// (repo-relatively) in that run's `raw/params.json` and no location map
/// matches its arch. Walk up from the log directory instead, and fall back to
/// the service cwd only when `log_dir` sits outside any checkout.
pub(crate) fn owning_repo_root(log_dir: &Path) -> Result<PathBuf> {
    let absolute = if log_dir.is_absolute() {
        log_dir.to_path_buf()
    } else {
        repo_root()?.join(log_dir)
    };
    // Canonicalize so a registry-relative `../../wt-topic/logs` still has real
    // ancestors to walk; keep the literal path when the directory is gone.
    let absolute = absolute.canonicalize().unwrap_or(absolute);
    absolute
        .ancestors()
        .find(|ancestor| ancestor.join(CHECKOUT_MARKER).is_file())
        .map(Path::to_path_buf)
        .map_or_else(repo_root, Ok)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Lay out `<root>/model/work/floors.py` plus a nested log directory, the
    /// shape every checkout shares.
    fn checkout_with_log_dir(root: &Path, log_dir_suffix: &str) -> PathBuf {
        std::fs::create_dir_all(root.join("model/work")).expect("create labeler directory");
        std::fs::write(root.join(CHECKOUT_MARKER), "").expect("write checkout marker");
        let log_dir = root.join(log_dir_suffix);
        std::fs::create_dir_all(&log_dir).expect("create log directory");
        log_dir
    }

    #[test]
    fn a_log_dir_resolves_to_the_checkout_that_owns_it_not_the_process_cwd() {
        let workspace = tempfile::tempdir().expect("temp workspace");
        let worktree = workspace.path().join("wt-topic");
        let log_dir = checkout_with_log_dir(&worktree, "logs/run_0");
        // Nested runs (a sweep member, say) still resolve to the same checkout.
        let nested = checkout_with_log_dir(&worktree, "logs/sweep_0/member_1");

        let expected = worktree.canonicalize().expect("canonicalize worktree");
        assert_eq!(owning_repo_root(&log_dir).expect("resolve root"), expected);
        assert_eq!(
            owning_repo_root(&nested).expect("resolve nested root"),
            expected
        );
    }

    #[test]
    fn the_innermost_checkout_wins_over_an_enclosing_one() {
        let workspace = tempfile::tempdir().expect("temp workspace");
        let outer = workspace.path().join("main");
        checkout_with_log_dir(&outer, "logs");
        let inner = outer.join("agent-workspaces/w_x/repo");
        let log_dir = checkout_with_log_dir(&inner, "logs/run_0");

        assert_eq!(
            owning_repo_root(&log_dir).expect("resolve root"),
            inner.canonicalize().expect("canonicalize inner checkout")
        );
    }

    #[test]
    fn a_log_dir_outside_any_checkout_falls_back_to_the_process_cwd() {
        let orphan = tempfile::tempdir().expect("temp orphan");
        assert_eq!(
            owning_repo_root(orphan.path()).expect("resolve root"),
            repo_root().expect("process cwd")
        );
    }
}
