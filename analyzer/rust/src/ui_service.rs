//! Minimal read-only HTTP wiring between analyzer run directories and viz-ui.
//!
//! This first slice deliberately exposes only `GET /api/v1/runs`. It discovers
//! run directories from server-configured roots; request data is never turned
//! into a filesystem path. Descriptor and artifact reads belong to later,
//! independently reviewable slices.

use std::collections::HashSet;
use std::fs;
use std::net::SocketAddr;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};

const PROTOCOL_VERSION: u32 = 1;

#[derive(Clone, Debug)]
struct ConfiguredRoot {
    ordinal: usize,
    path: PathBuf,
}

#[derive(Clone, Debug)]
struct ServiceState {
    roots: Arc<Vec<ConfiguredRoot>>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum StageStatus {
    NotStarted,
    Pending,
    Complete,
    Failed,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct Lifecycle {
    simulation: StageStatus,
    analysis: StageStatus,
}

#[derive(Debug, Serialize)]
struct RunCatalog {
    protocol_version: u32,
    generated_at: String,
    runs: Vec<RunCatalogEntry>,
}

#[derive(Debug, Serialize)]
struct RunCatalogEntry {
    run_id: String,
    kind: &'static str,
    display_name: String,
    descriptor_href: String,
    lifecycle: Lifecycle,
    updated_at: String,
    #[serde(skip)]
    updated_time: SystemTime,
}

pub(crate) async fn serve(bind: SocketAddr, logs_roots: Vec<PathBuf>) -> Result<()> {
    let roots = configure_logs_roots(logs_roots)?;
    let state = ServiceState {
        roots: Arc::new(roots),
    };
    let app = Router::new()
        .route("/api/v1/runs", get(list_runs))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("bind analyzer UI service to {bind}"))?;
    eprintln!("[analyze] serving run catalog at http://{bind}/api/v1/runs");
    axum::serve(listener, app)
        .await
        .context("serve analyzer run catalog")
}

fn configure_logs_roots(logs_roots: Vec<PathBuf>) -> Result<Vec<ConfiguredRoot>> {
    let mut seen = HashSet::new();
    let mut roots = Vec::new();
    for requested in logs_roots {
        let path = requested
            .canonicalize()
            .with_context(|| format!("canonicalize logs root {}", requested.display()))?;
        if !path.is_dir() {
            anyhow::bail!("logs root is not a directory: {}", path.display());
        }
        if seen.insert(path.clone()) {
            roots.push(ConfiguredRoot {
                ordinal: roots.len(),
                path,
            });
        }
    }
    Ok(roots)
}

async fn list_runs(State(state): State<ServiceState>) -> Response {
    let roots = Arc::clone(&state.roots);
    match tokio::task::spawn_blocking(move || build_catalog(&roots)).await {
        Ok(Ok(catalog)) => Json(catalog).into_response(),
        Ok(Err(error)) => catalog_error(error),
        Err(error) => catalog_error(anyhow::anyhow!("catalog worker failed: {error}")),
    }
}

fn catalog_error(error: anyhow::Error) -> Response {
    eprintln!("[analyze] run discovery failed: {error:#}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "code": "run_discovery_failed",
            "detail": "The configured logs roots could not be scanned."
        })),
    )
        .into_response()
}

fn build_catalog(roots: &[ConfiguredRoot]) -> Result<RunCatalog> {
    let mut runs = discover_runs(roots)?;
    runs.sort_by(|left, right| {
        right
            .updated_time
            .cmp(&left.updated_time)
            .then_with(|| left.run_id.cmp(&right.run_id))
    });
    Ok(RunCatalog {
        protocol_version: PROTOCOL_VERSION,
        generated_at: timestamp(SystemTime::now()),
        runs,
    })
}

fn discover_runs(roots: &[ConfiguredRoot]) -> Result<Vec<RunCatalogEntry>> {
    let mut runs = Vec::new();
    for root in roots {
        discover_runs_under_root(root, &mut runs)?;
    }
    Ok(runs)
}

fn discover_runs_under_root(root: &ConfiguredRoot, runs: &mut Vec<RunCatalogEntry>) -> Result<()> {
    let mut pending = vec![root.path.clone()];
    while let Some(directory) = pending.pop() {
        if is_run_directory(&directory) {
            let relative = directory
                .strip_prefix(&root.path)
                .expect("discovery only queues paths below its configured root");
            let run_id = opaque_run_id(root.ordinal, relative);
            let updated_time = run_updated_at(&directory);
            runs.push(RunCatalogEntry {
                descriptor_href: format!("runs/{run_id}/descriptor"),
                run_id,
                kind: "simulation",
                display_name: display_name(&root.path, relative),
                lifecycle: run_lifecycle(&directory),
                updated_at: timestamp(updated_time),
                updated_time,
            });
            // Raw parquet and generated artifacts live below a run. Once found,
            // do not recursively scan that potentially large subtree.
            continue;
        }

        let mut children = fs::read_dir(&directory)
            .with_context(|| format!("read logs directory {}", directory.display()))?
            .filter_map(|entry| entry.ok())
            .collect::<Vec<_>>();
        children.sort_by_key(|entry| entry.file_name());
        for entry in children.into_iter().rev() {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_symlink() || !file_type.is_dir() {
                continue;
            }
            let name = entry.file_name();
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            pending.push(entry.path());
        }
    }
    Ok(())
}

fn is_run_directory(directory: &Path) -> bool {
    fs::symlink_metadata(directory.join("raw/params.json"))
        .map(|metadata| metadata.file_type().is_file())
        .unwrap_or(false)
}

fn run_lifecycle(run: &Path) -> Lifecycle {
    let simulation = if regular_file(&run.join(".failed")) {
        StageStatus::Failed
    } else if regular_file(&run.join(".complete")) {
        StageStatus::Complete
    } else {
        StageStatus::Pending
    };
    let analysis = if regular_file(&run.join("reports/analyzer_timing.json")) {
        StageStatus::Complete
    } else {
        StageStatus::NotStarted
    };
    Lifecycle {
        simulation,
        analysis,
    }
}

fn regular_file(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|metadata| metadata.file_type().is_file())
        .unwrap_or(false)
}

fn run_updated_at(run: &Path) -> SystemTime {
    [
        "raw/params.json",
        "raw/run_meta.json",
        ".complete",
        ".failed",
        "reports/analyzer_timing.json",
    ]
    .into_iter()
    .filter_map(|relative| fs::metadata(run.join(relative)).ok()?.modified().ok())
    .max()
    .unwrap_or(UNIX_EPOCH)
}

fn opaque_run_id(root_ordinal: usize, relative: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vibesim-analyzer-run-id-v1\0");
    hasher.update((root_ordinal as u64).to_be_bytes());
    hasher.update(b"\0");
    hasher.update(relative.as_os_str().as_encoded_bytes());
    format!("r_{:x}", hasher.finalize())
}

fn display_name(root: &Path, relative: &Path) -> String {
    let components = relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if components.is_empty() {
        root.file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string())
    } else {
        components.join("/")
    }
}

fn timestamp(time: SystemTime) -> String {
    DateTime::<Utc>::from(time).to_rfc3339_opts(SecondsFormat::Secs, true)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::Value;
    use tempfile::TempDir;

    use super::*;

    fn make_run(path: &Path, complete: bool, analyzed: bool) {
        fs::create_dir_all(path.join("raw")).expect("create raw directory");
        fs::write(path.join("raw/params.json"), "{}").expect("write params");
        if complete {
            fs::write(path.join(".complete"), "").expect("write completion marker");
        }
        if analyzed {
            fs::create_dir_all(path.join("reports")).expect("create reports directory");
            fs::write(path.join("reports/analyzer_timing.json"), "{}").expect("write timing");
        }
    }

    #[test]
    fn empty_logs_root_has_empty_protocol_v1_catalog() {
        let temporary = TempDir::new().expect("temporary logs root");
        let roots = configure_logs_roots(vec![temporary.path().to_path_buf()])
            .expect("configure logs root");

        let value = serde_json::to_value(build_catalog(&roots).expect("build catalog"))
            .expect("serialize catalog");

        assert_eq!(value["protocol_version"], 1);
        assert_eq!(value["runs"], Value::Array(Vec::new()));
        assert!(value["generated_at"].as_str().is_some());
    }

    #[test]
    fn discovers_nested_runs_and_ignores_directory_shells() {
        let temporary = TempDir::new().expect("temporary logs root");
        let completed = temporary.path().join("sweep/tp4/simulation");
        let pending = temporary.path().join("sweep/tp8/simulation");
        make_run(&completed, true, true);
        make_run(&pending, false, false);
        fs::create_dir_all(temporary.path().join("empty/folder")).expect("create shell");
        let roots = configure_logs_roots(vec![temporary.path().to_path_buf()])
            .expect("configure logs root");

        let catalog = build_catalog(&roots).expect("build catalog");

        assert_eq!(catalog.runs.len(), 2);
        let completed = catalog
            .runs
            .iter()
            .find(|run| run.display_name == "sweep/tp4/simulation")
            .expect("completed run");
        let completed_value = serde_json::to_value(completed).expect("serialize completed run");
        assert_eq!(completed_value["lifecycle"]["simulation"], "complete");
        assert_eq!(completed_value["lifecycle"]["analysis"], "complete");
        assert!(completed.run_id.starts_with("r_"));
        assert_eq!(
            completed.descriptor_href,
            format!("runs/{}/descriptor", completed.run_id)
        );

        let pending = catalog
            .runs
            .iter()
            .find(|run| run.display_name == "sweep/tp8/simulation")
            .expect("pending run");
        let pending_value = serde_json::to_value(pending).expect("serialize pending run");
        assert_eq!(pending_value["lifecycle"]["simulation"], "pending");
        assert_eq!(pending_value["lifecycle"]["analysis"], "not_started");
    }

    #[cfg(unix)]
    #[test]
    fn does_not_follow_directory_symlinks() {
        use std::os::unix::fs::symlink;

        let root = TempDir::new().expect("temporary logs root");
        let outside = TempDir::new().expect("outside directory");
        make_run(&outside.path().join("escaped-run"), true, true);
        symlink(outside.path(), root.path().join("linked-outside")).expect("create symlink");
        let roots =
            configure_logs_roots(vec![root.path().to_path_buf()]).expect("configure logs root");

        let catalog = build_catalog(&roots).expect("build catalog");

        assert!(catalog.runs.is_empty());
    }
}
