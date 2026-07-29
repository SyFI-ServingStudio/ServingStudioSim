//! Configured logs roots, recursive run discovery, and opaque-id resolution.

use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::RunNotFound;

#[derive(Clone, Debug)]
pub(super) struct ConfiguredRoot {
    workspace_id: String,
    path: PathBuf,
}

impl ConfiguredRoot {
    pub(super) fn workspace_id(&self) -> &str {
        &self.workspace_id
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum StageStatus {
    NotStarted,
    Pending,
    Complete,
    Failed,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(super) struct Lifecycle {
    pub(super) simulation: StageStatus,
    pub(super) analysis: StageStatus,
}

#[derive(Clone, Debug)]
pub(super) struct DiscoveredRun {
    pub(super) workspace_id: String,
    pub(super) run_id: String,
    pub(super) display_name: String,
    pub(super) path: PathBuf,
    pub(super) lifecycle: Lifecycle,
    pub(super) updated_time: SystemTime,
}

pub(super) fn configure_logs_roots(logs_roots: Vec<PathBuf>) -> Result<Vec<ConfiguredRoot>> {
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
                workspace_id: format!("root_{}", roots.len()),
                path,
            });
        }
    }
    Ok(roots)
}

#[derive(Debug, Deserialize)]
struct WorkspaceRegistry {
    schema_version: u32,
    workspaces: Vec<WorkspaceRegistryEntry>,
}

#[derive(Debug, Deserialize)]
struct WorkspaceRegistryEntry {
    workspace_id: String,
    state: String,
    logs_root: PathBuf,
}

/// Load the dynamic workspace registry on every catalog/resource request.
///
/// Workspace identity, not registry order, is part of each opaque ID. Adding or
/// reordering roots therefore cannot invalidate a previously emitted URL.
pub(super) fn configure_workspace_registry(path: &Path) -> Result<Vec<ConfiguredRoot>> {
    let bytes =
        fs::read(path).with_context(|| format!("read workspace registry {}", path.display()))?;
    let registry: WorkspaceRegistry = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse workspace registry {}", path.display()))?;
    if registry.schema_version != 1 {
        anyhow::bail!(
            "unsupported workspace registry schema_version {}",
            registry.schema_version
        );
    }
    let registry_dir = path
        .parent()
        .context("workspace registry path has no parent directory")?;
    let mut seen_ids = HashSet::new();
    let mut seen_paths = HashSet::new();
    let mut roots = Vec::new();
    for entry in registry.workspaces {
        if entry.state == "archived" {
            continue;
        }
        if entry.state != "active" {
            anyhow::bail!(
                "workspace {} has unsupported state {:?}",
                entry.workspace_id,
                entry.state
            );
        }
        validate_workspace_id(&entry.workspace_id)?;
        if !seen_ids.insert(entry.workspace_id.clone()) {
            anyhow::bail!("duplicate workspace id in registry: {}", entry.workspace_id);
        }
        let requested = if entry.logs_root.is_absolute() {
            entry.logs_root
        } else {
            registry_dir.join(entry.logs_root)
        };
        let canonical = match requested.canonicalize() {
            Ok(canonical) => canonical,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                // A workspace exists before its first simulation creates
                // repo/logs. Registry reloads are request-scoped, so omitting
                // this root now makes it discoverable automatically once the
                // Launcher creates the directory.
                continue;
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("canonicalize logs root {}", requested.display()));
            }
        };
        if !canonical.is_dir() {
            anyhow::bail!("logs root is not a directory: {}", canonical.display());
        }
        if !seen_paths.insert(canonical.clone()) {
            anyhow::bail!(
                "multiple active workspaces point at logs root {}",
                canonical.display()
            );
        }
        roots.push(ConfiguredRoot {
            workspace_id: entry.workspace_id,
            path: canonical,
        });
    }
    Ok(roots)
}

pub(super) fn discover_runs(roots: &[ConfiguredRoot]) -> Result<Vec<DiscoveredRun>> {
    let mut runs = Vec::new();
    for root in roots {
        discover_runs_under_root(root, &mut runs)?;
    }
    Ok(runs)
}

fn discover_runs_under_root(root: &ConfiguredRoot, runs: &mut Vec<DiscoveredRun>) -> Result<()> {
    let mut pending = vec![root.path.clone()];
    while let Some(directory) = pending.pop() {
        if is_run_directory(&directory) {
            let relative = directory
                .strip_prefix(&root.path)
                .expect("discovery only queues paths below its configured root");
            runs.push(DiscoveredRun {
                workspace_id: root.workspace_id.clone(),
                run_id: opaque_run_id(&root.workspace_id, relative),
                display_name: display_name(&root.path, relative),
                lifecycle: run_lifecycle(&directory),
                updated_time: run_updated_at(&directory),
                path: directory,
            });
            // Run-owned raw and generated trees are not nested experiment roots.
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
            if file_type.is_symlink()
                || !file_type.is_dir()
                || ignored_directory_name(&entry.file_name())
            {
                continue;
            }
            pending.push(entry.path());
        }
    }
    Ok(())
}

pub(super) fn ignored_directory_name(name: &std::ffi::OsStr) -> bool {
    name.to_string_lossy().starts_with('.') || name == "old-logs"
}

pub(super) fn resolve_run(roots: &[ConfiguredRoot], run_id: &str) -> Result<DiscoveredRun> {
    discover_runs(roots)?
        .into_iter()
        .find(|run| run.run_id == run_id)
        .ok_or_else(|| RunNotFound.into())
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

pub(super) fn regular_file(path: &Path) -> bool {
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

fn opaque_run_id(workspace_id: &str, relative: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vibesim-analyzer-run-id-v2\0");
    hasher.update(workspace_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(relative.as_os_str().as_encoded_bytes());
    format!("r_{:x}", hasher.finalize())
}

fn validate_workspace_id(workspace_id: &str) -> Result<()> {
    if !workspace_id.starts_with("w_")
        || workspace_id.len() > 66
        || !workspace_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        anyhow::bail!("invalid workspace id in registry: {workspace_id:?}");
    }
    Ok(())
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

pub(super) fn timestamp(time: SystemTime) -> String {
    DateTime::<Utc>::from(time).to_rfc3339_opts(SecondsFormat::Secs, true)
}
