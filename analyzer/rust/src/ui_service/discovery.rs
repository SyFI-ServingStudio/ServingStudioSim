//! Recursive run discovery and single-flight catalog caching.

use super::*;

const OPAQUE_RUN_ID_PREFIX: &str = "r_";
// `opaque_run_id` hashes with SHA-256 and `hex` emits two lowercase digits per
// byte. Keep validation exact so arbitrary path segments cannot trigger scans.
const OPAQUE_RUN_ID_HEX_LENGTH: usize = 64;

pub(super) fn discovery_problem(error: anyhow::Error) -> ApiProblem {
    eprintln!("[analyze] run discovery failed: {error:#}");
    ApiProblem::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "run_discovery_failed",
        "Run discovery failed",
        "One or more configured logs roots could not be scanned.",
    )
}

pub(super) async fn discover_runs_cached(state: &Arc<ServiceState>) -> Result<Vec<RunRecord>> {
    refresh_discovery(state).await
}

pub(super) async fn refresh_discovery(state: &Arc<ServiceState>) -> Result<Vec<RunRecord>> {
    let observed_refresh = {
        let cache = state
            .discovery
            .read()
            .map_err(|_| anyhow::anyhow!("run discovery cache lock is poisoned"))?;
        if cache
            .refreshed_at
            .is_some_and(|refreshed_at| refreshed_at.elapsed() < CATALOG_REFRESH_INTERVAL)
        {
            return Ok(cache.runs.clone());
        }
        cache.refreshed_at
    };

    // Only one request scans the logs tree. Waiters reuse its result rather
    // than immediately launching another 26+ GB traversal.
    let _refresh_guard = state.discovery_refresh.lock().await;
    {
        let cache = state
            .discovery
            .read()
            .map_err(|_| anyhow::anyhow!("run discovery cache lock is poisoned"))?;
        if cache.refreshed_at != observed_refresh
            || cache
                .refreshed_at
                .is_some_and(|refreshed_at| refreshed_at.elapsed() < CATALOG_REFRESH_INTERVAL)
        {
            return Ok(cache.runs.clone());
        }
    }

    let roots = state.roots.clone();
    #[cfg(test)]
    state
        .discovery_scan_count
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let runs = tokio::task::spawn_blocking(move || discover_runs_uncached(&roots))
        .await
        .context("run discovery worker task failed")??;
    let mut cache = state
        .discovery
        .write()
        .map_err(|_| anyhow::anyhow!("run discovery cache lock is poisoned"))?;
    cache.refreshed_at = Some(Instant::now());
    cache.runs = runs.clone();
    Ok(runs)
}

#[cfg(test)]
pub(super) fn discover_runs(state: &ServiceState) -> Result<Vec<RunRecord>> {
    let runs = discover_runs_uncached(&state.roots)?;
    let mut cache = state
        .discovery
        .write()
        .map_err(|_| anyhow::anyhow!("run discovery cache lock is poisoned"))?;
    cache.refreshed_at = Some(Instant::now());
    cache.runs = runs.clone();
    Ok(runs)
}

pub(super) fn discover_runs_uncached(roots: &[ConfiguredRoot]) -> Result<Vec<RunRecord>> {
    let mut runs = Vec::new();
    let mut seen_paths = HashSet::new();
    for root in roots {
        discover_root(root, &mut seen_paths, &mut runs)?;
    }
    runs.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| left.run_id.cmp(&right.run_id))
    });
    Ok(runs)
}

pub(super) fn discover_root(
    root: &ConfiguredRoot,
    seen_paths: &mut HashSet<PathBuf>,
    runs: &mut Vec<RunRecord>,
) -> Result<()> {
    let mut pending = vec![root.path.clone()];
    while let Some(directory) = pending.pop() {
        if let Some(run_path) = canonical_run_path(&root.path, &directory) {
            if seen_paths.insert(run_path.clone()) {
                let relative = run_path
                    .strip_prefix(&root.path)
                    .expect("contained run is relative to configured root");
                runs.push(RunRecord {
                    run_id: opaque_run_id(root.ordinal, relative),
                    display_name: relative_label(relative),
                    root: root.path.clone(),
                    #[cfg(target_os = "linux")]
                    root_directory: Arc::clone(&root.directory),
                    updated_at: run_updated_at(&root.path, &run_path),
                    path: run_path,
                });
            }
            // A run owns potentially huge raw/cache/artifact trees. Nested sweep
            // siblings are discovered from their common parent, never by walking
            // inside a resolved run.
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
            if !file_type.is_dir() || file_type.is_symlink() {
                continue;
            }
            if should_skip_discovery_dir(&entry.file_name().to_string_lossy()) {
                continue;
            }
            pending.push(entry.path());
        }
    }
    Ok(())
}

pub(super) fn canonical_run_path(root: &Path, directory: &Path) -> Option<PathBuf> {
    let run = directory.canonicalize().ok()?;
    if !run.starts_with(root) {
        return None;
    }
    let params = run.join("raw/params.json").canonicalize().ok()?;
    if !params.is_file() || !params.starts_with(root) || !params.starts_with(&run) {
        return None;
    }
    Some(run)
}

pub(super) fn should_skip_discovery_dir(name: &str) -> bool {
    name.starts_with('.')
        || matches!(
            name,
            "raw"
                | "reports"
                | "payloads"
                | "plots"
                | "traces"
                | "git_snapshot"
                | "target"
                | "node_modules"
                | "__pycache__"
        )
}

pub(super) async fn resolve_run(
    state: &Arc<ServiceState>,
    run_id: &str,
) -> Result<RunRecord, ApiProblem> {
    if !is_canonical_run_id(run_id) {
        return Err(ApiProblem::run_not_found(run_id));
    }

    discover_runs_cached(state)
        .await
        .map_err(discovery_problem)?
        .into_iter()
        .find(|run| run.run_id == run_id)
        .ok_or_else(|| ApiProblem::run_not_found(run_id))
}

pub(super) fn is_canonical_run_id(run_id: &str) -> bool {
    let Some(digest) = run_id.strip_prefix(OPAQUE_RUN_ID_PREFIX) else {
        return false;
    };
    digest.len() == OPAQUE_RUN_ID_HEX_LENGTH
        && digest
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

pub(super) fn opaque_run_id(root_ordinal: usize, relative: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vibesim-analyzer-run-id-v1\0");
    hasher.update((root_ordinal as u64).to_be_bytes());
    hasher.update(b"\0");
    hasher.update(relative.as_os_str().as_encoded_bytes());
    let digest = hasher.finalize();
    format!("{OPAQUE_RUN_ID_PREFIX}{}", hex(&digest))
}

pub(super) fn relative_label(relative: &Path) -> String {
    let parts = relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if parts.is_empty() {
        ".".to_string()
    } else {
        parts.join("/")
    }
}

pub(super) fn run_updated_at(root: &Path, run: &Path) -> SystemTime {
    let mut updated = UNIX_EPOCH;
    let mut consider = |relative: &Path| {
        if let Ok(path) = resolve_contained_file(root, run, relative, "run metadata") {
            if let Ok(modified) = path.metadata().and_then(|meta| meta.modified()) {
                updated = updated.max(modified);
            }
        }
    };
    for relative in [
        Path::new("raw/params.json"),
        Path::new("raw/run_meta.json"),
        Path::new("summary.json"),
        Path::new(".complete"),
        Path::new(".failed"),
        Path::new("reports/analyzer_timing.json"),
        Path::new(PIPELINE_STATE_PATH),
    ] {
        consider(relative);
    }
    if let Ok(trace) = select_trace_from(root, run) {
        if let Ok(modified) = trace.metadata().and_then(|meta| meta.modified()) {
            updated = updated.max(modified);
        }
    }
    updated
}
