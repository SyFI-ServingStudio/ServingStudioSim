//! Read-only protocol-v1 HTTP boundary for analyzer artifacts.
//!
//! The service deliberately is not a file server. Every public resource is
//! selected from either the fixed core endpoints or [`crate::registry::SUBJECTS`]
//! after an opaque run id has resolved to a canonical, contained run directory.
//! This keeps parquet, cache-build trees, and arbitrary paths outside the HTTP
//! surface even when a caller guesses their on-disk names.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::io::ErrorKind;
use std::net::SocketAddr;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use axum::body::Body;
use axum::extract::{Path as RoutePath, Request, State};
use axum::http::{header, HeaderMap, StatusCode, Uri};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use chrono::{DateTime, SecondsFormat, Utc};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::io::SCHEMA_VERSION;
use crate::registry::{Scope, Subject, SUBJECTS};

const PROTOCOL_VERSION: u32 = 1;
const DEFAULT_GENERATOR_VERSION: &str = concat!("analyzer-", env!("CARGO_PKG_VERSION"));
const MAX_JSON_BYTES: u64 = 128 * 1024 * 1024;
const MAX_TRACE_BYTES: u64 = 512 * 1024 * 1024;
const CATALOG_REFRESH_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Clone, Debug)]
struct ConfiguredRoot {
    ordinal: usize,
    path: PathBuf,
}

#[derive(Debug)]
struct ServiceState {
    roots: Vec<ConfiguredRoot>,
    discovery: Mutex<DiscoveryCache>,
}

#[derive(Clone, Debug, Default)]
struct DiscoveryCache {
    refreshed_at: Option<Instant>,
    runs: Vec<RunRecord>,
}

#[derive(Clone, Debug)]
struct RunRecord {
    run_id: String,
    display_name: String,
    root: PathBuf,
    path: PathBuf,
    updated_at: SystemTime,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum StageStatus {
    NotStarted,
    Pending,
    Complete,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
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
}

#[derive(Debug, Serialize)]
struct ArtifactRef {
    href: &'static str,
    media_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema_version: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct WorkerRef {
    pool_tag: String,
    worker_id: u64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum SubjectState {
    Ready {
        schema_version: u32,
        report_href: String,
        payload_href: String,
    },
    Pending {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    Unavailable {
        reason: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        code: Option<String>,
    },
    NotGenerated {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    Failed {
        code: String,
        reason: String,
    },
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum TraceState {
    Ready {
        href: &'static str,
        media_type: &'static str,
        byte_length: u64,
    },
    Pending {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    NotGenerated {
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    Failed {
        code: String,
        reason: String,
    },
}

#[derive(Debug, Serialize)]
struct AnalysisIdentity {
    revision: String,
    generated_at: String,
    generator_version: &'static str,
}

#[derive(Debug, Serialize)]
struct AnalyzerProvenance {
    source: &'static str,
    synthetic: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    generated_at: Option<String>,
    generator_version: &'static str,
}

#[derive(Debug, Serialize)]
struct RunDescriptor {
    protocol_version: u32,
    run_id: String,
    kind: &'static str,
    display_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_name: Option<String>,
    deployment: String,
    lifecycle: Lifecycle,
    summary: ArtifactRef,
    topology: ArtifactRef,
    #[serde(skip_serializing_if = "Option::is_none")]
    workers: Option<Vec<WorkerRef>>,
    subjects: BTreeMap<String, SubjectState>,
    details: BTreeMap<&'static str, Value>,
    traces: BTreeMap<&'static str, TraceState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    analysis: Option<AnalysisIdentity>,
    provenance: AnalyzerProvenance,
}

#[derive(Clone, Debug)]
struct TimingInfo {
    statuses: HashMap<String, String>,
    bytes: Vec<u8>,
    modified_at: SystemTime,
}

#[derive(Clone, Debug)]
enum TimingState {
    Missing,
    Valid(TimingInfo),
    Invalid(String),
}

#[derive(Clone, Debug)]
enum JsonProbe {
    Missing,
    Valid { schema_version: u32, value: Value },
    Invalid { code: String, reason: String },
}

#[derive(Debug)]
struct ApiProblem {
    status: StatusCode,
    code: String,
    title: &'static str,
    detail: String,
}

#[derive(Serialize)]
struct ProblemBody<'a> {
    r#type: &'static str,
    title: &'a str,
    status: u16,
    detail: &'a str,
    code: &'a str,
}

impl ApiProblem {
    fn new(
        status: StatusCode,
        code: impl Into<String>,
        title: &'static str,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            status,
            code: code.into(),
            title,
            detail: detail.into(),
        }
    }

    fn run_not_found(run_id: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "run_not_found",
            "Run not found",
            format!("No discovered run has opaque id {run_id:?}."),
        )
    }

    fn artifact_missing(resource: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "artifact_missing",
            "Artifact not found",
            format!("The bounded {resource} artifact does not exist for this run."),
        )
    }

    fn artifact_incompatible(resource: &str, detail: impl Into<String>) -> Self {
        Self::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "artifact_incompatible",
            "Artifact is incompatible",
            format!(
                "The bounded {resource} artifact is invalid: {}",
                detail.into()
            ),
        )
    }
}

impl IntoResponse for ApiProblem {
    fn into_response(self) -> Response {
        let body = serde_json::to_vec(&ProblemBody {
            r#type: "about:blank",
            title: self.title,
            status: self.status.as_u16(),
            detail: &self.detail,
            code: &self.code,
        })
        .expect("problem details are serializable");
        Response::builder()
            .status(self.status)
            .header(header::CONTENT_TYPE, "application/problem+json")
            .header(header::CACHE_CONTROL, "no-store")
            .body(Body::from(body))
            .expect("static problem headers are valid")
    }
}

/// Run the loopback-by-default analyzer artifact service.
pub async fn serve(logs_roots: Vec<PathBuf>, bind: SocketAddr) -> Result<()> {
    let app = router(logs_roots)?;
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("bind analyzer UI service at {bind}"))?;
    let local_addr = listener.local_addr()?;
    eprintln!("[analyze] serving read-only UI artifacts at http://{local_addr}/api/v1/runs");
    axum::serve(listener, app)
        .await
        .context("serve analyzer UI API")
}

fn router(logs_roots: Vec<PathBuf>) -> Result<Router> {
    let state = Arc::new(ServiceState {
        roots: configure_roots(logs_roots)?,
        discovery: Mutex::new(DiscoveryCache::default()),
    });
    Ok(Router::new()
        .route("/api/v1/runs", get(list_runs))
        .route("/api/v1/runs/{run_id}/descriptor", get(get_descriptor))
        .route("/api/v1/runs/{run_id}/summary", get(get_summary))
        .route("/api/v1/runs/{run_id}/topology", get(get_topology))
        .route(
            "/api/v1/runs/{run_id}/reports/{subject_id}",
            get(get_subject_report),
        )
        .route(
            "/api/v1/runs/{run_id}/payloads/{subject_id}",
            get(get_subject_payload),
        )
        .route(
            "/api/v1/runs/{run_id}/traces/perfetto",
            get(get_perfetto_trace),
        )
        .fallback(unknown_resource)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(middleware::from_fn(reject_unsafe_request_path))
        .with_state(state))
}

fn configure_roots(logs_roots: Vec<PathBuf>) -> Result<Vec<ConfiguredRoot>> {
    if logs_roots.is_empty() {
        bail!("at least one --logs-root is required");
    }
    logs_roots
        .into_iter()
        .enumerate()
        .map(|(ordinal, configured)| {
            let canonical = configured
                .canonicalize()
                .with_context(|| format!("canonicalize logs root {}", configured.display()))?;
            if !canonical.is_dir() {
                bail!("logs root is not a directory: {}", configured.display());
            }
            Ok(ConfiguredRoot {
                ordinal,
                path: canonical,
            })
        })
        .collect()
}

async fn list_runs(
    State(state): State<Arc<ServiceState>>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    let records = discover_runs(&state).map_err(discovery_problem)?;
    let generated_at = records
        .iter()
        .map(|run| run.updated_at)
        .max()
        .unwrap_or(UNIX_EPOCH);
    let mut runs = Vec::with_capacity(records.len());
    for run in records {
        let lifecycle = lifecycle(&run);
        runs.push(RunCatalogEntry {
            descriptor_href: format!("runs/{}/descriptor", run.run_id),
            run_id: run.run_id,
            kind: "simulation",
            display_name: run.display_name,
            lifecycle,
            updated_at: timestamp(run.updated_at),
        });
    }
    let catalog = RunCatalog {
        protocol_version: PROTOCOL_VERSION,
        generated_at: timestamp(generated_at),
        runs,
    };
    json_response(&headers, &catalog)
}

async fn get_descriptor(
    State(state): State<Arc<ServiceState>>,
    RoutePath(run_id): RoutePath<String>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    let run = resolve_run(&state, &run_id)?;
    let descriptor = build_descriptor(&run)?;
    json_response(&headers, &descriptor)
}

async fn get_summary(
    State(state): State<Arc<ServiceState>>,
    RoutePath(run_id): RoutePath<String>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    let run = resolve_run(&state, &run_id)?;
    let bytes = read_json_artifact(&run, Path::new("summary.json"), "summary")?;
    conditional_bytes(&headers, "application/json", bytes)
}

async fn get_topology(
    State(state): State<Arc<ServiceState>>,
    RoutePath(run_id): RoutePath<String>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    let run = resolve_run(&state, &run_id)?;
    let params = read_json_value(&run, Path::new("raw/params.json"), "topology params")?;
    let run_meta = read_json_value(&run, Path::new("raw/run_meta.json"), "topology run_meta")?;
    if !params.is_object() || !run_meta.is_object() {
        return Err(ApiProblem::artifact_incompatible(
            "topology",
            "params and run_meta must both be JSON objects",
        ));
    }
    let envelope = serde_json::json!({
        "schema_version": SCHEMA_VERSION,
        "params": params,
        "run_meta": run_meta,
    });
    json_response(&headers, &envelope)
}

async fn get_subject_report(
    State(state): State<Arc<ServiceState>>,
    RoutePath((run_id, subject_id)): RoutePath<(String, String)>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    get_subject_artifact(&state, &run_id, &subject_id, true, &headers)
}

async fn get_subject_payload(
    State(state): State<Arc<ServiceState>>,
    RoutePath((run_id, subject_id)): RoutePath<(String, String)>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    get_subject_artifact(&state, &run_id, &subject_id, false, &headers)
}

fn get_subject_artifact(
    state: &ServiceState,
    run_id: &str,
    subject_id: &str,
    report: bool,
    headers: &HeaderMap,
) -> Result<Response, ApiProblem> {
    let run = resolve_run(state, run_id)?;
    let subject = SUBJECTS
        .iter()
        .find(|subject| subject.scope == Scope::Run && subject.name == subject_id)
        .ok_or_else(|| {
            ApiProblem::new(
                StatusCode::NOT_FOUND,
                "resource_not_found",
                "Resource not found",
                format!("{subject_id:?} is not a registered simulation analyzer subject."),
            )
        })?;
    let deployment = read_deployment(&run)?;
    let lifecycle = lifecycle(&run);
    let timing = read_timing(&run);
    if !matches!(
        subject_state(&run, subject, &deployment, lifecycle, &timing),
        SubjectState::Ready { .. }
    ) {
        return Err(ApiProblem::new(
            StatusCode::NOT_FOUND,
            "resource_not_ready",
            "Resource is not ready",
            format!("Analyzer subject {subject_id:?} does not declare a ready artifact."),
        ));
    }
    let (directory, file_name, label) = if report {
        ("reports", subject.report_name, "subject report")
    } else {
        ("payloads", subject.payload_name, "subject payload")
    };
    let relative = Path::new(directory).join(file_name);
    let bytes = read_json_artifact(&run, &relative, label)?;
    conditional_bytes(headers, "application/json", bytes)
}

async fn get_perfetto_trace(
    State(state): State<Arc<ServiceState>>,
    RoutePath(run_id): RoutePath<String>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    let run = resolve_run(&state, &run_id)?;
    if !matches!(trace_state(&run, lifecycle(&run)), TraceState::Ready { .. }) {
        return Err(ApiProblem::new(
            StatusCode::NOT_FOUND,
            "resource_not_ready",
            "Resource is not ready",
            "This run does not declare a ready Perfetto trace.",
        ));
    }
    let path = select_trace(&run)?;
    let bytes = read_bounded_file(&path, MAX_TRACE_BYTES, "Perfetto trace")?;
    conditional_bytes(&headers, trace_media_type(&path), bytes)
}

async fn unknown_resource(uri: Uri) -> ApiProblem {
    ApiProblem::new(
        StatusCode::NOT_FOUND,
        "resource_not_found",
        "Resource not found",
        format!("No bounded analyzer resource is routed at {}.", uri.path()),
    )
}

async fn method_not_allowed() -> ApiProblem {
    ApiProblem::new(
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
        "Method not allowed",
        "Analyzer UI resources are read-only and accept GET requests only.",
    )
}

async fn reject_unsafe_request_path(request: Request, next: Next) -> Response {
    let path = request.uri().path();
    let has_invalid_segment = path
        .split('/')
        .skip(1)
        .any(|segment| segment.is_empty() || matches!(segment, "." | ".."));
    if path.contains('%') || path.contains('\\') || has_invalid_segment {
        return ApiProblem::new(
            StatusCode::BAD_REQUEST,
            "invalid_resource_path",
            "Invalid resource path",
            "Protocol-v1 resource paths must be normalized, unencoded relative segments.",
        )
        .into_response();
    }
    next.run(request).await
}

fn discovery_problem(error: anyhow::Error) -> ApiProblem {
    eprintln!("[analyze] run discovery failed: {error:#}");
    ApiProblem::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "run_discovery_failed",
        "Run discovery failed",
        "One or more configured logs roots could not be scanned.",
    )
}

fn discover_runs(state: &ServiceState) -> Result<Vec<RunRecord>> {
    refresh_discovery(state, false)
}

fn refresh_discovery(state: &ServiceState, force: bool) -> Result<Vec<RunRecord>> {
    if !force {
        let cache = state
            .discovery
            .lock()
            .map_err(|_| anyhow::anyhow!("run discovery cache lock is poisoned"))?;
        if cache
            .refreshed_at
            .is_some_and(|refreshed_at| refreshed_at.elapsed() < CATALOG_REFRESH_INTERVAL)
        {
            return Ok(cache.runs.clone());
        }
    }
    let runs = discover_runs_uncached(&state.roots)?;
    // Do not hold the cache lock while walking the filesystem: artifact reads
    // for already-known ids remain responsive during a catalog refresh.
    let mut cache = state
        .discovery
        .lock()
        .map_err(|_| anyhow::anyhow!("run discovery cache lock is poisoned"))?;
    cache.refreshed_at = Some(Instant::now());
    cache.runs = runs.clone();
    Ok(runs)
}

fn discover_runs_uncached(roots: &[ConfiguredRoot]) -> Result<Vec<RunRecord>> {
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

fn discover_root(
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

fn canonical_run_path(root: &Path, directory: &Path) -> Option<PathBuf> {
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

fn should_skip_discovery_dir(name: &str) -> bool {
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

fn resolve_run(state: &ServiceState, run_id: &str) -> Result<RunRecord, ApiProblem> {
    let cached = state
        .discovery
        .lock()
        .map_err(|_| discovery_problem(anyhow::anyhow!("run discovery cache lock is poisoned")))?
        .runs
        .iter()
        .find(|run| run.run_id == run_id)
        .cloned();
    if let Some(run) = cached {
        return Ok(run);
    }
    refresh_discovery(state, true)
        .map_err(discovery_problem)?
        .into_iter()
        .find(|run| run.run_id == run_id)
        .ok_or_else(|| ApiProblem::run_not_found(run_id))
}

fn opaque_run_id(root_ordinal: usize, relative: &Path) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vibesim-analyzer-run-id-v1\0");
    hasher.update((root_ordinal as u64).to_be_bytes());
    hasher.update(b"\0");
    hasher.update(relative.as_os_str().as_encoded_bytes());
    let digest = hasher.finalize();
    format!("r_{}", hex(&digest))
}

fn relative_label(relative: &Path) -> String {
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

fn run_updated_at(root: &Path, run: &Path) -> SystemTime {
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

fn lifecycle(run: &RunRecord) -> Lifecycle {
    let simulation = if contained_marker(run, ".failed") {
        StageStatus::Failed
    } else if contained_marker(run, ".complete") {
        StageStatus::Complete
    } else {
        StageStatus::Pending
    };
    let analysis = match read_timing(run) {
        TimingState::Valid(_) => StageStatus::Complete,
        TimingState::Invalid(_) => StageStatus::Failed,
        TimingState::Missing => StageStatus::NotStarted,
    };
    Lifecycle {
        simulation,
        analysis,
    }
}

fn contained_marker(run: &RunRecord, name: &str) -> bool {
    resolve_contained_file(&run.root, &run.path, Path::new(name), name).is_ok()
}

fn build_descriptor(run: &RunRecord) -> Result<RunDescriptor, ApiProblem> {
    let params = read_json_value(run, Path::new("raw/params.json"), "params")?;
    let deployment = deployment_from_params(&params)?;
    let lifecycle = lifecycle(run);
    let timing = read_timing(run);
    let run_meta = read_json_value_optional(run, Path::new("raw/run_meta.json"));
    let workers = run_meta
        .as_ref()
        .and_then(|run_meta| workers_from_run_meta(&deployment, run_meta));

    let subjects = SUBJECTS
        .iter()
        .filter(|subject| subject.scope == Scope::Run)
        .map(|subject| {
            (
                subject.name.to_string(),
                subject_state(run, subject, &deployment, lifecycle, &timing),
            )
        })
        .collect();
    let mut details = BTreeMap::new();
    details.insert(
        "iteration-detail",
        serde_json::json!({
            "status": "not_generated",
            "reason": "No on-demand iteration detail endpoint is available."
        }),
    );
    details.insert(
        "worker-cost-tree",
        serde_json::json!({
            "status": "not_generated",
            "reason": "No versioned hierarchical worker CostTree endpoint is available."
        }),
    );
    details.insert(
        "worker-iteration-index",
        serde_json::json!({
            "status": "not_generated",
            "reason": "No paginated worker iteration index is available."
        }),
    );
    let mut traces = BTreeMap::new();
    traces.insert("perfetto", trace_state(run, lifecycle));

    let analysis = match &timing {
        TimingState::Valid(timing) => Some(AnalysisIdentity {
            revision: analysis_revision(run, timing),
            generated_at: timestamp(timing.modified_at),
            generator_version: DEFAULT_GENERATOR_VERSION,
        }),
        TimingState::Missing | TimingState::Invalid(_) => None,
    };
    let generated_at = analysis
        .as_ref()
        .map(|identity| identity.generated_at.clone());
    Ok(RunDescriptor {
        protocol_version: PROTOCOL_VERSION,
        run_id: run.run_id.clone(),
        kind: "simulation",
        display_name: run.display_name.clone(),
        model_name: model_name_from_params(&deployment, &params),
        deployment,
        lifecycle,
        summary: ArtifactRef {
            href: "summary",
            media_type: "application/json",
            schema_version: None,
        },
        topology: ArtifactRef {
            href: "topology",
            media_type: "application/json",
            schema_version: Some(SCHEMA_VERSION),
        },
        workers,
        subjects,
        details,
        traces,
        analysis,
        provenance: AnalyzerProvenance {
            source: "analyzer",
            synthetic: false,
            generated_at,
            generator_version: DEFAULT_GENERATOR_VERSION,
        },
    })
}

fn read_deployment(run: &RunRecord) -> Result<String, ApiProblem> {
    let params = read_json_value(run, Path::new("raw/params.json"), "params")?;
    deployment_from_params(&params)
}

fn deployment_from_params(params: &Value) -> Result<String, ApiProblem> {
    let Some(deployment) = params.get("deployment").and_then(Value::as_str) else {
        return Err(ApiProblem::artifact_incompatible(
            "params",
            "deployment is missing or is not a string",
        ));
    };
    if !matches!(deployment, "unified" | "pd" | "afd") {
        return Err(ApiProblem::artifact_incompatible(
            "params",
            format!("unsupported deployment {deployment:?}"),
        ));
    }
    Ok(deployment.to_string())
}

fn pool_roles(deployment: &str) -> Option<&'static [&'static str]> {
    match deployment {
        "unified" => Some(&["main"]),
        "pd" => Some(&["prefill", "decode"]),
        "afd" => Some(&["attn", "ffn"]),
        _ => None,
    }
}

fn model_name_from_params(deployment: &str, params: &Value) -> Option<String> {
    let roles = pool_roles(deployment)?;
    let pools = params.get("pools")?.as_object()?;
    let models = roles
        .iter()
        .filter_map(|role| pools.get(*role)?.get("groups")?.as_array())
        .flatten()
        .filter_map(|group| {
            group
                .get("arch")?
                .get("model_config")?
                .as_str()
                .map(str::to_owned)
        })
        .collect::<HashSet<_>>();
    if models.len() == 1 {
        models.into_iter().next()
    } else {
        None
    }
}

fn workers_from_run_meta(deployment: &str, run_meta: &Value) -> Option<Vec<WorkerRef>> {
    let roles = pool_roles(deployment)?;
    let raw_workers = run_meta.get("workers")?.as_array()?;
    if raw_workers.is_empty() {
        return None;
    }
    let mut workers = Vec::with_capacity(raw_workers.len());
    let mut identities = HashSet::new();
    for raw_worker in raw_workers {
        let numeric_pool = usize::try_from(raw_worker.get("pool")?.as_u64()?).ok()?;
        let pool_tag = *roles.get(numeric_pool)?;
        let worker_id = raw_worker.get("worker_id")?.as_u64()?;
        if let Some(observed_tag) = raw_worker.get("pool_tag").and_then(Value::as_str) {
            if observed_tag != pool_tag {
                return None;
            }
        }
        if !identities.insert((pool_tag, worker_id)) {
            return None;
        }
        workers.push(WorkerRef {
            pool_tag: pool_tag.to_string(),
            worker_id,
        });
    }
    workers.sort_by_key(|worker| {
        (
            roles
                .iter()
                .position(|role| *role == worker.pool_tag)
                .unwrap_or(usize::MAX),
            worker.worker_id,
        )
    });
    Some(workers)
}

fn read_timing(run: &RunRecord) -> TimingState {
    let relative = Path::new("reports/analyzer_timing.json");
    let path = match resolve_contained_file(&run.root, &run.path, relative, "analyzer timing") {
        Ok(path) => path,
        Err(problem) if problem.code == "artifact_missing" => return TimingState::Missing,
        Err(problem) => return TimingState::Invalid(problem.detail),
    };
    let bytes = match read_bounded_file(&path, MAX_JSON_BYTES, "analyzer timing") {
        Ok(bytes) => bytes,
        Err(problem) => return TimingState::Invalid(problem.detail),
    };
    let modified_at = path
        .metadata()
        .and_then(|meta| meta.modified())
        .unwrap_or(UNIX_EPOCH);
    let value: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            return TimingState::Invalid(format!("invalid analyzer timing JSON: {error}"))
        }
    };
    let Some(subjects) = value.get("subjects").and_then(Value::as_array) else {
        return TimingState::Invalid("analyzer timing has no subjects array".to_string());
    };
    let mut statuses = HashMap::new();
    for row in subjects {
        let (Some(name), Some(status)) = (
            row.get("name").and_then(Value::as_str),
            row.get("status").and_then(Value::as_str),
        ) else {
            return TimingState::Invalid(
                "analyzer timing subject row lacks string name/status".to_string(),
            );
        };
        if !matches!(status, "ok" | "failed") {
            return TimingState::Invalid(format!(
                "analyzer timing subject {name:?} has unknown status {status:?}"
            ));
        }
        if statuses
            .insert(name.to_string(), status.to_string())
            .is_some()
        {
            return TimingState::Invalid(format!("analyzer timing repeats subject {name:?}"));
        }
    }
    TimingState::Valid(TimingInfo {
        statuses,
        bytes,
        modified_at,
    })
}

fn subject_state(
    run: &RunRecord,
    subject: &Subject,
    deployment: &str,
    lifecycle: Lifecycle,
    timing: &TimingState,
) -> SubjectState {
    if !subject.applies.matches(Some(deployment)) {
        return SubjectState::Unavailable {
            reason: format!(
                "Analyzer subject {:?} does not apply to deployment {deployment:?}.",
                subject.name
            ),
            code: Some("subject_not_applicable".to_string()),
        };
    }
    if matches!(
        timing,
        TimingState::Valid(TimingInfo { statuses, .. })
            if statuses.get(subject.name).is_some_and(|status| status == "failed")
    ) {
        return SubjectState::Failed {
            code: "analysis_failed".to_string(),
            reason: format!(
                "Analyzer subject {:?} failed during generation.",
                subject.name
            ),
        };
    }

    let report = probe_json(
        run,
        &Path::new("reports").join(subject.report_name),
        "subject report",
    );
    let payload = probe_json(
        run,
        &Path::new("payloads").join(subject.payload_name),
        "subject payload",
    );
    if let JsonProbe::Invalid { code, reason } = &report {
        return SubjectState::Failed {
            code: code.clone(),
            reason: reason.clone(),
        };
    }
    if let JsonProbe::Invalid { code, reason } = &payload {
        return SubjectState::Failed {
            code: code.clone(),
            reason: reason.clone(),
        };
    }

    match (&report, &payload) {
        (
            JsonProbe::Valid {
                schema_version: report_version,
                value: report_value,
            },
            JsonProbe::Valid {
                schema_version: payload_version,
                value: payload_value,
            },
        ) => {
            if report_version != payload_version {
                return SubjectState::Failed {
                    code: "artifact_incompatible".to_string(),
                    reason: format!(
                        "Analyzer subject {:?} report/payload schema versions disagree ({report_version} vs {payload_version}).",
                        subject.name
                    ),
                };
            }
            if artifact_available(report_value) == Some(false)
                || artifact_available(payload_value) == Some(false)
            {
                return SubjectState::Unavailable {
                    reason: artifact_reason(report_value)
                        .or_else(|| artifact_reason(payload_value))
                        .unwrap_or_else(|| {
                            format!(
                                "Analyzer subject {:?} reported no available data.",
                                subject.name
                            )
                        }),
                    code: artifact_code(report_value)
                        .or_else(|| artifact_code(payload_value))
                        .or_else(|| Some("subject_unavailable".to_string())),
                };
            }
            SubjectState::Ready {
                schema_version: *report_version,
                report_href: format!("reports/{}", subject.name),
                payload_href: format!("payloads/{}", subject.name),
            }
        }
        _ if timing_status(timing, subject.name) == Some("ok") => SubjectState::Failed {
            code: "artifact_missing".to_string(),
            reason: format!(
                "Analyzer subject {:?} completed but its report/payload pair is incomplete.",
                subject.name
            ),
        },
        _ if lifecycle.simulation == StageStatus::Pending
            || lifecycle.analysis == StageStatus::Pending =>
        {
            SubjectState::Pending {
                reason: Some(format!(
                    "Analyzer subject {:?} has not finished generating.",
                    subject.name
                )),
            }
        }
        _ if lifecycle.analysis == StageStatus::Failed => {
            let detail = match timing {
                TimingState::Invalid(detail) => detail.as_str(),
                TimingState::Missing | TimingState::Valid(_) => "unknown analyzer lifecycle error",
            };
            SubjectState::Failed {
                code: "analysis_state_invalid".to_string(),
                reason: format!(
                    "Analyzer lifecycle failed before subject {:?} produced a complete artifact pair: {detail}.",
                    subject.name
                ),
            }
        }
        _ => SubjectState::NotGenerated {
            reason: Some(format!(
                "Analyzer subject {:?} was not generated for this run.",
                subject.name
            )),
        },
    }
}

fn timing_status<'a>(timing: &'a TimingState, subject: &str) -> Option<&'a str> {
    match timing {
        TimingState::Valid(info) => info.statuses.get(subject).map(String::as_str),
        TimingState::Missing | TimingState::Invalid(_) => None,
    }
}

fn probe_json(run: &RunRecord, relative: &Path, resource: &str) -> JsonProbe {
    let path = match resolve_contained_file(&run.root, &run.path, relative, resource) {
        Ok(path) => path,
        Err(problem) if problem.code == "artifact_missing" => return JsonProbe::Missing,
        Err(problem) => {
            return JsonProbe::Invalid {
                code: problem.code,
                reason: problem.detail,
            }
        }
    };
    let bytes = match read_bounded_file(&path, MAX_JSON_BYTES, resource) {
        Ok(bytes) => bytes,
        Err(problem) => {
            return JsonProbe::Invalid {
                code: problem.code,
                reason: problem.detail,
            }
        }
    };
    let value: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(error) => {
            return JsonProbe::Invalid {
                code: "artifact_incompatible".to_string(),
                reason: format!("The {resource} is not valid JSON: {error}"),
            }
        }
    };
    let Some(schema_version) = value.get("schema_version").and_then(Value::as_u64) else {
        return JsonProbe::Invalid {
            code: "artifact_incompatible".to_string(),
            reason: format!("The {resource} has no positive integer schema_version."),
        };
    };
    let Ok(schema_version) = u32::try_from(schema_version) else {
        return JsonProbe::Invalid {
            code: "artifact_incompatible".to_string(),
            reason: format!("The {resource} schema_version does not fit protocol-v1."),
        };
    };
    if schema_version == 0 {
        return JsonProbe::Invalid {
            code: "artifact_incompatible".to_string(),
            reason: format!("The {resource} schema_version must be positive."),
        };
    }
    JsonProbe::Valid {
        schema_version,
        value,
    }
}

fn artifact_available(value: &Value) -> Option<bool> {
    value
        .get("available")
        .and_then(Value::as_bool)
        .or_else(|| value.get("meta")?.get("available")?.as_bool())
}

fn artifact_reason(value: &Value) -> Option<String> {
    value
        .get("reason")
        .and_then(Value::as_str)
        .or_else(|| value.get("meta")?.get("reason")?.as_str())
        .filter(|reason| !reason.trim().is_empty())
        .map(str::to_owned)
}

fn artifact_code(value: &Value) -> Option<String> {
    value
        .get("code")
        .and_then(Value::as_str)
        .or_else(|| value.get("meta")?.get("code")?.as_str())
        .filter(|code| !code.trim().is_empty())
        .map(str::to_owned)
}

fn trace_state(run: &RunRecord, lifecycle: Lifecycle) -> TraceState {
    match select_trace(run) {
        Ok(path) => match path.metadata() {
            Ok(metadata) => TraceState::Ready {
                href: "traces/perfetto",
                media_type: trace_media_type(&path),
                byte_length: metadata.len(),
            },
            Err(error) => TraceState::Failed {
                code: "artifact_missing".to_string(),
                reason: format!("Perfetto trace metadata is unavailable: {error}"),
            },
        },
        Err(problem) if problem.code == "artifact_missing" => {
            if lifecycle.simulation == StageStatus::Pending
                || lifecycle.analysis == StageStatus::Pending
            {
                TraceState::Pending {
                    reason: Some("Perfetto trace generation has not completed.".to_string()),
                }
            } else {
                TraceState::NotGenerated {
                    reason: Some("No Perfetto trace was generated for this run.".to_string()),
                }
            }
        }
        Err(problem) => TraceState::Failed {
            code: problem.code,
            reason: problem.detail,
        },
    }
}

fn trace_media_type(path: &Path) -> &'static str {
    if path.extension().and_then(|extension| extension.to_str()) == Some("gz") {
        "application/gzip"
    } else {
        "application/x-protobuf"
    }
}

fn select_trace(run: &RunRecord) -> Result<PathBuf, ApiProblem> {
    select_trace_from(&run.root, &run.path)
}

fn select_trace_from(root: &Path, run: &Path) -> Result<PathBuf, ApiProblem> {
    let trace_dir = run.join("traces");
    let entries = match fs::read_dir(&trace_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            return Err(ApiProblem::artifact_missing("Perfetto trace"))
        }
        Err(error) => {
            return Err(ApiProblem::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "artifact_read_failed",
                "Artifact read failed",
                format!("Cannot enumerate the bounded Perfetto trace directory: {error}"),
            ))
        }
    };
    let mut candidates = Vec::new();
    let mut containment_error = None;
    for entry in entries.filter_map(|entry| entry.ok()) {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".pftrace") && !name.ends_with(".pftrace.gz") {
            continue;
        }
        match resolve_contained_file(
            root,
            run,
            &Path::new("traces").join(&name),
            "Perfetto trace",
        ) {
            Ok(path) => {
                let modified_at = path
                    .metadata()
                    .and_then(|metadata| metadata.modified())
                    .unwrap_or(UNIX_EPOCH);
                candidates.push((modified_at, name, path));
            }
            Err(problem) if problem.code == "artifact_missing" => {}
            Err(problem) => containment_error = Some(problem),
        }
    }
    candidates.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    if let Some((_, _, path)) = candidates.pop() {
        return Ok(path);
    }
    Err(containment_error.unwrap_or_else(|| ApiProblem::artifact_missing("Perfetto trace")))
}

fn analysis_revision(run: &RunRecord, timing: &TimingInfo) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vibesim-analyzer-analysis-revision-v1\0");
    hasher.update(&timing.bytes);
    for subject in SUBJECTS
        .iter()
        .filter(|subject| subject.scope == Scope::Run)
    {
        fingerprint_artifact(
            &mut hasher,
            run,
            &Path::new("reports").join(subject.report_name),
        );
        fingerprint_artifact(
            &mut hasher,
            run,
            &Path::new("payloads").join(subject.payload_name),
        );
    }
    if let Ok(trace) = select_trace(run) {
        fingerprint_metadata(&mut hasher, Path::new("traces/perfetto"), &trace);
    }
    format!("analyzer-sha256-{}", hex(&hasher.finalize()))
}

fn fingerprint_artifact(hasher: &mut Sha256, run: &RunRecord, relative: &Path) {
    hasher.update(relative_label(relative).as_bytes());
    match resolve_contained_file(&run.root, &run.path, relative, "analysis artifact") {
        Ok(path) => fingerprint_metadata(hasher, relative, &path),
        Err(_) => hasher.update(b"\0missing\0"),
    }
}

fn fingerprint_metadata(hasher: &mut Sha256, relative: &Path, path: &Path) {
    hasher.update(relative_label(relative).as_bytes());
    match path.metadata() {
        Ok(metadata) => {
            hasher.update(metadata.len().to_be_bytes());
            if let Ok(modified) = metadata.modified() {
                let nanos = modified
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos();
                hasher.update(nanos.to_be_bytes());
            }
        }
        Err(_) => hasher.update(b"\0metadata-error\0"),
    }
}

fn resolve_contained_file(
    root: &Path,
    run: &Path,
    relative: &Path,
    resource: &str,
) -> Result<PathBuf, ApiProblem> {
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ApiProblem::new(
            StatusCode::BAD_REQUEST,
            "invalid_resource_path",
            "Invalid resource path",
            "The server-built artifact path is not a normalized relative path.",
        ));
    }
    let candidate = run.join(relative);
    let canonical = match candidate.canonicalize() {
        Ok(path) => path,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            return Err(ApiProblem::artifact_missing(resource))
        }
        Err(error) => {
            return Err(ApiProblem::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "artifact_read_failed",
                "Artifact read failed",
                format!("Cannot resolve the bounded {resource} artifact: {error}"),
            ))
        }
    };
    if !canonical.starts_with(root) || !canonical.starts_with(run) {
        return Err(ApiProblem::new(
            StatusCode::FORBIDDEN,
            "artifact_outside_run",
            "Artifact escaped its run",
            format!("The bounded {resource} artifact resolves outside its canonical run."),
        ));
    }
    if !canonical.is_file() {
        return Err(ApiProblem::artifact_missing(resource));
    }
    Ok(canonical)
}

fn read_json_value(run: &RunRecord, relative: &Path, resource: &str) -> Result<Value, ApiProblem> {
    let bytes = read_json_artifact(run, relative, resource)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| ApiProblem::artifact_incompatible(resource, error.to_string()))
}

fn read_json_value_optional(run: &RunRecord, relative: &Path) -> Option<Value> {
    read_json_value(run, relative, "optional run metadata").ok()
}

fn read_json_artifact(
    run: &RunRecord,
    relative: &Path,
    resource: &str,
) -> Result<Vec<u8>, ApiProblem> {
    let path = resolve_contained_file(&run.root, &run.path, relative, resource)?;
    let bytes = read_bounded_file(&path, MAX_JSON_BYTES, resource)?;
    serde_json::from_slice::<Value>(&bytes)
        .map_err(|error| ApiProblem::artifact_incompatible(resource, error.to_string()))?;
    Ok(bytes)
}

fn read_bounded_file(path: &Path, maximum: u64, resource: &str) -> Result<Vec<u8>, ApiProblem> {
    let metadata = path.metadata().map_err(|error| {
        ApiProblem::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "artifact_read_failed",
            "Artifact read failed",
            format!("Cannot read bounded {resource} metadata: {error}"),
        )
    })?;
    if metadata.len() > maximum {
        return Err(ApiProblem::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "artifact_too_large",
            "Artifact is too large",
            format!("The bounded {resource} exceeds the service safety limit."),
        ));
    }
    fs::read(path).map_err(|error| {
        ApiProblem::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "artifact_read_failed",
            "Artifact read failed",
            format!("Cannot read the bounded {resource}: {error}"),
        )
    })
}

fn json_response<T: Serialize>(headers: &HeaderMap, value: &T) -> Result<Response, ApiProblem> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        ApiProblem::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "response_encoding_failed",
            "Response encoding failed",
            error.to_string(),
        )
    })?;
    conditional_bytes(headers, "application/json", bytes)
}

fn conditional_bytes(
    headers: &HeaderMap,
    content_type: &'static str,
    bytes: Vec<u8>,
) -> Result<Response, ApiProblem> {
    let etag = format!("\"sha256-{}\"", hex(&Sha256::digest(&bytes)));
    if if_none_match(headers, &etag) {
        return Response::builder()
            .status(StatusCode::NOT_MODIFIED)
            .header(header::ETAG, etag)
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::empty())
            .map_err(response_build_problem);
    }
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, content_type)
        .header(header::ETAG, etag)
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from(bytes))
        .map_err(response_build_problem)
}

fn if_none_match(headers: &HeaderMap, etag: &str) -> bool {
    headers
        .get_all(header::IF_NONE_MATCH)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .any(|candidate| candidate == "*" || candidate.trim_start_matches("W/") == etag)
}

fn response_build_problem(error: axum::http::Error) -> ApiProblem {
    ApiProblem::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "response_build_failed",
        "Response build failed",
        error.to_string(),
    )
}

fn timestamp(time: SystemTime) -> String {
    DateTime::<Utc>::from(time).to_rfc3339_opts(SecondsFormat::Nanos, true)
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use std::fs;

    use axum::body::{to_bytes, Body};
    use axum::http::{header, Request, StatusCode};
    use serde_json::{json, Value};
    use tempfile::TempDir;
    use tower::ServiceExt;

    use super::*;

    fn write_json(path: &Path, value: &Value) {
        fs::create_dir_all(path.parent().expect("test artifact has parent")).unwrap();
        fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
    }

    fn params(deployment: &str) -> Value {
        let roles: &[&str] = match deployment {
            "unified" => &["main"],
            "pd" => &["prefill", "decode"],
            "afd" => &["attn", "ffn"],
            other => panic!("unsupported test deployment {other}"),
        };
        let pools = roles
            .iter()
            .map(|role| {
                (
                    (*role).to_string(),
                    json!({
                        "placement": "least-queued",
                        "groups": [{
                            "gpu": "NVIDIA H200",
                            "replicas": 1,
                            "arch": {
                                "type": "test",
                                "model_config": "model/test.json",
                                "fp8": false
                            },
                            "worker": {"type": "test"}
                        }]
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        json!({"deployment": deployment, "pools": pools})
    }

    fn run_meta(deployment: &str) -> Value {
        let roles = pool_roles(deployment).unwrap();
        let gpus = roles
            .iter()
            .enumerate()
            .map(|(pool, _)| json!({"id": pool, "name": "NVIDIA H200", "pool": pool, "worker_id": 0}))
            .collect::<Vec<_>>();
        let workers = roles
            .iter()
            .enumerate()
            .map(|(pool, role)| {
                json!({
                    "worker_id": 0,
                    "pool": pool,
                    "pool_tag": role,
                    "gpu_ids": [pool],
                    "kv_pools": []
                })
            })
            .collect::<Vec<_>>();
        json!({
            "schema_version": 3,
            "num_gpus": roles.len(),
            "gpus": gpus,
            "workers": workers,
            "comm_groups": []
        })
    }

    fn create_run(root: &Path, relative: &str, deployment: &str, complete: bool) -> PathBuf {
        let run = root.join(relative);
        write_json(&run.join("raw/params.json"), &params(deployment));
        write_json(&run.join("raw/run_meta.json"), &run_meta(deployment));
        write_json(
            &run.join("summary.json"),
            &json!({"total_tok_s": 1.0, "num_gpus": pool_roles(deployment).unwrap().len(), "requests_finished": 1}),
        );
        if complete {
            fs::write(run.join(".complete"), b"").unwrap();
        }
        run
    }

    fn write_subject(run: &Path, subject_name: &str, available: bool, reason: Option<&str>) {
        let subject = SUBJECTS
            .iter()
            .find(|subject| subject.name == subject_name)
            .unwrap();
        let mut report = json!({"schema_version": 1, "available": available});
        let mut payload = json!({"schema_version": 1, "meta": {"available": available}});
        if let Some(reason) = reason {
            report["reason"] = Value::String(reason.to_string());
            payload["meta"]["reason"] = Value::String(reason.to_string());
        }
        write_json(&run.join("reports").join(subject.report_name), &report);
        write_json(&run.join("payloads").join(subject.payload_name), &payload);
    }

    fn write_timing(run: &Path, statuses: &[(&str, &str)]) {
        write_json(
            &run.join("reports/analyzer_timing.json"),
            &json!({
                "schema_version": 1,
                "subjects": statuses.iter().map(|(name, status)| json!({"name": name, "status": status})).collect::<Vec<_>>()
            }),
        );
    }

    fn state(root: &TempDir) -> ServiceState {
        ServiceState {
            roots: configure_roots(vec![root.path().to_path_buf()]).unwrap(),
            discovery: Mutex::new(DiscoveryCache::default()),
        }
    }

    async fn body_json(response: Response) -> Value {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn nested_duplicate_basenames_have_stable_distinct_ids() {
        let root = TempDir::new().unwrap();
        create_run(root.path(), "sweep-a/simulation", "unified", true);
        create_run(root.path(), "sweep-b/simulation", "unified", true);
        // A cache-build copy has a params sidecar too, but is not a public run.
        create_run(root.path(), ".cache_build/simulation", "unified", true);
        let state = state(&root);

        let first = discover_runs(&state).unwrap();
        let second = discover_runs(&state).unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(
            first.iter().map(|run| &run.run_id).collect::<Vec<_>>(),
            second.iter().map(|run| &run.run_id).collect::<Vec<_>>()
        );
        assert_ne!(first[0].run_id, first[1].run_id);
        let labels = first
            .iter()
            .map(|run| run.display_name.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(
            labels,
            HashSet::from(["sweep-a/simulation", "sweep-b/simulation"])
        );
        assert!(first.iter().all(|run| run.run_id.starts_with("r_")));
        assert!(first.iter().all(|run| !run.run_id.contains("simulation")));
    }

    #[test]
    fn repeated_logs_roots_keep_identical_relative_paths_distinct() {
        let first_root = TempDir::new().unwrap();
        let second_root = TempDir::new().unwrap();
        create_run(first_root.path(), "simulation", "unified", true);
        create_run(second_root.path(), "simulation", "unified", true);
        let state = ServiceState {
            roots: configure_roots(vec![
                first_root.path().to_path_buf(),
                second_root.path().to_path_buf(),
            ])
            .unwrap(),
            discovery: Mutex::new(DiscoveryCache::default()),
        };

        let runs = discover_runs(&state).unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(
            runs.iter()
                .map(|run| run.display_name.as_str())
                .collect::<Vec<_>>(),
            vec!["simulation", "simulation"]
        );
        assert_ne!(runs[0].run_id, runs[1].run_id);
    }

    #[test]
    fn pd_descriptor_preserves_deployment_and_composite_workers() {
        let root = TempDir::new().unwrap();
        create_run(root.path(), "pd-run", "pd", true);
        let record = discover_runs(&state(&root)).unwrap().remove(0);

        let descriptor = build_descriptor(&record).unwrap();
        assert_eq!(descriptor.deployment, "pd");
        assert_eq!(descriptor.topology.href, "topology");
        assert_eq!(descriptor.topology.schema_version, Some(1));
        assert_eq!(
            descriptor.workers.unwrap(),
            vec![
                WorkerRef {
                    pool_tag: "prefill".to_string(),
                    worker_id: 0,
                },
                WorkerRef {
                    pool_tag: "decode".to_string(),
                    worker_id: 0,
                },
            ]
        );
    }

    #[tokio::test]
    async fn registry_is_the_only_report_payload_allowlist() {
        let root = TempDir::new().unwrap();
        let run = create_run(root.path(), "run", "unified", true);
        write_subject(&run, "slo-general", true, None);
        write_timing(&run, &[("slo-general", "ok")]);
        let state = state(&root);
        let run_id = discover_runs(&state).unwrap().remove(0).run_id;
        let app = router(vec![root.path().to_path_buf()]).unwrap();

        let ready = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/v1/runs/{run_id}/reports/slo-general"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(ready.status(), StatusCode::OK);

        for path in [
            format!("/api/v1/runs/{run_id}/reports/not-in-registry"),
            format!("/api/v1/runs/{run_id}/raw/params.json"),
        ] {
            let response = app
                .clone()
                .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NOT_FOUND);
            assert_eq!(
                response.headers()[header::CONTENT_TYPE],
                "application/problem+json"
            );
            assert_eq!(body_json(response).await["code"], "resource_not_found");
        }
    }

    #[tokio::test]
    async fn summary_is_passed_through_without_rewriting_its_shape_or_bytes() {
        let root = TempDir::new().unwrap();
        let run = create_run(root.path(), "run", "unified", true);
        let original = b"{\n  \"future_summary_field\": [3, 2, 1],\n  \"total_tok_s\": 7.5\n}\n";
        fs::write(run.join("summary.json"), original).unwrap();
        let run_id = discover_runs(&state(&root)).unwrap().remove(0).run_id;
        let app = router(vec![root.path().to_path_buf()]).unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/v1/runs/{run_id}/summary"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert_eq!(bytes.as_ref(), original);
    }

    #[tokio::test]
    async fn unknown_run_is_a_structured_problem() {
        let root = TempDir::new().unwrap();
        let app = router(vec![root.path().to_path_buf()]).unwrap();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/runs/r_unknown/descriptor")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(body_json(response).await["code"], "run_not_found");
    }

    #[tokio::test]
    async fn api_is_same_origin_only_and_rejects_noncanonical_paths_and_writes() {
        let root = TempDir::new().unwrap();
        let run = create_run(root.path(), "run", "unified", true);
        write_subject(&run, "slo-general", true, None);
        write_timing(&run, &[("slo-general", "ok")]);
        let run_id = discover_runs(&state(&root)).unwrap().remove(0).run_id;
        let app = router(vec![root.path().to_path_buf()]).unwrap();

        let catalog = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/v1/runs")
                    .header(header::ORIGIN, "https://untrusted.example")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(catalog.status(), StatusCode::OK);
        assert!(!catalog
            .headers()
            .contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN));

        let encoded = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/v1/runs/{run_id}/reports/%73lo-general"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(encoded.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_json(encoded).await["code"], "invalid_resource_path");

        let write_attempt = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/runs")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(write_attempt.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(body_json(write_attempt).await["code"], "method_not_allowed");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn symlink_escape_is_rejected_after_run_resolution() {
        use std::os::unix::fs::symlink;

        let root = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        let run = create_run(root.path(), "run", "unified", true);
        let outside_summary = outside.path().join("summary.json");
        write_json(&outside_summary, &json!({"secret": true}));
        fs::remove_file(run.join("summary.json")).unwrap();
        symlink(outside_summary, run.join("summary.json")).unwrap();
        let run_id = discover_runs(&state(&root)).unwrap().remove(0).run_id;
        let app = router(vec![root.path().to_path_buf()]).unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/v1/runs/{run_id}/summary"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        assert_eq!(body_json(response).await["code"], "artifact_outside_run");
    }

    #[tokio::test]
    async fn catalog_descriptor_and_artifact_honor_etag() {
        let root = TempDir::new().unwrap();
        let run = create_run(root.path(), "run", "unified", true);
        write_subject(&run, "slo-general", true, None);
        write_timing(&run, &[("slo-general", "ok")]);
        let run_id = discover_runs(&state(&root)).unwrap().remove(0).run_id;
        let app = router(vec![root.path().to_path_buf()]).unwrap();

        for uri in [
            "/api/v1/runs".to_string(),
            format!("/api/v1/runs/{run_id}/descriptor"),
            format!("/api/v1/runs/{run_id}/payloads/slo-general"),
        ] {
            let first = app
                .clone()
                .oneshot(Request::builder().uri(&uri).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(first.status(), StatusCode::OK);
            let etag = first.headers()[header::ETAG].clone();
            let conditional = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(&uri)
                        .header(header::IF_NONE_MATCH, etag)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(conditional.status(), StatusCode::NOT_MODIFIED, "{uri}");
            assert!(to_bytes(conditional.into_body(), usize::MAX)
                .await
                .unwrap()
                .is_empty());
        }
    }

    #[test]
    fn descriptor_reports_subject_states_independently() {
        let root = TempDir::new().unwrap();
        let run = create_run(root.path(), "run", "unified", true);
        write_subject(&run, "slo-general", true, None);
        write_subject(
            &run,
            "slo-detailed",
            false,
            Some("output token times were not logged"),
        );
        write_timing(
            &run,
            &[
                ("slo-general", "ok"),
                ("slo-detailed", "ok"),
                ("throughput", "failed"),
            ],
        );
        let record = discover_runs(&state(&root)).unwrap().remove(0);
        let descriptor = build_descriptor(&record).unwrap();

        assert!(matches!(
            descriptor.subjects["slo-general"],
            SubjectState::Ready { .. }
        ));
        assert!(matches!(
            descriptor.subjects["slo-detailed"],
            SubjectState::Unavailable { .. }
        ));
        assert!(matches!(
            descriptor.subjects["throughput"],
            SubjectState::Failed { .. }
        ));
        assert!(matches!(
            descriptor.subjects["batch"],
            SubjectState::NotGenerated { .. }
        ));
        assert_eq!(descriptor.lifecycle.analysis, StageStatus::Complete);
        assert!(descriptor.analysis.is_some());
    }

    #[test]
    fn unfinished_run_marks_missing_subjects_pending() {
        let root = TempDir::new().unwrap();
        create_run(root.path(), "run", "unified", false);
        let record = discover_runs(&state(&root)).unwrap().remove(0);
        let descriptor = build_descriptor(&record).unwrap();
        assert_eq!(descriptor.lifecycle.simulation, StageStatus::Pending);
        assert!(matches!(
            descriptor.subjects["slo-general"],
            SubjectState::Pending { .. }
        ));
    }

    #[test]
    fn completed_legacy_run_does_not_call_missing_subjects_pending() {
        let root = TempDir::new().unwrap();
        let run = create_run(root.path(), "run", "unified", true);
        write_subject(&run, "slo-general", true, None);
        let record = discover_runs(&state(&root)).unwrap().remove(0);
        let descriptor = build_descriptor(&record).unwrap();

        assert_eq!(descriptor.lifecycle.analysis, StageStatus::NotStarted);
        assert!(matches!(
            descriptor.subjects["slo-general"],
            SubjectState::Ready { .. }
        ));
        assert!(matches!(
            descriptor.subjects["throughput"],
            SubjectState::NotGenerated { .. }
        ));
    }

    #[tokio::test]
    async fn topology_is_the_exact_v1_params_run_meta_envelope() {
        let root = TempDir::new().unwrap();
        create_run(root.path(), "run", "afd", true);
        let run_id = discover_runs(&state(&root)).unwrap().remove(0).run_id;
        let app = router(vec![root.path().to_path_buf()]).unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/v1/runs/{run_id}/topology"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = body_json(response).await;
        assert_eq!(
            body.as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<HashSet<_>>(),
            HashSet::from(["schema_version", "params", "run_meta"])
        );
        assert_eq!(body["schema_version"], 1);
        assert_eq!(body["params"]["deployment"], "afd");
        assert_eq!(body["run_meta"]["num_gpus"], 2);
    }
}
