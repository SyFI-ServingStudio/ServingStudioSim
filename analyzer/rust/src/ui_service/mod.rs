//! Read-only protocol-v1 HTTP boundary for analyzer artifacts.
//!
//! The service deliberately is not a file server. Every public resource is
//! selected from either the fixed core endpoints or [`crate::registry::SUBJECTS`]
//! after an opaque run id has resolved to a canonical, contained run directory.
//! This keeps parquet, cache-build trees, and arbitrary paths outside the HTTP
//! surface even when a caller guesses their on-disk names.

mod artifact;
mod discovery;
mod lifecycle;
#[cfg(test)]
mod tests;

use artifact::*;
use discovery::*;
use lifecycle::*;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::io::{ErrorKind, Read, Seek, SeekFrom};
use std::net::SocketAddr;
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use axum::body::Body;
use axum::extract::{Path as RoutePath, Request, State};
use axum::http::{header, uri::Authority, HeaderMap, StatusCode, Uri};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::io::ReaderStream;

use crate::registry::{Scope, Subject, SUBJECTS};

const PROTOCOL_VERSION: u32 = 1;
const TOPOLOGY_SCHEMA_VERSION: u32 = 1;
const PIPELINE_SCHEMA_VERSION: u32 = 1;
const PIPELINE_STATE_PATH: &str = "reports/analyzer_pipeline_state.json";
const LEGACY_GENERATOR_VERSION: &str = "legacy-unknown";
const MAX_PIPELINE_STATE_BYTES: u64 = 1024 * 1024;
const MAX_JSON_BYTES: u64 = 128 * 1024 * 1024;
const MAX_TRACE_BYTES: u64 = 512 * 1024 * 1024;
const CATALOG_REFRESH_INTERVAL: Duration = Duration::from_secs(30);
const MAX_DESCRIPTOR_CACHE_ENTRIES: usize = 256;
const MAX_GENERATION_READ_ATTEMPTS: usize = 3;

#[derive(Clone, Debug)]
struct ConfiguredRoot {
    ordinal: usize,
    path: PathBuf,
    #[cfg(target_os = "linux")]
    directory: Arc<fs::File>,
}

#[derive(Debug)]
struct ServiceState {
    roots: Vec<ConfiguredRoot>,
    discovery: RwLock<DiscoveryCache>,
    discovery_refresh: AsyncMutex<()>,
    #[cfg(test)]
    discovery_scan_count: AtomicUsize,
    descriptors: RwLock<HashMap<String, CachedDescriptor>>,
    allowed_hosts: HashSet<String>,
}

#[derive(Clone, Debug)]
struct CachedDescriptor {
    stamp: String,
    bytes: Vec<u8>,
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
    #[cfg(target_os = "linux")]
    root_directory: Arc<fs::File>,
    path: PathBuf,
    updated_at: SystemTime,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum StageStatus {
    NotStarted,
    Pending,
    Complete,
    Failed,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
struct PipelineProducer {
    name: String,
    version: String,
    revision: String,
    binary_sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
struct PipelineStage {
    status: StageStatus,
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    artifact: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
struct PipelineStages {
    compute: PipelineStage,
    render: PipelineStage,
    trace: PipelineStage,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
struct PipelineStateV1 {
    schema_version: u32,
    generation_id: String,
    artifact_revision: String,
    status: StageStatus,
    started_at: String,
    updated_at: String,
    #[serde(default)]
    completed_at: Option<String>,
    producer: PipelineProducer,
    #[serde(default)]
    requested_subjects: Option<Vec<String>>,
    stages: PipelineStages,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PipelineStateRead {
    Missing,
    Valid(Box<PipelineStateV1>),
    Invalid { code: String, reason: String },
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
        href: String,
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
    generator_version: String,
}

#[derive(Debug, Serialize)]
struct AnalyzerProvenance {
    source: &'static str,
    synthetic: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    generated_at: Option<String>,
    generator_version: String,
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct TimingInfo {
    statuses: HashMap<String, String>,
    bytes: Vec<u8>,
    modified_at: SystemTime,
    generation_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TimingState {
    Missing,
    Valid(TimingInfo),
    Invalid(String),
}

#[derive(Clone, Debug)]
enum JsonProbe {
    Missing,
    Valid {
        schema_version: u32,
        value: Value,
        bytes: Vec<u8>,
    },
    Invalid {
        code: String,
        reason: String,
    },
}

#[derive(Default)]
struct SubjectArtifactBytes {
    report: Vec<u8>,
    payload: Vec<u8>,
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

    fn artifact_generation_changed(detail: impl Into<String>) -> Self {
        Self::new(
            StatusCode::CONFLICT,
            "artifact_generation_changed",
            "Artifact generation changed",
            detail,
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
        let mut response = Response::builder()
            .status(self.status)
            .header(header::CONTENT_TYPE, "application/problem+json")
            .header(header::CACHE_CONTROL, "no-store")
            .body(Body::from(body))
            .expect("static problem headers are valid");
        if self.status == StatusCode::METHOD_NOT_ALLOWED {
            response
                .headers_mut()
                .insert(header::ALLOW, header::HeaderValue::from_static("GET"));
        }
        response
    }
}

/// Run the loopback-by-default analyzer artifact service.
pub async fn serve(
    logs_roots: Vec<PathBuf>,
    bind: SocketAddr,
    allow_hosts: Vec<String>,
) -> Result<()> {
    let app = router_with_hosts(logs_roots, allow_hosts)?;
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("bind analyzer UI service at {bind}"))?;
    let local_addr = listener.local_addr()?;
    eprintln!("[analyze] serving read-only UI artifacts at http://{local_addr}/api/v1/runs");
    axum::serve(listener, app)
        .await
        .context("serve analyzer UI API")
}

#[cfg(test)]
fn router(logs_roots: Vec<PathBuf>) -> Result<Router> {
    router_with_hosts(logs_roots, Vec::new())
}

fn router_with_hosts(logs_roots: Vec<PathBuf>, allow_hosts: Vec<String>) -> Result<Router> {
    let state = Arc::new(ServiceState {
        roots: configure_roots(logs_roots)?,
        discovery: RwLock::new(DiscoveryCache::default()),
        discovery_refresh: AsyncMutex::new(()),
        #[cfg(test)]
        discovery_scan_count: AtomicUsize::new(0),
        descriptors: RwLock::new(HashMap::new()),
        allowed_hosts: configure_allowed_hosts(allow_hosts)?,
    });
    Ok(Router::new()
        .route("/api/v1/runs", get(list_runs))
        .route("/api/v1/runs/{run_id}/descriptor", get(get_descriptor))
        .route("/api/v1/runs/{run_id}/summary", get(get_summary))
        .route("/api/v1/runs/{run_id}/topology", get(get_topology))
        .route(
            "/api/v1/runs/{run_id}/revisions/{revision}/reports/{subject_id}",
            get(get_subject_report),
        )
        .route(
            "/api/v1/runs/{run_id}/revisions/{revision}/payloads/{subject_id}",
            get(get_subject_payload),
        )
        .route(
            "/api/v1/runs/{run_id}/revisions/{revision}/traces/perfetto",
            get(get_perfetto_trace),
        )
        .fallback(unknown_resource)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(middleware::from_fn(reject_unsafe_request_path))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            enforce_allowed_host,
        ))
        .with_state(state))
}

fn configure_allowed_hosts(configured: Vec<String>) -> Result<HashSet<String>> {
    let mut allowed = HashSet::from([
        "localhost".to_string(),
        "127.0.0.1".to_string(),
        "::1".to_string(),
    ]);
    for candidate in configured {
        let authority = Authority::from_str(candidate.trim())
            .with_context(|| format!("invalid --allow-host {candidate:?}"))?;
        if authority.port_u16().is_some() {
            bail!("--allow-host names must not include a port: {candidate:?}");
        }
        let host = normalize_host(authority.host());
        if host.is_empty() {
            bail!("--allow-host must contain a hostname");
        }
        allowed.insert(host);
    }
    Ok(allowed)
}

fn normalize_host(host: &str) -> String {
    host.trim_start_matches('[')
        .trim_end_matches(']')
        .to_ascii_lowercase()
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
            #[cfg(target_os = "linux")]
            let directory = Arc::new(fs::File::open(&canonical).with_context(|| {
                format!("open stable logs root descriptor {}", canonical.display())
            })?);
            Ok(ConfiguredRoot {
                ordinal,
                path: canonical,
                #[cfg(target_os = "linux")]
                directory,
            })
        })
        .collect()
}

async fn list_runs(
    State(state): State<Arc<ServiceState>>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    let records = discover_runs_cached(&state)
        .await
        .map_err(discovery_problem)?;
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
    let run = resolve_run(&state, &run_id).await?;
    let stamp = descriptor_stamp(&run);
    if let Some(cached) = state
        .descriptors
        .read()
        .map_err(|_| {
            ApiProblem::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "descriptor_cache_failed",
                "Descriptor cache failed",
                "The descriptor cache lock is poisoned.",
            )
        })?
        .get(&run_id)
        .filter(|cached| cached.stamp == stamp)
        .cloned()
    {
        return conditional_bytes(&headers, "application/json", cached.bytes);
    }
    let descriptor = build_descriptor(&run)?;
    let bytes = encode_json(&descriptor)?;
    {
        let mut cache = state.descriptors.write().map_err(|_| {
            ApiProblem::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "descriptor_cache_failed",
                "Descriptor cache failed",
                "The descriptor cache lock is poisoned.",
            )
        })?;
        if cache.len() >= MAX_DESCRIPTOR_CACHE_ENTRIES && !cache.contains_key(&run_id) {
            cache.clear();
        }
        cache.insert(
            run_id,
            CachedDescriptor {
                stamp,
                bytes: bytes.clone(),
            },
        );
    }
    conditional_bytes(&headers, "application/json", bytes)
}

async fn get_summary(
    State(state): State<Arc<ServiceState>>,
    RoutePath(run_id): RoutePath<String>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    let run = resolve_run(&state, &run_id).await?;
    let bytes = read_json_artifact(&run, Path::new("summary.json"), "summary")?;
    conditional_bytes(&headers, "application/json", bytes)
}

async fn get_topology(
    State(state): State<Arc<ServiceState>>,
    RoutePath(run_id): RoutePath<String>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    let run = resolve_run(&state, &run_id).await?;
    let params = read_json_value(&run, Path::new("raw/params.json"), "topology params")?;
    let run_meta = read_json_value(&run, Path::new("raw/run_meta.json"), "topology run_meta")?;
    if !params.is_object() || !run_meta.is_object() {
        return Err(ApiProblem::artifact_incompatible(
            "topology",
            "params and run_meta must both be JSON objects",
        ));
    }
    let envelope = serde_json::json!({
        "schema_version": TOPOLOGY_SCHEMA_VERSION,
        "params": params,
        "run_meta": run_meta,
    });
    json_response(&headers, &envelope)
}

async fn get_subject_report(
    State(state): State<Arc<ServiceState>>,
    RoutePath((run_id, revision, subject_id)): RoutePath<(String, String, String)>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    get_subject_artifact(&state, &run_id, &revision, &subject_id, true, &headers).await
}

async fn get_subject_payload(
    State(state): State<Arc<ServiceState>>,
    RoutePath((run_id, revision, subject_id)): RoutePath<(String, String, String)>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    get_subject_artifact(&state, &run_id, &revision, &subject_id, false, &headers).await
}

async fn get_subject_artifact(
    state: &Arc<ServiceState>,
    run_id: &str,
    revision: &str,
    subject_id: &str,
    report: bool,
    headers: &HeaderMap,
) -> Result<Response, ApiProblem> {
    let run = resolve_run(state, run_id).await?;
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
    let bytes = with_consistent_analysis_snapshot(&run, |snapshot| {
        snapshot.require_revision(revision)?;
        let lifecycle = lifecycle_from(&run, snapshot.pipeline(), snapshot.timing());
        let mut artifact_bytes = SubjectArtifactBytes::default();
        if !matches!(
            subject_state(
                &run,
                subject,
                &deployment,
                lifecycle,
                snapshot.pipeline(),
                snapshot.timing(),
                snapshot.artifact_revision(),
                Some(&mut artifact_bytes),
            ),
            SubjectState::Ready { .. }
        ) {
            return Err(ApiProblem::new(
                StatusCode::NOT_FOUND,
                "resource_not_ready",
                "Resource is not ready",
                format!("Analyzer subject {subject_id:?} does not declare a ready artifact."),
            ));
        }
        Ok(if report {
            artifact_bytes.report
        } else {
            artifact_bytes.payload
        })
    })?;
    conditional_bytes(headers, "application/json", bytes)
}

async fn get_perfetto_trace(
    State(state): State<Arc<ServiceState>>,
    RoutePath((run_id, revision)): RoutePath<(String, String)>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    let run = resolve_run(&state, &run_id).await?;
    let prepared = tokio::task::spawn_blocking(move || {
        with_consistent_analysis_snapshot(&run, |snapshot| {
            snapshot.require_revision(&revision)?;
            let lifecycle = lifecycle_from(&run, snapshot.pipeline(), snapshot.timing());
            if !matches!(
                trace_state(
                    &run,
                    lifecycle,
                    snapshot.pipeline(),
                    snapshot.timing(),
                    snapshot.artifact_revision(),
                ),
                TraceState::Ready { .. }
            ) {
                return Err(ApiProblem::new(
                    StatusCode::NOT_FOUND,
                    "resource_not_ready",
                    "Resource is not ready",
                    "This run does not declare a ready Perfetto trace.",
                ));
            }
            let path = select_trace_for_pipeline(&run, snapshot.pipeline())?;
            let media_type = trace_media_type(&path);
            // Keep this fd open across the final snapshot validation. Atomic
            // replacement cannot change the inode that the response streams.
            Ok((media_type, prepare_trace(&run, path)?))
        })
    })
    .await
    .map_err(|error| {
        ApiProblem::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "artifact_read_failed",
            "Artifact read failed",
            format!("Perfetto trace reader task failed: {error}"),
        )
    })??;
    conditional_prepared_trace(&headers, prepared.0, prepared.1)
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

async fn enforce_allowed_host(
    State(state): State<Arc<ServiceState>>,
    request: Request,
    next: Next,
) -> Response {
    let host_headers = request.headers().get_all(header::HOST);
    let mut values = host_headers.iter();
    let host = values
        .next()
        .and_then(|value| value.to_str().ok())
        .and_then(|value| Authority::from_str(value).ok())
        .map(|authority| normalize_host(authority.host()));
    if values.next().is_some()
        || !host.is_some_and(|host| state.allowed_hosts.contains(host.as_str()))
    {
        return ApiProblem::new(
            StatusCode::MISDIRECTED_REQUEST,
            "host_not_allowed",
            "Host is not allowed",
            "The request Host is not in the analyzer service allowlist.",
        )
        .into_response();
    }
    next.run(request).await
}
