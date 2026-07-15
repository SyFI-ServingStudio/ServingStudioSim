//! Read-only protocol-v1 HTTP boundary for analyzer artifacts.
//!
//! The service deliberately is not a file server. Every public resource is
//! selected from either the fixed core endpoints or [`crate::registry::SUBJECTS`]
//! after an opaque run id has resolved to a canonical, contained run directory.
//! This keeps parquet, cache-build trees, and arbitrary paths outside the HTTP
//! surface even when a caller guesses their on-disk names.

mod artifact;
mod catalog;
mod discovery;
mod lifecycle;
#[cfg(test)]
mod tests;

use artifact::*;
use catalog::*;
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
use std::sync::{Arc, Mutex as StdMutex, RwLock};
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
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tokio_util::io::ReaderStream;

use crate::registry::{Scope, Subject, SUBJECTS};

const PROTOCOL_VERSION: u32 = 1;
const TOPOLOGY_SCHEMA_VERSION: u32 = 1;
const PIPELINE_SCHEMA_VERSION: u32 = 1;
const PIPELINE_STATE_PATH: &str = "reports/analyzer_pipeline_state.json";
const LEGACY_GENERATOR_VERSION: &str = "legacy-unknown";
const MAX_PIPELINE_STATE_BYTES: u64 = 1024 * 1024;
// Run-scope payloads are deliberately downsampled by the analyzer contract. The
// largest checked-in/live artifact observed while setting this boundary was
// 1.49 MiB, so 16 MiB leaves ample evolution room without allowing one request
// to reserve an unreviewed 128 MiB JSON document.
const MAX_JSON_BYTES: u64 = 16 * 1024 * 1024;
// A catalog cold build reads pipeline/timing JSON across every discovered run.
// Keep that fan-in within one service-wide budget instead of multiplying the
// per-file JSON limit by the number of runs.
const MAX_CATALOG_LIFECYCLE_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TRACE_BYTES: u64 = 512 * 1024 * 1024;
const CATALOG_REFRESH_INTERVAL: Duration = Duration::from_secs(30);
const MAX_DESCRIPTOR_CACHE_ENTRIES: usize = 256;
const MAX_SUBJECT_PROOF_CACHE_ENTRIES: usize = 2048;
const MAX_GENERATION_READ_ATTEMPTS: usize = 3;
// JSON parsing temporarily retains both input bytes and serde's value tree.
// Four readers bound that amplification while leaving enough parallelism for
// the UI's summary/topology/subject startup fan-out.
const MAX_IN_FLIGHT_ARTIFACT_READS: usize = 4;

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
    catalog: RwLock<Option<Arc<CachedCatalog>>>,
    catalog_build: StdMutex<()>,
    descriptors: RwLock<HashMap<String, CachedDescriptor>>,
    descriptor_build: StdMutex<()>,
    subject_proofs: RwLock<HashMap<SubjectProofKey, SubjectReadinessProof>>,
    artifact_reads: Arc<Semaphore>,
    allowed_hosts: HashSet<String>,
}

#[derive(Clone, Debug)]
struct CachedDescriptor {
    stamp: String,
    etag: String,
    bytes: Vec<u8>,
    subject_proofs: Vec<(String, String, SubjectReadinessProof)>,
    analysis_proof: Option<AnalysisReadinessProof>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct SubjectProofKey {
    run_id: String,
    revision: String,
    subject: String,
}

#[derive(Clone, Debug)]
struct SubjectReadinessProof {
    /// Analysis-wide metadata fence. For legacy runs its value memoizes the
    /// content-derived revision without replacing that revision's SHA contract.
    analysis_fence: String,
    pair_fence: String,
    legacy: bool,
}

#[derive(Clone, Debug)]
struct AnalysisReadinessProof {
    revision: String,
    analysis_fence: String,
    legacy: bool,
    trace_ready: bool,
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
        if self.code == "artifact_read_busy" {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, header::HeaderValue::from_static("1"));
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
        catalog: RwLock::new(None),
        catalog_build: StdMutex::new(()),
        descriptors: RwLock::new(HashMap::new()),
        descriptor_build: StdMutex::new(()),
        subject_proofs: RwLock::new(HashMap::new()),
        artifact_reads: Arc::new(Semaphore::new(MAX_IN_FLIGHT_ARTIFACT_READS)),
        allowed_hosts: configure_allowed_hosts(allow_hosts)?,
    });
    Ok(router_from_state(state))
}

fn router_from_state(state: Arc<ServiceState>) -> Router {
    Router::new()
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
        .with_state(state)
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
    let task_state = Arc::clone(&state);
    let task_headers = headers.clone();
    let prepared = run_blocking_artifact_task(&state, "catalog lifecycle", move || {
        prepare_catalog(&task_state, &task_headers, &records)
    })
    .await?;
    conditional_response("application/json", prepared)
}

async fn get_descriptor(
    State(state): State<Arc<ServiceState>>,
    RoutePath(run_id): RoutePath<String>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    let run = resolve_run(&state, &run_id).await?;
    let task_state = Arc::clone(&state);
    let task_headers = headers.clone();
    let prepared = run_blocking_artifact_task(&state, "descriptor build", move || {
        let cached = get_or_build_descriptor(&task_state, &run, &run_id)?;
        if if_none_match(&task_headers, &cached.etag) {
            Ok(ConditionalBody::NotModified { etag: cached.etag })
        } else {
            Ok(ConditionalBody::Bytes {
                etag: cached.etag,
                bytes: cached.bytes,
            })
        }
    })
    .await?;
    conditional_response("application/json", prepared)
}

fn descriptor_cache_problem() -> ApiProblem {
    ApiProblem::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "descriptor_cache_failed",
        "Descriptor cache failed",
        "The descriptor cache lock is poisoned.",
    )
}

fn get_or_build_descriptor(
    state: &ServiceState,
    run: &RunRecord,
    run_id: &str,
) -> Result<CachedDescriptor, ApiProblem> {
    let stamp = descriptor_stamp(run);
    if let Some(cached) = state
        .descriptors
        .read()
        .map_err(|_| descriptor_cache_problem())?
        .get(run_id)
        .filter(|cached| cached.stamp == stamp)
        .cloned()
    {
        return Ok(cached);
    }

    // This synchronous mutex is acquired only inside the bounded blocking
    // pool. It single-flights cold descriptor/content-hash work without ever
    // parking a Tokio async worker.
    let _build_guard = state
        .descriptor_build
        .lock()
        .map_err(|_| descriptor_cache_problem())?;
    let stamp = descriptor_stamp(run);
    if let Some(cached) = state
        .descriptors
        .read()
        .map_err(|_| descriptor_cache_problem())?
        .get(run_id)
        .filter(|cached| cached.stamp == stamp)
        .cloned()
    {
        return Ok(cached);
    }

    let (descriptor, stable_stamp, readiness_proofs, analysis_proof) =
        build_descriptor_with_readiness_proofs(run)?;
    let cached = CachedDescriptor {
        etag: format!("W/\"descriptor-v1-{stable_stamp}\""),
        stamp: stable_stamp,
        bytes: encode_json(&descriptor)?,
        subject_proofs: readiness_proofs.clone(),
        analysis_proof,
    };
    cache_readiness_proofs(state, run_id, readiness_proofs)?;
    let mut descriptors = state
        .descriptors
        .write()
        .map_err(|_| descriptor_cache_problem())?;
    if descriptors.len() >= MAX_DESCRIPTOR_CACHE_ENTRIES && !descriptors.contains_key(run_id) {
        descriptors.clear();
    }
    descriptors.insert(run_id.to_string(), cached.clone());
    Ok(cached)
}

fn build_descriptor_with_readiness_proofs(
    run: &RunRecord,
) -> Result<
    (
        RunDescriptor,
        String,
        Vec<(String, String, SubjectReadinessProof)>,
        Option<AnalysisReadinessProof>,
    ),
    ApiProblem,
> {
    with_consistent_analysis_snapshot(run, |snapshot| {
        let fence_before = descriptor_stamp(run);
        let descriptor = build_descriptor_at_snapshot(run, snapshot)?;
        let revision = snapshot.artifact_revision().map(str::to_owned);
        let legacy = matches!(snapshot.pipeline(), PipelineStateRead::Missing);
        let mut readiness_proofs = Vec::new();
        if let Some(revision) = &revision {
            for subject in SUBJECTS
                .iter()
                .filter(|subject| subject.scope == Scope::Run)
            {
                if matches!(
                    descriptor.subjects.get(subject.name),
                    Some(SubjectState::Ready { .. })
                ) {
                    readiness_proofs.push((
                        revision.clone(),
                        subject.name.to_string(),
                        SubjectReadinessProof {
                            analysis_fence: String::new(),
                            pair_fence: subject_pair_fence(run, subject)?,
                            legacy,
                        },
                    ));
                }
            }
        }
        let fence_after = descriptor_stamp(run);
        if fence_before != fence_after {
            return Err(ApiProblem::artifact_generation_changed(
                "Analyzer artifact metadata changed while descriptor readiness proofs were built.",
            ));
        }
        for (_, _, proof) in &mut readiness_proofs {
            proof.analysis_fence.clone_from(&fence_after);
        }
        let analysis_proof = revision.map(|revision| AnalysisReadinessProof {
            revision,
            analysis_fence: fence_after.clone(),
            legacy,
            trace_ready: matches!(
                descriptor.traces.get("perfetto"),
                Some(TraceState::Ready { .. })
            ),
        });
        Ok((descriptor, fence_after, readiness_proofs, analysis_proof))
    })
}

fn cache_readiness_proofs(
    state: &ServiceState,
    run_id: &str,
    readiness_proofs: Vec<(String, String, SubjectReadinessProof)>,
) -> Result<(), ApiProblem> {
    let mut cache = state
        .subject_proofs
        .write()
        .map_err(|_| descriptor_cache_problem())?;
    if cache.len().saturating_add(readiness_proofs.len()) > MAX_SUBJECT_PROOF_CACHE_ENTRIES {
        cache.clear();
    }
    for (revision, subject, proof) in readiness_proofs {
        cache.insert(
            SubjectProofKey {
                run_id: run_id.to_string(),
                revision,
                subject,
            },
            proof,
        );
    }
    Ok(())
}

fn subject_pair_fence(run: &RunRecord, subject: &Subject) -> Result<String, ApiProblem> {
    let report_relative = Path::new("reports").join(subject.report_name);
    let payload_relative = Path::new("payloads").join(subject.payload_name);
    let report = open_bounded_artifact(run, &report_relative, MAX_JSON_BYTES, "subject report")?;
    let payload = open_bounded_artifact(run, &payload_relative, MAX_JSON_BYTES, "subject payload")?;
    Ok(metadata_etag(
        "subject-pair-v1",
        &[
            (&report_relative, &report.metadata),
            (&payload_relative, &payload.metadata),
        ],
    ))
}

async fn get_summary(
    State(state): State<Arc<ServiceState>>,
    RoutePath(run_id): RoutePath<String>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    let run = resolve_run(&state, &run_id).await?;
    let task_headers = headers.clone();
    let prepared = run_blocking_artifact_task(&state, "summary read", move || {
        prepare_json_artifact(
            &task_headers,
            &run,
            Path::new("summary.json"),
            MAX_JSON_BYTES,
            "summary",
            |_| Ok(()),
        )
    })
    .await?;
    conditional_response("application/json", prepared)
}

async fn get_topology(
    State(state): State<Arc<ServiceState>>,
    RoutePath(run_id): RoutePath<String>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    let run = resolve_run(&state, &run_id).await?;
    let task_headers = headers.clone();
    let prepared = run_blocking_artifact_task(&state, "topology build", move || {
        prepare_topology(&task_headers, &run)
    })
    .await?;
    conditional_response("application/json", prepared)
}

fn prepare_topology(headers: &HeaderMap, run: &RunRecord) -> Result<ConditionalBody, ApiProblem> {
    let params_relative = Path::new("raw/params.json");
    let run_meta_relative = Path::new("raw/run_meta.json");
    let params_artifact =
        open_bounded_artifact(run, params_relative, MAX_JSON_BYTES, "topology params")?;
    let run_meta_artifact =
        open_bounded_artifact(run, run_meta_relative, MAX_JSON_BYTES, "topology run_meta")?;
    let etag = metadata_etag(
        "topology-v1",
        &[
            (params_relative, &params_artifact.metadata),
            (run_meta_relative, &run_meta_artifact.metadata),
        ],
    );
    if if_none_match_exact(headers, &etag) {
        return Ok(ConditionalBody::NotModified { etag });
    }
    let (params_bytes, _) =
        read_bounded_open_file(params_artifact, MAX_JSON_BYTES, "topology params")?;
    let (run_meta_bytes, _) =
        read_bounded_open_file(run_meta_artifact, MAX_JSON_BYTES, "topology run_meta")?;
    let params: Value = serde_json::from_slice(&params_bytes)
        .map_err(|error| ApiProblem::artifact_incompatible("topology params", error.to_string()))?;
    let run_meta: Value = serde_json::from_slice(&run_meta_bytes).map_err(|error| {
        ApiProblem::artifact_incompatible("topology run_meta", error.to_string())
    })?;
    if !params.is_object() || !run_meta.is_object() {
        return Err(ApiProblem::artifact_incompatible(
            "topology",
            "params and run_meta must both be JSON objects",
        ));
    }
    if if_none_match(headers, &etag) {
        return Ok(ConditionalBody::NotModified { etag });
    }
    let bytes = encode_json(&serde_json::json!({
        "schema_version": TOPOLOGY_SCHEMA_VERSION,
        "params": params,
        "run_meta": run_meta,
    }))?;
    Ok(ConditionalBody::Bytes { etag, bytes })
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
    let task_headers = headers.clone();
    let task_state = Arc::clone(state);
    let run_id = run_id.to_string();
    let revision = revision.to_string();
    let subject_id = subject_id.to_string();
    let prepared = run_blocking_artifact_task(state, "subject artifact read", move || {
        let proof = get_subject_readiness_proof(&task_state, &run, &run_id, subject, &revision)?;
        prepare_subject_artifact(
            &task_headers,
            &run,
            subject,
            &revision,
            &subject_id,
            report,
            &proof,
        )
    })
    .await?;
    conditional_response("application/json", prepared)
}

fn prepare_subject_artifact(
    headers: &HeaderMap,
    run: &RunRecord,
    subject: &'static Subject,
    revision: &str,
    subject_id: &str,
    report: bool,
    proof: &SubjectReadinessProof,
) -> Result<ConditionalBody, ApiProblem> {
    if proof.legacy {
        return prepare_legacy_subject_artifact(headers, run, subject, subject_id, report, proof);
    }

    let deployment = read_deployment(run)?;
    with_consistent_analysis_snapshot(run, |snapshot| {
        snapshot.require_revision(revision)?;
        if subject_preflight_state(subject, &deployment, snapshot.pipeline(), snapshot.timing())
            .is_some()
        {
            return Err(subject_not_ready(subject_id));
        }
        prepare_proven_subject_artifact(headers, run, subject, subject_id, report, proof)
    })
}

fn get_subject_readiness_proof(
    state: &ServiceState,
    run: &RunRecord,
    run_id: &str,
    subject: &Subject,
    revision: &str,
) -> Result<SubjectReadinessProof, ApiProblem> {
    let key = SubjectProofKey {
        run_id: run_id.to_string(),
        revision: revision.to_string(),
        subject: subject.name.to_string(),
    };
    if let Some(proof) = state
        .subject_proofs
        .read()
        .map_err(|_| descriptor_cache_problem())?
        .get(&key)
        .cloned()
    {
        return Ok(proof);
    }

    let cached = get_or_build_descriptor(state, run, run_id)?;
    cache_readiness_proofs(state, run_id, cached.subject_proofs.clone())?;
    let current_revision = cached
        .analysis_proof
        .as_ref()
        .map(|proof| proof.revision.as_str());
    cached
        .subject_proofs
        .into_iter()
        .find_map(|(proof_revision, proof_subject, proof)| {
            (proof_revision == revision && proof_subject == subject.name).then_some(proof)
        })
        .ok_or_else(|| match current_revision {
            Some(current) if current == revision => subject_not_ready(subject.name),
            _ => ApiProblem::artifact_generation_changed(format!(
                "Revision {revision:?} has no cached ready pair for subject {:?}; refresh the descriptor.",
                subject.name
            )),
        })
}

fn prepare_legacy_subject_artifact(
    headers: &HeaderMap,
    run: &RunRecord,
    subject: &Subject,
    subject_id: &str,
    report: bool,
    proof: &SubjectReadinessProof,
) -> Result<ConditionalBody, ApiProblem> {
    if !matches!(read_pipeline_state(run), PipelineStateRead::Missing)
        || descriptor_stamp(run) != proof.analysis_fence
    {
        return Err(ApiProblem::artifact_generation_changed(
            "Legacy analysis metadata changed after its content-derived revision was cached.",
        ));
    }
    let prepared =
        prepare_proven_subject_artifact(headers, run, subject, subject_id, report, proof)?;
    if !matches!(read_pipeline_state(run), PipelineStateRead::Missing)
        || descriptor_stamp(run) != proof.analysis_fence
    {
        return Err(ApiProblem::artifact_generation_changed(
            "Legacy analysis metadata changed while its artifact was being read.",
        ));
    }
    Ok(prepared)
}

fn prepare_proven_subject_artifact(
    headers: &HeaderMap,
    run: &RunRecord,
    subject: &Subject,
    subject_id: &str,
    report: bool,
    proof: &SubjectReadinessProof,
) -> Result<ConditionalBody, ApiProblem> {
    if subject_pair_fence(run, subject)? != proof.pair_fence {
        return Err(ApiProblem::artifact_generation_changed(
            "The analyzer report/payload pair changed after descriptor validation.",
        ));
    }
    let (relative, resource) = if report {
        (
            Path::new("reports").join(subject.report_name),
            "subject report",
        )
    } else {
        (
            Path::new("payloads").join(subject.payload_name),
            "subject payload",
        )
    };
    let prepared =
        prepare_json_artifact(headers, run, &relative, MAX_JSON_BYTES, resource, |value| {
            validate_subject_artifact(value, resource, subject_id)
        })?;
    if subject_pair_fence(run, subject)? != proof.pair_fence {
        return Err(ApiProblem::artifact_generation_changed(
            "The analyzer report/payload pair changed while its artifact was being read.",
        ));
    }
    Ok(prepared)
}

fn validate_subject_artifact(
    value: &Value,
    resource: &str,
    subject_id: &str,
) -> Result<(), ApiProblem> {
    let schema_version = value.get("schema_version").and_then(Value::as_u64);
    if !schema_version.is_some_and(|version| version > 0 && u32::try_from(version).is_ok()) {
        return Err(ApiProblem::artifact_incompatible(
            resource,
            "schema_version must be a positive protocol-v1 integer",
        ));
    }
    if artifact_available(value) == Some(false) {
        return Err(subject_not_ready(subject_id));
    }
    Ok(())
}

fn subject_not_ready(subject_id: &str) -> ApiProblem {
    ApiProblem::new(
        StatusCode::NOT_FOUND,
        "resource_not_ready",
        "Resource is not ready",
        format!("Analyzer subject {subject_id:?} does not declare a ready artifact."),
    )
}

async fn get_perfetto_trace(
    State(state): State<Arc<ServiceState>>,
    RoutePath((run_id, revision)): RoutePath<(String, String)>,
    headers: HeaderMap,
) -> Result<Response, ApiProblem> {
    let run = resolve_run(&state, &run_id).await?;
    let task_state = Arc::clone(&state);
    let ((media_type, prepared), permit) =
        run_blocking_stream_task(&state, "Perfetto trace preparation", move || {
            let cached = get_or_build_descriptor(&task_state, &run, &run_id)?;
            let proof = cached.analysis_proof.ok_or_else(|| {
                ApiProblem::artifact_generation_changed(
                    "The requested trace has no cached analysis revision proof.",
                )
            })?;
            if proof.revision != revision {
                return Err(ApiProblem::artifact_generation_changed(format!(
                    "Trace revision {revision:?} is no longer current."
                )));
            }
            if !proof.trace_ready {
                return Err(ApiProblem::new(
                    StatusCode::NOT_FOUND,
                    "resource_not_ready",
                    "Resource is not ready",
                    "This run does not declare a ready Perfetto trace.",
                ));
            }
            if proof.legacy {
                return prepare_legacy_trace(&run, &proof);
            }
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
                // Keep this fd open across the final snapshot validation. Atomic
                // replacement cannot change the inode that the response streams.
                Ok((trace_media_type(&path), prepare_trace(&run, path)?))
            })
        })
        .await?;
    conditional_prepared_trace(&headers, media_type, prepared, permit)
}

fn prepare_legacy_trace(
    run: &RunRecord,
    proof: &AnalysisReadinessProof,
) -> Result<(&'static str, PreparedTrace), ApiProblem> {
    if !matches!(read_pipeline_state(run), PipelineStateRead::Missing)
        || descriptor_stamp(run) != proof.analysis_fence
    {
        return Err(ApiProblem::artifact_generation_changed(
            "Legacy analysis metadata changed after its trace revision was cached.",
        ));
    }
    let path = select_trace(run)?;
    let prepared = (trace_media_type(&path), prepare_trace(run, path)?);
    if !matches!(read_pipeline_state(run), PipelineStateRead::Missing)
        || descriptor_stamp(run) != proof.analysis_fence
    {
        return Err(ApiProblem::artifact_generation_changed(
            "Legacy analysis metadata changed while its trace was being opened.",
        ));
    }
    Ok(prepared)
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
