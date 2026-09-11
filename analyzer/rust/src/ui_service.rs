//! Minimal read-only HTTP wiring between analyzer run directories and viz-ui.
//!
//! Transport stays here. Filesystem discovery, catalog projection, and core run
//! resources live in separate modules so later subjects do not grow one service
//! file into a second analyzer.

mod alignment;
mod artifact;
mod artifact_kind;
mod batch;
mod catalog;
mod concurrency;
mod core;
mod discovery;
mod hardware;
mod kernel_input_distribution;
mod kernel_measurement;
mod kernel_profile;
mod kernel_throughput_analysis;
mod kernel_time_share;
mod kv_occupancy;
mod model;
mod optimality;
mod prediction;
mod request_state;
mod slo;
mod sweep;
#[cfg(test)]
mod tests;
mod throughput;
mod topology;
mod utilization;
mod worker_detail;
mod workload;
mod workload_conservation;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Path as RoutePath, Query, Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tower_http::compression::CompressionLayer;

use crate::optimality::{
    iteration_kernel_ladder, iteration_waterfall, prediction_kernel_ladder, prediction_waterfall,
};
use alignment::{
    alignment_artifact_bytes, alignment_descriptor,
    alignment_detail_index as build_alignment_detail_index,
    alignment_sequence_index as build_alignment_sequence_index, build_alignment_catalog,
    discover_alignments, operation_split_projection, read_alignment_iteration_detail,
    read_alignment_sequence, reference_lane_projection, AlignmentDetailIndex,
    AlignmentSequenceIndex, DiscoveredAlignment,
};
use artifact::read_json;
use batch::{read_batch_payload, read_batch_report};
use catalog::build_catalog;
use concurrency::{read_concurrency_payload, read_concurrency_report};
use core::{build_descriptor, read_summary};
use discovery::{
    configure_logs_roots, configure_workspace_registry, resolve_run, ConfiguredRoot, DiscoveredRun,
};
use hardware::{hardware_gpu_response, resolve_gpu};
use kernel_input_distribution::{
    read_kernel_input_distribution_payload, read_kernel_input_distribution_report,
};
use kernel_measurement::{
    build_kernel_measurement_catalog, measurement_descriptor, measurement_plot,
    measurement_summary, resolve_kernel_measurement,
};
use kernel_profile::{
    build_kernel_profile_catalog, profile_curve, profile_descriptor, resolve_kernel_profile,
};
use kernel_throughput_analysis::analyze_kernel_throughput;
use kernel_time_share::{
    read_kernel_time_share_payload, read_kernel_time_share_report,
    read_kernel_time_share_worker_payload,
};
use kv_occupancy::{read_kv_occupancy_payload, read_kv_occupancy_report};
use model::read_model;
use optimality::{
    read_locked_optimality_payload, read_locked_optimality_report, read_optimality_payload,
    read_optimality_report,
};
use prediction::{
    build_prediction_catalog, prediction_cases, prediction_descriptor, resolve_prediction,
    DiscoveredPrediction,
};
use request_state::{read_request_state_payload, read_request_state_report};
use slo::{read_slo_general_payload, read_slo_general_report};
use sweep::{
    build_filtered_sweep_catalog, read_sweep_payload, resolve_sweep, SweepCatalogFilter,
    SweepStatus,
};
use throughput::{read_throughput_payload, read_throughput_report};
use topology::build_topology;
use utilization::{read_utilization_payload, read_utilization_report};
use worker_detail::{
    exact_operation_cost_tree, operation_range, operation_seek, prediction_operation_cost_tree,
    OperationIndexCache, OperationRange,
};
use workload::read_workload;
use workload_conservation::{
    read_workload_conservation_payload, read_workload_conservation_report,
};

const PROTOCOL_VERSION: u32 = 1;
const TIMELINE_PROFILE_BODY_LIMIT: usize = 16 * 1024;
/// Directory discovery and shard-index parsing are process-local caches. A
/// one-second TTL still made a user click after a short pause repeat the full
/// workspace walk, so keep one page's repeated cycle selection hot while
/// allowing newly generated alignment bundles to appear without a restart.
const ALIGNMENT_DISCOVERY_CACHE_TTL: Duration = Duration::from_secs(60);
static NEXT_WORKER_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct ServiceState {
    root_source: Arc<RootSource>,
    repo_root: Arc<PathBuf>,
    operation_indexes: Arc<OperationIndexCache>,
    alignment_discovery: Arc<Mutex<Option<CachedAlignmentDiscovery>>>,
    alignment_detail_indexes: Arc<Mutex<HashMap<(String, String), CachedAlignmentDetailIndex>>>,
    alignment_sequence_indexes: Arc<Mutex<HashMap<String, CachedAlignmentSequenceIndex>>>,
}

struct CachedAlignmentDiscovery {
    loaded_at: Instant,
    alignments: Vec<DiscoveredAlignment>,
}

struct CachedAlignmentDetailIndex {
    loaded_at: Instant,
    index: Arc<AlignmentDetailIndex>,
}

struct CachedAlignmentSequenceIndex {
    loaded_at: Instant,
    index: Arc<AlignmentSequenceIndex>,
}

#[derive(Clone)]
enum RootSource {
    Static(Arc<Vec<ConfiguredRoot>>),
    WorkspaceRegistry(PathBuf),
}

impl RootSource {
    fn load(&self) -> Result<Vec<ConfiguredRoot>> {
        match self {
            Self::Static(roots) => Ok(roots.as_ref().clone()),
            Self::WorkspaceRegistry(path) => configure_workspace_registry(path),
        }
    }
}

impl ServiceState {
    fn resolve_run(&self, run_id: &str) -> Result<DiscoveredRun> {
        resolve_run(&self.root_source.load()?, run_id)
    }

    fn resolve_prediction(&self, prediction_id: &str) -> Result<DiscoveredPrediction> {
        resolve_prediction(&self.root_source.load()?, prediction_id)
    }

    fn resolve_alignment(&self, alignment_id: &str) -> Result<DiscoveredAlignment> {
        let mut cache = self
            .alignment_discovery
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cache
            .as_ref()
            .is_some_and(|value| value.loaded_at.elapsed() < ALIGNMENT_DISCOVERY_CACHE_TTL)
        {
            return cache
                .as_ref()
                .and_then(|value| {
                    value
                        .alignments
                        .iter()
                        .find(|alignment| alignment.alignment_id == alignment_id)
                        .cloned()
                })
                .ok_or_else(|| AlignmentNotFound.into());
        }

        // Hold the lock through the scan. Several page sections issue their
        // first request together; serializing only this cache fill prevents
        // every request from recursively walking the same workspace tree.
        let roots = self.root_source.load()?;
        let alignments = discover_alignments(&roots)?;
        let found = alignments
            .iter()
            .find(|alignment| alignment.alignment_id == alignment_id)
            .cloned();
        *cache = Some(CachedAlignmentDiscovery {
            loaded_at: Instant::now(),
            alignments,
        });
        found.ok_or_else(|| AlignmentNotFound.into())
    }

    /// Parse the small `byte_ranges` document once for each subject while it
    /// is being used by a page. The old path reparsed this index for every
    /// selected iteration, even though the following shard read only needs a
    /// single `(offset, length)` pair.
    fn alignment_detail_index(
        &self,
        alignment_id: &str,
        alignment: &DiscoveredAlignment,
        subject: &str,
    ) -> Result<Arc<AlignmentDetailIndex>> {
        let key = (alignment_id.to_owned(), subject.to_owned());
        let mut cache = self
            .alignment_detail_indexes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = cache.get(&key) {
            if cached.loaded_at.elapsed() < ALIGNMENT_DISCOVERY_CACHE_TTL {
                return Ok(Arc::clone(&cached.index));
            }
        }

        // Keep the mutex held through construction just like discovery above:
        // concurrent timeline requests for the same subject should share one
        // index parse, not all perform the same JSON read.
        let index = Arc::new(build_alignment_detail_index(alignment, subject)?);
        cache.insert(
            key,
            CachedAlignmentDetailIndex {
                loaded_at: Instant::now(),
                index: Arc::clone(&index),
            },
        );
        Ok(index)
    }

    fn alignment_sequence_index(
        &self,
        alignment_id: &str,
        alignment: &DiscoveredAlignment,
    ) -> Result<Arc<AlignmentSequenceIndex>> {
        let mut cache = self
            .alignment_sequence_indexes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = cache.get(alignment_id) {
            if cached.loaded_at.elapsed() < ALIGNMENT_DISCOVERY_CACHE_TTL {
                return Ok(Arc::clone(&cached.index));
            }
        }
        let index = Arc::new(build_alignment_sequence_index(alignment)?);
        cache.insert(
            alignment_id.to_owned(),
            CachedAlignmentSequenceIndex {
                loaded_at: Instant::now(),
                index: Arc::clone(&index),
            },
        );
        Ok(index)
    }

    fn resolve_kernel_profile(
        &self,
        profile_id: &str,
    ) -> Result<kernel_profile::DiscoveredKernelProfile> {
        resolve_kernel_profile(&self.root_source.load()?, profile_id)
    }

    fn resolve_kernel_measurement(
        &self,
        measurement_id: &str,
    ) -> Result<kernel_measurement::DiscoveredKernelMeasurement> {
        resolve_kernel_measurement(&self.root_source.load()?, measurement_id)
    }
}

pub(crate) async fn serve(
    bind: SocketAddr,
    logs_roots: Vec<PathBuf>,
    workspace_registry: Option<PathBuf>,
) -> Result<()> {
    let root_source = if let Some(path) = workspace_registry {
        // Validate once at startup, then reload on every request.
        configure_workspace_registry(&path)?;
        RootSource::WorkspaceRegistry(path)
    } else {
        RootSource::Static(Arc::new(configure_logs_roots(logs_roots)?))
    };
    let repo_root = std::env::current_dir().context("resolve analyzer repository root")?;
    let state = ServiceState {
        root_source: Arc::new(root_source),
        repo_root: Arc::new(repo_root),
        operation_indexes: Arc::new(OperationIndexCache::default()),
        alignment_discovery: Arc::new(Mutex::new(None)),
        alignment_detail_indexes: Arc::new(Mutex::new(HashMap::new())),
        alignment_sequence_indexes: Arc::new(Mutex::new(HashMap::new())),
    };
    let app = service_router(state);
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("bind analyzer UI service to {bind}"))?;
    eprintln!(
        "[analyze] serving run, sweep, prediction, alignment, kernel-profile, kernel-measurement, and hardware catalogs at http://{bind}/api/analyzer/v1/{{runs,sweeps,predictions,alignments,kernel-profiles,kernel-measurements,hardware}}"
    );
    axum::serve(listener, app)
        .await
        .context("serve analyzer run catalog")
}

/// Build the complete read-only HTTP application independently of its socket.
/// Keeping transport startup outside this function lets contract tests exercise
/// the exact production route table without binding a port or duplicating
/// handler wiring.
fn service_router(state: ServiceState) -> Router {
    read_only_router(state).merge(dev_router())
}

/// The browser-profiling sink.
///
/// It is a `POST` that writes a line to the analyzer's stderr, and it used to
/// sit on `/api/v1/profile/timeline` — inside the tree that otherwise only ever
/// reads finished analysis off disk. Sharing a prefix with the read API made
/// the read API look like it accepts writes: a reverse proxy or a deployment
/// policy that wants to expose the artifacts read-only had no prefix to say so
/// with.
///
/// `/api/dev/v1/` is that prefix. The body limit lives here too, because this
/// is the only route in the service that receives a body at all.
fn dev_router() -> Router {
    Router::new()
        .route("/api/dev/v1/profile/timeline", post(profile_timeline))
        .layer(DefaultBodyLimit::max(TIMELINE_PROFILE_BODY_LIMIT))
}

/// Every route that reads an artifact. No handler here takes a request body.
fn read_only_router(state: ServiceState) -> Router {
    Router::new()
        .route("/api/analyzer/v1/alignments", get(list_alignments))
        .route(
            "/api/analyzer/v1/alignments/{alignment_id}/descriptor",
            get(get_alignment_descriptor),
        )
        .route(
            "/api/analyzer/v1/alignments/{alignment_id}/subjects/{subject}/report",
            get(get_alignment_report),
        )
        .route(
            "/api/analyzer/v1/alignments/{alignment_id}/subjects/{subject}/payload",
            get(get_alignment_payload),
        )
        .route(
            "/api/analyzer/v1/alignments/{alignment_id}/subjects/{subject}/iterations/{iteration_id}",
            get(get_alignment_iteration),
        )
        .route(
            "/api/analyzer/v1/alignments/{alignment_id}/subjects/iteration/sequences/{phase}/{sequence_id}",
            get(get_alignment_sequence),
        )
        .route("/api/analyzer/v1/predictions", get(list_predictions))
        .route(
            "/api/analyzer/v1/predictions/{prediction_id}/descriptor",
            get(get_prediction_descriptor),
        )
        .route(
            "/api/analyzer/v1/predictions/{prediction_id}/subjects/cases/payload",
            get(get_prediction_cases),
        )
        .route(
            "/api/analyzer/v1/predictions/{prediction_id}/subjects/kernel-input-distribution/payload",
            get(get_prediction_kernel_input_distribution),
        )
        .route(
            "/api/analyzer/v1/predictions/{prediction_id}/cases/{case_id}/operations/{operation_id}/subjects/cost-tree/payload",
            get(get_prediction_cost_tree),
        )
        .route(
            "/api/analyzer/v1/predictions/{prediction_id}/cases/{case_id}/operations/{operation_id}/leaves/{leaf_id}/subjects/kernel-throughput-analysis/payload",
            get(get_prediction_kernel_throughput_analysis),
        )
        .route(
            "/api/analyzer/v1/predictions/{prediction_id}/cases/{case_id}/subjects/optimality-kernel-ladder/payload",
            get(get_prediction_optimality_kernel_ladder),
        )
        .route(
            "/api/analyzer/v1/predictions/{prediction_id}/cases/{case_id}/subjects/optimality-waterfall/payload",
            get(get_prediction_optimality_waterfall),
        )
        .route("/api/analyzer/v1/kernel-profiles", get(list_kernel_profiles))
        .route(
            "/api/analyzer/v1/kernel-profiles/{profile_id}/descriptor",
            get(get_kernel_profile_descriptor),
        )
        .route(
            "/api/analyzer/v1/kernel-profiles/{profile_id}/subjects/curve/payload",
            get(get_kernel_profile_curve),
        )
        .route("/api/analyzer/v1/kernel-measurements", get(list_kernel_measurements))
        .route(
            "/api/analyzer/v1/kernel-measurements/{measurement_id}/descriptor",
            get(get_kernel_measurement_descriptor),
        )
        .route(
            "/api/analyzer/v1/kernel-measurements/{measurement_id}/subjects/summary/report",
            get(get_kernel_measurement_summary),
        )
        .route(
            "/api/analyzer/v1/kernel-measurements/{measurement_id}/plots/{plot_name}",
            get(get_kernel_measurement_plot),
        )
        .route("/api/analyzer/v1/hardware/gpus", get(get_hardware_gpus))
        .route("/api/analyzer/v1/runs", get(list_runs))
        .route("/api/analyzer/v1/sweeps", get(list_sweeps))
        .route("/api/analyzer/v1/sweeps/latest", get(get_latest_sweep))
        .route("/api/analyzer/v1/sweeps/{sweep_id}/subjects/sweep/payload", get(get_sweep_payload))
        .route("/api/analyzer/v1/runs/{run_id}/descriptor", get(get_descriptor))
        .route("/api/analyzer/v1/runs/{run_id}/subjects/summary/report", get(get_summary))
        .route("/api/analyzer/v1/runs/{run_id}/subjects/topology/payload", get(get_topology))
        .route("/api/analyzer/v1/runs/{run_id}/subjects/model/payload", get(get_model))
        .route("/api/analyzer/v1/runs/{run_id}/subjects/workload/payload", get(get_workload))
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/concurrency/report",
            get(get_concurrency_report),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/concurrency/payload",
            get(get_concurrency_payload),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/request-state/report",
            get(get_request_state_report),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/request-state/payload",
            get(get_request_state_payload),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/slo-general/report",
            get(get_slo_general_report),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/slo-general/payload",
            get(get_slo_general_payload),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/throughput/report",
            get(get_throughput_report),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/throughput/payload",
            get(get_throughput_payload),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/utilization/report",
            get(get_utilization_report),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/utilization/payload",
            get(get_utilization_payload),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/batch/report",
            get(get_batch_report),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/batch/payload",
            get(get_batch_payload),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/kv-occupancy/report",
            get(get_kv_occupancy_report),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/kv-occupancy/payload",
            get(get_kv_occupancy_payload),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/kernel-input-distribution/report",
            get(get_kernel_input_distribution_report),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/kernel-input-distribution/payload",
            get(get_kernel_input_distribution_payload),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/kernel-time-share/report",
            get(get_kernel_time_share_report),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/kernel-time-share/payload",
            get(get_kernel_time_share_payload),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/workers/{pool_tag}/{worker_id}/subjects/kernel-time-share/payload",
            get(get_worker_kernel_time_share_payload),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/optimality/report",
            get(get_optimality_report),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/optimality/payload",
            get(get_optimality_payload),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/optimality/variants/batch_locked/report",
            get(get_locked_optimality_report),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/optimality/variants/batch_locked/payload",
            get(get_locked_optimality_payload),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/workload-conservation/report",
            get(get_workload_conservation_report),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/subjects/workload-conservation/payload",
            get(get_workload_conservation_payload),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/workers/{pool_tag}/{worker_id}/subjects/operations/payload",
            get(get_worker_operations),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/workers/{pool_tag}/{worker_id}/subjects/operations/seek",
            get(seek_worker_operation),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/workers/{pool_tag}/{worker_id}/iterations/{iter_id}/subjects/optimality-kernel-ladder/payload",
            get(get_iteration_optimality_kernel_ladder),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/workers/{pool_tag}/{worker_id}/iterations/{iter_id}/subjects/optimality-waterfall/payload",
            get(get_iteration_optimality_waterfall),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/workers/{pool_tag}/{worker_id}/operations/{iter_id}/{batch_id}/{operation_id}/subjects/cost-tree/payload",
            get(get_worker_cost_tree),
        )
        .route(
            "/api/analyzer/v1/runs/{run_id}/workers/{pool_tag}/{worker_id}/operations/{iter_id}/{batch_id}/{operation_id}/leaves/{leaf_id}/subjects/kernel-throughput-analysis/payload",
            get(get_kernel_throughput_analysis),
        )
        .with_state(state)
        // Compress every response, not one of them.
        //
        // These payloads are analysis JSON — long arrays of numbers with
        // repeated keys — which is close to the best case for a text codec, and
        // the readers that matter are browsers on the other side of a network.
        // The default predicate skips bodies under 32 bytes and anything
        // already typed as an image, so the measurement plots are not
        // re-compressed. SVG plots are excluded by that same rule and would
        // compress well; serving them under a non-image content type is the way
        // to change that, not loosening the predicate.
        .layer(CompressionLayer::new())
        // Inside the compression layer on the response path, so the validator is
        // computed over the bytes the analyzer produced rather than over one
        // particular encoding of them.
        .layer(middleware::from_fn(revalidate))
}

/// An address that does not name a revision: store it, but ask before reusing.
///
/// `no-cache` does not mean "do not store". Paired with the ETag below, a reload
/// of a page whose analysis has not changed costs one conditional request per
/// artifact and no payload bytes at all.
const REVALIDATE: &str = "no-cache";

/// An address that names the revision it got. Safe to keep for a year.
///
/// The promise is only sound because the revision IS the content hash: if the
/// analysis is re-run and the bytes change, the hash changes, the pinned URL no
/// longer matches, and this header is not sent. A stale `immutable` hit is
/// therefore not expressible.
const IMMUTABLE: &str = "max-age=31536000, immutable";

/// Query parameter a client uses to pin the revision it already has.
///
/// The client does not have to know a revision in advance. It reads once
/// without one, gets the ETag, and pins it — which is what
/// `ResultRef.revision` in `VibeSimUI/new-design.md` is for, and why that field
/// is optional with "absent means latest".
const REVISION_PARAM: &str = "rev";

/// The revision inside a validator: the hash with `W/` and the quotes removed.
///
/// A revision travels in a URL, so it is spelled as a bare token; an ETag is an
/// HTTP header field with its own syntax. They are the same value in two
/// notations, and this is the one place that converts between them.
fn revision_of(etag: &str) -> &str {
    etag.trim_start_matches("W/").trim_matches('"')
}

/// Read `rev=` out of a raw query string.
///
/// Hand-parsed rather than through an extractor because this runs for every
/// route, and the routes have their own, differently-shaped query types.
fn pinned_revision(query: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (key, value) = pair.split_once('=')?;
        (key == REVISION_PARAM).then(|| value.to_owned())
    })
}

/// Attach a validator to every response, and answer a matching request with 304.
///
/// The service serves files whose content is a pure function of an analysis
/// that has already finished. Re-sending an unchanged 540 KB payload because
/// the user pressed Back is waste that neither compression nor caching headers
/// alone remove: compression still sends the bytes, and a plain `max-age` would
/// have to guess how long the analysis stays put.
///
/// The tag is weak (`W/`). A strong validator has to differ per content-coding,
/// and this one is computed before compression precisely so it identifies the
/// analyzer's output rather than one negotiated encoding of it — which is what
/// "weak" means.
async fn revalidate(request: Request, next: Next) -> Response {
    let if_none_match = request
        .headers()
        .get(header::IF_NONE_MATCH)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let pinned_revision = request.uri().query().and_then(pinned_revision);
    let response = next.run(request).await;

    // Only successful, fully-buffered responses have a stable identity worth
    // validating. Errors and redirects are left exactly as the handler wrote
    // them.
    if response.status() != StatusCode::OK {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    let bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        // The body was already consumed or failed mid-stream; there is nothing
        // left to hash and nothing useful to say about it.
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    let etag = format!("W/\"{:x}\"", Sha256::digest(&bytes));
    if let Ok(value) = HeaderValue::from_str(&etag) {
        parts.headers.insert(header::ETAG, value);
    }
    let revision = revision_of(&etag);
    let pinned = pinned_revision.as_deref() == Some(revision);
    parts.headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static(if pinned { IMMUTABLE } else { REVALIDATE }),
    );

    // `If-None-Match` may list several tags, and `*` matches any existing
    // representation.
    let matched = if_none_match.is_some_and(|header| {
        header
            .split(',')
            .map(str::trim)
            .any(|candidate| candidate == "*" || candidate == etag)
    });
    if matched {
        parts.status = StatusCode::NOT_MODIFIED;
        // A 304 carries the validator and no body; Content-Length would
        // describe bytes that are not being sent.
        parts.headers.remove(header::CONTENT_LENGTH);
        return Response::from_parts(parts, Body::empty());
    }
    Response::from_parts(parts, Body::from(bytes))
}

#[derive(Deserialize)]
struct TimelineProfileEvent {
    session_id: String,
    event: String,
    #[serde(default)]
    elapsed_ms: Option<f64>,
    #[serde(default)]
    cursor_ms: Option<f64>,
    #[serde(default)]
    worker: Option<String>,
    #[serde(default)]
    detail: Option<String>,
}

async fn profile_timeline(Json(event): Json<TimelineProfileEvent>) -> Response {
    match timeline_profile_log_line(&event) {
        Ok(line) => {
            eprintln!("[timeline-prof-client] {line}");
            StatusCode::NO_CONTENT.into_response()
        }
        Err(detail) => (
            StatusCode::BAD_REQUEST,
            Json(json!({"code": "invalid_timeline_profile", "detail": detail})),
        )
            .into_response(),
    }
}

fn timeline_profile_log_line(
    event: &TimelineProfileEvent,
) -> std::result::Result<String, &'static str> {
    if event.session_id.is_empty() || event.session_id.len() > 128 {
        return Err("session_id must contain 1 to 128 bytes");
    }
    if event.event.is_empty() || event.event.len() > 128 {
        return Err("event must contain 1 to 128 bytes");
    }
    if event.worker.as_ref().is_some_and(|value| value.len() > 128)
        || event
            .detail
            .as_ref()
            .is_some_and(|value| value.len() > 1024)
    {
        return Err("worker or detail exceeds its diagnostic size limit");
    }
    if event.elapsed_ms.is_some_and(|value| !value.is_finite())
        || event.cursor_ms.is_some_and(|value| !value.is_finite())
    {
        return Err("timing values must be finite");
    }
    serde_json::to_string(&json!({
        "session_id": event.session_id,
        "event": event.event,
        "elapsed_ms": event.elapsed_ms,
        "cursor_ms": event.cursor_ms,
        "worker": event.worker,
        "detail": event.detail,
    }))
    .map_err(|_| "timeline profile event could not be encoded")
}

async fn list_alignments(State(state): State<ServiceState>) -> Response {
    let root_source = Arc::clone(&state.root_source);
    match tokio::task::spawn_blocking(move || {
        let roots = root_source.load()?;
        build_alignment_catalog(&roots)
    })
    .await
    {
        Ok(Ok(catalog)) => Json(catalog).into_response(),
        Ok(Err(error)) => alignment_resource_error(error),
        Err(error) => alignment_resource_error(anyhow::anyhow!(error)),
    }
}

async fn get_alignment_descriptor(
    RoutePath(alignment_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    // The descriptor names the timing prediction the bundle was paired against,
    // and that name is only servable relative to the configured roots.
    match state
        .resolve_alignment(&alignment_id)
        .and_then(|alignment| Ok((alignment, state.root_source.load()?)))
    {
        Ok((alignment, roots)) => Json(alignment_descriptor(&alignment, &roots)).into_response(),
        Err(error) => alignment_resource_error(error),
    }
}

async fn get_alignment_report(
    RoutePath((alignment_id, subject)): RoutePath<(String, String)>,
    State(state): State<ServiceState>,
) -> Response {
    match state
        .resolve_alignment(&alignment_id)
        .and_then(|alignment| alignment_artifact_bytes(&alignment, &subject, false))
    {
        Ok(bytes) => alignment_json_bytes_response(bytes),
        Err(error) => alignment_resource_error(error),
    }
}

async fn get_alignment_payload(
    RoutePath((alignment_id, subject)): RoutePath<(String, String)>,
    State(state): State<ServiceState>,
) -> Response {
    match state
        .resolve_alignment(&alignment_id)
        .and_then(|alignment| alignment_artifact_bytes(&alignment, &subject, true))
    {
        Ok(bytes) => alignment_json_bytes_response(bytes),
        Err(error) => alignment_resource_error(error),
    }
}

/// One iteration of one subject, straight out of the shard.
///
/// The bytes are already the JSON object the client wants, so they are returned
/// unparsed — the point of the byte-range layout is that no request touches more
/// than one iteration's worth of the file.
async fn get_alignment_iteration(
    RoutePath((alignment_id, subject, iteration_id)): RoutePath<(String, String, String)>,
    Query(query): Query<AlignmentIterationQuery>,
    State(state): State<ServiceState>,
) -> Response {
    let detail = state
        .resolve_alignment(&alignment_id)
        .and_then(|alignment| {
            let index = state.alignment_detail_index(&alignment_id, &alignment, &subject)?;
            let bytes = read_alignment_iteration_detail(&index, &iteration_id)?;
            if query.projection.as_deref() == Some("reference-lane") && subject == "timeline" {
                reference_lane_projection(&index, bytes)
            } else if query.projection.as_deref() == Some("operation-split")
                && subject == "iteration"
            {
                operation_split_projection(bytes)
            } else {
                Ok(bytes)
            }
        });
    match detail {
        Ok(bytes) => alignment_json_bytes_response(bytes),
        Err(error) => alignment_resource_error(error),
    }
}

/// One folded multi-stream program selected by the mapping board.
async fn get_alignment_sequence(
    RoutePath((alignment_id, phase, sequence_id)): RoutePath<(String, String, String)>,
    State(state): State<ServiceState>,
) -> Response {
    let detail = state
        .resolve_alignment(&alignment_id)
        .and_then(|alignment| {
            let index = state.alignment_sequence_index(&alignment_id, &alignment)?;
            read_alignment_sequence(&index, &phase, &sequence_id)
        });
    match detail {
        Ok(bytes) => alignment_json_bytes_response(bytes),
        Err(error) => alignment_resource_error(error),
    }
}

#[derive(Debug, Deserialize)]
struct AlignmentIterationQuery {
    projection: Option<String>,
}

/// Alignment artifacts are already the JSON the client wants, so they go out as
/// bytes rather than through `Json` — parsing and re-serializing a payload this
/// size to change nothing about it is pure cost.
///
/// This function used to gzip the body itself, sniffing `Accept-Encoding` and
/// building the header set by hand. It was the only compressed response in the
/// service: the other 30 went out uncompressed because compression lived in one
/// handler instead of in front of all of them. `CompressionLayer` in
/// `service_router` now covers every route, so this is back to being what it
/// reads as.
fn alignment_json_bytes_response(bytes: Vec<u8>) -> Response {
    ([(header::CONTENT_TYPE, "application/json")], bytes).into_response()
}

async fn list_predictions(State(state): State<ServiceState>) -> Response {
    let root_source = Arc::clone(&state.root_source);
    match tokio::task::spawn_blocking(move || {
        let roots = root_source.load()?;
        build_prediction_catalog(&roots)
    })
    .await
    {
        Ok(Ok(catalog)) => Json(catalog).into_response(),
        Ok(Err(error)) => prediction_resource_error(error),
        Err(error) => prediction_resource_error(anyhow::anyhow!(error)),
    }
}

async fn get_prediction_descriptor(
    RoutePath(prediction_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    match state.resolve_prediction(&prediction_id) {
        Ok(prediction) => Json(prediction_descriptor(&prediction)).into_response(),
        Err(error) => prediction_resource_error(error),
    }
}

async fn get_prediction_cases(
    RoutePath(prediction_id): RoutePath<String>,
    Query(query): Query<OperationRangeQuery>,
    State(state): State<ServiceState>,
) -> Response {
    let prediction = match state.resolve_prediction(&prediction_id) {
        Ok(prediction) => prediction,
        Err(error) => return prediction_resource_error(error),
    };
    let request_id = NEXT_WORKER_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    match prediction_cases(
        &prediction,
        &state.operation_indexes,
        query.offset,
        query.limit,
        request_id,
    )
    .await
    {
        Ok(value) => Json(value).into_response(),
        Err(error) => prediction_resource_error(error),
    }
}

async fn get_prediction_kernel_input_distribution(
    RoutePath(prediction_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    let prediction = match state.resolve_prediction(&prediction_id) {
        Ok(prediction) => prediction,
        Err(error) => return prediction_resource_error(error),
    };
    match read_json(
        &prediction
            .path
            .join("payloads/kernel_input_distribution_scatter.json"),
    ) {
        Ok(value) => Json(value).into_response(),
        Err(error) => prediction_resource_error(error),
    }
}

async fn get_prediction_cost_tree(
    RoutePath((prediction_id, case_id, operation_id)): RoutePath<(String, u64, usize)>,
    State(state): State<ServiceState>,
) -> Response {
    let prediction = match state.resolve_prediction(&prediction_id) {
        Ok(prediction) => prediction,
        Err(error) => return prediction_resource_error(error),
    };
    let source = match prediction.cost_source() {
        Ok(source) => source,
        Err(error) => return prediction_resource_error(error),
    };
    let request_id = NEXT_WORKER_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    match prediction_operation_cost_tree(
        &state.operation_indexes,
        &source,
        prediction.prediction_id(),
        case_id,
        operation_id,
        request_id,
    )
    .await
    {
        Ok(value) => Json(value).into_response(),
        Err(error) => prediction_resource_error(error),
    }
}

async fn get_prediction_kernel_throughput_analysis(
    RoutePath((prediction_id, case_id, operation_id, leaf_id)): RoutePath<(
        String,
        u64,
        usize,
        usize,
    )>,
    State(state): State<ServiceState>,
) -> Response {
    let prediction = match state.resolve_prediction(&prediction_id) {
        Ok(prediction) => prediction,
        Err(error) => return prediction_resource_error(error),
    };
    let source = match prediction.cost_source() {
        Ok(source) => source,
        Err(error) => return prediction_resource_error(error),
    };
    let request_id = NEXT_WORKER_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    let detail = match prediction_operation_cost_tree(
        &state.operation_indexes,
        &source,
        prediction.prediction_id(),
        case_id,
        operation_id,
        request_id,
    )
    .await
    {
        Ok(value) => value,
        Err(error) => return prediction_resource_error(error),
    };
    let repo_root = Arc::clone(&state.repo_root);
    match tokio::task::spawn_blocking(move || {
        analyze_kernel_throughput(&repo_root, detail, leaf_id)
    })
    .await
    {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(error)) => prediction_resource_error(error),
        Err(error) => prediction_resource_error(anyhow::anyhow!(error)),
    }
}

async fn get_prediction_optimality_kernel_ladder(
    RoutePath((prediction_id, case_id)): RoutePath<(String, u64)>,
    Query(query): Query<OptimalityModeQuery>,
    State(state): State<ServiceState>,
) -> Response {
    let prediction = match state.resolve_prediction(&prediction_id) {
        Ok(prediction) => prediction,
        Err(error) => return prediction_resource_error(error),
    };
    let source = match prediction.cost_source() {
        Ok(source) => source,
        Err(error) => return prediction_resource_error(error),
    };
    match prediction_kernel_ladder(
        &prediction.path,
        source.pool_tag(),
        source.worker_id(),
        prediction.prediction_id(),
        case_id,
        prediction.gpu_name(),
        prediction.gpu_count(),
        matches!(query.mode, Some(OptimalityMode::BatchLocked)),
    )
    .await
    {
        Ok(value) => Json(value).into_response(),
        Err(error) => prediction_resource_error(error),
    }
}

async fn get_prediction_optimality_waterfall(
    RoutePath((prediction_id, case_id)): RoutePath<(String, u64)>,
    Query(query): Query<OptimalityModeQuery>,
    State(state): State<ServiceState>,
) -> Response {
    let prediction = match state.resolve_prediction(&prediction_id) {
        Ok(prediction) => prediction,
        Err(error) => return prediction_resource_error(error),
    };
    let source = match prediction.cost_source() {
        Ok(source) => source,
        Err(error) => return prediction_resource_error(error),
    };
    match prediction_waterfall(
        &prediction.path,
        source.pool_tag(),
        source.worker_id(),
        prediction.prediction_id(),
        case_id,
        prediction.gpu_name(),
        prediction.gpu_count(),
        matches!(query.mode, Some(OptimalityMode::BatchLocked)),
    )
    .await
    {
        Ok(value) => Json(value).into_response(),
        Err(error) => prediction_resource_error(error),
    }
}

async fn list_kernel_profiles(State(state): State<ServiceState>) -> Response {
    let root_source = Arc::clone(&state.root_source);
    match tokio::task::spawn_blocking(move || {
        let roots = root_source.load()?;
        build_kernel_profile_catalog(&roots)
    })
    .await
    {
        Ok(Ok(catalog)) => Json(catalog).into_response(),
        Ok(Err(error)) => kernel_profile_resource_error(error),
        Err(error) => kernel_profile_resource_error(anyhow::anyhow!(error)),
    }
}

async fn get_kernel_profile_descriptor(
    RoutePath(profile_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    match state.resolve_kernel_profile(&profile_id) {
        Ok(profile) => Json(profile_descriptor(&profile)).into_response(),
        Err(error) => kernel_profile_resource_error(error),
    }
}

async fn get_kernel_profile_curve(
    RoutePath(profile_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    let repo_root = Arc::clone(&state.repo_root);
    match state.resolve_kernel_profile(&profile_id) {
        Ok(profile) => match profile_curve(&profile, &repo_root) {
            Ok(value) => Json(value).into_response(),
            Err(error) => kernel_profile_resource_error(error),
        },
        Err(error) => kernel_profile_resource_error(error),
    }
}

async fn list_kernel_measurements(State(state): State<ServiceState>) -> Response {
    let root_source = Arc::clone(&state.root_source);
    match tokio::task::spawn_blocking(move || {
        let roots = root_source.load()?;
        build_kernel_measurement_catalog(&roots)
    })
    .await
    {
        Ok(Ok(catalog)) => Json(catalog).into_response(),
        Ok(Err(error)) => kernel_measurement_resource_error(error),
        Err(error) => kernel_measurement_resource_error(anyhow::anyhow!(error)),
    }
}

async fn get_kernel_measurement_descriptor(
    RoutePath(measurement_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    match state.resolve_kernel_measurement(&measurement_id) {
        Ok(measurement) => Json(measurement_descriptor(&measurement)).into_response(),
        Err(error) => kernel_measurement_resource_error(error),
    }
}

async fn get_kernel_measurement_summary(
    RoutePath(measurement_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    match state.resolve_kernel_measurement(&measurement_id) {
        Ok(measurement) => match measurement_summary(&measurement) {
            Ok(value) => Json(value).into_response(),
            Err(error) => kernel_measurement_resource_error(error),
        },
        Err(error) => kernel_measurement_resource_error(error),
    }
}

async fn get_kernel_measurement_plot(
    RoutePath((measurement_id, plot_name)): RoutePath<(String, String)>,
    State(state): State<ServiceState>,
) -> Response {
    let measurement = match state.resolve_kernel_measurement(&measurement_id) {
        Ok(measurement) => measurement,
        Err(error) => return kernel_measurement_resource_error(error),
    };
    match tokio::task::spawn_blocking(move || {
        let bytes = measurement_plot(&measurement, &plot_name)
            .map_err(|error| (plot_name.clone(), error))?;
        Ok::<_, (String, anyhow::Error)>((plot_name, bytes))
    })
    .await
    {
        Ok(Ok((name, bytes))) => {
            let content_type = plot_content_type(&name);
            let content_type = axum::http::HeaderValue::from_static(content_type);
            ([(axum::http::header::CONTENT_TYPE, content_type)], bytes).into_response()
        }
        Ok(Err((_, error))) => kernel_measurement_resource_error(error),
        Err(error) => kernel_measurement_resource_error(anyhow::anyhow!(error)),
    }
}

fn plot_content_type(plot_name: &str) -> &'static str {
    if plot_name.to_ascii_lowercase().ends_with(".jpg")
        || plot_name.to_ascii_lowercase().ends_with(".jpeg")
    {
        "image/jpeg"
    } else {
        "image/png"
    }
}

async fn get_hardware_gpus(
    Query(query): Query<GpuNameQuery>,
    State(state): State<ServiceState>,
) -> Response {
    let requested = match query.name {
        Some(name) if !name.trim().is_empty() => name.trim().to_owned(),
        _ => {
            return problem(
                StatusCode::BAD_REQUEST,
                "gpu_name_required",
                "hardware/gpus requires a non-empty ?name=<gpu_name>",
            )
        }
    };
    let repo_root = Arc::clone(&state.repo_root);
    match tokio::task::spawn_blocking(move || {
        let resolved = resolve_gpu(&repo_root, &requested)?;
        Ok::<_, anyhow::Error>(hardware_gpu_response(&requested, resolved.as_ref()))
    })
    .await
    {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(error)) => {
            eprintln!("[analyze] hardware resolution failed: {error:#}");
            problem(
                StatusCode::INTERNAL_SERVER_ERROR,
                "hardware_resolution_failed",
                "The GPU catalog could not be read.",
            )
        }
        Err(error) => {
            eprintln!("[analyze] hardware resolution worker failed: {error}");
            problem(
                StatusCode::INTERNAL_SERVER_ERROR,
                "hardware_resolution_failed",
                "The GPU catalog could not be read.",
            )
        }
    }
}

#[derive(Deserialize)]
struct GpuNameQuery {
    name: Option<String>,
}

async fn list_runs(State(state): State<ServiceState>) -> Response {
    let root_source = Arc::clone(&state.root_source);
    match tokio::task::spawn_blocking(move || {
        let roots = root_source.load()?;
        build_catalog(&roots)
    })
    .await
    {
        Ok(Ok(catalog)) => Json(catalog).into_response(),
        Ok(Err(error)) => catalog_error(error),
        Err(error) => catalog_error(anyhow::anyhow!("catalog worker failed: {error}")),
    }
}

const MAX_SWEEP_CATALOG_LIMIT: usize = 100;

#[derive(Debug, Default, Deserialize)]
struct SweepCatalogQuery {
    status: Option<SweepStatus>,
    limit: Option<usize>,
}

impl SweepCatalogQuery {
    fn filter(self) -> std::result::Result<SweepCatalogFilter, Response> {
        if self
            .limit
            .is_some_and(|limit| limit == 0 || limit > MAX_SWEEP_CATALOG_LIMIT)
        {
            return Err(problem(
                StatusCode::BAD_REQUEST,
                "invalid_sweep_catalog_limit",
                "Sweep catalog limit must be between 1 and 100.",
            ));
        }
        Ok(SweepCatalogFilter {
            status: self.status,
            limit: self.limit,
        })
    }
}

async fn list_sweeps(
    Query(query): Query<SweepCatalogQuery>,
    State(state): State<ServiceState>,
) -> Response {
    let filter = match query.filter() {
        Ok(filter) => filter,
        Err(response) => return response,
    };
    sweep_catalog_response(state, filter).await
}

async fn get_latest_sweep(State(state): State<ServiceState>) -> Response {
    sweep_catalog_response(
        state,
        SweepCatalogFilter {
            status: Some(SweepStatus::Ready),
            limit: Some(1),
        },
    )
    .await
}

async fn sweep_catalog_response(state: ServiceState, filter: SweepCatalogFilter) -> Response {
    let root_source = Arc::clone(&state.root_source);
    match tokio::task::spawn_blocking(move || {
        let roots = root_source.load()?;
        build_filtered_sweep_catalog(&roots, filter)
    })
    .await
    {
        Ok(Ok(catalog)) => Json(catalog).into_response(),
        Ok(Err(error)) => sweep_catalog_error(error),
        Err(error) => sweep_catalog_error(anyhow::anyhow!("sweep catalog worker failed: {error}")),
    }
}

async fn get_sweep_payload(
    RoutePath(sweep_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    let root_source = Arc::clone(&state.root_source);
    match tokio::task::spawn_blocking(move || {
        let roots = root_source.load()?;
        let sweep = resolve_sweep(&roots, &sweep_id)?;
        read_sweep_payload(&roots, &sweep)
    })
    .await
    {
        Ok(Ok(payload)) => Json(payload).into_response(),
        Ok(Err(error)) if error.downcast_ref::<SweepNotFound>().is_some() => problem(
            StatusCode::NOT_FOUND,
            "sweep_not_found",
            "The requested sweep is not present below the configured logs roots.",
        ),
        Ok(Err(error)) if error.downcast_ref::<ArtifactNotFound>().is_some() => problem(
            StatusCode::NOT_FOUND,
            "artifact_missing",
            "The requested sweep payload has not been generated.",
        ),
        Ok(Err(error)) => {
            eprintln!("[analyze] sweep resource failed: {error:#}");
            problem(
                StatusCode::INTERNAL_SERVER_ERROR,
                "artifact_read_failed",
                "The requested sweep resource could not be read.",
            )
        }
        Err(error) => {
            eprintln!("[analyze] sweep resource worker failed: {error}");
            problem(
                StatusCode::INTERNAL_SERVER_ERROR,
                "artifact_read_failed",
                "The requested sweep resource could not be read.",
            )
        }
    }
}

async fn get_descriptor(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| build_descriptor(&run)).await
}

async fn get_summary(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_summary(&run)).await
}

async fn get_topology(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| build_topology(&run)).await
}

async fn get_model(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    let repo_root = Arc::clone(&state.repo_root);
    read_run_resource(state, run_id, move |run| read_model(&run, &repo_root)).await
}

async fn get_workload(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| {
        let logs_root = run.logs_root.clone();
        read_workload(&run, &logs_root)
    })
    .await
}

async fn get_concurrency_report(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_concurrency_report(&run)).await
}

async fn get_concurrency_payload(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_concurrency_payload(&run)).await
}

async fn get_request_state_report(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_request_state_report(&run)).await
}

async fn get_request_state_payload(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_request_state_payload(&run)).await
}

async fn get_slo_general_report(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_slo_general_report(&run)).await
}

async fn get_slo_general_payload(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_slo_general_payload(&run)).await
}

async fn get_throughput_report(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_throughput_report(&run)).await
}

async fn get_throughput_payload(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_throughput_payload(&run)).await
}

async fn get_utilization_report(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_utilization_report(&run)).await
}

async fn get_utilization_payload(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_utilization_payload(&run)).await
}

async fn get_batch_report(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_batch_report(&run)).await
}

async fn get_batch_payload(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_batch_payload(&run)).await
}

async fn get_kv_occupancy_report(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_kv_occupancy_report(&run)).await
}

async fn get_kv_occupancy_payload(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_kv_occupancy_payload(&run)).await
}

async fn get_kernel_time_share_report(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_kernel_time_share_report(&run)).await
}

async fn get_kernel_input_distribution_report(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| {
        read_kernel_input_distribution_report(&run)
    })
    .await
}

async fn get_kernel_input_distribution_payload(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| {
        read_kernel_input_distribution_payload(&run)
    })
    .await
}

async fn get_kernel_time_share_payload(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_kernel_time_share_payload(&run)).await
}

async fn get_worker_kernel_time_share_payload(
    RoutePath((run_id, pool_tag, worker_id)): RoutePath<(String, String, u16)>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, move |run| {
        read_kernel_time_share_worker_payload(&run, &pool_tag, worker_id)
    })
    .await
}

async fn get_optimality_report(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_optimality_report(&run)).await
}

async fn get_optimality_payload(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_optimality_payload(&run)).await
}

async fn get_locked_optimality_report(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_locked_optimality_report(&run)).await
}

async fn get_locked_optimality_payload(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_locked_optimality_payload(&run)).await
}

async fn get_workload_conservation_report(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_workload_conservation_report(&run)).await
}

async fn get_workload_conservation_payload(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| {
        read_workload_conservation_payload(&run)
    })
    .await
}

#[derive(Deserialize)]
struct OperationRangeQuery {
    #[serde(default)]
    offset: usize,
    #[serde(default = "default_operation_limit")]
    limit: usize,
}

#[derive(Deserialize)]
struct OperationSeekQuery {
    at_ms: f64,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OptimalityMode {
    Unlocked,
    BatchLocked,
}

#[derive(Deserialize)]
struct OptimalityModeQuery {
    #[serde(default)]
    mode: Option<OptimalityMode>,
}

fn default_operation_limit() -> usize {
    128
}

async fn get_worker_operations(
    RoutePath((run_id, pool_tag, worker_id)): RoutePath<(String, String, u16)>,
    Query(query): Query<OperationRangeQuery>,
    State(state): State<ServiceState>,
) -> Response {
    let request_id = NEXT_WORKER_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    let request_started = Instant::now();
    eprintln!(
        "[worker-prof] request_id={request_id} endpoint=operations event=enter run_id={run_id} worker={pool_tag}/{worker_id} offset={} limit={}",
        query.offset, query.limit
    );
    let run = match state.resolve_run(&run_id) {
        Ok(run) => run,
        Err(error) => {
            eprintln!(
                "[worker-prof] request_id={request_id} endpoint=operations event=exit status=error elapsed_ms={:.3}",
                request_started.elapsed().as_secs_f64() * 1000.0
            );
            return worker_resource_error(error);
        }
    };
    let response = match operation_range(
        &state.operation_indexes,
        &run,
        &pool_tag,
        worker_id,
        OperationRange::bounded(query.offset, query.limit),
        request_id,
    )
    .await
    {
        Ok(value) => Json(value).into_response(),
        Err(error) => worker_resource_error(error),
    };
    eprintln!(
        "[worker-prof] request_id={request_id} endpoint=operations event=exit status={} elapsed_ms={:.3}",
        response.status(),
        request_started.elapsed().as_secs_f64() * 1000.0
    );
    response
}

async fn seek_worker_operation(
    RoutePath((run_id, pool_tag, worker_id)): RoutePath<(String, String, u16)>,
    Query(query): Query<OperationSeekQuery>,
    State(state): State<ServiceState>,
) -> Response {
    let request_id = NEXT_WORKER_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    let request_started = Instant::now();
    eprintln!(
        "[worker-prof] request_id={request_id} endpoint=operation_seek event=enter run_id={run_id} worker={pool_tag}/{worker_id} at_ms={}",
        query.at_ms
    );
    let run = match state.resolve_run(&run_id) {
        Ok(run) => run,
        Err(error) => {
            eprintln!(
                "[worker-prof] request_id={request_id} endpoint=operation_seek event=exit status=error elapsed_ms={:.3}",
                request_started.elapsed().as_secs_f64() * 1000.0
            );
            return worker_resource_error(error);
        }
    };
    let response = match operation_seek(
        &state.operation_indexes,
        &run,
        &pool_tag,
        worker_id,
        query.at_ms,
        request_id,
    )
    .await
    {
        Ok(value) => Json(value).into_response(),
        Err(error) => worker_resource_error(error),
    };
    eprintln!(
        "[worker-prof] request_id={request_id} endpoint=operation_seek event=exit status={} elapsed_ms={:.3}",
        response.status(),
        request_started.elapsed().as_secs_f64() * 1000.0
    );
    response
}

async fn get_iteration_optimality_kernel_ladder(
    RoutePath((run_id, pool_tag, worker_id, iter_id)): RoutePath<(String, String, u16, u64)>,
    Query(query): Query<OptimalityModeQuery>,
    State(state): State<ServiceState>,
) -> Response {
    let run = match state.resolve_run(&run_id) {
        Ok(run) => run,
        Err(error) => return worker_resource_error(error),
    };
    match iteration_kernel_ladder(
        &run.path,
        &pool_tag,
        worker_id,
        iter_id,
        matches!(query.mode, Some(OptimalityMode::BatchLocked)),
    )
    .await
    {
        Ok(value) => Json(value).into_response(),
        Err(error) => worker_resource_error(error),
    }
}

async fn get_iteration_optimality_waterfall(
    RoutePath((run_id, pool_tag, worker_id, iter_id)): RoutePath<(String, String, u16, u64)>,
    Query(query): Query<OptimalityModeQuery>,
    State(state): State<ServiceState>,
) -> Response {
    let run = match state.resolve_run(&run_id) {
        Ok(run) => run,
        Err(error) => return worker_resource_error(error),
    };
    match iteration_waterfall(
        &run.path,
        &pool_tag,
        worker_id,
        iter_id,
        matches!(query.mode, Some(OptimalityMode::BatchLocked)),
    )
    .await
    {
        Ok(value) => Json(value).into_response(),
        Err(error) => worker_resource_error(error),
    }
}

async fn get_worker_cost_tree(
    RoutePath((run_id, pool_tag, worker_id, iter_id, batch_id, operation_id)): RoutePath<(
        String,
        String,
        u16,
        u64,
        u64,
        u64,
    )>,
    State(state): State<ServiceState>,
) -> Response {
    let request_id = NEXT_WORKER_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    let request_started = Instant::now();
    eprintln!(
        "[worker-prof] request_id={request_id} endpoint=cost_tree event=enter run_id={run_id} worker={pool_tag}/{worker_id} iter_id={iter_id} batch_id={batch_id} operation_id={operation_id}"
    );
    let run = match state.resolve_run(&run_id) {
        Ok(run) => run,
        Err(error) => {
            eprintln!(
                "[worker-prof] request_id={request_id} endpoint=cost_tree event=exit status=error elapsed_ms={:.3}",
                request_started.elapsed().as_secs_f64() * 1000.0
            );
            return worker_resource_error(error);
        }
    };
    let response = match exact_operation_cost_tree(
        &run,
        &pool_tag,
        worker_id,
        iter_id,
        batch_id,
        operation_id,
        request_id,
    )
    .await
    {
        Ok(value) => Json(value).into_response(),
        Err(error) => worker_resource_error(error),
    };
    eprintln!(
        "[worker-prof] request_id={request_id} endpoint=cost_tree event=exit status={} elapsed_ms={:.3}",
        response.status(),
        request_started.elapsed().as_secs_f64() * 1000.0
    );
    response
}

async fn get_kernel_throughput_analysis(
    RoutePath((run_id, pool_tag, worker_id, iter_id, batch_id, operation_id, leaf_id)): RoutePath<
        (String, String, u16, u64, u64, u64, usize),
    >,
    State(state): State<ServiceState>,
) -> Response {
    let request_id = NEXT_WORKER_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    let run = match state.resolve_run(&run_id) {
        Ok(run) => run,
        Err(error) => return worker_resource_error(error),
    };
    let detail = match exact_operation_cost_tree(
        &run,
        &pool_tag,
        worker_id,
        iter_id,
        batch_id,
        operation_id,
        request_id,
    )
    .await
    {
        Ok(value) => value,
        Err(error) => return worker_resource_error(error),
    };
    let repo_root = Arc::clone(&state.repo_root);
    match tokio::task::spawn_blocking(move || {
        analyze_kernel_throughput(&repo_root, detail, leaf_id)
    })
    .await
    {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(error)) => worker_resource_error(error),
        Err(error) => worker_resource_error(anyhow::anyhow!(error)),
    }
}

fn worker_resource_error(error: anyhow::Error) -> Response {
    if error.downcast_ref::<RunNotFound>().is_some() {
        return problem(
            StatusCode::NOT_FOUND,
            "run_not_found",
            "The requested run is not present below the configured logs roots.",
        );
    }
    eprintln!("[analyze] worker detail failed: {error:#}");
    problem(
        StatusCode::INTERNAL_SERVER_ERROR,
        "worker_detail_failed",
        "The requested worker detail could not be reconstructed.",
    )
}

async fn read_run_resource(
    state: ServiceState,
    run_id: String,
    read: impl FnOnce(DiscoveredRun) -> Result<Value> + Send + 'static,
) -> Response {
    let root_source = Arc::clone(&state.root_source);
    match tokio::task::spawn_blocking(move || {
        let roots = root_source.load()?;
        let run = resolve_run(&roots, &run_id)?;
        read(run)
    })
    .await
    {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(error)) if error.downcast_ref::<RunNotFound>().is_some() => problem(
            StatusCode::NOT_FOUND,
            "run_not_found",
            "The requested run is not present below the configured logs roots.",
        ),
        Ok(Err(error)) if error.downcast_ref::<ArtifactNotFound>().is_some() => problem(
            StatusCode::NOT_FOUND,
            "artifact_missing",
            "The requested run artifact has not been generated.",
        ),
        Ok(Err(error)) => {
            eprintln!("[analyze] run resource failed: {error:#}");
            problem(
                StatusCode::INTERNAL_SERVER_ERROR,
                "artifact_read_failed",
                "The requested run resource could not be read.",
            )
        }
        Err(error) => {
            eprintln!("[analyze] run resource worker failed: {error}");
            problem(
                StatusCode::INTERNAL_SERVER_ERROR,
                "artifact_read_failed",
                "The requested run resource could not be read.",
            )
        }
    }
}

#[derive(Debug)]
struct RunNotFound;

impl std::fmt::Display for RunNotFound {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("run not found")
    }
}

impl std::error::Error for RunNotFound {}

#[derive(Debug)]
struct PredictionNotFound;

impl std::fmt::Display for PredictionNotFound {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("prediction not found")
    }
}

impl std::error::Error for PredictionNotFound {}

#[derive(Debug)]
struct KernelProfileNotFound;

impl std::fmt::Display for KernelProfileNotFound {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("kernel profile not found")
    }
}

impl std::error::Error for KernelProfileNotFound {}

#[derive(Debug)]
struct KernelMeasurementNotFound;

impl std::fmt::Display for KernelMeasurementNotFound {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("kernel measurement not found")
    }
}

impl std::error::Error for KernelMeasurementNotFound {}

#[derive(Debug)]
struct SweepNotFound;

impl std::fmt::Display for SweepNotFound {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("sweep not found")
    }
}

impl std::error::Error for SweepNotFound {}

#[derive(Debug)]
struct AlignmentNotFound;

impl std::fmt::Display for AlignmentNotFound {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("alignment not found")
    }
}

impl std::error::Error for AlignmentNotFound {}

#[derive(Debug)]
struct ArtifactNotFound;

impl std::fmt::Display for ArtifactNotFound {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("artifact not found")
    }
}

impl std::error::Error for ArtifactNotFound {}

fn catalog_error(error: anyhow::Error) -> Response {
    eprintln!("[analyze] run discovery failed: {error:#}");
    problem(
        StatusCode::INTERNAL_SERVER_ERROR,
        "run_discovery_failed",
        "The configured logs roots could not be scanned.",
    )
}

fn kernel_profile_resource_error(error: anyhow::Error) -> Response {
    if error.downcast_ref::<KernelProfileNotFound>().is_some() {
        return problem(
            StatusCode::NOT_FOUND,
            "kernel_profile_not_found",
            "The requested kernel profile is not present below the configured logs roots.",
        );
    }
    if error.downcast_ref::<ArtifactNotFound>().is_some() {
        return problem(
            StatusCode::NOT_FOUND,
            "artifact_missing",
            "The requested kernel-profile artifact has not been generated.",
        );
    }
    eprintln!("[analyze] kernel-profile resource failed: {error:#}");
    problem(
        StatusCode::INTERNAL_SERVER_ERROR,
        "kernel_profile_resource_failed",
        "The requested kernel-profile resource could not be reconstructed.",
    )
}

fn kernel_measurement_resource_error(error: anyhow::Error) -> Response {
    if error.downcast_ref::<KernelMeasurementNotFound>().is_some() {
        return problem(
            StatusCode::NOT_FOUND,
            "kernel_measurement_not_found",
            "The requested kernel measurement is not present below the configured logs roots.",
        );
    }
    if error.downcast_ref::<ArtifactNotFound>().is_some() {
        return problem(
            StatusCode::NOT_FOUND,
            "artifact_missing",
            "The requested kernel-measurement artifact has not been generated.",
        );
    }
    eprintln!("[analyze] kernel-measurement resource failed: {error:#}");
    problem(
        StatusCode::INTERNAL_SERVER_ERROR,
        "kernel_measurement_resource_failed",
        "The requested kernel-measurement resource could not be reconstructed.",
    )
}

fn alignment_resource_error(error: anyhow::Error) -> Response {
    if error.downcast_ref::<AlignmentNotFound>().is_some() {
        return problem(
            StatusCode::NOT_FOUND,
            "alignment_not_found",
            "The requested alignment bundle is not present below the configured logs roots.",
        );
    }
    if error.downcast_ref::<ArtifactNotFound>().is_some() {
        // Also the answer for an unknown subject name and for an iteration the
        // payload index does not carry: from outside, all three are "that is not
        // a thing here", and distinguishing them would describe the layout.
        return problem(
            StatusCode::NOT_FOUND,
            "artifact_missing",
            "The requested alignment artifact has not been generated.",
        );
    }
    eprintln!("[analyze] alignment resource failed: {error:#}");
    problem(
        StatusCode::INTERNAL_SERVER_ERROR,
        "alignment_resource_failed",
        "The requested alignment resource could not be reconstructed.",
    )
}

fn prediction_resource_error(error: anyhow::Error) -> Response {
    if error.downcast_ref::<PredictionNotFound>().is_some() {
        return problem(
            StatusCode::NOT_FOUND,
            "prediction_not_found",
            "The requested timing prediction is not present below the configured logs roots.",
        );
    }
    if error.downcast_ref::<ArtifactNotFound>().is_some() {
        return problem(
            StatusCode::NOT_FOUND,
            "artifact_missing",
            "The requested timing-prediction artifact has not been generated.",
        );
    }
    eprintln!("[analyze] timing-prediction resource failed: {error:#}");
    problem(
        StatusCode::INTERNAL_SERVER_ERROR,
        "prediction_resource_failed",
        "The requested timing-prediction resource could not be reconstructed.",
    )
}

fn sweep_catalog_error(error: anyhow::Error) -> Response {
    eprintln!("[analyze] sweep discovery failed: {error:#}");
    problem(
        StatusCode::INTERNAL_SERVER_ERROR,
        "sweep_discovery_failed",
        "The configured logs roots could not be scanned for sweeps.",
    )
}

fn problem(status: StatusCode, code: &'static str, detail: &'static str) -> Response {
    (status, Json(json!({ "code": code, "detail": detail }))).into_response()
}
