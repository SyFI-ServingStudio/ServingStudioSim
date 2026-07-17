//! Minimal read-only HTTP wiring between analyzer run directories and viz-ui.
//!
//! Transport stays here. Filesystem discovery, catalog projection, and core run
//! resources live in separate modules so later subjects do not grow one service
//! file into a second analyzer.

mod artifact;
mod catalog;
mod concurrency;
mod core;
mod discovery;
mod kernel_time_share;
mod kv_occupancy;
mod model;
mod slo;
#[cfg(test)]
mod tests;
mod throughput;
mod topology;
mod utilization;
mod worker_detail;
mod workload;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, Path as RoutePath, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use catalog::build_catalog;
use concurrency::{read_concurrency_payload, read_concurrency_report};
use core::{build_descriptor, read_summary};
use discovery::{configure_logs_roots, resolve_run, ConfiguredRoot, DiscoveredRun};
use kernel_time_share::{read_kernel_time_share_payload, read_kernel_time_share_report};
use kv_occupancy::{read_kv_occupancy_payload, read_kv_occupancy_report};
use model::read_model;
use slo::{read_slo_general_payload, read_slo_general_report};
use throughput::{read_throughput_payload, read_throughput_report};
use topology::build_topology;
use utilization::{read_utilization_payload, read_utilization_report};
use worker_detail::{
    exact_operation_cost_tree, operation_range, operation_seek, OperationIndexCache, OperationRange,
};
use workload::read_workload;

const PROTOCOL_VERSION: u32 = 1;
const TIMELINE_PROFILE_BODY_LIMIT: usize = 16 * 1024;
static NEXT_WORKER_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
struct ServiceState {
    roots: Arc<Vec<ConfiguredRoot>>,
    repo_root: Arc<PathBuf>,
    operation_indexes: Arc<OperationIndexCache>,
}

pub(crate) async fn serve(bind: SocketAddr, logs_roots: Vec<PathBuf>) -> Result<()> {
    let roots = configure_logs_roots(logs_roots)?;
    let repo_root = std::env::current_dir().context("resolve analyzer repository root")?;
    let state = ServiceState {
        roots: Arc::new(roots),
        repo_root: Arc::new(repo_root),
        operation_indexes: Arc::new(OperationIndexCache::default()),
    };
    let app = Router::new()
        .route("/api/v1/profile/timeline", post(profile_timeline))
        .route("/api/v1/runs", get(list_runs))
        .route("/api/v1/runs/{run_id}/descriptor", get(get_descriptor))
        .route("/api/v1/runs/{run_id}/summary", get(get_summary))
        .route("/api/v1/runs/{run_id}/topology", get(get_topology))
        .route("/api/v1/runs/{run_id}/model", get(get_model))
        .route("/api/v1/runs/{run_id}/workload", get(get_workload))
        .route(
            "/api/v1/runs/{run_id}/subjects/concurrency/report",
            get(get_concurrency_report),
        )
        .route(
            "/api/v1/runs/{run_id}/subjects/concurrency/payload",
            get(get_concurrency_payload),
        )
        .route(
            "/api/v1/runs/{run_id}/subjects/slo-general/report",
            get(get_slo_general_report),
        )
        .route(
            "/api/v1/runs/{run_id}/subjects/slo-general/payload",
            get(get_slo_general_payload),
        )
        .route(
            "/api/v1/runs/{run_id}/subjects/throughput/report",
            get(get_throughput_report),
        )
        .route(
            "/api/v1/runs/{run_id}/subjects/throughput/payload",
            get(get_throughput_payload),
        )
        .route(
            "/api/v1/runs/{run_id}/subjects/utilization/report",
            get(get_utilization_report),
        )
        .route(
            "/api/v1/runs/{run_id}/subjects/utilization/payload",
            get(get_utilization_payload),
        )
        .route(
            "/api/v1/runs/{run_id}/subjects/kv-occupancy/report",
            get(get_kv_occupancy_report),
        )
        .route(
            "/api/v1/runs/{run_id}/subjects/kv-occupancy/payload",
            get(get_kv_occupancy_payload),
        )
        .route(
            "/api/v1/runs/{run_id}/subjects/kernel-time-share/report",
            get(get_kernel_time_share_report),
        )
        .route(
            "/api/v1/runs/{run_id}/subjects/kernel-time-share/payload",
            get(get_kernel_time_share_payload),
        )
        .route(
            "/api/v1/runs/{run_id}/workers/{pool_tag}/{worker_id}/operations",
            get(get_worker_operations),
        )
        .route(
            "/api/v1/runs/{run_id}/workers/{pool_tag}/{worker_id}/operations/seek",
            get(seek_worker_operation),
        )
        .route(
            "/api/v1/runs/{run_id}/workers/{pool_tag}/{worker_id}/operations/{iter_id}/{batch_id}/{operation_id}/cost-tree",
            get(get_worker_cost_tree),
        )
        .with_state(state)
        // The service otherwise receives no bodies. Bound this temporary
        // browser-profiling sink in case DevTools is left running.
        .layer(DefaultBodyLimit::max(TIMELINE_PROFILE_BODY_LIMIT));
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("bind analyzer UI service to {bind}"))?;
    eprintln!("[analyze] serving run catalog at http://{bind}/api/v1/runs");
    axum::serve(listener, app)
        .await
        .context("serve analyzer run catalog")
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

async fn list_runs(State(state): State<ServiceState>) -> Response {
    let roots = Arc::clone(&state.roots);
    match tokio::task::spawn_blocking(move || build_catalog(&roots)).await {
        Ok(Ok(catalog)) => Json(catalog).into_response(),
        Ok(Err(error)) => catalog_error(error),
        Err(error) => catalog_error(anyhow::anyhow!("catalog worker failed: {error}")),
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
    let repo_root = Arc::clone(&state.repo_root);
    read_run_resource(state, run_id, move |run| read_workload(&run, &repo_root)).await
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

async fn get_kernel_time_share_payload(
    RoutePath(run_id): RoutePath<String>,
    State(state): State<ServiceState>,
) -> Response {
    read_run_resource(state, run_id, |run| read_kernel_time_share_payload(&run)).await
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
    let run = match resolve_run(&state.roots, &run_id) {
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
    let run = match resolve_run(&state.roots, &run_id) {
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
    let run = match resolve_run(&state.roots, &run_id) {
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
    let roots = Arc::clone(&state.roots);
    match tokio::task::spawn_blocking(move || {
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

fn problem(status: StatusCode, code: &'static str, detail: &'static str) -> Response {
    (status, Json(json!({ "code": code, "detail": detail }))).into_response()
}
