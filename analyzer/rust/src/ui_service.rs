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
mod model;
mod slo;
#[cfg(test)]
mod tests;
mod throughput;
mod topology;
mod utilization;
mod workload;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::extract::{Path as RoutePath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};

use catalog::build_catalog;
use concurrency::{read_concurrency_payload, read_concurrency_report};
use core::{build_descriptor, read_summary};
use discovery::{configure_logs_roots, resolve_run, ConfiguredRoot, DiscoveredRun};
use model::read_model;
use slo::{read_slo_general_payload, read_slo_general_report};
use throughput::{read_throughput_payload, read_throughput_report};
use topology::build_topology;
use utilization::{read_utilization_payload, read_utilization_report};
use workload::read_workload;

const PROTOCOL_VERSION: u32 = 1;

#[derive(Clone, Debug)]
struct ServiceState {
    roots: Arc<Vec<ConfiguredRoot>>,
    repo_root: Arc<PathBuf>,
}

pub(crate) async fn serve(bind: SocketAddr, logs_roots: Vec<PathBuf>) -> Result<()> {
    let roots = configure_logs_roots(logs_roots)?;
    let repo_root = std::env::current_dir().context("resolve analyzer repository root")?;
    let state = ServiceState {
        roots: Arc::new(roots),
        repo_root: Arc::new(repo_root),
    };
    let app = Router::new()
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
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("bind analyzer UI service to {bind}"))?;
    eprintln!("[analyze] serving run catalog at http://{bind}/api/v1/runs");
    axum::serve(listener, app)
        .await
        .context("serve analyzer run catalog")
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
