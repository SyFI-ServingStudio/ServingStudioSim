use super::*;

use std::fs;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;

static ARTIFACT_READ_TEST_LOCK: StdMutex<()> = StdMutex::new(());

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
    write_timing_generation(run, statuses, None);
}

fn write_timing_generation(run: &Path, statuses: &[(&str, &str)], generation_id: Option<&str>) {
    write_json(
        &run.join("reports/analyzer_timing.json"),
        &json!({
            "schema_version": 1,
            "generation_id": generation_id,
            "subjects": statuses.iter().map(|(name, status)| json!({"name": name, "status": status})).collect::<Vec<_>>()
        }),
    );
}

fn write_pipeline(
    run: &Path,
    generation_id: &str,
    status: &str,
    compute: &str,
    render: &str,
    trace: &str,
    requested_subjects: Option<&[&str]>,
    trace_artifact: Option<&str>,
) {
    let stage = |stage_name: &str, stage_status: &str, artifact: Option<&str>| {
        let code = (stage_status == "failed").then(|| format!("{stage_name}_failed"));
        json!({
            "status": stage_status,
            "updated_at": "2026-07-15T00:00:00.000Z",
            "code": code,
            "artifact": artifact,
        })
    };
    write_json(
        &run.join(PIPELINE_STATE_PATH),
        &json!({
            "schema_version": 1,
            "generation_id": generation_id,
            "artifact_revision": format!("pipeline-{generation_id}"),
            "status": status,
            "started_at": "2026-07-15T00:00:00.000Z",
            "updated_at": "2026-07-15T00:00:01.000Z",
            "completed_at": matches!(status, "complete" | "failed").then_some("2026-07-15T00:00:01.000Z"),
            "producer": {
                "name": "vibesim-analyzer",
                "version": "0.1.0",
                "revision": "a".repeat(40),
                "binary_sha256": format!("sha256:{}", "b".repeat(64)),
            },
            "requested_subjects": requested_subjects,
            "stages": {
                "compute": stage("compute", compute, None),
                "render": stage("render", render, None),
                "trace": stage("trace", trace, trace_artifact),
            },
        }),
    );
}

fn pipeline_from_lifecycle_row(row: &Value) -> PipelineStateV1 {
    let status = |name: &str| {
        serde_json::from_value::<StageStatus>(row[name].clone())
            .unwrap_or_else(|error| panic!("invalid {name} status in lifecycle fixture: {error}"))
    };
    let pipeline = status("pipeline");
    let compute = status("compute");
    let render = status("render");
    let trace = status("trace");
    let stage = |name: &str, stage_status: StageStatus| PipelineStage {
        status: stage_status,
        code: (stage_status == StageStatus::Failed).then(|| format!("{name}_failed")),
        artifact: (name == "trace" && stage_status == StageStatus::Complete)
            .then(|| "traces/test.pftrace.gz".to_string()),
    };
    PipelineStateV1 {
        schema_version: PIPELINE_SCHEMA_VERSION,
        generation_id: "fixture-generation".to_string(),
        artifact_revision: "pipeline-fixture-generation".to_string(),
        status: pipeline,
        started_at: "2026-07-15T00:00:00.000Z".to_string(),
        updated_at: "2026-07-15T00:00:01.000Z".to_string(),
        completed_at: matches!(pipeline, StageStatus::Complete | StageStatus::Failed)
            .then(|| "2026-07-15T00:00:01.000Z".to_string()),
        producer: PipelineProducer {
            name: "vibesim-analyzer".to_string(),
            version: "0.1.0".to_string(),
            revision: "a".repeat(40),
            binary_sha256: format!("sha256:{}", "b".repeat(64)),
        },
        requested_subjects: None,
        stages: PipelineStages {
            compute: stage("compute", compute),
            render: stage("render", render),
            trace: stage("trace", trace),
        },
    }
}

#[test]
fn shared_publisher_lifecycle_table_matches_rust_validation() {
    // Python publication tests consume this same table, so a lifecycle change
    // cannot silently make the producer and reader disagree again.
    let contract: Value = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/analyzer_pipeline_lifecycle_v1.json"
    ))
    .unwrap();
    assert_eq!(
        contract["schema_version"],
        Value::from(PIPELINE_SCHEMA_VERSION)
    );

    for row in contract["valid_states"].as_array().unwrap() {
        let state = pipeline_from_lifecycle_row(row);
        if let Err(reason) = validate_pipeline_state(&state) {
            panic!("shared lifecycle fixture row {row} was rejected: {reason}");
        }
    }
    for row in contract["invalid_regressions"].as_array().unwrap() {
        let state = pipeline_from_lifecycle_row(row);
        assert!(
            validate_pipeline_state(&state).is_err(),
            "regression lifecycle fixture row unexpectedly validated: {row}"
        );
    }
}

fn publish_subject_generation(run: &Path, generation_id: &str, marker: &str) {
    write_pipeline(
        run,
        generation_id,
        "pending",
        "pending",
        "not_started",
        "not_started",
        Some(&["slo-general"]),
        None,
    );
    let subject = SUBJECTS
        .iter()
        .find(|subject| subject.name == "slo-general")
        .unwrap();
    write_json(
        &run.join("reports").join(subject.report_name),
        &json!({"schema_version": 1, "available": true, "marker": marker}),
    );
    write_json(
        &run.join("payloads").join(subject.payload_name),
        &json!({"schema_version": 1, "meta": {"available": true}, "marker": marker}),
    );
    write_timing_generation(run, &[("slo-general", "ok")], Some(generation_id));
    write_pipeline(
        run,
        generation_id,
        "complete",
        "complete",
        "complete",
        "failed",
        Some(&["slo-general"]),
        None,
    );
}

fn state(root: &TempDir) -> ServiceState {
    ServiceState {
        roots: configure_roots(vec![root.path().to_path_buf()]).unwrap(),
        discovery: RwLock::new(DiscoveryCache::default()),
        discovery_refresh: AsyncMutex::new(()),
        discovery_scan_count: AtomicUsize::new(0),
        catalog: RwLock::new(None),
        catalog_build: StdMutex::new(()),
        descriptors: RwLock::new(HashMap::new()),
        descriptor_build: StdMutex::new(()),
        subject_proofs: RwLock::new(HashMap::new()),
        artifact_reads: Arc::new(Semaphore::new(MAX_IN_FLIGHT_ARTIFACT_READS)),
        allowed_hosts: configure_allowed_hosts(Vec::new()).unwrap(),
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

#[tokio::test]
async fn malformed_run_id_is_rejected_without_discovery() {
    let root = TempDir::new().unwrap();
    let state = Arc::new(state(&root));

    let problem = resolve_run(&state, "r_unknown")
        .await
        .expect_err("a non-canonical run id must not resolve");

    assert_eq!(problem.code, "run_not_found");
    assert_eq!(
        state
            .discovery_scan_count
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
}

#[tokio::test]
async fn different_unknown_run_ids_share_one_fresh_discovery_scan() {
    let root = TempDir::new().unwrap();
    let state = Arc::new(state(&root));
    let missing_ids = [
        opaque_run_id(0, Path::new("missing-a")),
        opaque_run_id(0, Path::new("missing-b")),
        opaque_run_id(0, Path::new("missing-c")),
        opaque_run_id(0, Path::new("missing-d")),
    ];

    let (first, second, third) = tokio::join!(
        resolve_run(&state, &missing_ids[0]),
        resolve_run(&state, &missing_ids[1]),
        resolve_run(&state, &missing_ids[2]),
    );
    for result in [first, second, third] {
        assert_eq!(result.unwrap_err().code, "run_not_found");
    }
    assert_eq!(
        state
            .discovery_scan_count
            .load(std::sync::atomic::Ordering::Relaxed),
        1,
        "single-flight must collapse concurrent cold-cache misses"
    );

    assert_eq!(
        resolve_run(&state, &missing_ids[3]).await.unwrap_err().code,
        "run_not_found"
    );
    assert_eq!(
        state
            .discovery_scan_count
            .load(std::sync::atomic::Ordering::Relaxed),
        1,
        "a different unknown id cannot bypass the fresh catalog TTL"
    );
}

#[tokio::test]
async fn expired_discovery_refreshes_and_finds_a_new_run() {
    let root = TempDir::new().unwrap();
    let state = Arc::new(state(&root));
    let missing_id = opaque_run_id(0, Path::new("initial-miss"));

    assert_eq!(
        resolve_run(&state, &missing_id).await.unwrap_err().code,
        "run_not_found"
    );
    create_run(root.path(), "new-run", "unified", true);
    let new_run_id = opaque_run_id(0, Path::new("new-run"));

    assert_eq!(
        resolve_run(&state, &new_run_id).await.unwrap_err().code,
        "run_not_found",
        "a fresh catalog is also the bounded negative cache"
    );
    assert_eq!(
        state
            .discovery_scan_count
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );

    state.discovery.write().unwrap().refreshed_at =
        Some(Instant::now() - CATALOG_REFRESH_INTERVAL - Duration::from_millis(1));
    let discovered = resolve_run(&state, &new_run_id)
        .await
        .expect("an expired catalog must refresh");

    assert_eq!(discovered.display_name, "new-run");
    assert_eq!(
        state
            .discovery_scan_count
            .load(std::sync::atomic::Ordering::Relaxed),
        2
    );
}

#[test]
fn canonical_run_id_validation_matches_the_sha256_generator() {
    let run_id = opaque_run_id(7, Path::new("sweep/simulation"));

    assert_eq!(run_id.len(), 2 + 64);
    assert!(is_canonical_run_id(&run_id));
    assert!(!is_canonical_run_id("r_deadbeef"));
    assert!(!is_canonical_run_id(&format!("r_{}", "A".repeat(64))));
    assert!(!is_canonical_run_id(&format!("x_{}", "a".repeat(64))));
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
        discovery: RwLock::new(DiscoveryCache::default()),
        discovery_refresh: AsyncMutex::new(()),
        discovery_scan_count: AtomicUsize::new(0),
        catalog: RwLock::new(None),
        catalog_build: StdMutex::new(()),
        descriptors: RwLock::new(HashMap::new()),
        descriptor_build: StdMutex::new(()),
        subject_proofs: RwLock::new(HashMap::new()),
        artifact_reads: Arc::new(Semaphore::new(MAX_IN_FLIGHT_ARTIFACT_READS)),
        allowed_hosts: configure_allowed_hosts(Vec::new()).unwrap(),
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
    let record = discover_runs(&state).unwrap().remove(0);
    let run_id = record.run_id.clone();
    let revision = build_descriptor(&record)
        .unwrap()
        .analysis
        .unwrap()
        .revision;
    let app = router(vec![root.path().to_path_buf()]).unwrap();

    let ready = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/api/v1/runs/{run_id}/revisions/{revision}/reports/slo-general"
                ))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ready.status(), StatusCode::OK);

    for path in [
        format!("/api/v1/runs/{run_id}/revisions/{revision}/reports/not-in-registry"),
        format!("/api/v1/runs/{run_id}/raw/params.json"),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(path)
                    .header(header::HOST, "localhost")
                    .body(Body::empty())
                    .unwrap(),
            )
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
async fn report_request_does_not_read_or_parse_the_payload() {
    let _observer_guard = ARTIFACT_READ_TEST_LOCK.lock().unwrap();
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_subject(&run, "slo-general", true, None);
    write_timing_generation(&run, &[("slo-general", "ok")], Some("generation-current"));
    write_pipeline(
        &run,
        "generation-current",
        "pending",
        "complete",
        "pending",
        "not_started",
        Some(&["slo-general"]),
        None,
    );
    let subject = SUBJECTS
        .iter()
        .find(|subject| subject.name == "slo-general")
        .unwrap();
    let run_id = discover_runs(&state(&root)).unwrap().remove(0).run_id;
    let app = router(vec![root.path().to_path_buf()]).unwrap();
    let (reads_tx, reads_rx) = std::sync::mpsc::channel();
    set_artifact_read_observer(Some(reads_tx));

    let descriptor = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/runs/{run_id}/descriptor"))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(descriptor.status(), StatusCode::OK);
    while reads_rx.try_recv().is_ok() {}

    let report = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/api/v1/runs/{run_id}/revisions/pipeline-generation-current/reports/slo-general"
                ))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(report.status(), StatusCode::OK);
    let reads = reads_rx.try_iter().collect::<Vec<_>>();
    set_artifact_read_observer(None);
    assert!(reads.contains(&run.join("reports").join(subject.report_name)));
    assert!(!reads.contains(&run.join("payloads").join(subject.payload_name)));
}

#[tokio::test]
async fn legacy_subject_reuses_content_revision_without_rehashing_counterpart() {
    let _observer_guard = ARTIFACT_READ_TEST_LOCK.lock().unwrap();
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_subject(&run, "slo-general", true, None);
    write_timing(&run, &[("slo-general", "ok")]);
    let subject = SUBJECTS
        .iter()
        .find(|subject| subject.name == "slo-general")
        .unwrap();
    let run_id = discover_runs(&state(&root)).unwrap().remove(0).run_id;
    let app = router(vec![root.path().to_path_buf()]).unwrap();
    let descriptor = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/runs/{run_id}/descriptor"))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let descriptor = body_json(descriptor).await;
    let revision = descriptor["analysis"]["revision"].as_str().unwrap();
    let (reads_tx, reads_rx) = std::sync::mpsc::channel();
    set_artifact_read_observer(Some(reads_tx));

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/api/v1/runs/{run_id}/revisions/{revision}/reports/slo-general"
                ))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let reads = reads_rx.try_iter().collect::<Vec<_>>();
    set_artifact_read_observer(None);
    assert!(reads.contains(&run.join("reports").join(subject.report_name)));
    assert!(!reads.contains(&run.join("payloads").join(subject.payload_name)));
}

#[tokio::test]
async fn changed_counterpart_invalidates_cached_subject_readiness() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    publish_subject_generation(&run, "generation-current", "before");
    let run_id = discover_runs(&state(&root)).unwrap().remove(0).run_id;
    let app = router(vec![root.path().to_path_buf()]).unwrap();
    let descriptor = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/runs/{run_id}/descriptor"))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(descriptor.status(), StatusCode::OK);

    let subject = SUBJECTS
        .iter()
        .find(|subject| subject.name == "slo-general")
        .unwrap();
    fs::write(run.join("payloads").join(subject.payload_name), b"not json").unwrap();
    let response = app
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/api/v1/runs/{run_id}/revisions/pipeline-generation-current/reports/slo-general"
                ))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::CONFLICT);
    assert_eq!(
        body_json(response).await["code"],
        "artifact_generation_changed"
    );
}

#[tokio::test]
async fn saturated_artifact_reader_limit_fails_fast_with_stable_problem() {
    let root = TempDir::new().unwrap();
    create_run(root.path(), "run", "unified", true);
    let mut service_state = state(&root);
    service_state.artifact_reads = Arc::new(Semaphore::new(1));
    let service_state = Arc::new(service_state);
    let held_permit = Arc::clone(&service_state.artifact_reads)
        .try_acquire_owned()
        .unwrap();
    let run_id = discover_runs(&service_state).unwrap().remove(0).run_id;
    let app = router_from_state(Arc::clone(&service_state));

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/runs/{run_id}/summary"))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(response.headers()[header::RETRY_AFTER], "1");
    assert_eq!(body_json(response).await["code"], "artifact_read_busy");
    drop(held_permit);
}

#[tokio::test]
async fn trace_stream_holds_its_artifact_reader_permit_until_body_drop() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_subject(&run, "slo-general", true, None);
    write_timing_generation(&run, &[("slo-general", "ok")], Some("generation-trace"));
    fs::create_dir_all(run.join("traces")).unwrap();
    fs::write(run.join("traces/current.pftrace.gz"), b"trace bytes").unwrap();
    write_pipeline(
        &run,
        "generation-trace",
        "complete",
        "complete",
        "complete",
        "complete",
        Some(&["slo-general"]),
        Some("traces/current.pftrace.gz"),
    );

    let mut service_state = state(&root);
    service_state.artifact_reads = Arc::new(Semaphore::new(1));
    let service_state = Arc::new(service_state);
    let run_id = discover_runs(&service_state).unwrap().remove(0).run_id;
    let app = router_from_state(Arc::clone(&service_state));
    let trace_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/api/v1/runs/{run_id}/revisions/pipeline-generation-trace/traces/perfetto"
                ))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(trace_response.status(), StatusCode::OK);
    assert_eq!(service_state.artifact_reads.available_permits(), 0);

    let saturated = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/runs/{run_id}/summary"))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(saturated.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body_json(saturated).await["code"], "artifact_read_busy");

    drop(trace_response);
    assert_eq!(service_state.artifact_reads.available_permits(), 1);
    let after_drop = app
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/runs/{run_id}/summary"))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(after_drop.status(), StatusCode::OK);
}

#[tokio::test(flavor = "current_thread")]
async fn artifact_io_runs_on_a_blocking_worker() {
    let root = TempDir::new().unwrap();
    let service_state = Arc::new(state(&root));
    let async_worker = std::thread::current().id();

    let blocking_worker =
        run_blocking_artifact_task(&service_state, "thread-boundary test", move || {
            Ok(std::thread::current().id())
        })
        .await
        .unwrap();

    assert_ne!(blocking_worker, async_worker);
}

#[tokio::test]
async fn matching_metadata_etag_returns_before_reading_invalid_json() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    fs::write(run.join("summary.json"), b"not json").unwrap();
    let record = discover_runs(&state(&root)).unwrap().remove(0);
    let relative = Path::new("summary.json");
    let artifact = open_bounded_artifact(&record, relative, MAX_JSON_BYTES, "summary").unwrap();
    let etag = metadata_etag("json-artifact-v1", &[(relative, &artifact.metadata)]);
    let app = router(vec![root.path().to_path_buf()]).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/runs/{}/summary", record.run_id))
                .header(header::HOST, "localhost")
                .header(header::IF_NONE_MATCH, etag)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_MODIFIED);
    assert!(to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn wildcard_etag_does_not_bypass_json_validation() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    fs::write(run.join("summary.json"), b"not json").unwrap();
    let run_id = discover_runs(&state(&root)).unwrap().remove(0).run_id;
    let app = router(vec![root.path().to_path_buf()]).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/runs/{run_id}/summary"))
                .header(header::HOST, "localhost")
                .header(header::IF_NONE_MATCH, "*")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body_json(response).await["code"], "artifact_incompatible");
}

#[tokio::test]
async fn json_artifact_limit_is_enforced_before_allocation() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    fs::File::create(run.join("summary.json"))
        .unwrap()
        .set_len(MAX_JSON_BYTES + 1)
        .unwrap();
    let run_id = discover_runs(&state(&root)).unwrap().remove(0).run_id;
    let app = router(vec![root.path().to_path_buf()]).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/runs/{run_id}/summary"))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(body_json(response).await["code"], "artifact_too_large");
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
                .header(header::HOST, "localhost")
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
                .header(header::HOST, "localhost")
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
    let record = discover_runs(&state(&root)).unwrap().remove(0);
    let run_id = record.run_id.clone();
    let revision = build_descriptor(&record)
        .unwrap()
        .analysis
        .unwrap()
        .revision;
    let app = router(vec![root.path().to_path_buf()]).unwrap();

    let rejected_host = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/runs")
                .header(header::HOST, "attacker.example")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rejected_host.status(), StatusCode::MISDIRECTED_REQUEST);
    assert_eq!(body_json(rejected_host).await["code"], "host_not_allowed");

    let catalog = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/runs")
                .header(header::HOST, "localhost")
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
                .uri(format!(
                    "/api/v1/runs/{run_id}/revisions/{revision}/reports/%73lo-general"
                ))
                .header(header::HOST, "localhost")
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
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(write_attempt.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(write_attempt.headers()[header::ALLOW], "GET");
    assert_eq!(body_json(write_attempt).await["code"], "method_not_allowed");
}

#[tokio::test]
async fn explicitly_configured_proxy_host_is_accepted() {
    let root = TempDir::new().unwrap();
    create_run(root.path(), "run", "unified", true);
    let app = router_with_hosts(
        vec![root.path().to_path_buf()],
        vec!["ui.internal".to_string()],
    )
    .unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/runs")
                .header(header::HOST, "UI.INTERNAL:5177")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
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
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    assert_eq!(body_json(response).await["code"], "artifact_outside_run");
}

#[cfg(unix)]
#[test]
fn replacing_a_discovered_run_with_a_symlink_cannot_escape_the_root() {
    use std::os::unix::fs::symlink;

    let root = TempDir::new().unwrap();
    let outside = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_json(
        &outside.path().join("summary.json"),
        &json!({"secret": true}),
    );
    let discovered = discover_runs(&state(&root)).unwrap().remove(0);

    fs::rename(&run, root.path().join("discovered-run-moved")).unwrap();
    symlink(outside.path(), &run).unwrap();

    let problem = open_contained_artifact(&discovered, Path::new("summary.json"), "summary")
        .expect_err("replacement symlink must not escape the configured root");
    assert_eq!(problem.status, StatusCode::FORBIDDEN);
    assert_eq!(problem.code, "artifact_outside_run");
}

#[tokio::test]
async fn catalog_descriptor_and_artifact_honor_etag() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_subject(&run, "slo-general", true, None);
    write_timing(&run, &[("slo-general", "ok")]);
    let record = discover_runs(&state(&root)).unwrap().remove(0);
    let run_id = record.run_id.clone();
    let revision = build_descriptor(&record)
        .unwrap()
        .analysis
        .unwrap()
        .revision;
    let app = router(vec![root.path().to_path_buf()]).unwrap();

    for uri in [
        "/api/v1/runs".to_string(),
        format!("/api/v1/runs/{run_id}/descriptor"),
        format!("/api/v1/runs/{run_id}/revisions/{revision}/payloads/slo-general"),
    ] {
        let first = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(&uri)
                    .header(header::HOST, "localhost")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let etag = first.headers()[header::ETAG].clone();
        let conditional = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(&uri)
                    .header(header::HOST, "localhost")
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn catalog_cold_build_is_singleflight_and_cache_hits_skip_lifecycle_bodies() {
    let _observer_guard = ARTIFACT_READ_TEST_LOCK.lock().unwrap();
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_timing_generation(&run, &[("slo-general", "ok")], Some("catalog-generation"));
    write_pipeline(
        &run,
        "catalog-generation",
        "complete",
        "complete",
        "complete",
        "failed",
        Some(&["slo-general"]),
        None,
    );
    let app = router(vec![root.path().to_path_buf()]).unwrap();
    let (reads_tx, reads_rx) = std::sync::mpsc::channel();
    set_artifact_read_observer(Some(reads_tx));

    let first_request = app.clone().oneshot(
        Request::builder()
            .uri("/api/v1/runs")
            .header(header::HOST, "localhost")
            .body(Body::empty())
            .unwrap(),
    );
    let second_request = app.clone().oneshot(
        Request::builder()
            .uri("/api/v1/runs")
            .header(header::HOST, "localhost")
            .body(Body::empty())
            .unwrap(),
    );
    let (first, second) = tokio::join!(first_request, second_request);
    let first = first.unwrap();
    let second = second.unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(second.status(), StatusCode::OK);
    let etag = first.headers()[header::ETAG].clone();
    assert_eq!(second.headers()[header::ETAG], etag);

    let cold_reads = reads_rx.try_iter().collect::<Vec<_>>();
    assert_eq!(
        cold_reads
            .iter()
            .filter(|path| *path == &run.join(PIPELINE_STATE_PATH))
            .count(),
        1
    );
    assert_eq!(
        cold_reads
            .iter()
            .filter(|path| *path == &run.join("reports/analyzer_timing.json"))
            .count(),
        1
    );

    let cached = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/runs")
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cached.status(), StatusCode::OK);
    let conditional = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/runs")
                .header(header::HOST, "localhost")
                .header(header::IF_NONE_MATCH, etag)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(conditional.status(), StatusCode::NOT_MODIFIED);
    set_artifact_read_observer(None);
    assert!(reads_rx
        .try_iter()
        .all(|observed| !observed.starts_with(&run)));
}

#[tokio::test]
async fn catalog_cache_invalidates_when_lifecycle_metadata_changes() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", false);
    let app = router(vec![root.path().to_path_buf()]).unwrap();

    let first = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/v1/runs")
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    let first_etag = first.headers()[header::ETAG].clone();
    let first_body = body_json(first).await;
    assert_eq!(first_body["runs"][0]["lifecycle"]["simulation"], "pending");
    assert_eq!(
        first_body["runs"][0]["lifecycle"]["analysis"],
        "not_started"
    );

    fs::write(run.join(".complete"), b"").unwrap();
    write_timing(&run, &[("slo-general", "ok")]);

    let second = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/runs")
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(second.status(), StatusCode::OK);
    assert_ne!(second.headers()[header::ETAG], first_etag);
    let second_body = body_json(second).await;
    assert_eq!(
        second_body["runs"][0]["lifecycle"]["simulation"],
        "complete"
    );
    assert_eq!(second_body["runs"][0]["lifecycle"]["analysis"], "complete");
}

#[tokio::test]
async fn catalog_rejects_aggregate_lifecycle_input_before_reading_bodies() {
    let _observer_guard = ARTIFACT_READ_TEST_LOCK.lock().unwrap();
    let root = TempDir::new().unwrap();
    let per_run_bytes = MAX_CATALOG_LIFECYCLE_BYTES / 2 + 1;
    for name in ["run-a", "run-b"] {
        let run = create_run(root.path(), name, "unified", true);
        fs::create_dir_all(run.join("reports")).unwrap();
        let timing = fs::File::create(run.join("reports/analyzer_timing.json")).unwrap();
        timing.set_len(per_run_bytes).unwrap();
    }
    let app = router(vec![root.path().to_path_buf()]).unwrap();
    let (reads_tx, reads_rx) = std::sync::mpsc::channel();
    set_artifact_read_observer(Some(reads_tx));

    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/v1/runs")
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let problem = body_json(response).await;
    set_artifact_read_observer(None);
    assert_eq!(problem["code"], "catalog_state_too_large");
    assert!(reads_rx
        .try_iter()
        .all(|observed| !observed.starts_with(root.path())));
}

#[test]
fn catalog_returns_stable_conflict_after_bounded_metadata_churn() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_timing(&run, &[("slo-general", "ok")]);
    let service_state = state(&root);
    let records = discover_runs(&service_state).unwrap();
    let timing_path = run.join("reports/analyzer_timing.json");
    let hook_calls = std::cell::Cell::new(0);

    let result =
        prepare_catalog_with_build_hook(&service_state, &HeaderMap::new(), &records, |attempt| {
            hook_calls.set(hook_calls.get() + 1);
            fs::write(&timing_path, vec![b' '; attempt + 1]).unwrap();
        });
    let problem = match result {
        Err(problem) => problem,
        Ok(_) => panic!("continuous metadata churn must exhaust the bounded fence"),
    };

    assert_eq!(hook_calls.get(), MAX_GENERATION_READ_ATTEMPTS);
    assert_eq!(problem.status, StatusCode::CONFLICT);
    assert_eq!(problem.code, "artifact_generation_changed");
    assert_eq!(
        problem.detail,
        "Analyzer catalog lifecycle metadata changed during all 3 bounded read attempts."
    );
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
    let identity = descriptor
        .analysis
        .expect("ready legacy artifact has identity");
    assert_eq!(identity.generator_version, LEGACY_GENERATOR_VERSION);
    let rebuilt = build_descriptor(&record).unwrap().analysis.unwrap();
    assert_eq!(identity.revision, rebuilt.revision);
}

#[test]
fn pending_generation_hides_legacy_subject_and_trace_files() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_subject(&run, "slo-general", true, None);
    write_timing(&run, &[("slo-general", "ok")]);
    fs::create_dir_all(run.join("traces")).unwrap();
    fs::write(run.join("traces/old.pftrace.gz"), b"old trace").unwrap();
    write_pipeline(
        &run,
        "generation-new",
        "pending",
        "pending",
        "not_started",
        "not_started",
        Some(&["slo-general"]),
        None,
    );

    let record = discover_runs(&state(&root)).unwrap().remove(0);
    let descriptor = build_descriptor(&record).unwrap();
    assert_eq!(descriptor.lifecycle.analysis, StageStatus::Pending);
    assert!(matches!(
        descriptor.subjects["slo-general"],
        SubjectState::Pending { .. }
    ));
    assert!(matches!(
        descriptor.subjects["throughput"],
        SubjectState::NotGenerated { .. }
    ));
    assert!(matches!(
        descriptor.traces["perfetto"],
        TraceState::Pending { .. }
    ));
    assert!(descriptor.analysis.is_none());
}

#[test]
fn compute_generation_and_requested_subjects_gate_current_ready_files() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_subject(&run, "slo-general", true, None);
    write_subject(&run, "throughput", true, None);
    write_timing_generation(&run, &[("slo-general", "ok")], Some("generation-current"));
    write_pipeline(
        &run,
        "generation-current",
        "pending",
        "complete",
        "pending",
        "not_started",
        Some(&["slo-general"]),
        None,
    );

    let record = discover_runs(&state(&root)).unwrap().remove(0);
    let descriptor = build_descriptor(&record).unwrap();
    assert_eq!(descriptor.lifecycle.analysis, StageStatus::Pending);
    assert!(matches!(
        &descriptor.subjects["slo-general"],
        SubjectState::Ready {
            report_href,
            payload_href,
            ..
        } if report_href == "revisions/pipeline-generation-current/reports/slo-general"
            && payload_href == "revisions/pipeline-generation-current/payloads/slo-general"
    ));
    assert!(matches!(
        descriptor.subjects["throughput"],
        SubjectState::NotGenerated { .. }
    ));
    let identity = descriptor.analysis.expect("compute identity");
    assert_eq!(identity.revision, "pipeline-generation-current");
    assert_ne!(identity.generator_version, LEGACY_GENERATOR_VERSION);
}

#[test]
fn complete_pipeline_preserves_json_when_trace_stage_failed() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_subject(&run, "slo-general", true, None);
    write_timing_generation(
        &run,
        &[("slo-general", "ok")],
        Some("generation-trace-failed"),
    );
    write_pipeline(
        &run,
        "generation-trace-failed",
        "complete",
        "complete",
        "complete",
        "failed",
        Some(&["slo-general"]),
        None,
    );

    let record = discover_runs(&state(&root)).unwrap().remove(0);
    let descriptor = build_descriptor(&record).unwrap();
    assert_eq!(descriptor.lifecycle.analysis, StageStatus::Complete);
    assert!(matches!(
        descriptor.subjects["slo-general"],
        SubjectState::Ready { .. }
    ));
    assert!(matches!(
        &descriptor.traces["perfetto"],
        TraceState::Failed { code, .. } if code == "trace_failed"
    ));
}

#[test]
fn generation_mismatch_never_revives_old_ready_files() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_subject(&run, "slo-general", true, None);
    write_timing_generation(&run, &[("slo-general", "ok")], Some("generation-old"));
    fs::create_dir_all(run.join("traces")).unwrap();
    fs::write(run.join("traces/current.pftrace.gz"), b"trace").unwrap();
    write_pipeline(
        &run,
        "generation-new",
        "complete",
        "complete",
        "complete",
        "complete",
        Some(&["slo-general"]),
        Some("traces/current.pftrace.gz"),
    );

    let record = discover_runs(&state(&root)).unwrap().remove(0);
    let descriptor = build_descriptor(&record).unwrap();
    assert_eq!(descriptor.lifecycle.analysis, StageStatus::Failed);
    assert!(matches!(
        &descriptor.subjects["slo-general"],
        SubjectState::Failed { code, .. } if code == "analysis_generation_mismatch"
    ));
    assert!(matches!(
        &descriptor.traces["perfetto"],
        TraceState::Failed { code, .. } if code == "analysis_generation_mismatch"
    ));
    assert!(descriptor.analysis.is_none());
}

#[test]
fn incompatible_pipeline_state_never_falls_back_to_legacy_artifacts() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_subject(&run, "slo-general", true, None);
    write_timing(&run, &[("slo-general", "ok")]);
    write_json(
        &run.join(PIPELINE_STATE_PATH),
        &json!({"schema_version": 99, "status": "complete"}),
    );

    let record = discover_runs(&state(&root)).unwrap().remove(0);
    let descriptor = build_descriptor(&record).unwrap();
    assert_eq!(descriptor.lifecycle.analysis, StageStatus::Failed);
    assert!(matches!(
        &descriptor.subjects["slo-general"],
        SubjectState::Failed { code, .. } if code == "pipeline_state_incompatible"
    ));
    assert!(descriptor.analysis.is_none());
}

#[tokio::test]
async fn stale_revision_link_fails_closed_after_generation_cutover() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    publish_subject_generation(&run, "generation-old", "old-bytes");
    let record = discover_runs(&state(&root)).unwrap().remove(0);
    let run_id = record.run_id.clone();
    let old_descriptor = build_descriptor(&record).unwrap();
    let old_report_href = match &old_descriptor.subjects["slo-general"] {
        SubjectState::Ready { report_href, .. } => report_href.clone(),
        state => panic!("expected old ready subject, got {state:?}"),
    };
    assert!(old_report_href.contains("pipeline-generation-old"));

    publish_subject_generation(&run, "generation-new", "new-bytes");
    let app = router(vec![root.path().to_path_buf()]).unwrap();
    let stale = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/runs/{run_id}/{old_report_href}"))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(stale.status(), StatusCode::CONFLICT);
    assert_eq!(
        body_json(stale).await["code"],
        "artifact_generation_changed"
    );

    let new_descriptor = build_descriptor(&record).unwrap();
    let new_report_href = match &new_descriptor.subjects["slo-general"] {
        SubjectState::Ready { report_href, .. } => report_href.clone(),
        state => panic!("expected new ready subject, got {state:?}"),
    };
    let current = app
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/runs/{run_id}/{new_report_href}"))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(current.status(), StatusCode::OK);
    assert_eq!(body_json(current).await["marker"], "new-bytes");
}

#[tokio::test]
async fn current_revision_non_ready_subject_is_not_a_generation_conflict() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    publish_subject_generation(&run, "generation-current", "current-bytes");
    let run_id = discover_runs(&state(&root)).unwrap().remove(0).run_id;
    let app = router(vec![root.path().to_path_buf()]).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/api/v1/runs/{run_id}/revisions/pipeline-generation-current/reports/throughput"
                ))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(body_json(response).await["code"], "resource_not_ready");
}

#[tokio::test]
async fn descriptor_ready_subject_remains_readable_while_simulation_is_pending() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", false);
    publish_subject_generation(&run, "generation-current", "current-bytes");
    let run_id = discover_runs(&state(&root)).unwrap().remove(0).run_id;
    let app = router(vec![root.path().to_path_buf()]).unwrap();

    let descriptor = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/runs/{run_id}/descriptor"))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(descriptor.status(), StatusCode::OK);
    let descriptor = body_json(descriptor).await;
    assert_eq!(descriptor["lifecycle"]["simulation"], "pending");
    assert_eq!(descriptor["subjects"]["slo-general"]["status"], "ready");
    let report_href = descriptor["subjects"]["slo-general"]["report_href"]
        .as_str()
        .unwrap();

    let report = app
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/runs/{run_id}/{report_href}"))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(report.status(), StatusCode::OK);
    assert_eq!(body_json(report).await["marker"], "current-bytes");
}

#[test]
fn descriptor_seqlock_discards_a_descriptor_crossed_by_cutover() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    publish_subject_generation(&run, "generation-old", "old-bytes");
    let record = discover_runs(&state(&root)).unwrap().remove(0);
    let mut attempts = 0usize;

    let descriptor = with_consistent_analysis_snapshot(&record, |snapshot| {
        attempts += 1;
        let descriptor = build_descriptor_at_snapshot(&record, snapshot)?;
        if attempts == 1 {
            publish_subject_generation(&run, "generation-new", "new-bytes");
        }
        Ok(descriptor)
    })
    .unwrap();

    assert_eq!(attempts, 2);
    assert_eq!(
        descriptor.analysis.unwrap().revision,
        "pipeline-generation-new"
    );
    assert!(matches!(
        &descriptor.subjects["slo-general"],
        SubjectState::Ready { report_href, .. }
            if report_href.contains("pipeline-generation-new")
    ));
}

#[test]
fn generation_seqlock_retries_the_whole_read_after_cutover() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    publish_subject_generation(&run, "generation-old", "old-bytes");
    let record = discover_runs(&state(&root)).unwrap().remove(0);
    let subject = SUBJECTS
        .iter()
        .find(|subject| subject.name == "slo-general")
        .unwrap();
    let mut attempts = 0usize;

    let observed_marker = with_consistent_analysis_snapshot(&record, |snapshot| {
        attempts += 1;
        snapshot
            .artifact_revision()
            .expect("published generation has a revision");
        let report = read_json_value(
            &record,
            &Path::new("reports").join(subject.report_name),
            "subject report",
        )?;
        let marker = report["marker"].as_str().unwrap().to_string();
        if attempts == 1 {
            publish_subject_generation(&run, "generation-new", "new-bytes");
        }
        Ok(marker)
    })
    .unwrap();

    assert_eq!(attempts, 2);
    assert_eq!(observed_marker, "new-bytes");
}

#[test]
fn generation_seqlock_returns_stable_conflict_under_repeated_cutover() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    publish_subject_generation(&run, "generation-0", "bytes-0");
    let record = discover_runs(&state(&root)).unwrap().remove(0);
    let mut attempts = 0usize;

    let result: Result<String, ApiProblem> =
        with_consistent_analysis_snapshot(&record, |snapshot| {
            let observed = snapshot.artifact_revision().unwrap().to_string();
            attempts += 1;
            publish_subject_generation(
                &run,
                &format!("generation-{attempts}"),
                &format!("bytes-{attempts}"),
            );
            Ok(observed)
        });

    let problem = match result {
        Ok(revision) => panic!("repeated cutover unexpectedly returned {revision}"),
        Err(problem) => problem,
    };
    assert_eq!(attempts, MAX_GENERATION_READ_ATTEMPTS);
    assert_eq!(problem.status, StatusCode::CONFLICT);
    assert_eq!(problem.code, "artifact_generation_changed");
}

#[test]
fn versioned_descriptor_stamp_ignores_unclaimed_newer_trace() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_subject(&run, "slo-general", true, None);
    write_timing_generation(&run, &[("slo-general", "ok")], Some("generation-trace"));
    fs::create_dir_all(run.join("traces")).unwrap();
    fs::write(run.join("traces/current.pftrace.gz"), b"current").unwrap();
    write_pipeline(
        &run,
        "generation-trace",
        "complete",
        "complete",
        "complete",
        "complete",
        Some(&["slo-general"]),
        Some("traces/current.pftrace.gz"),
    );
    let record = discover_runs(&state(&root)).unwrap().remove(0);
    let before = descriptor_stamp(&record);

    fs::write(run.join("traces/unclaimed-newer.pftrace.gz"), b"unclaimed").unwrap();
    assert_eq!(descriptor_stamp(&record), before);

    fs::write(
        run.join("traces/current.pftrace.gz"),
        b"changed-current-trace",
    )
    .unwrap();
    assert_ne!(descriptor_stamp(&record), before);
}

#[tokio::test]
async fn pipeline_selects_exact_trace_and_trace_endpoint_honors_etag() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_subject(&run, "slo-general", true, None);
    write_timing_generation(&run, &[("slo-general", "ok")], Some("generation-trace"));
    fs::create_dir_all(run.join("traces")).unwrap();
    fs::write(run.join("traces/old.pftrace.gz"), b"old").unwrap();
    let current = b"current generation trace";
    fs::write(run.join("traces/current.pftrace.gz"), current).unwrap();
    write_pipeline(
        &run,
        "generation-trace",
        "complete",
        "complete",
        "complete",
        "complete",
        Some(&["slo-general"]),
        Some("traces/current.pftrace.gz"),
    );
    let run_id = discover_runs(&state(&root)).unwrap().remove(0).run_id;
    let app = router(vec![root.path().to_path_buf()]).unwrap();
    let uri = format!("/api/v1/runs/{run_id}/revisions/pipeline-generation-trace/traces/perfetto");

    let first = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(&uri)
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(
        first.headers()[header::CONTENT_LENGTH],
        current.len().to_string()
    );
    let etag = first.headers()[header::ETAG].clone();
    assert!(etag.to_str().unwrap().starts_with("W/\"trace-v1-"));
    assert_eq!(
        to_bytes(first.into_body(), usize::MAX)
            .await
            .unwrap()
            .as_ref(),
        current
    );

    let conditional = app
        .oneshot(
            Request::builder()
                .uri(&uri)
                .header(header::HOST, "localhost")
                .header(header::IF_NONE_MATCH, etag)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(conditional.status(), StatusCode::NOT_MODIFIED);
}

#[tokio::test]
async fn legacy_trace_conditional_get_reuses_cached_content_revision() {
    let _observer_guard = ARTIFACT_READ_TEST_LOCK.lock().unwrap();
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_subject(&run, "slo-general", true, None);
    write_timing(&run, &[("slo-general", "ok")]);
    fs::create_dir_all(run.join("traces")).unwrap();
    fs::write(run.join("traces/legacy.pftrace.gz"), b"legacy trace").unwrap();
    let run_id = discover_runs(&state(&root)).unwrap().remove(0).run_id;
    let app = router(vec![root.path().to_path_buf()]).unwrap();
    let descriptor = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/api/v1/runs/{run_id}/descriptor"))
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let descriptor = body_json(descriptor).await;
    let href = descriptor["traces"]["perfetto"]["href"].as_str().unwrap();
    let uri = format!("/api/v1/runs/{run_id}/{href}");
    let (reads_tx, reads_rx) = std::sync::mpsc::channel();
    set_artifact_read_observer(Some(reads_tx));

    let first = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(&uri)
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(first.status(), StatusCode::OK);
    let etag = first.headers()[header::ETAG].clone();
    let conditional = app
        .oneshot(
            Request::builder()
                .uri(&uri)
                .header(header::HOST, "localhost")
                .header(header::IF_NONE_MATCH, etag)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(conditional.status(), StatusCode::NOT_MODIFIED);
    set_artifact_read_observer(None);
    assert!(reads_rx.try_iter().collect::<Vec<_>>().is_empty());
}

#[test]
fn trace_fd_is_not_served_when_generation_changes_after_open() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_subject(&run, "slo-general", true, None);
    write_timing_generation(&run, &[("slo-general", "ok")], Some("generation-old"));
    fs::create_dir_all(run.join("traces")).unwrap();
    let trace = run.join("traces/current.pftrace.gz");
    fs::write(&trace, b"old-trace").unwrap();
    write_pipeline(
        &run,
        "generation-old",
        "complete",
        "complete",
        "complete",
        "complete",
        Some(&["slo-general"]),
        Some("traces/current.pftrace.gz"),
    );
    let record = discover_runs(&state(&root)).unwrap().remove(0);
    let mut cut_over = false;

    let result: Result<PreparedTrace, ApiProblem> =
        with_consistent_analysis_snapshot(&record, |snapshot| {
            snapshot.require_revision("pipeline-generation-old")?;
            let path = select_trace_for_pipeline(&record, snapshot.pipeline())?;
            let prepared = prepare_trace(&record, path)?;
            if !cut_over {
                cut_over = true;
                write_pipeline(
                    &run,
                    "generation-new",
                    "pending",
                    "pending",
                    "not_started",
                    "not_started",
                    Some(&["slo-general"]),
                    None,
                );
                let replacement = run.join("traces/replacement.tmp");
                fs::write(&replacement, b"new-trace").unwrap();
                fs::rename(replacement, &trace).unwrap();
                write_timing_generation(&run, &[("slo-general", "ok")], Some("generation-new"));
                write_pipeline(
                    &run,
                    "generation-new",
                    "complete",
                    "complete",
                    "complete",
                    "complete",
                    Some(&["slo-general"]),
                    Some("traces/current.pftrace.gz"),
                );
            }
            Ok(prepared)
        });

    let problem = match result {
        Ok(_) => panic!("old trace fd must not survive a revision cutover response"),
        Err(problem) => problem,
    };
    assert!(cut_over);
    assert_eq!(problem.status, StatusCode::CONFLICT);
    assert_eq!(problem.code, "artifact_generation_changed");
}

#[test]
fn oversized_trace_is_not_declared_ready() {
    let root = TempDir::new().unwrap();
    let run = create_run(root.path(), "run", "unified", true);
    write_subject(&run, "slo-general", true, None);
    write_timing_generation(
        &run,
        &[("slo-general", "ok")],
        Some("generation-large-trace"),
    );
    fs::create_dir_all(run.join("traces")).unwrap();
    let trace = run.join("traces/large.pftrace.gz");
    fs::File::create(&trace)
        .unwrap()
        .set_len(MAX_TRACE_BYTES + 1)
        .unwrap();
    write_pipeline(
        &run,
        "generation-large-trace",
        "complete",
        "complete",
        "complete",
        "complete",
        Some(&["slo-general"]),
        Some("traces/large.pftrace.gz"),
    );

    let record = discover_runs(&state(&root)).unwrap().remove(0);
    let descriptor = build_descriptor(&record).unwrap();
    assert!(matches!(
        &descriptor.traces["perfetto"],
        TraceState::Failed { code, .. } if code == "artifact_too_large"
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
                .header(header::HOST, "localhost")
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
