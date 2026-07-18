use std::fs;
use std::path::Path;

use serde_json::{json, Value};
use tempfile::TempDir;

use super::batch::{read_batch_payload, read_batch_report};
use super::catalog::build_catalog;
use super::concurrency::{read_concurrency_payload, read_concurrency_report};
use super::core::{build_descriptor, read_summary};
use super::discovery::{configure_logs_roots, discover_runs};
use super::kernel_input_distribution::{
    read_kernel_input_distribution_payload, read_kernel_input_distribution_report,
};
use super::kernel_time_share::{read_kernel_time_share_payload, read_kernel_time_share_report};
use super::kv_occupancy::{read_kv_occupancy_payload, read_kv_occupancy_report};
use super::model::read_model;
use super::optimality::{read_optimality_payload, read_optimality_report};
use super::request_state::{read_request_state_payload, read_request_state_report};
use super::slo::{read_slo_general_payload, read_slo_general_report};
use super::throughput::{read_throughput_payload, read_throughput_report};
use super::topology::build_topology;
use super::utilization::{read_utilization_payload, read_utilization_report};
use super::workload::read_workload;
use super::workload_conservation::{
    read_workload_conservation_payload, read_workload_conservation_report,
};
use super::{timeline_profile_log_line, TimelineProfileEvent};

#[test]
fn timeline_profile_log_has_fixed_fields_and_rejects_non_finite_time() {
    let event = TimelineProfileEvent {
        session_id: "drag-1".to_owned(),
        event: "seek-settled".to_owned(),
        elapsed_ms: Some(42.5),
        cursor_ms: Some(1_000.0),
        worker: Some("attn/0".to_owned()),
        detail: None,
    };
    let line = timeline_profile_log_line(&event).expect("valid profiling event");
    let decoded: Value = serde_json::from_str(&line).expect("profiling line JSON");
    assert_eq!(decoded["session_id"], "drag-1");
    assert_eq!(decoded["event"], "seek-settled");
    assert_eq!(decoded["elapsed_ms"], 42.5);
    assert_eq!(decoded["cursor_ms"], 1_000.0);
    assert_eq!(decoded["worker"], "attn/0");
    assert!(decoded["detail"].is_null());

    let invalid = TimelineProfileEvent {
        elapsed_ms: Some(f64::NAN),
        ..event
    };
    assert!(timeline_profile_log_line(&invalid).is_err());
}

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

fn make_core_run(path: &Path) {
    fs::create_dir_all(path.join("raw")).expect("create raw directory");
    fs::create_dir_all(path.join("reports")).expect("create reports directory");
    fs::write(
        path.join("raw/params.json"),
        r#"{
            "deployment": "afd",
            "workload": {
                "trace_files": ["trace/workload.csv"],
                "request_rate": 2.0
            },
            "pools": {
                "attn": {
                    "groups": [{"arch": {"model_config": "model/config/qwen.json"}}]
                }
            }
        }"#,
    )
    .expect("write params");
    fs::write(
        path.join("raw/run_meta.json"),
        r#"{"gpus": [], "workers": [], "comm_groups": []}"#,
    )
    .expect("write run metadata");
    fs::write(
        path.join("summary.json"),
        r#"{"total_tok_s": 42.0, "num_gpus": 8}"#,
    )
    .expect("write summary");
    fs::write(
        path.join("reports/analyzer_timing.json"),
        r#"{"subjects": [
            {"name": "concurrency", "status": "ok"},
            {"name": "request-state", "status": "ok"},
            {"name": "slo-general", "status": "ok"},
            {"name": "throughput", "status": "ok"},
            {"name": "utilization", "status": "ok"},
            {"name": "batch", "status": "ok"},
            {"name": "kernel-input-distribution", "status": "ok"},
            {"name": "kernel-time-share", "status": "ok"},
            {"name": "optimality", "status": "ok"},
            {"name": "kv-occupancy", "status": "ok"},
            {"name": "workload-conservation", "status": "ok"}
        ]}"#,
    )
    .expect("write timing");
    fs::write(
        path.join("reports/concurrency_report.json"),
        r#"{"schema_version": 1, "available": true, "totals": {"peak_active": 2}}"#,
    )
    .expect("write concurrency report");
    fs::create_dir_all(path.join("payloads")).expect("create payloads directory");
    fs::write(
        path.join("payloads/concurrency_series.json"),
        r#"{"schema_version": 1, "t_ms": [5.0, 10.0], "active": [2.0, 1.0], "peak": 2}"#,
    )
    .expect("write concurrency payload");
    fs::write(
        path.join("reports/request_state_report.json"),
        r#"{"schema_version":1,"available":true,"totals":{"cluster_categories":[{"category":"pending","peak":2}]}}"#,
    )
    .expect("write request-state report");
    fs::write(
        path.join("payloads/request_state_series.json"),
        r#"{
            "schema_version": 1,
            "meta": {"log_dir": "logs/test"},
            "t_start_ms": [0.0, 5.0],
            "t_end_ms": [5.0, 10.0],
            "cluster_series": [
                {"category": "pending", "values": [1.0, 2.0]},
                {"category": "active", "values": [2.0, 1.0]},
                {"category": "done", "values": [0.0, 1.0]}
            ],
            "pools": []
        }"#,
    )
    .expect("write request-state payload");
    fs::write(
        path.join("reports/slo_general_report.json"),
        r#"{
            "schema_version": 1,
            "available": true,
            "metrics": {
                "ttft": {"mean": 12.0},
                "tpot": {"mean": 4.0},
                "e2e": {"mean": 40.0}
            }
        }"#,
    )
    .expect("write SLO report");
    fs::write(
        path.join("payloads/slo_general_cdf.json"),
        r#"{
            "schema_version": 1,
            "series": [
                {"key": "ttft", "x": [10.0], "y_pct": [100.0]},
                {"key": "tpot", "x": [4.0], "y_pct": [100.0]},
                {"key": "e2e", "x": [40.0], "y_pct": [100.0]}
            ]
        }"#,
    )
    .expect("write SLO payload");
    fs::write(
        path.join("reports/throughput_report.json"),
        r#"{
            "schema_version": 1,
            "available": true,
            "totals": {
                "prefill_tps": 1200.0,
                "decode_tps": 300.0,
                "total_tps": 1500.0,
                "total_tps_per_gpu": 187.5
            }
        }"#,
    )
    .expect("write throughput report");
    fs::write(
        path.join("payloads/throughput_segments.json"),
        r#"{
            "schema_version": 1,
            "t_start_ms": [0.0],
            "t_end_ms": [1000.0],
            "series": [
                {"key": "total", "per_gpu": [187.5]},
                {"key": "prefill", "per_gpu": [150.0]},
                {"key": "decode", "per_gpu": [37.5]}
            ]
        }"#,
    )
    .expect("write throughput payload");
    fs::write(
        path.join("reports/utilization_report.json"),
        r#"{
            "schema_version": 1,
            "available": true,
            "totals": {
                "per_pool": [{"pool": 0, "pool_tag": "attn", "avg_util": 0.6}],
                "per_worker": [
                    {"pool": 0, "pool_tag": "attn", "worker_id": 0, "avg_util": 0.8},
                    {"pool": 0, "pool_tag": "attn", "worker_id": 1, "avg_util": 0.4}
                ],
                "overall_avg": 0.6
            }
        }"#,
    )
    .expect("write utilization report");
    fs::write(
        path.join("payloads/utilization_series.json"),
        r#"{
            "schema_version": 1,
            "t_start_ms": [0.0],
            "t_end_ms": [1000.0],
            "series": [
                {"key": "pool_0", "label": "Pool 0", "pool_tag": "attn", "util": [0.6]}
            ],
            "worker_series": [
                {"key": "worker_0_0", "label": "attn/0", "pool_tag": "attn", "worker_id": 0, "util": [0.8]},
                {"key": "worker_0_1", "label": "attn/1", "pool_tag": "attn", "worker_id": 1, "util": [0.4]}
            ]
        }"#,
    )
    .expect("write utilization payload");
    fs::write(
        path.join("reports/batch_report.json"),
        r#"{
            "schema_version": 1,
            "available": true,
            "meta": {"num_calls": 2}
        }"#,
    )
    .expect("write batch report");
    fs::write(
        path.join("payloads/batch_scatter.json"),
        r#"{
            "schema_version": 1,
            "available": true,
            "meta": {"log_dir": "simulation", "num_calls": 2},
            "pools": [{
                "pool": "attn",
                "num_calls": 2,
                "plotted_points": 2,
                "time_ms": [0.0, 1.0],
                "series": [
                    {"key": "batch_tokens", "label": "Batch tokens", "values": [8, 16]},
                    {"key": "prefill_tokens", "label": "Prefill tokens", "values": [8, 0]},
                    {"key": "decode_request_count", "label": "Decode requests", "values": [0, 16]}
                ]
            }]
        }"#,
    )
    .expect("write batch payload");
    fs::write(
        path.join("reports/kernel_input_distribution_report.json"),
        r#"{
            "schema_version": 1,
            "available": true,
            "meta": {"num_positions_plotted": 1},
            "positions": [{"name": "m.layers.mlp", "num_points": 2}]
        }"#,
    )
    .expect("write kernel-input-distribution report");
    fs::write(
        path.join("payloads/kernel_input_distribution_scatter.json"),
        r#"{
            "schema_version": 1,
            "available": true,
            "positions": [{
                "name": "m.layers.mlp",
                "kind": "grouped_gemm",
                "candidate_backends": ["triton", "cutlass"],
                "selection": [
                    {"backend_index": 0, "backend_name": "triton", "count": 3, "ratio": 0.75},
                    {"backend_index": 1, "backend_name": "cutlass", "count": 1, "ratio": 0.25}
                ],
                "projection": "raw_2d",
                "axis_labels": ["tokens", "experts"],
                "explained_variance": null,
                "points": [
                    {"x": 64.0, "y": 8.0, "backend_index": 0, "backend_name": "triton", "count": 3},
                    {"x": 128.0, "y": 8.0, "backend_index": 1, "backend_name": "cutlass", "count": 1}
                ]
            }]
        }"#,
    )
    .expect("write kernel-input-distribution payload");
    fs::write(
        path.join("reports/kernel_time_share_report.json"),
        r#"{
            "schema_version": 1,
            "available": true,
            "totals": {"overall": {"kernel_time_ms": 10.0}}
        }"#,
    )
    .expect("write kernel-time-share report");
    fs::write(
        path.join("payloads/kernel_time_share_composition.json"),
        r#"{
            "schema_version": 1,
            "available": true,
            "overall": {
                "kernel_time_ms": 10.0,
                "segments": [{
                    "position": "attn.decode",
                    "kind": "flashinfer_attn_decode",
                    "kernel_time_ms": 10.0,
                    "share_pct": 100.0
                }]
            },
            "pools": [],
            "workers": [],
            "positions": [],
            "definitions": {
                "tree_attribution": "critical path with Sum/Scale/Max/overlap"
            }
        }"#,
    )
    .expect("write kernel-time-share payload");
    fs::write(
        path.join("reports/optimality_report.json"),
        r#"{
            "schema_version": 1,
            "available": true,
            "optimality_ratio": 0.33,
            "unit": "gpu_seconds",
            "cluster": {"real": 96.0, "hardware_limit": 31.8}
        }"#,
    )
    .expect("write optimality report");
    fs::write(
        path.join("payloads/optimality_waterfall.json"),
        r#"{
            "schema_version": 1,
            "available": true,
            "unit": "gpu_seconds",
            "optimality_ratio": 0.33,
            "bucket_keys": ["idle", "imbalance", "batching", "communication", "hardware_gap", "hardware_optimal"],
            "levels": [{
                "level": "cluster", "key": "cluster", "label": "Cluster", "total": 96.0,
                "buckets": {"idle": 29.0, "imbalance": 0.0, "batching": 15.6,
                            "communication": 1.8, "hardware_gap": 17.8, "hardware_optimal": 31.8},
                "optimality_ratio": 0.33
            }],
            "kernels": [{
                "name": "afd.attn.decode", "kind": "flashinfer_attn_decode", "is_comm": false,
                "real": 45.7,
                "buckets": {"batching": 4.3, "communication": 0.0, "hardware_gap": 16.7, "hardware_optimal": 24.7}
            }]
        }"#,
    )
    .expect("write optimality payload");
    fs::write(
        path.join("reports/workload_conservation_report.json"),
        r#"{
            "schema_version": 1,
            "available": true,
            "all_ok": true,
            "checks": [{
                "name": "prefill_tokens",
                "description": "actual prefill tokens equal expected request tokens",
                "actual": 42.0,
                "expected": 42.0,
                "delta": 0.0,
                "delta_pct": 0.0,
                "status": "OK"
            }]
        }"#,
    )
    .expect("write workload-conservation report");
    fs::write(
        path.join("payloads/workload_conservation_checks.json"),
        r#"{
            "schema_version": 1,
            "meta": {
                "log_dir": "simulation",
                "available": true,
                "deployment": "afd",
                "mode": "afd-layered",
                "tolerance_pct": 0.01,
                "warn_pct": 5.0,
                "all_ok": true
            },
            "checks": [{
                "name": "prefill_tokens",
                "description": "actual prefill tokens equal expected request tokens",
                "actual": 42.0,
                "expected": 42.0,
                "delta": 0.0,
                "delta_pct": 0.0,
                "status": "OK"
            }]
        }"#,
    )
    .expect("write workload-conservation payload");
    fs::write(
        path.join("reports/kv_occupancy_report.json"),
        r#"{
            "schema_version": 1,
            "available": true,
            "totals": {
                "per_series": [{
                    "pool_tag": "attn",
                    "group_id": 0,
                    "capacity_tokens": 100,
                    "n_workers": 2,
                    "peak_active_mean_tokens": 50
                }]
            }
        }"#,
    )
    .expect("write KV occupancy report");
    fs::write(
        path.join("payloads/kv_occupancy_series.json"),
        r#"{
            "schema_version": 1,
            "meta": {
                "log_dir": "simulation",
                "unit": "KV tokens (per shard); fraction = tokens / capacity_tokens",
                "has_capacity": true
            },
            "t_start_ms": [0.0],
            "t_end_ms": [1000.0],
            "series": [{
                "key": "attn/g0",
                "label": "attn",
                "pool_tag": "attn",
                "group_id": 0,
                "capacity_tokens": 100,
                "n_workers": 2,
                "workers": [
                    {"worker_id": 0, "active_tokens": [60], "projected_tokens": [70], "promised_tokens": [10]},
                    {"worker_id": 1, "active_tokens": [40], "projected_tokens": [50], "promised_tokens": [10]}
                ],
                "active": {"mean": [50], "min": [40], "max": [60]},
                "projected": {"mean": [60], "min": [50], "max": [70]},
                "promised": {"mean": [10], "min": [10], "max": [10]}
            }]
        }"#,
    )
    .expect("write KV occupancy payload");
    fs::write(path.join(".complete"), "").expect("write completion marker");
}

#[test]
fn empty_logs_root_has_empty_protocol_v1_catalog() {
    let temporary = TempDir::new().expect("temporary logs root");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");

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
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");

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

#[test]
fn descriptor_indexes_existing_core_resources() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_core_run(&temporary.path().join("experiment/simulation"));
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let descriptor = build_descriptor(&run).expect("build descriptor");

    assert_eq!(descriptor["run_id"], run.run_id);
    assert_eq!(descriptor["deployment"], "afd");
    assert_eq!(descriptor["model_name"], "model/config/qwen.json");
    assert_eq!(descriptor["model"]["href"], "model");
    assert_eq!(descriptor["model"]["schema_version"], 1);
    assert_eq!(descriptor["summary"]["href"], "summary");
    assert_eq!(descriptor["topology"]["href"], "topology");
    assert_eq!(descriptor["topology"]["schema_version"], 1);
    assert_eq!(descriptor["workload"]["href"], "workload");
    assert_eq!(descriptor["workload"]["schema_version"], 1);
    assert_eq!(descriptor["subjects"]["concurrency"]["status"], "ready");
    assert_eq!(
        descriptor["subjects"]["concurrency"]["payload_href"],
        "subjects/concurrency/payload"
    );
    assert_eq!(descriptor["subjects"]["request-state"]["status"], "ready");
    assert_eq!(
        descriptor["subjects"]["request-state"]["payload_href"],
        "subjects/request-state/payload"
    );
    assert_eq!(descriptor["subjects"]["slo-general"]["status"], "ready");
    assert_eq!(
        descriptor["subjects"]["slo-general"]["report_href"],
        "subjects/slo-general/report"
    );
    assert_eq!(
        descriptor["subjects"]["slo-general"]["payload_href"],
        "subjects/slo-general/payload"
    );
    assert_eq!(descriptor["subjects"]["throughput"]["status"], "ready");
    assert_eq!(
        descriptor["subjects"]["throughput"]["report_href"],
        "subjects/throughput/report"
    );
    assert_eq!(
        descriptor["subjects"]["throughput"]["payload_href"],
        "subjects/throughput/payload"
    );
    assert_eq!(descriptor["subjects"]["utilization"]["status"], "ready");
    assert_eq!(
        descriptor["subjects"]["utilization"]["report_href"],
        "subjects/utilization/report"
    );
    assert_eq!(
        descriptor["subjects"]["utilization"]["payload_href"],
        "subjects/utilization/payload"
    );
    assert_eq!(descriptor["subjects"]["kv-occupancy"]["status"], "ready");
    assert_eq!(
        descriptor["subjects"]["kv-occupancy"]["report_href"],
        "subjects/kv-occupancy/report"
    );
    assert_eq!(
        descriptor["subjects"]["kv-occupancy"]["payload_href"],
        "subjects/kv-occupancy/payload"
    );
    assert_eq!(
        descriptor["subjects"]["kernel-input-distribution"]["status"],
        "ready"
    );
    assert_eq!(
        descriptor["subjects"]["kernel-input-distribution"]["payload_href"],
        "subjects/kernel-input-distribution/payload"
    );
    assert_eq!(
        descriptor["subjects"]["kernel-time-share"]["status"],
        "ready"
    );
    assert_eq!(
        descriptor["subjects"]["kernel-time-share"]["payload_href"],
        "subjects/kernel-time-share/payload"
    );
    assert_eq!(
        descriptor["subjects"]["workload-conservation"]["status"],
        "ready"
    );
    assert_eq!(
        descriptor["subjects"]["workload-conservation"]["payload_href"],
        "subjects/workload-conservation/payload"
    );
    assert!(descriptor["analysis"]["revision"]
        .as_str()
        .is_some_and(|revision| revision.starts_with("legacy-sha256-")));
}

#[test]
fn summary_and_topology_reuse_simulator_artifacts() {
    let temporary = TempDir::new().expect("temporary logs root");
    let run_path = temporary.path().join("simulation");
    make_core_run(&run_path);
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let summary = read_summary(&run).expect("read summary");
    let topology = build_topology(&run).expect("build topology");

    assert_eq!(summary["total_tok_s"], 42.0);
    assert_eq!(summary["num_gpus"], 8);
    assert_eq!(topology["schema_version"], 1);
    assert_eq!(topology["params"]["deployment"], "afd");
    assert_eq!(topology["run_meta"]["workers"], json!([]));
}

#[test]
fn model_resource_reads_repo_config_named_by_run_params() {
    let temporary = TempDir::new().expect("temporary logs root");
    let run_path = temporary.path().join("simulation");
    make_core_run(&run_path);
    let repo = TempDir::new().expect("temporary repository");
    fs::create_dir_all(repo.path().join("model/config")).expect("create model config directory");
    fs::write(
        repo.path().join("model/config/qwen.json"),
        r#"{"hidden_size": 6144, "num_hidden_layers": 62}"#,
    )
    .expect("write model config");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let model = read_model(&run, repo.path()).expect("read model resource");

    assert_eq!(model["schema_version"], 1);
    assert_eq!(model["source_path"], "model/config/qwen.json");
    assert_eq!(model["config"]["hidden_size"], 6144);
    assert_eq!(model["config"]["num_hidden_layers"], 62);
}

#[test]
fn model_resource_rejects_paths_outside_model_config() {
    let temporary = TempDir::new().expect("temporary logs root");
    let run_path = temporary.path().join("simulation");
    make_core_run(&run_path);
    let params_path = run_path.join("raw/params.json");
    let params = fs::read_to_string(&params_path)
        .expect("read params")
        .replace("model/config/qwen.json", "../../outside.json");
    fs::write(params_path, params).expect("write unsafe params");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let error = read_model(&run, temporary.path()).expect_err("reject path traversal");

    assert!(error.to_string().contains("below model/config"));
}

#[test]
fn workload_resource_summarizes_configured_trace() {
    let temporary = TempDir::new().expect("temporary logs root");
    let run_path = temporary.path().join("simulation");
    make_core_run(&run_path);
    let repo = TempDir::new().expect("temporary repository");
    fs::create_dir_all(repo.path().join("trace")).expect("create trace directory");
    fs::write(
        repo.path().join("trace/workload.csv"),
        "id,input_len,output_len,arrival_time\n\
         0,8,32,0\n\
         1,16,64,2000\n\
         2,32,128,4000\n",
    )
    .expect("write trace");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let workload = read_workload(&run, repo.path()).expect("read workload resource");

    assert_eq!(workload["schema_version"], 1);
    assert_eq!(workload["scope"], "configured_trace");
    assert_eq!(workload["source_paths"], json!(["trace/workload.csv"]));
    assert_eq!(workload["request_count"], 3);
    assert!((workload["average_input_tokens"].as_f64().unwrap() - 56.0 / 3.0).abs() < 1e-9);
    assert!((workload["average_output_tokens"].as_f64().unwrap() - 224.0 / 3.0).abs() < 1e-9);
    assert_eq!(workload["arrival_basis"], "effective_open_loop");
    let arrival_seconds = workload["arrival_seconds"]
        .as_array()
        .expect("arrival seconds");
    assert_eq!(arrival_seconds.len(), 3);
    assert!((arrival_seconds[0].as_f64().unwrap() - 1.0 / 3.0).abs() < 1e-9);
    assert_eq!(arrival_seconds[1], 1.0);
    assert!((arrival_seconds[2].as_f64().unwrap() - 5.0 / 3.0).abs() < 1e-9);
    assert_eq!(workload["arrivals"], json!([1, 1, 1]));
    assert_eq!(workload["peak_to_mean"], 1.0);
    assert_eq!(workload["token_lengths"].as_array().map(Vec::len), Some(72));
}

#[test]
fn workload_resource_rejects_paths_outside_trace() {
    let temporary = TempDir::new().expect("temporary logs root");
    let run_path = temporary.path().join("simulation");
    make_core_run(&run_path);
    let params_path = run_path.join("raw/params.json");
    let params = fs::read_to_string(&params_path)
        .expect("read params")
        .replace("trace/workload.csv", "../outside.csv");
    fs::write(params_path, params).expect("write unsafe params");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let error = read_workload(&run, temporary.path()).expect_err("reject path traversal");

    assert!(error.to_string().contains("below trace"));
}

#[test]
fn concurrency_resources_reuse_analyzer_artifacts() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_core_run(&temporary.path().join("simulation"));
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let report = read_concurrency_report(&run).expect("read concurrency report");
    let payload = read_concurrency_payload(&run).expect("read concurrency payload");

    assert_eq!(report["totals"]["peak_active"], 2);
    assert_eq!(payload["t_ms"], json!([5.0, 10.0]));
    assert_eq!(payload["active"], json!([2.0, 1.0]));
    assert_eq!(payload["peak"], 2);
}

#[test]
fn request_state_resources_reuse_hierarchical_analyzer_artifacts() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_core_run(&temporary.path().join("simulation"));
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let report = read_request_state_report(&run).expect("read request-state report");
    let payload = read_request_state_payload(&run).expect("read request-state payload");

    assert_eq!(report["totals"]["cluster_categories"][0]["peak"], 2);
    assert_eq!(payload["cluster_series"][0]["category"], "pending");
    assert_eq!(payload["cluster_series"][1]["values"], json!([2.0, 1.0]));
}

#[test]
fn unavailable_request_state_is_not_published_as_ready() {
    let temporary = TempDir::new().expect("temporary logs root");
    let run_path = temporary.path().join("simulation");
    make_core_run(&run_path);
    fs::write(
        run_path.join("reports/request_state_report.json"),
        r#"{"schema_version":1,"available":false,"reason":"stage logging disabled"}"#,
    )
    .expect("write unavailable request-state report");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let descriptor = build_descriptor(&run).expect("build descriptor");

    assert!(descriptor["subjects"].get("request-state").is_none());
}

#[test]
fn slo_general_resources_expose_ttft_tpot_and_e2e() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_core_run(&temporary.path().join("simulation"));
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let report = read_slo_general_report(&run).expect("read SLO report");
    let payload = read_slo_general_payload(&run).expect("read SLO payload");

    assert_eq!(report["metrics"]["ttft"]["mean"], 12.0);
    assert_eq!(report["metrics"]["tpot"]["mean"], 4.0);
    assert_eq!(report["metrics"]["e2e"]["mean"], 40.0);
    let series_keys = payload["series"]
        .as_array()
        .expect("SLO series")
        .iter()
        .map(|series| series["key"].as_str().expect("series key"))
        .collect::<Vec<_>>();
    assert_eq!(series_keys, ["ttft", "tpot", "e2e"]);
}

#[test]
fn unavailable_slo_general_is_not_published_as_ready() {
    let temporary = TempDir::new().expect("temporary logs root");
    let run_path = temporary.path().join("simulation");
    make_core_run(&run_path);
    fs::write(
        run_path.join("reports/slo_general_report.json"),
        r#"{"schema_version":1,"available":false,"reason":"request_slo.parquet not found"}"#,
    )
    .expect("write unavailable SLO report");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let descriptor = build_descriptor(&run).expect("build descriptor");

    assert!(descriptor["subjects"].get("slo-general").is_none());
}

#[test]
fn throughput_resources_expose_total_prefill_and_decode_rates() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_core_run(&temporary.path().join("simulation"));
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let report = read_throughput_report(&run).expect("read throughput report");
    let payload = read_throughput_payload(&run).expect("read throughput payload");

    assert_eq!(report["totals"]["total_tps"], 1500.0);
    assert_eq!(report["totals"]["total_tps_per_gpu"], 187.5);
    let series_keys = payload["series"]
        .as_array()
        .expect("throughput series")
        .iter()
        .map(|series| series["key"].as_str().expect("series key"))
        .collect::<Vec<_>>();
    assert_eq!(series_keys, ["total", "prefill", "decode"]);
}

#[test]
fn unavailable_throughput_is_not_published_as_ready() {
    let temporary = TempDir::new().expect("temporary logs root");
    let run_path = temporary.path().join("simulation");
    make_core_run(&run_path);
    fs::write(
        run_path.join("reports/throughput_report.json"),
        r#"{"schema_version":1,"available":false,"reason":"request_state.parquet not found"}"#,
    )
    .expect("write unavailable throughput report");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let descriptor = build_descriptor(&run).expect("build descriptor");

    assert!(descriptor["subjects"].get("throughput").is_none());
}

#[test]
fn utilization_resources_expose_workers_and_pool_average() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_core_run(&temporary.path().join("simulation"));
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let report = read_utilization_report(&run).expect("read utilization report");
    let payload = read_utilization_payload(&run).expect("read utilization payload");

    assert_eq!(report["totals"]["per_pool"][0]["avg_util"], 0.6);
    assert_eq!(report["totals"]["per_worker"][0]["worker_id"], 0);
    assert_eq!(payload["series"][0]["util"], json!([0.6]));
    assert_eq!(payload["worker_series"][0]["util"], json!([0.8]));
    assert_eq!(payload["worker_series"][1]["util"], json!([0.4]));
}

#[test]
fn batch_resources_publish_existing_pool_composition() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_core_run(&temporary.path().join("simulation"));
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let descriptor = build_descriptor(&run).expect("build descriptor");
    let report = read_batch_report(&run).expect("read batch report");
    let payload = read_batch_payload(&run).expect("read batch payload");

    assert_eq!(descriptor["subjects"]["batch"]["status"], "ready");
    assert_eq!(report["available"], true);
    assert_eq!(payload["pools"][0]["pool"], "attn");
    assert_eq!(payload["pools"][0]["series"][0]["values"], json!([8, 16]));
}

#[test]
fn kv_occupancy_resources_expose_workers_and_pool_average() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_core_run(&temporary.path().join("simulation"));
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let report = read_kv_occupancy_report(&run).expect("read KV occupancy report");
    let payload = read_kv_occupancy_payload(&run).expect("read KV occupancy payload");

    assert_eq!(
        report["totals"]["per_series"][0]["peak_active_mean_tokens"],
        50
    );
    assert_eq!(payload["series"][0]["active"]["mean"], json!([50]));
    assert_eq!(
        payload["series"][0]["workers"][0]["active_tokens"],
        json!([60])
    );
    assert_eq!(
        payload["series"][0]["workers"][1]["active_tokens"],
        json!([40])
    );
}

#[test]
fn kernel_time_share_resources_preserve_critical_path_attribution() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_core_run(&temporary.path().join("simulation"));
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let report = read_kernel_time_share_report(&run).expect("read kernel-time-share report");
    let payload = read_kernel_time_share_payload(&run).expect("read kernel-time-share payload");

    assert_eq!(report["totals"]["overall"]["kernel_time_ms"], 10.0);
    assert_eq!(payload["overall"]["segments"][0]["share_pct"], 100.0);
    assert!(payload["definitions"]["tree_attribution"]
        .as_str()
        .is_some_and(|definition| definition.contains("critical path")));
}

#[test]
fn optimality_resources_expose_waterfall_levels_and_kernels() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_core_run(&temporary.path().join("simulation"));
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let descriptor = build_descriptor(&run).expect("descriptor");
    assert_eq!(descriptor["subjects"]["optimality"]["status"], "ready");
    assert_eq!(
        descriptor["subjects"]["optimality"]["payload_href"],
        "subjects/optimality/payload"
    );
    let report = read_optimality_report(&run).expect("read optimality report");
    let payload = read_optimality_payload(&run).expect("read optimality payload");
    assert_eq!(report["optimality_ratio"], 0.33);
    // The cluster waterfall's telescoping buckets sum back to its Real GPU·s.
    let cluster = &payload["levels"][0];
    let sum: f64 = cluster["buckets"]
        .as_object()
        .expect("bucket object")
        .values()
        .map(|value| value.as_f64().unwrap_or(0.0))
        .sum();
    assert!((sum - cluster["total"].as_f64().unwrap()).abs() < 1e-6);
    assert_eq!(payload["kernels"][0]["name"], "afd.attn.decode");
}

#[test]
fn kernel_input_distribution_resources_preserve_backend_selection() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_core_run(&temporary.path().join("simulation"));
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let report =
        read_kernel_input_distribution_report(&run).expect("read kernel-input-distribution report");
    let payload = read_kernel_input_distribution_payload(&run)
        .expect("read kernel-input-distribution payload");

    assert_eq!(report["meta"]["num_positions_plotted"], 1);
    assert_eq!(payload["positions"][0]["candidate_backends"][1], "cutlass");
    assert_eq!(payload["positions"][0]["points"][0]["count"], 3);
}

#[test]
fn unavailable_kernel_input_distribution_remains_a_generated_subject() {
    let temporary = TempDir::new().expect("temporary logs root");
    let run_path = temporary.path().join("simulation");
    make_core_run(&run_path);
    fs::write(
        run_path.join("reports/kernel_input_distribution_report.json"),
        r#"{"schema_version":1,"available":false,"reason":"cost_log has no slot_backend column"}"#,
    )
    .expect("write unavailable kernel-input-distribution report");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let descriptor = build_descriptor(&run).expect("build descriptor");

    assert_eq!(
        descriptor["subjects"]["kernel-input-distribution"]["status"],
        "ready"
    );
}

#[test]
fn workload_conservation_resources_preserve_accounting_checks() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_core_run(&temporary.path().join("simulation"));
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let report =
        read_workload_conservation_report(&run).expect("read workload-conservation report");
    let payload =
        read_workload_conservation_payload(&run).expect("read workload-conservation payload");

    assert_eq!(report["all_ok"], true);
    assert_eq!(payload["meta"]["all_ok"], true);
    assert_eq!(payload["checks"][0]["name"], "prefill_tokens");
    assert_eq!(payload["checks"][0]["status"], "OK");
}

#[test]
fn unavailable_workload_conservation_remains_a_generated_subject() {
    let temporary = TempDir::new().expect("temporary logs root");
    let run_path = temporary.path().join("simulation");
    make_core_run(&run_path);
    fs::write(
        run_path.join("payloads/workload_conservation_checks.json"),
        r#"{
            "schema_version": 1,
            "meta": {
                "log_dir": "simulation",
                "available": false,
                "reason": "request_slo.parquet not found"
            },
            "checks": []
        }"#,
    )
    .expect("write unavailable workload-conservation payload");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let descriptor = build_descriptor(&run).expect("build descriptor");

    assert_eq!(
        descriptor["subjects"]["workload-conservation"]["status"],
        "ready"
    );
}

#[test]
fn unavailable_kv_occupancy_is_not_published_as_ready() {
    let temporary = TempDir::new().expect("temporary logs root");
    let run_path = temporary.path().join("simulation");
    make_core_run(&run_path);
    fs::write(
        run_path.join("reports/kv_occupancy_report.json"),
        r#"{"schema_version":1,"available":false,"reason":"kv_snapshot/ dir not found"}"#,
    )
    .expect("write unavailable KV occupancy report");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let descriptor = build_descriptor(&run).expect("build descriptor");

    assert!(descriptor["subjects"].get("kv-occupancy").is_none());
}

#[test]
fn unavailable_utilization_is_not_published_as_ready() {
    let temporary = TempDir::new().expect("temporary logs root");
    let run_path = temporary.path().join("simulation");
    make_core_run(&run_path);
    fs::write(
        run_path.join("reports/utilization_report.json"),
        r#"{"schema_version":1,"available":false,"reason":"cost_log/ dir not found"}"#,
    )
    .expect("write unavailable utilization report");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let descriptor = build_descriptor(&run).expect("build descriptor");

    assert!(descriptor["subjects"].get("utilization").is_none());
}

#[test]
fn unavailable_concurrency_is_not_published_as_ready() {
    let temporary = TempDir::new().expect("temporary logs root");
    let run_path = temporary.path().join("simulation");
    make_core_run(&run_path);
    fs::write(
        run_path.join("reports/concurrency_report.json"),
        r#"{"schema_version":1,"available":false,"reason":"request_slo.parquet not found"}"#,
    )
    .expect("write unavailable concurrency report");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let descriptor = build_descriptor(&run).expect("build descriptor");

    assert!(descriptor["subjects"].get("concurrency").is_none());
}

#[cfg(unix)]
#[test]
fn does_not_follow_directory_symlinks() {
    use std::os::unix::fs::symlink;

    let root = TempDir::new().expect("temporary logs root");
    let outside = TempDir::new().expect("outside directory");
    make_run(&outside.path().join("escaped-run"), true, true);
    symlink(outside.path(), root.path().join("linked-outside")).expect("create symlink");
    let roots = configure_logs_roots(vec![root.path().to_path_buf()]).expect("configure logs root");

    let catalog = build_catalog(&roots).expect("build catalog");

    assert!(catalog.runs.is_empty());
}
