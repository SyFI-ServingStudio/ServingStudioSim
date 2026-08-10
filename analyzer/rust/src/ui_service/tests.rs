use std::fs;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int16Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use axum::Router;
use parquet::arrow::ArrowWriter;
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt;

use super::alignment::{
    alignment_descriptor, alignment_iteration_detail, alignment_payload, alignment_report,
    build_alignment_catalog, discover_alignments, resolve_alignment,
};
use super::batch::{read_batch_payload, read_batch_report};
use super::catalog::build_catalog;
use super::concurrency::{read_concurrency_payload, read_concurrency_report};
use super::core::{build_descriptor, read_summary};
use super::discovery::{configure_logs_roots, configure_workspace_registry, discover_runs};
use super::hardware::{hardware_gpu_response, resolve_gpu};
use super::kernel_input_distribution::{
    read_kernel_input_distribution_payload, read_kernel_input_distribution_report,
};
use super::kernel_measurement::{
    build_kernel_measurement_catalog, discover_kernel_measurements, measurement_descriptor,
    measurement_plot, measurement_summary, resolve_kernel_measurement,
};
use super::kernel_profile::{
    build_kernel_profile_catalog, discover_kernel_profiles, profile_curve, profile_descriptor,
    resolve_kernel_profile,
};
use super::kernel_time_share::{read_kernel_time_share_payload, read_kernel_time_share_report};
use super::kv_occupancy::{read_kv_occupancy_payload, read_kv_occupancy_report};
use super::model::read_model;
use super::optimality::{
    read_locked_optimality_payload, read_locked_optimality_report, read_optimality_payload,
    read_optimality_report,
};
use super::prediction::{
    build_prediction_catalog, discover_predictions, prediction_descriptor, resolve_prediction,
};
use super::request_state::{read_request_state_payload, read_request_state_report};
use super::slo::{read_slo_general_payload, read_slo_general_report};
use super::sweep::{build_sweep_catalog, read_sweep_payload, resolve_sweep};
use super::throughput::{read_throughput_payload, read_throughput_report};
use super::topology::build_topology;
use super::utilization::{read_utilization_payload, read_utilization_report};
use super::workload::read_workload;
use super::workload_conservation::{
    read_workload_conservation_payload, read_workload_conservation_report,
};
use super::{
    service_router, timeline_profile_log_line, OperationIndexCache, RootSource, ServiceState,
    TimelineProfileEvent,
};

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

fn make_sweep(path: &Path, with_payload: bool) {
    fs::create_dir_all(path).expect("create sweep directory");
    fs::write(
        path.join("sweep_manifest.json"),
        r#"{
            "schema_version": 1,
            "axes": ["request_rate", "tensor_parallel"],
            "runs": [{
                "path": "rate10/tp2",
                "coordinates": {"request_rate": 10.0, "tensor_parallel": 2},
                "labels": {}
            }]
        }"#,
    )
    .expect("write sweep manifest");
    make_run(&path.join("rate10/tp2"), true, true);
    fs::write(
        path.join("rate10/tp2/raw/params.json"),
        r#"{
            "deployment": "unified",
            "workload": {"trace_files": ["logs/sweep/trace/aime_long.csv"]}
        }"#,
    )
    .expect("write sweep member params");
    if with_payload {
        fs::create_dir_all(path.join("payloads")).expect("create sweep payload directory");
        fs::write(
            path.join("payloads/sweep_metrics_grid.json"),
            r#"{
                "schema_version": 1,
                "meta": {
                    "experiment_dir": "/private/logs/sweep",
                    "num_axes": 2,
                    "num_runs": 1
                },
                "axes": ["request_rate", "tensor_parallel"],
                "domains": {
                    "request_rate": [10.0],
                    "tensor_parallel": [2]
                },
                "metrics": [{
                    "group": "throughput",
                    "key": "total_tps",
                    "label": "Total throughput",
                    "unit": "tok/s"
                }],
                "runs": [{
                    "path": "rate10/tp2",
                    "coordinates": {"request_rate": 10.0, "tensor_parallel": 2},
                    "labels": {},
                    "lifecycle": {"simulation": "complete", "analysis": "complete"},
                    "metrics": {"total_tps": 42.0}
                }],
                "definitions": {}
            }"#,
        )
        .expect("write sweep payload");
    }
}

fn make_prediction(path: &Path, prediction_id: &str) {
    fs::create_dir_all(path).expect("create prediction directory");
    fs::write(path.join("predict.json"), "{}").expect("write prediction config snapshot");
    fs::write(
        path.join("prediction.cases.json"),
        r#"[{"groups":[{"decode_count":8,"average_decode_length":128}]}]"#,
    )
    .expect("write normalized prediction cases");
    fs::write(
        path.join("prediction.meta.json"),
        serde_json::to_vec_pretty(&json!({
            "schema_version": 1,
            "prediction_id": prediction_id,
            "selector": "iter",
            "arch_type": "llama3_dense",
            "gpu": "NVIDIA H200",
            "gpu_count": 1,
            "config_file": "predict.json",
            "cases_file": "prediction.cases.json",
            "case_count": 1,
        }))
        .expect("serialize prediction metadata"),
    )
    .expect("write prediction metadata");
}

fn make_prediction_cost_source(path: &Path) {
    let cost_log_directory = path.join("raw/cost_log");
    let cost_manifest_directory = path.join("raw/cost_manifest");
    fs::create_dir_all(&cost_log_directory).expect("create prediction cost-log directory");
    fs::create_dir_all(&cost_manifest_directory)
        .expect("create prediction cost-manifest directory");
    fs::write(
        cost_manifest_directory.join("worker_predict_0.json"),
        r#"{
            "sections": [{
                "section": "iter",
                "slots": [{
                    "name": "post_norm",
                    "kind": "rms_norm",
                    "kernel_config": {"hidden": 4096, "backends": ["flashinfer"]}
                }],
                "nodes": [{"Leaf": 0}],
                "node_labels": [null]
            }]
        }"#,
    )
    .expect("write prediction cost manifest");

    let schema = Arc::new(Schema::new(vec![
        Field::new("iter_id", DataType::UInt64, false),
        Field::new("batch_id", DataType::UInt64, false),
        Field::new("section", DataType::Utf8, false),
        Field::new("layer", DataType::Int16, false),
        Field::new("wall_start_ms", DataType::Float64, false),
        Field::new("total_time_ms", DataType::Float64, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from(vec![0])),
        Arc::new(UInt64Array::from(vec![0])),
        Arc::new(StringArray::from(vec!["iter"])),
        Arc::new(Int16Array::from(vec![-1])),
        Arc::new(Float64Array::from(vec![0.0])),
        Arc::new(Float64Array::from(vec![4.75])),
    ];
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)
        .expect("build prediction cost-log batch");
    let file = fs::File::create(cost_log_directory.join("worker_predict_0.parquet"))
        .expect("create prediction cost-log parquet");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("create parquet writer");
    writer.write(&batch).expect("write prediction cost-log row");
    writer.close().expect("close prediction cost-log parquet");
}

fn write_gpu_spec_fixture(repo_root: &Path) {
    fs::create_dir_all(repo_root.join("gpu")).expect("create gpu catalog directory");
    fs::write(
        repo_root.join("gpu/spec.json"),
        r#"{"gpus": [
            {
                "name": "H200-SXM-141GB",
                "aliases": ["NVIDIA H200", "H200", "H200-SXM"],
                "mem_bandwidth_gbps": 4800,
                "fp16_tflops": 990,
                "bf16_tflops": 990,
                "fp8_tflops": 1979,
                "fp32_tflops": 67,
                "int8_tops": 1979,
                "interconnect": "NVLink 4.0",
                "interconnect_bandwidth_gbps": 900,
                "nvl_domain_size": 8
            },
            {
                "name": "H100-SXM5-80GB",
                "aliases": ["NVIDIA H100", "H100"],
                "mem_bandwidth_gbps": 3350,
                "fp16_tflops": 989,
                "bf16_tflops": 989,
                "fp8_tflops": 1979,
                "fp32_tflops": 67,
                "int8_tops": 1979,
                "interconnect": "NVLink 4.0",
                "interconnect_bandwidth_gbps": 900,
                "nvl_domain_size": 8
            }
        ]}"#,
    )
    .expect("write gpu spec fixture");
}

fn make_curve_file(path: &Path, kind: &str, family: &str, dtypes: &[&str]) {
    let series_comm = r#"[
        {"metric": "time_ms", "unit": "ms", "lowerIsBetter": true},
        {"metric": "algbw_gbps", "unit": "GB/s", "lowerIsBetter": false},
        {"metric": "busbw_gbps", "unit": "GB/s", "lowerIsBetter": false},
        {"metric": "energy_j", "unit": "J", "lowerIsBetter": true}
    ]"#;
    let series_compute = r#"[
        {"metric": "time_ms", "unit": "ms", "lowerIsBetter": true},
        {"metric": "tflops", "unit": "TFLOP/s", "lowerIsBetter": false},
        {"metric": "memory_bandwidth_gbps", "unit": "GB/s", "lowerIsBetter": false},
        {"metric": "energy_j", "unit": "J", "lowerIsBetter": true}
    ]"#;
    let rows = dtypes
        .iter()
        .enumerate()
        .map(|(index, dtype)| {
            json!({
                "index": index,
                "coordinates": {"m": index as u64 + 1},
                "args": {"m": index as u64 + 1, "k": 64, "dtype": dtype, "backend": "torch"},
                "status": "ok",
                "metrics": {"time_ms": 1.0, "tflops": 2.0, "memory_bandwidth_gbps": 3.0, "energy_j": 0.1}
            })
        })
        .collect::<Vec<_>>();
    fs::create_dir_all(path).expect("create curve directory");
    fs::write(
        path.join("curve.json"),
        serde_json::to_vec(&json!({
            "schemaVersion": 1,
            "resourceKind": "kernel_profile_curve",
            "kernelKind": kind,
            "table": kind,
            "backend": "torch",
            "metricFamily": family,
            "axes": [{"key": "m", "values": [1, 2]}],
            "fixedArgs": {"k": 64},
            "layout": {"xAxis": "m", "yAxis": null, "facets": []},
            "series": serde_json::from_str::<Value>(if family == "comm" { series_comm } else { series_compute })
                .expect("series"),
            "rows": rows,
        }))
        .expect("serialize curve"),
    )
    .expect("write curve");
}

fn make_kernel_profile(path: &Path, profile_id: &str, observed: Option<&str>) {
    fs::create_dir_all(path).expect("create kernel profile directory");
    make_curve_file(path, "single_gemm", "compute", &["bf16", "fp16"]);
    fs::write(
        path.join("job.meta.json"),
        serde_json::to_vec(&json!({
            "schemaVersion": 1,
            "jobKind": "kernel_profile",
            "resourceId": profile_id,
            "descriptor": {"table": "single_gemm", "kernelKind": "single_gemm", "backend": "torch", "metricFamily": "compute"},
            "origin": {"kind": "development"}
        }))
        .expect("serialize profile job metadata"),
    )
    .expect("write profile job metadata");
    fs::write(
        path.join("kernel-profile.meta.json"),
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "profile_id": profile_id,
            "kernel": {"kind": "single_gemm", "table": "single_gemm", "backend": "torch", "metric_family": "compute"},
            "gpu": {"cache_key": "NVIDIA H200", "observed_name": observed, "count": 1},
            "provenance": {"source": if observed.is_some() { "measurement" } else { "cache_key" }, "resolved_canonical_name": "H200-SXM-141GB"},
            "mode": "force-refresh",
            "args": [{"m": 1, "k": 64, "dtype": "bf16"}],
            "created_at": "2026-07-02T00:00:00Z",
            "artifacts": {"request": "request.json", "results": "results.json", "curve": "curve.json", "job_metadata": "job.meta.json"}
        }))
        .expect("serialize profile metadata"),
    )
    .expect("write profile metadata");
}

fn make_legacy_kernel_profile(path: &Path) {
    fs::create_dir_all(path).expect("create legacy kernel profile directory");
    make_curve_file(path, "single_gemm", "compute", &["bf16"]);
    fs::write(
        path.join("job.meta.json"),
        serde_json::to_vec(&json!({
            "schemaVersion": 1,
            "jobKind": "kernel_profile",
            "descriptor": {"table": "single_gemm", "kernelKind": "single_gemm", "backend": "torch", "metricFamily": "compute"},
            "origin": {"kind": "development"}
        }))
        .expect("serialize legacy job metadata"),
    )
    .expect("write legacy job metadata");
}

fn valid_summary_json() -> &'static str {
    r#"{
        "schema_version": 1,
        "measurement": "one continuous CUPTI activity window",
        "label": "single_gemm:torch",
        "shape": {},
        "metadata": {},
        "runtime_ms": {"mean": 1.0, "median": 1.0, "min": 0.5, "max": 1.5, "p10": 0.6, "p90": 1.4, "p99": 1.5, "first_1s_mean": 1.0, "last_1s_mean": 1.0, "linear_slope_ms_per_s": 0.0}
    }"#
}

fn make_kernel_measurement(path: &Path, measurement_id: &str) {
    fs::create_dir_all(path).expect("create kernel measurement directory");
    fs::write(path.join("summary.json"), valid_summary_json()).expect("write measurement summary");
    fs::write(
        path.join("runtimes.csv"),
        "start_s,duration_ms\\n0.0,1.0\\n",
    )
    .expect("write csv");
    let png = [0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    fs::write(path.join("runtime_trend.png"), png).expect("write trend png");
    fs::write(path.join("runtime_telemetry.png"), png).expect("write telemetry png");
    fs::write(
        path.join("kernel-measurement.meta.json"),
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "measurement_id": measurement_id,
            "kernel": {"kind": "single_gemm", "table": "single_gemm", "backend": "torch", "metric_family": "compute"},
            "gpu": {"cache_key": "NVIDIA H200", "observed_name": "NVIDIA H200", "count": 1},
            "shape": {"m": 8, "n": 8, "k": 8},
            "duration_s": 10.0,
            "telemetry": true,
            "created_at": "2026-07-02T00:00:00Z",
            "summary_file": "summary.json",
            "plots": ["runtime_trend.png", "runtime_telemetry.png"],
            "artifacts": ["runtimes.csv", "telemetry.csv", "summary.json", "runtime_trend.png", "runtime_telemetry.png"]
        }))
        .expect("serialize measurement metadata"),
    )
    .expect("write measurement metadata");
}

fn make_legacy_kernel_measurement(path: &Path) {
    fs::create_dir_all(path).expect("create legacy kernel measurement directory");
    fs::write(path.join("summary.json"), valid_summary_json()).expect("write legacy summary");
    fs::write(
        path.join("runtimes.csv"),
        "start_s,duration_ms\\n0.0,1.0\\n",
    )
    .expect("write csv");
    let png = [0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    fs::write(path.join("runtime_trend.png"), png).expect("write trend png");
}

fn prediction_test_router(logs_root: &Path) -> Router {
    let roots = configure_logs_roots(vec![logs_root.to_path_buf()])
        .expect("configure prediction test logs root");
    service_router(ServiceState {
        root_source: Arc::new(RootSource::Static(Arc::new(roots))),
        repo_root: Arc::new(logs_root.to_path_buf()),
        operation_indexes: Arc::new(OperationIndexCache::default()),
        alignment_discovery: Arc::new(std::sync::Mutex::new(None)),
        alignment_detail_indexes: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
    })
}

async fn get_json(router: Router, uri: &str) -> (StatusCode, Value) {
    let response = router
        .oneshot(
            Request::builder()
                .uri(uri)
                .body(Body::empty())
                .expect("build HTTP test request"),
        )
        .await
        .expect("route HTTP test request");
    let status = response.status();
    let body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("read HTTP test response body");
    let value = serde_json::from_slice(&body).expect("decode HTTP test response JSON");
    (status, value)
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
        path.join("reports/optimality_batch_locked_report.json"),
        r#"{
            "schema_version": 1,
            "available": true,
            "meta": {"batch_size_locked": true},
            "optimality_ratio": 0.33,
            "unit": "gpu_seconds"
        }"#,
    )
    .expect("write batch-locked optimality report");
    fs::copy(
        path.join("payloads/optimality_waterfall.json"),
        path.join("payloads/optimality_batch_locked_waterfall.json"),
    )
    .expect("write batch-locked optimality payload");
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
fn prediction_catalog_is_first_class_and_does_not_create_a_run() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_prediction(&temporary.path().join("predict-llama"), "p_test");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");

    let catalog =
        serde_json::to_value(build_prediction_catalog(&roots).expect("build prediction catalog"))
            .expect("serialize prediction catalog");
    assert_eq!(catalog["protocol_version"], 1);
    assert_eq!(catalog["predictions"][0]["prediction_id"], "p_test");
    assert_eq!(catalog["predictions"][0]["kind"], "timing_predict");
    assert_eq!(catalog["predictions"][0]["status"], "pending");
    assert!(discover_runs(&roots).expect("discover runs").is_empty());

    let prediction = resolve_prediction(&roots, "p_test").expect("resolve prediction");
    let descriptor = prediction_descriptor(&prediction);
    assert_eq!(descriptor["arch"]["type"], "llama3_dense");
    assert_eq!(
        descriptor["gpu"],
        json!({"name": "NVIDIA H200", "count": 1})
    );
    assert!(descriptor.get("worker").is_none());
    assert!(descriptor.get("deployment").is_none());
}

#[tokio::test]
async fn prediction_http_routes_publish_catalog_descriptor_cases_and_problem_json() {
    let temporary = TempDir::new().expect("temporary logs root");
    let prediction_path = temporary.path().join("predict-llama");
    make_prediction(&prediction_path, "p_http_test");
    make_prediction_cost_source(&prediction_path);
    let router = prediction_test_router(temporary.path());

    let (catalog_status, catalog) = get_json(router.clone(), "/api/v1/predictions").await;
    assert_eq!(catalog_status, StatusCode::OK);
    assert_eq!(catalog["predictions"][0]["prediction_id"], "p_http_test");
    assert_eq!(catalog["predictions"][0]["status"], "ready");

    let (descriptor_status, descriptor) =
        get_json(router.clone(), "/api/v1/predictions/p_http_test/descriptor").await;
    assert_eq!(descriptor_status, StatusCode::OK);
    assert_eq!(descriptor["kind"], "timing_predict");
    assert_eq!(descriptor["lifecycle"]["prediction"], "complete");
    assert!(descriptor.get("worker").is_none());
    assert!(descriptor.get("deployment").is_none());

    let (cases_status, cases) = get_json(
        router.clone(),
        "/api/v1/predictions/p_http_test/cases?offset=0&limit=10",
    )
    .await;
    assert_eq!(cases_status, StatusCode::OK);
    assert_eq!(cases["prediction_id"], "p_http_test");
    assert_eq!(cases["range"]["total"], 1);
    assert_eq!(cases["cases"][0]["case_id"], "0");
    assert_eq!(cases["cases"][0]["total_time_ms"], 4.75);
    assert_eq!(cases["cases"][0]["operations"][0]["section"], "iter");
    assert_eq!(
        cases["cases"][0]["input"],
        json!({"groups": [{"decode_count": 8, "average_decode_length": 128}]})
    );
    assert!(cases["cases"][0].get("worker").is_none());

    let (missing_status, missing) =
        get_json(router, "/api/v1/predictions/p_absent/descriptor").await;
    assert_eq!(missing_status, StatusCode::NOT_FOUND);
    assert_eq!(missing["code"], "prediction_not_found");
    assert!(missing["detail"]
        .as_str()
        .is_some_and(|detail| !detail.contains(temporary.path().to_string_lossy().as_ref())));
}

#[test]
fn prediction_discovery_rejects_duplicate_ids_and_ignores_old_logs() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_prediction(&temporary.path().join("first"), "p_duplicate");
    make_prediction(&temporary.path().join("second"), "p_duplicate");
    make_prediction(&temporary.path().join("old-logs/archived"), "p_archived");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");

    assert!(discover_predictions(&roots).is_err());
}

#[test]
fn prediction_discovery_rejects_noncanonical_resource_ids() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_prediction(&temporary.path().join("empty"), "p_");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    assert!(discover_predictions(&roots).is_err());

    fs::remove_dir_all(temporary.path().join("empty")).expect("remove invalid prediction");
    make_prediction(&temporary.path().join("unicode"), "p_预测");
    assert!(discover_predictions(&roots).is_err());
}

#[test]
fn workspace_registry_ids_are_stable_across_reordering_and_archives() {
    let temporary = TempDir::new().expect("temporary registry root");
    let first_logs = temporary.path().join("first/logs");
    let second_logs = temporary.path().join("second/logs");
    make_run(&first_logs.join("experiment/run"), true, true);
    make_run(&second_logs.join("experiment/run"), true, true);
    let registry_path = temporary.path().join("registry.json");
    let write_registry = |entries: Value| {
        fs::write(
            &registry_path,
            serde_json::to_vec(&json!({
                "schema_version": 1,
                "workspaces": entries,
            }))
            .expect("serialize registry"),
        )
        .expect("write registry");
    };
    write_registry(json!([
        {
            "workspace_id": "w_first",
            "display_name": "First",
            "state": "active",
            "logs_root": "first/logs"
        },
        {
            "workspace_id": "w_second",
            "display_name": "Second",
            "state": "active",
            "logs_root": "second/logs"
        }
    ]));
    let initial_roots = configure_workspace_registry(&registry_path).expect("configure registry");
    let initial_runs = discover_runs(&initial_roots).expect("discover initial runs");
    let initial_ids = initial_runs
        .iter()
        .map(|run| (run.workspace_id.clone(), run.run_id.clone()))
        .collect::<std::collections::HashMap<_, _>>();

    write_registry(json!([
        {
            "workspace_id": "w_second",
            "display_name": "Second",
            "state": "active",
            "logs_root": "second/logs"
        },
        {
            "workspace_id": "w_first",
            "display_name": "First",
            "state": "active",
            "logs_root": "first/logs"
        }
    ]));
    let reordered_roots =
        configure_workspace_registry(&registry_path).expect("reload reordered registry");
    let reordered_runs = discover_runs(&reordered_roots).expect("discover reordered runs");
    let reordered_ids = reordered_runs
        .iter()
        .map(|run| (run.workspace_id.clone(), run.run_id.clone()))
        .collect::<std::collections::HashMap<_, _>>();
    assert_eq!(initial_ids, reordered_ids);

    write_registry(json!([
        {
            "workspace_id": "w_second",
            "display_name": "Second",
            "state": "archived",
            "logs_root": "second/logs"
        },
        {
            "workspace_id": "w_first",
            "display_name": "First",
            "state": "active",
            "logs_root": "first/logs"
        }
    ]));
    let active_roots =
        configure_workspace_registry(&registry_path).expect("reload archived registry");
    let active_runs = discover_runs(&active_roots).expect("discover active runs");
    assert_eq!(active_runs.len(), 1);
    assert_eq!(active_runs[0].workspace_id, "w_first");
}

#[test]
fn workspace_registry_defers_active_workspaces_without_logs() {
    let temporary = TempDir::new().expect("temporary registry root");
    let ready_logs = temporary.path().join("ready/logs");
    let pending_logs = temporary.path().join("pending/logs");
    make_run(&ready_logs.join("experiment/run"), true, true);
    let registry_path = temporary.path().join("registry.json");
    fs::write(
        &registry_path,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "workspaces": [
                {
                    "workspace_id": "w_ready",
                    "display_name": "Ready",
                    "state": "active",
                    "logs_root": "ready/logs"
                },
                {
                    "workspace_id": "w_pending",
                    "display_name": "Pending",
                    "state": "active",
                    "logs_root": "pending/logs"
                }
            ],
        }))
        .expect("serialize registry"),
    )
    .expect("write registry");

    let initial_roots = configure_workspace_registry(&registry_path).expect("configure registry");
    assert_eq!(initial_roots.len(), 1);
    assert_eq!(initial_roots[0].workspace_id(), "w_ready");

    make_run(&pending_logs.join("new-experiment/run"), true, true);
    let reloaded_roots = configure_workspace_registry(&registry_path).expect("reload registry");
    assert_eq!(reloaded_roots.len(), 2);
    assert!(reloaded_roots
        .iter()
        .any(|root| root.workspace_id() == "w_pending"));
}

#[test]
fn discovers_nested_runs_and_ignores_directory_shells() {
    let temporary = TempDir::new().expect("temporary logs root");
    let completed = temporary.path().join("sweep/tp4/simulation");
    let pending = temporary.path().join("sweep/tp8/simulation");
    make_run(&completed, true, true);
    make_run(&pending, false, false);
    make_run(&temporary.path().join("old-logs/archived"), true, true);
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
fn sweep_catalog_preserves_manifest_axes_and_pending_state() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_sweep(&temporary.path().join("ready-sweep"), true);
    make_sweep(&temporary.path().join("pending-sweep"), false);
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");

    let value = serde_json::to_value(build_sweep_catalog(&roots).expect("build sweep catalog"))
        .expect("serialize sweep catalog");

    assert_eq!(value["protocol_version"], 1);
    let sweeps = value["sweeps"].as_array().expect("sweep catalog entries");
    assert_eq!(sweeps.len(), 2);
    let ready = sweeps
        .iter()
        .find(|sweep| sweep["display_name"] == "ready-sweep")
        .expect("ready sweep");
    assert_eq!(ready["axes"], json!(["request_rate", "tensor_parallel"]));
    assert_eq!(ready["num_runs"], 1);
    assert_eq!(ready["status"], "ready");
    assert_eq!(ready["deployments"], json!(["unified"]));
    assert_eq!(ready["traces"], json!(["aime_long.csv"]));
    assert!(ready["sweep_id"]
        .as_str()
        .is_some_and(|sweep_id| sweep_id.starts_with("s_")));
    let pending = sweeps
        .iter()
        .find(|sweep| sweep["display_name"] == "pending-sweep")
        .expect("pending sweep");
    assert_eq!(pending["status"], "pending");
}

#[tokio::test]
async fn sweep_catalog_supports_bounded_ready_discovery_and_latest_alias() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_sweep(&temporary.path().join("20260803_0_pending"), false);
    make_sweep(&temporary.path().join("20260802_0_latest_ready"), true);
    make_sweep(&temporary.path().join("20260801_0_older_ready"), true);
    let router = prediction_test_router(temporary.path());

    let (status, catalog) = get_json(router.clone(), "/api/v1/sweeps?status=ready&limit=1").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(catalog["sweeps"].as_array().map(Vec::len), Some(1));
    assert_eq!(
        catalog["sweeps"][0]["display_name"],
        "20260802_0_latest_ready"
    );
    assert_eq!(catalog["sweeps"][0]["status"], "ready");

    let (status, latest) = get_json(router.clone(), "/api/v1/sweeps/latest").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(latest["sweeps"].as_array().map(Vec::len), Some(1));
    assert_eq!(latest["sweeps"][0], catalog["sweeps"][0]);

    let (status, problem) = get_json(router, "/api/v1/sweeps?limit=0").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(problem["code"], "invalid_sweep_catalog_limit");
}

#[test]
fn sweep_catalog_uses_launcher_experiment_identity_when_present() {
    let temporary = TempDir::new().expect("temporary logs root");
    let sweep_path = temporary.path().join("managed-sweep");
    make_sweep(&sweep_path, true);
    fs::write(
        sweep_path.join("experiment.meta.json"),
        r#"{
            "schema_version": 1,
            "experiment_id": "e_managed_test",
            "origin": {"kind": "managed"}
        }"#,
    )
    .expect("write experiment metadata");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");

    let catalog = serde_json::to_value(build_sweep_catalog(&roots).expect("build sweep catalog"))
        .expect("serialize sweep catalog");

    assert_eq!(catalog["sweeps"][0]["sweep_id"], "e_managed_test");
    let sweep = resolve_sweep(&roots, "e_managed_test").expect("resolve managed identity");
    let payload = read_sweep_payload(&roots, &sweep).expect("read managed sweep");
    assert_eq!(payload["sweep_id"], "e_managed_test");
}

#[test]
fn sweep_catalog_chooses_one_canonical_copy_for_a_reused_experiment_identity() {
    let temporary = TempDir::new().expect("temporary logs root");
    let older_path = temporary.path().join("20260803_0_copied/simulation");
    let newer_path = temporary.path().join("20260805_0_canonical/simulation");
    make_core_run(&older_path);
    make_core_run(&newer_path);
    let shared_metadata = r#"{
        "schema_version": 1,
        "experiment_id": "e_shared_copy",
        "origin": {"kind": "development"}
    }"#;
    fs::write(older_path.join("experiment.meta.json"), shared_metadata)
        .expect("write older experiment metadata");
    fs::write(newer_path.join("experiment.meta.json"), shared_metadata)
        .expect("write newer experiment metadata");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let newer_run_id = discover_runs(&roots)
        .expect("discover copied runs")
        .into_iter()
        .find(|run| run.display_name == "20260805_0_canonical/simulation")
        .expect("newer copied run")
        .run_id;

    let catalog = serde_json::to_value(build_sweep_catalog(&roots).expect("build sweep catalog"))
        .expect("serialize sweep catalog");

    let entries = catalog["sweeps"].as_array().expect("aggregate entries");
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["sweep_id"], "e_shared_copy");
    assert_eq!(
        entries[0]["display_name"],
        "20260805_0_canonical/simulation"
    );
    let sweep = resolve_sweep(&roots, "e_shared_copy").expect("resolve canonical copy");
    let payload = read_sweep_payload(&roots, &sweep).expect("read canonical payload");
    assert_eq!(payload["runs"][0]["run_id"], newer_run_id);
}

#[test]
fn sweep_payload_replaces_filesystem_paths_with_opaque_run_ids() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_sweep(&temporary.path().join("sweep"), true);
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let catalog = serde_json::to_value(build_sweep_catalog(&roots).expect("build sweep catalog"))
        .expect("serialize sweep catalog");
    let sweep_id = catalog["sweeps"][0]["sweep_id"].as_str().expect("sweep id");
    let sweep = resolve_sweep(&roots, sweep_id).expect("resolve sweep");

    let payload = read_sweep_payload(&roots, &sweep).expect("read sweep payload");

    assert_eq!(payload["protocol_version"], 1);
    assert_eq!(payload["sweep_id"], sweep_id);
    assert!(payload["meta"].get("experiment_dir").is_none());
    assert!(payload["runs"][0].get("path").is_none());
    assert!(payload["runs"][0]["run_id"]
        .as_str()
        .is_some_and(|run_id| run_id.starts_with("r_")));
    assert_eq!(payload["runs"][0]["metrics"]["total_tps"], 42.0);
    assert_eq!(payload["metrics"][0]["objective"], "maximize");
}

#[test]
fn sweep_catalog_adds_only_unclaimed_runs_as_singletons() {
    let temporary = TempDir::new().expect("temporary logs root");
    make_sweep(&temporary.path().join("sweep"), true);
    let standalone_path = temporary.path().join("20260725_0_standalone");
    make_core_run(&standalone_path);
    fs::write(
        standalone_path.join("reports/analyzer_timing.json"),
        r#"{"subjects": [{"name": "slo-goodput", "status": "ok"}]}"#,
    )
    .expect("replace timing with a later partial analysis");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let catalog = serde_json::to_value(build_sweep_catalog(&roots).expect("build sweep catalog"))
        .expect("serialize sweep catalog");
    let entries = catalog["sweeps"].as_array().expect("aggregate entries");

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0]["kind"], "singleton");
    assert_eq!(
        entries
            .iter()
            .filter(|entry| entry["kind"] == "sweep")
            .count(),
        1
    );
    let singleton = entries
        .iter()
        .find(|entry| entry["kind"] == "singleton")
        .expect("singleton entry");
    assert_eq!(singleton["display_name"], "20260725_0_standalone");
    assert_eq!(singleton["axes"], json!([]));
    assert_eq!(singleton["num_runs"], 1);
    assert_eq!(singleton["experiment_date"], "2026-07-25");
    assert_eq!(singleton["deployments"], json!(["afd"]));
    assert_eq!(singleton["traces"], json!(["workload.csv"]));

    let singleton_id = singleton["sweep_id"].as_str().expect("singleton id");
    let discovered = resolve_sweep(&roots, singleton_id).expect("resolve singleton");
    let payload = read_sweep_payload(&roots, &discovered).expect("read singleton payload");
    assert_eq!(payload["meta"]["num_axes"], 0);
    assert_eq!(payload["axes"], json!([]));
    assert_eq!(payload["domains"], json!({}));
    assert!(payload["runs"][0]["run_id"]
        .as_str()
        .is_some_and(|run_id| run_id.starts_with("r_")));
    assert_eq!(payload["runs"][0]["metrics"]["tpot_mean_ms"], 4.0);
    assert_eq!(payload["metrics"][0]["objective"], "minimize");
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
    assert_eq!(descriptor["model"]["schema_version"], 2);
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

    assert_eq!(model["schema_version"], 2);
    assert_eq!(model["source_path"], "model/config/qwen.json");
    assert_eq!(model["config"]["hidden_size"], 6144);
    assert_eq!(model["config"]["num_hidden_layers"], 62);
    // The temporary repository intentionally has no model.work package; raw
    // config remains available while optional enrichment degrades to null.
    assert!(model["parameter_counts"].is_null());
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
    assert_eq!(workload["arrival_basis"], "effective_trace_timed");
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
fn workload_arrival_basis_reads_the_arrival_axis_not_the_capacity_cap() {
    let temporary = TempDir::new().expect("temporary logs root");
    let run_path = temporary.path().join("simulation");
    make_core_run(&run_path);
    // A capped run is still trace-timed unless it says otherwise, so the
    // recorded timeline must stay rescaled by request_rate.
    let params = fs::read_to_string(run_path.join("raw/params.json")).expect("read params");
    fs::write(
        run_path.join("raw/params.json"),
        params.replace(
            r#""request_rate": 2.0"#,
            r#""request_rate": 2.0, "max_concurrency": 2, "arrival_mode": "trace_timed""#,
        ),
    )
    .expect("write params");
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

    assert_eq!(workload["arrival_basis"], "effective_trace_timed");
    let arrival_seconds = workload["arrival_seconds"]
        .as_array()
        .expect("arrival seconds");
    assert!((arrival_seconds[1].as_f64().unwrap() - 1.0).abs() < 1e-9);
}

#[test]
fn workload_resource_accepts_launcher_experiment_trace_path() {
    let repository = TempDir::new().expect("temporary repository");
    let logs_root = repository.path().join("logs");
    let run_path = repository.path().join("logs/experiment/rate90.0/tp1");
    make_core_run(&run_path);
    let source_path = "logs/experiment/trace/workload.csv";
    let params_path = run_path.join("raw/params.json");
    let params = fs::read_to_string(&params_path)
        .expect("read params")
        .replace("trace/workload.csv", source_path);
    fs::write(params_path, params).expect("write launcher params");
    fs::create_dir_all(repository.path().join("logs/experiment/trace"))
        .expect("create experiment trace directory");
    fs::write(
        repository.path().join(source_path),
        "id,input_len,output_len,arrival_time\n0,8,32,0\n1,16,64,2000\n",
    )
    .expect("write experiment trace");
    let roots = configure_logs_roots(vec![logs_root.clone()]).expect("configure logs root");
    let run = discover_runs(&roots)
        .expect("discover runs")
        .pop()
        .expect("one run");

    let workload = read_workload(&run, &logs_root).expect("read launcher experiment workload");

    assert_eq!(workload["source_paths"], json!([source_path]));
    assert_eq!(workload["request_count"], 2);
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

    assert!(error.to_string().contains("inside a trace directory"));
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
    let run_path = temporary.path().join("simulation");
    make_core_run(&run_path);
    fs::create_dir_all(run_path.join("cost_log")).expect("create cost log detail root");
    fs::create_dir_all(run_path.join("cost_manifest")).expect("create cost manifest detail root");
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
    assert_eq!(
        descriptor["subjects"]["optimality"]["variants"]["batch_locked"]["payload_href"],
        "subjects/optimality/variants/batch-locked/payload"
    );
    assert_eq!(
        descriptor["details"]["iteration-optimality-kernel-ladder"]["status"],
        "ready"
    );
    assert_eq!(
        descriptor["details"]["iteration-optimality-waterfall"]["status"],
        "ready"
    );
    let report = read_optimality_report(&run).expect("read optimality report");
    let payload = read_optimality_payload(&run).expect("read optimality payload");
    let locked_report =
        read_locked_optimality_report(&run).expect("read batch-locked optimality report");
    let locked_payload =
        read_locked_optimality_payload(&run).expect("read batch-locked optimality payload");
    assert_eq!(report["optimality_ratio"], 0.33);
    assert_eq!(locked_report["optimality_ratio"], 0.33);
    assert_eq!(locked_payload["optimality_ratio"], 0.33);
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

async fn get_bytes(router: Router, uri: &str) -> (StatusCode, axum::body::Bytes) {
    let response = router
        .oneshot(
            Request::builder()
                .uri(uri)
                .body(Body::empty())
                .expect("build HTTP test request"),
        )
        .await
        .expect("route HTTP test request");
    let status = response.status();
    let body = to_bytes(response.into_body(), 1024 * 1024)
        .await
        .expect("read HTTP test response body");
    (status, body)
}

#[test]
fn kernel_profiles_and_measurements_are_first_class_and_do_not_create_runs() {
    let temporary = TempDir::new().expect("temporary logs root");
    let logs = temporary.path();
    make_kernel_profile(&logs.join("profiles/one"), "kp_test", Some("NVIDIA H200"));
    make_kernel_measurement(&logs.join("measure-1"), "km_test");
    let roots = configure_logs_roots(vec![logs.to_path_buf()]).expect("configure logs root");

    let profile_catalog = serde_json::to_value(
        build_kernel_profile_catalog(&roots).expect("build kernel profile catalog"),
    )
    .expect("serialize profile catalog");
    assert_eq!(profile_catalog["protocol_version"], 1);
    assert_eq!(
        profile_catalog["kernel_profiles"][0]["profile_id"],
        "kp_test"
    );
    assert_eq!(
        profile_catalog["kernel_profiles"][0]["kind"],
        "kernel_profile"
    );
    assert_eq!(
        profile_catalog["kernel_profiles"][0]["provenance_source"],
        "measurement"
    );
    assert_eq!(profile_catalog["kernel_profiles"][0]["status"], "ready");
    assert!(discover_runs(&roots).expect("discover runs").is_empty());

    let measurement_catalog = serde_json::to_value(
        build_kernel_measurement_catalog(&roots).expect("build measurement catalog"),
    )
    .expect("serialize measurement catalog");
    assert_eq!(measurement_catalog["protocol_version"], 1);
    assert_eq!(
        measurement_catalog["kernel_measurements"][0]["measurement_id"],
        "km_test"
    );
    assert_eq!(
        measurement_catalog["kernel_measurements"][0]["kind"],
        "kernel_measurement"
    );
}

#[tokio::test]
async fn kernel_profile_http_routes_publish_descriptor_and_enriched_curve() {
    let temporary = TempDir::new().expect("temporary logs root");
    let logs = temporary.path();
    make_kernel_profile(
        &logs.join("profiles/one"),
        "kp_http_test",
        Some("NVIDIA H200"),
    );
    write_gpu_spec_fixture(logs);
    let router = prediction_test_router(logs);

    let (status, catalog) = get_json(router.clone(), "/api/v1/kernel-profiles").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(catalog["kernel_profiles"][0]["profile_id"], "kp_http_test");

    let (status, descriptor) = get_json(
        router.clone(),
        "/api/v1/kernel-profiles/kp_http_test/descriptor",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(descriptor["kind"], "kernel_profile");
    assert_eq!(descriptor["workspace_id"], "root_0");
    assert_eq!(descriptor["kernel"]["metric_family"], "compute");
    assert_eq!(descriptor["gpu"]["cache_key"], "NVIDIA H200");
    assert_eq!(descriptor["gpu"]["observed_name"], "NVIDIA H200");
    assert_eq!(descriptor["gpu_provenance"]["source"], "measurement");
    assert_eq!(descriptor["lifecycle"]["profile"], "complete");
    assert_eq!(
        descriptor["resources"]["curve_href"],
        "kernel-profiles/kp_http_test/curve"
    );

    let (status, curve) = get_json(router, "/api/v1/kernel-profiles/kp_http_test/curve").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(curve["schemaVersion"], 1);
    assert_eq!(curve["hardware"]["matched"], true);
    assert_eq!(curve["hardware"]["gpu"]["canonical_name"], "H200-SXM-141GB");
    assert_eq!(curve["hardware"]["gpu"]["matched_alias"], "NVIDIA H200");
    // Per-row dtype-aware limits: bf16 row reaches the dense 990 TFLOP/s peak.
    assert_eq!(curve["rows"][0]["hardware"]["tflops_limit"]["limit"], 990.0);
    assert_eq!(curve["rows"][1]["hardware"]["tflops_limit"]["limit"], 990.0);
    // HBM H200 4.8 TB/s; time/energy have no theoretical line.
    assert_eq!(
        curve["rows"][0]["hardware"]["memory_bandwidth_gbps_limit"]["limit"],
        4800.0
    );
    assert_eq!(
        curve["hardware"]["metrics"]["memory_bandwidth_gbps"]["limit"],
        4800.0
    );
    assert_eq!(
        curve["hardware"]["metrics"]["time_ms"]["reason"],
        "no_theoretical_line"
    );
    assert_eq!(curve["rows"][0]["status"], "ok");
}

#[tokio::test]
async fn kernel_measurement_http_routes_publish_descriptor_summary_and_plot() {
    let temporary = TempDir::new().expect("temporary logs root");
    let logs = temporary.path();
    make_kernel_measurement(&logs.join("measure-1"), "km_http_test");
    let router = prediction_test_router(logs);

    let (status, descriptor) = get_json(
        router.clone(),
        "/api/v1/kernel-measurements/km_http_test/descriptor",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(descriptor["kind"], "kernel_measurement");
    assert_eq!(descriptor["workspace_id"], "root_0");
    assert_eq!(descriptor["gpu"]["observed_name"], "NVIDIA H200");
    assert_eq!(descriptor["gpu_provenance"]["source"], "measurement");
    assert_eq!(descriptor["duration_s"], 10.0);
    assert_eq!(descriptor["shape"]["k"], 8);
    assert_eq!(
        descriptor["resources"]["summary_href"],
        "kernel-measurements/km_http_test/summary"
    );
    let plots = descriptor["resources"]["plots"].as_array().expect("plots");
    assert_eq!(plots.len(), 2);
    assert!(plots
        .iter()
        .any(|p| p.as_str() == Some("kernel-measurements/km_http_test/plots/runtime_trend.png")));

    let (status, summary) = get_json(
        router.clone(),
        "/api/v1/kernel-measurements/km_http_test/summary",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(summary["schema_version"], 1);
    assert_eq!(summary["runtime_ms"]["median"], 1.0);

    let (status, bytes) = get_bytes(
        router.clone(),
        "/api/v1/kernel-measurements/km_http_test/plots/runtime_trend.png",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        bytes.as_ref(),
        &[0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    );

    // A traversal-shaped request never resolves to OS files (route mismatch → 4xx).
    let (status, bytes) = get_bytes(
        router.clone(),
        "/api/v1/kernel-measurements/km_http_test/plots/../curve.json",
    )
    .await;
    assert_ne!(status, StatusCode::OK);
    let _ = bytes;
    // Undeclared names are rejected with the artifact_missing problem body.
    let (status, error) = get_json(
        router,
        "/api/v1/kernel-measurements/km_http_test/plots/summary.json",
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error["code"], "artifact_missing");
    let _ = error;
}

#[tokio::test]
async fn kernel_measurement_plot_path_rejects_traversal_and_undeclared_files() {
    let temporary = TempDir::new().expect("temporary logs root");
    let logs = temporary.path();
    make_kernel_measurement(&logs.join("measure-1"), "km_traverse");
    let roots = configure_logs_roots(vec![logs.to_path_buf()]).expect("configure logs root");
    let measurement = resolve_kernel_measurement(&roots, "km_traverse").expect("resolve");
    let png = [0x89u8, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

    // Exact declared basenames serve the declared image bytes.
    assert_eq!(
        measurement_plot(&measurement, "runtime_trend.png").unwrap(),
        png
    );
    // `..` escapes, sub-paths, and undeclared files are all rejected.
    assert!(measurement_plot(&measurement, "../kernel-measurement.meta.json").is_err());
    assert!(measurement_plot(&measurement, "sub/curve.json").is_err());
    assert!(measurement_plot(&measurement, "summary.json").is_err());
    assert!(measurement_plot(&measurement, "runtimes.csv").is_err());
    assert!(measurement_plot(&measurement, ".hidden").is_err());
    assert!(measurement_plot(&measurement, "").is_err());
}

#[test]
fn kernel_discovery_is_workspace_aware_and_ignores_old_logs() {
    let temporary = TempDir::new().expect("temporary registry root");
    let first_logs = temporary.path().join("first/logs");
    let second_logs = temporary.path().join("second/logs");
    make_kernel_profile(
        &first_logs.join("experiment/one"),
        "kp_w1",
        Some("NVIDIA H200"),
    );
    make_kernel_profile(
        &first_logs.join("old-logs/archived"),
        "kp_archive",
        Some("NVIDIA H200"),
    );
    make_kernel_measurement(&second_logs.join("measurement-2"), "km_w2");
    let registry_path = temporary.path().join("registry.json");
    fs::write(
        &registry_path,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "workspaces": [
                {"workspace_id": "w_first", "display_name": "First", "state": "active", "logs_root": "first/logs"},
                {"workspace_id": "w_second", "display_name": "Second", "state": "active", "logs_root": "second/logs"}
            ]
        }))
        .expect("serialize registry"),
    )
    .expect("write registry");
    let roots = configure_workspace_registry(&registry_path).expect("configure registry");

    let profile_catalog = build_kernel_profile_catalog(&roots).expect("build catalogs");
    assert_eq!(profile_catalog.kernel_profiles.len(), 1);
    assert_eq!(profile_catalog.kernel_profiles[0].profile_id, "kp_w1");
    assert_eq!(profile_catalog.kernel_profiles[0].workspace_id, "w_first");

    let measurement_catalog =
        build_kernel_measurement_catalog(&roots).expect("build measurement catalog");
    assert_eq!(measurement_catalog.kernel_measurements.len(), 1);
    assert_eq!(
        measurement_catalog.kernel_measurements[0].workspace_id,
        "w_second"
    );
    // The old-logs archive boundary never surfaces as a kernel resource.
    assert!(!profile_catalog
        .kernel_profiles
        .iter()
        .any(|entry| entry.display_name.contains("old-logs")));
}

#[test]
fn kernel_discovery_rejects_duplicate_and_invalid_ids() {
    let temporary = TempDir::new().expect("temporary logs root");
    let logs = temporary.path();
    make_kernel_profile(&logs.join("first"), "kp_dup", None);
    make_kernel_profile(&logs.join("second"), "kp_dup", None);
    let roots = configure_logs_roots(vec![logs.to_path_buf()]).expect("configure logs root");
    assert!(discover_kernel_profiles(&roots).is_err());

    fs::remove_dir_all(logs.join("first")).expect("remove first");
    fs::remove_dir_all(logs.join("second")).expect("remove second");
    fs::create_dir_all(logs.join("invalid")).expect("create invalid");
    fs::write(
        logs.join("invalid/kernel-profile.meta.json"),
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "profile_id": "not-an-id",
            "kernel": {"kind": "single_gemm", "table": "single_gemm", "backend": "torch", "metric_family": "compute"},
            "gpu": {"cache_key": "NVIDIA H200", "observed_name": null, "count": 1},
            "provenance": {"source": "cache_key", "resolved_canonical_name": "H200-SXM-141GB"}
        }))
        .expect("serialize invalid metadata"),
    )
    .expect("write invalid metadata");
    assert!(discover_kernel_profiles(&roots).is_err());

    fs::remove_dir_all(logs.join("invalid")).expect("remove invalid");

    // Legacy ids are stable hashes of workspace + relative path.
    let legacy_a = logs.join("legacy");
    make_legacy_kernel_profile(&legacy_a);
    let roots = configure_logs_roots(vec![logs.to_path_buf()]).expect("configure logs root");
    let found = resolve_kernel_profile(&roots, "kp_absent").unwrap_err();
    assert!(found.to_string().contains("not found"));
    let resolved = resolve_kernel_profile(
        &roots,
        &discover_kernel_profiles(&roots).unwrap()[0].profile_id,
    )
    .expect("resolve legacy");
    assert!(resolved.profile_id.starts_with("kp_legacy_"));
    assert!(resolved.legacy);
}

#[test]
fn kernel_measurement_discovery_rejects_duplicate_and_incomplete_legacy() {
    let temporary = TempDir::new().expect("temporary logs root");
    let logs = temporary.path();
    make_kernel_measurement(&logs.join("first"), "km_dup");
    make_kernel_measurement(&logs.join("second"), "km_dup");
    let roots = configure_logs_roots(vec![logs.to_path_buf()]).expect("configure logs root");
    assert!(discover_kernel_measurements(&roots).is_err());

    fs::remove_dir_all(logs.join("first")).expect("remove first");
    fs::remove_dir_all(logs.join("second")).expect("remove second");
    // An empty/incomplete summary has no verifiable signature → not discovered.
    fs::create_dir_all(logs.join("incomplete")).expect("create dir");
    fs::write(logs.join("incomplete/summary.json"), "{}").expect("write incomplete summary");
    let roots = configure_logs_roots(vec![logs.to_path_buf()]).expect("configure logs root");
    assert!(discover_kernel_measurements(&roots)
        .expect("discover")
        .is_empty());

    make_legacy_kernel_measurement(&logs.join("legacy-measure"));
    let roots = configure_logs_roots(vec![logs.to_path_buf()]).expect("configure logs root");
    let measurements = discover_kernel_measurements(&roots).expect("discover legacy");
    assert_eq!(measurements.len(), 1);
    assert!(measurements[0].legacy);
    assert!(measurements[0].measurement_id.starts_with("km_legacy_"));
    let descriptor = measurement_descriptor(&measurements[0]);
    assert_eq!(descriptor["gpu_provenance"]["source"], "unavailable");
    assert!(
        descriptor["resources"]["plots"]
            .as_array()
            .expect("plots")
            .len()
            == 1
    );
}

#[test]
fn measurement_descriptor_is_pending_when_declared_summary_is_absent() {
    let temporary = TempDir::new().expect("temporary logs root");
    let logs = temporary.path();
    let dir = logs.join("measure-1");
    make_kernel_measurement(&dir, "km_no_summary");
    fs::remove_file(dir.join("summary.json")).expect("remove declared summary");
    let roots = configure_logs_roots(vec![logs.to_path_buf()]).expect("configure logs root");
    let measurement = resolve_kernel_measurement(&roots, "km_no_summary").expect("resolve");
    let descriptor = measurement_descriptor(&measurement);
    // A description must never report complete while its declared summary is absent.
    assert_eq!(descriptor["lifecycle"]["measurement"], "pending");
    // Summary serving still fails cleanly for the missing artifact.
    assert!(measurement_summary(&measurement).is_err());
}

#[test]
fn kernel_measurement_metadata_rejects_traversal_artifacts() {
    let temporary = TempDir::new().expect("temporary logs root");
    let logs = temporary.path();
    let dir = logs.join("measure-1");
    make_kernel_measurement(&dir, "km_traverse");
    fs::write(
        dir.join("kernel-measurement.meta.json"),
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "measurement_id": "km_traverse",
            "kernel": {"kind": "single_gemm", "table": "single_gemm", "backend": "torch", "metric_family": "compute"},
            "gpu": {"cache_key": "NVIDIA H200", "observed_name": "NVIDIA H200", "count": 1},
            "duration_s": 10.0,
            "summary_file": "summary.json",
            "plots": ["runtime_trend.png"],
            "artifacts": ["../escaped.png"]
        }))
        .expect("serialize"),
    )
    .expect("write metadata");
    let roots = configure_logs_roots(vec![logs.to_path_buf()]).expect("configure logs root");
    // Validation is lexical (basenames only), so a traversal entry is rejected at
    // discovery without any filesystem membership check.
    assert!(discover_kernel_measurements(&roots).is_err());
}

#[test]
fn kernel_profile_rejects_path_like_artifact_declarations() {
    let temporary = TempDir::new().expect("temporary logs root");
    let logs = temporary.path();
    let dir = logs.join("profiles/one");
    make_kernel_profile(&dir, "kp_traverse", Some("NVIDIA H200"));
    fs::write(
        dir.join("kernel-profile.meta.json"),
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "profile_id": "kp_traverse",
            "kernel": {"kind": "single_gemm", "table": "single_gemm", "backend": "torch", "metric_family": "compute"},
            "gpu": {"cache_key": "NVIDIA H200", "observed_name": "NVIDIA H200", "count": 1},
            "provenance": {"source": "measurement", "resolved_canonical_name": "H200-SXM-141GB"},
            "artifacts": {"curve": "sub/dir/curve.json"}
        }))
        .expect("serialize"),
    )
    .expect("write metadata");
    let roots = configure_logs_roots(vec![logs.to_path_buf()]).expect("configure logs root");
    assert!(discover_kernel_profiles(&roots).is_err());
}

#[test]
fn hardware_api_resolves_aliases_and_reports_unmatched_explicitly() {
    let temporary = TempDir::new().expect("temporary repo");
    write_gpu_spec_fixture(temporary.path());

    let h200 = resolve_gpu(temporary.path(), "NVIDIA H200")
        .expect("resolve")
        .expect("matched");
    assert_eq!(h200.canonical_name, "H200-SXM-141GB");
    assert_eq!(h200.matched_alias, "NVIDIA H200");
    assert_eq!(h200.dense_tflops("bf16"), Some(990.0));
    assert_eq!(h200.dense_tflops("fp8_e4m3"), Some(1979.0));
    assert_eq!(h200.mem_bandwidth_gbps, Some(4800.0));
    assert_eq!(h200.interconnect_bandwidth_gbps, Some(900.0));
    assert_eq!(h200.one_way_gbps(), Some(450.0));
    assert_eq!(h200.dense_tflops("made-up-dtype"), None);

    let response = hardware_gpu_response("NVIDIA H200", Some(&h200));
    assert_eq!(response["matched"], true);
    assert_eq!(response["canonical_name"], "H200-SXM-141GB");
    assert_eq!(response["interconnect"]["bidirectional_gbps"], 900.0);
    assert_eq!(response["interconnect"]["one_way_gbps"], 450.0);
    assert_eq!(response["peaks"]["bf16_tflops"], 990.0);
    assert_eq!(response["hbm_bandwidth_gbps"], 4800.0);

    let unknown = resolve_gpu(temporary.path(), "Totally Made Up GPU").expect("resolve unknown");
    assert!(unknown.is_none());
    let response = hardware_gpu_response("Totally Made Up GPU", unknown.as_ref());
    assert_eq!(response["matched"], false);
    assert_eq!(response["available"], false);
    assert!(response["reason"]
        .as_str()
        .is_some_and(|r| r.contains("unmatched")));
    assert!(response.get("canonical_name").is_none());
    assert!(response.get("peaks").is_none());
    assert!(response.get("hbm_bandwidth_gbps").is_none());
}

#[tokio::test]
async fn hardware_gpus_http_route_resolves_and_requires_name() {
    let temporary = TempDir::new().expect("temporary repo");
    write_gpu_spec_fixture(temporary.path());
    let router = prediction_test_router(temporary.path());

    let (status, value) =
        get_json(router.clone(), "/api/v1/hardware/gpus?name=NVIDIA%20H200").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(value["matched"], true);
    assert_eq!(value["canonical_name"], "H200-SXM-141GB");
    assert_eq!(value["interconnect"]["name"], "NVLink 4.0");
    assert_eq!(value["interconnect"]["bidirectional_gbps"], 900.0);
    assert_eq!(value["interconnect"]["one_way_gbps"], 450.0);
    // H200 BF16 must resolve to dense 990 TFLOP/s; HBM 4800 GB/s.
    assert_eq!(value["peaks"]["bf16_tflops"], 990.0);
    assert_eq!(value["hbm_bandwidth_gbps"], 4800.0);

    let (status, value) = get_json(router.clone(), "/api/v1/hardware/gpus").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(value["code"], "gpu_name_required");
}

#[test]
fn curve_hardware_bindings_follow_dtype_and_interconnect_direction() {
    let temporary = TempDir::new().expect("temporary repo");
    let logs = temporary.path();
    write_gpu_spec_fixture(logs);
    // Computed bf16 row + a made-up dtype row (must have no fake line).
    fs::create_dir_all(logs.join("profiles/compute")).expect("create dir");
    make_curve_file(
        &logs.join("profiles/compute"),
        "single_gemm",
        "compute",
        &["bf16", "made-up"],
    );
    fs::write(
        logs.join("profiles/compute/kernel-profile.meta.json"),
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "profile_id": "kp_limits",
            "kernel": {"kind": "single_gemm", "table": "single_gemm", "backend": "torch", "metric_family": "compute"},
            "gpu": {"cache_key": "H200-SXM-141GB", "observed_name": null, "count": 1},
            "provenance": {"source": "cache_key", "resolved_canonical_name": "H200-SXM-141GB"},
            "mode": "jit-fill"
        }))
        .expect("serialize"),
    )
    .expect("write metadata");
    // P2P collective curve: algbw/busbw use the one-way peak.
    fs::create_dir_all(logs.join("profiles/p2p")).expect("create p2p dir");
    make_curve_file(&logs.join("profiles/p2p"), "p2p_intra", "comm", &[]);
    fs::write(
        logs.join("profiles/p2p/kernel-profile.meta.json"),
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "profile_id": "kp_p2p",
            "kernel": {"kind": "p2p_intra", "table": "p2p_intra", "backend": "torch", "metric_family": "comm"},
            "gpu": {"cache_key": "NVIDIA H200", "observed_name": null, "count": 1},
            "provenance": {"source": "cache_key", "resolved_canonical_name": "H200-SXM-141GB"},
            "mode": "jit-fill"
        }))
        .expect("serialize p2p"),
    )
    .expect("write p2p metadata");
    // Collective curve: busbw uses the bidirectional peak, algbw has no generic line.
    fs::create_dir_all(logs.join("profiles/ar")).expect("create ar dir");
    make_curve_file(&logs.join("profiles/ar"), "all_reduce", "comm", &[]);
    fs::write(
        logs.join("profiles/ar/kernel-profile.meta.json"),
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "profile_id": "kp_ar",
            "kernel": {"kind": "all_reduce", "table": "all_reduce", "backend": "torch", "metric_family": "comm"},
            "gpu": {"cache_key": "NVIDIA H200", "observed_name": null, "count": 1},
            "provenance": {"source": "cache_key", "resolved_canonical_name": "H200-SXM-141GB"},
            "mode": "jit-fill"
        }))
        .expect("serialize ar"),
    )
    .expect("write ar metadata");
    let roots = configure_logs_roots(vec![logs.to_path_buf()]).expect("configure logs root");

    let compute_curve = profile_curve(
        &resolve_kernel_profile(&roots, "kp_limits").expect("resolve"),
        logs,
    )
    .expect("compute curve");
    assert_eq!(compute_curve["hardware"]["matched"], true);
    assert_eq!(
        compute_curve["rows"][0]["hardware"]["tflops_limit"]["limit"],
        990.0
    );
    // Made-up dtype → no fabricated line, explicit reason.
    assert_eq!(
        compute_curve["rows"][1]["hardware"]["tflops_limit"]["available"],
        false
    );
    assert_eq!(
        compute_curve["rows"][1]["hardware"]["tflops_limit"]["reason"],
        "no_peak_for_dtype_or_device"
    );
    assert_eq!(
        compute_curve["rows"][0]["hardware"]["memory_bandwidth_gbps_limit"]["limit"],
        4800.0
    );

    let p2p_curve = profile_curve(
        &resolve_kernel_profile(&roots, "kp_p2p").expect("resolve"),
        logs,
    )
    .expect("p2p curve");
    assert_eq!(
        p2p_curve["hardware"]["metrics"]["algbw_gbps"]["limit"],
        450.0
    );
    assert_eq!(
        p2p_curve["hardware"]["metrics"]["busbw_gbps"]["limit"],
        450.0
    );

    let ar_curve = profile_curve(
        &resolve_kernel_profile(&roots, "kp_ar").expect("resolve"),
        logs,
    )
    .expect("all-reduce curve");
    assert_eq!(
        ar_curve["hardware"]["metrics"]["busbw_gbps"]["limit"],
        900.0
    );
    assert_eq!(
        ar_curve["hardware"]["metrics"]["algbw_gbps"]["available"],
        false
    );
    assert_eq!(
        ar_curve["hardware"]["metrics"]["time_ms"]["available"],
        false
    );
}

#[test]
fn legacy_profile_has_data_but_no_hardware_limits() {
    let temporary = TempDir::new().expect("temporary repo");
    let logs = temporary.path();
    make_legacy_kernel_profile(&logs.join("legacy-profile"));
    let roots = configure_logs_roots(vec![logs.to_path_buf()]).expect("configure logs root");

    let profile = resolve_kernel_profile(
        &roots,
        &discover_kernel_profiles(&roots).expect("discover")[0].profile_id,
    )
    .expect("resolve legacy");
    let descriptor = profile_descriptor(&profile);
    assert_eq!(descriptor["legacy"], true);
    assert_eq!(descriptor["gpu_provenance"]["source"], "unavailable");
    assert!(descriptor["gpu"].is_null());

    let curve = profile_curve(&profile, logs).expect("legacy curve");
    // Data rows survive, but hardware limits are never synthesized.
    assert_eq!(curve["rows"][0]["status"], "ok");
    assert_eq!(curve["hardware"]["available"], false);
    assert_eq!(curve["hardware"]["matched"], false);
    assert_eq!(curve["hardware"]["reason"], "gpu_provenance_unavailable");
    assert_eq!(
        curve["rows"][0]["hardware"]["reason"],
        "gpu_provenance_unavailable"
    );
    assert_eq!(
        curve["rows"][0]["hardware"]["tflops_limit"]["available"],
        false
    );
}

// ---------------------------------------------------------------------------
// alignment bundles
// ---------------------------------------------------------------------------

/// A bundle with the kernel half analysed and the e2e half only configured.
///
/// The two halves are independent on purpose: a capture is routinely aligned at
/// the kernel level long before a simulation exists to compare end-to-end.
fn write_alignment_bundle(root: &Path) -> (Vec<u8>, [usize; 2]) {
    let bundle = root.join("20260720_0_llama3_8b_tp_alignment/tp4/rate32");
    let kernel = bundle.join("analysis_kernel");
    fs::create_dir_all(kernel.join("reports")).expect("kernel reports");
    fs::create_dir_all(kernel.join("payloads")).expect("kernel payloads");
    // Two prediction directories, as a re-analysed capture really has, so the
    // descriptor cannot pass by picking whichever one the layout offers first.
    make_prediction(&bundle.join("timing_predict"), "p_paired");
    make_prediction(&bundle.join("timing_predict_second_pass"), "p_unused");
    fs::write(
        kernel.join("alignment_manifest.json"),
        json!({"predict_log_dir": bundle.join("timing_predict")}).to_string(),
    )
    .expect("kernel manifest");
    fs::write(
        kernel.join("reports/alignment_timeline_report.json"),
        json!({"iterations": [{"iteration_id": 8}]}).to_string(),
    )
    .expect("timeline report");

    let first = json!({"iteration_id": 8, "measured": "eight"}).to_string();
    let second = json!({"iteration_id": 9, "measured": "nine"}).to_string();
    let mut shard = Vec::new();
    let ranges = [[shard.len(), first.len()], {
        shard.extend_from_slice(first.as_bytes());
        shard.push(b'\n');
        [shard.len(), second.len()]
    }];
    shard.extend_from_slice(second.as_bytes());
    shard.push(b'\n');
    fs::write(
        kernel.join("payloads/alignment_timeline_iterations.jsonl"),
        &shard,
    )
    .expect("timeline shard");
    fs::write(
        kernel.join("payloads/alignment_timeline.json"),
        json!({
            "iterations": [{"iteration_id": 8}, {"iteration_id": 9}],
            "iteration_detail": {
                "file": "alignment_timeline_iterations.jsonl",
                "byte_ranges": {"8": ranges[0], "9": ranges[1]},
            },
        })
        .to_string(),
    )
    .expect("timeline payload");

    // Configured but never run: manifest present, no report.
    let e2e = bundle.join("analysis_e2e");
    fs::create_dir_all(&e2e).expect("e2e dir");
    fs::write(e2e.join("alignment_manifest.json"), "{}").expect("e2e manifest");

    // Sibling directories a bundle owns; discovery must not descend into them
    // and find a second bundle.
    fs::create_dir_all(bundle.join("profile/nsys")).expect("profile dir");
    fs::create_dir_all(bundle.join("simulation")).expect("simulation dir");
    (first.into_bytes(), [ranges[0][0], ranges[1][0]])
}

#[test]
fn alignment_catalog_reports_each_analysis_half_separately() {
    let temporary = TempDir::new().expect("temp dir");
    write_alignment_bundle(temporary.path());
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");

    let catalog = serde_json::to_value(build_alignment_catalog(&roots).expect("catalog"))
        .expect("serialize catalog");
    let entries = catalog["alignments"].as_array().expect("entries");

    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["kind"], "alignment");
    assert_eq!(
        entries[0]["display_name"],
        "20260720_0_llama3_8b_tp_alignment/tp4/rate32"
    );
    // A manifest with a report is complete; a manifest without one is pending.
    assert_eq!(entries[0]["kernel_analysis"], "complete");
    assert_eq!(entries[0]["e2e_analysis"], "pending");
    assert!(entries[0]["alignment_id"]
        .as_str()
        .is_some_and(|id| id.starts_with("al_")));
}

#[test]
fn the_bundle_is_the_analysis_parent_and_discovery_stops_there() {
    let temporary = TempDir::new().expect("temp dir");
    write_alignment_bundle(temporary.path());
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");

    let discovered = discover_alignments(&roots).expect("discover");

    assert_eq!(discovered.len(), 1);
    // The bundle owns the capture and the prediction both halves point at, so it
    // is the parent of `analysis_kernel/`, not that directory itself.
    assert!(discovered[0].path.join("analysis_kernel").is_dir());
    assert!(discovered[0].path.join("profile").is_dir());
}

#[test]
fn an_ungenerated_subject_is_named_but_offers_no_href() {
    let temporary = TempDir::new().expect("temp dir");
    write_alignment_bundle(temporary.path());
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let alignment = &discover_alignments(&roots).expect("discover")[0];

    let descriptor = alignment_descriptor(alignment, &roots);

    assert_eq!(descriptor["subjects"]["timeline"]["status"], "ready");
    assert!(descriptor["subjects"]["timeline"]["payload_href"].is_string());
    // Missing is a state, not an absence: the client must be able to say "not
    // generated" rather than infer it from a key that is not there.
    assert_eq!(descriptor["subjects"]["e2e"]["status"], "not_generated");
    assert!(descriptor["subjects"]["e2e"]["payload_href"].is_null());
    // Only the sharded subjects offer a per-iteration href.
    assert!(descriptor["subjects"]["timeline"]["iteration_href"].is_string());
    assert!(descriptor["subjects"]["e2e"]["iteration_href"].is_null());
}

#[test]
fn the_descriptor_names_the_prediction_the_kernel_half_was_paired_against() {
    let temporary = TempDir::new().expect("temp dir");
    write_alignment_bundle(temporary.path());
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let alignment = &discover_alignments(&roots).expect("discover")[0];

    let descriptor = alignment_descriptor(alignment, &roots);

    // The manifest recorded which of the bundle's two prediction directories
    // produced these numbers. Anything that reads the layout instead would be
    // free to answer `p_unused`.
    assert_eq!(
        descriptor["prediction"]["prediction_id"], "p_paired",
        "the descriptor must name the prediction the manifest recorded"
    );
    assert_eq!(
        descriptor["prediction"]["display_name"],
        "20260720_0_llama3_8b_tp_alignment/tp4/rate32/timing_predict"
    );
}

#[test]
fn a_bundle_whose_manifest_names_no_prediction_offers_none() {
    let temporary = TempDir::new().expect("temp dir");
    write_alignment_bundle(temporary.path());
    let bundle = temporary
        .path()
        .join("20260720_0_llama3_8b_tp_alignment/tp4/rate32");
    // As every bundle analysed before the manifest carried the path looks.
    fs::write(bundle.join("analysis_kernel/alignment_manifest.json"), "{}")
        .expect("rewrite kernel manifest");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let alignment = &discover_alignments(&roots).expect("discover")[0];

    let descriptor = alignment_descriptor(alignment, &roots);

    // Null, not absent: the client distinguishes "no prediction to offer" from
    // "this service is too old to say".
    assert!(descriptor.get("prediction").is_some());
    assert!(descriptor["prediction"].is_null());
}

#[test]
fn a_prediction_directory_outside_the_served_roots_is_not_offered() {
    let temporary = TempDir::new().expect("temp dir");
    let elsewhere = TempDir::new().expect("temp dir");
    write_alignment_bundle(temporary.path());
    // A prediction that exists on disk but that no walk of the configured roots
    // would reach. Naming its id would hand the client a link to a 404.
    make_prediction(&elsewhere.path().join("timing_predict"), "p_unreachable");
    let bundle = temporary
        .path()
        .join("20260720_0_llama3_8b_tp_alignment/tp4/rate32");
    fs::write(
        bundle.join("analysis_kernel/alignment_manifest.json"),
        json!({"predict_log_dir": elsewhere.path().join("timing_predict")}).to_string(),
    )
    .expect("rewrite kernel manifest");
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let alignment = &discover_alignments(&roots).expect("discover")[0];

    let descriptor = alignment_descriptor(alignment, &roots);

    assert!(descriptor["prediction"].is_null());
}

#[test]
fn one_iteration_is_read_by_byte_range_not_by_parsing_the_shard() {
    let temporary = TempDir::new().expect("temp dir");
    let (expected_first, offsets) = write_alignment_bundle(temporary.path());
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let alignment = &discover_alignments(&roots).expect("discover")[0];

    let first = alignment_iteration_detail(alignment, "timeline", "8").expect("iteration 8");
    let second = alignment_iteration_detail(alignment, "timeline", "9").expect("iteration 9");

    // Bytes, verbatim — the shard already holds the JSON the client wants.
    assert_eq!(first, expected_first);
    assert_eq!(
        serde_json::from_slice::<Value>(&second).expect("json")["measured"],
        "nine"
    );
    // The second record really starts after the first, so the range is a seek
    // and not a full scan that happened to work.
    assert!(offsets[1] > offsets[0]);
}

#[test]
fn an_unknown_subject_or_iteration_is_a_missing_artifact_not_a_path() {
    let temporary = TempDir::new().expect("temp dir");
    write_alignment_bundle(temporary.path());
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");
    let alignment = &discover_alignments(&roots).expect("discover")[0];

    for subject in ["../../etc/passwd", "nonsense"] {
        assert!(alignment_report(alignment, subject).is_err(), "{subject}");
        assert!(alignment_payload(alignment, subject).is_err(), "{subject}");
    }
    // A subject that exists but was never run, and an iteration the index does
    // not carry, are both "not here" rather than an error about the layout.
    assert!(alignment_report(alignment, "e2e").is_err());
    assert!(alignment_iteration_detail(alignment, "timeline", "999").is_err());
    assert!(alignment_iteration_detail(alignment, "workload", "8").is_err());
}

#[test]
fn an_unknown_alignment_id_resolves_to_not_found() {
    let temporary = TempDir::new().expect("temp dir");
    write_alignment_bundle(temporary.path());
    let roots =
        configure_logs_roots(vec![temporary.path().to_path_buf()]).expect("configure logs root");

    assert!(resolve_alignment(&roots, "al_nope").is_err());
    let known = &discover_alignments(&roots).expect("discover")[0].alignment_id;
    assert!(resolve_alignment(&roots, known).is_ok());
}
