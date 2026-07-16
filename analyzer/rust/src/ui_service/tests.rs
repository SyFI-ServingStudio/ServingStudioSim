use std::fs;
use std::path::Path;

use serde_json::{json, Value};
use tempfile::TempDir;

use super::catalog::build_catalog;
use super::core::{build_descriptor, build_topology, read_summary};
use super::discovery::{configure_logs_roots, discover_runs};

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
            "pools": {
                "attn": {
                    "groups": [{"arch": {"model_config": "model/qwen.json"}}]
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
        r#"{"subjects": []}"#,
    )
    .expect("write timing");
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
    assert_eq!(descriptor["model_name"], "model/qwen.json");
    assert_eq!(descriptor["summary"]["href"], "summary");
    assert_eq!(descriptor["topology"]["href"], "topology");
    assert_eq!(descriptor["topology"]["schema_version"], 1);
    assert_eq!(descriptor["subjects"], json!({}));
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
