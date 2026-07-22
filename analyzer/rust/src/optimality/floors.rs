//! Labeler-derived necessary-work floors — two independent lower bounds that sit
//! BELOW the R5 `hardware_limit` green. R5 is a roofline of the sim's *actual* per
//! -kernel work (weights re-loaded every iteration, activation I/O, sim's attention
//! approximation), so it is not the true floor. Here we aggregate each level's
//! workload (reusing the `workload-conservation` subject's `groups` parser — never
//! the cost tree's `slot_flops`/`slot_bytes`, to stay independent) and hand it to the
//! Python `model.work` labeler, which returns two roofline floors per level in GPU·s:
//!
//! - **necessary** — global roofline `max(ΣFLOPs/peak, Σbytes/bw)` of the run's whole
//!   token workload as one fused mega-forward (weights counted once): the loosest,
//!   truly irreducible floor.
//! - **segmented** — `Σ_seg max(compute, memory)`: per-op serial bound (≥ necessary).
//!
//! Run-level waterfalls use one unlocked mega-batch per level. The same batched
//! request also carries one 10,000× saturated row per worker; those normalized
//! semantic labels feed analyzer-owned worker → pool → cluster kernel ladders.
//! Exact iteration detail uses the observed workload directly in locked mode;
//! unlocked mode replicates its independent batch entries before labeling and
//! normalizes the result back to one iteration. Replication amortizes weights without
//! changing sequence geometry. The labeler is a subprocess
//! (`uv run python -m model.work.floors`); any failure degrades gracefully to the
//! plain R0..R5 ladder (the caller records a caveat).

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{anyhow, Context, Result};
use datafusion::prelude::SessionContext;
use serde_json::{json, Map, Value};

use crate::conservation::workload::{
    collect_iteration_workload, collect_workload_by_worker, WorkloadTotals,
};
use crate::kernel_query::repo_root;

/// One level's two labeler floors, in GPU·seconds. Ordered `necessary ≤ segmented`.
#[derive(Clone, Copy, Default)]
pub(super) struct Floors {
    pub(super) necessary: f64,
    pub(super) segmented: f64,
}

#[derive(Clone, Debug)]
pub(super) struct SemanticWork {
    pub(super) name: String,
    pub(super) flops: f64,
    pub(super) bytes: f64,
}

pub(super) struct IterationLabel {
    pub(super) floors: Floors,
    pub(super) segments: Vec<SemanticWork>,
}

pub(super) struct RunLabels {
    pub(super) floors: FloorsByLevel,
    pub(super) saturated_workers: HashMap<(String, u16), IterationLabel>,
}

/// Per-level floors keyed exactly like `levels.rs`'s level `key`:
/// `"cluster"` | `<pool_tag>` | `"<pool_tag>/<worker_id>"`.
pub(super) type FloorsByLevel = HashMap<String, Floors>;

/// Aggregate the run's workload per level, hand it to the labeler, and return the two
/// floors per level. Errors (SQL, missing labeler, non-zero exit, bad JSON) propagate
/// so the caller can turn them into a caveat + graceful degrade — they never abort the
/// optimality subject.
pub(super) async fn compute_run_labels(
    ctx: &SessionContext,
    log_dir: &Path,
    replication_factor: u32,
) -> Result<RunLabels> {
    if replication_factor == 0 {
        return Err(anyhow!("worker replication factor must be positive"));
    }
    let by_worker = collect_workload_by_worker(ctx).await?;
    if by_worker.is_empty() {
        return Err(anyhow!("no cost_log workload rows to aggregate"));
    }
    let mut levels = rollup_levels(&by_worker);
    let normalization = f64::from(replication_factor);
    for ((pool_tag, worker_id), totals) in &by_worker {
        let mut saturated_totals = totals.clone();
        saturated_totals.scale(normalization);
        levels.insert(saturated_worker_key(pool_tag, *worker_id), saturated_totals);
    }
    let response = run_labeler_json(log_dir, &levels)?;
    let mut floors = parse_response(&response)?;
    let mut saturated_workers = HashMap::new();
    for (pool_tag, worker_id) in by_worker.keys() {
        let key = saturated_worker_key(pool_tag, *worker_id);
        floors.remove(&key);
        saturated_workers.insert(
            (pool_tag.clone(), *worker_id),
            parse_label(&response, &key, normalization)?,
        );
    }
    Ok(RunLabels {
        floors,
        saturated_workers,
    })
}

/// Label one worker iteration after replicating its independent batch entries.
/// `replication_factor=1` is the exact locked batch. A larger factor is the unlocked
/// large-batch counterfactual: additive compute/KV work scales, while model weights
/// remain loaded once inside `model.work`; all output is normalized back to one
/// original iteration before returning.
pub(super) async fn compute_iteration_label(
    ctx: &SessionContext,
    log_dir: &Path,
    pool_tag: &str,
    worker_id: u16,
    iter_id: u64,
    replication_factor: u32,
) -> Result<IterationLabel> {
    if replication_factor == 0 {
        return Err(anyhow!("iteration replication factor must be positive"));
    }
    let mut totals = collect_iteration_workload(ctx, pool_tag, worker_id, iter_id).await?;
    if totals.matmul_tokens <= 0.0 {
        return Err(anyhow!("iteration has no model workload to label"));
    }
    let normalization = f64::from(replication_factor);
    totals.scale(normalization);
    let level_key = format!("{pool_tag}/{worker_id}");
    let levels = HashMap::from([(level_key.clone(), totals)]);
    let response = run_labeler_json(log_dir, &levels)?;
    parse_label(&response, &level_key, normalization)
}

fn saturated_worker_key(pool_tag: &str, worker_id: u16) -> String {
    // `model.work.floors` resolves the model spec from the first slash-delimited
    // component, so this private key remains batchable with ordinary level keys.
    format!("{pool_tag}/__saturated_worker__/{worker_id}")
}

fn parse_label(response: &Value, level_key: &str, normalization: f64) -> Result<IterationLabel> {
    let mut floors = parse_response(response)?
        .remove(level_key)
        .context("labeler output missing semantic floor")?;
    floors.necessary /= normalization;
    floors.segmented /= normalization;
    let level = response
        .get("levels")
        .and_then(Value::as_object)
        .and_then(|levels| levels.get(level_key))
        .context("labeler output missing semantic level")?;
    let segments = level
        .get("segments")
        .and_then(Value::as_array)
        .context("labeler output missing semantic segments")?
        .iter()
        .map(|segment| {
            Ok(SemanticWork {
                name: segment
                    .get("name")
                    .and_then(Value::as_str)
                    .context("semantic segment missing name")?
                    .to_string(),
                flops: segment
                    .get("flops")
                    .and_then(Value::as_f64)
                    .context("semantic segment missing flops")?
                    / normalization,
                bytes: segment
                    .get("bytes")
                    .and_then(Value::as_f64)
                    .context("semantic segment missing bytes")?
                    / normalization,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(IterationLabel { floors, segments })
}

/// Roll the per-worker totals up into the three level granularities the waterfall
/// draws. All `WorkloadTotals` fields are additive, so pool = Σ its workers and
/// cluster = Σ all — the roofline (a max) is applied per level by the labeler.
fn rollup_levels(
    by_worker: &HashMap<(String, u16), WorkloadTotals>,
) -> HashMap<String, WorkloadTotals> {
    let mut levels: HashMap<String, WorkloadTotals> = HashMap::new();
    for ((pool_tag, worker_id), totals) in by_worker {
        levels.entry("cluster".to_string()).or_default().add(totals);
        levels.entry(pool_tag.clone()).or_default().add(totals);
        levels
            .entry(format!("{pool_tag}/{worker_id}"))
            .or_default()
            .add(totals);
    }
    levels
}

fn run_labeler_json(log_dir: &Path, levels: &HashMap<String, WorkloadTotals>) -> Result<Value> {
    let root = repo_root().context("repo root not found for the labeler subprocess")?;
    let request = build_request(levels);
    let log_dir_abs: PathBuf = if log_dir.is_absolute() {
        log_dir.to_path_buf()
    } else {
        root.join(log_dir)
    };

    let mut child = Command::new("uv")
        .args(["run", "python", "-m", "model.work.floors"])
        .arg(&log_dir_abs)
        .current_dir(&root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn `uv run python -m model.work.floors`")?;
    child
        .stdin
        .take()
        .context("labeler subprocess stdin unavailable")?
        .write_all(&serde_json::to_vec(&request)?)?; // dropping the handle sends EOF
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(anyhow!(
            "labeler floors exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    serde_json::from_slice(&output.stdout).context("parse labeler stdout as JSON")
}

fn build_request(levels: &HashMap<String, WorkloadTotals>) -> Value {
    let mut level_json = Map::new();
    for (key, totals) in levels {
        level_json.insert(
            key.clone(),
            json!({
                "matmul_tokens": totals.matmul_tokens,
                "prefill_tokens": totals.prefill_tokens,
                "decode_passes": totals.decode_passes,
                "prefill_pairs": totals.prefill_pairs,
                "prefill_cached": totals.prefill_cached,
                "decode_kv": totals.decode_kv,
                "prefill_requests": totals.prefill_requests,
            }),
        );
    }
    json!({ "levels": Value::Object(level_json) })
}

fn parse_response(parsed: &Value) -> Result<FloorsByLevel> {
    let level_map = parsed
        .get("levels")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow!("labeler output missing `levels` object"))?;
    let mut floors = FloorsByLevel::new();
    for (key, value) in level_map {
        floors.insert(
            key.clone(),
            Floors {
                necessary: value
                    .get("necessary")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                segmented: value
                    .get("segmented")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
            },
        );
    }
    Ok(floors)
}
