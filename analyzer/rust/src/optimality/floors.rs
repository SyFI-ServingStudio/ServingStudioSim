//! Labeler-derived necessary-work floors — two independent lower bounds that sit
//! BELOW the R5 `hardware_limit` green. R5 is a roofline of the sim's *actual*
//! per-kernel work (weights re-loaded every iteration, activation I/O, sim's attention
//! approximation), so it is not the true floor. Here we aggregate each level's
//! workload (reusing the `workload-conservation` subject's `groups` parser — never
//! the cost tree's `slot_flops`/`slot_bytes`, to stay independent) and hand it to the
//! Python `model.work` labeler, which returns two roofline floors per level in GPU·s:
//!
//! - **scope-fused** — global roofline `max(ΣFLOPs/peak, Σbytes/bw)` of the run's
//!   whole token workload as one fused mega-forward (weights counted once): the
//!   loosest, truly irreducible floor. The stable labeler wire field remains
//!   `necessary` for schema compatibility.
//! - **segmented** — `Σ_seg max(compute, memory)`: per-op serial bound (≥ necessary).
//!
//! Run-level waterfalls use one unlocked mega-batch per level. The same batched
//! request also carries one 10,000× saturated row per worker; those normalized
//! semantic labels feed analyzer-owned worker → pool → cluster kernel ladders.
//! Exact iteration detail uses the observed workload directly in locked mode;
//! unlocked mode replicates its independent batch entries before labeling and
//! normalizes the result back to one iteration. Replication amortizes weights without
//! changing sequence geometry. The labeler is a subprocess
//! (`uv run python -m model.work.floors`). Transport/parse failure degrades the
//! request; a per-level label failure degrades only that scope and leaves other
//! worker/pool labels available.

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

/// One level's two labeler floors, in GPU·seconds. Ordered `fused ≤ segmented`.
#[derive(Clone, Copy, Default)]
pub(super) struct Floors {
    pub(super) fused: f64,
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

impl IterationLabel {
    fn normalize(&mut self, normalization: f64) {
        self.floors.fused /= normalization;
        self.floors.segmented /= normalization;
        for segment in &mut self.segments {
            segment.flops /= normalization;
            segment.bytes /= normalization;
        }
    }
}

pub(super) struct RunLabels {
    pub(super) floors: FloorsByLevel,
    pub(super) saturated_workers: HashMap<(String, u16), IterationLabel>,
    pub(super) errors: HashMap<String, String>,
}

struct ParsedLabels {
    labels: HashMap<String, IterationLabel>,
    errors: HashMap<String, String>,
}

/// Per-level floors keyed exactly like `levels.rs`'s level `key`:
/// `"cluster"` | `<pool_tag>` | `"<pool_tag>/<worker_id>"`.
pub(super) type FloorsByLevel = HashMap<String, Floors>;

/// Aggregate the run's workload per level, hand it to the labeler, and return the two
/// floors per level. Transport errors propagate so the caller can degrade the whole
/// label stage; individual level errors stay in `RunLabels::errors`, preserving every
/// independently valid scope.
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
    let ParsedLabels { mut labels, errors } = parse_labels(&response)?;
    let mut saturated_workers = HashMap::new();
    for (pool_tag, worker_id) in by_worker.keys() {
        let key = saturated_worker_key(pool_tag, *worker_id);
        if let Some(mut label) = labels.remove(&key) {
            label.normalize(normalization);
            saturated_workers.insert((pool_tag.clone(), *worker_id), label);
        }
    }
    let floors = labels
        .into_iter()
        .map(|(key, label)| (key, label.floors))
        .collect();
    Ok(RunLabels {
        floors,
        saturated_workers,
        errors,
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
    let mut parsed = parse_labels(&response)?;
    if let Some(error) = parsed.errors.remove(&level_key) {
        return Err(anyhow!("labeler could not label iteration: {error}"));
    }
    let mut label = parsed
        .labels
        .remove(&level_key)
        .context("labeler output missing iteration")?;
    label.normalize(normalization);
    Ok(label)
}

fn saturated_worker_key(pool_tag: &str, worker_id: u16) -> String {
    // `model.work.floors` resolves the model spec from the first slash-delimited
    // component, so this private key remains batchable with ordinary level keys.
    format!("{pool_tag}/__saturated_worker__/{worker_id}")
}

fn parse_labels(response: &Value) -> Result<ParsedLabels> {
    let levels = response
        .get("levels")
        .and_then(Value::as_object)
        .context("labeler output missing `levels` object")?;
    let mut labels = HashMap::new();
    let mut errors = HashMap::new();
    for (key, level) in levels {
        if let Some(error) = level.get("error").and_then(Value::as_str) {
            errors.insert(key.clone(), error.to_string());
            continue;
        }
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
                        .context("semantic segment missing flops")?,
                    bytes: segment
                        .get("bytes")
                        .and_then(Value::as_f64)
                        .context("semantic segment missing bytes")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        labels.insert(
            key.clone(),
            IterationLabel {
                floors: Floors {
                    fused: level
                        .get("necessary")
                        .and_then(Value::as_f64)
                        .context("semantic level missing fused floor")?,
                    segmented: level
                        .get("segmented")
                        .and_then(Value::as_f64)
                        .context("semantic level missing segmented floor")?,
                },
                segments,
            },
        );
    }
    Ok(ParsedLabels { labels, errors })
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::parse_labels;

    #[test]
    fn one_level_error_does_not_discard_other_labels() {
        let parsed = parse_labels(&json!({
            "levels": {
                "cluster": {"error": "heterogeneous models"},
                "main/0": {
                    "necessary": 2.0,
                    "segmented": 3.0,
                    "segments": [{"name": "gemm", "flops": 4.0, "bytes": 5.0}]
                }
            }
        }))
        .unwrap();

        assert_eq!(parsed.errors["cluster"], "heterogeneous models");
        assert_eq!(parsed.labels["main/0"].floors.fused, 2.0);
        assert_eq!(parsed.labels["main/0"].segments[0].name, "gemm");
    }
}
