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
//! Unlocked run-level waterfalls use one mega-batch per level and one 10,000×
//! saturated row per worker. Batch-locked run-level waterfalls instead deduplicate
//! exact iteration shapes and add occurrence-weighted per-iteration rooflines, so
//! work never crosses a fixed-batch boundary. Exact iteration detail uses the
//! observed workload directly in locked mode; unlocked mode replicates its
//! independent batch entries before labeling and normalizes the result back to one
//! iteration. Replication amortizes weights without changing sequence geometry.
//! The labeler is a subprocess
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
    collect_iteration_workload, collect_workload_by_worker, collect_workload_shapes_by_worker,
    WeightedWorkload, WorkloadTotals,
};
use crate::kernel_query::repo_root;

/// One level's two labeler floors, in GPU·seconds. Ordered `fused ≤ segmented`.
#[derive(Clone, Copy, Default)]
pub(super) struct Floors {
    pub(super) fused: f64,
    pub(super) segmented: f64,
}

impl Floors {
    fn add_scaled(&mut self, other: Self, scale: f64) {
        self.fused += other.fused * scale;
        self.segmented += other.segmented * scale;
    }
}

#[derive(Clone, Debug)]
pub(super) struct SemanticWork {
    pub(super) name: String,
    pub(super) flops: f64,
    pub(super) bytes: f64,
    pub(super) necessary_gpu_s: f64,
    /// Precision this row's math runs at. The labeler decides it per row because a
    /// checkpoint is mixed — an FP8 MoE still keeps its router and its BF16
    /// FlashMLA kernel off the FP8 tensor cores.
    pub(super) compute_dtype: String,
}

pub(super) struct IterationLabel {
    pub(super) floors: Floors,
    pub(super) segments: Vec<SemanticWork>,
}

pub(super) struct WeightedIterationLabel {
    pub(super) label: IterationLabel,
    pub(super) occurrences: u64,
}

pub(super) struct WorkerComposition {
    pub(super) labels: Vec<WeightedIterationLabel>,
    pub(super) floors: Floors,
}

#[derive(Clone, Copy, Debug, Default)]
struct CompositionStats {
    unique_shapes: u64,
    iterations: u64,
    affine_bases: u64,
    direct_fallback_bases: u64,
}

impl IterationLabel {
    fn normalize(&mut self, normalization: f64) {
        self.floors.fused /= normalization;
        self.floors.segmented /= normalization;
        for segment in &mut self.segments {
            segment.flops /= normalization;
            segment.bytes /= normalization;
            segment.necessary_gpu_s /= normalization;
        }
    }
}

pub(super) struct RunLabels {
    pub(super) floors: FloorsByLevel,
    pub(super) workers: HashMap<(String, u16), WorkerComposition>,
    pub(super) errors: HashMap<String, String>,
    pub(super) batch_locked_unique_shapes: Option<usize>,
    pub(super) batch_locked_iterations: Option<u64>,
    pub(super) batch_locked_affine_bases: Option<u64>,
    pub(super) batch_locked_direct_fallback_bases: Option<u64>,
}

struct ParsedLabels {
    labels: HashMap<String, IterationLabel>,
    errors: HashMap<String, String>,
    composition_stats: HashMap<String, CompositionStats>,
}

/// Per-level floors keyed exactly like `levels.rs`'s level `key`:
/// `"cluster"` | `<pool_tag>` | `"<pool_tag>/<worker_id>"`.
pub(super) type FloorsByLevel = HashMap<String, Floors>;

/// Aggregate the run's workload per level, hand it to the labeler, and return the two
/// floors per level. Transport errors propagate so the caller can degrade the whole
/// label stage; individual level errors stay in `RunLabels::errors`, preserving every
/// independently valid scope.
pub(super) async fn compute_saturated_run_labels(
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
    let ParsedLabels {
        mut labels, errors, ..
    } = parse_labels(&response)?;
    let mut workers = HashMap::new();
    for (pool_tag, worker_id) in by_worker.keys() {
        let key = saturated_worker_key(pool_tag, *worker_id);
        if let Some(mut label) = labels.remove(&key) {
            label.normalize(normalization);
            workers.insert(
                (pool_tag.clone(), *worker_id),
                WorkerComposition {
                    floors: label.floors,
                    labels: vec![WeightedIterationLabel {
                        label,
                        occurrences: 1,
                    }],
                },
            );
        }
    }
    let floors = labels
        .into_iter()
        .map(|(key, label)| (key, label.floors))
        .collect();
    Ok(RunLabels {
        floors,
        workers,
        errors,
        batch_locked_unique_shapes: None,
        batch_locked_iterations: None,
        batch_locked_affine_bases: None,
        batch_locked_direct_fallback_bases: None,
    })
}

/// Label every distinct observed iteration shape once and compose the results while
/// preserving iteration boundaries. Equal shapes are weighted by occurrence count;
/// no weights are amortized across separate batches. Worker floors then add into
/// pool/cluster floors only when that whole scope is available.
pub(super) async fn compute_batch_locked_run_labels(
    ctx: &SessionContext,
    log_dir: &Path,
) -> Result<RunLabels> {
    let shapes_by_worker = collect_workload_shapes_by_worker(ctx).await?;
    if shapes_by_worker.is_empty() {
        return Err(anyhow!("no cost_log iteration workloads to label"));
    }

    let mut worker_keys: Vec<(String, u16)> = shapes_by_worker.keys().cloned().collect();
    worker_keys.sort();
    let batch_locked_unique_shapes = shapes_by_worker.values().map(Vec::len).sum();
    let batch_locked_iterations = shapes_by_worker
        .values()
        .flatten()
        .map(|shape| shape.occurrences)
        .sum();
    let request = build_locked_request(&shapes_by_worker);
    let response = run_labeler_request(log_dir, request)?;
    let ParsedLabels {
        mut labels,
        errors,
        mut composition_stats,
    } = parse_labels(&response)?;

    let mut workers = HashMap::new();
    let mut returned_stats = CompositionStats::default();
    for worker_key in &worker_keys {
        let worker_level_key = format!("{}/{}", worker_key.0, worker_key.1);
        if errors.contains_key(&worker_level_key) {
            continue;
        }
        let Some(label) = labels.remove(&worker_level_key) else {
            continue;
        };
        let stats = composition_stats
            .remove(&worker_level_key)
            .with_context(|| {
                format!("locked label {worker_level_key:?} missing composition stats")
            })?;
        let expected_shapes = &shapes_by_worker[worker_key];
        let expected_iterations: u64 = expected_shapes.iter().map(|shape| shape.occurrences).sum();
        if stats.unique_shapes != expected_shapes.len() as u64
            || stats.iterations != expected_iterations
        {
            return Err(anyhow!(
                "locked labeler composition mismatch for {worker_level_key:?}: Rust observed {} unique shapes / {expected_iterations} iterations, labeler returned {} / {}",
                expected_shapes.len(),
                stats.unique_shapes,
                stats.iterations,
            ));
        }
        returned_stats.unique_shapes += stats.unique_shapes;
        returned_stats.iterations += stats.iterations;
        returned_stats.affine_bases += stats.affine_bases;
        returned_stats.direct_fallback_bases += stats.direct_fallback_bases;
        workers.insert(
            worker_key.clone(),
            WorkerComposition {
                floors: label.floors,
                labels: vec![WeightedIterationLabel {
                    label,
                    occurrences: 1,
                }],
            },
        );
    }
    let floors = rollup_locked_floors(&worker_keys, &workers);
    Ok(RunLabels {
        floors,
        workers,
        errors,
        batch_locked_unique_shapes: Some(batch_locked_unique_shapes),
        batch_locked_iterations: Some(batch_locked_iterations),
        batch_locked_affine_bases: Some(returned_stats.affine_bases),
        batch_locked_direct_fallback_bases: Some(returned_stats.direct_fallback_bases),
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

fn rollup_locked_floors(
    expected_workers: &[(String, u16)],
    workers: &HashMap<(String, u16), WorkerComposition>,
) -> FloorsByLevel {
    let mut floors = HashMap::new();
    let mut expected_by_pool: HashMap<&str, usize> = HashMap::new();
    let mut available_by_pool: HashMap<&str, (usize, Floors)> = HashMap::new();
    for (pool_tag, worker_id) in expected_workers {
        *expected_by_pool.entry(pool_tag.as_str()).or_default() += 1;
        if let Some(composition) = workers.get(&(pool_tag.clone(), *worker_id)) {
            floors.insert(format!("{pool_tag}/{worker_id}"), composition.floors);
            let (count, pool_floors) = available_by_pool.entry(pool_tag).or_default();
            *count += 1;
            pool_floors.add_scaled(composition.floors, 1.0);
        }
    }
    let mut cluster_floors = Floors::default();
    let mut complete_cluster = true;
    for (pool_tag, expected_count) in expected_by_pool {
        match available_by_pool.get(pool_tag) {
            Some((available_count, pool_floors)) if *available_count == expected_count => {
                floors.insert(pool_tag.to_string(), *pool_floors);
                cluster_floors.add_scaled(*pool_floors, 1.0);
            }
            _ => complete_cluster = false,
        }
    }
    if complete_cluster && workers.len() == expected_workers.len() {
        floors.insert("cluster".to_string(), cluster_floors);
    }
    floors
}

fn parse_labels(response: &Value) -> Result<ParsedLabels> {
    let levels = response
        .get("levels")
        .and_then(Value::as_object)
        .context("labeler output missing `levels` object")?;
    let mut labels = HashMap::new();
    let mut errors = HashMap::new();
    let mut composition_stats = HashMap::new();
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
                    necessary_gpu_s: segment
                        .get("necessary")
                        .and_then(Value::as_f64)
                        .context("semantic segment missing necessary roofline")?,
                    compute_dtype: segment
                        .get("compute_dtype")
                        .and_then(Value::as_str)
                        .context("semantic segment missing compute_dtype")?
                        .to_string(),
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
        if let Some(composition) = level.get("composition") {
            composition_stats.insert(
                key.clone(),
                CompositionStats {
                    unique_shapes: composition
                        .get("unique_shapes")
                        .and_then(Value::as_u64)
                        .context("composition missing unique_shapes")?,
                    iterations: composition
                        .get("iterations")
                        .and_then(Value::as_u64)
                        .context("composition missing iterations")?,
                    affine_bases: composition
                        .get("affine_bases")
                        .and_then(Value::as_u64)
                        .context("composition missing affine_bases")?,
                    direct_fallback_bases: composition
                        .get("direct_fallback_bases")
                        .and_then(Value::as_u64)
                        .context("composition missing direct_fallback_bases")?,
                },
            );
        }
    }
    Ok(ParsedLabels {
        labels,
        errors,
        composition_stats,
    })
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
    let request = build_request(levels);
    run_labeler_request(log_dir, request)
}

fn run_labeler_request(log_dir: &Path, request: Value) -> Result<Value> {
    let root = repo_root().context("repo root not found for the labeler subprocess")?;
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
    let request_bytes = serde_json::to_vec(&request)?;
    let write_result = child
        .stdin
        .take()
        .context("labeler subprocess stdin unavailable")?
        .write_all(&request_bytes); // dropping the handle sends EOF
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(anyhow!(
            "labeler floors exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    write_result.context("write labeler request to stdin")?;
    serde_json::from_slice(&output.stdout).context("parse labeler stdout as JSON")
}

fn build_request(levels: &HashMap<String, WorkloadTotals>) -> Value {
    let mut level_json = Map::new();
    for (key, totals) in levels {
        level_json.insert(key.clone(), workload_json(*totals));
    }
    json!({ "levels": Value::Object(level_json) })
}

fn build_locked_request(shapes_by_worker: &HashMap<(String, u16), Vec<WeightedWorkload>>) -> Value {
    let mut compositions = Map::new();
    for ((pool_tag, worker_id), shapes) in shapes_by_worker {
        compositions.insert(
            format!("{pool_tag}/{worker_id}"),
            Value::Array(
                shapes
                    .iter()
                    .map(|shape| {
                        json!({
                            "occurrences": shape.occurrences,
                            "totals": workload_json(shape.totals),
                        })
                    })
                    .collect(),
            ),
        );
    }
    json!({"locked_compositions": Value::Object(compositions)})
}

fn workload_json(totals: WorkloadTotals) -> Value {
    json!({
        "matmul_tokens": totals.matmul_tokens,
        "prefill_tokens": totals.prefill_tokens,
        "decode_passes": totals.decode_passes,
        "prefill_pairs": totals.prefill_pairs,
        "prefill_cached": totals.prefill_cached,
        "decode_kv": totals.decode_kv,
        "prefill_requests": totals.prefill_requests,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{parse_labels, rollup_locked_floors, Floors, WorkerComposition};

    #[test]
    fn one_level_error_does_not_discard_other_labels() {
        let parsed = parse_labels(&json!({
            "levels": {
                "cluster": {"error": "heterogeneous models"},
                "main/0": {
                    "necessary": 2.0,
                    "segmented": 3.0,
                    "segments": [{
                        "name": "gemm",
                        "flops": 4.0,
                        "bytes": 5.0,
                        "compute_dtype": "fp8",
                        "necessary": 0.5
                    }],
                    "composition": {
                        "unique_shapes": 3,
                        "iterations": 9,
                        "affine_bases": 1,
                        "direct_fallback_bases": 0
                    }
                }
            }
        }))
        .unwrap();

        assert_eq!(parsed.errors["cluster"], "heterogeneous models");
        assert_eq!(parsed.labels["main/0"].floors.fused, 2.0);
        assert_eq!(parsed.labels["main/0"].segments[0].name, "gemm");
        assert_eq!(parsed.labels["main/0"].segments[0].necessary_gpu_s, 0.5);
        assert_eq!(parsed.labels["main/0"].segments[0].compute_dtype, "fp8");
        let stats = parsed.composition_stats["main/0"];
        assert_eq!(stats.unique_shapes, 3);
        assert_eq!(stats.iterations, 9);
        assert_eq!(stats.affine_bases, 1);
        assert_eq!(stats.direct_fallback_bases, 0);
    }

    #[test]
    fn locked_floor_rollup_requires_complete_scopes_and_adds_worker_floors() {
        let expected = vec![
            ("main".to_string(), 0),
            ("main".to_string(), 1),
            ("decode".to_string(), 0),
        ];
        let composition = |fused, segmented| WorkerComposition {
            labels: Vec::new(),
            floors: Floors { fused, segmented },
        };
        let partial = std::collections::HashMap::from([
            (("main".to_string(), 0), composition(2.0, 3.0)),
            (("decode".to_string(), 0), composition(5.0, 7.0)),
        ]);
        let partial_floors = rollup_locked_floors(&expected, &partial);
        assert_eq!(partial_floors["main/0"].fused, 2.0);
        assert_eq!(partial_floors["decode"].fused, 5.0);
        assert!(!partial_floors.contains_key("main"));
        assert!(!partial_floors.contains_key("cluster"));

        let mut complete = partial;
        complete.insert(("main".to_string(), 1), composition(11.0, 13.0));
        let complete_floors = rollup_locked_floors(&expected, &complete);
        assert_eq!(complete_floors["main"].fused, 13.0);
        assert_eq!(complete_floors["cluster"].fused, 18.0);
        assert_eq!(complete_floors["cluster"].segmented, 23.0);
    }
}
