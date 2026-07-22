//! Kernel tier — the per-location bars. Unlike the worker/pool/cluster tiers a
//! location is a single leaf pooled across workers, so it carries only the four
//! attributable-to-a-leaf buckets (batching / communication / hw-gap / hw-optimal),
//! no idle/imbalance. The fold's per-`(location, worker)` cells are anchored with
//! each worker's factor and summed per location.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use super::fold::WorkerFoldAccumulator;
use super::prepare::KernelLocation;
use super::{ms_to_s, R0, R1, R2, RUNG_KEYS, TOP_KERNELS};

const LADDER_KERNEL_RUNG_KEYS: [&str; 5] = [
    "balanced",
    "per_config_best",
    "ignore_network",
    "hardware_limit",
    "necessary_limit",
];

#[derive(Default)]
struct NecessaryWorkAccumulator {
    semantics: BTreeSet<String>,
    min_flops: f64,
    min_bytes: f64,
    compute_gpu_s: f64,
    memory_gpu_s: f64,
    necessary_gpu_s: f64,
    wall_s: f64,
}

struct KernelLadderAccumulator {
    kind: String,
    is_comm: bool,
    rungs: BTreeMap<&'static str, f64>,
    necessary_work: Option<NecessaryWorkAccumulator>,
}

/// Anchor the fold's per-`(location, worker)` R2..R5 cells with each worker's factor
/// and sum per location → one `[R2, R3, R4, R5]` (GPU·ms) row per location.
pub(super) fn aggregate_kernel_rungs(
    sampled_rungs_by_location_worker: &HashMap<(u32, usize), [f64; 4]>,
    worker_anchor_factors: &[f64],
    num_locations: usize,
) -> Vec<[f64; 4]> {
    let mut kernel_rungs_by_location = vec![[0.0; 4]; num_locations];
    for ((location_id, worker_index), sampled_rungs_ms) in sampled_rungs_by_location_worker {
        let anchor_factor = worker_anchor_factors[*worker_index];
        let location_rungs = &mut kernel_rungs_by_location[*location_id as usize];
        for rung_index in 0..4 {
            location_rungs[rung_index] += sampled_rungs_ms[rung_index] * anchor_factor;
        }
    }
    kernel_rungs_by_location
}

/// Kernel-level bars: per location the Real (balanced) GPU·s split into batching /
/// communication / hw-gap / hw-optimal. Top-N by Real + an `other` roll-up.
pub(super) fn kernel_levels_json(
    kernel_locations: &[KernelLocation],
    kernel_rungs_by_location: &[[f64; 4]],
) -> Vec<Value> {
    let mut location_indices: Vec<usize> = (0..kernel_locations.len())
        .filter(|&location_index| kernel_rungs_by_location[location_index][0] > 0.0)
        .collect();
    location_indices.sort_by(|&left, &right| {
        kernel_rungs_by_location[right][0].total_cmp(&kernel_rungs_by_location[left][0])
    });
    let mut kernel_entries = Vec::new();
    let mut other_rungs = [0.0f64; 4];
    for (rank, &location_index) in location_indices.iter().enumerate() {
        if rank < TOP_KERNELS {
            let location = &kernel_locations[location_index];
            kernel_entries.push(kernel_entry_json(
                &location.name,
                &location.kind,
                location.is_communication,
                &kernel_rungs_by_location[location_index],
            ));
        } else {
            for rung_index in 0..4 {
                other_rungs[rung_index] += kernel_rungs_by_location[location_index][rung_index];
            }
        }
    }
    if other_rungs[0] > 0.0 {
        kernel_entries.push(kernel_entry_json("other", "other", false, &other_rungs));
    }
    kernel_entries
}

fn kernel_entry_json(
    name: &str,
    kind: &str,
    is_communication: bool,
    kernel_rungs: &[f64; 4],
) -> Value {
    // kernel_rungs = [R2 real, R3 per-config-best, R4 ignore-network, R5 hardware].
    json!({
        "name": name,
        "kind": kind,
        "is_comm": is_communication,
        "real": ms_to_s(kernel_rungs[0]),
        "buckets": {
            "batching": ms_to_s((kernel_rungs[0] - kernel_rungs[1]).max(0.0)),
            "communication": ms_to_s((kernel_rungs[1] - kernel_rungs[2]).max(0.0)),
            "hardware_gap": ms_to_s((kernel_rungs[2] - kernel_rungs[3]).max(0.0)),
            "hardware_optimal": ms_to_s(kernel_rungs[3].max(0.0)),
        },
    })
}

/// The worst-headroom kernels for the report (by batching GPU·s), a quick "where
/// to look first" list next to the full kernel array in the payload.
pub(super) fn worst_batching_json(
    kernel_locations: &[KernelLocation],
    kernel_rungs_by_location: &[[f64; 4]],
) -> Vec<Value> {
    let mut location_indices: Vec<usize> = (0..kernel_locations.len()).collect();
    location_indices.sort_by(|&left, &right| {
        let right_batching =
            kernel_rungs_by_location[right][0] - kernel_rungs_by_location[right][1];
        let left_batching = kernel_rungs_by_location[left][0] - kernel_rungs_by_location[left][1];
        right_batching.total_cmp(&left_batching)
    });
    location_indices
        .iter()
        .take(8)
        .filter(|&&location_index| {
            kernel_rungs_by_location[location_index][0]
                - kernel_rungs_by_location[location_index][1]
                > 0.0
        })
        .map(|&location_index| {
            let location = &kernel_locations[location_index];
            let kernel_rungs = &kernel_rungs_by_location[location_index];
            json!({
                "name": location.name,
                "kind": location.kind,
                "batching_gpu_s": ms_to_s(kernel_rungs[0] - kernel_rungs[1]),
                "real_gpu_s": ms_to_s(kernel_rungs[0]),
            })
        })
        .collect()
}

/// Per-worker kernel contributions for the rung-ladder renderer. Kernel values
/// are attributable only from R2 onward; the renderer represents `R1-R2`
/// imbalance and `R0-R1` idle as two explicit aggregate chunks instead of
/// inventing a per-kernel critical-path attribution for either gap. Consumes the
/// fold-order worker rungs + anchors (both in original worker order, matching
/// `sampled_rungs_by_location_worker`'s worker index).
pub(super) fn worker_kernel_ladders_json(
    kernel_locations: &[KernelLocation],
    workers: &[WorkerFoldAccumulator],
    worker_rungs_in_fold_order: &[[f64; 6]],
    sampled_rungs_by_location_worker: &HashMap<(u32, usize), [f64; 4]>,
    worker_anchor_factors: &[f64],
) -> Vec<Value> {
    let mut worker_indices: Vec<usize> = (0..workers.len()).collect();
    worker_indices.sort_by(|&left, &right| {
        (&workers[left].pool_tag, workers[left].worker_id)
            .cmp(&(&workers[right].pool_tag, workers[right].worker_id))
    });

    worker_indices
        .into_iter()
        .map(|worker_index| {
            let worker = &workers[worker_index];
            let worker_rungs = &worker_rungs_in_fold_order[worker_index];
            let mut kernel_rows: Vec<(usize, [f64; 4])> = kernel_locations
                .iter()
                .enumerate()
                .filter_map(|(location_index, _)| {
                    let sampled_rungs = sampled_rungs_by_location_worker
                        .get(&(location_index as u32, worker_index))?;
                    let anchored_rungs =
                        sampled_rungs.map(|value| value * worker_anchor_factors[worker_index]);
                    (anchored_rungs[0] > 0.0).then_some((location_index, anchored_rungs))
                })
                .collect();
            kernel_rows.sort_by(|(left_id, left_rungs), (right_id, right_rungs)| {
                right_rungs[0].total_cmp(&left_rungs[0]).then_with(|| {
                    kernel_locations[*left_id]
                        .name
                        .cmp(&kernel_locations[*right_id].name)
                })
            });
            let kernels: Vec<Value> = kernel_rows
                .into_iter()
                .map(|(location_index, kernel_rungs)| {
                    let location = &kernel_locations[location_index];
                    json!({
                        "name": location.name,
                        "kind": location.kind,
                        "is_comm": location.is_communication,
                        "rungs": {
                            "balanced": ms_to_s(kernel_rungs[0]),
                            "per_config_best": ms_to_s(kernel_rungs[1]),
                            "ignore_network": ms_to_s(kernel_rungs[2]),
                            "hardware_limit": ms_to_s(kernel_rungs[3]),
                        },
                    })
                })
                .collect();
            let worker_key = format!("{}/{}", worker.pool_tag, worker.worker_id);
            json!({
                "key": worker_key,
                "label": worker_key,
                "pool_tag": worker.pool_tag,
                "worker_id": worker.worker_id,
                "rungs": RUNG_KEYS
                    .iter()
                    .zip(worker_rungs.iter())
                    .map(|(key, value)| (key.to_string(), json!(ms_to_s(*value))))
                    .collect::<serde_json::Map<String, Value>>(),
                "special_chunks": {
                    "idle": ms_to_s((worker_rungs[R0] - worker_rungs[R1]).max(0.0)),
                    "imbalance": ms_to_s((worker_rungs[R1] - worker_rungs[R2]).max(0.0)),
                },
                "kernels": kernels,
            })
        })
        .collect()
}

/// Roll analyzer-owned worker ladders into explicit pool and cluster ladders.
/// The UI selects one emitted scope; it never redefines rung or location addition.
/// R0/R1 retain aggregate-only idle/imbalance, R2..R6 reconcile to the sum of their
/// kernel locations, and R7 remains an aggregate globally-fused floor.
pub(super) fn aggregate_kernel_ladders_json(worker_ladders: &[Value]) -> Result<Vec<Value>> {
    if worker_ladders.is_empty() {
        return Ok(Vec::new());
    }
    let all_indices: Vec<usize> = (0..worker_ladders.len()).collect();
    let mut pools: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, ladder) in worker_ladders.iter().enumerate() {
        let pool_tag = ladder["pool_tag"]
            .as_str()
            .context("worker ladder missing pool_tag")?;
        pools.entry(pool_tag.to_string()).or_default().push(index);
    }

    let mut aggregates = vec![sum_kernel_ladders(
        "cluster",
        "cluster",
        "Cluster aggregate",
        worker_ladders,
        &all_indices,
    )?];
    for (pool_tag, indices) in pools {
        aggregates.push(sum_kernel_ladders(
            "pool",
            &pool_tag,
            &format!("{pool_tag} aggregate"),
            worker_ladders,
            &indices,
        )?);
    }
    Ok(aggregates)
}

fn sum_kernel_ladders(
    level: &str,
    key: &str,
    label: &str,
    worker_ladders: &[Value],
    member_indices: &[usize],
) -> Result<Value> {
    let members: Vec<&Value> = member_indices
        .iter()
        .map(|index| &worker_ladders[*index])
        .collect();
    let has_necessary_work = members.iter().all(|ladder| {
        ladder["rungs"]["segmented_necessary"].is_number()
            && ladder["rungs"]["hardware_necessary"].is_number()
    });
    if members.iter().any(|ladder| {
        ladder["rungs"]["segmented_necessary"].is_number()
            || ladder["rungs"]["hardware_necessary"].is_number()
    }) && !has_necessary_work
    {
        bail!("aggregate ladder necessary-work rungs are only partially available");
    }

    let mut rung_values = serde_json::Map::new();
    for rung_key in RUNG_KEYS {
        rung_values.insert(
            rung_key.to_string(),
            json!(sum_path(&members, &["rungs", rung_key])?),
        );
    }
    if has_necessary_work {
        for rung_key in ["segmented_necessary", "hardware_necessary"] {
            rung_values.insert(
                rung_key.to_string(),
                json!(sum_path(&members, &["rungs", rung_key])?),
            );
        }
    }

    let mut special_chunks = serde_json::Map::new();
    for chunk_key in ["idle", "imbalance"] {
        special_chunks.insert(
            chunk_key.to_string(),
            json!(sum_path(&members, &["special_chunks", chunk_key])?),
        );
    }
    if has_necessary_work {
        special_chunks.insert(
            "fusion".to_string(),
            json!(sum_path(&members, &["special_chunks", "fusion"])?),
        );
    }

    let mut kernels: BTreeMap<String, KernelLadderAccumulator> = BTreeMap::new();
    for ladder in &members {
        for kernel in ladder["kernels"]
            .as_array()
            .context("worker ladder missing kernels")?
        {
            let name = kernel["name"]
                .as_str()
                .context("worker ladder kernel missing name")?;
            let kind = kernel["kind"]
                .as_str()
                .context("worker ladder kernel missing kind")?;
            let is_comm = kernel["is_comm"]
                .as_bool()
                .context("worker ladder kernel missing is_comm")?;
            let accumulator =
                kernels
                    .entry(name.to_string())
                    .or_insert_with(|| KernelLadderAccumulator {
                        kind: kind.to_string(),
                        is_comm,
                        rungs: BTreeMap::new(),
                        necessary_work: None,
                    });
            if accumulator.kind != kind || accumulator.is_comm != is_comm {
                bail!(
                    "kernel location {name:?} changes kind or communication class across workers"
                );
            }
            for rung_key in LADDER_KERNEL_RUNG_KEYS {
                let value = kernel["rungs"][rung_key].as_f64();
                if rung_key == "necessary_limit" && !has_necessary_work {
                    continue;
                }
                let value = value.with_context(|| {
                    format!("kernel location {name:?} missing rung {rung_key:?}")
                })?;
                *accumulator.rungs.entry(rung_key).or_default() += value;
            }
            if has_necessary_work {
                accumulate_necessary_work(accumulator, kernel, name)?;
            }
        }
    }

    let kernel_rows: Vec<Value> = kernels
        .into_iter()
        .map(|(name, accumulator)| kernel_accumulator_json(name, accumulator))
        .collect();
    reconcile_aggregate_ladder(
        &rung_values,
        &special_chunks,
        &kernel_rows,
        has_necessary_work,
    )?;

    Ok(json!({
        "level": level,
        "key": key,
        "label": label,
        "rungs": rung_values,
        "special_chunks": special_chunks,
        "kernels": kernel_rows,
        "necessary_work_mode": has_necessary_work.then_some("replicated_large_batch"),
        "necessary_work_replication_factor": has_necessary_work
            .then_some(super::UNLOCKED_ITERATION_REPLICATION_FACTOR),
    }))
}

fn sum_path(members: &[&Value], path: &[&str]) -> Result<f64> {
    members.iter().try_fold(0.0, |sum, member| {
        let value = path.iter().fold(*member, |value, key| &value[*key]);
        Ok(sum
            + value
                .as_f64()
                .with_context(|| format!("kernel ladder path {path:?} is not numeric"))?)
    })
}

fn accumulate_necessary_work(
    accumulator: &mut KernelLadderAccumulator,
    kernel: &Value,
    name: &str,
) -> Result<()> {
    let necessary = kernel
        .get("necessary_work")
        .context("mapped kernel missing necessary_work")?;
    let target = accumulator
        .necessary_work
        .get_or_insert_with(NecessaryWorkAccumulator::default);
    for semantic in necessary["semantics"]
        .as_array()
        .context("necessary_work semantics is not an array")?
    {
        target.semantics.insert(
            semantic
                .as_str()
                .with_context(|| format!("kernel {name:?} has a non-string semantic"))?
                .to_string(),
        );
    }
    for (field, target_value) in [
        ("min_flops", &mut target.min_flops),
        ("min_bytes", &mut target.min_bytes),
        ("compute_gpu_s", &mut target.compute_gpu_s),
        ("memory_gpu_s", &mut target.memory_gpu_s),
        ("necessary_gpu_s", &mut target.necessary_gpu_s),
        ("wall_s", &mut target.wall_s),
    ] {
        *target_value += necessary[field]
            .as_f64()
            .with_context(|| format!("kernel {name:?} necessary_work missing {field:?}"))?;
    }
    Ok(())
}

fn kernel_accumulator_json(name: String, accumulator: KernelLadderAccumulator) -> Value {
    let hardware_limit = accumulator.rungs["hardware_limit"];
    let necessary_limit = accumulator.rungs.get("necessary_limit").copied();
    let necessary_work = accumulator.necessary_work.map(|work| {
        json!({
            "semantics": work.semantics,
            "min_flops": work.min_flops,
            "min_bytes": work.min_bytes,
            "compute_gpu_s": work.compute_gpu_s,
            "memory_gpu_s": work.memory_gpu_s,
            "necessary_gpu_s": work.necessary_gpu_s,
            "wall_s": work.wall_s,
            "redundant_gpu_s": (hardware_limit - work.necessary_gpu_s).max(0.0),
            "under_accounted_gpu_s": (work.necessary_gpu_s - hardware_limit).max(0.0),
            "bound": if work.compute_gpu_s >= work.memory_gpu_s { "compute" } else { "memory" },
        })
    });
    json!({
        "name": name,
        "kind": accumulator.kind,
        "is_comm": accumulator.is_comm,
        "rungs": {
            "balanced": accumulator.rungs["balanced"],
            "per_config_best": accumulator.rungs["per_config_best"],
            "ignore_network": accumulator.rungs["ignore_network"],
            "hardware_limit": hardware_limit,
            "necessary_limit": necessary_limit,
        },
        "necessary_work": necessary_work,
    })
}

fn reconcile_aggregate_ladder(
    rungs: &serde_json::Map<String, Value>,
    special_chunks: &serde_json::Map<String, Value>,
    kernels: &[Value],
    has_necessary_work: bool,
) -> Result<()> {
    for rung_key in [
        "balanced",
        "per_config_best",
        "ignore_network",
        "hardware_limit",
    ] {
        let kernel_sum: f64 = kernels
            .iter()
            .map(|kernel| kernel["rungs"][rung_key].as_f64().unwrap_or(0.0))
            .sum();
        let expected = rungs[rung_key].as_f64().unwrap_or(0.0);
        let tolerance = expected.abs().max(1.0) * 1e-9;
        if (kernel_sum - expected).abs() > tolerance {
            bail!(
                "aggregate kernel rung {rung_key:?} does not reconcile: {kernel_sum} vs {expected}"
            );
        }
    }
    let balanced = rungs["balanced"].as_f64().unwrap_or(0.0);
    let busy = rungs["busy"].as_f64().unwrap_or(0.0);
    let real = rungs["real"].as_f64().unwrap_or(0.0);
    let imbalance = special_chunks["imbalance"].as_f64().unwrap_or(0.0);
    let idle = special_chunks["idle"].as_f64().unwrap_or(0.0);
    if ((balanced + imbalance) - busy).abs() > busy.abs().max(1.0) * 1e-9
        || ((busy + idle) - real).abs() > real.abs().max(1.0) * 1e-9
    {
        bail!("aggregate kernel ladder special chunks do not reconcile with R0/R1/R2");
    }
    if has_necessary_work {
        let kernel_sum: f64 = kernels
            .iter()
            .map(|kernel| kernel["rungs"]["necessary_limit"].as_f64().unwrap_or(0.0))
            .sum();
        let segmented = rungs["segmented_necessary"].as_f64().unwrap_or(0.0);
        let hardware = rungs["hardware_necessary"].as_f64().unwrap_or(0.0);
        let fusion = special_chunks["fusion"].as_f64().unwrap_or(0.0);
        let tolerance = segmented.abs().max(1.0) * 1e-9;
        if (kernel_sum - segmented).abs() > tolerance
            || ((hardware + fusion) - segmented).abs() > tolerance
        {
            bail!("aggregate R6/R7 necessary-work rungs do not reconcile");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::aggregate_kernel_ladders_json;

    fn worker_ladder(worker_id: u16, scale: f64) -> serde_json::Value {
        json!({
            "pool_tag": "main",
            "worker_id": worker_id,
            "rungs": {
                "real": 10.0 * scale,
                "busy": 9.0 * scale,
                "balanced": 8.0 * scale,
                "per_config_best": 7.0 * scale,
                "ignore_network": 6.0 * scale,
                "hardware_limit": 5.0 * scale,
                "segmented_necessary": 3.0 * scale,
                "hardware_necessary": 2.0 * scale,
            },
            "special_chunks": {
                "idle": 1.0 * scale,
                "imbalance": 1.0 * scale,
                "fusion": 1.0 * scale,
            },
            "kernels": [{
                "name": "model.gemm",
                "kind": "single_gemm",
                "is_comm": false,
                "rungs": {
                    "balanced": 8.0 * scale,
                    "per_config_best": 7.0 * scale,
                    "ignore_network": 6.0 * scale,
                    "hardware_limit": 5.0 * scale,
                    "necessary_limit": 3.0 * scale,
                },
                "necessary_work": {
                    "semantics": ["gemm"],
                    "min_flops": 3e12 * scale,
                    "min_bytes": 1e9 * scale,
                    "compute_gpu_s": 3.0 * scale,
                    "memory_gpu_s": 1.0 * scale,
                    "necessary_gpu_s": 3.0 * scale,
                    "wall_s": 3.0 * scale,
                    "redundant_gpu_s": 2.0 * scale,
                    "under_accounted_gpu_s": 0.0,
                    "bound": "compute",
                },
            }],
        })
    }

    #[test]
    fn analyzer_rolls_worker_kernel_ladders_up_and_reconciles_r6_r7() {
        let aggregates =
            aggregate_kernel_ladders_json(&[worker_ladder(0, 1.0), worker_ladder(1, 2.0)]).unwrap();
        let cluster = &aggregates[0];
        assert_eq!(cluster["level"], "cluster");
        assert_eq!(cluster["rungs"]["hardware_limit"], 15.0);
        assert_eq!(cluster["rungs"]["segmented_necessary"], 9.0);
        assert_eq!(cluster["rungs"]["hardware_necessary"], 6.0);
        assert_eq!(cluster["special_chunks"]["fusion"], 3.0);
        assert_eq!(cluster["kernels"][0]["rungs"]["necessary_limit"], 9.0);
        assert_eq!(aggregates[1]["level"], "pool");
    }
}
