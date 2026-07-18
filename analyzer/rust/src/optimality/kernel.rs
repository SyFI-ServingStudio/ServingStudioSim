//! Kernel tier — the per-location bars. Unlike the worker/pool/cluster tiers a
//! location is a single leaf pooled across workers, so it carries only the four
//! attributable-to-a-leaf buckets (batching / communication / hw-gap / hw-optimal),
//! no idle/imbalance. The fold's per-`(location, worker)` cells are anchored with
//! each worker's factor and summed per location.

use std::collections::HashMap;

use serde_json::{json, Value};

use super::fold::WorkerFoldAccumulator;
use super::prepare::KernelLocation;
use super::{ms_to_s, R0, R1, R2, RUNG_KEYS, TOP_KERNELS};

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
