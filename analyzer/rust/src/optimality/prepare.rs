//! Preparation stage — turn the per-worker CostTree manifests + grid peaks + GPU
//! spec into the precomputed structures the fold consumes. Interns every manifest
//! leaf into a global location and, per `(pool_tag, worker_id, section)`, precomputes
//! the mean-fold weight `α` per slot, the slot→location map, and each slot's rate
//! ceilings. No folding happens here — this is all input precompute.

use std::collections::{BTreeMap, HashMap};

use serde_json::Value;

use crate::trace::manifest::{fold_mean, ManifestDoc};

use super::grid_peaks::GridPeakCatalog;
use super::spec::GpuSpec;

/// One cost-tree location (leaf identity) pooled across workers, keyed by the
/// manifest `name` — exactly like `kernel-throughput`, so `Max` siblings and DP
/// replicas of a location pool together.
#[derive(Clone, Debug)]
pub(super) struct KernelLocation {
    pub(super) name: String,
    pub(super) kind: String,
    pub(super) is_communication: bool,
}

/// Per `(pool_tag, worker_id, section)` structural metadata, precomputed once so
/// the hot row loop is array indexing: the mean-fold weight `α` per slot, the
/// slot→location map, and each slot's rate ceilings.
pub(super) struct SectionFoldPlan {
    /// Mean-mode fold weight per slot (`Σ` of leaf weights if a slot recurs).
    pub(super) mean_fold_weight_by_slot: Vec<f64>,
    pub(super) location_id_by_slot: Vec<u32>,
    pub(super) is_communication_by_slot: Vec<bool>,
    /// Grid-peak ceilings for R3 (`0` = no sidecar entry → that leaf's R3 = R2).
    pub(super) grid_peak_tflops_by_slot: Vec<f64>,
    pub(super) grid_peak_gbps_by_slot: Vec<f64>,
    /// R3 chooses exactly one grid throughput basis. A config is compute-capable
    /// when any fitted point reaches the GPU-spec ridge intensity; otherwise it
    /// uses bandwidth. This deliberately models the large-batch regime rather
    /// than rooflining the current small-batch point.
    pub(super) r3_uses_compute_throughput_by_slot: Vec<bool>,
    /// Hardware spec compute peak for R5 by the slot's dtype (`0` = no spec).
    pub(super) hardware_peak_tflops_by_slot: Vec<f64>,
}

/// Intern every manifest leaf into a global location, and precompute each
/// `(pool_tag, worker_id, section)`'s fold weights + per-slot rate ceilings.
pub(super) fn build_section_fold_plans(
    manifests_by_worker: &BTreeMap<(String, u16), ManifestDoc>,
    grid_peak_catalog: &GridPeakCatalog,
    gpu_spec: &GpuSpec,
) -> (
    Vec<KernelLocation>,
    HashMap<(String, u16, String), SectionFoldPlan>,
) {
    let mut location_id_by_name: HashMap<String, u32> = HashMap::new();
    let mut kernel_locations: Vec<KernelLocation> = Vec::new();
    let mut section_fold_plans = HashMap::new();

    for ((pool_tag, worker_id), manifest_doc) in manifests_by_worker {
        for manifest_section in &manifest_doc.sections {
            let manifest = &manifest_section.manifest;
            let num_slots = manifest.slots.len();
            let mut mean_fold_weight_by_slot = vec![0.0; num_slots];
            // Root is node 0 (BFS layout: parent precedes children).
            if !manifest.nodes.is_empty() {
                fold_mean(manifest, 0, 1.0, &mut |slot, leaf_weight| {
                    if let Some(accumulated_weight) = mean_fold_weight_by_slot.get_mut(slot) {
                        *accumulated_weight += leaf_weight;
                    }
                });
            }

            let mut location_id_by_slot = Vec::with_capacity(num_slots);
            let mut is_communication_by_slot = Vec::with_capacity(num_slots);
            let mut grid_peak_tflops_by_slot = Vec::with_capacity(num_slots);
            let mut grid_peak_gbps_by_slot = Vec::with_capacity(num_slots);
            let mut r3_uses_compute_throughput_by_slot = Vec::with_capacity(num_slots);
            let mut hardware_peak_tflops_by_slot = Vec::with_capacity(num_slots);
            for leaf in &manifest.slots {
                let is_communication = is_communication_kind(&leaf.kind);
                let location_id =
                    *location_id_by_name
                        .entry(leaf.name.clone())
                        .or_insert_with(|| {
                            let location_id = kernel_locations.len() as u32;
                            kernel_locations.push(KernelLocation {
                                name: leaf.name.clone(),
                                kind: leaf.kind.clone(),
                                is_communication,
                            });
                            location_id
                        });
                location_id_by_slot.push(location_id);
                is_communication_by_slot.push(is_communication);
                let peak = grid_peak_catalog
                    .get(&leaf.kind, &leaf.kernel_config)
                    .unwrap_or_default();
                grid_peak_tflops_by_slot.push(peak.tflops);
                grid_peak_gbps_by_slot.push(peak.gbps);
                let dtype = compute_dtype(&leaf.kernel_config);
                let hardware_peak_tflops = if is_communication {
                    0.0
                } else {
                    gpu_spec.peak_tflops(&dtype)
                };
                hardware_peak_tflops_by_slot.push(hardware_peak_tflops);
                let hardware_bandwidth_gbps = gpu_spec.mem_bandwidth_gbps;
                let hardware_ridge_flops_per_byte =
                    if hardware_peak_tflops > 0.0 && hardware_bandwidth_gbps > 0.0 {
                        hardware_peak_tflops * 1_000.0 / hardware_bandwidth_gbps
                    } else {
                        f64::INFINITY
                    };
                r3_uses_compute_throughput_by_slot.push(r3_uses_compute_throughput(
                    peak.max_arithmetic_intensity_flops_per_byte,
                    peak.tflops,
                    is_communication,
                    hardware_ridge_flops_per_byte,
                ));
            }
            section_fold_plans.insert(
                (
                    pool_tag.clone(),
                    *worker_id,
                    manifest_section.section.clone(),
                ),
                SectionFoldPlan {
                    mean_fold_weight_by_slot,
                    location_id_by_slot,
                    is_communication_by_slot,
                    grid_peak_tflops_by_slot,
                    grid_peak_gbps_by_slot,
                    r3_uses_compute_throughput_by_slot,
                    hardware_peak_tflops_by_slot,
                },
            );
        }
    }
    (kernel_locations, section_fold_plans)
}

fn r3_uses_compute_throughput(
    max_arithmetic_intensity_flops_per_byte: f64,
    peak_tflops: f64,
    is_communication: bool,
    hardware_ridge_flops_per_byte: f64,
) -> bool {
    !is_communication
        && peak_tflops > 0.0
        && max_arithmetic_intensity_flops_per_byte >= hardware_ridge_flops_per_byte
}

/// Best-effort compute dtype for a leaf's roofline: the first present of the
/// GEMM/norm `dtype`, then attention `q_dtype`, then an input dtype; else bf16.
fn compute_dtype(config: &Value) -> String {
    for key in ["dtype", "q_dtype", "input_dtype", "kv_dtype"] {
        if let Some(s) = config.get(key).and_then(Value::as_str) {
            return s.to_string();
        }
    }
    "bf16".to_string()
}

/// Collective / point-to-point leaves — dropped at R4 ("ignore network") and R5.
fn is_communication_kind(kind: &str) -> bool {
    matches!(
        kind,
        "all_reduce"
            | "all_gather"
            | "reduce_scatter"
            | "all_to_all"
            | "broadcast"
            | "gather"
            | "scatter"
            | "send"
            | "recv"
    ) || kind.starts_with("p2p")
        || kind.starts_with("nccl")
        || kind.starts_with("comm")
}

#[cfg(test)]
mod tests {
    use super::r3_uses_compute_throughput;

    #[test]
    fn r3_basis_follows_whether_the_grid_crosses_the_hardware_ridge() {
        assert!(r3_uses_compute_throughput(400.0, 700.0, false, 300.0));
        assert!(!r3_uses_compute_throughput(100.0, 700.0, false, 300.0));
        assert!(!r3_uses_compute_throughput(400.0, 0.0, true, 300.0));
    }
}
