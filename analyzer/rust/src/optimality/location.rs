//! Strict semantic-work → exact CostTree-location attribution for one iteration.
//!
//! Mapping files are model-independent of simulator shapes: they connect stable
//! `model.work` semantic rows to versioned manifest location names. Attribution is
//! all-or-nothing so an arch change cannot silently turn missing necessary work into
//! zero. Communication leaves are exempt because necessary model work is local work.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

use super::floors::{IterationLabel, SemanticWork};
use super::prepare::KernelLocation;
use super::spec::GpuSpec;
use super::under_accounted_difference;

#[derive(Deserialize)]
struct LocationMap {
    schema_version: u64,
    mapping_id: String,
    arch_types: Vec<String>,
    locations: Vec<LocationRule>,
}

#[derive(Deserialize)]
struct LocationRule {
    location: String,
    semantics: Vec<String>,
}

pub(super) struct LocationAttribution {
    pub(super) mapping_id: String,
}

#[derive(Clone, Copy, Default)]
struct NecessaryWork {
    flops: f64,
    bytes: f64,
}

/// Add R6 and per-location necessary-work diagnostics to one worker ladder. The
/// semantic label may describe an exact iteration or a saturated run aggregate.
/// Any validation failure returns before mutation, preserving the plain R0..R5 view.
#[allow(clippy::too_many_arguments)]
pub(super) fn attribute_ladder(
    repo_root: &Path,
    log_dir: &Path,
    pool_tag: &str,
    kernel_locations: &[KernelLocation],
    ladder: &mut Value,
    label: &IterationLabel,
    gpu_spec: GpuSpec,
    gpu_count: f64,
) -> Result<LocationAttribution> {
    let arch_type = pool_arch_type(log_dir, pool_tag)?;
    let location_map = load_location_map(repo_root, &arch_type)?;
    validate_mapping(&location_map, kernel_locations, &label.segments)?;

    let semantic_work: BTreeMap<&str, NecessaryWork> = label
        .segments
        .iter()
        .map(|segment| {
            (
                segment.name.as_str(),
                NecessaryWork {
                    flops: segment.flops,
                    bytes: segment.bytes,
                },
            )
        })
        .collect();
    let dtype = pool_compute_dtype(log_dir, pool_tag)?;
    let peak_tflops = gpu_spec.peak_tflops(&dtype);
    let bandwidth_gbps = gpu_spec.mem_bandwidth_gbps;
    if peak_tflops <= 0.0 || bandwidth_gbps <= 0.0 {
        bail!("GPU spec lacks positive {dtype} compute peak or memory bandwidth");
    }

    let kind_by_location: BTreeMap<&str, &str> = kernel_locations
        .iter()
        .map(|location| (location.name.as_str(), location.kind.as_str()))
        .collect();
    let mut details_by_location = BTreeMap::new();
    let mut segmented_necessary_gpu_s = 0.0;
    for rule in &location_map.locations {
        let work =
            rule.semantics
                .iter()
                .try_fold(NecessaryWork::default(), |mut total, semantic| {
                    let row = semantic_work
                        .get(semantic.as_str())
                        .with_context(|| format!("mapped semantic row {semantic:?} is absent"))?;
                    total.flops += row.flops;
                    total.bytes += row.bytes;
                    Ok::<_, anyhow::Error>(total)
                })?;
        let compute_gpu_s = work.flops / (peak_tflops * 1e12);
        let memory_gpu_s = work.bytes / (bandwidth_gbps * 1e9);
        let necessary_gpu_s = compute_gpu_s.max(memory_gpu_s);
        segmented_necessary_gpu_s += necessary_gpu_s;
        details_by_location.insert(
            rule.location.as_str(),
            json!({
                "semantics": rule.semantics,
                "min_flops": work.flops,
                "min_bytes": work.bytes,
                "compute_gpu_s": compute_gpu_s,
                "memory_gpu_s": memory_gpu_s,
                "necessary_gpu_s": necessary_gpu_s,
                "wall_s": necessary_gpu_s / gpu_count,
                "bound": if compute_gpu_s >= memory_gpu_s { "compute" } else { "memory" },
            }),
        );
    }
    let tolerance = (label.floors.segmented.abs() * 1e-7).max(1e-12);
    if (segmented_necessary_gpu_s - label.floors.segmented).abs() > tolerance {
        bail!(
            "mapped location floor {segmented_necessary_gpu_s} does not reconcile with semantic floor {}",
            label.floors.segmented
        );
    }

    let kernels = ladder
        .get_mut("kernels")
        .and_then(Value::as_array_mut)
        .context("iteration ladder missing kernels")?;
    let existing_names: BTreeSet<String> = kernels
        .iter()
        .filter_map(|kernel| {
            kernel
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect();
    for rule in &location_map.locations {
        if !existing_names.contains(&rule.location) {
            let detail = &details_by_location[rule.location.as_str()];
            if detail["necessary_gpu_s"].as_f64().unwrap_or(0.0) > 0.0 {
                kernels.push(json!({
                    "name": rule.location,
                    "kind": kind_by_location.get(rule.location.as_str()).copied().unwrap_or("unknown"),
                    "is_comm": false,
                    "rungs": {
                        "balanced": 0.0,
                        "per_config_best": 0.0,
                        "ignore_network": 0.0,
                        "hardware_limit": 0.0,
                    },
                }));
            }
        }
    }
    for kernel in kernels {
        let Some(name) = kernel.get("name").and_then(Value::as_str) else {
            continue;
        };
        let Some(mut detail) = details_by_location.remove(name) else {
            continue;
        };
        let hardware_limit = kernel["rungs"]["hardware_limit"].as_f64().unwrap_or(0.0);
        let necessary_gpu_s = detail["necessary_gpu_s"].as_f64().unwrap_or(0.0);
        let (under_accounted_gpu_s, under_accounted_raw_gpu_s, accounting_tolerance_gpu_s) =
            under_accounted_difference(hardware_limit, necessary_gpu_s);
        detail["redundant_gpu_s"] = json!((hardware_limit - necessary_gpu_s).max(0.0));
        detail["under_accounted_gpu_s"] = json!(under_accounted_gpu_s);
        detail["under_accounted_raw_gpu_s"] = json!(under_accounted_raw_gpu_s);
        detail["accounting_tolerance_gpu_s"] = json!(accounting_tolerance_gpu_s);
        kernel["rungs"]["necessary_limit"] = json!(necessary_gpu_s);
        kernel["necessary_work"] = detail;
    }
    ladder["rungs"]["segmented_necessary"] = json!(segmented_necessary_gpu_s);
    ladder["rungs"]["hardware_necessary"] = json!(label.floors.necessary);
    ladder["special_chunks"]["fusion"] =
        json!((segmented_necessary_gpu_s - label.floors.necessary).max(0.0));

    Ok(LocationAttribution {
        mapping_id: location_map.mapping_id,
    })
}

fn validate_mapping(
    location_map: &LocationMap,
    kernel_locations: &[KernelLocation],
    segments: &[SemanticWork],
) -> Result<()> {
    if location_map.schema_version != 1 {
        bail!(
            "unsupported location-map schema {}",
            location_map.schema_version
        );
    }
    let expected_locations: BTreeSet<&str> = kernel_locations
        .iter()
        .filter(|location| !location.is_communication)
        .map(|location| location.name.as_str())
        .collect();
    let mapped_locations: BTreeSet<&str> = location_map
        .locations
        .iter()
        .map(|rule| rule.location.as_str())
        .collect();
    if expected_locations != mapped_locations {
        bail!(
            "location map mismatch; missing={:?}, extra={:?}",
            expected_locations
                .difference(&mapped_locations)
                .collect::<Vec<_>>(),
            mapped_locations
                .difference(&expected_locations)
                .collect::<Vec<_>>()
        );
    }
    if mapped_locations.len() != location_map.locations.len() {
        bail!("location map contains duplicate location rows");
    }
    let expected_semantics: BTreeSet<&str> = segments
        .iter()
        .map(|segment| segment.name.as_str())
        .collect();
    let mapped_semantics: Vec<&str> = location_map
        .locations
        .iter()
        .flat_map(|rule| rule.semantics.iter().map(String::as_str))
        .collect();
    let unique_semantics: BTreeSet<&str> = mapped_semantics.iter().copied().collect();
    if unique_semantics.len() != mapped_semantics.len() {
        bail!("location map consumes a semantic row more than once");
    }
    if expected_semantics != unique_semantics {
        bail!(
            "semantic map mismatch; missing={:?}, extra={:?}",
            expected_semantics
                .difference(&unique_semantics)
                .collect::<Vec<_>>(),
            unique_semantics
                .difference(&expected_semantics)
                .collect::<Vec<_>>()
        );
    }
    Ok(())
}

fn load_location_map(repo_root: &Path, arch_type: &str) -> Result<LocationMap> {
    let directory = repo_root.join("model/work/location_maps");
    for entry in std::fs::read_dir(&directory)
        .with_context(|| format!("read location-map directory {}", directory.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
            continue;
        }
        let map: LocationMap = serde_json::from_slice(&std::fs::read(&path)?)
            .with_context(|| format!("parse {}", path.display()))?;
        if map
            .arch_types
            .iter()
            .any(|candidate| candidate == arch_type)
        {
            return Ok(map);
        }
    }
    Err(anyhow!(
        "no semantic location map for arch type {arch_type:?}"
    ))
}

fn pool_arch(log_dir: &Path, pool_tag: &str) -> Result<Value> {
    let params: Value = serde_json::from_slice(&std::fs::read(log_dir.join("raw/params.json"))?)?;
    params["pools"][pool_tag]["groups"]
        .as_array()
        .and_then(|groups| groups.first())
        .and_then(|group| group.get("arch"))
        .cloned()
        .with_context(|| format!("params missing first arch for pool {pool_tag:?}"))
}

fn pool_arch_type(log_dir: &Path, pool_tag: &str) -> Result<String> {
    pool_arch(log_dir, pool_tag)?["type"]
        .as_str()
        .map(str::to_string)
        .context("pool arch missing type")
}

fn pool_compute_dtype(log_dir: &Path, pool_tag: &str) -> Result<String> {
    Ok(
        if pool_arch(log_dir, pool_tag)?["fp8"]
            .as_bool()
            .unwrap_or(false)
        {
            "fp8".to_string()
        } else {
            "bf16".to_string()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn location(name: &str) -> KernelLocation {
        KernelLocation {
            name: name.to_string(),
            kind: "single_gemm".to_string(),
            is_communication: false,
        }
    }

    fn segment(name: &str) -> SemanticWork {
        SemanticWork {
            name: name.to_string(),
            flops: 1.0,
            bytes: 2.0,
        }
    }

    #[test]
    fn mapping_requires_exact_location_and_semantic_coverage() {
        let complete = LocationMap {
            schema_version: 1,
            mapping_id: "test".to_string(),
            arch_types: vec!["test".to_string()],
            locations: vec![LocationRule {
                location: "model.gemm".to_string(),
                semantics: vec!["gemm".to_string()],
            }],
        };
        assert!(validate_mapping(&complete, &[location("model.gemm")], &[segment("gemm")]).is_ok());

        let duplicate_semantic = LocationMap {
            locations: vec![
                LocationRule {
                    location: "model.gemm".to_string(),
                    semantics: vec!["gemm".to_string()],
                },
                LocationRule {
                    location: "model.norm".to_string(),
                    semantics: vec!["gemm".to_string()],
                },
            ],
            ..complete
        };
        assert!(validate_mapping(
            &duplicate_semantic,
            &[location("model.gemm"), location("model.norm")],
            &[segment("gemm")],
        )
        .is_err());
    }
}
