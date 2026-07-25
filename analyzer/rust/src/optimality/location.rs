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

use super::floors::{Floors, IterationLabel, SemanticWork, WorkerComposition};
use super::ladder::{
    KernelContribution, KernelLadder, KernelNecessaryWork, KernelRungs, NecessaryWorkPolicy,
};
use super::prepare::KernelLocation;
use super::spec::GpuSpec;

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

#[derive(Deserialize)]
struct ParamsDocument {
    pools: BTreeMap<String, PoolParams>,
}

#[derive(Deserialize)]
struct PoolParams {
    groups: Vec<GroupParams>,
}

#[derive(Deserialize)]
struct GroupParams {
    arch: ArchParams,
}

#[derive(Deserialize)]
struct ArchParams {
    #[serde(rename = "type")]
    arch_type: String,
    #[serde(default)]
    fp8: bool,
}

struct PoolModelSpec {
    arch_type: String,
    compute_dtype: &'static str,
}

/// Run-scoped semantic attribution inputs. Params and every mapping file are
/// parsed once, then reused across workers and exact-iteration projections.
pub(super) struct LocationCatalog {
    maps: Vec<LocationMap>,
    pool_specs: BTreeMap<String, PoolModelSpec>,
}

pub(super) struct LocationAttribution {
    pub(super) mapping_id: String,
}

#[derive(Clone, Copy, Default)]
struct NecessaryWork {
    flops: f64,
    bytes: f64,
    roofline_gpu_s: f64,
}

impl LocationCatalog {
    pub(super) fn load(repo_root: &Path, log_dir: &Path) -> Result<Self> {
        let params: ParamsDocument =
            serde_json::from_slice(&std::fs::read(log_dir.join("raw/params.json"))?)
                .context("parse raw/params.json for semantic attribution")?;
        let mut pool_specs = BTreeMap::new();
        for (pool_tag, pool) in params.pools {
            let group = pool
                .groups
                .into_iter()
                .next()
                .with_context(|| format!("params has no group for pool {pool_tag:?}"))?;
            pool_specs.insert(
                pool_tag,
                PoolModelSpec {
                    arch_type: group.arch.arch_type,
                    compute_dtype: if group.arch.fp8 { "fp8" } else { "bf16" },
                },
            );
        }

        let directory = repo_root.join("model/work/location_maps");
        let mut maps = Vec::new();
        for entry in std::fs::read_dir(&directory)
            .with_context(|| format!("read location-map directory {}", directory.display()))?
        {
            let path = entry?.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                continue;
            }
            maps.push(
                serde_json::from_slice(&std::fs::read(&path)?)
                    .with_context(|| format!("parse {}", path.display()))?,
            );
        }
        maps.sort_by(|left: &LocationMap, right: &LocationMap| {
            left.mapping_id.cmp(&right.mapping_id)
        });
        Ok(Self { maps, pool_specs })
    }

    /// Add R6/R7 and per-location work to one typed ladder. The semantic label
    /// may describe an exact iteration or a saturated run aggregate. Mutation is
    /// transactional: every validation and reconciliation succeeds before the
    /// caller's ladder is replaced.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attribute_ladder(
        &self,
        pool_tag: &str,
        kernel_locations: &[KernelLocation],
        ladder: &mut KernelLadder,
        label: &IterationLabel,
        gpu_spec: GpuSpec,
        gpu_count: f64,
        policy: NecessaryWorkPolicy,
    ) -> Result<LocationAttribution> {
        self.attribute_label_refs(
            pool_tag,
            kernel_locations,
            ladder,
            &[(label, 1)],
            gpu_spec,
            gpu_count,
            policy,
        )
    }

    /// Attribute a run-level worker composition. A batch-locked composition may
    /// contain many distinct iteration shapes with occurrence counts; saturated
    /// composition contains one already-normalized large-batch label.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attribute_composed_ladder(
        &self,
        pool_tag: &str,
        kernel_locations: &[KernelLocation],
        ladder: &mut KernelLadder,
        composition: &WorkerComposition,
        gpu_spec: GpuSpec,
        gpu_count: f64,
        policy: NecessaryWorkPolicy,
    ) -> Result<LocationAttribution> {
        let label_refs: Vec<(&IterationLabel, u64)> = composition
            .labels
            .iter()
            .map(|weighted| (&weighted.label, weighted.occurrences))
            .collect();
        self.attribute_label_refs(
            pool_tag,
            kernel_locations,
            ladder,
            &label_refs,
            gpu_spec,
            gpu_count,
            policy,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attribute_label_refs(
        &self,
        pool_tag: &str,
        kernel_locations: &[KernelLocation],
        ladder: &mut KernelLadder,
        labels: &[(&IterationLabel, u64)],
        gpu_spec: GpuSpec,
        gpu_count: f64,
        policy: NecessaryWorkPolicy,
    ) -> Result<LocationAttribution> {
        if labels.is_empty() {
            bail!("semantic attribution has no workload labels");
        }
        let pool_spec = self
            .pool_specs
            .get(pool_tag)
            .with_context(|| format!("semantic attribution missing pool {pool_tag:?}"))?;
        let location_map = select_location_map(&self.maps, &pool_spec.arch_type, kernel_locations)?;
        let peak_tflops = gpu_spec.peak_tflops(pool_spec.compute_dtype);
        let bandwidth_gbps = gpu_spec.mem_bandwidth_gbps;
        if peak_tflops <= 0.0 || bandwidth_gbps <= 0.0 {
            bail!(
                "GPU spec lacks positive {} compute peak or memory bandwidth",
                pool_spec.compute_dtype
            );
        }

        let kind_by_location: BTreeMap<&str, &str> = kernel_locations
            .iter()
            .map(|location| (location.name.as_str(), location.kind.as_str()))
            .collect();
        let mut work_by_location: BTreeMap<&str, KernelNecessaryWork> = BTreeMap::new();
        let mut expected_floors = Floors::default();
        for (label, occurrences) in labels {
            validate_mapping(location_map, kernel_locations, &label.segments)?;
            let occurrence_scale = *occurrences as f64;
            expected_floors.fused += label.floors.fused * occurrence_scale;
            expected_floors.segmented += label.floors.segmented * occurrence_scale;
            let semantic_work: BTreeMap<&str, NecessaryWork> = label
                .segments
                .iter()
                .map(|segment| {
                    (
                        segment.name.as_str(),
                        NecessaryWork {
                            flops: segment.flops,
                            bytes: segment.bytes,
                            roofline_gpu_s: segment.necessary_gpu_s,
                        },
                    )
                })
                .collect();
            for rule in &location_map.locations {
                let work = rule.semantics.iter().try_fold(
                    NecessaryWork::default(),
                    |mut total, semantic| {
                        let row = semantic_work.get(semantic.as_str()).with_context(|| {
                            format!("mapped semantic row {semantic:?} is absent")
                        })?;
                        total.flops += row.flops;
                        total.bytes += row.bytes;
                        total.roofline_gpu_s += row.roofline_gpu_s;
                        Ok::<_, anyhow::Error>(total)
                    },
                )?;
                let mut weighted_work = KernelNecessaryWork::from_rates_with_roofline(
                    rule.semantics.iter().cloned(),
                    work.flops,
                    work.bytes,
                    peak_tflops,
                    bandwidth_gbps,
                    work.roofline_gpu_s,
                    gpu_count,
                );
                weighted_work.scale(occurrence_scale);
                work_by_location
                    .entry(rule.location.as_str())
                    .or_default()
                    .add_assign_for_policy(&weighted_work, policy);
            }
        }
        for work in work_by_location.values_mut() {
            work.set_worker_wall_time(gpu_count);
        }

        let mut attributed_ladder = ladder.clone();
        let existing_names: BTreeSet<String> = attributed_ladder
            .kernels
            .iter()
            .map(|kernel| kernel.name.clone())
            .collect();
        for rule in &location_map.locations {
            if !existing_names.contains(&rule.location)
                && work_by_location[rule.location.as_str()].necessary_gpu_s() > 0.0
            {
                attributed_ladder.kernels.push(KernelContribution {
                    name: rule.location.clone(),
                    kind: kind_by_location
                        .get(rule.location.as_str())
                        .copied()
                        .unwrap_or("unknown")
                        .to_string(),
                    is_comm: false,
                    rungs: KernelRungs::default(),
                    necessary_work: None,
                });
            }
        }
        for kernel in &mut attributed_ladder.kernels {
            kernel.necessary_work = if kernel.is_comm {
                Some(KernelNecessaryWork::zero())
            } else {
                work_by_location.remove(kernel.name.as_str())
            };
        }
        attributed_ladder.finalize_necessary_work(policy, Some(expected_floors.fused))?;
        let segmented_necessary_gpu_s = attributed_ladder
            .rungs
            .segmented_necessary
            .context("finalized ladder missing segmented floor")?;
        let scope_fused_necessary_gpu_s = attributed_ladder
            .rungs
            .scope_fused_necessary
            .context("finalized ladder missing fused floor")?;
        let tolerance = (expected_floors.segmented.abs() * 1e-7).max(1e-12);
        if (segmented_necessary_gpu_s - expected_floors.segmented).abs() > tolerance
            || (scope_fused_necessary_gpu_s - expected_floors.fused).abs() > tolerance
        {
            bail!(
                "mapped location floors ({segmented_necessary_gpu_s}, {scope_fused_necessary_gpu_s}) do not reconcile with semantic floors ({}, {})",
                expected_floors.segmented,
                expected_floors.fused
            );
        }
        *ladder = attributed_ladder;

        Ok(LocationAttribution {
            mapping_id: location_map.mapping_id.clone(),
        })
    }
}

fn select_location_map<'map>(
    maps: &'map [LocationMap],
    arch_type: &str,
    kernel_locations: &[KernelLocation],
) -> Result<&'map LocationMap> {
    let expected_locations: BTreeSet<&str> = kernel_locations
        .iter()
        .filter(|location| !location.is_communication)
        .map(|location| location.name.as_str())
        .collect();
    let matches: Vec<&LocationMap> = maps
        .iter()
        .filter(|location_map| {
            location_map
                .arch_types
                .iter()
                .any(|candidate| candidate == arch_type)
        })
        .filter(|location_map| {
            location_map
                .locations
                .iter()
                .map(|rule| rule.location.as_str())
                .collect::<BTreeSet<_>>()
                == expected_locations
        })
        .collect();
    match matches.as_slice() {
        [location_map] => Ok(*location_map),
        [] => Err(anyhow!(
            "no semantic location map matches arch type {arch_type:?} and its exact manifest locations"
        )),
        _ => Err(anyhow!(
            "multiple semantic location maps match arch type {arch_type:?} and its exact manifest locations: {:?}",
            matches
                .iter()
                .map(|location_map| location_map.mapping_id.as_str())
                .collect::<Vec<_>>()
        )),
    }
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
            necessary_gpu_s: 2.0,
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

    #[test]
    fn map_selection_disambiguates_layouts_that_share_an_arch_type() {
        let maps = vec![
            LocationMap {
                schema_version: 1,
                mapping_id: "unified".to_string(),
                arch_types: vec!["llama3_dense".to_string()],
                locations: vec![LocationRule {
                    location: "unified.gemm".to_string(),
                    semantics: vec!["gemm".to_string()],
                }],
            },
            LocationMap {
                schema_version: 1,
                mapping_id: "pd".to_string(),
                arch_types: vec!["llama3_dense".to_string()],
                locations: vec![LocationRule {
                    location: "pd.gemm".to_string(),
                    semantics: vec!["gemm".to_string()],
                }],
            },
        ];

        let selected = select_location_map(&maps, "llama3_dense", &[location("pd.gemm")]).unwrap();
        assert_eq!(selected.mapping_id, "pd");
    }
}
