//! Typed kernel-ladder domain and the single worker -> pool -> cluster reducer.
//!
//! R0..R5 are additive GPU-second measurements. R6/R7 follow the declared policy:
//! batch-locked composition adds rooflines already evaluated at each fixed-batch
//! boundary, while saturated composition adds (`flops`, `bytes`) and reevaluates
//! rooflines at the wider scope. Keeping this distinction here prevents a hierarchy
//! caller from silently rebatching locked iterations.
//! JSON exists only at [`KernelLadder::to_json`], the publication boundary.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use super::levels::BaseRungs;
use super::{ms_to_s, under_accounted_difference};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AggregateLevel {
    Cluster,
    Pool,
}

impl AggregateLevel {
    fn wire_name(self) -> &'static str {
        match self {
            Self::Cluster => "cluster",
            Self::Pool => "pool",
        }
    }
}

#[derive(Clone, Debug)]
pub(super) enum LadderScope {
    Worker {
        pool_tag: String,
        worker_id: u16,
    },
    Aggregate {
        level: AggregateLevel,
        key: String,
        label: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum NecessaryWorkPolicy {
    BatchLocked,
    Saturated { replication_factor: u32 },
}

impl NecessaryWorkPolicy {
    fn wire_mode(self) -> &'static str {
        match self {
            Self::BatchLocked => "batch_locked",
            Self::Saturated { .. } => "replicated_large_batch",
        }
    }

    fn replication_factor(self) -> u32 {
        match self {
            Self::BatchLocked => 1,
            Self::Saturated { replication_factor } => replication_factor,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct LadderRungs {
    pub(super) real: f64,
    pub(super) busy: f64,
    pub(super) balanced: f64,
    pub(super) per_config_best: f64,
    pub(super) ignore_network: f64,
    pub(super) hardware_limit: f64,
    pub(super) segmented_necessary: Option<f64>,
    pub(super) scope_fused_necessary: Option<f64>,
}

impl LadderRungs {
    pub(super) fn from_gpu_ms(rungs: &BaseRungs) -> Self {
        Self {
            real: ms_to_s(rungs.real),
            busy: ms_to_s(rungs.busy),
            balanced: ms_to_s(rungs.balanced),
            per_config_best: ms_to_s(rungs.per_config_best),
            ignore_network: ms_to_s(rungs.ignore_network),
            hardware_limit: ms_to_s(rungs.hardware_limit),
            segmented_necessary: None,
            scope_fused_necessary: None,
        }
    }

    fn add_base(&mut self, other: Self) {
        self.real += other.real;
        self.busy += other.busy;
        self.balanced += other.balanced;
        self.per_config_best += other.per_config_best;
        self.ignore_network += other.ignore_network;
        self.hardware_limit += other.hardware_limit;
    }

    fn to_json(self) -> Value {
        let mut value = json!({
            "real": self.real,
            "busy": self.busy,
            "balanced": self.balanced,
            "per_config_best": self.per_config_best,
            "ignore_network": self.ignore_network,
            "hardware_limit": self.hardware_limit,
        });
        if let Some(segmented_necessary) = self.segmented_necessary {
            value["segmented_necessary"] = json!(segmented_necessary);
        }
        if let Some(scope_fused_necessary) = self.scope_fused_necessary {
            // Keep the v1 wire name for compatibility. Saturated scopes recompute
            // this rung from work; batch-locked scopes add their fixed-batch R7s.
            value["hardware_necessary"] = json!(scope_fused_necessary);
        }
        value
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SpecialChunks {
    pub(super) idle: f64,
    pub(super) imbalance: f64,
    pub(super) fusion: Option<f64>,
}

impl SpecialChunks {
    fn from_rungs(rungs: LadderRungs) -> Self {
        Self {
            idle: (rungs.real - rungs.busy).max(0.0),
            imbalance: (rungs.busy - rungs.balanced).max(0.0),
            fusion: None,
        }
    }

    fn add_base(&mut self, other: Self) {
        self.idle += other.idle;
        self.imbalance += other.imbalance;
    }

    fn to_json(self) -> Value {
        let mut value = json!({
            "idle": self.idle,
            "imbalance": self.imbalance,
        });
        if let Some(fusion) = self.fusion {
            value["fusion"] = json!(fusion);
        }
        value
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct KernelRungs {
    pub(super) balanced: f64,
    pub(super) per_config_best: f64,
    pub(super) ignore_network: f64,
    pub(super) hardware_limit: f64,
}

impl KernelRungs {
    pub(super) fn from_gpu_ms(rungs: [f64; 4]) -> Self {
        Self {
            balanced: ms_to_s(rungs[0]),
            per_config_best: ms_to_s(rungs[1]),
            ignore_network: ms_to_s(rungs[2]),
            hardware_limit: ms_to_s(rungs[3]),
        }
    }

    fn add_assign(&mut self, other: Self) {
        self.balanced += other.balanced;
        self.per_config_best += other.per_config_best;
        self.ignore_network += other.ignore_network;
        self.hardware_limit += other.hardware_limit;
    }
}

/// Minimum semantic work assigned to one exact manifest location.
///
/// `compute_gpu_s` and `memory_gpu_s` are the two additive roofline terms.
/// `necessary_gpu_s` is deliberately derived as their max at the current scope.
#[derive(Clone, Debug, Default)]
pub(super) struct KernelNecessaryWork {
    pub(super) semantics: BTreeSet<String>,
    pub(super) min_flops: f64,
    pub(super) min_bytes: f64,
    pub(super) compute_gpu_s: f64,
    pub(super) memory_gpu_s: f64,
    roofline_gpu_s: f64,
    pub(super) wall_s: Option<f64>,
}

impl KernelNecessaryWork {
    /// Roll several semantic rows up into one location.
    ///
    /// `compute_gpu_s` arrives already summed by the caller rather than being
    /// derived here from one peak: a model is mixed-precision, so the rows behind
    /// one location can divide by different peaks (an FP8 GEMM and the BF16
    /// FlashMLA kernel next to it).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_rates_with_roofline(
        semantics: impl IntoIterator<Item = String>,
        min_flops: f64,
        min_bytes: f64,
        compute_gpu_s: f64,
        bandwidth_gbps: f64,
        roofline_gpu_s: f64,
        gpu_count: f64,
    ) -> Self {
        let memory_gpu_s = min_bytes / (bandwidth_gbps * 1e9);
        Self::from_gpu_seconds(
            semantics,
            min_flops,
            min_bytes,
            compute_gpu_s,
            memory_gpu_s,
            roofline_gpu_s,
            gpu_count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_gpu_seconds(
        semantics: impl IntoIterator<Item = String>,
        min_flops: f64,
        min_bytes: f64,
        compute_gpu_s: f64,
        memory_gpu_s: f64,
        roofline_gpu_s: f64,
        gpu_count: f64,
    ) -> Self {
        // Per-kernel GPU·seconds are published in the same footing as the ladder's
        // per-rank rungs: the labeler labels the WHOLE model on one GPU, while each
        // worker contributes F/(G·peak) GPU·seconds. Normalize compute/memory/
        // roofline by the worker's GPU count so per-kernel necessary == hardware
        // limit; min_flops/min_bytes keep the whole-model scale as information.
        let gpus = gpu_count.max(1.0);
        Self {
            semantics: semantics.into_iter().collect(),
            min_flops,
            min_bytes,
            compute_gpu_s: compute_gpu_s / gpus,
            memory_gpu_s: memory_gpu_s / gpus,
            roofline_gpu_s: roofline_gpu_s / gpus,
            wall_s: Some(roofline_gpu_s / gpus),
        }
    }

    pub(super) fn zero() -> Self {
        Self {
            wall_s: Some(0.0),
            ..Self::default()
        }
    }

    pub(super) fn necessary_gpu_s(&self) -> f64 {
        self.roofline_gpu_s
    }

    pub(super) fn scale(&mut self, factor: f64) {
        self.min_flops *= factor;
        self.min_bytes *= factor;
        self.compute_gpu_s *= factor;
        self.memory_gpu_s *= factor;
        self.roofline_gpu_s *= factor;
        self.wall_s = self.wall_s.map(|wall_s| wall_s * factor);
    }

    pub(super) fn add_assign_for_policy(&mut self, other: &Self, policy: NecessaryWorkPolicy) {
        self.semantics.extend(other.semantics.iter().cloned());
        self.min_flops += other.min_flops;
        self.min_bytes += other.min_bytes;
        self.compute_gpu_s += other.compute_gpu_s;
        self.memory_gpu_s += other.memory_gpu_s;
        self.roofline_gpu_s = match policy {
            NecessaryWorkPolicy::BatchLocked => self.roofline_gpu_s + other.roofline_gpu_s,
            NecessaryWorkPolicy::Saturated { .. } => self.compute_gpu_s.max(self.memory_gpu_s),
        };
        // Wall time is not additive across parallel workers. Aggregate ladders
        // intentionally publish only GPU seconds.
        self.wall_s = None;
    }

    pub(super) fn set_worker_wall_time(&mut self, gpu_count: f64) {
        self.wall_s = Some(self.roofline_gpu_s / gpu_count.max(1.0));
    }

    fn to_json(&self, hardware_limit_gpu_s: f64) -> Value {
        let necessary_gpu_s = self.necessary_gpu_s();
        let (under_accounted_gpu_s, under_accounted_raw_gpu_s, accounting_tolerance_gpu_s) =
            under_accounted_difference(hardware_limit_gpu_s, necessary_gpu_s);
        let mut value = json!({
            "semantics": self.semantics,
            "min_flops": self.min_flops,
            "min_bytes": self.min_bytes,
            "compute_gpu_s": self.compute_gpu_s,
            "memory_gpu_s": self.memory_gpu_s,
            "necessary_gpu_s": necessary_gpu_s,
            "redundant_gpu_s": (hardware_limit_gpu_s - necessary_gpu_s).max(0.0),
            "under_accounted_gpu_s": under_accounted_gpu_s,
            "under_accounted_raw_gpu_s": under_accounted_raw_gpu_s,
            "accounting_tolerance_gpu_s": accounting_tolerance_gpu_s,
            "bound": if self.compute_gpu_s >= self.memory_gpu_s { "compute" } else { "memory" },
        });
        if let Some(wall_s) = self.wall_s {
            value["wall_s"] = json!(wall_s);
        }
        value
    }
}

#[derive(Clone, Debug)]
pub(super) struct KernelContribution {
    pub(super) name: String,
    pub(super) kind: String,
    pub(super) is_comm: bool,
    pub(super) rungs: KernelRungs,
    pub(super) necessary_work: Option<KernelNecessaryWork>,
}

impl KernelContribution {
    fn to_json(&self) -> Value {
        let necessary_limit = self
            .necessary_work
            .as_ref()
            .map(KernelNecessaryWork::necessary_gpu_s);
        let mut rung_value = json!({
            "balanced": self.rungs.balanced,
            "per_config_best": self.rungs.per_config_best,
            "ignore_network": self.rungs.ignore_network,
            "hardware_limit": self.rungs.hardware_limit,
        });
        if let Some(necessary_limit) = necessary_limit {
            rung_value["necessary_limit"] = json!(necessary_limit);
        }
        let mut value = json!({
            "name": self.name,
            "kind": self.kind,
            "is_comm": self.is_comm,
            "rungs": rung_value,
        });
        if let Some(necessary_work) = &self.necessary_work {
            value["necessary_work"] = necessary_work.to_json(self.rungs.hardware_limit);
        }
        value
    }
}

#[derive(Clone, Debug)]
pub(super) struct KernelLadder {
    pub(super) scope: LadderScope,
    pub(super) rungs: LadderRungs,
    pub(super) special_chunks: SpecialChunks,
    pub(super) kernels: Vec<KernelContribution>,
    pub(super) necessary_work_policy: Option<NecessaryWorkPolicy>,
}

impl KernelLadder {
    pub(super) fn worker(
        pool_tag: &str,
        worker_id: u16,
        rungs_gpu_ms: &BaseRungs,
        kernels: Vec<KernelContribution>,
    ) -> Self {
        let rungs = LadderRungs::from_gpu_ms(rungs_gpu_ms);
        Self {
            scope: LadderScope::Worker {
                pool_tag: pool_tag.to_string(),
                worker_id,
            },
            rungs,
            special_chunks: SpecialChunks::from_rungs(rungs),
            kernels,
            necessary_work_policy: None,
        }
    }

    pub(super) fn worker_ref(&self) -> Option<(&str, u16)> {
        match &self.scope {
            LadderScope::Worker {
                pool_tag,
                worker_id,
            } => Some((pool_tag, *worker_id)),
            LadderScope::Aggregate { .. } => None,
        }
    }

    pub(super) fn has_necessary_work(&self) -> bool {
        self.rungs.segmented_necessary.is_some()
            && self.rungs.scope_fused_necessary.is_some()
            && self.necessary_work_policy.is_some()
            && self
                .kernels
                .iter()
                .all(|kernel| kernel.necessary_work.is_some())
    }

    /// Finish one strict location attribution. Both lower rungs are derived from
    /// the typed location work, so callers cannot create a non-reconciling ladder.
    pub(super) fn finalize_necessary_work(
        &mut self,
        policy: NecessaryWorkPolicy,
        scope_fused_override: Option<f64>,
    ) -> Result<()> {
        if self
            .kernels
            .iter()
            .any(|kernel| kernel.necessary_work.is_none())
        {
            bail!("necessary work must cover every emitted kernel location");
        }
        let segmented_necessary: f64 = self
            .kernels
            .iter()
            .filter_map(|kernel| kernel.necessary_work.as_ref())
            .map(KernelNecessaryWork::necessary_gpu_s)
            .sum();
        let compute_gpu_s: f64 = self
            .kernels
            .iter()
            .filter_map(|kernel| kernel.necessary_work.as_ref())
            .map(|work| work.compute_gpu_s)
            .sum();
        let memory_gpu_s: f64 = self
            .kernels
            .iter()
            .filter_map(|kernel| kernel.necessary_work.as_ref())
            .map(|work| work.memory_gpu_s)
            .sum();
        let scope_fused_necessary =
            scope_fused_override.unwrap_or_else(|| compute_gpu_s.max(memory_gpu_s));
        self.rungs.segmented_necessary = Some(segmented_necessary);
        self.rungs.scope_fused_necessary = Some(scope_fused_necessary);
        self.special_chunks.fusion = Some((segmented_necessary - scope_fused_necessary).max(0.0));
        self.necessary_work_policy = Some(policy);
        self.validate()
    }

    /// The only hierarchy reducer for kernel ladders. R0..R5 add directly.
    /// Saturated R6/R7 reevaluate after adding work; batch-locked R6/R7 add child
    /// fixed-batch rooflines. A partial child set keeps the parent at R0..R5 instead
    /// of misrepresenting missing work as zero.
    pub(super) fn aggregate(
        level: AggregateLevel,
        key: &str,
        label: &str,
        members: &[&Self],
    ) -> Result<Self> {
        if members.is_empty() {
            bail!("cannot aggregate an empty kernel-ladder scope");
        }

        let mut rungs = LadderRungs::default();
        let mut special_chunks = SpecialChunks::default();
        for member in members {
            rungs.add_base(member.rungs);
            special_chunks.add_base(member.special_chunks);
        }

        let any_necessary_work = members.iter().any(|member| member.has_necessary_work());
        let all_necessary_work = members.iter().all(|member| member.has_necessary_work());
        let common_policy = if all_necessary_work {
            let policy = members[0]
                .necessary_work_policy
                .context("complete ladder missing necessary-work policy")?;
            if members
                .iter()
                .any(|member| member.necessary_work_policy != Some(policy))
            {
                bail!("aggregate ladder mixes necessary-work policies");
            }
            Some(policy)
        } else {
            None
        };

        let mut kernels_by_name: BTreeMap<String, KernelContribution> = BTreeMap::new();
        for member in members {
            for kernel in &member.kernels {
                let aggregate = kernels_by_name
                    .entry(kernel.name.clone())
                    .or_insert_with(|| KernelContribution {
                        name: kernel.name.clone(),
                        kind: kernel.kind.clone(),
                        is_comm: kernel.is_comm,
                        rungs: KernelRungs::default(),
                        necessary_work: all_necessary_work.then(KernelNecessaryWork::default),
                    });
                if aggregate.kind != kernel.kind || aggregate.is_comm != kernel.is_comm {
                    bail!(
                        "kernel location {:?} changes kind or communication class across workers",
                        kernel.name
                    );
                }
                aggregate.rungs.add_assign(kernel.rungs);
                if all_necessary_work {
                    aggregate
                        .necessary_work
                        .as_mut()
                        .expect("initialized when every member has necessary work")
                        .add_assign_for_policy(
                            kernel
                                .necessary_work
                                .as_ref()
                                .context("complete ladder kernel missing necessary work")?,
                            common_policy.expect("complete members have one policy"),
                        );
                }
            }
        }
        if any_necessary_work && !all_necessary_work {
            for kernel in kernels_by_name.values_mut() {
                kernel.necessary_work = None;
            }
        }

        let mut kernels: Vec<KernelContribution> = kernels_by_name.into_values().collect();
        kernels.sort_by(|left, right| {
            right
                .rungs
                .balanced
                .total_cmp(&left.rungs.balanced)
                .then_with(|| left.name.cmp(&right.name))
        });
        let mut aggregate = Self {
            scope: LadderScope::Aggregate {
                level,
                key: key.to_string(),
                label: label.to_string(),
            },
            rungs,
            special_chunks,
            kernels,
            necessary_work_policy: None,
        };
        if let Some(policy) = common_policy {
            match policy {
                NecessaryWorkPolicy::BatchLocked => {
                    let segmented: f64 = members
                        .iter()
                        .filter_map(|member| member.rungs.segmented_necessary)
                        .sum();
                    let scope_fused: f64 = members
                        .iter()
                        .filter_map(|member| member.rungs.scope_fused_necessary)
                        .sum();
                    aggregate.rungs.segmented_necessary = Some(segmented);
                    aggregate.rungs.scope_fused_necessary = Some(scope_fused);
                    aggregate.special_chunks.fusion = Some((segmented - scope_fused).max(0.0));
                    aggregate.necessary_work_policy = Some(policy);
                    aggregate.validate()?;
                }
                NecessaryWorkPolicy::Saturated { .. } => {
                    aggregate.finalize_necessary_work(policy, None)?;
                }
            }
        } else {
            aggregate.validate()?;
        }
        Ok(aggregate)
    }

    pub(super) fn to_json(&self) -> Result<Value> {
        self.validate()?;
        let (key, label, identity) = match &self.scope {
            LadderScope::Worker {
                pool_tag,
                worker_id,
            } => {
                let key = format!("{pool_tag}/{worker_id}");
                (
                    key.clone(),
                    key,
                    json!({"pool_tag": pool_tag, "worker_id": worker_id}),
                )
            }
            LadderScope::Aggregate { level, key, label } => (
                key.clone(),
                label.clone(),
                json!({"level": level.wire_name()}),
            ),
        };
        let mut value = json!({
            "key": key,
            "label": label,
            "rungs": self.rungs.to_json(),
            "special_chunks": self.special_chunks.to_json(),
            "kernels": self.kernels.iter().map(KernelContribution::to_json).collect::<Vec<_>>(),
        });
        let object = value
            .as_object_mut()
            .expect("kernel ladder serialization starts as an object");
        object.extend(
            identity
                .as_object()
                .expect("ladder identity is an object")
                .clone(),
        );
        if let Some(policy) = self.necessary_work_policy {
            value["necessary_work_mode"] = json!(policy.wire_mode());
            value["necessary_work_replication_factor"] = json!(policy.replication_factor());
        }
        Ok(value)
    }

    fn validate(&self) -> Result<()> {
        for (rung_name, kernel_sum, expected) in [
            (
                "balanced",
                self.kernels
                    .iter()
                    .map(|kernel| kernel.rungs.balanced)
                    .sum::<f64>(),
                self.rungs.balanced,
            ),
            (
                "per_config_best",
                self.kernels
                    .iter()
                    .map(|kernel| kernel.rungs.per_config_best)
                    .sum::<f64>(),
                self.rungs.per_config_best,
            ),
            (
                "ignore_network",
                self.kernels
                    .iter()
                    .map(|kernel| kernel.rungs.ignore_network)
                    .sum::<f64>(),
                self.rungs.ignore_network,
            ),
            (
                "hardware_limit",
                self.kernels
                    .iter()
                    .map(|kernel| kernel.rungs.hardware_limit)
                    .sum::<f64>(),
                self.rungs.hardware_limit,
            ),
        ] {
            // Per-worker anchoring and hierarchy reduction visit the same terms
            // in different orders. Keep this stricter than the wire decoder's
            // 1e-6 tolerance while allowing ordinary multi-worker FP summation.
            let tolerance = expected.abs().max(1.0) * 1e-7;
            if (kernel_sum - expected).abs() > tolerance {
                bail!(
                    "kernel ladder rung {rung_name:?} does not reconcile: {kernel_sum} vs {expected}"
                );
            }
        }
        let base_tolerance = self.rungs.real.abs().max(1.0) * 1e-7;
        if ((self.rungs.balanced + self.special_chunks.imbalance) - self.rungs.busy).abs()
            > base_tolerance
            || ((self.rungs.busy + self.special_chunks.idle) - self.rungs.real).abs()
                > base_tolerance
        {
            bail!("kernel ladder special chunks do not reconcile with R0/R1/R2");
        }
        match (
            self.rungs.segmented_necessary,
            self.rungs.scope_fused_necessary,
            self.special_chunks.fusion,
        ) {
            (Some(segmented), Some(scope_fused), Some(fusion)) => {
                let kernel_sum: f64 = self
                    .kernels
                    .iter()
                    .map(|kernel| {
                        kernel
                            .necessary_work
                            .as_ref()
                            .map(KernelNecessaryWork::necessary_gpu_s)
                            .unwrap_or(0.0)
                    })
                    .sum();
                let tolerance = segmented.abs().max(1.0) * 1e-9;
                if (kernel_sum - segmented).abs() > tolerance
                    || ((scope_fused + fusion) - segmented).abs() > tolerance
                    || scope_fused > segmented + tolerance
                {
                    bail!("kernel ladder R6/R7 necessary work does not reconcile");
                }
            }
            (None, None, None) => {}
            _ => bail!("kernel ladder has partially available R6/R7 fields"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel(
        name: &str,
        balanced: f64,
        compute_gpu_s: f64,
        memory_gpu_s: f64,
    ) -> KernelContribution {
        KernelContribution {
            name: name.to_string(),
            kind: "single_gemm".to_string(),
            is_comm: false,
            rungs: KernelRungs {
                balanced,
                per_config_best: balanced,
                ignore_network: balanced,
                hardware_limit: balanced,
            },
            necessary_work: Some(KernelNecessaryWork {
                semantics: BTreeSet::from([name.to_string()]),
                min_flops: compute_gpu_s,
                min_bytes: memory_gpu_s,
                compute_gpu_s,
                memory_gpu_s,
                roofline_gpu_s: compute_gpu_s.max(memory_gpu_s),
                wall_s: Some(compute_gpu_s.max(memory_gpu_s)),
            }),
        }
    }

    fn worker(
        worker_id: u16,
        kernel: KernelContribution,
        policy: NecessaryWorkPolicy,
    ) -> KernelLadder {
        let balanced = kernel.rungs.balanced;
        let mut ladder = KernelLadder {
            scope: LadderScope::Worker {
                pool_tag: "main".to_string(),
                worker_id,
            },
            rungs: LadderRungs {
                real: balanced,
                busy: balanced,
                balanced,
                per_config_best: balanced,
                ignore_network: balanced,
                hardware_limit: balanced,
                ..LadderRungs::default()
            },
            special_chunks: SpecialChunks::default(),
            kernels: vec![kernel],
            necessary_work_policy: None,
        };
        ladder.finalize_necessary_work(policy, None).unwrap();
        ladder
    }

    #[test]
    fn per_kernel_necessary_is_normalized_by_gpu_count() {
        // The labeler labels the whole model on one GPU; per-kernel GPU·seconds
        // must be divided by the worker GPU count to match the per-rank rungs.
        let work = KernelNecessaryWork::from_gpu_seconds(
            ["qkv".to_string()],
            1.0,
            1.0,
            2.0,
            1.0,
            2.0,
            4.0,
        );
        assert_eq!(work.compute_gpu_s, 0.5);
        assert_eq!(work.memory_gpu_s, 0.25);
        assert_eq!(work.necessary_gpu_s(), 0.5);
        assert_eq!(work.wall_s, Some(0.5));
    }

    #[test]
    fn parent_recomputes_rooflines_after_bound_switching() {
        let policy = NecessaryWorkPolicy::Saturated {
            replication_factor: 10_000,
        };
        let compute_worker = worker(0, kernel("compute", 10.0, 10.0, 1.0), policy);
        let memory_worker = worker(1, kernel("memory", 10.0, 1.0, 10.0), policy);
        let aggregate = KernelLadder::aggregate(
            AggregateLevel::Cluster,
            "cluster",
            "Cluster aggregate",
            &[&compute_worker, &memory_worker],
        )
        .unwrap();

        // Segmented keeps the two locations separate: max(10,1)+max(1,10)=20.
        assert_eq!(aggregate.rungs.segmented_necessary, Some(20.0));
        // Scope-fused adds work first: max(10+1, 1+10)=11, not Σ child max=20.
        assert_eq!(aggregate.rungs.scope_fused_necessary, Some(11.0));
        assert_eq!(aggregate.special_chunks.fusion, Some(9.0));
        assert!(aggregate.kernels.iter().all(|kernel| kernel
            .necessary_work
            .as_ref()
            .unwrap()
            .wall_s
            .is_none()));
    }

    #[test]
    fn batch_locked_parent_adds_fixed_batch_rooflines_across_bound_switches() {
        let compute_iteration = worker(
            0,
            kernel("model.gemm", 10.0, 10.0, 1.0),
            NecessaryWorkPolicy::BatchLocked,
        );
        let memory_iteration = worker(
            1,
            kernel("model.gemm", 10.0, 1.0, 10.0),
            NecessaryWorkPolicy::BatchLocked,
        );
        let aggregate = KernelLadder::aggregate(
            AggregateLevel::Cluster,
            "cluster",
            "Cluster aggregate",
            &[&compute_iteration, &memory_iteration],
        )
        .unwrap();

        assert_eq!(
            aggregate.kernels[0]
                .necessary_work
                .as_ref()
                .unwrap()
                .necessary_gpu_s(),
            20.0
        );
        assert_eq!(aggregate.rungs.segmented_necessary, Some(20.0));
        assert_eq!(aggregate.rungs.scope_fused_necessary, Some(20.0));
        assert_eq!(aggregate.special_chunks.fusion, Some(0.0));
    }

    #[test]
    fn partial_child_attribution_does_not_turn_missing_work_into_zero() {
        let policy = NecessaryWorkPolicy::Saturated {
            replication_factor: 10_000,
        };
        let complete = worker(0, kernel("model.gemm", 10.0, 3.0, 1.0), policy);
        let mut missing = worker(1, kernel("model.gemm", 10.0, 3.0, 1.0), policy);
        missing.rungs.segmented_necessary = None;
        missing.rungs.scope_fused_necessary = None;
        missing.special_chunks.fusion = None;
        missing.necessary_work_policy = None;
        missing.kernels[0].necessary_work = None;

        let aggregate = KernelLadder::aggregate(
            AggregateLevel::Pool,
            "main",
            "main aggregate",
            &[&complete, &missing],
        )
        .unwrap();
        assert_eq!(aggregate.rungs.segmented_necessary, None);
        assert_eq!(aggregate.rungs.scope_fused_necessary, None);
        assert!(aggregate.kernels[0].necessary_work.is_none());
    }
}
