//! Dry-run profile coverage tree used before building runtime caches.
//!
//! Agent note: `JitPlan` mirrors `LookupResult` structurally. Keep aggregation
//! generic here so L2/L3/L4 can compose dry-run plans without kernel-specific code.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::timing::bridge::{BuildError, KernelKind, PerfApiBridge};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JitPlan {
    pub name: String,
    pub total_specs: usize,
    pub cached: usize,
    pub missing: usize,
    pub per_backend: Vec<BackendJitPlan>,
    pub breakdown: Vec<JitPlan>,
}

impl JitPlan {
    pub fn leaf(name: String, backend: &'static str, cached: usize, missing: usize) -> Self {
        Self {
            name,
            total_specs: cached + missing,
            cached,
            missing,
            per_backend: vec![BackendJitPlan {
                backend: backend.to_string(),
                cached,
                missing,
            }],
            breakdown: Vec::new(),
        }
    }

    /// Safe constructor from a `bridge.count_missing(...)` result: rejects the
    /// nonsensical `missing > total` case as `BuildError::FitFailed` before
    /// turning the counts into a `JitPlan::leaf`. Used by the engine's
    /// `dry_run` per-backend loop.
    pub fn from_missing_count(
        kernel_kind: KernelKind,
        name: &str,
        backend: &'static str,
        total_specs: usize,
        missing: usize,
    ) -> Result<Self, BuildError> {
        let cached = total_specs
            .checked_sub(missing)
            .ok_or_else(|| BuildError::FitFailed {
                kind: kernel_kind,
                reason: format!(
                    "count_missing returned {missing} missing rows for {total_specs} specs"
                ),
            })?;
        Ok(Self::leaf(
            format!("{name}.{backend}"),
            backend,
            cached,
            missing,
        ))
    }

    pub fn sum(name: String, parts: Vec<JitPlan>) -> Self {
        let cached = parts.iter().map(|part| part.cached).sum();
        let missing = parts.iter().map(|part| part.missing).sum();
        let per_backend = aggregate_backend_plans(&parts);
        Self {
            name,
            total_specs: cached + missing,
            cached,
            missing,
            per_backend,
            breakdown: parts,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendJitPlan {
    pub backend: String,
    pub cached: usize,
    pub missing: usize,
}

pub trait DryRun {
    type Config;

    fn dry_run(
        name: &str,
        config: &Self::Config,
        bridge: &PerfApiBridge,
    ) -> Result<JitPlan, BuildError>;
}

fn aggregate_backend_plans(parts: &[JitPlan]) -> Vec<BackendJitPlan> {
    let mut totals: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for part in parts {
        for backend_plan in &part.per_backend {
            let entry = totals.entry(backend_plan.backend.clone()).or_default();
            entry.0 += backend_plan.cached;
            entry.1 += backend_plan.missing;
        }
    }
    totals
        .into_iter()
        .map(|(backend, (cached, missing))| BackendJitPlan {
            backend,
            cached,
            missing,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::JitPlan;
    use crate::timing::BuildError;

    #[test]
    fn from_missing_count_rejects_impossible_counts() {
        assert!(matches!(
            JitPlan::from_missing_count("single_gemm", "gemm", "torch", 1, 2),
            Err(BuildError::FitFailed { .. })
        ));
    }

    #[test]
    fn jit_plan_sum_rolls_up_backend_counts() {
        let plan = JitPlan::sum(
            "model".to_string(),
            vec![
                JitPlan::leaf("a".to_string(), "torch", 3, 1),
                JitPlan::leaf("b".to_string(), "torch", 2, 4),
                JitPlan::leaf("c".to_string(), "triton", 5, 0),
            ],
        );

        assert_eq!(plan.total_specs, 15);
        assert_eq!(plan.cached, 10);
        assert_eq!(plan.missing, 5);
        assert_eq!(plan.per_backend.len(), 2);
        assert_eq!(plan.per_backend[0].backend, "torch");
        assert_eq!(plan.per_backend[0].cached, 5);
        assert_eq!(plan.per_backend[0].missing, 5);
        assert_eq!(plan.per_backend[1].backend, "triton");
        assert_eq!(plan.per_backend[1].cached, 5);
        assert_eq!(plan.per_backend[1].missing, 0);
    }
}
