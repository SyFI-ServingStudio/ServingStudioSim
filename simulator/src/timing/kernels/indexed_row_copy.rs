//! Eager PyTorch indexed copy of whole rows: gather, scatter or masked zero.
//!
//! Recurrent-state models move one state row per prefilling request between a
//! slot pool and a dense tensor with plain tensor indexing. The row size and the
//! operation are static identity; the number of rows is the runtime axis. Small
//! row counts dominate (tens of prefill requests per step), and the per-launch
//! cost there is not a byte rate, so the axis is dense at the low end.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IndexedRowCopyKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub row_elements: Dim,
    #[compute_dtype]
    pub dtype: DType,
    /// `gather`, `scatter` or `masked_zero`.
    pub operation: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct IndexedRowCopyKernelInput {
    pub num_rows: u32,
}

pub struct IndexedRowCopySpec;

impl KernelSpec for IndexedRowCopySpec {
    type Config = IndexedRowCopyKernelConfig;
    type Input = IndexedRowCopyKernelInput;

    const KIND: KernelKind = "indexed_row_copy";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::values([
            1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128,
        ])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_rows| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_rows", num_rows as u32)
                .with("row_elements", config.row_elements.get())
                .with("dtype", config.dtype.as_str())
                .with("operation", config.operation.clone())
        })
    }
}

register_kernel!(IndexedRowCopyKernel, IndexedRowCopySpec);

#[cfg(test)]
mod tests {
    use super::{IndexedRowCopyKernelConfig, IndexedRowCopySpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;
    use crate::timing::Dim;
    use serde_json::Value;

    /// Catches a payload that drifts from the Python `IndexedRowCopyArgs`
    /// (a renamed or dropped field turns every lookup into a miss).
    #[test]
    fn enumerate_matches_python_wire_schema() {
        let config = IndexedRowCopyKernelConfig {
            backends: vec!["torch"],
            gpu_name: "NVIDIA B200".to_string(),
            row_elements: Dim::from(786_432),
            dtype: DType::Fp32,
            operation: "gather".to_string(),
        };
        let grid = IndexedRowCopySpec::sweep_grid(&config);
        let payloads = IndexedRowCopySpec::enumerate(&config, &grid, "torch");
        assert_eq!(payloads.len(), 14);
        let fields = payloads[2].fields();
        assert_eq!(fields.len(), 5);
        assert_eq!(fields.get("num_rows"), Some(&Value::from(3_u32)));
        assert_eq!(fields.get("row_elements"), Some(&Value::from(786_432_u32)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("fp32")));
        assert_eq!(fields.get("operation"), Some(&Value::from("gather")));
    }
}
