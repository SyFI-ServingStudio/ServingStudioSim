//! Eager PyTorch indexed copy of whole rows: gather, scatter or masked zero of
//! state-pool rows, and index_select / index_copy of token rows.
//!
//! Recurrent-state models move one state row per prefilling request between a
//! slot pool and a dense tensor with plain tensor indexing. The row size and the
//! operation are static identity; the number of rows is the runtime axis. Small
//! row counts dominate (tens of prefill requests per step), and the per-launch
//! cost there is not a byte rate, so the axis is dense at the low end. The
//! token-row operations split and merge the rows of a step that mixes
//! speculative and plain rows, so their axis runs to a whole step.

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
    /// `gather`, `scatter` or `masked_zero` (state rows), or `index_select` /
    /// `index_copy` (token rows of a step mixing speculative and plain rows).
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

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        if is_token_row_operation(&config.operation) {
            // Token rows run to a whole step. PyTorch's index_select takes a
            // per-index path up to 16 indices and a vectorized gather from 17, a
            // step down in time, so both sides of that boundary are grid points.
            return SweepGrid::new(vec![Axis::values([
                1, 2, 4, 8, 12, 16, 17, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
                1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384,
            ])]);
        }
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

/// Operations over the token rows of a step rather than state-pool slots.
fn is_token_row_operation(operation: &str) -> bool {
    matches!(operation, "index_select" | "index_copy")
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

    /// Token-row operations sweep a whole step and keep both sides of
    /// index_select's 16/17-index path switch; state-row operations keep
    /// their grid so existing rows stay valid.
    #[test]
    fn token_row_operations_sweep_a_whole_step() {
        let config = |operation: &str| IndexedRowCopyKernelConfig {
            backends: vec!["torch"],
            gpu_name: "NVIDIA B200".to_string(),
            row_elements: Dim::from(10_240),
            dtype: DType::Bf16,
            operation: operation.to_string(),
        };
        for operation in ["index_select", "index_copy"] {
            let grid = IndexedRowCopySpec::sweep_grid(&config(operation));
            let payloads = IndexedRowCopySpec::enumerate(&config(operation), &grid, "torch");
            let rows: Vec<u64> = payloads
                .iter()
                .map(|p| p.fields().get("num_rows").and_then(Value::as_u64).unwrap())
                .collect();
            assert!(rows.windows(2).all(|w| w[0] < w[1]));
            assert!(rows.contains(&16) && rows.contains(&17));
            assert_eq!(rows.last(), Some(&16_384));
        }
        let grid = IndexedRowCopySpec::sweep_grid(&config("scatter"));
        assert_eq!(IndexedRowCopySpec::enumerate(&config("scatter"), &grid, "torch").len(), 14);
    }
}
