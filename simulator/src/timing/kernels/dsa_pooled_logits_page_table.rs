//! Pooled DSA indexer back: indexer logits to the packed page table, one launch.
//!
//! The decode top-k over pooled indexer logits (`dsa_persistent_topk_decode`
//! at `top_k` pools) and the page-table build after it
//! (`dsa_pooled_topk_page_table`) fused: each row's top pools are written out
//! directly as physical page-table slots with a valid count. Coordinates and
//! the batch axis follow `dsa_pooled_topk_page_table`; the context axis spans
//! the long-context decode range, where time grows with the logits read.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::dsa_pooled_topk_page_table::batch_axis;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaPooledLogitsPageTableKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub next_n: u32,
    /// Tokens one request's block table addresses.
    pub max_model_len: Dim,
    /// Elements between logit rows (the fork's `max_model_len / kpool`).
    pub logits_row_stride: Dim,
    pub top_k: u32,
    pub kpool: u32,
    pub page_table_width: u32,
    pub block_size: u32,
    #[compute_dtype]
    pub logits_dtype: DType,
    pub index_dtype: String,
    pub context_mode: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DsaPooledLogitsPageTableKernelInput {
    /// Decode requests.
    pub batch_size: u32,
    /// Complete pools of the longest row.
    pub context_len: u32,
}

/// Pools of the longest row: short rows (<= top_k, every pool kept), the
/// threshold kernel's 4096-candidate capacity, and powers of two to 1M tokens.
const CONTEXT_AXIS: [u32; 13] = [
    16, 128, 511, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
];

pub struct DsaPooledLogitsPageTableSpec;

impl KernelSpec for DsaPooledLogitsPageTableSpec {
    type Config = DsaPooledLogitsPageTableKernelConfig;
    type Input = DsaPooledLogitsPageTableKernelInput;

    const KIND: KernelKind = "dsa_pooled_logits_page_table";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![batch_axis(config.next_n), Axis::values(CONTEXT_AXIS)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // One CTA per row streams the row's logits: work is rows x pools.
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        let max_pools = f64::from(config.max_model_len.get() / config.kpool.max(1))
            .min(f64::from(config.logits_row_stride.get()));
        let min_pools = f64::from((config.next_n + 3).div_ceil(config.kpool.max(1)));
        grid.expand_2d(|_, context_len| context_len > max_pools || context_len < min_pools)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|batch_size, context_len| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", batch_size as u32)
                .with("context_len", context_len as u32)
                .with("next_n", config.next_n)
                .with("max_model_len", config.max_model_len.get())
                .with("logits_row_stride", config.logits_row_stride.get())
                .with("top_k", config.top_k)
                .with("kpool", config.kpool)
                .with("page_table_width", config.page_table_width)
                .with("block_size", config.block_size)
                .with("logits_dtype", config.logits_dtype.as_str())
                .with("index_dtype", config.index_dtype.clone())
                .with("context_mode", config.context_mode.clone())
        })
    }
}

register_kernel!(DsaPooledLogitsPageTableKernel, DsaPooledLogitsPageTableSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn config(next_n: u32) -> DsaPooledLogitsPageTableKernelConfig {
        DsaPooledLogitsPageTableKernelConfig {
            backends: vec!["cuda_threshold_filter_radix"],
            gpu_name: "NVIDIA B200".to_string(),
            next_n,
            max_model_len: Dim::param("max_model_len", 1_048_576),
            logits_row_stride: Dim::param("logits_row_stride", 262_144),
            top_k: 512,
            kpool: 4,
            page_table_width: 2176,
            block_size: 64,
            logits_dtype: DType::Fp32,
            index_dtype: "int32".to_string(),
            context_mode: "uniform_measured".to_string(),
        }
    }

    #[test]
    fn enumerate_matches_python_wire_schema() {
        let config = config(8);
        let grid = DsaPooledLogitsPageTableSpec::sweep_grid(&config);
        let payloads =
            DsaPooledLogitsPageTableSpec::enumerate(&config, &grid, "cuda_threshold_filter_radix");
        let fields = payloads[0].fields();
        assert_eq!(fields.len(), 13);
        assert_eq!(fields.get("logits_row_stride"), Some(&Value::from(262_144_u32)));
        assert_eq!(fields.get("logits_dtype"), Some(&Value::from("fp32")));
        assert_eq!(fields.get("context_mode"), Some(&Value::from("uniform_measured")));
    }

    #[test]
    fn grid_brackets_short_rows_and_long_context_within_budget() {
        for next_n in [1, 4, 8] {
            let config = config(next_n);
            let grid = DsaPooledLogitsPageTableSpec::sweep_grid(&config);
            let context = &grid.axes()[1];
            assert!(context.contains(&512.0) && context.contains(&4096.0));
            assert_eq!(context.last(), Some(&262_144.0));
            let feasible = DsaPooledLogitsPageTableSpec::infeasible_mask(&config, &grid)
                .iter()
                .filter(|&&masked| !masked)
                .count();
            assert!(feasible <= 500);
        }
    }

    #[test]
    fn input_coordinates_are_requests_and_pools() {
        let input = DsaPooledLogitsPageTableKernelInput {
            batch_size: 32,
            context_len: 54_000,
        };
        assert_eq!(&*input.coords(), &[32.0, 54_000.0]);
    }
}
