//! Pooled DSA indexer back: selected pools to the packed sparse-attention page table.
//!
//! One launch per decode step and DSA layer: `batch_size * next_n` rows, each
//! with `top_k` selected pools of `kpool` tokens and its trailing partial pool,
//! mapped through the block table and packed to the front of a
//! `page_table_width` row with a valid count. It takes the place of the fork's
//! six post-top-k launches (buffer fill, pool-id cast, seq-len + 1, pool
//! expand, buffer copy, index remap).
//!
//! Physical coordinates are `(batch_size, context_len)`, `context_len` in
//! pools of the longest row; `next_n` is config identity (as for
//! `dsa_persistent_topk_decode`). The batch axis is a generic row axis
//! (1..2048 rows, wave edges of 148 SMs) divided by `next_n`.

use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaPooledTopkPageTableKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub next_n: u32,
    /// Tokens one request's block table addresses.
    pub max_model_len: Dim,
    /// Pools selected per row.
    pub top_k: u32,
    pub kpool: u32,
    pub page_table_width: u32,
    pub block_size: u32,
    pub index_dtype: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DsaPooledTopkPageTableKernelInput {
    /// Decode requests.
    pub batch_size: u32,
    /// Complete pools of the longest row.
    pub context_len: u32,
}

/// Rows of one decode launch the grid brackets: powers of two, the 148-SM
/// wave edges (1 and 2 CTAs per SM) and the decode/verify batches between.
pub(crate) const ROW_AXIS: [u32; 24] = [
    1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 296, 320, 384, 448, 512, 592, 640, 768, 896, 1024,
    1280, 1536, 2048,
];

/// The row axis in requests of `next_n` rows each.
pub(crate) fn batch_axis(next_n: u32) -> Vec<f64> {
    let mut batch: Vec<u32> = ROW_AXIS
        .iter()
        .map(|rows| rows.div_ceil(next_n.max(1)))
        .collect();
    batch.dedup();
    Axis::values(batch)
}

/// Pools of the longest row. The page-table build reads one block-table entry
/// per selected pool, so time follows the rows and barely the context (rows
/// shorter than `top_k` pools carry -1 holes).
const CONTEXT_AXIS: [u32; 7] = [16, 128, 512, 2048, 16384, 65536, 262144];

pub struct DsaPooledTopkPageTableSpec;

impl KernelSpec for DsaPooledTopkPageTableSpec {
    type Config = DsaPooledTopkPageTableKernelConfig;
    type Input = DsaPooledTopkPageTableKernelInput;

    const KIND: KernelKind = "dsa_pooled_topk_page_table";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![batch_axis(config.next_n), Axis::values(CONTEXT_AXIS)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        let max_pools = f64::from(config.max_model_len.get() / config.kpool.max(1));
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
                .with("top_k", config.top_k)
                .with("kpool", config.kpool)
                .with("page_table_width", config.page_table_width)
                .with("block_size", config.block_size)
                .with("index_dtype", config.index_dtype.clone())
        })
    }
}

register_kernel!(DsaPooledTopkPageTableKernel, DsaPooledTopkPageTableSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn config(next_n: u32) -> DsaPooledTopkPageTableKernelConfig {
        DsaPooledTopkPageTableKernelConfig {
            backends: vec!["cuda_row_scan_compact"],
            gpu_name: "NVIDIA B200".to_string(),
            next_n,
            max_model_len: Dim::param("max_model_len", 1_048_576),
            top_k: 512,
            kpool: 4,
            page_table_width: 2176,
            block_size: 64,
            index_dtype: "int32".to_string(),
        }
    }

    #[test]
    fn enumerate_matches_python_wire_schema() {
        let config = config(8);
        let grid = DsaPooledTopkPageTableSpec::sweep_grid(&config);
        let payloads =
            DsaPooledTopkPageTableSpec::enumerate(&config, &grid, "cuda_row_scan_compact");
        let fields = payloads[0].fields();
        assert_eq!(fields.len(), 10);
        assert_eq!(fields.get("batch_size"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("context_len"), Some(&Value::from(16_u32)));
        assert_eq!(fields.get("next_n"), Some(&Value::from(8_u32)));
        assert_eq!(
            fields.get("max_model_len"),
            Some(&Value::from(1_048_576_u32))
        );
        assert_eq!(fields.get("page_table_width"), Some(&Value::from(2176_u32)));
    }

    #[test]
    fn batch_axis_covers_rows_1_to_2048_per_next_n() {
        for next_n in [1, 4, 8] {
            let grid = DsaPooledTopkPageTableSpec::sweep_grid(&config(next_n));
            let batch = &grid.axes()[0];
            assert_eq!(batch.first(), Some(&1.0));
            assert_eq!(batch.last(), Some(&f64::from(2048 / next_n)));
            assert!(batch.windows(2).all(|w| w[0] < w[1]));
            // The reference decode shapes: 16 / 32 / 80 requests of 8 rows.
            if next_n == 8 {
                assert!([16.0, 32.0, 80.0].iter().all(|b| batch.contains(b)));
            }
            assert!(batch.len() * grid.axes()[1].len() <= 500);
        }
        assert!(matches!(
            DsaPooledTopkPageTableSpec::cache_kind("cuda_row_scan_compact"),
            CacheKind::Cache2DLinear(Extrapolation::Product)
        ));
    }

    #[test]
    fn input_coordinates_are_requests_and_pools() {
        let input = DsaPooledTopkPageTableKernelInput {
            batch_size: 16,
            context_len: 59_000,
        };
        assert_eq!(&*input.coords(), &[16.0, 59_000.0]);
    }
}
