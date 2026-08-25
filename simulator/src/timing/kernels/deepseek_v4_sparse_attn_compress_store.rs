//! DeepSeek V4 sparse-attention compressor/store compound operation.

use std::collections::BTreeSet;

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeepseekV4SparseAttnCompressStoreKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub compress_ratio: u32,
    pub num_kv_heads: u32,
    pub head_dim: Dim,
    pub rope_head_dim: Dim,
    pub logical_block_size: u32,
    #[compute_dtype]
    pub state_dtype: DType,
    pub norm_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
    pub cache_dtype: String,
    pub cache_layout: String,
    pub scale_format: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeepseekV4SparseAttnCompressStoreKernelInput {
    pub row_positions: Vec<u32>,
    pub row_request_ids: Vec<u32>,
    pub state_block_table_width: u32,
}

impl DeepseekV4SparseAttnCompressStoreKernelInput {
    fn work(&self, compress_ratio: u32) -> (u32, u32) {
        assert!(!self.row_positions.is_empty());
        assert_eq!(self.row_positions.len(), self.row_request_ids.len());
        assert!(self.row_positions.len() <= 8192);
        assert!(self.state_block_table_width > 0);
        let request_count = self.row_request_ids.iter().copied().max().unwrap() + 1;
        assert!(request_count <= 64);
        assert_eq!(
            self.row_request_ids
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
            (0..request_count).collect()
        );
        let active = self
            .row_positions
            .iter()
            .filter(|&&position| (position + 1) % compress_ratio == 0)
            .count() as u32;
        (self.row_positions.len() as u32, active)
    }
}

impl SweepCoords for DeepseekV4SparseAttnCompressStoreKernelInput {
    fn coords(&self) -> Coords {
        // The active-row coordinate depends on the Config's compression ratio.
        Coords::new([self.row_positions.len() as f64, 0.0])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["num_tokens", "num_active_tokens"]
    }
}

fn canonical_input(
    num_tokens: u32,
    num_active_tokens: u32,
    compress_ratio: u32,
) -> DeepseekV4SparseAttnCompressStoreKernelInput {
    let request_count = num_tokens.min(64);
    let row_request_ids = (0..num_tokens)
        .map(|row| row % request_count)
        .collect::<Vec<_>>();
    let row_positions = (0..num_tokens)
        .map(|row| {
            let request_ordinal = row / request_count;
            if row < num_active_tokens {
                compress_ratio - 1 + request_ordinal * compress_ratio
            } else {
                request_ordinal * compress_ratio
            }
        })
        .collect::<Vec<_>>();
    let state_block_size = if compress_ratio == 4 { 4 } else { 8 };
    let state_block_table_width =
        row_positions.iter().copied().max().unwrap_or(0) / state_block_size + 1;
    DeepseekV4SparseAttnCompressStoreKernelInput {
        row_positions,
        row_request_ids,
        state_block_table_width,
    }
}

pub struct DeepseekV4SparseAttnCompressStoreSpec;

fn validate_config(config: &DeepseekV4SparseAttnCompressStoreKernelConfig) {
    assert_eq!(config.gpu_name, "NVIDIA H200");
    assert_eq!(config.num_kv_heads, 1);
    assert_eq!(config.rope_head_dim.get(), 64);
    assert_eq!(config.logical_block_size, 256);
    assert_eq!(config.state_dtype, DType::Fp32);
    assert_eq!(config.norm_dtype, DType::Bf16);
    assert_eq!(config.kv_dtype, DType::Fp8E4m3);
    assert_eq!(config.cache_layout, "block_segregated_data_then_scales");
    match config.head_dim.get() {
        512 => {
            assert!(matches!(config.compress_ratio, 4 | 128));
            assert_eq!(config.backends, ["vllm_deepseek_v4_cutedsl"]);
            assert_eq!(config.cache_dtype, "fp8_ds_mla");
            assert_eq!(config.scale_format, "ue8m0");
        }
        128 => {
            assert_eq!(config.compress_ratio, 4);
            assert_eq!(config.backends, ["vllm_deepseek_v4_triton"]);
            assert_eq!(config.cache_dtype, "fp8_indexer");
            assert_eq!(config.scale_format, "fp32_per_token");
        }
        head_dim => panic!("unsupported compressor head_dim {head_dim}"),
    }
}

impl KernelSpec for DeepseekV4SparseAttnCompressStoreSpec {
    type Config = DeepseekV4SparseAttnCompressStoreKernelConfig;
    type Input = DeepseekV4SparseAttnCompressStoreKernelInput;

    const KIND: KernelKind = "deepseek_v4_sparse_attn_compress_store";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        validate_config(config);
        // This compound has two smooth work terms: save-partial-state scales
        // with all rows, while compress/store scales with boundary-active rows.
        // Crossing the dense shared token axis with itself would profile 1,595
        // feasible points per config without introducing a device dispatch
        // boundary. Power-of-two anchors retain decode through 8K prefill while
        // keeping the 2D interpolation surface measurable and auditable.
        SweepGrid::new(vec![
            Axis::pow2(0, 13),
            Axis::chain([Axis::values([0]), Axis::pow2(0, 13)]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        let (num_tokens, num_active_tokens) = input.work(config.compress_ratio);
        Coords::new([f64::from(num_tokens), f64::from(num_active_tokens)])
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_2d(|num_tokens, num_active_tokens| num_active_tokens > num_tokens)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|num_tokens, num_active_tokens| {
            let input = canonical_input(
                num_tokens as u32,
                num_active_tokens.min(num_tokens) as u32,
                config.compress_ratio,
            );
            ArgsPayload::new()
                .with("backend", backend)
                .with("row_positions", input.row_positions)
                .with("row_request_ids", input.row_request_ids)
                .with("state_block_table_width", input.state_block_table_width)
                .with("compress_ratio", config.compress_ratio)
                .with("num_kv_heads", config.num_kv_heads)
                .with("head_dim", config.head_dim.get())
                .with("rope_head_dim", config.rope_head_dim.get())
                .with("logical_block_size", config.logical_block_size)
                .with("rms_eps", 1.0e-6_f64)
                .with("state_dtype", config.state_dtype.as_str())
                .with("norm_dtype", config.norm_dtype.as_str())
                .with("cache_dtype", config.cache_dtype.clone())
                .with("cache_layout", config.cache_layout.clone())
                .with("scale_format", config.scale_format.clone())
        })
    }
}

register_kernel!(
    DeepseekV4SparseAttnCompressStoreKernel,
    DeepseekV4SparseAttnCompressStoreSpec
);

#[cfg(test)]
mod tests {
    use super::canonical_input;

    #[test]
    fn canonical_topology_has_the_requested_boundary_count() {
        for compress_ratio in [4, 128] {
            let input = canonical_input(128, 17, compress_ratio);
            assert_eq!(input.work(compress_ratio), (128, 17));
            assert_eq!(input.row_request_ids.iter().copied().max(), Some(63));
            for request in 0..64 {
                let positions = input
                    .row_positions
                    .iter()
                    .zip(&input.row_request_ids)
                    .filter_map(|(&position, &row_request)| {
                        (row_request == request).then_some(position)
                    })
                    .collect::<Vec<_>>();
                assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
            }
        }
    }
}
