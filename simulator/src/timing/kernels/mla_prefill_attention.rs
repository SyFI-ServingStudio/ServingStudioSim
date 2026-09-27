//! Kimi-K3 TRT-LLM ragged MLA prefill attention.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

const MAX_PROFILE_BYTES: u64 = 16 * 1024 * 1024 * 1024;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MlaPrefillAttentionKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub qk_head_dim: Dim,
    pub v_head_dim: Dim,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
    pub o_dtype: DType,
    pub causal: bool,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct MlaPrefillAttentionKernelInput {
    pub batch_size: u32,
    pub q_len: u32,
    pub kv_len: u32,
    pub prefix_len: u32,
    pub num_prefix_chunks: u32,
}

impl SweepCoords for MlaPrefillAttentionKernelInput {
    fn coords(&self) -> Coords {
        // The physical runner consumes kv_len per launch, while the cache
        // axis must represent the total prefix work across all launches.
        Coords::new([
            self.batch_size as f64,
            self.q_len as f64,
            self.prefix_len as f64,
        ])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["batch_size", "q_len", "prefix_len"]
    }
}

pub struct MlaPrefillAttentionSpec;

impl KernelSpec for MlaPrefillAttentionSpec {
    type Config = MlaPrefillAttentionKernelConfig;
    type Input = MlaPrefillAttentionKernelInput;

    const KIND: KernelKind = "mla_prefill_attention";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values([1, 4]),
            Axis::values([4096, 16_384, 32_768]),
            Axis::values([0, 49_152, 131_072, 229_376, 245_760]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_3d(|batch_size, q_len, prefix_len| {
            let batch_size = batch_size.round() as u64;
            let q_len = q_len.round() as u64;
            let prefix_len = prefix_len.round() as u64;
            let chunk_len = if prefix_len == 0 {
                q_len
            } else {
                prefix_len.min(131_072 / batch_size.max(1))
            };
            let bytes = batch_size
                .saturating_mul(chunk_len)
                .saturating_mul((config.qk_head_dim.get() + config.v_head_dim.get()) as u64)
                .saturating_mul(config.kv_dtype.size_bytes() as u64);
            bytes > MAX_PROFILE_BYTES
                || (config.causal && prefix_len != 0)
                || (!config.causal && prefix_len == 0)
                || (!config.causal && batch_size != 1)
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_3d(|batch_size, q_len, prefix_len| {
            let batch_size = batch_size.round() as u32;
            let q_len = q_len.round() as u32;
            let prefix_len = prefix_len.round() as u32;
            let (kv_len, num_prefix_chunks) = if prefix_len == 0 {
                (q_len, 0)
            } else {
                let chunk_capacity = (131_072 / u64::from(batch_size.max(1))) as u32;
                let chunk_len = prefix_len.min(chunk_capacity);
                let chunks = (prefix_len + chunk_len - 1) / chunk_len;
                (chunk_len, chunks)
            };
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_heads", config.num_heads.get())
                .with("qk_head_dim", config.qk_head_dim.get())
                .with("v_head_dim", config.v_head_dim.get())
                .with("q_dtype", config.q_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("o_dtype", config.o_dtype.as_str())
                .with("causal", config.causal)
                .with("batch_size", batch_size)
                .with("q_len", q_len)
                .with("kv_len", kv_len)
                .with("prefix_len", prefix_len)
                .with("num_prefix_chunks", num_prefix_chunks)
        })
    }
}

register_kernel!(MlaPrefillAttentionKernel, MlaPrefillAttentionSpec);
