//! Kimi-K3 TRT-LLM ragged MLA prefill attention.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

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

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MlaPrefillAttentionKernelInput {
    pub batch_size: u32,
    pub q_len: u32,
    pub kv_len: u32,
}

pub struct MlaPrefillAttentionSpec;

impl KernelSpec for MlaPrefillAttentionSpec {
    type Config = MlaPrefillAttentionKernelConfig;
    type Input = MlaPrefillAttentionKernelInput;

    const KIND: KernelKind = "mla_prefill_attention";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values([1, 2, 4, 8]),
            Axis::values([1024, 4096, 16_384]),
            Axis::values([1024, 4096, 16_384, 49_152]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_3d(|batch_size, q_len, kv_len| {
            let batch_size = batch_size.round() as u64;
            let q_len = q_len.round() as u64;
            let kv_len = kv_len.round() as u64;
            let bytes = batch_size
                .saturating_mul(kv_len)
                .saturating_mul((config.qk_head_dim.get() + config.v_head_dim.get()) as u64)
                .saturating_mul(config.kv_dtype.size_bytes() as u64);
            bytes > MAX_PROFILE_BYTES
                || (config.causal && q_len != kv_len)
                || (!config.causal && kv_len < q_len)
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_3d(|batch_size, q_len, kv_len| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_heads", config.num_heads.get())
                .with("qk_head_dim", config.qk_head_dim.get())
                .with("v_head_dim", config.v_head_dim.get())
                .with("q_dtype", config.q_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("o_dtype", config.o_dtype.as_str())
                .with("causal", config.causal)
                .with("batch_size", batch_size.round() as u32)
                .with("q_len", q_len.round() as u32)
                .with("kv_len", kv_len.round() as u32)
        })
    }
}

register_kernel!(MlaPrefillAttentionKernel, MlaPrefillAttentionSpec);
