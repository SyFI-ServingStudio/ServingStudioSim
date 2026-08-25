//! DeepSeek V4 fused QR/KV RMSNorm launch.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeepseekV4FusedQKvRmsnormKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub q_dim: Dim,
    pub kv_dim: Dim,
    pub rms_eps_bits: u64,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DeepseekV4FusedQKvRmsnormKernelInput {
    pub num_tokens: u32,
}

pub struct DeepseekV4FusedQKvRmsnormSpec;

impl KernelSpec for DeepseekV4FusedQKvRmsnormSpec {
    type Config = DeepseekV4FusedQKvRmsnormKernelConfig;
    type Input = DeepseekV4FusedQKvRmsnormKernelInput;
    const KIND: KernelKind = "deepseek_v4_fused_q_kv_rmsnorm";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert_eq!(config.q_dim.get(), 1536);
        assert_eq!(config.kv_dim.get(), 512);
        assert_eq!(f64::from_bits(config.rms_eps_bits), 1.0e-6);
        let mut tokens = Axis::chain([Axis::pow2(0, 4), Axis::token_axis()]);
        tokens.sort_by(f64::total_cmp);
        tokens.dedup();
        SweepGrid::new(vec![tokens])
    }

    fn cache_kind(_: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", tokens as u32)
                .with("q_dim", config.q_dim.get())
                .with("kv_dim", config.kv_dim.get())
                .with("rms_eps", f64::from_bits(config.rms_eps_bits))
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(
    DeepseekV4FusedQKvRmsnormKernel,
    DeepseekV4FusedQKvRmsnormSpec
);
