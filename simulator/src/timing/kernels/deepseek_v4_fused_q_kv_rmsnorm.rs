//! Fused QR/KV RMSNorm launch (DeepSeek V4; GLM-5.3 via the `vllm_fork_triton` backend).

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
        // DeepSeek V4 uses 1e-6, GLM-5.3 1e-5; the scalar never changes the launch.
        let eps = f64::from_bits(config.rms_eps_bits);
        assert!(eps == 1.0e-6 || eps == 1.0e-5, "unsupported rms_eps {eps}");
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

#[cfg(test)]
mod tests {
    use super::{DeepseekV4FusedQKvRmsnormKernelConfig, DeepseekV4FusedQKvRmsnormSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;
    use serde_json::Value;

    #[test]
    fn glm53_epsilon_is_forwarded_unchanged() {
        let config = DeepseekV4FusedQKvRmsnormKernelConfig {
            backends: vec!["vllm_fork_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            q_dim: 1536.into(),
            kv_dim: 512.into(),
            rms_eps_bits: 1.0e-5_f64.to_bits(),
            dtype: DType::Bf16,
        };
        let grid = DeepseekV4FusedQKvRmsnormSpec::sweep_grid(&config);
        let payload =
            &DeepseekV4FusedQKvRmsnormSpec::enumerate(&config, &grid, "vllm_fork_triton")[0];
        assert_eq!(payload.fields().get("rms_eps"), Some(&Value::from(1.0e-5)));
    }
}
