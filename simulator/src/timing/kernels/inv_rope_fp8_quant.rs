//! Fused inverse-RoPE and grouped FP8 quantization.

use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InvRopeFp8QuantKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct InvRopeFp8QuantKernelInput {
    pub num_tokens: u32,
}

pub struct InvRopeFp8QuantSpec;

impl KernelSpec for InvRopeFp8QuantSpec {
    type Config = InvRopeFp8QuantKernelConfig;
    type Input = InvRopeFp8QuantKernelInput;

    const KIND: KernelKind = "inv_rope_fp8_quant";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // The production Triton kernel switches from one to four tokens per
        // program at 512, so retain both adjacent sides of that boundary.
        let mut tokens = Axis::chain([
            Axis::pow2(0, 8),
            Axis::values([511, 512, 513]),
            Axis::token_axis()
                .into_iter()
                .filter(|&num_tokens| num_tokens <= 8192.0)
                .collect(),
        ]);
        tokens.sort_by(f64::total_cmp);
        tokens.dedup();
        SweepGrid::new(vec![tokens])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        _config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
        })
    }
}

register_kernel!(InvRopeFp8QuantKernel, InvRopeFp8QuantSpec);

#[cfg(test)]
mod tests {
    use super::{InvRopeFp8QuantKernelConfig, InvRopeFp8QuantSpec};
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn sweep_preserves_the_production_coarsening_boundary() {
        let config = InvRopeFp8QuantKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA H200".to_string(),
        };
        let grid = InvRopeFp8QuantSpec::sweep_grid(&config);
        let axis = &grid.axes()[0];
        for num_tokens in [1.0, 511.0, 512.0, 513.0, 8192.0] {
            assert!(axis.contains(&num_tokens));
        }
        assert!(axis.iter().all(|&num_tokens| num_tokens <= 8192.0));
    }
}
