//! DeepSeek routed-expert reduction over the top-k expert axis.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeSumKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub top_k: u32,
    pub hidden_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MoeSumKernelInput {
    pub num_tokens: u32,
}

pub struct MoeSumSpec;

impl KernelSpec for MoeSumSpec {
    type Config = MoeSumKernelConfig;
    type Input = MoeSumKernelInput;

    const KIND: KernelKind = "moe_sum";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::chain([Axis::pow2(0, 4), Axis::token_axis()])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
                .with("top_k", config.top_k)
                .with("hidden_dim", config.hidden_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(MoeSumKernel, MoeSumSpec);

#[cfg(test)]
mod tests {
    use super::{MoeSumKernelConfig, MoeSumSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn decode_and_prefill_token_sizes_share_one_axis() {
        let config = MoeSumKernelConfig {
            backends: vec!["vllm_cuda"],
            gpu_name: "NVIDIA H200".to_string(),
            top_k: 6,
            hidden_dim: 4096.into(),
            dtype: DType::Bf16,
        };
        let payloads =
            MoeSumSpec::enumerate(&config, &MoeSumSpec::sweep_grid(&config), "vllm_cuda");
        assert_eq!(payloads.first().unwrap().fields()["num_tokens"], 1);
        assert_eq!(payloads.last().unwrap().fields()["num_tokens"], 65_536);
        assert_eq!(payloads[0].fields()["top_k"], 6);
        assert_eq!(payloads[0].fields()["hidden_dim"], 4096);
    }
}
