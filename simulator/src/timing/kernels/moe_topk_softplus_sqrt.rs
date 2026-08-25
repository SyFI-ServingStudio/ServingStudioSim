//! DeepSeek learned/hash sqrt-softplus router selection.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeTopkSoftplusSqrtKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub selection_mode: String,
    pub num_experts: Dim,
    pub top_k: u32,
    pub hash_vocab_size: u32,
    #[compute_dtype]
    pub logits_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MoeTopkSoftplusSqrtKernelInput {
    pub num_tokens: u32,
}

pub struct MoeTopkSoftplusSqrtSpec;

impl KernelSpec for MoeTopkSoftplusSqrtSpec {
    type Config = MoeTopkSoftplusSqrtKernelConfig;
    type Input = MoeTopkSoftplusSqrtKernelInput;

    const KIND: KernelKind = "moe_topk_softplus_sqrt";

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
                .with("selection_mode", config.selection_mode.clone())
                .with("num_tokens", num_tokens as u32)
                .with("num_experts", config.num_experts.get())
                .with("top_k", config.top_k)
                .with("hash_vocab_size", config.hash_vocab_size)
                .with("logits_dtype", config.logits_dtype.as_str())
        })
    }
}

register_kernel!(MoeTopkSoftplusSqrtKernel, MoeTopkSoftplusSqrtSpec);

#[cfg(test)]
mod tests {
    use super::{MoeTopkSoftplusSqrtKernelConfig, MoeTopkSoftplusSqrtSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn learned_and_hash_modes_remain_distinct_cache_configs() {
        let learned = MoeTopkSoftplusSqrtKernelConfig {
            backends: vec!["vllm_cuda"],
            gpu_name: "NVIDIA H200".to_string(),
            selection_mode: "learned".to_string(),
            num_experts: 256.into(),
            top_k: 6,
            hash_vocab_size: 0,
            logits_dtype: DType::Fp32,
        };
        let mut hash = learned.clone();
        hash.selection_mode = "hash".to_string();
        hash.hash_vocab_size = 129_280;
        assert_ne!(learned, hash);

        let payload = MoeTopkSoftplusSqrtSpec::enumerate(
            &hash,
            &MoeTopkSoftplusSqrtSpec::sweep_grid(&hash),
            "vllm_cuda",
        )
        .remove(0);
        assert_eq!(payload.fields()["selection_mode"], "hash");
        assert_eq!(payload.fields()["hash_vocab_size"], 129_280);
    }
}
