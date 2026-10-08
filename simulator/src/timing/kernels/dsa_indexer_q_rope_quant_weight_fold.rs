//! Fused DSA indexer-query RoPE and FP8 quantization, scales folded into the head weights.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::causal_rows::query_axis;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::SweepGrid;
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaIndexerQRopeQuantWeightFoldKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub rope_dim: Dim,
    pub max_model_len: u32,
    pub max_num_batched_tokens: u32,
    pub positions_dtype: String,
    #[compute_dtype]
    pub q_dtype: DType,
    pub rope_dtype: DType,
    pub weight_dtype: DType,
    pub q_output_dtype: DType,
    pub weight_output_dtype: DType,
    pub rope_style: String,
    pub quant_mode: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DsaIndexerQRopeQuantWeightFoldKernelInput {
    pub num_tokens: u32,
}

pub struct DsaIndexerQRopeQuantWeightFoldSpec;

impl KernelSpec for DsaIndexerQRopeQuantWeightFoldSpec {
    type Config = DsaIndexerQRopeQuantWeightFoldKernelConfig;
    type Input = DsaIndexerQRopeQuantWeightFoldKernelInput;

    const KIND: KernelKind = "dsa_indexer_q_rope_quant_weight_fold";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![query_axis(
            &[
                1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 511, 512, 513, 768, 1024, 2048, 4096, 8192,
            ],
            config.max_num_batched_tokens,
        )])
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
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("rope_dim", config.rope_dim.get())
                .with("max_model_len", config.max_model_len)
                .with("max_num_batched_tokens", config.max_num_batched_tokens)
                .with("index_weights_softmax_scale", 0.088_388_347_648_318_45_f64)
                .with("index_weights_head_scale", 0.125_f64)
                .with("fp8_max", 448.0_f64)
                .with("scale_epsilon", 1.0e-4_f64)
                .with("positions_dtype", config.positions_dtype.clone())
                .with("q_dtype", config.q_dtype.as_str())
                .with("rope_dtype", config.rope_dtype.as_str())
                .with("weight_dtype", config.weight_dtype.as_str())
                .with("q_output_dtype", config.q_output_dtype.as_str())
                .with("weight_output_dtype", config.weight_output_dtype.as_str())
                .with("rope_style", config.rope_style.clone())
                .with("quant_mode", config.quant_mode.clone())
        })
    }
}

register_kernel!(
    DsaIndexerQRopeQuantWeightFoldKernel,
    DsaIndexerQRopeQuantWeightFoldSpec
);

#[cfg(test)]
mod tests {
    use super::{DsaIndexerQRopeQuantWeightFoldKernelConfig, DsaIndexerQRopeQuantWeightFoldSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn sweep_keeps_the_source_coarsening_boundary() {
        let config = DsaIndexerQRopeQuantWeightFoldKernelConfig {
            backends: vec!["vllm_cutedsl_fp8"],
            gpu_name: "NVIDIA H200".to_string(),
            num_heads: 64.into(),
            head_dim: 128.into(),
            rope_dim: 64.into(),
            max_model_len: 1_048_576,
            max_num_batched_tokens: 8192,
            positions_dtype: "int64".to_string(),
            q_dtype: DType::Bf16,
            rope_dtype: DType::Fp32,
            weight_dtype: DType::Bf16,
            q_output_dtype: DType::Fp8E4m3,
            weight_output_dtype: DType::Fp32,
            rope_style: "gptj_interleaved_trailing".to_string(),
            quant_mode: "per_token_head_fp8_pow2_ceil_folded_weight".to_string(),
        };
        let grid = DsaIndexerQRopeQuantWeightFoldSpec::sweep_grid(&config);
        assert_eq!(&grid.axes()[0][9..13], &[384.0, 511.0, 512.0, 513.0]);
        let boundary =
            DsaIndexerQRopeQuantWeightFoldSpec::enumerate(&config, &grid, "vllm_cutedsl_fp8");
        assert_eq!(boundary[10].fields()["num_tokens"], 511);
        assert_eq!(boundary[11].fields()["num_tokens"], 512);
    }
}
