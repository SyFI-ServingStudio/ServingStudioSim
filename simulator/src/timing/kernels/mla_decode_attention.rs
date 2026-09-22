//! Kimi-K3 absorbed MLA decode attention over a paged latent KV cache.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MlaDecodeAttentionKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub kv_lora_rank: Dim,
    pub rope_dim: Dim,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
    pub page_size: Dim,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MlaDecodeAttentionKernelInput {
    pub batch_size: u32,
    pub kv_len: u32,
}

pub struct MlaDecodeAttentionSpec;

impl KernelSpec for MlaDecodeAttentionSpec {
    type Config = MlaDecodeAttentionKernelConfig;
    type Input = MlaDecodeAttentionKernelInput;

    const KIND: KernelKind = "mla_decode_attention";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::pow2(0, 8), Axis::pow2(6, 22)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|batch_size, kv_len| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_heads", config.num_heads.get())
                .with("kv_lora_rank", config.kv_lora_rank.get())
                .with("rope_dim", config.rope_dim.get())
                .with("q_dtype", config.q_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("page_size", config.page_size.get())
                .with("batch_size", batch_size as u32)
                .with("kv_len", kv_len as u32)
        })
    }
}

register_kernel!(MlaDecodeAttentionKernel, MlaDecodeAttentionSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::bridge::DType;
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::KernelSpec;
    use crate::timing::SweepCoords;

    fn config() -> MlaDecodeAttentionKernelConfig {
        MlaDecodeAttentionKernelConfig {
            backends: vec!["sglang_cutedsl_mla", "sglang_trtllm_mla", "sglang_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads: 12.into(),
            kv_lora_rank: 512.into(),
            rope_dim: 64.into(),
            q_dtype: DType::Bf16,
            kv_dtype: DType::Bf16,
            page_size: 64.into(),
        }
    }

    #[test]
    fn config_and_two_dimensional_grid_match_k3() {
        let cfg = config();
        assert_eq!(MlaDecodeAttentionSpec::KIND, "mla_decode_attention");
        assert_eq!(cfg.num_heads, 12);
        assert_eq!(MlaDecodeAttentionSpec::sweep_grid(&cfg).axes().len(), 2);
        assert_eq!(MlaDecodeAttentionSpec::sweep_grid(&cfg).axes()[1][0], 64.0);
        assert_eq!(
            MlaDecodeAttentionSpec::cache_kind("sglang_cutedsl_mla"),
            CacheKind::Cache2DLinear(Extrapolation::Weighted)
        );
    }

    #[test]
    fn payload_and_coords_are_exact() {
        let input = MlaDecodeAttentionKernelInput {
            batch_size: 8,
            kv_len: 8192,
        };
        assert_eq!(&*input.coords(), &[8.0, 8192.0]);
        assert_eq!(
            MlaDecodeAttentionKernelInput::coord_field_names(),
            &["batch_size", "kv_len"]
        );
        let payload = MlaDecodeAttentionSpec::enumerate(
            &config(),
            &SweepGrid::new(vec![Axis::values([1]), Axis::values([64])]),
            "sglang_triton",
        )[0]
        .clone();
        assert_eq!(payload.fields().len(), 9);
        assert_eq!(payload.fields()["page_size"], serde_json::json!(64));
        assert_eq!(payload.fields()["kv_dtype"], serde_json::json!("bf16"));
    }
}
