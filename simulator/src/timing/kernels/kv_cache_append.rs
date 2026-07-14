//! Paged KV-cache append: write newly produced K/V rows into cache slots.
//!
//! Static identity captures the physical write path (KV heads/head size, page
//! size, input/cache dtypes, NHD/HND layout, and tensor/head scale). Runtime is
//! the number of actual appended tokens, matching vLLM's use of
//! `slot_mapping.size(0)` rather than any CUDA-graph padded tensor length.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Deserialize)]
pub struct KvCacheAppendKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub block_size: u32,
    #[compute_dtype]
    pub input_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
    pub cache_layout: String,
    pub scale_granularity: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct KvCacheAppendKernelInput {
    pub num_tokens: u32,
}

pub struct KvCacheAppendSpec;

impl KernelSpec for KvCacheAppendSpec {
    type Config = KvCacheAppendKernelConfig;
    type Input = KvCacheAppendKernelInput;

    const KIND: KernelKind = "kv_cache_append";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // Decode needs coverage below the shared token_axis's first point (32).
        // H200 sustains two token-blocks per 132 SMs: 264 is the last one-wave
        // point and 265 is the first two-wave point. Keeping the slow-side point
        // prevents interpolation to 384 from smoothing over the measured cliff.
        SweepGrid::new(vec![Axis::chain([
            Axis::pow2(0, 8),
            Axis::values([264, 265]),
            Axis::token_axis(),
        ])])
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
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("block_size", config.block_size)
                .with("input_dtype", config.input_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("cache_layout", config.cache_layout.clone())
                .with("scale_granularity", config.scale_granularity.clone())
                .with("num_tokens", num_tokens as u32)
        })
    }
}

register_kernel!(KvCacheAppendKernel, KvCacheAppendSpec);

#[cfg(test)]
mod tests {
    use super::{KvCacheAppendKernelConfig, KvCacheAppendKernelInput, KvCacheAppendSpec};
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> KvCacheAppendKernelConfig {
        KvCacheAppendKernelConfig {
            backends: vec!["vllm_cuda"],
            gpu_name: "NVIDIA H200".to_string(),
            num_kv_heads: 8.into(),
            head_dim: 128.into(),
            block_size: 16,
            input_dtype: DType::Bf16,
            kv_dtype: DType::Bf16,
            cache_layout: "NHD".to_string(),
            scale_granularity: "tensor".to_string(),
        }
    }

    #[test]
    fn config_identity_describes_the_physical_write_path() {
        assert_eq!(
            config().describe_config(),
            r#"backends=["vllm_cuda"] gpu_name="NVIDIA H200" num_kv_heads=8 head_dim=128 block_size=16 input_dtype=Bf16 kv_dtype=Bf16 cache_layout="NHD" scale_granularity="tensor""#
        );
    }

    #[test]
    fn input_is_the_actual_append_count() {
        let input = KvCacheAppendKernelInput { num_tokens: 64 };
        assert_eq!(&*input.coords(), &[64.0]);
        assert_eq!(
            KvCacheAppendKernelInput::coord_field_names(),
            &["num_tokens"]
        );
    }

    #[test]
    fn grid_covers_decode_and_prefill_sizes() {
        let grid = KvCacheAppendSpec::sweep_grid(&config());
        assert_eq!(grid.axes().len(), 1);
        assert_eq!(&grid.axes()[0][..5], &[1.0, 2.0, 4.0, 8.0, 16.0]);
        assert!(grid.axes()[0].contains(&32.0));
        assert!(grid.axes()[0].windows(2).any(|pair| pair == [264.0, 265.0]));
        assert!(grid.axes()[0].contains(&2048.0));
        assert_eq!(
            KvCacheAppendSpec::cache_kind("vllm_cuda"),
            CacheKind::Cache1DLinear
        );
    }

    #[test]
    fn enumerate_matches_python_args_exactly() {
        let cfg = config();
        let grid = KvCacheAppendSpec::sweep_grid(&cfg);
        let payload = &KvCacheAppendSpec::enumerate(&cfg, &grid, "vllm_cuda")[0];
        let fields = payload.fields();
        assert_eq!(fields.len(), 9);
        assert_eq!(fields.get("backend"), Some(&Value::from("vllm_cuda")));
        assert_eq!(fields.get("num_kv_heads"), Some(&Value::from(8_u32)));
        assert_eq!(fields.get("head_dim"), Some(&Value::from(128_u32)));
        assert_eq!(fields.get("block_size"), Some(&Value::from(16_u32)));
        assert_eq!(fields.get("input_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("kv_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("cache_layout"), Some(&Value::from("NHD")));
        assert_eq!(
            fields.get("scale_granularity"),
            Some(&Value::from("tensor"))
        );
        assert_eq!(fields.get("num_tokens"), Some(&Value::from(1_u32)));
    }
}
