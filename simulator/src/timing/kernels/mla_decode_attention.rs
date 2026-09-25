//! Kimi-K3 absorbed MLA decode attention over a paged latent KV cache.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

const MAX_PROFILE_KV_BYTES: f64 = 48.0 * 1024.0 * 1024.0 * 1024.0;

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
        // Decode groups reach B=512, and the rank-1 oracle also exercises a
        // 40,960-token per-request context. Keep the established power-of-two
        // coverage while making those production points measured cache cells.
        SweepGrid::new(vec![
            Axis::pow2(0, 9),
            Axis::chain([
                Axis::pow2(7, 15),
                Axis::values([40_960]),
                Axis::pow2(16, 20),
            ]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        let latent_dim = f64::from(config.kv_lora_rank.get() + config.rope_dim.get());
        let kv_bytes_per_token = latent_dim * f64::from(config.kv_dtype.size_bytes());
        grid.expand_2d(|batch_size, kv_len| {
            batch_size * kv_len * kv_bytes_per_token > MAX_PROFILE_KV_BYTES
        })
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
        let grid = MlaDecodeAttentionSpec::sweep_grid(&cfg);
        assert_eq!(grid.axes().len(), 2);
        assert_eq!(grid.axes()[0], Axis::pow2(0, 9));
        assert_eq!(
            grid.axes()[1],
            Axis::chain([
                Axis::pow2(7, 15),
                Axis::values([40_960]),
                Axis::pow2(16, 20),
            ])
        );
        assert_eq!(
            MlaDecodeAttentionSpec::cache_kind("sglang_cutedsl_mla"),
            CacheKind::Cache2DLinear(Extrapolation::Weighted)
        );
    }

    #[test]
    fn memory_mask_preserves_the_k3_points_for_bf16_and_fp8() {
        let cfg = config();
        let grid = MlaDecodeAttentionSpec::sweep_grid(&cfg);
        let columns = grid.axes()[1].len();
        let cell = |mask: &[bool], batch_size: f64, kv_len: f64| {
            let row = grid.axes()[0]
                .iter()
                .position(|&value| value == batch_size)
                .unwrap();
            let column = grid.axes()[1]
                .iter()
                .position(|&value| value == kv_len)
                .unwrap();
            mask[row * columns + column]
        };

        let bf16_mask = MlaDecodeAttentionSpec::infeasible_mask(&cfg, &grid);
        assert_eq!(bf16_mask.len(), 10 * 15);
        assert!(!cell(&bf16_mask, 1.0, 1_048_576.0));
        assert!(!cell(&bf16_mask, 16.0, 65_536.0));
        assert!(!cell(&bf16_mask, 128.0, 8_192.0));
        assert!(!cell(&bf16_mask, 512.0, 65_536.0));
        assert!(cell(&bf16_mask, 512.0, 131_072.0));

        let mut fp8_cfg = cfg;
        fp8_cfg.kv_dtype = DType::Fp8E4m3;
        let fp8_mask = MlaDecodeAttentionSpec::infeasible_mask(&fp8_cfg, &grid);
        assert!(!cell(&fp8_mask, 64.0, 1_048_576.0));
        assert!(cell(&fp8_mask, 128.0, 1_048_576.0));
    }

    #[test]
    fn large_decode_points_are_grid_covered_instead_of_extrapolated() {
        use crate::timing::bridge::KernelMetrics;
        use crate::timing::cache::interp::CoverageFlags;
        use crate::timing::cache::{Cache, Cache2DLinear};

        let cfg = config();
        let grid = MlaDecodeAttentionSpec::sweep_grid(&cfg);
        let samples = grid.expand_2d(|batch_size, kv_len| KernelMetrics {
            time_ms: batch_size + kv_len,
            tflops: Some(1.0),
            memory_bandwidth_gbps: Some(1.0),
            algbw_gbps: None,
            busbw_gbps: None,
            energy_j: 1.0,
        });
        let (cache, _warnings) =
            Cache2DLinear::from_samples_with(&grid, &samples, Extrapolation::Weighted);

        for [batch_size, kv_len] in [
            [1.0, 8_192.0],
            [32.0, 8_192.0],
            [128.0, 8_192.0],
            [256.0, 8_192.0],
            [512.0, 8_192.0],
            [16.0, 40_960.0],
        ] {
            let result = cache.eval(&[batch_size, kv_len]);
            assert_eq!(
                result.coverage,
                CoverageFlags::EMPTY,
                "MLA decode point ({batch_size}, {kv_len}) must be measured-grid covered"
            );
        }
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
            &SweepGrid::new(vec![Axis::values([1]), Axis::values([128])]),
            "sglang_triton",
        )[0]
        .clone();
        assert_eq!(payload.fields().len(), 9);
        assert_eq!(payload.fields()["page_size"], serde_json::json!(64));
        assert_eq!(payload.fields()["kv_dtype"], serde_json::json!("bf16"));
    }
}
