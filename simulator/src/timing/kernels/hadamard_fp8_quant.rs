//! Hadamard rotation of each row followed by per-row power-of-two FP8 quantization.
//!
//! One launch: `num_rows` BF16 rows of `head_dim` elements are rotated by the
//! normalized Walsh-Hadamard transform and quantized to FP8 E4M3 with one
//! UE8M0-valued FP32 scale per row. A sparse-attention indexer calls it on its
//! query heads, so `num_rows = tokens * heads`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HadamardFp8QuantKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Row width and Hadamard transform size.
    pub head_dim: u32,
    #[compute_dtype]
    pub input_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct HadamardFp8QuantKernelInput {
    /// Rows rotated and quantized together (tokens times heads).
    pub num_rows: u32,
}

pub struct HadamardFp8QuantSpec;

impl KernelSpec for HadamardFp8QuantSpec {
    type Config = HadamardFp8QuantKernelConfig;
    type Input = HadamardFp8QuantKernelInput;

    const KIND: KernelKind = "hadamard_fp8_quant";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // Rows are tokens x heads: one head of one token up to a 32k-token
        // prefill of 32 heads (2^20 rows). The shared token curve gives the
        // dense mid-range where decode and verify batches land.
        SweepGrid::new(vec![Axis::chain([
            Axis::pow2(0, 5),
            Axis::token_axis(),
            Axis::pow2(17, 20),
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
        grid.expand_1d(|num_rows| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_rows", num_rows as u32)
                .with("head_dim", config.head_dim)
                .with("input_dtype", config.input_dtype.as_str())
        })
    }
}

register_kernel!(HadamardFp8QuantKernel, HadamardFp8QuantSpec);

#[cfg(test)]
mod tests {
    use super::{HadamardFp8QuantKernelConfig, HadamardFp8QuantKernelInput, HadamardFp8QuantSpec};
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::KernelSpec;
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> HadamardFp8QuantKernelConfig {
        HadamardFp8QuantKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            head_dim: 128,
            input_dtype: DType::Bf16,
        }
    }

    #[test]
    fn enumerate_matches_python_wire_schema() {
        let config = config();
        let grid = HadamardFp8QuantSpec::sweep_grid(&config);
        let payloads = HadamardFp8QuantSpec::enumerate(&config, &grid, "vllm_triton");
        let fields = payloads[0].fields();
        assert_eq!(fields.len(), 4);
        assert_eq!(fields.get("backend"), Some(&Value::from("vllm_triton")));
        assert_eq!(fields.get("num_rows"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("head_dim"), Some(&Value::from(128_u32)));
        assert_eq!(fields.get("input_dtype"), Some(&Value::from("bf16")));
    }

    #[test]
    fn grid_spans_single_rows_to_long_prefill_and_stays_small() {
        let grid = HadamardFp8QuantSpec::sweep_grid(&config());
        let axis = &grid.axes()[0];
        assert_eq!(axis.first(), Some(&1.0));
        assert_eq!(axis.last(), Some(&1_048_576.0));
        assert!(axis.windows(2).all(|w| w[0] < w[1]));
        assert!(axis.len() <= 500);
        // The two reference decode shapes (16 and 32 requests x 8 tokens x 32 heads).
        assert!(axis.contains(&4096.0) && axis.contains(&8192.0));
        assert!(matches!(
            HadamardFp8QuantSpec::cache_kind("vllm_triton"),
            CacheKind::Cache1DLinear
        ));
    }

    #[test]
    fn input_coordinate_is_the_row_count() {
        let input = HadamardFp8QuantKernelInput { num_rows: 4096 };
        assert_eq!(&*input.coords(), &[4096.0]);
    }
}
