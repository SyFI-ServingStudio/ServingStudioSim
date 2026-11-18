//! Hadamard rotation and per-head power-of-two FP8 quantization of the
//! indexer query heads, read in place from a wider GEMM row, plus the head
//! weights scaled by each head's quant scale, in one launch.
//!
//! When the indexer's query projection is a column span of a concatenated
//! GEMM (`[q_b | wq_b]`), this kind replaces `hadamard_fp8_quant` on the
//! contiguous projection output and the separate weight-scale launch.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HadamardFp8QuantWeightScaleKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Indexer query heads per token.
    pub num_heads: Dim,
    /// Elements per head; also the Hadamard transform size.
    pub head_dim: Dim,
    /// Columns of each input row before the query heads (the other
    /// projections of the concatenated GEMM).
    pub lead_cols: Dim,
    #[compute_dtype]
    pub input_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct HadamardFp8QuantWeightScaleKernelInput {
    pub num_tokens: u32,
}

pub struct HadamardFp8QuantWeightScaleSpec;

impl KernelSpec for HadamardFp8QuantWeightScaleSpec {
    type Config = HadamardFp8QuantWeightScaleKernelConfig;
    type Input = HadamardFp8QuantWeightScaleKernelInput;

    const KIND: KernelKind = "hadamard_fp8_quant_weight_scale";

    /// The shared token curve: single decode rows to long prefill chunks.
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
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("lead_cols", config.lead_cols.get())
                .with("input_dtype", config.input_dtype.as_str())
        })
    }
}

register_kernel!(
    HadamardFp8QuantWeightScaleKernel,
    HadamardFp8QuantWeightScaleSpec
);

#[cfg(test)]
mod tests {
    use super::{
        HadamardFp8QuantWeightScaleKernelConfig, HadamardFp8QuantWeightScaleKernelInput,
        HadamardFp8QuantWeightScaleSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{Dim, SweepCoords};

    fn config() -> HadamardFp8QuantWeightScaleKernelConfig {
        HadamardFp8QuantWeightScaleKernelConfig {
            backends: vec!["triton"],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads: Dim::param("index_n_heads", 32),
            head_dim: Dim::param("index_head_dim", 128),
            lead_cols: Dim::param("q_b_cols", 16384),
            input_dtype: DType::Bf16,
        }
    }

    /// Catches a payload that drifts from the Python
    /// `HadamardFp8QuantWeightScaleArgs` wire form.
    #[test]
    fn enumerate_matches_python_wire_schema() {
        let cfg = config();
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        let grid = HadamardFp8QuantWeightScaleSpec::sweep_grid(&cfg);
        let payloads = HadamardFp8QuantWeightScaleSpec::enumerate(&cfg, &grid, "triton");
        assert_eq!(payloads.len(), 68);
        assert_eq!(
            serde_json::to_value(payloads.last().unwrap().fields()).unwrap(),
            serde_json::json!({
                "backend": "triton",
                "num_tokens": 65536,
                "num_heads": 32,
                "head_dim": 128,
                "lead_cols": 16384,
                "input_dtype": "bf16",
            })
        );
    }

    #[test]
    fn grid_covers_decode_and_verify_rows() {
        let grid = HadamardFp8QuantWeightScaleSpec::sweep_grid(&config());
        let axis = &grid.axes()[0];
        assert!(axis.windows(2).all(|w| w[0] < w[1]));
        for t in [1.0, 8.0, 128.0, 256.0, 384.0, 512.0, 640.0] {
            assert!(axis.contains(&t), "{t} not on the grid");
        }
        assert!(matches!(
            HadamardFp8QuantWeightScaleSpec::cache_kind("triton"),
            CacheKind::Cache1DLinear
        ));
        let input = HadamardFp8QuantWeightScaleKernelInput { num_tokens: 128 };
        assert_eq!(&*input.coords(), &[128.0]);
    }
}
