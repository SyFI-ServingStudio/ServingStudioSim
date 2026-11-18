//! Batched GEMM whose bf16 result is quantized to FP8 E4M3 under one static
//! scale, in one launch.
//!
//! MLA query absorption (`[m, b, k] x W_UK[b, k, n]`) feeding an FP8
//! sparse-attention kernel: the unfused path is a bf16 `batched_gemm` plus a
//! static FP8 quant launch; this kind is the pair as one kernel that quantizes
//! in its epilogue (bit-identical output). Not a `batched_gemm` backend: that
//! kind's output is the bf16 product.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BatchedGemmStaticFp8QuantKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Attention heads on this GPU.
    pub num_batches: Dim,
    /// Output features per head.
    pub n: Dim,
    /// Input features per head.
    pub k: Dim,
    /// Element type of both inputs; the output is always FP8 E4M3.
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct BatchedGemmStaticFp8QuantKernelInput {
    /// Token rows per head.
    pub m: u32,
}

pub struct BatchedGemmStaticFp8QuantSpec;

impl KernelSpec for BatchedGemmStaticFp8QuantSpec {
    type Config = BatchedGemmStaticFp8QuantKernelConfig;
    type Input = BatchedGemmStaticFp8QuantKernelInput;

    const KIND: KernelKind = "batched_gemm_static_fp8_quant";

    /// `batched_gemm`'s token curve: the two kinds price the same rows.
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
        grid.expand_1d(|m| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_batches", config.num_batches.get())
                .with("m", m as u32)
                .with("n", config.n.get())
                .with("k", config.k.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(
    BatchedGemmStaticFp8QuantKernel,
    BatchedGemmStaticFp8QuantSpec
);

#[cfg(test)]
mod tests {
    use super::{
        BatchedGemmStaticFp8QuantKernelConfig, BatchedGemmStaticFp8QuantKernelInput,
        BatchedGemmStaticFp8QuantSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{Dim, SweepCoords};

    fn config() -> BatchedGemmStaticFp8QuantKernelConfig {
        BatchedGemmStaticFp8QuantKernelConfig {
            backends: vec!["cuda_tcgen05"],
            gpu_name: "NVIDIA B200".to_string(),
            num_batches: Dim::param("num_heads", 64),
            n: Dim::param("kv_lora_rank", 512),
            k: Dim::param("qk_nope_head_dim", 256),
            dtype: DType::Bf16,
        }
    }

    /// Catches a payload that drifts from the Python
    /// `BatchedGemmStaticFp8QuantArgs` wire form (it would read as missing rows).
    #[test]
    fn enumerate_matches_python_wire_schema() {
        let cfg = config();
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        let grid = BatchedGemmStaticFp8QuantSpec::sweep_grid(&cfg);
        let payloads = BatchedGemmStaticFp8QuantSpec::enumerate(&cfg, &grid, "cuda_tcgen05");
        assert_eq!(payloads.len(), 68);
        assert_eq!(
            serde_json::to_value(payloads[0].fields()).unwrap(),
            serde_json::json!({
                "backend": "cuda_tcgen05",
                "num_batches": 64,
                "m": 1,
                "n": 512,
                "k": 256,
                "dtype": "bf16",
            })
        );
    }

    #[test]
    fn grid_covers_decode_and_verify_rows() {
        let grid = BatchedGemmStaticFp8QuantSpec::sweep_grid(&config());
        let axis = &grid.axes()[0];
        assert_eq!(axis.first(), Some(&1.0));
        assert!(axis.windows(2).all(|w| w[0] < w[1]));
        // Decode and verify batches at 8 query rows per request.
        for m in [8.0, 128.0, 256.0, 384.0, 512.0, 640.0] {
            assert!(axis.contains(&m), "{m} not on the grid");
        }
        assert!(matches!(
            BatchedGemmStaticFp8QuantSpec::cache_kind("cuda_tcgen05"),
            CacheKind::Cache1DLinear
        ));
        let input = BatchedGemmStaticFp8QuantKernelInput { m: 256 };
        assert_eq!(&*input.coords(), &[256.0]);
    }
}
