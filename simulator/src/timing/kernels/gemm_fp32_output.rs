//! GEMM (BF16 or FP32 inputs) whose production callable materializes an FP32 output.
//!
//! Backends: `torch_cublas` is `torch.mm(..., out_dtype=fp32)` alone;
//! `sglang_router_auto` and `vllm_inkling_router_auto` time a framework's whole
//! router dispatch, which switches kernels with `m` inside one call.
//!
//! Coverage gap: the DeepSeek-V4.1 router GateLinear (n=384, k=5120) runs a
//! different kernel for m <= 16 in production (the cute-DSL `ll_bf16_gemm`
//! tier of vLLM's `GateLinear`); only m > 16 falls through to
//! `torch.mm(..., out_dtype=fp32)`. No backend profiles that tier yet, so the
//! m <= 16 rows of that shape are the cuBLAS/nvjet time, not the production one.
//! (Inkling's router reaches the same `ll_bf16_gemm` through its own m <= 64
//! dispatch, which `vllm_inkling_router_auto` measures.)

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GemmFp32OutputKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub n: Dim,
    pub k: Dim,
    #[compute_dtype]
    pub input_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct GemmFp32OutputKernelInput {
    pub m: u32,
}

pub struct GemmFp32OutputSpec;

/// The alignment fork's Inkling router dispatch (`moe.py::_linear_with_fp32_out`).
const VLLM_INKLING_ROUTER_AUTO: &str = "vllm_inkling_router_auto";

/// `m` points added for `vllm_inkling_router_auto`. The call switches kernels
/// at two points: `ll_bf16` dot-product (m <= 4) to split-K (m >= 5), and
/// `ll_bf16` (m <= 64) to cuBLAS (m >= 65). Both sides of each switch are
/// measured so linear interpolation never spans one, and every CUDA-graph
/// capture size up to 128 (multiples of 8), where decode batches are padded to,
/// is a measured point.
fn inkling_router_m_points() -> Vec<f64> {
    Axis::chain([Axis::values([3, 5, 65]), Axis::arithmetic(8, 128, 8)])
}

impl KernelSpec for GemmFp32OutputSpec {
    type Config = GemmFp32OutputKernelConfig;
    type Input = GemmFp32OutputKernelInput;

    const KIND: KernelKind = "gemm_fp32_output";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let mut m = Axis::chain([Axis::pow2(0, 4), Axis::token_axis()]);
        // Only the Inkling dispatch has the in-call switches; keeping its extra
        // points off the other backends avoids resampling their existing rows.
        if config.backends.contains(&VLLM_INKLING_ROUTER_AUTO) {
            m = Axis::chain([m, inkling_router_m_points()]);
            m.sort_by(f64::total_cmp);
        }
        SweepGrid::new(vec![m])
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
                .with("m", m as u32)
                .with("n", config.n.get())
                .with("k", config.k.get())
                .with("input_dtype", config.input_dtype.as_str())
        })
    }
}

register_kernel!(GemmFp32OutputKernel, GemmFp32OutputSpec);

#[cfg(test)]
mod tests {
    use super::{GemmFp32OutputKernelConfig, GemmFp32OutputSpec, VLLM_INKLING_ROUTER_AUTO};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn output_precision_is_kind_semantics_not_a_payload_knob() {
        let config = GemmFp32OutputKernelConfig {
            backends: vec!["torch_cublas"],
            gpu_name: "NVIDIA H200".to_string(),
            n: 256.into(),
            k: 4096.into(),
            input_dtype: DType::Bf16,
        };
        let payload = GemmFp32OutputSpec::enumerate(
            &config,
            &GemmFp32OutputSpec::sweep_grid(&config),
            "torch_cublas",
        )
        .remove(0);
        assert_eq!(payload.fields()["m"], 1);
        assert_eq!(payload.fields()["n"], 256);
        assert_eq!(payload.fields()["k"], 4096);
        assert!(payload.fields().get("output_dtype").is_none());
    }

    fn m_axis(backends: Vec<&'static str>) -> Vec<f64> {
        let config = GemmFp32OutputKernelConfig {
            backends,
            gpu_name: "NVIDIA B200".to_string(),
            n: 264.into(),
            k: 4096.into(),
            input_dtype: DType::Bf16,
        };
        GemmFp32OutputSpec::sweep_grid(&config).axes()[0].clone()
    }

    #[test]
    fn inkling_router_grid_brackets_both_kernel_switches() {
        let m = m_axis(vec![VLLM_INKLING_ROUTER_AUTO]);
        // Linear interpolation must never span a kernel switch inside the call.
        for switch in [4.0, 5.0, 64.0, 65.0] {
            assert!(m.contains(&switch), "missing m={switch}");
        }
        assert!(
            m.windows(2).all(|w| w[0] < w[1]),
            "axis must be strictly increasing"
        );
        // The other backends keep the rows they were profiled on.
        let shared = m_axis(vec!["torch_cublas"]);
        assert!(!shared.contains(&65.0));
        assert!(shared.iter().all(|v| m.contains(v)));
        assert!(m.len() <= 500);
    }
}
