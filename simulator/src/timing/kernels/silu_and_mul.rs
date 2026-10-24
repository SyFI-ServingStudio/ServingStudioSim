//! SiLU-and-mul activation of a dense gated MLP, as Inductor compiles vLLM's
//! native `SiluAndMul` into one Triton pointwise kernel.
//!
//! Backends: `vllm_inductor`, `vllm_cuda`, and `tensorfold_triton`
//! (TensorFold's Triton `_swiglu` on the stacked `[gate | up]` projection
//! output, bf16 only). All share the grid and the linear 1-D cache.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SiluAndMulKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Width of the gate half and of the up half (the output width).
    pub hidden_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct SiluAndMulKernelInput {
    pub num_tokens: u32,
}

pub struct SiluAndMulSpec;

/// TensorFold's stacked `_swiglu` and the largest stacked input it can index.
const TENSORFOLD_TRITON: &str = "tensorfold_triton";
const TENSORFOLD_MAX_STACKED_ELEMENTS: f64 = 2_147_483_647.0;

impl KernelSpec for SiluAndMulSpec {
    type Config = SiluAndMulKernelConfig;
    type Input = SiluAndMulKernelInput;

    const KIND: KernelKind = "silu_and_mul";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::chain([Axis::pow2(0, 4), Axis::token_axis()])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    /// TensorFold's `_swiglu` offsets each row by `row * 2 * hidden_dim` in
    /// int32, so a stacked input of 2^31 elements or more wraps and faults
    /// (65536 rows at hidden 17408 on GB10); its verify windows stay at or
    /// under 256 rows. A config that names `tensorfold_triton` drops those grid
    /// rows; queries past the last row extrapolate the linear cache.
    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        if !config.backends.contains(&TENSORFOLD_TRITON) {
            return Vec::new();
        }
        let row = 2.0 * f64::from(config.hidden_dim.get());
        grid.axes()[0]
            .iter()
            .map(|&num_tokens| num_tokens * row > TENSORFOLD_MAX_STACKED_ELEMENTS)
            .collect()
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
                .with("hidden_dim", config.hidden_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(SiluAndMulKernel, SiluAndMulSpec);

#[cfg(test)]
mod tests {
    use super::{SiluAndMulKernelConfig, SiluAndMulSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;
    use serde_json::Value;

    #[test]
    fn tensorfold_triton_payloads_carry_the_backend_and_python_fields() {
        // Catches the TensorFold rows being requested under another backend or
        // with fields SiluAndMulArgs does not have.
        let cfg = SiluAndMulKernelConfig {
            backends: vec!["tensorfold_triton"],
            gpu_name: "NVIDIA GB10".to_string(),
            hidden_dim: 17408.into(),
            dtype: DType::Bf16,
        };
        let grid = SiluAndMulSpec::sweep_grid(&cfg);
        let payloads = SiluAndMulSpec::enumerate(&cfg, &grid, "tensorfold_triton");
        assert_eq!(payloads[0].fields()["num_tokens"], Value::from(1_u32));
        for payload in &payloads {
            let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
            assert_eq!(names, ["backend", "dtype", "hidden_dim", "num_tokens"]);
            assert_eq!(payload.backend(), Some("tensorfold_triton"));
        }
        // Only the 65536-row point wraps the int32 row offset at hidden 17408; a
        // mask that kept it would fault the whole profiling batch.
        let mask = SiluAndMulSpec::infeasible_mask(&cfg, &grid);
        let dropped: Vec<f64> = grid.axes()[0]
            .iter()
            .zip(&mask)
            .filter_map(|(&m, &drop)| drop.then_some(m))
            .collect();
        assert_eq!(dropped, vec![65536.0]);
        let vllm = SiluAndMulKernelConfig {
            backends: vec!["vllm_cuda"],
            ..cfg
        };
        assert!(SiluAndMulSpec::infeasible_mask(&vllm, &grid).is_empty());
    }
}
