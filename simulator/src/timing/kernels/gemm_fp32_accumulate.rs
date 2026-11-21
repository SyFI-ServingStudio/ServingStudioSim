//! GEMM that accumulates a bf16 product into an existing fp32 matrix:
//! `C[m, n] += A[m, k] · B[n, k]ᵀ` (cuBLASLt with beta = 1).
//!
//! DeepInfer's DFlash head adds its Markov correction to each draft step's
//! logits this way (`markov_w2(w1, out_acc=logits)` → DeepGEMM `bf16_gemm_nt`
//! with `c = d`). Unlike `gemm_fp32_output` the output is also read, so the
//! two kinds do not share rows.
//!
//! `n` and `k` are weight geometry (Config); `m` (rows: requests in one
//! Markov step) is the only runtime axis.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GemmFp32AccumulateKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub n: Dim,
    pub k: Dim,
    #[compute_dtype]
    pub input_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct GemmFp32AccumulateKernelInput {
    pub m: u32,
}

pub struct GemmFp32AccumulateSpec;

impl KernelSpec for GemmFp32AccumulateSpec {
    type Config = GemmFp32AccumulateKernelConfig;
    type Input = GemmFp32AccumulateKernelInput;

    const KIND: KernelKind = "gemm_fp32_accumulate";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // The GEMM token axis through 8192 rows (requests per step).
        let m: Vec<f64> = Axis::chain([Axis::pow2(0, 4), Axis::token_axis()])
            .into_iter()
            .filter(|&m| m <= 8192.0)
            .collect();
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

register_kernel!(GemmFp32AccumulateKernel, GemmFp32AccumulateSpec);

#[cfg(test)]
mod tests {
    use super::{
        GemmFp32AccumulateKernelConfig, GemmFp32AccumulateKernelInput, GemmFp32AccumulateSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;

    fn config() -> GemmFp32AccumulateKernelConfig {
        GemmFp32AccumulateKernelConfig {
            backends: vec!["deepgemm_cublaslt"],
            gpu_name: "NVIDIA B200".to_string(),
            n: 129280.into(),
            k: 256.into(),
            input_dtype: DType::Bf16,
        }
    }

    /// The payload must carry exactly the Python `GemmFp32AccumulateArgs`
    /// fields; the m axis spans 1..=8192.
    #[test]
    fn payload_carries_exactly_the_python_args_over_m() {
        let cfg = config();
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        let grid = GemmFp32AccumulateSpec::sweep_grid(&cfg);
        let payloads = GemmFp32AccumulateSpec::enumerate(&cfg, &grid, "deepgemm_cublaslt");
        for p in &payloads {
            let names: Vec<&str> = p.fields().keys().map(String::as_str).collect();
            assert_eq!(names, ["backend", "input_dtype", "k", "m", "n"]);
        }
        let m: Vec<u64> = payloads
            .iter()
            .map(|p| p.fields()["m"].as_u64().unwrap())
            .collect();
        assert_eq!(m.first(), Some(&1));
        assert_eq!(m.iter().max(), Some(&8192));
        let input = GemmFp32AccumulateKernelInput { m: 64 };
        assert_eq!(&*input.coords(), &[64.0]);
    }
}
