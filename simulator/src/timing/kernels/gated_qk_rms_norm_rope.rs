//! Per-head QK RMSNorm + partial NeoX RoPE of a gated attention whose output
//! gate stays in the `[q | gate]` projection rows: TensorFold's Triton
//! `_attn_prep`, one program per (token, query-or-key head). Unlike
//! `qk_rms_norm_rope_gate` it neither reads nor copies the gate; the later
//! `attn_output_gate` reads it in place.
//!
//! Backend: `tensorfold_triton` (bf16 only). Heads, head width, rotated width
//! and `mrope` (two extra position loads a program) are static identity; the
//! token count is the one runtime axis, cached 1-D linear: time is a
//! row-count staircase fine enough that linear interpolation holds.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GatedQkRmsNormRopeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_q_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub rotary_dim: Dim,
    pub mrope: bool,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct GatedQkRmsNormRopeKernelInput {
    pub num_tokens: u32,
}

pub struct GatedQkRmsNormRopeSpec;

impl KernelSpec for GatedQkRmsNormRopeSpec {
    type Config = GatedQkRmsNormRopeKernelConfig;
    type Input = GatedQkRmsNormRopeKernelInput;

    const KIND: KernelKind = "gated_qk_rms_norm_rope";

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
                .with("num_q_heads", config.num_q_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("rotary_dim", config.rotary_dim.get())
                .with("mrope", config.mrope)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(GatedQkRmsNormRopeKernel, GatedQkRmsNormRopeSpec);

#[cfg(test)]
mod tests {
    use super::{GatedQkRmsNormRopeKernelConfig, GatedQkRmsNormRopeSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;
    use serde_json::Value;

    #[test]
    fn payloads_carry_exactly_the_python_fields() {
        // Catches a field the Python GatedQkRmsNormRopeArgs does not have (or
        // lacks one it has), which would key rows the profiler never writes.
        let cfg = GatedQkRmsNormRopeKernelConfig {
            backends: vec!["tensorfold_triton"],
            gpu_name: "NVIDIA GB10".to_string(),
            num_q_heads: 24.into(),
            num_kv_heads: 4.into(),
            head_dim: 256.into(),
            rotary_dim: 64.into(),
            mrope: false,
            dtype: DType::Bf16,
        };
        let grid = GatedQkRmsNormRopeSpec::sweep_grid(&cfg);
        let payloads = GatedQkRmsNormRopeSpec::enumerate(&cfg, &grid, "tensorfold_triton");
        let first = payloads[0].fields();
        let names: Vec<&str> = first.keys().map(String::as_str).collect();
        assert_eq!(
            names,
            [
                "backend",
                "dtype",
                "head_dim",
                "mrope",
                "num_kv_heads",
                "num_q_heads",
                "num_tokens",
                "rotary_dim"
            ]
        );
        assert_eq!(first["num_tokens"], Value::from(1_u32));
        assert_eq!(first["mrope"], Value::from(false));
        let last = payloads.last().unwrap().fields();
        assert_eq!(last["num_tokens"], Value::from(65536_u32));
    }
}
