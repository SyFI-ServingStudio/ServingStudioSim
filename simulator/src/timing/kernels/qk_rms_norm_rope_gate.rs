//! Fused per-head QK RMSNorm + partial NeoX RoPE + output-gate split of a gated
//! attention: vLLM's `fused_qk_rmsnorm_rope_gate` Triton kernel, one program
//! per (token, query-or-key head).
//!
//! `mrope` is a field because multimodal positions add two position loads per
//! program: about 5% at prefill sizes on B200. The cos/sin table length and the
//! position values are not: they moved the time by under 1%.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QkRmsNormRopeGateKernelConfig {
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
pub struct QkRmsNormRopeGateKernelInput {
    pub num_tokens: u32,
}

pub struct QkRmsNormRopeGateSpec;

impl KernelSpec for QkRmsNormRopeGateSpec {
    type Config = QkRmsNormRopeGateKernelConfig;
    type Input = QkRmsNormRopeGateKernelInput;

    const KIND: KernelKind = "qk_rms_norm_rope_gate";

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

register_kernel!(QkRmsNormRopeGateKernel, QkRmsNormRopeGateSpec);
