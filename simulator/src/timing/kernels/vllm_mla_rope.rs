//! vLLM MLA query `RoPE`, priced as the inductor fusion actually runs.
//!
//! This leaf is NOT rope-slice sized. vLLM writes `q[..., qk_nope:] = q_pe` and
//! then hands `q` to the attention op, so the functionalized inductor graph
//! materialises a whole new `q`: the generated triton kernel walks the full
//! `num_tokens * num_heads * (qk_nope + rope)` space, loads and stores every
//! column, and selects `tl.where(col >= qk_nope, roped, original)`. It reads and
//! writes 100% of q to change `rope / (qk_nope + rope)` of it.
//!
//! Consequently it sustains ~0.9 TB/s where a streaming pointwise kernel reaches
//! ~4.3 TB/s on the same H200. Pricing the slot as an `elementwise` leaf over
//! the rope bytes made it 74% too fast on GLM-5.2 prefill; this kernel lands
//! within 5.3% of the measured 600.21 us at 8192 tokens.
//!
//! `max_position` is a config axis rather than a value-only detail because the
//! cos/sin table is gathered per token with an indirect index, so its row count
//! decides whether that gather hits cache. `rope_theta` is deliberately absent:
//! it changes the table's values, never its shape or access pattern.
//!
//! Cost is linear in `num_tokens` at a fixed row width, so a 1-D linear cache on
//! the token axis is the right shape. The curve saturates at ~945 GB/s from
//! ~1k tokens up and stays there to 65k, so the prefill end needs no
//! extrapolation. Folded against the GLM-5.2 capture it lands at -5.4% on the
//! 8192-token prefill steps and -9.7% at decode.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VllmMlaRopeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub qk_nope_head_dim: Dim,
    pub rope_dim: Dim,
    pub max_position: Dim,
    pub is_neox_style: bool,
    #[compute_dtype]
    pub input_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct VllmMlaRopeKernelInput {
    pub num_tokens: u32,
}

pub struct VllmMlaRopeSpec;

impl KernelSpec for VllmMlaRopeSpec {
    type Config = VllmMlaRopeKernelConfig;
    type Input = VllmMlaRopeKernelInput;

    const KIND: KernelKind = "vllm_mla_rope";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // Explicit low counts before the shared token curve: decode steps land
        // at a few dozen rows, far below the token axis's first point, and the
        // kernel is launch-bound there rather than bandwidth-bound.
        SweepGrid::new(vec![Axis::chain([
            Axis::values([1, 4, 8, 16, 32, 48]),
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
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "num_tokens is a grid point from Axis::pow2/token_axis, always a small non-negative integer"
            )]
            let num_tokens_u32 = num_tokens as u32;
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens_u32)
                .with("num_heads", config.num_heads.get())
                .with("qk_nope_head_dim", config.qk_nope_head_dim.get())
                .with("rope_dim", config.rope_dim.get())
                .with("max_position", config.max_position.get())
                .with("is_neox_style", config.is_neox_style)
                .with("input_dtype", config.input_dtype.as_str())
        })
    }
}

register_kernel!(VllmMlaRopeKernel, VllmMlaRopeSpec);

#[cfg(test)]
mod tests {
    use super::{VllmMlaRopeKernelConfig, VllmMlaRopeKernelInput, VllmMlaRopeSpec};
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::KernelSpec;
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> VllmMlaRopeKernelConfig {
        VllmMlaRopeKernelConfig {
            backends: vec!["vllm_inductor"],
            gpu_name: "NVIDIA H200".to_string(),
            num_heads: 64.into(),
            qk_nope_head_dim: 192.into(),
            rope_dim: 64.into(),
            max_position: 1_048_576.into(),
            is_neox_style: false,
            input_dtype: DType::Bf16,
        }
    }

    #[test]
    fn enumerate_emits_every_python_arg_for_each_swept_token_count() {
        let grid = VllmMlaRopeSpec::sweep_grid(&config());
        let payloads = VllmMlaRopeSpec::enumerate(&config(), &grid, "vllm_inductor");
        assert_eq!(payloads.len(), grid.axes()[0].len());

        let first = payloads[0].fields();
        for key in [
            "backend",
            "num_tokens",
            "num_heads",
            "qk_nope_head_dim",
            "rope_dim",
            "max_position",
            "is_neox_style",
            "input_dtype",
        ] {
            assert!(first.contains_key(key), "payload is missing {key}");
        }
        assert_eq!(first["num_heads"], Value::from(64u32));
        assert_eq!(first["is_neox_style"], Value::from(false));
    }

    #[test]
    fn the_token_axis_reaches_a_full_prefill_chunk_and_starts_below_a_decode_batch() {
        let grid = VllmMlaRopeSpec::sweep_grid(&config());
        let axis = &grid.axes()[0];
        assert_eq!(axis[0], 1.0, "the curve must start at a single token");
        assert!(
            axis.iter().any(|&value| value <= 48.0 && value >= 32.0),
            "decode batches land in the tens of rows and need a sample there"
        );
        assert!(
            axis.iter().any(|&value| value >= 8192.0),
            "a max_batch_tokens=8192 prefill chunk must be inside the curve, not extrapolated"
        );
    }

    #[test]
    fn the_cache_is_one_dimensional_on_the_token_axis() {
        assert!(matches!(
            VllmMlaRopeSpec::cache_kind("vllm_inductor"),
            CacheKind::Cache1DLinear
        ));
        let input = VllmMlaRopeKernelInput { num_tokens: 8192 };
        assert_eq!(&*input.coords(), &[8192.0]);
    }

    #[test]
    fn rope_dim_and_qk_nope_are_separate_axes_because_only_one_is_roped() {
        // The kernel walks qk_nope + rope columns but applies the rotation to
        // rope columns alone, so collapsing them into a single head_dim would
        // lose the ratio that sets how much of the traffic is pure copy.
        let mut wider_nope = config();
        wider_nope.qk_nope_head_dim = 256.into();
        assert_ne!(config(), wider_nope);

        let mut wider_rope = config();
        wider_rope.rope_dim = 128.into();
        assert_ne!(config(), wider_rope);
    }
}
