//! Qwen Gated `DeltaNet` prefill chunked delta rule as ONE fused launch.
//!
//! Deliberately a separate kind from the six `gdn_chunk_*` kinds. Those model
//! FLA's Triton realization, which splits the chunked gated delta rule into six
//! launches (local cumsum, K.Kt, triangular solve, w/u recompute, inter-chunk
//! state scan, output). This kind models the realization vLLM actually selects
//! on Hopper: `FlashInfer`'s `chunk_gated_delta_rule`, a single CUTLASS TMA
//! warp-specialized kernel (`flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule`)
//! that keeps every intermediate on chip. `_resolve_gdn_prefill_backend` picks
//! it for the default `auto` request on any SM90 part, so an H200 capture never
//! launches the six Triton kernels at all.
//!
//! The cache axes are `(L, N)` = longest sequence and `T/L`, NOT `(T, N)`.
//! The inter-chunk recurrence is sequential *within* a sequence and independent
//! *across* sequences, so one CTA per (sequence, value head) walks its own
//! chunks: `L` is the critical path and `T` is the aggregate work. The profiled
//! rows show it directly on an H200 (16/32 heads, 128/128 dims): 4096x1 /
//! 4096x2 / 4096x4 are flat at 198.5 / 198.7 / 212.6 us while 32*N CTAs still
//! fit the 132 SMs, then 4096x8 and 4096x16 rise to 425.3 / 868.2 us. A ragged
//! batch of one 8192-token sequence plus sixteen 64-token ones measures 1.005x
//! its own session's 8192x1 baseline, which rules out a `ceil(32*N/132)` wave
//! model -- short sequences retire immediately, so the launch is
//! work-conserving. Keying on `(T, N)` instead would collapse the 8163+22 split
//! vLLM actually produces (390.7 us, matching the in-situ capture's 389.9 us)
//! onto a balanced 4092+4093 split of the same token total (198.7 us).
//!
//! `L` on a power-of-two axis and `N` on a small integer axis makes the sweep
//! rectangular; only the `T = N*L` ceiling needs masking.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

/// Ceiling on `T = N*L`. A row allocates 28,800 bytes per token plus 4 MiB per
/// recurrent state, so 65,536 tokens is ~1.9 GiB of operands -- profilable
/// beside a resident serving job, unlike the 262,144 the FLA kinds sweep.
const MAX_TOKENS: u64 = 65_536;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnChunkDeltaRuleKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_key_heads: Dim,
    pub num_heads: Dim,
    pub key_head_dim: Dim,
    pub value_head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

/// The physical caller shape. `max_sequence_length` is the longest prefill
/// sequence in the batch, so `num_tokens / max_sequence_length` is the number
/// of full-length sequences the canonical partition realizes.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct GdnChunkDeltaRuleKernelInput {
    pub num_tokens: u32,
    pub max_sequence_length: u32,
}

impl SweepCoords for GdnChunkDeltaRuleKernelInput {
    fn coords(&self) -> Coords {
        Coords::new([
            f64::from(self.max_sequence_length),
            f64::from(self.num_tokens) / f64::from(self.max_sequence_length),
        ])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["num_tokens", "max_sequence_length"]
    }
}

pub struct GdnChunkDeltaRuleSpec;

impl KernelSpec for GdnChunkDeltaRuleSpec {
    type Config = GdnChunkDeltaRuleKernelConfig;
    type Input = GdnChunkDeltaRuleKernelInput;

    const KIND: KernelKind = "gdn_chunk_delta_rule";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // L from one chunk (64) to the 16,384-token chunked-prefill budget;
        // N up to 16, which is where the SM-saturation knee has been passed.
        SweepGrid::new(vec![Axis::pow2(6, 14), Axis::pow2(0, 4)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_2d(|max_sequence_length, full_sequences| {
            token_total(max_sequence_length, full_sequences) > MAX_TOKENS
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|max_sequence_length, full_sequences| {
            let max_sequence_length = max_sequence_length.round() as u64;
            let num_tokens = token_total(max_sequence_length as f64, full_sequences);
            ArgsPayload::new()
                .with("backend", backend)
                .with(
                    "num_tokens",
                    u32::try_from(num_tokens).expect("token sweep must fit u32"),
                )
                .with(
                    "max_sequence_length",
                    u32::try_from(max_sequence_length).expect("length sweep must fit u32"),
                )
                .with("num_key_heads", config.num_key_heads.get())
                .with("num_heads", config.num_heads.get())
                .with("key_head_dim", config.key_head_dim.get())
                .with("value_head_dim", config.value_head_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

fn token_total(max_sequence_length: f64, full_sequences: f64) -> u64 {
    let max_sequence_length = max_sequence_length.round() as u64;
    let full_sequences = full_sequences.round() as u64;
    max_sequence_length
        .checked_mul(full_sequences)
        .expect("GDN chunk-delta-rule token total must fit u64")
}

register_kernel!(GdnChunkDeltaRuleKernel, GdnChunkDeltaRuleSpec);

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{
        GdnChunkDeltaRuleKernelConfig, GdnChunkDeltaRuleKernelInput, GdnChunkDeltaRuleSpec,
        MAX_TOKENS,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};

    fn config() -> GdnChunkDeltaRuleKernelConfig {
        GdnChunkDeltaRuleKernelConfig {
            backends: vec!["flashinfer"],
            gpu_name: "NVIDIA H200".to_string(),
            num_key_heads: 16.into(),
            num_heads: 32.into(),
            key_head_dim: 128.into(),
            value_head_dim: 128.into(),
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_description_and_serde_round_trip_are_exact() {
        let cfg = config();
        assert_eq!(GdnChunkDeltaRuleSpec::KIND, "gdn_chunk_delta_rule");
        assert_eq!(
            GdnChunkDeltaRuleSpec::profile_kind(),
            "gdn_chunk_delta_rule"
        );
        assert_eq!(cfg.backends(), &["flashinfer"]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
        assert_eq!(
            cfg.describe_config(),
            serde_json::json!({
                "backends": ["flashinfer"],
                "gpu_name": "NVIDIA H200",
                "num_key_heads": {"value": 16, "expression": null, "bindings": {}},
                "num_heads": {"value": 32, "expression": null, "bindings": {}},
                "key_head_dim": {"value": 128, "expression": null, "bindings": {}},
                "value_head_dim": {"value": 128, "expression": null, "bindings": {}},
                "dtype": "bf16",
            })
        );
        let encoded = serde_json::to_value(&cfg).unwrap();
        let decoded: GdnChunkDeltaRuleKernelConfig =
            serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
    }

    #[test]
    fn cache_coords_are_longest_sequence_and_full_sequence_count() {
        let cfg = config();
        let coords = |num_tokens, max_sequence_length| {
            GdnChunkDeltaRuleSpec::cache_coords(
                &cfg,
                &GdnChunkDeltaRuleKernelInput {
                    num_tokens,
                    max_sequence_length,
                },
            )
        };
        assert_eq!(&*coords(4096, 4096), &[4096.0, 1.0]);
        assert_eq!(&*coords(16384, 4096), &[4096.0, 4.0]);
        // The capture's 8163+22 iteration lands just past one full sequence,
        // where a (num_tokens, num_sequences) key would have read a balanced
        // two-sequence row worth roughly half the measured time.
        assert_eq!(&*coords(8185, 8163), &[8163.0, 8185.0 / 8163.0]);
    }

    #[test]
    fn physical_input_serde_names_and_slot_payload_are_exact() {
        let input = GdnChunkDeltaRuleKernelInput {
            num_tokens: 8185,
            max_sequence_length: 8163,
        };
        assert_eq!(
            GdnChunkDeltaRuleKernelInput::coord_field_names(),
            &["num_tokens", "max_sequence_length"]
        );
        let slot: SlotInput = input.clone().into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_tokens": 8185, "max_sequence_length": 8163})
        );
        let decoded: GdnChunkDeltaRuleKernelInput =
            serde_json::from_value(serde_json::to_value(input).unwrap()).unwrap();
        assert_eq!(&*decoded.coords(), &[8163.0, 8185.0 / 8163.0]);
    }

    #[test]
    fn sweep_axes_mask_and_enumerated_rows_respect_the_token_ceiling() {
        let cfg = config();
        let grid = GdnChunkDeltaRuleSpec::sweep_grid(&cfg);
        assert_eq!(
            grid.axes()[0],
            vec![64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0]
        );
        assert_eq!(grid.axes()[1], vec![1.0, 2.0, 4.0, 8.0, 16.0]);
        assert_eq!(grid.axes().iter().map(Vec::len).product::<usize>(), 45);

        let mask = GdnChunkDeltaRuleSpec::infeasible_mask(&cfg, &grid);
        assert_eq!(mask.len(), 45);
        // Only 8192x16, 16384x8 and 16384x16 exceed 65,536 tokens.
        assert_eq!(mask.iter().filter(|&&masked| masked).count(), 3);

        let payloads = GdnChunkDeltaRuleSpec::enumerate(&cfg, &grid, "flashinfer");
        for (payload, &masked) in payloads.iter().zip(&mask) {
            let fields = payload.fields();
            let num_tokens = fields["num_tokens"].as_u64().unwrap();
            let max_sequence_length = fields["max_sequence_length"].as_u64().unwrap();
            assert_eq!(num_tokens % max_sequence_length, 0);
            assert_eq!(masked, num_tokens > MAX_TOKENS);
        }
    }

    #[test]
    fn every_payload_carries_exactly_the_python_args_schema() {
        let cfg = config();
        let grid = GdnChunkDeltaRuleSpec::sweep_grid(&cfg);
        let expected = BTreeSet::from([
            "backend",
            "dtype",
            "key_head_dim",
            "max_sequence_length",
            "num_heads",
            "num_key_heads",
            "num_tokens",
            "value_head_dim",
        ]);
        let mut found_qwen_anchor = false;
        for payload in GdnChunkDeltaRuleSpec::enumerate(&cfg, &grid, "flashinfer") {
            assert_eq!(
                payload
                    .fields()
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>(),
                expected
            );
            let fields = payload.fields();
            assert_eq!(fields["backend"], "flashinfer");
            assert_eq!(fields["num_key_heads"], 16);
            assert_eq!(fields["num_heads"], 32);
            assert_eq!(fields["key_head_dim"], 128);
            assert_eq!(fields["value_head_dim"], 128);
            assert_eq!(fields["dtype"], "bf16");
            found_qwen_anchor |=
                fields["num_tokens"] == 8192 && fields["max_sequence_length"] == 8192;
        }
        assert!(found_qwen_anchor);
    }

    #[test]
    fn the_single_backend_uses_product_extrapolated_cache2d() {
        assert_eq!(
            GdnChunkDeltaRuleSpec::cache_kind("flashinfer"),
            CacheKind::Cache2DLinear(Extrapolation::Product)
        );
    }
}
