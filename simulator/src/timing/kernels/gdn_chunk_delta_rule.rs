//! Gated DeltaNet prefill chunked delta rule: one call writing every token's
//! output and each sequence's final state.
//!
//! Two backends share the kind, its args and its `(L, N)` cache; they differ in
//! implementation and in where `exp` is applied to the decay gate:
//!
//! - `flashinfer` -- FlashInfer's `chunk_gated_delta_rule`, a single CUTLASS
//!   TMA warp-specialized kernel (`flat::kernel::FlatKernelTmaWarpSpecializedDeltaRule`)
//!   on SM90 and a persistent CuTe DSL kernel on SM100, every intermediate on
//!   chip. It takes `exp(g)`, which vLLM computes in a separate elementwise
//!   launch the arch costs as its own leaf. `_resolve_gdn_prefill_backend`
//!   picks it for the default `auto` request on any SM90 part (and SM10x/SM12x
//!   with head dim 128 on CUDA 13+), so an H200 capture never launches the
//!   Triton chain below.
//! - `fla_triton` -- vLLM's vendored FLA `chunk_gated_delta_rule`, every other
//!   CUDA device (A100, L40S, ...): seven launches (local cumsum, K.Kt, a BF16
//!   zero fill and the triangular solve, w/u recompute, inter-chunk state scan,
//!   output). It takes the log-space `g` directly, so an arch selecting it has
//!   no `exp(g)` leaf.
//!
//! The cache axes are `(L, N)` = longest sequence and `T/L`, NOT `(T, N)`.
//! The inter-chunk recurrence is sequential *within* a sequence and independent
//! *across* sequences, so one CTA per (sequence, value head) walks its own
//! chunks (FlashInfer's kernel; FLA's `chunk_gated_delta_rule_fwd_kernel_h`,
//! the largest of its launches, whose other launches are chunk-parallel over
//! tokens): `L` is the critical path and `T` is the aggregate work. The profiled
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
//!
//! `fla_triton` gets a wider grid. Its time at large `T` is nearly linear in
//! `T` whatever `L` is (L40S, Qwen3.8-27B 16/48 heads: 16,384 tokens take
//! 6.59-6.76 ms from 1x16384 to 16x1024), but every sequence costs at least a
//! whole 64-token chunk, so many short prefills cost more per token than one
//! long one (16x64 = 1,024 tokens: 346 us against 180 us for 1x1024). On the
//! FlashInfer grid that region is reached only by extrapolation, which on the
//! L40S under-predicted 128x64 by 18% and 75x8 by 93%. The FLA grid therefore
//! adds `L` = 8/16/32 below one chunk, `L` = 1536/3072 where single-sequence
//! time bends (1024: 180 us, 2048: 522 us), and `N` up to 256. A config whose
//! backends include `fla_triton` uses it; FlashInfer-only configs keep the grid
//! their H200/B200 rows were measured on.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

/// Ceiling on `T = N*L`. A row allocates 28,800 bytes per token plus 4 MiB per
/// recurrent state, so 65,536 tokens is ~1.9 GiB of operands -- profilable
/// beside a resident serving job.
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
            self.max_sequence_length as f64,
            self.num_tokens as f64 / self.max_sequence_length as f64,
        ])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["num_tokens", "max_sequence_length"]
    }
}

pub struct GdnChunkDeltaRuleSpec;

const FLA_TRITON: &str = "fla_triton";

impl KernelSpec for GdnChunkDeltaRuleSpec {
    type Config = GdnChunkDeltaRuleKernelConfig;
    type Input = GdnChunkDeltaRuleKernelInput;

    const KIND: KernelKind = "gdn_chunk_delta_rule";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        if config.backends.contains(&FLA_TRITON) {
            // Sub-chunk L, the 1-2K and 2-4K single-sequence bend, and N up to
            // 256 for many short prefills (see the module docs).
            return SweepGrid::new(vec![
                Axis::values([
                    8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 8192, 16384,
                ]),
                Axis::pow2(0, 8),
            ]);
        }
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
    fn both_backends_use_product_extrapolated_cache2d() {
        for backend in ["flashinfer", "fla_triton"] {
            assert_eq!(
                GdnChunkDeltaRuleSpec::cache_kind(backend),
                CacheKind::Cache2DLinear(Extrapolation::Product)
            );
        }
    }

    #[test]
    fn fla_triton_grid_reaches_sub_chunk_and_many_short_prefills() {
        // Without these cells a batch of 75 8-token prefills was extrapolated
        // to 7% of its measured L40S time; the FlashInfer grid must not grow,
        // or its measured H200/B200 rows stop covering the build.
        let mut cfg = config();
        cfg.backends = vec!["fla_triton"];
        let grid = GdnChunkDeltaRuleSpec::sweep_grid(&cfg);
        assert_eq!(
            grid.axes()[0],
            vec![
                8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 1536.0, 2048.0, 3072.0, 4096.0,
                8192.0, 16384.0
            ]
        );
        assert_eq!(grid.axes()[1].last(), Some(&256.0));
        let mask = GdnChunkDeltaRuleSpec::infeasible_mask(&cfg, &grid);
        assert_eq!(mask.iter().filter(|&&masked| !masked).count(), 98);
        assert_eq!(
            &*GdnChunkDeltaRuleSpec::cache_coords(
                &cfg,
                &GdnChunkDeltaRuleKernelInput {
                    num_tokens: 600,
                    max_sequence_length: 8,
                },
            ),
            &[8.0, 75.0]
        );
        let flashinfer = GdnChunkDeltaRuleSpec::sweep_grid(&config());
        assert_eq!(flashinfer.axes()[0].first(), Some(&64.0));
        assert_eq!(flashinfer.axes()[1].last(), Some(&16.0));
    }

    #[test]
    fn fla_triton_on_l40s_deserializes_and_routes_the_same_args() {
        // The arch switches backends by name alone: an L40S Qwen3.8-27B config
        // (16 key / 48 value heads) must deserialize with `fla_triton` and emit
        // the identical payload schema, only the backend string differing.
        let cfg: GdnChunkDeltaRuleKernelConfig = serde_json::from_value(serde_json::json!({
            "backends": ["fla_triton"],
            "gpu_name": "NVIDIA L40S",
            "num_key_heads": 16,
            "num_heads": 48,
            "key_head_dim": 128,
            "value_head_dim": 128,
            "dtype": "bf16",
        }))
        .unwrap();
        assert_eq!(cfg.backends(), &["fla_triton"]);
        assert_eq!(cfg.gpu_name(), "NVIDIA L40S");
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        let grid = GdnChunkDeltaRuleSpec::sweep_grid(&cfg);
        let fla = GdnChunkDeltaRuleSpec::enumerate(&cfg, &grid, "fla_triton");
        let flashinfer = GdnChunkDeltaRuleSpec::enumerate(&cfg, &grid, "flashinfer");
        assert_eq!(fla.len(), flashinfer.len());
        for (fla, flashinfer) in fla.iter().zip(&flashinfer) {
            let mut fla = fla.fields().clone();
            let mut flashinfer = flashinfer.fields().clone();
            assert_eq!(fla.remove("backend").unwrap(), "fla_triton");
            assert_eq!(flashinfer.remove("backend").unwrap(), "flashinfer");
            assert_eq!(fla, flashinfer);
            assert_eq!(fla["num_heads"], 48);
        }
    }
}
