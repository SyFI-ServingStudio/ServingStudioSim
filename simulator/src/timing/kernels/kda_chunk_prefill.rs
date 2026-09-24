//! GLM-5.3-Flash Kimi Delta Attention (KDA) chunked prefill as ONE callable.
//!
//! Times one `chunk_kda_with_fused_gate` call exactly as the vLLM fork's
//! prefill branch issues it: the vendored FLA Triton chain of 15 launches
//! (q/k/v contiguous copies, qk l2norm, fused safe gate + chunk cumsum, K.Kt,
//! triangular solve, w/u recompute, inter-chunk state scan, output). It is a
//! separate kind from `gdn_chunk_delta_rule` (FlashInfer's single CUTLASS
//! kernel with a scalar per-head gate) and from the six per-stage `gdn_chunk_*`
//! kinds: KDA's gate is per channel and produced inside the callable from
//! `raw_g`, `A_log` and `dt_bias`.
//!
//! The physical input is `(T, L, D)`: total tokens, longest prefill sequence
//! and co-scheduled length-1 decode sequences. The cache axes are the re-axis
//! `(L, R = (T-D)/L, D)`:
//! - `L` is the critical path of the sequential inter-chunk state scan
//!   (~`ceil(L/64)` chunks per sequence), as in `gdn_chunk_delta_rule`;
//! - `R` counts full-length prefill sequences, so `P = T-D >= L` is the
//!   structural `R >= 1` and the single-prefill diagonal `P = L` is the grid
//!   line `R = 1`;
//! - `D` adds one whole chunk (and one state) per decode sequence to every
//!   chunk-parallel launch -- 29 decodes add +36% at equal token count.
//!
//! For `L >= 64` the chunk count `R*L/64 + D` is bilinear in `(L, R)`, which
//! trilinear interpolation reproduces inside a cell.
//!
//! The shortest `L` anchor is 2, not 1: a length-1 "prefill" never reaches
//! this callable (vLLM's GDN-family metadata uses decode threshold 1, so every
//! query-length-1 request is a decode and lands on the `D` axis), and the
//! Python runner rejects the all-length-1, no-decode batch. For `2 <= L <= 64`
//! each prefill sequence is one chunk, so at fixed `R` both the chunk count
//! (`R + D`) and the token-proportional copies (`R*L`) are linear in `L` --
//! exact for the `[2, 64]` cell. Only the prefill-token ceiling `L*R` is
//! masked.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

/// Ceiling on prefill tokens `P = L*R`. Twice the 8,192-token query domain,
/// so cells covering `P <= 8192` keep their upper corners profiled.
const MAX_PREFILL_TOKENS: u64 = 16_384;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KdaChunkPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Local KDA heads (64 / TP).
    pub num_heads: Dim,
    /// K = V head dim (128).
    pub head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

/// The physical caller shape of the non-spec varlen call: `num_tokens` counts
/// prefill plus decode tokens, `max_sequence_length` is the longest prefill
/// sequence, and `num_decode_sequences` length-1 sequences precede the
/// prefills. Requires `num_tokens - num_decode_sequences >= max_sequence_length >= 1`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KdaChunkPrefillKernelInput {
    pub num_tokens: u32,
    pub max_sequence_length: u32,
    pub num_decode_sequences: u32,
}

impl SweepCoords for KdaChunkPrefillKernelInput {
    /// Panics on an input outside the callable's domain: a pure-decode batch
    /// (no prefill) is `fused_recurrent_kda`, a different kind, and anything
    /// with `T - D < L` would otherwise silently read an `R < 1` extrapolation.
    fn coords(&self) -> Coords {
        let prefill_tokens = self.num_tokens.checked_sub(self.num_decode_sequences);
        assert!(
            self.max_sequence_length >= 1
                && prefill_tokens.is_some_and(|p| p >= self.max_sequence_length),
            "kda_chunk_prefill requires num_tokens - num_decode_sequences >= \
             max_sequence_length >= 1, got num_tokens={}, max_sequence_length={}, \
             num_decode_sequences={}",
            self.num_tokens,
            self.max_sequence_length,
            self.num_decode_sequences,
        );
        let prefill_tokens = prefill_tokens.unwrap();
        Coords::new([
            self.max_sequence_length as f64,
            prefill_tokens as f64 / self.max_sequence_length as f64,
            self.num_decode_sequences as f64,
        ])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["num_tokens", "max_sequence_length", "num_decode_sequences"]
    }
}

pub struct KdaChunkPrefillSpec;

impl KernelSpec for KdaChunkPrefillSpec {
    type Config = KdaChunkPrefillKernelConfig;
    type Input = KdaChunkPrefillKernelInput;

    const KIND: KernelKind = "kda_chunk_prefill";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            // L = 2 anchors the sub-chunk regime (chunks = R + D); from one
            // chunk (64) up, powers of two to the 8,192-token query domain.
            Axis::chain([vec![2.0], Axis::pow2(6, 13)]),
            Axis::pow2(0, 9),
            Axis::values([0, 1, 2, 4, 8, 16, 32, 64]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand(|coordinates| {
            prefill_tokens(coordinates[0], coordinates[1]) > MAX_PREFILL_TOKENS
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand(|coordinates| {
            let max_sequence_length = coordinates[0].round() as u64;
            let num_decode_sequences = coordinates[2].round() as u64;
            let num_tokens = prefill_tokens(coordinates[0], coordinates[1])
                .checked_add(num_decode_sequences)
                .expect("KDA token total must fit u64");
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
                .with(
                    "num_decode_sequences",
                    u32::try_from(num_decode_sequences).expect("decode sweep must fit u32"),
                )
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

fn prefill_tokens(max_sequence_length: f64, full_sequences: f64) -> u64 {
    let max_sequence_length = max_sequence_length.round() as u64;
    let full_sequences = full_sequences.round() as u64;
    max_sequence_length
        .checked_mul(full_sequences)
        .expect("KDA prefill token total must fit u64")
}

register_kernel!(KdaChunkPrefillKernel, KdaChunkPrefillSpec);

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{
        KdaChunkPrefillKernelConfig, KdaChunkPrefillKernelInput, KdaChunkPrefillSpec,
        MAX_PREFILL_TOKENS,
    };
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;
    use crate::timing::SweepCoords;

    fn config() -> KdaChunkPrefillKernelConfig {
        KdaChunkPrefillKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads: 16.into(),
            head_dim: 128.into(),
            dtype: DType::Bf16,
        }
    }

    fn input(
        num_tokens: u32,
        max_sequence_length: u32,
        num_decode_sequences: u32,
    ) -> KdaChunkPrefillKernelInput {
        KdaChunkPrefillKernelInput {
            num_tokens,
            max_sequence_length,
            num_decode_sequences,
        }
    }

    /// A physical `(T, L, D)` query (cost log or kernel-query JSON) must land on
    /// `(L, (T-D)/L, D)`; keying on raw T would fold the 29 decodes' +36% into
    /// the prefill-token axis.
    #[test]
    fn physical_input_projects_to_length_full_sequences_and_decodes() {
        let cfg = config();
        let decoded: KdaChunkPrefillKernelInput = serde_json::from_value(serde_json::json!({
            "num_tokens": 2048, "max_sequence_length": 2019, "num_decode_sequences": 29
        }))
        .unwrap();
        assert_eq!(
            &*KdaChunkPrefillSpec::cache_coords(&cfg, &decoded),
            &[2019.0, 1.0, 29.0]
        );
        assert_eq!(
            &*KdaChunkPrefillSpec::cache_coords(&cfg, &input(4108, 1000, 12)),
            &[1000.0, 4.096, 12.0]
        );
        assert_eq!(
            KdaChunkPrefillKernelInput::coord_field_names(),
            &["num_tokens", "max_sequence_length", "num_decode_sequences"]
        );
    }

    /// Inputs outside the callable's domain must not silently extrapolate.
    #[test]
    fn inputs_with_fewer_prefill_tokens_than_the_longest_sequence_are_rejected() {
        for bad in [input(2048, 2020, 29), input(10, 0, 0), input(5, 1, 6)] {
            let result = std::panic::catch_unwind(|| bad.coords());
            assert!(result.is_err(), "{bad:?} must be rejected");
        }
        // The boundary P == L is legal.
        assert_eq!(&*input(2048, 2019, 29).coords(), &[2019.0, 1.0, 29.0]);
    }

    #[test]
    fn mask_keeps_exactly_the_432_cells_within_the_prefill_ceiling() {
        let cfg = config();
        let grid = KdaChunkPrefillSpec::sweep_grid(&cfg);
        assert_eq!(
            grid.axes()[0],
            vec![2.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0]
        );
        assert_eq!(grid.axes()[1].first(), Some(&1.0));
        assert_eq!(grid.axes()[1].last(), Some(&512.0));
        assert_eq!(
            grid.axes()[2],
            vec![0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        );

        let mask = KdaChunkPrefillSpec::infeasible_mask(&cfg, &grid);
        assert_eq!(mask.len(), 720);
        // 54 (L, R) pairs x 8 D.
        assert_eq!(mask.iter().filter(|&&masked| !masked).count(), 432);

        let payloads = KdaChunkPrefillSpec::enumerate(&cfg, &grid, "vllm_triton");
        assert_eq!(payloads.len(), 720);
        for (payload, &masked) in payloads.iter().zip(&mask) {
            let fields = payload.fields();
            let prefill = fields["num_tokens"].as_u64().unwrap()
                - fields["num_decode_sequences"].as_u64().unwrap();
            let length = fields["max_sequence_length"].as_u64().unwrap();
            assert_eq!(prefill % length, 0);
            // Boundary: 64x256 and 8192x2 (P = 16384) stay profiled.
            assert_eq!(masked, prefill > MAX_PREFILL_TOKENS);
        }
    }

    #[test]
    fn every_payload_carries_exactly_the_python_args_schema() {
        let cfg = config();
        let grid = KdaChunkPrefillSpec::sweep_grid(&cfg);
        let expected = BTreeSet::from([
            "backend",
            "dtype",
            "head_dim",
            "max_sequence_length",
            "num_decode_sequences",
            "num_heads",
            "num_tokens",
        ]);
        let mut found_capture_diagonal = false;
        for payload in KdaChunkPrefillSpec::enumerate(&cfg, &grid, "vllm_triton") {
            let fields = payload.fields();
            assert_eq!(
                fields.keys().map(String::as_str).collect::<BTreeSet<_>>(),
                expected
            );
            assert_eq!(fields["backend"], "vllm_triton");
            assert_eq!(fields["num_heads"], 16);
            assert_eq!(fields["head_dim"], 128);
            assert_eq!(fields["dtype"], "bf16");
            found_capture_diagonal |= fields["num_tokens"] == 2080
                && fields["max_sequence_length"] == 2048
                && fields["num_decode_sequences"] == 32;
        }
        assert!(found_capture_diagonal);
    }
}
