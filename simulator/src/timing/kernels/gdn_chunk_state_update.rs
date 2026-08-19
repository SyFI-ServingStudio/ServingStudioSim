//! Qwen Gated `DeltaNet` fused chunk-state update kernel.
//!
//! Stable H200 measurements require three cache axes: `C` captures aggregate
//! chunk work and H snapshots, `N` controls launch programs, and the normalized
//! feasible position `R` retains `M`'s critical recurrence-depth effect without
//! interpolating through impossible `(C,N,M)` corners. Token occupancy at fixed
//! `(C,N,M)` and residual sequence distributions were measured equivalent, so
//! the production envelope profiles canonical full-occupancy rows with `T=64*C`.
//! The public input and profiler payload retain all four physical fields.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

const CHUNK_SIZE: u64 = 64;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnChunkStateUpdateKernelConfig {
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

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct GdnChunkStateUpdateKernelInput {
    pub num_tokens: u32,
    pub num_chunks: u32,
    pub num_sequences: u32,
    pub max_chunks_per_sequence: u32,
}

impl SweepCoords for GdnChunkStateUpdateKernelInput {
    fn coords(&self) -> Coords {
        Coords::new([
            f64::from(self.num_tokens),
            f64::from(self.num_chunks),
            f64::from(self.num_sequences),
            f64::from(self.max_chunks_per_sequence),
        ])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &[
            "num_tokens",
            "num_chunks",
            "num_sequences",
            "max_chunks_per_sequence",
        ]
    }
}

fn geometry_is_feasible(num_chunks: u64, num_sequences: u64, max_chunks: u64) -> bool {
    let Some(minimum_chunks) = max_chunks
        .checked_add(num_sequences)
        .and_then(|value| value.checked_sub(1))
    else {
        return false;
    };
    let Some(maximum_chunks) = num_sequences.checked_mul(max_chunks) else {
        return false;
    };
    minimum_chunks <= num_chunks && num_chunks <= maximum_chunks
}

/// Exact-M bounds implied by `M+N-1 <= C <= N*M` for fixed `(C,N)`.
fn feasible_max_chunk_bounds(num_chunks: u64, num_sequences: u64) -> Option<(u64, u64)> {
    if num_chunks == 0 || num_sequences == 0 || num_sequences > num_chunks {
        return None;
    }
    let minimum = num_chunks.checked_add(num_sequences.checked_sub(1)?)? / num_sequences;
    let maximum = num_chunks.checked_sub(num_sequences)?.checked_add(1)?;
    Some((minimum, maximum))
}

#[allow(
    clippy::cast_precision_loss,
    reason = "num_chunks/minimum/maximum/span are kernel cache sweep-grid coordinates (chunk \
              counts), bounded by the sweep grid (Axis::pow2(0, 6) => <= 64) — far below 2^52"
)]
fn normalized_max_chunk_position(num_chunks: u64, num_sequences: u64, max_chunks: u64) -> f64 {
    let Some((minimum, maximum)) = feasible_max_chunk_bounds(num_chunks, num_sequences) else {
        return -1.0;
    };
    let span = maximum - minimum;
    if span == 0 {
        return match max_chunks.cmp(&minimum) {
            std::cmp::Ordering::Less => -1.0,
            std::cmp::Ordering::Equal => 1.0,
            std::cmp::Ordering::Greater => 2.0,
        };
    }
    (max_chunks as f64 - minimum as f64) / span as f64
}

/// Place the frozen quarter/half landmarks with nearest-integer half-up rounding.
fn max_chunks_at_landmark(num_chunks: u64, num_sequences: u64, position: f64) -> u64 {
    let Some((minimum, maximum)) = feasible_max_chunk_bounds(num_chunks, num_sequences) else {
        // Masked placeholders are never profiled, but enumerate must still emit
        // one schema-valid payload for every raw rectangular cache cell.
        return 1;
    };
    let span = maximum - minimum;
    let offset = if position == 0.0 {
        0
    } else if position == 0.25 {
        span.checked_add(2)
            .expect("quarter-position rounding must fit u64")
            / 4
    } else if position == 0.5 {
        span.checked_add(1)
            .expect("half-position rounding must fit u64")
            / 2
    } else if position == 1.0 {
        span
    } else {
        panic!("unsupported state-update M-position landmark {position}");
    };
    minimum
        .checked_add(offset)
        .expect("state-update M-position placement must fit u64")
}

/// Canonical exact-M sequence chunk counts used by the Python runner.
fn canonical_chunk_counts(
    num_chunks: u64,
    num_sequences: u64,
    max_chunks: u64,
) -> Option<Vec<u64>> {
    if !geometry_is_feasible(num_chunks, num_sequences, max_chunks) {
        return None;
    }
    if num_sequences == 1 {
        return Some(vec![max_chunks]);
    }

    let minimum_chunks = max_chunks
        .checked_add(num_sequences)
        .and_then(|value| value.checked_sub(1))?;
    let residual = num_chunks.checked_sub(minimum_chunks)?;
    let slots = num_sequences - 1;
    let quotient = residual / slots;
    let remainder = residual % slots;
    let mut counts = Vec::with_capacity(usize::try_from(num_sequences).ok()?);
    counts.push(max_chunks);
    for index in 0..slots {
        counts.push(quotient + 1 + u64::from(index < remainder));
    }
    Some(counts)
}

pub struct GdnChunkStateUpdateSpec;

impl KernelSpec for GdnChunkStateUpdateSpec {
    type Config = GdnChunkStateUpdateKernelConfig;
    type Input = GdnChunkStateUpdateKernelInput;

    const KIND: KernelKind = "gdn_chunk_state_update";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::pow2(0, 6),
            // N=3 is an empirically identified launch-program interpolation breakpoint.
            Axis::values([1, 2, 3, 4, 8]),
            vec![0.0, 0.25, 0.5, 1.0],
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn cache_coords(_config: &Self::Config, input: &Self::Input) -> Coords {
        Coords::new([
            f64::from(input.num_chunks),
            f64::from(input.num_sequences),
            normalized_max_chunk_position(
                input.num_chunks.into(),
                input.num_sequences.into(),
                input.max_chunks_per_sequence.into(),
            ),
        ])
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "coordinates are sweep-grid axis values (chunk/sequence counts, always small \
                  non-negative integers per Axis::pow2/values); .round() before the cast keeps \
                  them exact and non-negative"
    )]
    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand(|coordinates| coordinates[1].round() as u64 > coordinates[0].round() as u64)
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "coordinates are sweep-grid axis values (chunk/sequence counts, always small \
                  non-negative integers per Axis::pow2/values); .round() before the cast keeps \
                  them exact and non-negative"
    )]
    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand(|coordinates| {
            let num_chunks = coordinates[0].round() as u64;
            let num_sequences = coordinates[1].round() as u64;
            let max_chunks = max_chunks_at_landmark(num_chunks, num_sequences, coordinates[2]);
            let num_tokens = num_chunks
                .checked_mul(CHUNK_SIZE)
                .expect("state-update full-occupancy token count must fit u64");

            debug_assert!(
                !geometry_is_feasible(num_chunks, num_sequences, max_chunks)
                    || canonical_chunk_counts(num_chunks, num_sequences, max_chunks).is_some()
            );

            ArgsPayload::new()
                .with("backend", backend)
                .with(
                    "num_tokens",
                    u32::try_from(num_tokens).expect("token sweep must fit u32"),
                )
                .with(
                    "num_chunks",
                    u32::try_from(num_chunks).expect("chunk sweep must fit u32"),
                )
                .with(
                    "num_sequences",
                    u32::try_from(num_sequences).expect("sequence sweep must fit u32"),
                )
                .with(
                    "max_chunks_per_sequence",
                    u32::try_from(max_chunks).expect("maximum-chunk sweep must fit u32"),
                )
                .with("num_key_heads", config.num_key_heads.get())
                .with("num_heads", config.num_heads.get())
                .with("key_head_dim", config.key_head_dim.get())
                .with("value_head_dim", config.value_head_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(GdnChunkStateUpdateKernel, GdnChunkStateUpdateSpec);

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashSet};

    use super::{
        canonical_chunk_counts, geometry_is_feasible, max_chunks_at_landmark,
        normalized_max_chunk_position, GdnChunkStateUpdateKernelConfig,
        GdnChunkStateUpdateKernelInput, GdnChunkStateUpdateSpec, CHUNK_SIZE,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};

    fn config() -> GdnChunkStateUpdateKernelConfig {
        GdnChunkStateUpdateKernelConfig {
            backends: vec!["torch", "vllm_triton"],
            gpu_name: "NVIDIA H200".to_string(),
            num_key_heads: 16.into(),
            num_heads: 32.into(),
            key_head_dim: 128.into(),
            value_head_dim: 128.into(),
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_description_serde_routing_and_dtype_tags_match_handoff() {
        let cfg = config();
        assert_eq!(GdnChunkStateUpdateSpec::KIND, "gdn_chunk_state_update");
        assert_eq!(
            GdnChunkStateUpdateSpec::profile_kind(),
            "gdn_chunk_state_update"
        );
        assert_eq!(cfg.backends(), &["torch", "vllm_triton"]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.num_key_heads, 16);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.key_head_dim, 128);
        assert_eq!(cfg.value_head_dim, 128);
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
        assert_eq!(
            cfg.describe_config(),
            serde_json::json!({
                "backends": ["torch", "vllm_triton"],
                "gpu_name": "NVIDIA H200",
                "num_key_heads": {"value": 16, "expression": null, "bindings": {}},
                "num_heads": {"value": 32, "expression": null, "bindings": {}},
                "key_head_dim": {"value": 128, "expression": null, "bindings": {}},
                "value_head_dim": {"value": 128, "expression": null, "bindings": {}},
                "dtype": "bf16",
            })
        );
        let encoded = serde_json::to_value(&cfg).unwrap();
        let decoded: GdnChunkStateUpdateKernelConfig =
            serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
    }

    #[test]
    fn physical_input_serde_names_slot_and_qwen_coordinates_are_exact() {
        let qwen = GdnChunkStateUpdateKernelInput {
            num_tokens: 128,
            num_chunks: 2,
            num_sequences: 1,
            max_chunks_per_sequence: 2,
        };
        assert_eq!(&*qwen.coords(), &[128.0, 2.0, 1.0, 2.0]);
        assert_eq!(
            GdnChunkStateUpdateKernelInput::coord_field_names(),
            &[
                "num_tokens",
                "num_chunks",
                "num_sequences",
                "max_chunks_per_sequence"
            ]
        );
        assert_eq!(
            &*GdnChunkStateUpdateSpec::cache_coords(&config(), &qwen),
            &[2.0, 1.0, 1.0]
        );
        let slot: SlotInput = qwen.clone().into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({
                "num_tokens": 128,
                "num_chunks": 2,
                "num_sequences": 1,
                "max_chunks_per_sequence": 2,
            })
        );
        let decoded: GdnChunkStateUpdateKernelInput =
            serde_json::from_value(serde_json::to_value(qwen).unwrap()).unwrap();
        assert_eq!(&*decoded.coords(), &[128.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn normalized_coordinates_cover_boundaries_interiors_and_degenerate_spans() {
        let cfg = config();
        let input = |c, n, m| GdnChunkStateUpdateKernelInput {
            num_tokens: 64 * c,
            num_chunks: c,
            num_sequences: n,
            max_chunks_per_sequence: m,
        };

        assert_eq!(
            &*GdnChunkStateUpdateSpec::cache_coords(&cfg, &input(64, 8, 8)),
            &[64.0, 8.0, 0.0]
        );
        assert_eq!(
            &*GdnChunkStateUpdateSpec::cache_coords(&cfg, &input(64, 8, 57)),
            &[64.0, 8.0, 1.0]
        );
        assert_eq!(
            &*GdnChunkStateUpdateSpec::cache_coords(&cfg, &input(64, 8, 20)),
            &[64.0, 8.0, 12.0 / 49.0]
        );
        assert_eq!(
            &*GdnChunkStateUpdateSpec::cache_coords(&cfg, &input(12, 3, 4)),
            &[12.0, 3.0, 0.0]
        );
        assert_eq!(
            &*GdnChunkStateUpdateSpec::cache_coords(&cfg, &input(4, 1, 4)),
            &[4.0, 1.0, 1.0]
        );
        assert_eq!(normalized_max_chunk_position(4, 1, 3), -1.0);
        assert_eq!(normalized_max_chunk_position(4, 1, 5), 2.0);
        assert_eq!(normalized_max_chunk_position(1, 2, 1), -1.0);

        let partial = GdnChunkStateUpdateKernelInput {
            num_tokens: 65,
            num_chunks: 2,
            num_sequences: 1,
            max_chunks_per_sequence: 2,
        };
        let full = GdnChunkStateUpdateKernelInput {
            num_tokens: 128,
            ..partial.clone()
        };
        assert_eq!(
            GdnChunkStateUpdateSpec::cache_coords(&cfg, &partial).as_slice(),
            GdnChunkStateUpdateSpec::cache_coords(&cfg, &full).as_slice()
        );
    }

    #[test]
    fn quarter_and_half_landmarks_use_checked_half_up_rounding() {
        assert_eq!(
            [0.0, 0.25, 0.5, 1.0].map(|r| max_chunks_at_landmark(8, 2, r)),
            [4, 5, 6, 7]
        );
        assert_eq!(
            [0.0, 0.25, 0.5, 1.0].map(|r| max_chunks_at_landmark(16, 4, r)),
            [4, 6, 9, 13]
        );
        assert_eq!(
            [0.0, 0.25, 0.5, 1.0].map(|r| max_chunks_at_landmark(2, 1, r)),
            [2, 2, 2, 2]
        );
        assert_eq!(max_chunks_at_landmark(1, 2, 0.5), 1);
    }

    #[test]
    fn sweep_axes_counts_mask_and_unique_feasible_rows_are_exact() {
        let grid = GdnChunkStateUpdateSpec::sweep_grid(&config());
        assert_eq!(grid.axes()[0], vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]);
        assert_eq!(grid.axes()[1], vec![1.0, 2.0, 3.0, 4.0, 8.0]);
        assert_eq!(grid.axes()[2], vec![0.0, 0.25, 0.5, 1.0]);
        assert_eq!(grid.axes().iter().map(Vec::len).product::<usize>(), 140);

        let mask = GdnChunkStateUpdateSpec::infeasible_mask(&config(), &grid);
        assert_eq!(mask.len(), 140);
        assert_eq!(mask.iter().filter(|&&masked| masked).count(), 32);
        assert_eq!(mask.iter().filter(|&&masked| !masked).count(), 108);

        let payloads = GdnChunkStateUpdateSpec::enumerate(&config(), &grid, "vllm_triton");
        let feasible = payloads
            .iter()
            .zip(&mask)
            .filter_map(|(payload, &masked)| {
                (!masked).then(|| {
                    let fields = payload.fields();
                    (
                        fields["num_tokens"].as_u64().unwrap(),
                        fields["num_chunks"].as_u64().unwrap(),
                        fields["num_sequences"].as_u64().unwrap(),
                        fields["max_chunks_per_sequence"].as_u64().unwrap(),
                    )
                })
            })
            .collect::<HashSet<_>>();
        assert_eq!(feasible.len(), 73);
    }

    #[test]
    fn every_feasible_row_has_exact_full_occupancy_and_canonical_exact_m_counts() {
        let grid = GdnChunkStateUpdateSpec::sweep_grid(&config());
        let payloads = GdnChunkStateUpdateSpec::enumerate(&config(), &grid, "vllm_triton");
        let mask = GdnChunkStateUpdateSpec::infeasible_mask(&config(), &grid);
        for (payload, &masked) in payloads.iter().zip(&mask) {
            let fields = payload.fields();
            let t = fields["num_tokens"].as_u64().unwrap();
            let c = fields["num_chunks"].as_u64().unwrap();
            let n = fields["num_sequences"].as_u64().unwrap();
            let m = fields["max_chunks_per_sequence"].as_u64().unwrap();
            assert_eq!(t, CHUNK_SIZE.checked_mul(c).unwrap());
            assert_eq!(masked, n > c);
            if masked {
                continue;
            }
            assert!(geometry_is_feasible(c, n, m));
            let counts = canonical_chunk_counts(c, n, m).unwrap();
            assert_eq!(counts.len(), n as usize);
            assert!(counts.iter().all(|&count| count > 0));
            assert_eq!(counts.iter().sum::<u64>(), c);
            assert_eq!(counts.iter().copied().max(), Some(m));
            assert!(counts.iter().all(|&count| count <= m));
            assert!(m + n - 1 <= c && c <= n * m);
        }
    }

    #[test]
    fn n3_breakpoint_preserves_all_56_rows_and_adds_exactly_17_specs() {
        let normalized_rows = |sequence_axis: &[u64]| {
            [1_u64, 2, 4, 8, 16, 32, 64]
                .into_iter()
                .flat_map(|c| {
                    sequence_axis.iter().copied().flat_map(move |n| {
                        [0.0, 0.25, 0.5, 1.0]
                            .into_iter()
                            .filter(move |_| n <= c)
                            .map(move |r| (CHUNK_SIZE * c, c, n, max_chunks_at_landmark(c, n, r)))
                    })
                })
                .collect::<HashSet<_>>()
        };
        let old = normalized_rows(&[1, 2, 4, 8]);
        assert_eq!(old.len(), 56);

        let grid = GdnChunkStateUpdateSpec::sweep_grid(&config());
        let mask = GdnChunkStateUpdateSpec::infeasible_mask(&config(), &grid);
        let new = GdnChunkStateUpdateSpec::enumerate(&config(), &grid, "vllm_triton")
            .into_iter()
            .zip(mask)
            .filter_map(|(payload, masked)| {
                (!masked).then(|| {
                    let fields = payload.fields();
                    (
                        fields["num_tokens"].as_u64().unwrap(),
                        fields["num_chunks"].as_u64().unwrap(),
                        fields["num_sequences"].as_u64().unwrap(),
                        fields["max_chunks_per_sequence"].as_u64().unwrap(),
                    )
                })
            })
            .collect::<HashSet<_>>();
        assert_eq!(new.len(), 73);
        assert_eq!(new.intersection(&old).count(), 56);
        assert!(old.is_subset(&new));

        let expected_additions = [
            (4, 3, 2),
            (8, 3, 3),
            (8, 3, 4),
            (8, 3, 5),
            (8, 3, 6),
            (16, 3, 6),
            (16, 3, 8),
            (16, 3, 10),
            (16, 3, 14),
            (32, 3, 11),
            (32, 3, 16),
            (32, 3, 21),
            (32, 3, 30),
            (64, 3, 22),
            (64, 3, 32),
            (64, 3, 42),
            (64, 3, 62),
        ]
        .into_iter()
        .map(|(c, n, m)| (CHUNK_SIZE * c, c, n, m))
        .collect::<HashSet<_>>();
        assert_eq!(
            new.difference(&old).copied().collect::<HashSet<_>>(),
            expected_additions
        );
        assert!(new.difference(&old).all(|&(_, _, n, _)| n == 3));
    }

    #[test]
    fn every_payload_has_exact_fields_static_values_and_qwen_anchor() {
        let cfg = config();
        let grid = GdnChunkStateUpdateSpec::sweep_grid(&cfg);
        let payloads = GdnChunkStateUpdateSpec::enumerate(&cfg, &grid, "vllm_triton");
        let expected_fields = BTreeSet::from([
            "backend",
            "dtype",
            "key_head_dim",
            "max_chunks_per_sequence",
            "num_chunks",
            "num_heads",
            "num_key_heads",
            "num_sequences",
            "num_tokens",
            "value_head_dim",
        ]);
        let mut found_qwen = false;
        for payload in payloads {
            assert_eq!(
                payload
                    .fields()
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>(),
                expected_fields
            );
            let fields = payload.fields();
            assert_eq!(fields["backend"], "vllm_triton");
            assert_eq!(fields["num_key_heads"], 16);
            assert_eq!(fields["num_heads"], 32);
            assert_eq!(fields["key_head_dim"], 128);
            assert_eq!(fields["value_head_dim"], 128);
            assert_eq!(fields["dtype"], "bf16");
            found_qwen |= fields["num_tokens"] == 128
                && fields["num_chunks"] == 2
                && fields["num_sequences"] == 1
                && fields["max_chunks_per_sequence"] == 2;
        }
        assert!(found_qwen);
    }

    #[test]
    fn both_backends_use_cache3d_and_preserve_static_payload_configuration() {
        let cfg = config();
        let grid = GdnChunkStateUpdateSpec::sweep_grid(&cfg);
        for backend in ["torch", "vllm_triton"] {
            assert_eq!(
                GdnChunkStateUpdateSpec::cache_kind(backend),
                CacheKind::Cache3DLinear
            );
            let payload = &GdnChunkStateUpdateSpec::enumerate(&cfg, &grid, backend)[0];
            let fields = payload.fields();
            assert_eq!(fields["backend"], backend);
            assert_eq!(fields["num_key_heads"], 16);
            assert_eq!(fields["num_heads"], 32);
            assert_eq!(fields["key_head_dim"], 128);
            assert_eq!(fields["value_head_dim"], 128);
            assert_eq!(fields["dtype"], "bf16");
        }
    }

    #[test]
    fn production_envelope_maximum_row_allocation_is_bounded() {
        // Packed Qwen operands and wrapper outputs, including int32 metadata:
        // 28,800*T + 1,048,576*C + 4,194,304*N + 8*C + 8*N + 8 bytes.
        let t = 64_u64 * 64;
        let c = 64_u64;
        let n = 8_u64;
        let bytes = 28_800_u64 * t + 1_048_576_u64 * c + 4_194_304_u64 * n + 8 * c + 8 * n + 8;
        assert_eq!(bytes, 218_628_680);
    }
}
