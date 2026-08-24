//! Qwen Gated `DeltaNet` chunk-local triangular solve kernel.
//!
//! `max_chunk_tokens` is deliberately static configuration identity: it changes
//! which chunk occupancies are valid, while runtime callers still provide only
//! physical `(num_tokens, num_chunks)`. The 2D cache projects those inputs to a
//! rectangular `(C, P)` domain, where `C` is the solve launch count and `P` is
//! the normalized position between that C's minimum and maximum feasible token
//! counts. `KernelSpec::cache_coords` owns this config-aware projection; the
//! Input's serialized fields and `coord_field_names` remain physical.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

const MAX_TOKENS: u64 = 262_144;
const MAX_CHUNK_TOKENS: u32 = 64;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GdnChunkSolveTrilKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub max_chunk_tokens: Dim,
    pub num_heads: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct GdnChunkSolveTrilKernelInput {
    pub num_tokens: u32,
    pub num_chunks: u32,
}

impl SweepCoords for GdnChunkSolveTrilKernelInput {
    fn coords(&self) -> Coords {
        Coords::new([f64::from(self.num_tokens), f64::from(self.num_chunks)])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["num_tokens", "num_chunks"]
    }
}

fn checked_max_chunk_tokens(config: &GdnChunkSolveTrilKernelConfig) -> u32 {
    let max_chunk_tokens = config.max_chunk_tokens.get();
    assert!(
        (1..=MAX_CHUNK_TOKENS).contains(&max_chunk_tokens),
        "gdn_chunk_solve_tril max_chunk_tokens must be in 1..={MAX_CHUNK_TOKENS}, got {max_chunk_tokens}"
    );
    max_chunk_tokens
}

fn chunk_count_axis(max_chunk_tokens: u32) -> Vec<f64> {
    let max_chunks = MAX_TOKENS - u64::from(max_chunk_tokens) + 1;
    let max_log2 = u64::BITS - 1 - max_chunks.leading_zeros();
    let mut axis = Axis::pow2(0, max_log2);
    if max_chunk_tokens == MAX_CHUNK_TOKENS {
        // C=12 and C=17 are empirically identified launch-count interpolation
        // breakpoints for the production M64 configuration.
        for breakpoint in [12.0, 17.0] {
            axis.insert(
                axis.partition_point(|&chunks| chunks < breakpoint),
                breakpoint,
            );
        }
    }
    axis
}

fn position_axis() -> Vec<f64> {
    vec![0.0, 0.25, 0.5, 1.0]
}

fn feasible_token_bounds(max_chunk_tokens: u32, num_chunks: u64) -> (u64, u64) {
    let max_chunk_tokens = u64::from(max_chunk_tokens);
    let minimum = max_chunk_tokens
        .checked_add(num_chunks)
        .and_then(|value| value.checked_sub(1))
        .expect("GDN triangular-solve minimum token count must fit u64");
    let maximum = num_chunks
        .checked_mul(max_chunk_tokens)
        .expect("GDN triangular-solve maximum token count must fit u64")
        .min(MAX_TOKENS);
    (minimum, maximum)
}

fn normalized_position(num_tokens: u64, minimum: u64, maximum: u64) -> f64 {
    if minimum < maximum {
        #[allow(
            clippy::cast_precision_loss,
            reason = "token/chunk counts here are cache-sweep coordinates, far below f64's 2^53 exact-integer bound"
        )]
        {
            (num_tokens as f64 - minimum as f64) / (maximum - minimum) as f64
        }
    } else if minimum == maximum {
        match num_tokens.cmp(&minimum) {
            std::cmp::Ordering::Less => -1.0,
            std::cmp::Ordering::Equal => 1.0,
            std::cmp::Ordering::Greater => 2.0,
        }
    } else if num_tokens < minimum {
        -1.0
    } else {
        2.0
    }
}

fn landmark_offset(span: u64, position: f64) -> u64 {
    if position == 0.0 {
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
        panic!("unexpected GDN triangular-solve position landmark {position}")
    }
}

fn physical_tokens_at_position(minimum: u64, maximum: u64, position: f64) -> u64 {
    let span = maximum
        .checked_sub(minimum)
        .expect("GDN triangular-solve cache cells must have a feasible token span");
    minimum
        .checked_add(landmark_offset(span, position))
        .expect("GDN triangular-solve token landmark must fit u64")
}

pub struct GdnChunkSolveTrilSpec;

impl KernelSpec for GdnChunkSolveTrilSpec {
    type Config = GdnChunkSolveTrilKernelConfig;
    type Input = GdnChunkSolveTrilKernelInput;

    const KIND: KernelKind = "gdn_chunk_solve_tril";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let max_chunk_tokens = checked_max_chunk_tokens(config);
        SweepGrid::new(vec![chunk_count_axis(max_chunk_tokens), position_axis()])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        let max_chunk_tokens = checked_max_chunk_tokens(config);
        let num_chunks = u64::from(input.num_chunks);
        let num_tokens = u64::from(input.num_tokens);
        let (minimum, maximum) = feasible_token_bounds(max_chunk_tokens, num_chunks);
        #[allow(
            clippy::cast_precision_loss,
            reason = "num_chunks is a GDN chunk count, far below f64's 2^53 exact-integer bound"
        )]
        Coords::new([
            num_chunks as f64,
            normalized_position(num_tokens, minimum, maximum),
        ])
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        let max_chunk_tokens = checked_max_chunk_tokens(config);
        grid.expand_2d(|num_chunks, position| {
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "rounded sweep-grid chunk coordinate is a non-negative small chunk count, well within u64"
            )]
            let num_chunks = num_chunks.round() as u64;
            let (minimum, maximum) = feasible_token_bounds(max_chunk_tokens, num_chunks);
            let num_tokens = physical_tokens_at_position(minimum, maximum, position);
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
                .with("max_chunk_tokens", max_chunk_tokens)
                .with("num_heads", config.num_heads.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(GdnChunkSolveTrilKernel, GdnChunkSolveTrilSpec);

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet, HashSet};

    use super::{
        feasible_token_bounds, GdnChunkSolveTrilKernelConfig, GdnChunkSolveTrilKernelInput,
        GdnChunkSolveTrilSpec, MAX_TOKENS,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config(max_chunk_tokens: u32) -> GdnChunkSolveTrilKernelConfig {
        GdnChunkSolveTrilKernelConfig {
            backends: vec!["torch", "vllm_triton"],
            gpu_name: "NVIDIA H200".to_string(),
            max_chunk_tokens: max_chunk_tokens.into(),
            num_heads: 32.into(),
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_description_serde_routing_and_dtype_tags_match_handoff() {
        let cfg = config(64);

        assert_eq!(GdnChunkSolveTrilSpec::KIND, "gdn_chunk_solve_tril");
        assert_eq!(
            GdnChunkSolveTrilSpec::profile_kind(),
            "gdn_chunk_solve_tril"
        );
        assert_eq!(cfg.backends(), &["torch", "vllm_triton"]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.max_chunk_tokens, 64);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.dtype, DType::Bf16);
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
        assert_eq!(
            cfg.describe_config(),
            serde_json::json!({
                "backends": ["torch", "vllm_triton"],
                "gpu_name": "NVIDIA H200",
                "max_chunk_tokens": {"value": 64, "expression": null, "bindings": {}},
                "num_heads": {"value": 32, "expression": null, "bindings": {}},
                "dtype": "bf16",
            })
        );

        let encoded = serde_json::to_value(&cfg).unwrap();
        let decoded: GdnChunkSolveTrilKernelConfig =
            serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
    }

    #[test]
    fn physical_input_serde_names_and_slot_payload_survive_config_aware_reaxis() {
        let cfg = config(64);
        let qwen = GdnChunkSolveTrilKernelInput {
            num_tokens: 128,
            num_chunks: 2,
        };
        assert_eq!(&*qwen.coords(), &[128.0, 2.0]);
        assert_eq!(
            &*GdnChunkSolveTrilSpec::cache_coords(&cfg, &qwen),
            &[2.0, 1.0]
        );
        assert_eq!(
            GdnChunkSolveTrilKernelInput::coord_field_names(),
            &["num_tokens", "num_chunks"]
        );
        let slot: SlotInput = qwen.clone().into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_tokens": 128, "num_chunks": 2})
        );
        let round_trip: GdnChunkSolveTrilKernelInput =
            serde_json::from_value(serde_json::to_value(qwen).unwrap()).unwrap();
        assert_eq!(&*round_trip.coords(), &[128.0, 2.0]);
    }

    #[test]
    fn config_aware_coords_cover_boundaries_interiors_and_safe_extrapolation() {
        let cfg = config(64);
        for (input, expected) in [
            (
                GdnChunkSolveTrilKernelInput {
                    num_tokens: 75,
                    num_chunks: 12,
                },
                [12.0, 0.0],
            ),
            (
                GdnChunkSolveTrilKernelInput {
                    num_tokens: 768,
                    num_chunks: 12,
                },
                [12.0, 1.0],
            ),
            (
                GdnChunkSolveTrilKernelInput {
                    num_tokens: 80,
                    num_chunks: 17,
                },
                [17.0, 0.0],
            ),
            (
                GdnChunkSolveTrilKernelInput {
                    num_tokens: 1088,
                    num_chunks: 17,
                },
                [17.0, 1.0],
            ),
        ] {
            assert_eq!(
                &*GdnChunkSolveTrilSpec::cache_coords(&cfg, &input),
                &expected
            );
        }

        let interior = GdnChunkSolveTrilKernelInput {
            num_tokens: 86,
            num_chunks: 2,
        };
        let interior_coords = GdnChunkSolveTrilSpec::cache_coords(&cfg, &interior);
        assert_eq!(interior_coords[0], 2.0);
        assert!((interior_coords[1] - 1.0 / 3.0).abs() < f64::EPSILON);

        let below = GdnChunkSolveTrilSpec::cache_coords(
            &cfg,
            &GdnChunkSolveTrilKernelInput {
                num_tokens: 64,
                num_chunks: 2,
            },
        );
        let above = GdnChunkSolveTrilSpec::cache_coords(
            &cfg,
            &GdnChunkSolveTrilKernelInput {
                num_tokens: 129,
                num_chunks: 2,
            },
        );
        let zero_chunks = GdnChunkSolveTrilSpec::cache_coords(
            &cfg,
            &GdnChunkSolveTrilKernelInput {
                num_tokens: 0,
                num_chunks: 0,
            },
        );
        assert!(below[1] < 0.0);
        assert!(above[1] > 1.0);
        assert_eq!(zero_chunks[0], 0.0);
        assert!(zero_chunks[1] < 0.0 || zero_chunks[1] > 1.0);
    }

    #[test]
    fn m64_sweep_is_exactly_twenty_by_four_and_unique() {
        let grid = GdnChunkSolveTrilSpec::sweep_grid(&config(64));
        let expected_chunks = [
            1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 17.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0,
            4096.0, 8192.0, 16384.0, 32768.0, 65536.0, 131072.0,
        ];
        let expected_positions = [0.0, 0.25, 0.5, 1.0];

        assert_eq!(
            grid.axes(),
            &[expected_chunks.to_vec(), expected_positions.to_vec()]
        );
        assert_eq!(grid.axes().len(), 2);
        assert_eq!(grid.axes()[0].len() * grid.axes()[1].len(), 80);
        for axis in grid.axes() {
            assert!(axis.windows(2).all(|pair| pair[0] < pair[1]));
            assert_eq!(
                axis.iter()
                    .map(|value| value.to_bits())
                    .collect::<HashSet<_>>()
                    .len(),
                axis.len()
            );
        }
    }

    #[test]
    fn m64_grid_is_rectangular_feasible_and_has_only_c1_duplicates() {
        let cfg = config(64);
        let grid = GdnChunkSolveTrilSpec::sweep_grid(&cfg);
        let mask = GdnChunkSolveTrilSpec::infeasible_mask(&cfg, &grid);
        let payloads = GdnChunkSolveTrilSpec::enumerate(&cfg, &grid, "vllm_triton");
        let mut counts = BTreeMap::new();

        assert!(mask.is_empty());
        assert_eq!(payloads.len(), 80);
        for payload in &payloads {
            let tokens = payload.fields()["num_tokens"].as_u64().unwrap();
            let chunks = payload.fields()["num_chunks"].as_u64().unwrap();
            let (minimum, maximum) = feasible_token_bounds(64, chunks);
            assert!((minimum..=maximum).contains(&tokens));
            assert!(tokens <= MAX_TOKENS);
            *counts.entry((tokens, chunks)).or_insert(0usize) += 1;
        }
        assert_eq!(counts.len(), 77);
        assert_eq!(counts.get(&(64, 1)), Some(&4));
        assert!(counts
            .iter()
            .all(|(&point, &count)| point == (64, 1) || count == 1));
    }

    #[test]
    fn c17_adds_exactly_four_physical_rows_and_preserves_the_previous_grid() {
        let cfg = config(64);
        let grid = GdnChunkSolveTrilSpec::sweep_grid(&cfg);
        let payloads = GdnChunkSolveTrilSpec::enumerate(&cfg, &grid, "vllm_triton");
        let physical_rows = payloads
            .iter()
            .map(|payload| {
                (
                    payload.fields()["num_tokens"].as_u64().unwrap(),
                    payload.fields()["num_chunks"].as_u64().unwrap(),
                )
            })
            .collect::<BTreeSet<_>>();

        let previous_chunks = [
            1_u64, 2, 4, 8, 12, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
            65536, 131072,
        ];
        let positions = [0.0, 0.25, 0.5, 1.0];
        let previous_rows = previous_chunks
            .into_iter()
            .flat_map(|chunks| {
                let (minimum, maximum) = feasible_token_bounds(64, chunks);
                positions.into_iter().map(move |position| {
                    (
                        super::physical_tokens_at_position(minimum, maximum, position),
                        chunks,
                    )
                })
            })
            .collect::<BTreeSet<_>>();

        assert_eq!(previous_rows.len(), 73);
        assert!(previous_rows.is_subset(&physical_rows));
        assert_eq!(
            physical_rows
                .difference(&previous_rows)
                .copied()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([(80, 17), (332, 17), (584, 17), (1088, 17)])
        );
    }

    #[test]
    fn enumerate_uses_half_up_landmarks_and_exact_python_payload() {
        let cfg = config(64);
        let grid = GdnChunkSolveTrilSpec::sweep_grid(&cfg);
        let payloads = GdnChunkSolveTrilSpec::enumerate(&cfg, &grid, "vllm_triton");
        let expected_names = BTreeSet::from([
            "backend",
            "dtype",
            "max_chunk_tokens",
            "num_chunks",
            "num_heads",
            "num_tokens",
        ]);

        assert_eq!(payloads.len(), 80);
        for payload in &payloads {
            assert_eq!(
                payload
                    .fields()
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>(),
                expected_names
            );
            assert_eq!(payload.fields().len(), 6);
            assert_eq!(payload.backend(), Some("vllm_triton"));
            assert_eq!(payload.fields()["max_chunk_tokens"], Value::from(64));
            assert_eq!(payload.fields()["num_heads"], Value::from(32));
            assert_eq!(payload.fields()["dtype"], Value::from("bf16"));
        }

        let qwen = &payloads[4 + 3];
        assert_eq!(qwen.fields()["num_tokens"], Value::from(128));
        assert_eq!(qwen.fields()["num_chunks"], Value::from(2));
        assert_eq!(
            payloads[4..2 * 4]
                .iter()
                .map(|payload| payload.fields()["num_tokens"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![65, 81, 97, 128]
        );
        let c12_payloads = &payloads[4 * 4..5 * 4];
        assert_eq!(
            c12_payloads
                .iter()
                .map(|payload| payload.fields()["num_tokens"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![75, 248, 422, 768]
        );
        assert!(c12_payloads.iter().all(|payload| {
            payload.fields()["num_chunks"] == 12
                && payload.fields()["max_chunk_tokens"] == 64
                && payload.fields()["num_heads"] == 32
                && payload.fields()["dtype"] == "bf16"
        }));
        let c17_payloads = &payloads[6 * 4..7 * 4];
        assert_eq!(
            c17_payloads
                .iter()
                .map(|payload| payload.fields()["num_tokens"].as_u64().unwrap())
                .collect::<Vec<_>>(),
            vec![80, 332, 584, 1088]
        );
        assert!(c17_payloads.iter().all(|payload| {
            payload.fields()["num_chunks"] == 17
                && payload.fields()["max_chunk_tokens"] == 64
                && payload.fields()["num_heads"] == 32
                && payload.fields()["dtype"] == "bf16"
        }));
        let maximum = &payloads[19 * 4 + 3];
        assert_eq!(maximum.fields()["num_tokens"], Value::from(262_144));
        assert_eq!(maximum.fields()["num_chunks"], Value::from(131_072));
    }

    #[test]
    fn both_backends_remain_linear_2d_without_profile_override() {
        for backend in ["torch", "vllm_triton"] {
            assert_eq!(
                GdnChunkSolveTrilSpec::cache_kind(backend),
                CacheKind::Cache2DLinear(Extrapolation::Product)
            );
        }
        assert_eq!(
            GdnChunkSolveTrilSpec::profile_kind(),
            GdnChunkSolveTrilSpec::KIND
        );
    }

    #[test]
    fn dynamic_configs_remain_rectangular_and_normalized() {
        let m1 = config(1);
        let m1_grid = GdnChunkSolveTrilSpec::sweep_grid(&m1);
        let m1_mask = GdnChunkSolveTrilSpec::infeasible_mask(&m1, &m1_grid);
        assert_eq!(
            m1_grid.axes()[0],
            vec![
                1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0,
                8192.0, 16384.0, 32768.0, 65536.0, 131072.0, 262144.0,
            ]
        );
        assert_eq!(m1_grid.axes()[1], vec![0.0, 0.25, 0.5, 1.0]);
        assert!(m1_mask.is_empty());
        let m1_payloads = GdnChunkSolveTrilSpec::enumerate(&m1, &m1_grid, "torch");
        assert_eq!(m1_payloads.len(), 76);
        assert_eq!(
            m1_payloads
                .iter()
                .map(|payload| {
                    (
                        payload.fields()["num_tokens"].as_u64().unwrap(),
                        payload.fields()["num_chunks"].as_u64().unwrap(),
                    )
                })
                .collect::<BTreeSet<_>>()
                .len(),
            19
        );
        assert_eq!(
            &*GdnChunkSolveTrilSpec::cache_coords(
                &m1,
                &GdnChunkSolveTrilKernelInput {
                    num_tokens: 7,
                    num_chunks: 7,
                },
            ),
            &[7.0, 1.0]
        );
        for (tokens, expected_position) in [(6, -1.0), (8, 2.0)] {
            assert_eq!(
                &*GdnChunkSolveTrilSpec::cache_coords(
                    &m1,
                    &GdnChunkSolveTrilKernelInput {
                        num_tokens: tokens,
                        num_chunks: 7,
                    },
                ),
                &[7.0, expected_position]
            );
        }

        let m33 = config(33);
        let m33_grid = GdnChunkSolveTrilSpec::sweep_grid(&m33);
        let m33_mask = GdnChunkSolveTrilSpec::infeasible_mask(&m33, &m33_grid);
        assert_eq!(
            m33_grid.axes()[0],
            vec![
                1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0,
                8192.0, 16384.0, 32768.0, 65536.0, 131072.0,
            ]
        );
        assert_eq!(m33_grid.axes()[1], vec![0.0, 0.25, 0.5, 1.0]);
        assert!(m33_mask.is_empty());
        let m33_payloads = GdnChunkSolveTrilSpec::enumerate(&m33, &m33_grid, "torch");
        assert_eq!(m33_payloads.len(), 72);
        assert!(m33_payloads.iter().all(|payload| {
            let tokens = payload.fields()["num_tokens"].as_u64().unwrap();
            let chunks = payload.fields()["num_chunks"].as_u64().unwrap();
            let (minimum, maximum) = feasible_token_bounds(33, chunks);
            (minimum..=maximum).contains(&tokens)
        }));
        assert_eq!(
            &*GdnChunkSolveTrilSpec::cache_coords(
                &m33,
                &GdnChunkSolveTrilKernelInput {
                    num_tokens: 51,
                    num_chunks: 3,
                },
            ),
            &[3.0, 0.25]
        );
    }

    #[test]
    fn invalid_static_max_chunk_tokens_is_rejected() {
        for invalid in [0, 65] {
            let result = std::panic::catch_unwind(|| {
                let _ = GdnChunkSolveTrilSpec::sweep_grid(&config(invalid));
            });
            assert!(result.is_err(), "M={invalid} must be rejected");
        }
    }
}
