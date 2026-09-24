//! GLM-5.2 DSA sparse MLA attention kernel.
//!
//! The cache uses physical `(num_queries, num_cache_tokens)` coordinates. The
//! static `valid_counts_pattern` deterministically derives Python's canonical
//! flattened valid-slot-count RLE during enumeration; RLE text is never a cache
//! coordinate. The first R.4 gate found pattern-specific interpolation defects:
//! uniform adds S6/S11/S91, causal pins its domain boundary and scattered-cache
//! misses, and speculative removes unsupported odd-Q grid holes while adding
//! Q134/Q266 and S3/S23/S45/S182. Cache math and the public query contract are
//! unchanged.
//!
//! A speculative verify step submits one group of `next_n` query rows per
//! request, so the group width is a physical property of the pattern rather than
//! a fixed two. `ValidCountsPattern::SpeculativeGroups` carries it, and derives
//! its axes from the group-of-two anchors by request count.
//!
//! GLM-5.3-Flash's pooled indexer (kpool) selects `index_topk` pools plus the
//! trailing partial pool, so a row's active count is
//! `min(n, index_topk + n mod index_kpool) <= index_topk + index_kpool - 1`
//! (2051), inside a page table `round_up(2051, 128) = 2176` wide
//! (`selected_k`). `ValidCountsPattern::PooledUniformFull` carries the pooling
//! so decode derives `u:min(ctx, index_topk + index_kpool - 1)` instead of
//! clipping at the page-table width, and sweeps its own smaller grid.

use crate::timing::bridge::{ArgsPayload, DType, KernelKind, de_backends};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{KernelSpec, register_kernel};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

const UNIFORM_QUERY_AXIS: [u32; 27] = [
    1, 2, 4, 8, 16, 32, 64, 127, 128, 129, 131, 132, 133, 255, 256, 257, 263, 264, 265, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536,
];
const UNIFORM_CACHE_AXIS: [u32; 30] = [
    1, 2, 4, 6, 8, 11, 16, 32, 63, 64, 65, 91, 127, 128, 129, 256, 512, 1024, 2047, 2048, 2049,
    4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
];
const CAUSAL_QUERY_AXIS: [u32; 37] = [
    1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 127, 128, 129, 130, 131, 132, 133, 200, 254, 255,
    256, 257, 258, 263, 264, 265, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];
const CAUSAL_CACHE_AXIS: [u32; 41] = [
    1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 63, 64, 65, 90, 127, 128, 129, 130, 182, 200, 255, 256,
    260, 511, 512, 513, 724, 1024, 2047, 2048, 2049, 4096, 8192, 16384, 32768, 65536, 131072,
    262144, 524288, 1048576,
];
const SPECULATIVE_QUERY_AXIS: [u32; 20] = [
    2, 4, 8, 16, 32, 64, 128, 132, 134, 256, 264, 266, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    65536,
];
const SPECULATIVE_CACHE_AXIS: [u32; 31] = [
    1, 2, 3, 4, 8, 16, 23, 32, 45, 63, 64, 65, 127, 128, 129, 182, 256, 512, 1024, 2047, 2048,
    2049, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
];
/// Pooled decode queries stop at the TRTLLM 32,768-query launch guard, so the
/// grid carries no cell the backend must mask. 96 splits the 64..127 decode
/// batch gap, where the saturated time bends (B200 b3 fidelity).
const POOLED_QUERY_AXIS: [u32; 21] = [
    1, 2, 4, 8, 16, 32, 64, 96, 127, 128, 129, 255, 256, 257, 512, 1024, 2048, 4096, 8192, 16384,
    32768,
];
/// Pooled context points below and above the saturating count. The
/// pooling-derived `index_topk` and cap points are added by
/// `pooled_cache_axis`; past the cap only the gather footprint grows.
/// 768 splits 512..1024, where small-batch time dips then climbs.
const POOLED_CACHE_AXIS: [u32; 21] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 1048576,
];
/// Backends whose TRTLLM-gen launch rejects more than 32,768 query rows.
const TRTLLM_BACKENDS: [&str; 2] = ["flashinfer_trtllm_fp8", "flashinfer_trtllm_fp8_vllm_fork"];

/// Which valid-slot-count shape a config sweeps.
///
/// One choice, not two: the shape decides both the measured grid and the
/// flattened count vector that grid enumerates.
#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidCountsPattern {
    /// Every query row sees the whole cache.
    UniformFull,
    /// One prefill request's rows walk a causal ramp over its own context.
    CausalTail,
    /// A speculative verify step: `group_size` rows per request, each seeing one
    /// fewer context token than the next.
    SpeculativeGroups { group_size: u32 },
    /// Pooled-indexer decode: every row sees its whole context, but the indexer
    /// keeps at most `index_topk` pools of `index_kpool` tokens plus the
    /// partial tail pool, so the count saturates at
    /// `index_topk + index_kpool - 1`, below the `selected_k` page-table width.
    PooledUniformFull { index_topk: u32, index_kpool: u32 },
}

impl ValidCountsPattern {
    /// Choose the decode shape for a step submitting `verify_width` query rows
    /// per request, or `None` if that is not a width.
    ///
    /// One row is ordinary decode and every row sees the whole cache. Two or
    /// more rows are a verify group walking a descending context ramp. The two
    /// sweep different measured grids — uniform is dense in query count, while
    /// speculative anchors on request count — so this picks a grid, not just an
    /// encoding.
    pub fn for_decode(verify_width: u32) -> Option<Self> {
        match verify_width {
            0 => None,
            1 => Some(Self::UniformFull),
            group_size => Some(Self::SpeculativeGroups { group_size }),
        }
    }

    /// Largest active count a row can reach under this pattern.
    ///
    /// Unpooled patterns clip at the `selected_k` page-table width. The pooled
    /// pattern keeps `index_topk` whole pools plus up to `index_kpool - 1`
    /// trailing tokens (vLLM `_expand_pools_and_append_tail`), which must fit
    /// the page table.
    fn valid_count_cap(self, selected_k: u32) -> u32 {
        match self {
            Self::PooledUniformFull {
                index_topk,
                index_kpool,
            } => {
                assert!(
                    index_topk > 0 && index_kpool > 0,
                    "pooled index_topk and index_kpool must be positive"
                );
                let cap = index_topk
                    .checked_add(index_kpool - 1)
                    .expect("pooled valid-count cap overflows u32");
                assert!(
                    cap <= selected_k,
                    "pooled valid-count cap {cap} exceeds selected_k {selected_k}"
                );
                cap
            }
            Self::UniformFull | Self::CausalTail | Self::SpeculativeGroups { .. } => selected_k,
        }
    }
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaSparseMlaAttentionKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub num_kv_heads: Dim,
    pub selected_k: u32,
    pub latent_dim: Dim,
    pub rope_dim: Dim,
    pub value_dim: Dim,
    pub softmax_scale_denominator: u32,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub cache_dtype: DType,
    pub index_dtype: String,
    pub output_dtype: DType,
    pub valid_counts_pattern: ValidCountsPattern,
    pub index_distribution: String,
    pub cache_layout: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DsaSparseMlaAttentionKernelInput {
    pub num_queries: u32,
    pub num_cache_tokens: u32,
}

pub struct DsaSparseMlaAttentionSpec;

impl KernelSpec for DsaSparseMlaAttentionSpec {
    type Config = DsaSparseMlaAttentionKernelConfig;
    type Input = DsaSparseMlaAttentionKernelInput;

    const KIND: KernelKind = "dsa_sparse_mla_attention";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let (query_axis, cache_axis): (Vec<u32>, Vec<u32>) = match config.valid_counts_pattern {
            ValidCountsPattern::UniformFull => {
                (UNIFORM_QUERY_AXIS.to_vec(), UNIFORM_CACHE_AXIS.to_vec())
            }
            ValidCountsPattern::CausalTail => {
                (CAUSAL_QUERY_AXIS.to_vec(), CAUSAL_CACHE_AXIS.to_vec())
            }
            ValidCountsPattern::SpeculativeGroups { group_size } => (
                speculative_query_axis(group_size),
                speculative_cache_axis(group_size, config.selected_k),
            ),
            ValidCountsPattern::PooledUniformFull { index_topk, .. } => (
                POOLED_QUERY_AXIS.to_vec(),
                pooled_cache_axis(
                    index_topk,
                    config
                        .valid_counts_pattern
                        .valid_count_cap(config.selected_k),
                ),
            ),
        };
        SweepGrid::new(vec![
            Axis::values(query_axis.iter().copied()),
            Axis::values(cache_axis.iter().copied()),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // Sparse attention reads `selected_k` tokens per query, and `selected_k`
        // is a Config field, not an axis. Work is therefore ~linear in
        // `num_queries`; `num_cache_tokens` sets the gather footprint the
        // indices point into, not how many of them are read.
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        // TRTLLM-gen rejects the 65,536-query launch on B200. Production caps
        // one iteration far below that, so 32,768 remains a generous measured
        // extrapolation guard without requiring an unsupported launch.
        let trtllm_query_too_large = |num_queries: f64| {
            TRTLLM_BACKENDS
                .iter()
                .any(|backend| config.backends.contains(backend))
                && num_queries > 32_768.0
        };
        match config.valid_counts_pattern {
            ValidCountsPattern::UniformFull | ValidCountsPattern::PooledUniformFull { .. } => {
                grid.expand_2d(|num_queries, _| trtllm_query_too_large(num_queries))
            }
            ValidCountsPattern::CausalTail => grid.expand_2d(|num_queries, num_cache_tokens| {
                num_queries > num_cache_tokens || trtllm_query_too_large(num_queries)
            }),
            ValidCountsPattern::SpeculativeGroups { group_size } => {
                grid.expand_2d(|num_queries, _| {
                    num_queries as u32 % group_size != 0 || trtllm_query_too_large(num_queries)
                })
            }
        }
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        assert!(
            config.softmax_scale_denominator > 0,
            "softmax_scale_denominator must be positive"
        );
        grid.expand_2d(|num_queries, num_cache_tokens| {
            let num_queries = num_queries as u32;
            let num_cache_tokens = num_cache_tokens as u32;
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_queries", num_queries)
                .with("num_cache_tokens", num_cache_tokens)
                .with("num_heads", config.num_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("selected_k", config.selected_k)
                .with("latent_dim", config.latent_dim.get())
                .with("rope_dim", config.rope_dim.get())
                .with("value_dim", config.value_dim.get())
                .with(
                    "softmax_scale",
                    1.0 / f64::from(config.softmax_scale_denominator),
                )
                .with("q_dtype", config.q_dtype.as_str())
                .with("cache_dtype", config.cache_dtype.as_str())
                .with("index_dtype", config.index_dtype.clone())
                .with("output_dtype", config.output_dtype.as_str())
                .with(
                    "valid_counts",
                    canonical_valid_counts(
                        config.valid_counts_pattern,
                        num_queries,
                        num_cache_tokens,
                        config
                            .valid_counts_pattern
                            .valid_count_cap(config.selected_k),
                    ),
                )
                .with("index_distribution", config.index_distribution.clone())
                .with("cache_layout", config.cache_layout.clone())
        })
    }
}

/// Derive the exact canonical Python valid-count encoding from physical axes.
///
/// `k` is the pattern's valid-count cap (`ValidCountsPattern::valid_count_cap`):
/// `selected_k` for unpooled patterns, `index_topk + index_kpool - 1` pooled.
fn canonical_valid_counts(pattern: ValidCountsPattern, q: u32, s: u32, k: u32) -> String {
    match pattern {
        ValidCountsPattern::UniformFull | ValidCountsPattern::PooledUniformFull { .. } => {
            format!("u:{}x{q}", s.min(k))
        }
        ValidCountsPattern::CausalTail => {
            if q > s {
                return masked_placeholder(q);
            }
            let first = s - q + 1;
            let last = s;
            if q == 1 || first >= k {
                format!("u:{}x{q}", first.min(k))
            } else if last <= k {
                format!("r:{first}..{last}")
            } else {
                debug_assert!(first < k && k < last);
                format!("c:{first}..{last}@{k}")
            }
        }
        ValidCountsPattern::SpeculativeGroups { group_size } => {
            if q % group_size != 0 {
                return masked_placeholder(q);
            }
            canonical_speculative_counts(q, s, k, group_size)
        }
    }
}

/// Scale the frozen group-of-two query anchors by request count.
///
/// Each grid point is `requests * group_size` query rows, so holding the request
/// count fixed keeps the measured batch sizes — and the odd-Q holes the R.4 gate
/// removed — aligned across group sizes.
fn speculative_query_axis(group_size: u32) -> Vec<u32> {
    assert!(group_size > 0, "speculative group_size must be positive");
    if group_size == 2 {
        return SPECULATIVE_QUERY_AXIS.to_vec();
    }
    SPECULATIVE_QUERY_AXIS
        .iter()
        .map(|num_queries| {
            (num_queries / 2)
                .checked_mul(group_size)
                .expect("speculative query axis overflows u32")
        })
        .collect()
}

/// Add the group-sensitive transition points a wider verify group crosses.
///
/// The RLE form changes shape where the group first fits in the context
/// (`num_cache_tokens` near `group_size`) and where it straddles the `selected_k`
/// clip (`selected_k ..= selected_k + group_size - 1`). Without samples at
/// those boundaries the cache interpolates across a discontinuity it never
/// measured. The frozen group-of-two axis is returned untouched.
fn speculative_cache_axis(group_size: u32, selected_k: u32) -> Vec<u32> {
    if group_size == 2 {
        return SPECULATIVE_CACHE_AXIS.to_vec();
    }
    let saturated_context = selected_k
        .checked_add(group_size - 1)
        .expect("speculative saturation boundary overflows u32");
    let mut axis = SPECULATIVE_CACHE_AXIS.to_vec();
    axis.extend(
        [
            group_size - 1,
            group_size,
            group_size + 1,
            selected_k.saturating_sub(group_size - 1),
            selected_k,
            selected_k + 1,
            saturated_context.saturating_sub(1),
            saturated_context,
            saturated_context
                .checked_add(1)
                .expect("speculative saturation guard overflows u32"),
        ]
        .into_iter()
        .filter(|value| *value > 0),
    );
    axis.sort_unstable();
    axis.dedup();
    axis
}

/// Mirror Python's canonical encoder for `q / group_size` repeats of one
/// speculative verify group.
///
/// Row `i` of a group sees `num_cache_tokens - (group_size - 1 - i)` context
/// tokens, clipped to `selected_k` above and to zero below, so the group is
/// non-decreasing and one group determines the whole vector. Building just that
/// group keeps enumeration `O(group_size)` instead of `O(num_queries)` at the
/// largest request anchor, and is exact:
///
/// * `u:` needs every row equal, which is a property of the group alone.
/// * `r:` and `c:` compare against a ramp seeded at row 0. With more than one
///   repeat, row `group_size` is back at the group's first value while the ramp
///   has advanced, and the ramp can only have stalled by reaching `selected_k` —
///   which would have made the group uniform. So both forms require exactly one
///   repeat.
/// * Python's `g:` uses the minimal period. A non-decreasing sequence whose
///   period divides its length is constant, so a non-uniform group's minimal
///   period is the whole group, and Fine and Wilf's theorem rules out a shorter
///   period appearing only once the group is repeated.
fn canonical_speculative_counts(q: u32, s: u32, k: u32, group_size: u32) -> String {
    let group: Vec<u32> = (0..group_size)
        .map(|row| s.saturating_sub(group_size - 1 - row).min(k))
        .collect();
    let first = group[0];
    let last = group[group.len() - 1];
    if first == last {
        return format!("u:{first}x{q}");
    }

    if q == group_size {
        let is_ramp = |cap: u32| {
            group
                .iter()
                .enumerate()
                .all(|(row, &count)| count == (first + row as u32).min(cap))
        };
        if is_ramp(u32::MAX) {
            return format!("r:{first}..{last}");
        }
        let unclipped_last = first + group_size - 1;
        if first < k && k < unclipped_last && unclipped_last <= s && is_ramp(k) {
            return format!("c:{first}..{unclipped_last}@{k}");
        }
    }

    let counts = group
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("g:({counts})x{}", q / group_size)
}

/// Pooled context axis: the fixed points plus `index_topk` (last unsaturated
/// power-of-two anchor) and the saturating cap, so the linear cache never
/// interpolates across the point where the count stops growing.
fn pooled_cache_axis(index_topk: u32, cap: u32) -> Vec<u32> {
    let mut axis = POOLED_CACHE_AXIS.to_vec();
    axis.extend([index_topk, cap]);
    axis.sort_unstable();
    axis.dedup();
    axis
}

fn masked_placeholder(q: u32) -> String {
    format!("u:0x{q}")
}

register_kernel!(DsaSparseMlaAttentionKernel, DsaSparseMlaAttentionSpec);

#[cfg(test)]
mod tests {
    use super::ValidCountsPattern;
    use super::{
        CAUSAL_CACHE_AXIS, CAUSAL_QUERY_AXIS, DsaSparseMlaAttentionKernelConfig,
        DsaSparseMlaAttentionKernelInput, DsaSparseMlaAttentionSpec, SPECULATIVE_CACHE_AXIS,
        SPECULATIVE_QUERY_AXIS, UNIFORM_CACHE_AXIS, UNIFORM_QUERY_AXIS, canonical_valid_counts,
        speculative_cache_axis, speculative_query_axis,
    };
    use crate::timing::bridge::{ArgsPayload, DType};
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{Dim, SlotInput, SweepCoords};
    use serde_json::Value;
    use std::collections::BTreeSet;

    const TORCH_BACKEND: &str = "torch";
    const FLASHMLA_BACKEND: &str = "vllm_flashmla_bf16";
    fn config(pattern: ValidCountsPattern) -> DsaSparseMlaAttentionKernelConfig {
        DsaSparseMlaAttentionKernelConfig {
            backends: vec![TORCH_BACKEND, FLASHMLA_BACKEND],
            gpu_name: "NVIDIA H200".to_string(),
            num_heads: Dim::param("num_attention_heads", 64),
            num_kv_heads: Dim::param("num_kv_heads", 1),
            selected_k: 2048,
            latent_dim: Dim::param("kv_lora_rank", 512),
            rope_dim: Dim::param("qk_rope_head_dim", 64),
            value_dim: Dim::param("kv_lora_rank", 512),
            softmax_scale_denominator: 16,
            q_dtype: DType::Bf16,
            cache_dtype: DType::Bf16,
            index_dtype: "int32".to_string(),
            output_dtype: DType::Bf16,
            valid_counts_pattern: pattern,
            index_distribution: "recent_contiguous".to_string(),
            cache_layout: "token_major_mqa_bf16_latent_rope".to_string(),
        }
    }

    #[test]
    fn config_kind_and_dtype_identity_match_the_python_handoff() {
        let cfg = config(ValidCountsPattern::UniformFull);

        assert_eq!(DsaSparseMlaAttentionSpec::KIND, "dsa_sparse_mla_attention");
        assert_eq!(
            DsaSparseMlaAttentionSpec::profile_kind(),
            "dsa_sparse_mla_attention"
        );
        assert_eq!(cfg.backends(), &[TORCH_BACKEND, FLASHMLA_BACKEND]);
        assert_eq!(cfg.gpu_name(), "NVIDIA H200");
        assert_eq!(cfg.num_heads, 64);
        assert_eq!(cfg.num_kv_heads, 1);
        assert_eq!(cfg.selected_k, 2048);
        assert_eq!(cfg.latent_dim, 512);
        assert_eq!(cfg.rope_dim, 64);
        assert_eq!(cfg.value_dim, 512);
        assert_eq!(cfg.softmax_scale_denominator, 16);
        assert_eq!(cfg.q_dtype, DType::Bf16);
        assert_eq!(cfg.cache_dtype, DType::Bf16);
        assert_eq!(cfg.output_dtype, DType::Bf16);
        assert_eq!(cfg.index_dtype, "int32");
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), Some(DType::Bf16));
    }

    #[test]
    fn describe_config_preserves_rich_dimensions_and_static_modes() {
        assert_eq!(
            config(ValidCountsPattern::CausalTail).describe_config(),
            serde_json::json!({
                "backends": [TORCH_BACKEND, FLASHMLA_BACKEND],
                "gpu_name": "NVIDIA H200",
                "num_heads": rich_dim("num_attention_heads", 64),
                "num_kv_heads": rich_dim("num_kv_heads", 1),
                "selected_k": 2048,
                "latent_dim": rich_dim("kv_lora_rank", 512),
                "rope_dim": rich_dim("qk_rope_head_dim", 64),
                "value_dim": rich_dim("kv_lora_rank", 512),
                "softmax_scale_denominator": 16,
                "q_dtype": "bf16",
                "cache_dtype": "bf16",
                "index_dtype": "int32",
                "output_dtype": "bf16",
                "valid_counts_pattern": ValidCountsPattern::CausalTail,
                "index_distribution": "recent_contiguous",
                "cache_layout": "token_major_mqa_bf16_latent_rope",
            })
        );
    }

    #[test]
    fn input_coordinates_deserialization_and_slot_input_are_physical() {
        let input: DsaSparseMlaAttentionKernelInput =
            serde_json::from_str(r#"{"num_queries":128,"num_cache_tokens":2049}"#).unwrap();

        assert_eq!(&*input.coords(), &[128.0, 2049.0]);
        assert_eq!(
            DsaSparseMlaAttentionKernelInput::coord_field_names(),
            &["num_queries", "num_cache_tokens"]
        );
        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_queries":128,"num_cache_tokens":2049})
        );
    }

    #[test]
    fn grids_have_the_exact_pattern_specific_axes_and_boundaries() {
        assert_grid(
            ValidCountsPattern::UniformFull,
            &UNIFORM_QUERY_AXIS,
            &UNIFORM_CACHE_AXIS,
            810,
        );
        assert_grid(
            ValidCountsPattern::CausalTail,
            &CAUSAL_QUERY_AXIS,
            &CAUSAL_CACHE_AXIS,
            1_517,
        );
        assert_grid(
            ValidCountsPattern::SpeculativeGroups { group_size: 2 },
            &SPECULATIVE_QUERY_AXIS,
            &SPECULATIVE_CACHE_AXIS,
            620,
        );

        // Every pattern's cache axis now reaches the arch's full 1,048,576-token
        // timing domain, and every query axis reaches 65,536.
        for pattern in [
            ValidCountsPattern::UniformFull,
            ValidCountsPattern::CausalTail,
            ValidCountsPattern::SpeculativeGroups { group_size: 2 },
        ] {
            let grid = DsaSparseMlaAttentionSpec::sweep_grid(&config(pattern));
            assert_eq!(grid.axes()[0].last(), Some(&65536.0));
            assert_eq!(grid.axes()[1].last(), Some(&1_048_576.0));
        }

        let uniform =
            DsaSparseMlaAttentionSpec::sweep_grid(&config(ValidCountsPattern::UniformFull));
        assert_contiguous(&uniform.axes()[0], &[127, 128, 129]);
        assert_contiguous(&uniform.axes()[0], &[131, 132, 133]);
        assert_contiguous(&uniform.axes()[0], &[255, 256, 257, 263, 264, 265]);
        assert_contiguous(&uniform.axes()[1], &[2, 4, 6, 8, 11, 16]);
        assert_contiguous(&uniform.axes()[1], &[65, 91, 127]);
        assert_contiguous(&uniform.axes()[1], &[2047, 2048, 2049]);

        let causal = DsaSparseMlaAttentionSpec::sweep_grid(&config(ValidCountsPattern::CausalTail));
        assert_contiguous(&causal.axes()[0], &[1, 2, 3, 4, 6, 8, 11]);
        assert_contiguous(&causal.axes()[0], &[127, 128, 129, 130, 131, 132, 133]);
        assert_contiguous(&causal.axes()[0], &[254, 255, 256, 257, 258, 263, 264, 265]);
        assert_contiguous(&causal.axes()[1], &[127, 128, 129, 130, 182, 200]);
        assert_contiguous(&causal.axes()[1], &[255, 256, 260, 511, 512, 513, 724]);
        assert_contiguous(&causal.axes()[1], &[2047, 2048, 2049]);

        let speculative =
            DsaSparseMlaAttentionSpec::sweep_grid(&config(ValidCountsPattern::SpeculativeGroups {
                group_size: 2,
            }));
        assert!(
            speculative.axes()[0]
                .iter()
                .all(|query| *query as u32 % 2 == 0)
        );
        assert_contiguous(&speculative.axes()[0], &[128, 132, 134]);
        assert_contiguous(&speculative.axes()[0], &[256, 264, 266]);
        assert_contiguous(&speculative.axes()[1], &[16, 23, 32, 45, 63]);
        assert_contiguous(&speculative.axes()[1], &[129, 182, 256]);
        assert_contiguous(&speculative.axes()[1], &[2047, 2048, 2049]);
    }

    #[test]
    fn masks_match_each_frozen_pattern_domain() {
        let uniform_grid =
            DsaSparseMlaAttentionSpec::sweep_grid(&config(ValidCountsPattern::UniformFull));
        let uniform = mask_for(ValidCountsPattern::UniformFull, &uniform_grid);
        assert_mask_split(&uniform, 810, 0);

        let causal_grid =
            DsaSparseMlaAttentionSpec::sweep_grid(&config(ValidCountsPattern::CausalTail));
        let causal = mask_for(ValidCountsPattern::CausalTail, &causal_grid);
        assert_mask_split(&causal, 858, 659);
        assert!(!masked(&causal, &causal_grid, 3, 3));
        assert!(!masked(&causal, &causal_grid, 3, 4));
        assert!(masked(&causal, &causal_grid, 3, 2));
        assert!(!masked(&causal, &causal_grid, 132, 182));
        assert!(masked(&causal, &causal_grid, 200, 182));
        assert!(!masked(&causal, &causal_grid, 255, 255));
        assert!(masked(&causal, &causal_grid, 256, 255));
        assert!(!masked(&causal, &causal_grid, 4096, 131072));
        assert!(!masked(&causal, &causal_grid, 65536, 1_048_576));
        assert!(masked(&causal, &causal_grid, 65536, 32768));

        let speculative_grid =
            DsaSparseMlaAttentionSpec::sweep_grid(&config(ValidCountsPattern::SpeculativeGroups {
                group_size: 2,
            }));
        let speculative = mask_for(
            ValidCountsPattern::SpeculativeGroups { group_size: 2 },
            &speculative_grid,
        );
        assert_mask_split(&speculative, 620, 0);
        assert!(!masked(&speculative, &speculative_grid, 2, 1));
        assert!(!masked(&speculative, &speculative_grid, 32, 2048));
        assert!(!masked(&speculative, &speculative_grid, 134, 65536));
        assert!(!masked(&speculative, &speculative_grid, 266, 65536));
        assert!(!masked(&speculative, &speculative_grid, 4096, 131072));
        assert!(!masked(&speculative, &speculative_grid, 65536, 1_048_576));
    }

    #[test]
    fn trtllm_fp8_masks_only_the_rejected_65536_query_row() {
        let mut cfg = config(ValidCountsPattern::CausalTail);
        cfg.backends = vec!["flashinfer_trtllm_fp8"];
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        let mask = DsaSparseMlaAttentionSpec::infeasible_mask(&cfg, &grid);

        assert!(!masked(&mask, &grid, 32768, 1_048_576));
        assert!(masked(&mask, &grid, 65536, 1_048_576));
    }

    const FORK_BACKEND: &str = "flashinfer_trtllm_fp8_vllm_fork";

    /// GLM-5.3-Flash TP4 decode on B200, the exact values of b2-dsa 5.1.
    fn pooled_fork_config() -> DsaSparseMlaAttentionKernelConfig {
        DsaSparseMlaAttentionKernelConfig {
            backends: vec![FORK_BACKEND],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads: Dim::param("num_attention_heads", 16),
            num_kv_heads: Dim::param("num_kv_heads", 1),
            selected_k: 2176,
            latent_dim: Dim::param("kv_lora_rank", 512),
            rope_dim: Dim::param("qk_rope_head_dim", 0),
            value_dim: Dim::param("kv_lora_rank", 512),
            softmax_scale_denominator: 16,
            q_dtype: DType::Fp8E4m3,
            cache_dtype: DType::Fp8E4m3,
            index_dtype: "int32".to_string(),
            output_dtype: DType::Bf16,
            valid_counts_pattern: ValidCountsPattern::PooledUniformFull {
                index_topk: 2048,
                index_kpool: 4,
            },
            index_distribution: "unique_scattered_pages".to_string(),
            cache_layout: "hnd_paged_mqa_fp8_latent".to_string(),
        }
    }

    /// Catches a pooled decode clipping at the 2176-wide page table (up to 125
    /// phantom active slots per row) or any drift from the fork runner's args.
    #[test]
    fn pooled_fork_payloads_cap_at_topk_plus_kpool_minus_one() {
        let cfg = pooled_fork_config();
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        let payloads = DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, FORK_BACKEND);

        for (s, expected) in [(1024, 1024), (2048, 2048), (2051, 2051), (4096, 2051)] {
            let fields = payload_for(&payloads, &grid, 32, s).fields();
            assert_eq!(
                fields.get("valid_counts"),
                Some(&Value::from(format!("u:{expected}x32")))
            );
        }
        let fields = payload_for(&payloads, &grid, 1, 1_048_576).fields();
        assert_eq!(fields.get("valid_counts"), Some(&Value::from("u:2051x1")));
        assert_eq!(
            serde_json::to_value(fields).unwrap(),
            serde_json::json!({
                "backend": FORK_BACKEND,
                "num_queries": 1,
                "num_cache_tokens": 1_048_576,
                "num_heads": 16,
                "num_kv_heads": 1,
                "selected_k": 2176,
                "latent_dim": 512,
                "rope_dim": 0,
                "value_dim": 512,
                "softmax_scale": 0.0625,
                "q_dtype": "fp8_e4m3",
                "cache_dtype": "fp8_e4m3",
                "index_dtype": "int32",
                "output_dtype": "bf16",
                "valid_counts": "u:2051x1",
                "index_distribution": "unique_scattered_pages",
                "cache_layout": "hnd_paged_mqa_fp8_latent",
            })
        );
        assert_eq!(cfg.compute_dtype(), Some(DType::Fp8E4m3));
        assert_eq!(cfg.kv_dtype(), Some(DType::Fp8E4m3));
    }

    /// Catches the pooled grid losing the saturation bracket or growing past
    /// the 500-feasible-coordinate ceiling.
    #[test]
    fn pooled_grid_brackets_saturation_within_the_coordinate_ceiling() {
        let cfg = pooled_fork_config();
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        assert_contiguous(&grid.axes()[1], &[1024, 1536, 2048, 2051, 4096]);
        assert_eq!(grid.axes()[0].last(), Some(&32768.0));
        assert_eq!(grid.axes()[1].last(), Some(&1_048_576.0));
        let mask = DsaSparseMlaAttentionSpec::infeasible_mask(&cfg, &grid);
        assert_mask_split(&mask, 21 * 23, 0);
    }

    #[test]
    #[should_panic(expected = "exceeds selected_k")]
    fn pooled_cap_must_fit_the_page_table() {
        let mut cfg = pooled_fork_config();
        cfg.selected_k = 2048;
        DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
    }

    /// Catches the fork backend escaping the TRTLLM >32,768-query guard.
    #[test]
    fn fork_backend_shares_the_trtllm_query_guard() {
        let mut cfg = config(ValidCountsPattern::UniformFull);
        cfg.backends = vec![FORK_BACKEND];
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        let mask = DsaSparseMlaAttentionSpec::infeasible_mask(&cfg, &grid);
        assert!(!masked(&mask, &grid, 32768, 1_048_576));
        assert!(masked(&mask, &grid, 65536, 1));

        let pooled = pooled_fork_config();
        let synthetic = crate::timing::sweep::SweepGrid::new(vec![
            crate::timing::sweep::Axis::values([32768, 65536]),
            crate::timing::sweep::Axis::values([2051]),
        ]);
        assert_eq!(
            DsaSparseMlaAttentionSpec::infeasible_mask(&pooled, &synthetic),
            vec![false, true]
        );
    }

    #[test]
    fn speculative_odd_query_defense_remains_for_synthetic_grids() {
        let grid = crate::timing::sweep::SweepGrid::new(vec![
            crate::timing::sweep::Axis::values([2, 3, 4]),
            crate::timing::sweep::Axis::values([1, 2]),
        ]);
        let mask = mask_for(
            ValidCountsPattern::SpeculativeGroups { group_size: 2 },
            &grid,
        );

        assert_mask_split(&mask, 4, 2);
        assert!(masked(&mask, &grid, 3, 1));
        assert!(masked(&mask, &grid, 3, 2));
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 2 },
                3,
                2,
                2048
            ),
            "u:0x3"
        );
    }

    #[test]
    fn canonical_valid_counts_obeys_python_precedence_for_every_pattern() {
        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::UniformFull, 1, 1, 2048),
            "u:1x1"
        );
        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::UniformFull, 256, 131072, 2048),
            "u:2048x256"
        );

        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::CausalTail, 1, 1, 2048),
            "u:1x1"
        );
        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::CausalTail, 128, 128, 2048),
            "r:1..128"
        );
        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::CausalTail, 128, 2048, 2048),
            "r:1921..2048"
        );
        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::CausalTail, 128, 2049, 2048),
            "c:1922..2049@2048"
        );
        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::CausalTail, 128, 4096, 2048),
            "u:2048x128"
        );
        assert_eq!(
            canonical_valid_counts(ValidCountsPattern::CausalTail, 4096, 1, 2048),
            "u:0x4096"
        );

        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 2 },
                2,
                1,
                2048
            ),
            "r:0..1"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 2 },
                32,
                2048,
                2048
            ),
            "g:(2047,2048)x16"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 2 },
                2,
                2049,
                2048
            ),
            "u:2048x2"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 2 },
                127,
                2048,
                2048
            ),
            "u:0x127"
        );

        // Group six, at coordinates its own grid actually visits. `r:`/`c:` need
        // a single repeat, so they only appear at the smallest query anchor.
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                6,
                5,
                2048
            ),
            "r:0..5"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                6,
                2043,
                2048
            ),
            "r:2038..2043"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                6,
                2049,
                2048
            ),
            "c:2044..2049@2048"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                6,
                3,
                2048
            ),
            "g:(0,0,0,1,2,3)x1"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                12,
                2048,
                2048
            ),
            "g:(2043,2044,2045,2046,2047,2048)x2"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                12,
                2049,
                2048
            ),
            "g:(2044,2045,2046,2047,2048,2048)x2"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                12,
                4096,
                2048
            ),
            "u:2048x12"
        );
        assert_eq!(
            canonical_valid_counts(
                ValidCountsPattern::SpeculativeGroups { group_size: 6 },
                7,
                4096,
                2048
            ),
            "u:0x7"
        );
    }

    /// Group two is a measured grid, not just an encoding. Generalizing the
    /// derivation must leave both its axes and its RLE bytes exactly where the
    /// R.4 gate put them.
    #[test]
    fn group_two_keeps_its_frozen_axes_and_bytes() {
        assert_eq!(speculative_query_axis(2), SPECULATIVE_QUERY_AXIS);
        assert_eq!(speculative_cache_axis(2, 2048), SPECULATIVE_CACHE_AXIS);

        // The generalized encoder must reproduce the frozen group-two bytes on
        // every coordinate of the frozen grid, not just the sampled ones.
        for &num_queries in SPECULATIVE_QUERY_AXIS.iter() {
            for &num_cache_tokens in SPECULATIVE_CACHE_AXIS.iter() {
                let first = num_cache_tokens.saturating_sub(1).min(2048);
                let second = num_cache_tokens.min(2048);
                let expected = if first == second {
                    format!("u:{first}x{num_queries}")
                } else if num_queries == 2 {
                    format!("r:{first}..{second}")
                } else {
                    format!("g:({first},{second})x{}", num_queries / 2)
                };
                assert_eq!(
                    canonical_valid_counts(
                        ValidCountsPattern::SpeculativeGroups { group_size: 2 },
                        num_queries,
                        num_cache_tokens,
                        2048
                    ),
                    expected,
                    "group-two identity drifted at ({num_queries}, {num_cache_tokens})"
                );
            }
        }
    }

    /// A wider verify group changes RLE shape where the group first fits the
    /// context and where it straddles the `selected_k` clip. Anchoring the query
    /// axis on request count keeps every cell feasible; dropping either cache
    /// boundary leaves the cache interpolating across a discontinuity.
    #[test]
    fn group_six_grid_anchors_requests_and_pins_its_transitions() {
        let cfg = config(ValidCountsPattern::SpeculativeGroups { group_size: 6 });
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);

        assert_eq!(grid.axes()[0].len(), SPECULATIVE_QUERY_AXIS.len());
        assert_eq!(grid.axes()[0][0], 6.0);
        assert_eq!(grid.axes()[0].last(), Some(&196_608.0));
        assert!(grid.axes()[0].iter().all(|q| *q as u32 % 6 == 0));
        for (index, &num_queries) in SPECULATIVE_QUERY_AXIS.iter().enumerate() {
            assert_eq!(grid.axes()[0][index], f64::from(num_queries / 2 * 6));
        }

        // Retain existing points and add both sides of the all-rows-saturated boundary.
        assert_eq!(grid.axes()[1].len(), SPECULATIVE_CACHE_AXIS.len() + 7);
        assert_contiguous(&grid.axes()[1], &[4, 5, 6, 7, 8]);
        assert_contiguous(&grid.axes()[1], &[1024, 2043, 2047, 2048, 2049]);
        assert_contiguous(&grid.axes()[1], &[2049, 2052, 2053, 2054, 4096]);
        assert_eq!(grid.axes()[1].last(), Some(&1_048_576.0));

        let mask = mask_for(
            ValidCountsPattern::SpeculativeGroups { group_size: 6 },
            &grid,
        );
        assert_mask_split(&mask, 760, 0);

        let payloads = DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, FLASHMLA_BACKEND);
        assert_eq!(payloads.len(), 760);
        assert_payload(
            payload_for(&payloads, &grid, 12, 2052),
            12,
            2052,
            "g:(2047,2048,2048,2048,2048,2048)x2".to_string(),
        );
        assert_payload(
            payload_for(&payloads, &grid, 12, 2053),
            12,
            2053,
            "u:2048x12".to_string(),
        );
        assert_payload(
            payload_for(&payloads, &grid, 6, 2049),
            6,
            2049,
            "c:2044..2049@2048".to_string(),
        );
        assert_payload(
            payload_for(&payloads, &grid, 12, 2043),
            12,
            2043,
            "g:(2038,2039,2040,2041,2042,2043)x2".to_string(),
        );
    }

    #[test]
    fn both_backends_use_the_existing_physical_bilinear_cache() {
        assert_eq!(
            DsaSparseMlaAttentionSpec::cache_kind(TORCH_BACKEND),
            CacheKind::Cache2DLinear(Extrapolation::Weighted)
        );
        assert_eq!(
            DsaSparseMlaAttentionSpec::cache_kind(FLASHMLA_BACKEND),
            CacheKind::Cache2DLinear(Extrapolation::Weighted)
        );
    }

    #[test]
    fn enumeration_emits_backend_plus_the_exact_python_schema() {
        for (pattern, expected_payloads) in [
            (ValidCountsPattern::UniformFull, 810),
            (ValidCountsPattern::CausalTail, 1_517),
            (ValidCountsPattern::SpeculativeGroups { group_size: 2 }, 620),
        ] {
            let cfg = config(pattern);
            let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
            let payloads = DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, FLASHMLA_BACKEND);

            assert_eq!(payloads.len(), expected_payloads);
            let expected_names = [
                "backend",
                "cache_dtype",
                "cache_layout",
                "index_distribution",
                "index_dtype",
                "latent_dim",
                "num_cache_tokens",
                "num_heads",
                "num_kv_heads",
                "num_queries",
                "output_dtype",
                "q_dtype",
                "rope_dim",
                "selected_k",
                "softmax_scale",
                "valid_counts",
                "value_dim",
            ];
            for payload in &payloads {
                let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
                assert_eq!(names, expected_names);
                assert_eq!(payload.fields().len(), 17);
            }

            let first_q = grid.axes()[0][0] as u32;
            let first_s = grid.axes()[1][0] as u32;
            assert_payload(
                &payloads[0],
                first_q,
                first_s,
                canonical_valid_counts(pattern, first_q, first_s, 2048),
            );
            assert_payload(
                payload_for(&payloads, &grid, 32, 2048),
                32,
                2048,
                canonical_valid_counts(pattern, 32, 2048, 2048),
            );
            assert_payload(
                payload_for(&payloads, &grid, 128, 2049),
                128,
                2049,
                canonical_valid_counts(pattern, 128, 2049, 2048),
            );
            let last_q = *grid.axes()[0].last().unwrap() as u32;
            let last_s = *grid.axes()[1].last().unwrap() as u32;
            assert_payload(
                payloads.last().unwrap(),
                last_q,
                last_s,
                canonical_valid_counts(pattern, last_q, last_s, 2048),
            );
        }

        assert_critical_payloads(
            ValidCountsPattern::UniformFull,
            &[(132, 6), (132, 11), (132, 91), (131, 8192), (133, 8192)],
        );
        assert_critical_payloads(
            ValidCountsPattern::CausalTail,
            &[
                (3, 3),
                (132, 182),
                (32, 511),
                (32, 513),
                (255, 724),
                (130, 130),
                (254, 255),
                (258, 260),
                (4096, 1),
            ],
        );
        assert_critical_payloads(
            ValidCountsPattern::SpeculativeGroups { group_size: 2 },
            &[
                (32, 3),
                (132, 23),
                (132, 45),
                (132, 182),
                (134, 65536),
                (266, 65536),
                (4096, 131072),
            ],
        );
    }

    #[test]
    fn remediated_pattern_union_matches_the_frozen_next_phase_inventory() {
        let distributions = [
            "recent_contiguous",
            "unique_scattered_pages",
            "clustered_pages",
            "uniform_stride",
        ];
        let mut all_distributions = BTreeSet::new();
        let mut memberships = 0;

        for distribution in distributions {
            let uniform = feasible_payload_keys(ValidCountsPattern::UniformFull, distribution);
            let causal = feasible_payload_keys(ValidCountsPattern::CausalTail, distribution);
            let speculative = feasible_payload_keys(
                ValidCountsPattern::SpeculativeGroups { group_size: 2 },
                distribution,
            );

            assert_eq!(uniform.len(), 810);
            assert_eq!(causal.len(), 858);
            assert_eq!(speculative.len(), 620);
            assert_eq!(uniform.intersection(&causal).count(), 249);
            assert_eq!(uniform.intersection(&speculative).count(), 180);
            assert_eq!(causal.intersection(&speculative).count(), 168);
            assert_eq!(
                uniform
                    .intersection(&causal)
                    .filter(|payload| speculative.contains(*payload))
                    .count(),
                148
            );

            memberships += uniform.len() + causal.len() + speculative.len();
            let per_distribution: BTreeSet<_> = uniform
                .into_iter()
                .chain(causal)
                .chain(speculative)
                .collect();
            assert_eq!(per_distribution.len(), 1_839);
            all_distributions.extend(per_distribution);
        }

        assert_eq!(memberships, 9_152);
        assert_eq!(all_distributions.len(), 7_356);
    }

    fn rich_dim(name: &str, value: u32) -> Value {
        serde_json::json!({
            "value": value,
            "expression": name,
            "bindings": {name: value},
        })
    }

    fn assert_grid(
        pattern: ValidCountsPattern,
        query_axis: &[u32],
        cache_axis: &[u32],
        cells: usize,
    ) {
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&config(pattern));
        let axes = grid.axes();

        assert_eq!(axes.len(), 2);
        assert_eq!(axes[0], axis_values(query_axis));
        assert_eq!(axes[1], axis_values(cache_axis));
        assert!(axes[0].windows(2).all(|pair| pair[0] < pair[1]));
        assert!(axes[1].windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(axes[0].first().copied(), Some(f64::from(query_axis[0])));
        assert_eq!(
            axes[0].last().copied(),
            Some(f64::from(*query_axis.last().unwrap()))
        );
        assert_eq!(axes[1].first().copied(), Some(f64::from(cache_axis[0])));
        assert_eq!(
            axes[1].last().copied(),
            Some(f64::from(*cache_axis.last().unwrap()))
        );
        assert_eq!(axes[0].len() * axes[1].len(), cells);
    }

    fn axis_values(axis: &[u32]) -> Vec<f64> {
        axis.iter().map(|&value| f64::from(value)).collect()
    }

    fn assert_contiguous(axis: &[f64], expected: &[u32]) {
        let expected = axis_values(expected);
        assert!(
            axis.windows(expected.len())
                .any(|window| window == expected.as_slice()),
            "axis {axis:?} does not contain contiguous sequence {expected:?}"
        );
    }

    fn assert_critical_payloads(pattern: ValidCountsPattern, coordinates: &[(u32, u32)]) {
        let cfg = config(pattern);
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        let payloads = DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, FLASHMLA_BACKEND);

        for &(q, s) in coordinates {
            assert_payload(
                payload_for(&payloads, &grid, q, s),
                q,
                s,
                canonical_valid_counts(pattern, q, s, 2048),
            );
        }
    }

    fn feasible_payload_keys(pattern: ValidCountsPattern, distribution: &str) -> BTreeSet<String> {
        let mut cfg = config(pattern);
        cfg.index_distribution = distribution.to_string();
        let grid = DsaSparseMlaAttentionSpec::sweep_grid(&cfg);
        let mask = DsaSparseMlaAttentionSpec::infeasible_mask(&cfg, &grid);
        DsaSparseMlaAttentionSpec::enumerate(&cfg, &grid, FLASHMLA_BACKEND)
            .into_iter()
            .zip(mask)
            .filter_map(|(payload, masked)| {
                (!masked).then(|| serde_json::to_string(payload.fields()).unwrap())
            })
            .collect()
    }

    fn mask_for(pattern: ValidCountsPattern, grid: &crate::timing::sweep::SweepGrid) -> Vec<bool> {
        DsaSparseMlaAttentionSpec::infeasible_mask(&config(pattern), grid)
    }

    fn assert_mask_split(mask: &[bool], feasible: usize, masked: usize) {
        assert_eq!(mask.len(), feasible + masked);
        assert_eq!(mask.iter().filter(|&&value| !value).count(), feasible);
        assert_eq!(mask.iter().filter(|&&value| value).count(), masked);
    }

    fn masked(mask: &[bool], grid: &crate::timing::sweep::SweepGrid, q: u32, s: u32) -> bool {
        let q_index = grid.axes()[0]
            .iter()
            .position(|&value| value == f64::from(q))
            .unwrap();
        let s_index = grid.axes()[1]
            .iter()
            .position(|&value| value == f64::from(s))
            .unwrap();
        mask[q_index * grid.axes()[1].len() + s_index]
    }

    fn payload_for<'a>(
        payloads: &'a [ArgsPayload],
        grid: &crate::timing::sweep::SweepGrid,
        q: u32,
        s: u32,
    ) -> &'a ArgsPayload {
        let q_index = grid.axes()[0]
            .iter()
            .position(|&value| value == f64::from(q))
            .unwrap();
        let s_index = grid.axes()[1]
            .iter()
            .position(|&value| value == f64::from(s))
            .unwrap();
        &payloads[q_index * grid.axes()[1].len() + s_index]
    }

    fn assert_payload(payload: &ArgsPayload, q: u32, s: u32, valid_counts: String) {
        let fields = payload.fields();
        assert_eq!(fields.get("backend"), Some(&Value::from(FLASHMLA_BACKEND)));
        assert_eq!(fields.get("num_queries"), Some(&Value::from(q)));
        assert_eq!(fields.get("num_cache_tokens"), Some(&Value::from(s)));
        assert_eq!(fields.get("num_heads"), Some(&Value::from(64_u32)));
        assert_eq!(fields.get("num_kv_heads"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("selected_k"), Some(&Value::from(2048_u32)));
        assert_eq!(fields.get("latent_dim"), Some(&Value::from(512_u32)));
        assert_eq!(fields.get("rope_dim"), Some(&Value::from(64_u32)));
        assert_eq!(fields.get("value_dim"), Some(&Value::from(512_u32)));
        assert_eq!(fields.get("softmax_scale"), Some(&Value::from(0.0625)));
        assert_eq!(fields.get("q_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("cache_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("index_dtype"), Some(&Value::from("int32")));
        assert_eq!(fields.get("output_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("valid_counts"), Some(&Value::from(valid_counts)));
        assert_eq!(
            fields.get("index_distribution"),
            Some(&Value::from("recent_contiguous"))
        );
        assert_eq!(
            fields.get("cache_layout"),
            Some(&Value::from("token_major_mqa_bf16_latent_rope"))
        );
        assert_eq!(payload.backend(), Some(FLASHMLA_BACKEND));
    }
}
