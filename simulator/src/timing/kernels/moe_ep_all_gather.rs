//! Ragged hidden-state plus router-logit all-gatherv for naive DP/EP MoE.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct MoeEpCollectiveKernelInput {
    pub per_rank_tokens: Vec<u32>,
}

/// vLLM runs a different NCCL call when every rank carries the same count:
/// `all_gather`/`reduce_scatter` instead of one grouped broadcast/reduce per
/// root (`cuda_communicator.py` `all_gatherv`/`reduce_scatterv`). The second
/// coordinate keeps the two apart: the largest rank's count when the ranks
/// differ, and zero when they are equal, so an equal split reads only the zero
/// line and a ragged one never interpolates toward it.
impl SweepCoords for MoeEpCollectiveKernelInput {
    fn coords(&self) -> Coords {
        let total = self.per_rank_tokens.iter().copied().sum::<u32>();
        let maximum = self.per_rank_tokens.iter().copied().max().unwrap_or(0);
        let equal = self.per_rank_tokens.iter().all(|&tokens| tokens == maximum);
        let ragged_max = if equal { 0 } else { maximum };
        Coords::new([f64::from(total), f64::from(ragged_max)])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["total_tokens", "ragged_max_rank_tokens"]
    }
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeEpAllGatherKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_gpus: u32,
    pub hidden_size: Dim,
    pub num_experts: Dim,
    #[compute_dtype]
    pub hidden_dtype: DType,
    pub router_dtype: DType,
    pub fabric: String,
}

pub struct MoeEpAllGatherSpec;

impl KernelSpec for MoeEpAllGatherSpec {
    type Config = MoeEpAllGatherKernelConfig;
    type Input = MoeEpCollectiveKernelInput;

    const KIND: KernelKind = "moe_ep_all_gather";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        collective_grid(8192)
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_2d(|total, ragged_max| {
            canonical_tokens(config.num_gpus, total as u32, ragged_max as u32).is_none()
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        validate_config(config);
        grid.expand_2d(|total, ragged_max| {
            let per_rank_tokens =
                canonical_tokens(config.num_gpus, total as u32, ragged_max as u32)
                    .unwrap_or_else(|| vec![total as u32; config.num_gpus as usize]);
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_gpus", config.num_gpus)
                .with("per_rank_tokens", per_rank_tokens)
                .with("hidden_size", config.hidden_size.get())
                .with("num_experts", config.num_experts.get())
                .with("hidden_dtype", config.hidden_dtype.as_str())
                .with("router_dtype", config.router_dtype.as_str())
                .with("fabric", config.fabric.clone())
        })
    }
}

/// Points of the largest per-rank batch, the second grid axis; 0 is the
/// equal-split line.
const RANK_TOKEN_POINTS: [u32; 9] = [0, 1, 4, 16, 64, 256, 1024, 4096, 8192];
/// Total-token points; the ones below 16384 are also per-rank points.
const TOTAL_TOKEN_POINTS: [u32; 11] = [1, 4, 16, 64, 256, 1024, 4096, 8192, 16384, 32768, 65536];

/// The (total tokens, ragged largest rank tokens) grid shared by the naive
/// DP/EP MoE collectives. The total axis stops at `max_total_tokens`.
pub(crate) fn collective_grid(max_total_tokens: u32) -> SweepGrid {
    let totals = TOTAL_TOKEN_POINTS
        .into_iter()
        .filter(|&tokens| tokens <= max_total_tokens);
    SweepGrid::new(vec![Axis::values(totals), Axis::values(RANK_TOKEN_POINTS)])
}

/// The per-rank counts profiled at one grid cell, or `None` when the cell
/// names no such split. `ragged_max_rank_tokens == 0` is the equal split, which
/// needs `total_tokens` divisible by the group; otherwise rank 0 carries the
/// maximum and the rest is spread as evenly as it fits, and a cell whose only
/// such split is equal is left to the zero line.
pub(crate) fn canonical_tokens(
    num_gpus: u32,
    total_tokens: u32,
    ragged_max_rank_tokens: u32,
) -> Option<Vec<u32>> {
    if num_gpus < 2 || total_tokens == 0 {
        return None;
    }
    if ragged_max_rank_tokens == 0 {
        return (total_tokens % num_gpus == 0)
            .then(|| vec![total_tokens / num_gpus; num_gpus as usize]);
    }
    let max_rank_tokens = ragged_max_rank_tokens;
    if max_rank_tokens > total_tokens || total_tokens >= max_rank_tokens.checked_mul(num_gpus)? {
        return None;
    }
    let mut tokens = vec![0; num_gpus as usize];
    tokens[0] = max_rank_tokens;
    let mut remaining = total_tokens - max_rank_tokens;
    for rank in 1..num_gpus as usize {
        let ranks_left = num_gpus as usize - rank;
        let count = remaining.div_ceil(ranks_left as u32).min(max_rank_tokens);
        tokens[rank] = count;
        remaining -= count;
    }
    (remaining == 0).then_some(tokens)
}

fn validate_config(config: &MoeEpAllGatherKernelConfig) {
    assert!(matches!(config.num_gpus, 2 | 4 | 8));
    assert_eq!(config.hidden_dtype, DType::Bf16);
    assert_eq!(config.router_dtype, DType::Fp32);
    assert_eq!(config.fabric, "nvlink");
}

register_kernel!(MoeEpAllGatherKernel, MoeEpAllGatherSpec);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_splits_read_the_zero_line_and_ragged_ones_their_maximum() {
        let ragged = MoeEpCollectiveKernelInput {
            per_rank_tokens: vec![128, 96, 0, 32],
        };
        assert_eq!(&*ragged.coords(), &[256.0, 128.0]);
        let nearly_equal = MoeEpCollectiveKernelInput {
            per_rank_tokens: vec![2, 1, 1, 1, 1, 1, 1, 1],
        };
        assert_eq!(&*nearly_equal.coords(), &[9.0, 2.0]);
        let equal = MoeEpCollectiveKernelInput {
            per_rank_tokens: vec![4; 8],
        };
        assert_eq!(&*equal.coords(), &[32.0, 0.0]);
    }

    #[test]
    fn canonical_tokens_profile_an_equal_split_only_on_the_zero_line() {
        assert_eq!(canonical_tokens(4, 256, 128), Some(vec![128, 43, 43, 42]));
        assert_eq!(canonical_tokens(4, 256, 32), None);
        // The ragged cell whose only split is equal belongs to the zero line.
        assert_eq!(canonical_tokens(4, 256, 64), None);
        assert_eq!(canonical_tokens(4, 256, 0), Some(vec![64; 4]));
        assert_eq!(canonical_tokens(8, 16, 0), Some(vec![2; 8]));
        assert_eq!(canonical_tokens(8, 4, 0), None);
    }

    #[test]
    fn an_equal_split_is_priced_from_equal_split_rows_only() {
        use crate::timing::bridge::KernelMetrics;
        use crate::timing::cache::{Cache, Cache2DLinear};
        let timed = |time_ms: f64| KernelMetrics {
            time_ms,
            tflops: None,
            memory_bandwidth_gbps: None,
            algbw_gbps: None,
            busbw_gbps: None,
            energy_j: 0.0,
        };
        // 8 ranks: the ragged rows cost 200, the equal-split rows 20.
        let grid = collective_grid(8192);
        let (totals, maxima) = (&grid.axes()[0], &grid.axes()[1]);
        let samples: Vec<KernelMetrics> = totals
            .iter()
            .flat_map(|&total| maxima.iter().map(move |&ragged_max| (total, ragged_max)))
            .map(
                |(total, ragged_max)| match canonical_tokens(8, total as u32, ragged_max as u32) {
                    None => KernelMetrics::non_finite(),
                    Some(_) if ragged_max == 0.0 => timed(0.020),
                    Some(_) => timed(0.200),
                },
            )
            .collect();
        let (cache, _) = Cache2DLinear::from_samples_with(&grid, &samples, Extrapolation::Weighted);
        let time_us = |per_rank_tokens: Vec<u32>| {
            let coords = MoeEpCollectiveKernelInput { per_rank_tokens }.coords();
            f64::from(cache.eval(&coords).m.time_ms) * 1e3
        };
        for tokens in [1, 2, 4, 8, 32, 88] {
            assert!(
                (time_us(vec![tokens; 8]) - 20.0).abs() < 1e-3,
                "{tokens} per rank"
            );
        }
        assert!((time_us(vec![2, 1, 1, 1, 1, 1, 1, 1]) - 200.0).abs() < 1e-3);
        assert!((time_us(vec![88, 87, 87, 87, 87, 87, 87, 87]) - 200.0).abs() < 1e-3);
    }

    #[test]
    fn collective_grid_extends_totals_only_to_the_requested_bound() {
        let small = collective_grid(8192);
        assert_eq!(small.axes()[0].last(), Some(&8192.0));
        let large = collective_grid(65536);
        assert_eq!(large.axes()[0].last(), Some(&65536.0));
        assert_eq!(large.axes()[1].first(), Some(&0.0));
        assert_eq!(large.axes()[1].last(), Some(&8192.0));
    }
}
