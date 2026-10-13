//! PLE n-gram hash ids: one launch of the fork's `_ple_ngram_ids_kernel`
//! (grid `ceil(tokens / 8)`). Each token binary-searches its request in
//! `query_start_loc`, so `num_reqs` adds a little work; both axes are kept.
//! The kernel is launch-latency bound (3-5 us across 1..8192 tokens on B200).

use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PleNgramIdsKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Predecessor tokens per request (`ngram_size - 1`).
    pub ngram_context_len: u32,
    pub heads_per_ngram: u32,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PleNgramIdsKernelInput {
    pub num_tokens: u32,
    pub num_reqs: u32,
}

pub struct PleNgramIdsSpec;

impl KernelSpec for PleNgramIdsSpec {
    type Config = PleNgramIdsKernelConfig;
    type Input = PleNgramIdsKernelInput;
    const KIND: KernelKind = "ple_ngram_ids";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert!(
            config.ngram_context_len > 0 && config.heads_per_ngram > 0,
            "ple_ngram_ids needs a positive n-gram geometry"
        );
        SweepGrid::new(vec![
            Axis::values([1, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768]),
            Axis::values([1, 16, 64, 256, 1024]),
        ])
    }

    fn cache_kind(_: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    /// A request has at least one token.
    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_2d(|tokens, reqs| reqs > tokens)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|tokens, reqs| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", tokens.round() as u32)
                .with("num_reqs", reqs.round() as u32)
                .with("ngram_context_len", config.ngram_context_len)
                .with("heads_per_ngram", config.heads_per_ngram)
        })
    }
}

register_kernel!(PleNgramIdsKernel, PleNgramIdsSpec);

#[cfg(test)]
mod tests {
    use super::*;

    /// Catches a grid cell with more requests than tokens (the runner rejects
    /// it) and a payload field the Python schema would reject.
    #[test]
    fn mask_drops_requests_without_tokens_and_payload_matches_python_args() {
        let cfg = PleNgramIdsKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            ngram_context_len: 2,
            heads_per_ngram: 8,
        };
        let grid = PleNgramIdsSpec::sweep_grid(&cfg);
        let mask = PleNgramIdsSpec::infeasible_mask(&cfg, &grid);
        let payloads = PleNgramIdsSpec::enumerate(&cfg, &grid, "vllm_triton");
        for (payload, masked) in payloads.iter().zip(&mask) {
            let f = payload.fields();
            let (t, r) = (
                f["num_tokens"].as_u64().unwrap(),
                f["num_reqs"].as_u64().unwrap(),
            );
            assert_eq!(*masked, r > t);
        }
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "heads_per_ngram",
                "ngram_context_len",
                "num_reqs",
                "num_tokens"
            ]
        );
    }
}
