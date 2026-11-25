//! vLLM ShardQ exchange around TP-sharded sparse-MLA prefill: `shard_q` moves
//! the query from head shards to zigzag token shards before attention and
//! `unshard_out` moves the attention output back. Each is one push kernel
//! into the peers' symmetric-memory buffers between two group barriers.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparseMlaShardQKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_gpus: u32,
    /// Heads one GPU holds under TP.
    pub num_heads: u32,
    /// Per-head width moved (query width for shard_q, value width for unshard_out).
    pub head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
    /// `shard_q` or `unshard_out`.
    pub direction: String,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct SparseMlaShardQKernelInput {
    /// Prefill tokens of the step (after the decodes), split zigzag over the group.
    pub num_tokens: u32,
}

pub struct SparseMlaShardQSpec;

impl KernelSpec for SparseMlaShardQSpec {
    type Config = SparseMlaShardQKernelConfig;
    type Input = SparseMlaShardQKernelInput;

    const KIND: KernelKind = "sparse_mla_shard_q";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::chain([
            Axis::pow2(5, 10),
            Axis::arithmetic(1024, 8192, 512),
            Axis::arithmetic(8192, 16384, 2048),
        ])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_1d(|tokens| (tokens as u32) < 2 * config.num_gpus)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        assert!(matches!(config.direction.as_str(), "shard_q" | "unshard_out"));
        grid.expand_1d(|num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_gpus", config.num_gpus)
                .with("num_tokens", num_tokens as u32)
                .with("num_heads", config.num_heads)
                .with("head_dim", config.head_dim.get())
                .with("dtype", config.dtype.as_str())
                .with("direction", config.direction.as_str())
        })
    }
}

register_kernel!(SparseMlaShardQKernel, SparseMlaShardQSpec);
