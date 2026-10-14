//! Sparse-indexer prefill scoring: each new token of one prefilling sequence
//! scores the compressed keys visible to it (causal over the prefix plus the
//! chunk), one launch per row piece the framework's logits budget allows
//! (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`, 512 MiB by default). The top-k that
//! follows is its own kind.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SparseIndexerLogitsPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub compress_ratio: u32,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, Debug, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct SparseIndexerLogitsPrefillKernelInput {
    /// New tokens of the sequence in this step.
    pub num_queries: u32,
    /// Tokens of the sequence already cached.
    pub prefix_len: u32,
}

const QUERIES_AXIS: [u32; 10] = [1, 16, 64, 256, 512, 1024, 2048, 4096, 8192, 16384];
const PREFIX_AXIS: [u32; 8] = [0, 1024, 4096, 16384, 32768, 65536, 131072, 262144];

pub struct SparseIndexerLogitsPrefillSpec;

impl KernelSpec for SparseIndexerLogitsPrefillSpec {
    type Config = SparseIndexerLogitsPrefillKernelConfig;
    type Input = SparseIndexerLogitsPrefillKernelInput;

    const KIND: KernelKind = "sparse_indexer_logits_prefill";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::values(QUERIES_AXIS), Axis::values(PREFIX_AXIS)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    /// A chunk past the grid runs as consecutive full-edge chunks, each seeing
    /// the tokens before it as prefix.
    fn off_grid(
        _config: &Self::Config,
        input: &Self::Input,
        _backend: &'static str,
    ) -> OffGrid<Self::Input> {
        let max_queries = QUERIES_AXIS[QUERIES_AXIS.len() - 1];
        if input.num_queries <= max_queries {
            return OffGrid::Cache;
        }
        let mut pieces = Vec::new();
        let mut done = 0;
        while done < input.num_queries {
            let piece = (input.num_queries - done).min(max_queries);
            pieces.push(SparseIndexerLogitsPrefillKernelInput {
                num_queries: piece,
                prefix_len: input.prefix_len + done,
            });
            done += piece;
        }
        OffGrid::Launches(pieces)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|num_queries, prefix_len| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_queries", num_queries as u32)
                .with("prefix_len", prefix_len as u32)
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("compress_ratio", config.compress_ratio)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(
    SparseIndexerLogitsPrefillKernel,
    SparseIndexerLogitsPrefillSpec
);

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SparseIndexerLogitsPrefillKernelConfig {
        SparseIndexerLogitsPrefillKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".into(),
            num_heads: Dim::param("num_heads", 4),
            head_dim: Dim::param("head_dim", 128),
            compress_ratio: 4,
            dtype: DType::Fp8E4m3,
        }
    }

    #[test]
    fn enumerate_emits_the_python_args() {
        let cfg = config();
        let grid = SparseIndexerLogitsPrefillSpec::sweep_grid(&cfg);
        let specs = SparseIndexerLogitsPrefillSpec::enumerate(&cfg, &grid, "vllm_triton");
        assert_eq!(specs.len(), 80);
        let names: Vec<&str> = specs[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            names,
            [
                "backend",
                "compress_ratio",
                "dtype",
                "head_dim",
                "num_heads",
                "num_queries",
                "prefix_len"
            ]
        );
    }

    #[test]
    fn long_chunks_split_with_growing_prefix() {
        let input = SparseIndexerLogitsPrefillKernelInput {
            num_queries: 20000,
            prefix_len: 100,
        };
        match SparseIndexerLogitsPrefillSpec::off_grid(&config(), &input, "vllm_triton") {
            OffGrid::Launches(pieces) => {
                let got: Vec<_> = pieces
                    .iter()
                    .map(|p| (p.num_queries, p.prefix_len))
                    .collect();
                assert_eq!(got, [(16384, 100), (3616, 16484)]);
            }
            _ => panic!("expected launches"),
        }
    }
}
