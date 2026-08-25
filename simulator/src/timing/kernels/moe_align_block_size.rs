//! vLLM MoE token-to-expert block alignment.
//!
//! `block_size` is a runtime coordinate: Qwen uses 16, while Marlin selects
//! 8/16/32/48/64 from the current routed-token shape.

use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeAlignBlockSizeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_experts: Dim,
    pub top_k: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MoeAlignBlockSizeKernelInput {
    pub num_tokens: u32,
    pub block_size: u32,
}

pub struct MoeAlignBlockSizeSpec;

impl KernelSpec for MoeAlignBlockSizeSpec {
    type Config = MoeAlignBlockSizeKernelConfig;
    type Input = MoeAlignBlockSizeKernelInput;

    const KIND: KernelKind = "moe_align_block_size";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // T<9 bypasses alignment in vLLM's real expert-assignment path. T31 and
        // T33 resolve the measured launch-count cliff around the T32 boundary;
        // the remaining landmarks follow powers of two through the context cap.
        SweepGrid::new(vec![
            Axis::values([
                9, 16, 31, 32, 33, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                131072, 262144,
            ]),
            Axis::values([8, 16, 32, 48, 64]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Clamp)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|num_tokens, block_size| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
                .with("num_experts", config.num_experts.get())
                .with("top_k", config.top_k)
                .with("block_size", block_size as u32)
        })
    }
}

register_kernel!(MoeAlignBlockSizeKernel, MoeAlignBlockSizeSpec);

#[cfg(test)]
mod tests {
    use super::{
        MoeAlignBlockSizeKernelConfig, MoeAlignBlockSizeKernelInput, MoeAlignBlockSizeSpec,
    };
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::KernelSpec;
    use crate::timing::SweepCoords;

    fn config() -> MoeAlignBlockSizeKernelConfig {
        MoeAlignBlockSizeKernelConfig {
            backends: vec!["vllm_cuda"],
            gpu_name: "NVIDIA H200".to_string(),
            num_experts: 256.into(),
            top_k: 8,
        }
    }

    #[test]
    fn runtime_block_size_produces_distinct_cache_cells_and_profile_rows() {
        let cfg = config();
        let grid = MoeAlignBlockSizeSpec::sweep_grid(&cfg);
        let payloads = MoeAlignBlockSizeSpec::enumerate(&cfg, &grid, "vllm_cuda");
        assert_eq!(grid.axes()[1], [8.0, 16.0, 32.0, 48.0, 64.0]);
        assert_eq!(payloads.len(), 18 * 5);
        assert_eq!(
            MoeAlignBlockSizeSpec::cache_kind("vllm_cuda"),
            CacheKind::Cache2DLinear(Extrapolation::Clamp)
        );

        let input_8 = MoeAlignBlockSizeKernelInput {
            num_tokens: 128,
            block_size: 8,
        };
        let input_64 = MoeAlignBlockSizeKernelInput {
            num_tokens: 128,
            block_size: 64,
        };
        assert_eq!(&*input_8.coords(), &[128.0, 8.0]);
        assert_eq!(&*input_64.coords(), &[128.0, 64.0]);
        assert_ne!(input_8.coords().as_slice(), input_64.coords().as_slice());
    }
}
