//! DeepSeek V4 indexer decode persistent top-k kernel.
//!
//! The public callable is vLLM's stable-libtorch `persistent_topk` at k=512.
//! Its static schema matches the older DSA kind, but its specialization and
//! max-ragged workload are separate profile evidence.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::dsa_persistent_topk_decode::{
    DsaPersistentTopkDecodeKernelConfig, DsaPersistentTopkDecodeKernelInput,
    DsaPersistentTopkDecodeSpec,
};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};

const BATCH_AXIS: [u32; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
const CONTEXT_AXIS: [u32; 21] = [
    0, 1, 128, 511, 512, 513, 1024, 2048, 4096, 8191, 8192, 8193, 16384, 32767, 32768, 32769,
    65536, 131072, 262144, 524288, 1048576,
];

pub type DeepseekV4IndexerTopkDecodeKernelConfig = DsaPersistentTopkDecodeKernelConfig;
pub type DeepseekV4IndexerTopkDecodeKernelInput = DsaPersistentTopkDecodeKernelInput;

pub struct DeepseekV4IndexerTopkDecodeSpec;

impl KernelSpec for DeepseekV4IndexerTopkDecodeSpec {
    type Config = DeepseekV4IndexerTopkDecodeKernelConfig;
    type Input = DeepseekV4IndexerTopkDecodeKernelInput;

    const KIND: KernelKind = "deepseek_v4_indexer_topk_decode";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::values(BATCH_AXIS), Axis::values(CONTEXT_AXIS)])
    }

    fn cache_kind(backend: &'static str) -> CacheKind {
        DsaPersistentTopkDecodeSpec::cache_kind(backend)
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        DsaPersistentTopkDecodeSpec::infeasible_mask(config, grid)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        DsaPersistentTopkDecodeSpec::enumerate(config, grid, backend)
    }
}

register_kernel!(
    DeepseekV4IndexerTopkDecodeKernel,
    DeepseekV4IndexerTopkDecodeSpec
);

#[cfg(test)]
mod tests {
    use super::{DeepseekV4IndexerTopkDecodeSpec, BATCH_AXIS, CONTEXT_AXIS};
    use crate::timing::kernels::dsa_persistent_topk_decode::DsaPersistentTopkDecodeSpec;
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn keeps_deepseek_k512_evidence_separate_and_brackets_its_cliffs() {
        assert_ne!(
            DeepseekV4IndexerTopkDecodeSpec::KIND,
            DsaPersistentTopkDecodeSpec::KIND
        );
        assert_eq!(BATCH_AXIS, [1, 2, 4, 8, 16, 32, 64, 128, 256]);
        assert!(CONTEXT_AXIS
            .windows(3)
            .any(|values| values == [511, 512, 513]));
        assert!(CONTEXT_AXIS
            .windows(3)
            .any(|values| values == [32767, 32768, 32769]));
    }
}
