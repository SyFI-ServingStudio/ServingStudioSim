//! DeepSeek V4 indexer decode MQA-logits kernel.
//!
//! DeepSeek and GLM share the same physical `(batch, context)` sweep and cache
//! interpolation, but not the profiler identity: DeepSeek uses a max-ragged,
//! request-contiguous, page-planar FP8 cache.  Keep a distinct kind/table while
//! delegating only the generic grid machinery to the established DSA spec.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::dsa_paged_mqa_logits_decode::{
    DsaPagedMqaLogitsDecodeKernelConfig, DsaPagedMqaLogitsDecodeKernelInput,
    DsaPagedMqaLogitsDecodeSpec,
};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::SweepGrid;

pub type DeepseekV4IndexerMqaLogitsDecodeKernelConfig = DsaPagedMqaLogitsDecodeKernelConfig;
pub type DeepseekV4IndexerMqaLogitsDecodeKernelInput = DsaPagedMqaLogitsDecodeKernelInput;

pub struct DeepseekV4IndexerMqaLogitsDecodeSpec;

impl KernelSpec for DeepseekV4IndexerMqaLogitsDecodeSpec {
    type Config = DeepseekV4IndexerMqaLogitsDecodeKernelConfig;
    type Input = DeepseekV4IndexerMqaLogitsDecodeKernelInput;

    const KIND: KernelKind = "deepseek_v4_indexer_mqa_logits_decode";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        DsaPagedMqaLogitsDecodeSpec::sweep_grid(config)
    }

    fn cache_kind(backend: &'static str) -> CacheKind {
        DsaPagedMqaLogitsDecodeSpec::cache_kind(backend)
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        DsaPagedMqaLogitsDecodeSpec::infeasible_mask(config, grid)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        DsaPagedMqaLogitsDecodeSpec::enumerate(config, grid, backend)
    }
}

register_kernel!(
    DeepseekV4IndexerMqaLogitsDecodeKernel,
    DeepseekV4IndexerMqaLogitsDecodeSpec
);

#[cfg(test)]
mod tests {
    use super::DeepseekV4IndexerMqaLogitsDecodeSpec;
    use crate::timing::kernels::dsa_paged_mqa_logits_decode::DsaPagedMqaLogitsDecodeSpec;
    use crate::timing::kernels::engine::KernelSpec;

    #[test]
    fn keeps_a_distinct_profile_identity_from_the_glm_layout() {
        assert_eq!(
            DeepseekV4IndexerMqaLogitsDecodeSpec::KIND,
            "deepseek_v4_indexer_mqa_logits_decode"
        );
        assert_ne!(
            DeepseekV4IndexerMqaLogitsDecodeSpec::KIND,
            DsaPagedMqaLogitsDecodeSpec::KIND
        );
    }
}
