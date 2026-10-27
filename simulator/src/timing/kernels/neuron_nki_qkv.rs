//! Standalone NKI fused QKV projection with RoPE of one prefill layer:
//! `nkilib.core.qkv` through `vllm_neuron.functional.qkv_proj`, per TP4 rank,
//! hidden `[1, T, 4096] @ W[4096, 1536]` in BF16. The unit is timed alone on
//! one LNC2 logical core; the row is raw device time (see `neuron_nki`).
//!
//! Prefill only: on the per-head block table decode path the projection is
//! fused into `neuron_attention_decode`.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_nki::{
    self, NeuronNkiKernelConfig, NeuronNkiKernelInput, NkiPhases,
};
use crate::timing::sweep::SweepGrid;
use crate::timing::Coords;

pub struct NeuronNkiQkvSpec;

impl KernelSpec for NeuronNkiQkvSpec {
    type Config = NeuronNkiKernelConfig;
    type Input = NeuronNkiKernelInput;

    const KIND: KernelKind = "neuron_nki_qkv";

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        neuron_nki::validate_config(config, NkiPhases::PrefillOnly)
    }

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        neuron_nki::sweep_grid(config)
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DDirect
    }

    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        neuron_nki::cache_coords(config, input)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        neuron_nki::enumerate_prefill_only(config, grid, backend)
    }
}

register_kernel!(NeuronNkiQkvKernel, NeuronNkiQkvSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::neuron_nki::{assert_rows, test_config, NKI_BACKEND};

    #[test]
    fn rows_are_the_python_fields_in_inventory_order() {
        let expected: Vec<_> = [("prefill", 128), ("prefill", 512)]
            .into_iter()
            .map(|(_phase, bucket)| {
                serde_json::json!({
                    "backend": NKI_BACKEND, "token_bucket": bucket,
                    "tp_size": 4, "dtype": "bf16"
                })
            })
            .collect();
        assert_rows::<NeuronNkiQkvSpec>(&test_config(vec![128, 512], vec![]), &expected);
    }
}
