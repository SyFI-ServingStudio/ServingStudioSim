//! Standalone NKI token-generation attention block of one decode layer:
//! `nkilib` `attention_block_tkg` through `vllm_neuron.functional.attention_decode`
//! with a per-head block table. The unit fuses QKV + RoPE, the paged KV gather
//! over the context, attention and the output projection for B requests, per
//! TP4 rank, BF16. It is timed alone on one LNC2 logical core; the row is raw
//! device time.
//!
//! The stock Llama3 decode never reaches this kernel at TP4: its 2-D block
//! table routes GQA to the torch fallback. This leaf models the per-head block
//! table path. Decode only; the context's KV pool caps the buckets.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_nki::{
    self, NeuronNkiKernelConfig, NeuronNkiKernelInput, NkiPhases,
};
use crate::timing::sweep::SweepGrid;
use crate::timing::Coords;

pub struct NeuronNkiAttentionTkgSpec;

impl KernelSpec for NeuronNkiAttentionTkgSpec {
    type Config = NeuronNkiKernelConfig;
    type Input = NeuronNkiKernelInput;

    const KIND: KernelKind = "neuron_nki_attention_tkg";

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        neuron_nki::validate_config(config, NkiPhases::DecodeOnly)
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
        neuron_nki::enumerate(config, grid, backend, |row, _phase, bucket| {
            neuron_nki::identity_fields(
                row.with("token_bucket", bucket)
                    .with("max_model_len", config.max_model_len.get())
                    .with("kv_blocks", config.kv_blocks.get())
                    .with("block_size", config.block_size.get()),
                config,
            )
        })
    }
}

register_kernel!(NeuronNkiAttentionTkgKernel, NeuronNkiAttentionTkgSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::neuron_nki::{assert_rows, test_config, NKI_BACKEND};

    #[test]
    fn rows_are_the_python_fields_in_inventory_order() {
        let expected: Vec<_> = [("decode", 1), ("decode", 16)]
            .into_iter()
            .map(|(_phase, bucket)| {
                serde_json::json!({
                    "backend": NKI_BACKEND, "token_bucket": bucket, "max_model_len": 512,
                    "kv_blocks": 6782, "block_size": 32, "tp_size": 4, "dtype": "bf16"
                })
            })
            .collect();
        assert_rows::<NeuronNkiAttentionTkgSpec>(&test_config(vec![], vec![1, 16]), &expected);
    }
}
