//! The per-step rest of the decode forward beside its layers, per TP4 rank,
//! BF16: the vocab-sharded embedding and its all-reduce, the RoPE tables, the
//! final norm, the lm_head shard, the engine's on-device sampler across the
//! shards and the graph's own fixed cost. It is the intercept
//! `t(8) - 8 x layer` of the same reduced-depth TP4 forward that gives
//! `neuron_decoder_layer` (`profiling/runners/neuron/forward_bench.py`); the
//! decode step is `fixed + num_layers x layer`. The sampler runs the engine's
//! distributed top-k path (default on-device sampling config), so the term
//! grows with the batch bucket.
//!
//! Same backends, inventory and row fields as `neuron_decoder_layer`.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_nki::{
    self, NeuronNkiKernelConfig, NeuronNkiKernelInput, NkiPhases, NEURON_TORCH_BACKEND, NKI_BACKEND,
};
use crate::timing::sweep::SweepGrid;
use crate::timing::Coords;

pub struct NeuronDecodeFixedSpec;

impl KernelSpec for NeuronDecodeFixedSpec {
    type Config = NeuronNkiKernelConfig;
    type Input = NeuronNkiKernelInput;

    const KIND: KernelKind = "neuron_decode_fixed";

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        neuron_nki::validate_leaf(
            config,
            NkiPhases::DecodeOnly,
            &[NKI_BACKEND, NEURON_TORCH_BACKEND],
        )
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

register_kernel!(NeuronDecodeFixedKernel, NeuronDecodeFixedSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::neuron_nki::{assert_rows, test_config, NKI_BACKEND};

    #[test]
    fn rows_are_the_python_fields_in_inventory_order() {
        let expected: Vec<_> = [1, 16]
            .into_iter()
            .map(|bucket| {
                serde_json::json!({
                    "backend": NKI_BACKEND, "token_bucket": bucket, "max_model_len": 512,
                    "kv_blocks": 6782, "block_size": 32, "tp_size": 4, "dtype": "bf16"
                })
            })
            .collect();
        assert_rows::<NeuronDecodeFixedSpec>(&test_config(vec![], vec![1, 16]), &expected);
        let mut torch = test_config(vec![], vec![1, 16]);
        torch.backends = vec![NEURON_TORCH_BACKEND];
        NeuronDecodeFixedSpec::validate_config(&torch).unwrap();
        let mut prefill = test_config(vec![512], vec![1]);
        prefill.backends = vec![NKI_BACKEND];
        assert!(NeuronDecodeFixedSpec::validate_config(&prefill).is_err());
    }
}
