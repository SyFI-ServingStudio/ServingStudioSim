//! One decoder layer of the decode forward, per TP4 rank, BF16, with both
//! block all-reduces. It is the marginal `(t(32) - t(8)) / 24` of the engine's
//! decode forward, run at reduced depth on the chip's four ranks
//! (`profiling/runners/neuron/forward_bench.py`). The engine's decode step is
//! one compiled graph per bucket on four ranks. Its units overlap and fuse,
//! and its collectives run inside it. The standalone rows of the same ops
//! therefore do not add up to the decode step (12-17% high on the kernel path;
//! +24% at B1 and -23% at B16 on the torch fallback, C512). With
//! `neuron_decode_fixed` the step is `fixed + num_layers x layer`, 2-5% below
//! the engine's measured decode step at every bucket from B1 to B128 (C512,
//! kernel path; doc/pr/llama31-8b-trn2/README.md, "Check 1"). The per-op rows remain
//! for attribution.
//!
//! One backend per attention path, like `neuron_attention_decode`:
//! `vllm_neuron_nki` (the per-head block-table NKI kernel, a patched Llama3
//! `forward_decode`) and `vllm_neuron_torch` (the stock call, the torch
//! fallback at TP4).
//!
//! Decode only; the context's KV pool caps the buckets. The row is raw device
//! time: the per-graph fixed cost is in `neuron_decode_fixed`.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_nki::{
    self, NeuronNkiKernelConfig, NeuronNkiKernelInput, NkiPhases, NEURON_TORCH_BACKEND, NKI_BACKEND,
};
use crate::timing::sweep::SweepGrid;
use crate::timing::Coords;

pub struct NeuronDecoderLayerSpec;

impl KernelSpec for NeuronDecoderLayerSpec {
    type Config = NeuronNkiKernelConfig;
    type Input = NeuronNkiKernelInput;

    const KIND: KernelKind = "neuron_decoder_layer";

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

register_kernel!(NeuronDecoderLayerKernel, NeuronDecoderLayerSpec);

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
        assert_rows::<NeuronDecoderLayerSpec>(&test_config(vec![], vec![1, 16]), &expected);
        let mut torch = test_config(vec![], vec![1, 16]);
        torch.backends = vec![NEURON_TORCH_BACKEND];
        NeuronDecoderLayerSpec::validate_config(&torch).unwrap();
        let mut prefill = test_config(vec![512], vec![1]);
        prefill.backends = vec![NKI_BACKEND];
        assert!(NeuronDecoderLayerSpec::validate_config(&prefill).is_err());
    }
}
