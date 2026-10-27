//! The decode attention block of one layer, `vllm_neuron.functional.attention_decode`
//! (fused QKV + RoPE, the paged KV read over the context, attention, the
//! output projection and the cache write for B requests, per rank, BF16),
//! measured on one of two paths the same call lowers to:
//!
//! - backend `vllm_neuron_nki`: the per-head block table, which routes the
//!   call to the `attention_block_tkg` NKI kernel (the kernel reads the paged
//!   cache itself); what a Llama3 `forward_decode` patched to build that table
//!   launches, and the arch's default (`decode_attention: nki_kernel`);
//! - backend `vllm_neuron_torch`: the stock Llama3 `forward_decode` call as
//!   is (2-D block table, explicit decode mask, in-kernel cache update), which
//!   at TP4 with two KV heads per rank the functional routes to its torch
//!   fallback (a torch gather of the K/V blocks on GpSimd, dense attention);
//!   the stock engine's path, the gate's truth (`decode_attention: torch_fallback`).
//!
//! Timed alone on one LNC2 logical core; the row is raw device time. Decode
//! only; the context's KV pool caps the buckets.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_nki::{
    self, NeuronNkiKernelConfig, NeuronNkiKernelInput, NkiPhases, NEURON_TORCH_BACKEND, NKI_BACKEND,
};
use crate::timing::sweep::SweepGrid;
use crate::timing::Coords;

pub struct NeuronAttentionDecodeSpec;

impl KernelSpec for NeuronAttentionDecodeSpec {
    type Config = NeuronNkiKernelConfig;
    type Input = NeuronNkiKernelInput;

    const KIND: KernelKind = "neuron_attention_decode";

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

register_kernel!(NeuronAttentionDecodeKernel, NeuronAttentionDecodeSpec);

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
        assert_rows::<NeuronAttentionDecodeSpec>(&test_config(vec![], vec![1, 16]), &expected);
    }

    #[test]
    fn either_path_is_one_backend_per_config() {
        let mut torch = test_config(vec![], vec![1, 16]);
        torch.backends = vec![NEURON_TORCH_BACKEND];
        NeuronAttentionDecodeSpec::validate_config(&torch).unwrap();
        let mut both = test_config(vec![], vec![1, 16]);
        both.backends = vec![NKI_BACKEND, NEURON_TORCH_BACKEND];
        assert!(NeuronAttentionDecodeSpec::validate_config(&both).is_err());
        let mut other = test_config(vec![], vec![1, 16]);
        other.backends = vec!["vllm_neuron"];
        assert!(NeuronAttentionDecodeSpec::validate_config(&other).is_err());
    }
}
