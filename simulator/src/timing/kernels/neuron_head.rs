//! The head unit of one Llama 3.1 8B TP4 forward on Trainium2: the embedding
//! row gather of the step's tokens, the final RMSNorm of the sampled rows, the
//! vocab-sharded `lm_head` GEMM (`[rows, 4096] @ [4096, 32064]` per rank) and
//! the greedy argmax, compiled as one graph and timed alone on one LNC2 logical
//! core. In prefill the sampled row is the prompt's last token (one row at
//! token bucket T, the embedding gathers T rows); in decode every request
//! samples (B rows at batch bucket B). The logits all-gather across the four
//! ranks is a separate collective leaf.
//!
//! Shares the NKI leaf config and input (`neuron_nki.rs`): a coordinate is one
//! compiled variant of the prefill then decode inventory, with no
//! interpolation. Measured in the vLLM Neuron 0.29 image (`vllm_neuron_nki`).

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_nki::{
    self, NeuronNkiKernelConfig, NeuronNkiKernelInput, NkiPhases,
};
use crate::timing::sweep::SweepGrid;
use crate::timing::Coords;

pub struct NeuronHeadSpec;

impl KernelSpec for NeuronHeadSpec {
    type Config = NeuronNkiKernelConfig;
    type Input = NeuronNkiKernelInput;

    const KIND: KernelKind = "neuron_head";

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        neuron_nki::validate_config(config, NkiPhases::Both)
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
        neuron_nki::enumerate(config, grid, backend, |row, phase, bucket| {
            neuron_nki::identity_fields(
                row.with("phase", phase.as_str())
                    .with("token_bucket", bucket),
                config,
            )
        })
    }
}

register_kernel!(NeuronHeadKernel, NeuronHeadSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::neuron_nki::{assert_rows, test_config, NKI_BACKEND};

    #[test]
    fn rows_are_the_python_fields_in_inventory_order() {
        let expected: Vec<_> = [("prefill", 128), ("prefill", 512), ("decode", 16)]
            .into_iter()
            .map(|(phase, bucket)| {
                serde_json::json!({
                    "backend": NKI_BACKEND, "phase": phase, "token_bucket": bucket,
                    "tp_size": 4, "dtype": "bf16"
                })
            })
            .collect();
        assert_rows::<NeuronHeadSpec>(&test_config(vec![128, 512], vec![16]), &expected);
    }
}
