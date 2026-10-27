//! Standalone NKI gated MLP of one layer: `nkilib` `mlp` through
//! `vllm_neuron.functional.mlp`, per TP4 rank, `[rows, 4096]` with gate/up
//! `[4096, 3584]` and down `[3584, 4096]` in BF16. `rows` is the prefill token
//! bucket or the decode batch bucket. The kernel selects its CTE or TKG path
//! from the row count. The unit is timed alone on one LNC2 logical core; the
//! row is raw device time.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_nki::{
    self, NeuronNkiKernelConfig, NeuronNkiKernelInput, NkiPhases,
};
use crate::timing::sweep::SweepGrid;
use crate::timing::Coords;

pub struct NeuronNkiMlpSpec;

impl KernelSpec for NeuronNkiMlpSpec {
    type Config = NeuronNkiKernelConfig;
    type Input = NeuronNkiKernelInput;

    const KIND: KernelKind = "neuron_nki_mlp";

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

register_kernel!(NeuronNkiMlpKernel, NeuronNkiMlpSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::neuron_nki::{assert_rows, test_config, NKI_BACKEND};

    #[test]
    fn rows_are_the_python_fields_in_inventory_order() {
        let expected: Vec<_> = [("prefill", 512), ("decode", 1), ("decode", 16)]
            .into_iter()
            .map(|(phase, bucket)| {
                serde_json::json!({
                    "backend": NKI_BACKEND, "phase": phase, "token_bucket": bucket,
                    "tp_size": 4, "dtype": "bf16"
                })
            })
            .collect();
        assert_rows::<NeuronNkiMlpSpec>(&test_config(vec![512], vec![1, 16]), &expected);
    }
}
