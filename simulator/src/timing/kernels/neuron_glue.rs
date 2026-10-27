//! The torch glue of one layer around its NKI kernels: two RMSNorms,
//! residual adds, QKV split and layouts, and (prefill) the paged KV write.
//! neuronx-cc compiles it. It is timed alone on one LNC2 logical core, per TP4
//! rank in BF16, at the prefill token bucket or the decode batch bucket; the
//! row is raw device time.
//!
//! An approximation: the model fuses its own glue into its graph differently.
//! This row approximates the non-kernel work; it does not reproduce it.

use crate::timing::bridge::{ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_nki::{
    self, NeuronNkiKernelConfig, NeuronNkiKernelInput, NkiPhases,
};
use crate::timing::sweep::SweepGrid;
use crate::timing::Coords;

pub struct NeuronGlueSpec;

impl KernelSpec for NeuronGlueSpec {
    type Config = NeuronNkiKernelConfig;
    type Input = NeuronNkiKernelInput;

    const KIND: KernelKind = "neuron_glue";

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

register_kernel!(NeuronGlueKernel, NeuronGlueSpec);

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
        assert_rows::<NeuronGlueSpec>(&test_config(vec![512], vec![1, 16]), &expected);
    }
}
