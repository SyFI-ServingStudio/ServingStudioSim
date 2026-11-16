//! Engram gate: mix the Engram lookup (after `wkv`) into each residual stream (SGLang `fused_engram_gate`).

use crate::timing::bridge::ArgsPayload;
use crate::timing::bridge::KernelKind;
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::mhc_pre_rms_norm::{
    enumerate, sweep_grid, MhcRmsNormKernelConfig, MhcRmsNormKernelInput,
};
use crate::timing::sweep::SweepGrid;

/// Same Python args as `mhc_pre_rms_norm` (`num_tokens`, `hidden_size`,
/// `hc_mult`, `hidden_dtype`), so it shares that kind's config, input and
/// token sweep. The sweep's token axis includes the row counts where SGLang
/// switches implementation (see the Python kind's doc).
pub struct EngramGateSpec;

impl KernelSpec for EngramGateSpec {
    type Config = MhcRmsNormKernelConfig;
    type Input = MhcRmsNormKernelInput;

    const KIND: KernelKind = "engram_gate";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        sweep_grid()
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        enumerate(config, grid, backend)
    }
}

register_kernel!(EngramGateKernel, EngramGateSpec);

#[cfg(test)]
mod tests {
    use super::EngramGateSpec;
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;
    use crate::timing::kernels::mhc_pre_rms_norm::MhcRmsNormKernelConfig;

    /// The payload must carry exactly the Python `*Args` columns, or the
    /// profiler rejects every row (and the cache never resolves).
    #[test]
    fn payload_matches_the_python_args_columns() {
        let config = MhcRmsNormKernelConfig {
            backends: vec!["sglang_triton"],
            gpu_name: "NVIDIA RTX PRO 6000 Blackwell Server Edition".to_string(),
            hidden_size: 4096.into(),
            hc_mult: 4,
            hidden_dtype: DType::Bf16,
        };
        let payloads = EngramGateSpec::enumerate(
            &config,
            &EngramGateSpec::sweep_grid(&config),
            "sglang_triton",
        );
        let mut fields: Vec<_> = payloads[0].fields().keys().cloned().collect();
        fields.sort();
        assert_eq!(
            fields,
            [
                "backend",
                "hc_mult",
                "hidden_dtype",
                "hidden_size",
                "num_tokens"
            ]
        );
    }
}
