//! Residual-stream sum + RMSNorm with a selectable output set, as Inductor
//! fuses it into one Triton reduction.
//!
//! vLLM's compiled graph reaches this kernel with one to three addends: a plain
//! norm, a residual add, or a residual add that re-adds the previous norm's two
//! inputs because Inductor does not store a sum that only this kernel reads.
//! A static FP8 input quantization of the next linear layer fuses in as an
//! output plane. Which planes are written is part of the kernel: each one is a
//! full `num_tokens x hidden_dim` store, so they are config fields, not a
//! post-hoc byte scale.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MultiAddRmsNormKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    /// Row-aligned tensors summed before the norm (1 = plain RMSNorm).
    pub num_addends: u32,
    /// The sum is written back as the new residual.
    pub store_sum: bool,
    /// The normalized rows are written in `dtype`.
    pub store_normed: bool,
    /// The normalized rows are written as static per-tensor FP8 E4M3.
    pub store_fp8: bool,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MultiAddRmsNormKernelInput {
    pub num_tokens: u32,
}

pub struct MultiAddRmsNormSpec;

impl KernelSpec for MultiAddRmsNormSpec {
    type Config = MultiAddRmsNormKernelConfig;
    type Input = MultiAddRmsNormKernelInput;

    const KIND: KernelKind = "multi_add_rms_norm";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::chain([Axis::pow2(0, 4), Axis::token_axis()])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
                .with("hidden_dim", config.hidden_dim.get())
                .with("num_addends", config.num_addends)
                .with("store_sum", config.store_sum)
                .with("store_normed", config.store_normed)
                .with("store_fp8", config.store_fp8)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(MultiAddRmsNormKernel, MultiAddRmsNormSpec);

#[cfg(test)]
mod tests {
    use super::{MultiAddRmsNormKernelConfig, MultiAddRmsNormSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;
    use serde_json::Value;

    #[test]
    fn payload_forwards_the_output_set_the_profiler_compiles() {
        // The output planes change the generated kernel; dropping one from the
        // payload would price a GQA input norm with the GDN one's extra store.
        let config = MultiAddRmsNormKernelConfig {
            backends: vec!["vllm_inductor"],
            gpu_name: "NVIDIA B200".to_string(),
            hidden_dim: 5120.into(),
            num_addends: 3,
            store_sum: true,
            store_normed: false,
            store_fp8: true,
            dtype: DType::Bf16,
        };
        let payloads = MultiAddRmsNormSpec::enumerate(
            &config,
            &MultiAddRmsNormSpec::sweep_grid(&config),
            "vllm_inductor",
        );
        let fields = payloads[0].fields();
        assert_eq!(fields["num_tokens"], Value::from(1u32));
        assert_eq!(fields["num_addends"], Value::from(3u32));
        assert_eq!(fields["store_sum"], Value::from(true));
        assert_eq!(fields["store_normed"], Value::from(false));
        assert_eq!(fields["store_fp8"], Value::from(true));
        assert_eq!(fields.len(), 8);
    }
}
