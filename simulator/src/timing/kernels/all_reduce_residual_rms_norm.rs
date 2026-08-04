//! FlashInfer TRT-LLM fused TP boundary:
//! all-reduce + residual add + RMSNorm in one device kernel.
//!
//! Unlike byte-keyed pure all-reduce, launch cost depends on the 2-D
//! `[num_tokens, hidden_dim]` shape. `hidden_dim` is static config and
//! `num_tokens` is the runtime/sweep axis. The grid is capped at vLLM's H200
//! fusion-size threshold for the TP degree; callers must use the unfused path
//! above that threshold.

use crate::common::Fabric;
use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AllReduceResidualRmsNormKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_gpus: u32,
    pub hidden_dim: u32,
    #[compute_dtype]
    pub dtype: DType,
    pub fabric: Fabric,
    pub strategy: String,
    pub launch_with_pdl: bool,
    pub fp32_acc: bool,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct AllReduceResidualRmsNormKernelInput {
    pub num_tokens: u32,
}

pub struct AllReduceResidualRmsNormSpec;

impl AllReduceResidualRmsNormSpec {
    /// Effective vLLM H200 fusion boundary for the compiled shape ranges used
    /// by the alignment target. The TP4 workspace itself can hold 4 MiB, but
    /// vLLM compiles ranges `(1, 256)` and `(257, 8192)` and only fuses a range
    /// when its end fits the workspace; therefore the observable TP4 boundary
    /// is 256 BF16 x 4096 tokens (2 MiB), not the raw 512-token capacity.
    fn max_fused_bytes(num_gpus: u32) -> u64 {
        match num_gpus {
            2 => 64 * 1024 * 1024,
            4 => 2 * 1024 * 1024,
            8 => 512 * 1024,
            _ => panic!("FlashInfer fused all-reduce supports TP 2/4/8"),
        }
    }

    /// Largest token count for which vLLM selects the fused SM90 recipe.
    ///
    /// L3 uses the same policy to choose between this fused leaf and the
    /// unfused all-reduce + RMSNorm fallback. Keeping the threshold here makes
    /// the runtime branch and this kernel's profiling grid share one owner.
    pub fn max_fused_tokens(config: &AllReduceResidualRmsNormKernelConfig) -> u32 {
        let bytes_per_token = (config.hidden_dim as u64) * (config.dtype.size_bytes() as u64);
        (Self::max_fused_bytes(config.num_gpus) / bytes_per_token) as u32
    }
}

impl KernelSpec for AllReduceResidualRmsNormSpec {
    type Config = AllReduceResidualRmsNormKernelConfig;
    type Input = AllReduceResidualRmsNormKernelInput;

    const KIND: KernelKind = "all_reduce_residual_rms_norm";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let max_fused_tokens = Self::max_fused_tokens(config);
        let mut tokens = Axis::chain([Axis::pow2(0, 4), Axis::token_axis()]);
        tokens.retain(|num_tokens| *num_tokens <= max_fused_tokens as f64);
        if tokens.last().copied() != Some(max_fused_tokens as f64) {
            tokens.push(max_fused_tokens as f64);
            tokens.sort_by(f64::total_cmp);
        }
        SweepGrid::new(vec![tokens])
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
                .with("num_gpus", config.num_gpus)
                .with("num_tokens", num_tokens as u32)
                .with("hidden_dim", config.hidden_dim)
                .with("dtype", config.dtype.as_str())
                .with("fabric", config.fabric.as_str())
                .with("strategy", config.strategy.as_str())
                .with("launch_with_pdl", config.launch_with_pdl)
                .with("trigger_completion_at_end", num_tokens as u32 > 16)
                .with("fp32_acc", config.fp32_acc)
        })
    }
}

register_kernel!(AllReduceResidualRmsNormKernel, AllReduceResidualRmsNormSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config(num_gpus: u32) -> AllReduceResidualRmsNormKernelConfig {
        AllReduceResidualRmsNormKernelConfig {
            backends: vec!["flashinfer_trtllm"],
            gpu_name: "NVIDIA H200".to_string(),
            num_gpus,
            hidden_dim: 4096,
            dtype: DType::Bf16,
            fabric: Fabric::Nvlink,
            strategy: "auto".to_string(),
            launch_with_pdl: true,
            fp32_acc: true,
        }
    }

    #[test]
    fn config_identity_and_dtype_are_explicit() {
        let config = config(2);
        assert_eq!(config.compute_dtype(), Some(DType::Bf16));
        assert_eq!(config.backends(), &["flashinfer_trtllm"]);
        assert_eq!(config.describe_config()["hidden_dim"], 4096);
        assert_eq!(config.describe_config()["strategy"], "auto");
    }

    #[test]
    fn input_axis_is_num_tokens() {
        let input = AllReduceResidualRmsNormKernelInput { num_tokens: 32 };
        assert_eq!(&*input.coords(), &[32.0]);
        assert_eq!(
            AllReduceResidualRmsNormKernelInput::coord_field_names(),
            &["num_tokens"]
        );
    }

    #[test]
    fn sweep_is_capped_by_vllm_h200_fusion_policy() {
        let tp2 = AllReduceResidualRmsNormSpec::sweep_grid(&config(2));
        let tp4 = AllReduceResidualRmsNormSpec::sweep_grid(&config(4));
        assert_eq!(tp2.axes()[0].last(), Some(&8192.0));
        assert_eq!(tp4.axes()[0].last(), Some(&256.0));
        assert!(matches!(
            AllReduceResidualRmsNormSpec::cache_kind("flashinfer_trtllm"),
            CacheKind::Cache1DLinear
        ));
    }

    #[test]
    fn enumerate_matches_python_args_exactly() {
        let config = config(4);
        let grid = AllReduceResidualRmsNormSpec::sweep_grid(&config);
        let first =
            &AllReduceResidualRmsNormSpec::enumerate(&config, &grid, "flashinfer_trtllm")[0];
        let fields = first.fields();
        assert_eq!(fields.len(), 10);
        assert_eq!(
            fields.get("backend"),
            Some(&Value::from("flashinfer_trtllm"))
        );
        assert_eq!(fields.get("num_gpus"), Some(&Value::from(4_u32)));
        assert_eq!(fields.get("num_tokens"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("hidden_dim"), Some(&Value::from(4096_u32)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("fabric"), Some(&Value::from("nvlink")));
        assert_eq!(fields.get("strategy"), Some(&Value::from("auto")));
        assert_eq!(fields.get("launch_with_pdl"), Some(&Value::from(true)));
        assert_eq!(
            fields.get("trigger_completion_at_end"),
            Some(&Value::from(false))
        );
        let thirty_two =
            &AllReduceResidualRmsNormSpec::enumerate(&config, &grid, "flashinfer_trtllm")[5];
        assert_eq!(
            thirty_two.fields().get("num_tokens"),
            Some(&Value::from(32_u32))
        );
        assert_eq!(
            thirty_two.fields().get("trigger_completion_at_end"),
            Some(&Value::from(true))
        );
        assert_eq!(fields.get("fp32_acc"), Some(&Value::from(true)));
    }
}
