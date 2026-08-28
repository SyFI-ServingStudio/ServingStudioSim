//! FlashInfer TRT-LLM standalone all-reduce used by vLLM's TP boundaries.
//!
//! This is shape-keyed rather than byte-keyed: vLLM passes a contiguous
//! `[num_tokens, hidden_dim]` tensor and changes the PDL completion policy at
//! 16 tokens. The generic `all_reduce` kind cannot represent either property.

use crate::common::Fabric;
use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AllReduceFusionKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_gpus: u32,
    pub hidden_dim: u32,
    #[compute_dtype]
    pub dtype: DType,
    pub fabric: Fabric,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct AllReduceFusionKernelInput {
    pub num_tokens: u32,
}

pub struct AllReduceFusionSpec;

impl AllReduceFusionSpec {
    /// vLLM's default FlashInfer workspace cap on SM100.
    fn max_fused_bytes(num_gpus: u32) -> u64 {
        let mib = match num_gpus {
            2 => 64,
            4 => 32,
            8 => 1,
            _ => panic!("FlashInfer all-reduce fusion supports TP 2/4/8"),
        };
        mib * 1024 * 1024
    }

    /// Largest token count for which vLLM selects FlashInfer on B200.
    pub fn max_fused_tokens(config: &AllReduceFusionKernelConfig) -> u32 {
        assert!(
            config.gpu_name.contains("B200"),
            "all_reduce_fusion is currently profiled only on B200"
        );
        let bytes_per_token = u64::from(config.hidden_dim)
            .checked_mul(config.dtype.size_bytes() as u64)
            .expect("all-reduce bytes per token overflow");
        (Self::max_fused_bytes(config.num_gpus) / bytes_per_token) as u32
    }
}

impl KernelSpec for AllReduceFusionSpec {
    type Config = AllReduceFusionKernelConfig;
    type Input = AllReduceFusionKernelInput;

    const KIND: KernelKind = "all_reduce_fusion";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let max_fused_tokens = Self::max_fused_tokens(config);
        let mut tokens = Axis::chain([Axis::values([1, 2, 4, 8, 16, 17]), Axis::token_axis()]);
        tokens.retain(|num_tokens| *num_tokens <= f64::from(max_fused_tokens));
        if tokens.last().copied() != Some(f64::from(max_fused_tokens)) {
            tokens.push(f64::from(max_fused_tokens));
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
        })
    }
}

register_kernel!(AllReduceFusionKernel, AllReduceFusionSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config(num_gpus: u32) -> AllReduceFusionKernelConfig {
        AllReduceFusionKernelConfig {
            backends: vec!["flashinfer_trtllm"],
            gpu_name: "NVIDIA B200".to_string(),
            num_gpus,
            hidden_dim: 6144,
            dtype: DType::Bf16,
            fabric: Fabric::Nvlink,
        }
    }

    #[test]
    fn config_and_input_identity_are_explicit() {
        let config = config(4);
        assert_eq!(config.compute_dtype(), Some(DType::Bf16));
        assert_eq!(config.backends(), &["flashinfer_trtllm"]);
        assert_eq!(config.describe_config()["hidden_dim"], 6144);
        let input = AllReduceFusionKernelInput { num_tokens: 8 };
        assert_eq!(&*input.coords(), &[8.0]);
        assert_eq!(
            AllReduceFusionKernelInput::coord_field_names(),
            &["num_tokens"]
        );
    }

    #[test]
    fn sweep_keeps_completion_boundary_and_workspace_cap() {
        let tp4 = AllReduceFusionSpec::sweep_grid(&config(4));
        assert!(tp4.axes()[0].contains(&16.0));
        assert!(tp4.axes()[0].contains(&17.0));
        assert_eq!(tp4.axes()[0].last(), Some(&2730.0));

        let tp8 = AllReduceFusionSpec::sweep_grid(&config(8));
        assert_eq!(tp8.axes()[0].last(), Some(&85.0));
    }

    #[test]
    fn enumerate_matches_python_schema() {
        let config = config(4);
        let grid = AllReduceFusionSpec::sweep_grid(&config);
        let payloads = AllReduceFusionSpec::enumerate(&config, &grid, "flashinfer_trtllm");
        let fields = payloads[0].fields();

        assert_eq!(fields.len(), 6);
        assert_eq!(
            fields.get("backend"),
            Some(&Value::from("flashinfer_trtllm"))
        );
        assert_eq!(fields.get("num_gpus"), Some(&Value::from(4_u32)));
        assert_eq!(fields.get("num_tokens"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("hidden_dim"), Some(&Value::from(6144_u32)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("fabric"), Some(&Value::from("nvlink")));
    }
}
