//! Kimi-K3 fused KDA decode (conv + recurrence + gated norm).

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KdaFusedDecodeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub head_k_dim: Dim,
    pub head_v_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
    pub state_dtype: DType,
    pub lower_bound: i32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct KdaFusedDecodeKernelInput {
    pub batch_size: u32,
}

pub struct KdaFusedDecodeSpec;

impl KernelSpec for KdaFusedDecodeSpec {
    type Config = KdaFusedDecodeKernelConfig;
    type Input = KdaFusedDecodeKernelInput;

    const KIND: KernelKind = "kda_fused_decode";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::pow2(0, 8)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|batch_size| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", batch_size as u32)
                .with("num_heads", config.num_heads.get())
                .with("head_k_dim", config.head_k_dim.get())
                .with("head_v_dim", config.head_v_dim.get())
                .with("dtype", config.dtype.as_str())
                .with("state_dtype", config.state_dtype.as_str())
                .with("lower_bound", config.lower_bound as f64)
        })
    }
}

register_kernel!(KdaFusedDecodeKernel, KdaFusedDecodeSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;

    fn config() -> KdaFusedDecodeKernelConfig {
        KdaFusedDecodeKernelConfig {
            backends: vec!["sglang_fused"],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads: 12.into(),
            head_k_dim: 128.into(),
            head_v_dim: 128.into(),
            dtype: DType::Bf16,
            state_dtype: DType::Fp32,
            lower_bound: -5,
        }
    }

    #[test]
    fn kind_backend_and_coords_are_stable() {
        let cfg = config();
        assert_eq!(KdaFusedDecodeSpec::KIND, "kda_fused_decode");
        assert_eq!(cfg.backends(), &["sglang_fused"]);
        assert_eq!(
            KdaFusedDecodeSpec::cache_kind("sglang_fused"),
            CacheKind::Cache1DLinear
        );
        assert_eq!(
            &*KdaFusedDecodeKernelInput { batch_size: 128 }.coords(),
            &[128.0]
        );
        assert_eq!(KdaFusedDecodeSpec::sweep_grid(&cfg).axes()[0][0], 1.0);
    }

    #[test]
    fn payload_has_same_args_as_recurrent_kda() {
        let payload = KdaFusedDecodeSpec::enumerate(
            &config(),
            &SweepGrid::new(vec![Axis::values([32])]),
            "sglang_fused",
        )[0]
        .clone();
        assert_eq!(payload.fields().len(), 8);
        assert_eq!(payload.fields()["batch_size"], serde_json::json!(32));
        assert_eq!(payload.fields()["lower_bound"], serde_json::json!(-5.0));
    }
}
