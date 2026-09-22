//! Kimi-K3 KDA recurrent decode, with batch as the physical sweep axis.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KdaRecurrentDecodeKernelConfig {
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
pub struct KdaRecurrentDecodeKernelInput {
    pub batch_size: u32,
}

pub struct KdaRecurrentDecodeSpec;

impl KernelSpec for KdaRecurrentDecodeSpec {
    type Config = KdaRecurrentDecodeKernelConfig;
    type Input = KdaRecurrentDecodeKernelInput;

    const KIND: KernelKind = "kda_recurrent_decode";

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

register_kernel!(KdaRecurrentDecodeKernel, KdaRecurrentDecodeSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> KdaRecurrentDecodeKernelConfig {
        KdaRecurrentDecodeKernelConfig {
            backends: vec!["torch", "sglang_triton"],
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
    fn wire_identity_and_sweep_are_k3_specific() {
        let cfg = config();
        assert_eq!(KdaRecurrentDecodeSpec::KIND, "kda_recurrent_decode");
        assert_eq!(cfg.backends(), &["torch", "sglang_triton"]);
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(KdaRecurrentDecodeSpec::sweep_grid(&cfg).axes()[0].len(), 9);
    }

    #[test]
    fn payload_and_slot_schema_match_python() {
        let input = KdaRecurrentDecodeKernelInput { batch_size: 32 };
        assert_eq!(&*input.coords(), &[32.0]);
        assert_eq!(
            KdaRecurrentDecodeKernelInput::coord_field_names(),
            &["batch_size"]
        );
        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"batch_size": 32})
        );
        let payload = KdaRecurrentDecodeSpec::enumerate(
            &config(),
            &SweepGrid::new(vec![Axis::values([1])]),
            "sglang_triton",
        )[0]
        .clone();
        assert_eq!(payload.fields().len(), 8);
        assert_eq!(payload.fields()["lower_bound"], Value::from(-5.0));
        assert_eq!(
            KdaRecurrentDecodeSpec::cache_kind("torch"),
            CacheKind::Cache1DLinear
        );
    }
}
