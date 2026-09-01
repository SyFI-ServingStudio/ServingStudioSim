//! SGLang deferred-MoE finalize with an optional fused shared-expert add.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeFinalizeFuseSharedKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub top_k: u32,
    pub hidden_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
    pub fuse_shared_output: bool,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MoeFinalizeFuseSharedKernelInput {
    pub num_tokens: u32,
}

pub struct MoeFinalizeFuseSharedSpec;

impl KernelSpec for MoeFinalizeFuseSharedSpec {
    type Config = MoeFinalizeFuseSharedKernelConfig;
    type Input = MoeFinalizeFuseSharedKernelInput;

    const KIND: KernelKind = "moe_finalize_fuse_shared";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::chain([
            Axis::values([1, 4, 8, 16, 32, 48]),
            Axis::token_axis(),
        ])])
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
                .with("top_k", config.top_k)
                .with("hidden_dim", config.hidden_dim.get())
                .with("dtype", config.dtype.as_str())
                .with("fuse_shared_output", config.fuse_shared_output)
        })
    }
}

register_kernel!(MoeFinalizeFuseSharedKernel, MoeFinalizeFuseSharedSpec);

#[cfg(test)]
mod tests {
    use super::{
        MoeFinalizeFuseSharedKernelConfig, MoeFinalizeFuseSharedKernelInput,
        MoeFinalizeFuseSharedSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config(fuse_shared_output: bool) -> MoeFinalizeFuseSharedKernelConfig {
        MoeFinalizeFuseSharedKernelConfig {
            backends: vec!["sglang_cuda"],
            gpu_name: "NVIDIA B200".to_string(),
            top_k: 8,
            hidden_dim: 6144.into(),
            dtype: DType::Bf16,
            fuse_shared_output,
        }
    }

    #[test]
    fn payload_matches_python_schema_and_token_coordinates() {
        let config = config(true);
        let grid = MoeFinalizeFuseSharedSpec::sweep_grid(&config);
        let axis = &grid.axes()[0];
        assert_eq!(axis.len(), 68);
        assert_eq!(axis.first(), Some(&1.0));
        assert_eq!(axis.last(), Some(&65_536.0));
        for anchor in [32.0, 48.0, 2_048.0, 8_192.0] {
            assert!(axis.contains(&anchor), "token grid is missing {anchor}");
        }
        assert!(matches!(
            MoeFinalizeFuseSharedSpec::cache_kind("sglang_cuda"),
            CacheKind::Cache1DLinear
        ));

        let payloads = MoeFinalizeFuseSharedSpec::enumerate(&config, &grid, "sglang_cuda");
        assert_eq!(payloads.last().unwrap().fields()["num_tokens"], 65_536);
        let payload = &payloads[0];
        let fields = payload.fields();
        assert_eq!(fields.len(), 6);
        assert_eq!(fields["backend"], Value::from("sglang_cuda"));
        assert_eq!(fields["num_tokens"], Value::from(1_u32));
        assert_eq!(fields["top_k"], Value::from(8_u32));
        assert_eq!(fields["hidden_dim"], Value::from(6144_u32));
        assert_eq!(fields["dtype"], Value::from("bf16"));
        assert_eq!(fields["fuse_shared_output"], Value::from(true));
        assert_eq!(config.compute_dtype(), Some(DType::Bf16));
        assert_eq!(config.kv_dtype(), None);

        let input = MoeFinalizeFuseSharedKernelInput { num_tokens: 32 };
        assert_eq!(&*input.coords(), &[32.0]);
        assert_eq!(
            MoeFinalizeFuseSharedKernelInput::coord_field_names(),
            &["num_tokens"]
        );
        let slot: SlotInput = input.into();
        assert!(matches!(&slot, SlotInput::MoeFinalizeFuseShared(_)));
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_tokens": 32})
        );
    }

    #[test]
    fn shared_output_fusion_is_config_identity() {
        assert_ne!(config(true), config(false));
    }
}
