//! Inter-domain point-to-point kernel: one cached perf model per
//! `(fabric, dtype)` config, swept over `message_size_bytes`.
//!
//! This is the inter-NVL-domain (cross-node NIC) leg of the MoE network model
//! (ref's `get_inter_device_p2p_times_batch` curve in `common_timing.rs`). A
//! single send/recv between two ranks in different NVLink domains, measured as
//! time vs message size. The MoE dispatch/combine L2 ops (`op/moe`) look this
//! curve up for stages tagged `P2pTier::InterDomain`.
//!
//! Identical shape to `p2p_intra` (no `num_gpus`; p2p is always 2 endpoints);
//! the separate kernel keeps the NIC bandwidth curve distinct from the NVLink
//! one, mirroring ref's two perf_api methods. Static config `(fabric, dtype)`;
//! runtime sweep axis `message_size_bytes` over a `pow2(10, 28)` ladder
//! (1 KB .. 256 MB); `Cache1DLinear`.

use crate::common::Fabric;
use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Deserialize)]
pub struct P2pInterKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub fabric: Fabric,
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct P2pInterKernelInput {
    /// Bytes moved over the single cross-domain src→dst link in this transfer.
    pub message_size_bytes: u64,
}

pub struct P2pInterSpec;

impl KernelSpec for P2pInterSpec {
    type Config = P2pInterKernelConfig;
    type Input = P2pInterKernelInput;

    const KIND: KernelKind = "p2p_inter";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // 1 KB .. 256 MB byte ladder (2^10 .. 2^28).
        SweepGrid::new(vec![Axis::pow2(10, 28)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|message_size| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("message_size_bytes", message_size as u64)
                .with("dtype", config.dtype.as_str())
                .with("fabric", config.fabric.as_str())
        })
    }
}

register_kernel!(P2pInterKernel, P2pInterSpec);

#[cfg(test)]
mod tests {
    use super::{P2pInterKernelConfig, P2pInterKernelInput, P2pInterSpec};
    use crate::common::Fabric;
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn cfg() -> P2pInterKernelConfig {
        P2pInterKernelConfig {
            backends: vec!["nccl"],
            gpu_name: "H100".to_string(),
            fabric: Fabric::Infiniband,
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_includes_backend_and_shape() {
        let c = cfg();
        assert_eq!(c.backends, vec!["nccl"]);
        assert_eq!(c.fabric, Fabric::Infiniband);
        assert_eq!(c.dtype, DType::Bf16);
    }

    #[test]
    fn describe_config_renders_tidy_field_list() {
        assert_eq!(
            cfg().describe_config(),
            r#"backends=["nccl"] gpu_name="H100" fabric=Infiniband dtype=Bf16"#
        );
    }

    #[test]
    fn input_sweep_coords_flatten_message_size_field() {
        let input = P2pInterKernelInput {
            message_size_bytes: 1 << 20,
        };
        assert_eq!(&*input.coords(), &[(1u64 << 20) as f64]);
    }

    #[test]
    fn sweep_grid_is_1d_message_size_ladder() {
        let grid = P2pInterSpec::sweep_grid(&cfg());
        assert_eq!(grid.axes().len(), 1);
        assert_eq!(grid.axes()[0].len(), 19);
        assert_eq!(grid.axes()[0].first().copied(), Some(1024.0));
        assert_eq!(grid.axes()[0].last().copied(), Some((1u64 << 28) as f64));
        assert!(matches!(
            P2pInterSpec::cache_kind("nccl"),
            CacheKind::Cache1DLinear
        ));
    }

    #[test]
    fn enumerate_emits_payload_with_all_wire_fields() {
        let c = cfg();
        let grid = P2pInterSpec::sweep_grid(&c);
        let payloads = P2pInterSpec::enumerate(&c, &grid, "nccl");

        assert!(!payloads.is_empty());
        let first = &payloads[0];
        let fields = first.fields();

        // Wire schema: { backend, message_size_bytes, dtype, fabric } — must
        // stay aligned with Python P2pInterArgs in profiling/kernels/p2p_inter.py.
        assert_eq!(fields.len(), 4);
        assert_eq!(fields.get("backend"), Some(&Value::from("nccl")));
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("fabric"), Some(&Value::from("infiniband")));
        assert!(fields
            .get("message_size_bytes")
            .and_then(Value::as_u64)
            .is_some());
        assert_eq!(first.backend(), Some("nccl"));
    }
}
