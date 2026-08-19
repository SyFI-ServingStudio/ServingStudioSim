//! Intra-domain point-to-point kernel: one cached perf model per `fabric`
//! config, swept over `message_size_bytes`.
//!
//! Comm cost is size-keyed, NOT dtype-keyed: a byte is a byte on the wire, so
//! the curve is `time_ms` vs `message_size_bytes` and dtype is not a cache axis.
//! An fp8 payload is simply fewer bytes on the same curve (the caller passes the
//! fp8-width `message_size_bytes`). The kernel is profiled once at bf16 (the
//! `enumerate` `dtype` field is a fixed profiling artifact, kept only to stay
//! aligned with the Python `P2pIntraArgs` wire schema).
//!
//! This is the intra-NVL-domain leg of the `MoE` network model (ref's
//! `get_p2p_metrics_batch` curve in `common_timing.rs`). A single send/recv
//! between two ranks sharing an `NVLink` domain, measured as time vs message
//! size. The `MoE` dispatch/combine L2 ops (`op/moe`) compute each network
//! stage's per-rank `max(send, recv)` byte load and look this curve up per
//! stage (`P2pTier::IntraDomain`).
//!
//! Comm vs compute: rows carry `algbw/busbw` (no `tflops`); `message_size_bytes`
//! is an args/cache-key axis only. The cost is `time_ms` interpolated over the
//! byte ladder. p2p is always 2 endpoints, so there is no `num_gpus` axis
//! (unlike `all_reduce`).
//!
//! Shape split: static config is `(fabric, dtype)`; per-pair `message_size_bytes`
//! is the runtime sweep axis. The sweep is a `pow2(10, 28)` byte ladder
//! (1 KB .. 256 MB), matching ref `P2P_MIN_BYTES`/`P2P_MAX_BYTES`;
//! `Cache1DLinear` interpolates between adjacent powers of two.

use crate::common::Fabric;
use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct P2pIntraKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub fabric: Fabric,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct P2pIntraKernelInput {
    /// Bytes moved over the single src→dst link in this transfer. The `MoE` net
    /// model passes the bottleneck rank's `max(send_bytes, recv_bytes)` for a
    /// stage.
    pub message_size_bytes: u64,
}

pub struct P2pIntraSpec;

impl KernelSpec for P2pIntraSpec {
    type Config = P2pIntraKernelConfig;
    type Input = P2pIntraKernelInput;

    const KIND: KernelKind = "p2p_intra";

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
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "message_size is a grid point from Axis::pow2, always a small non-negative power of two"
            )]
            let message_size_bytes = message_size as u64;
            ArgsPayload::new()
                .with("backend", backend)
                .with("message_size_bytes", message_size_bytes)
                // Fixed profiling dtype: comm is size-keyed (see module doc), the
                // curve is measured once at bf16 for every logical payload dtype.
                .with("dtype", DType::Bf16.as_str())
                .with("fabric", config.fabric.as_str())
        })
    }
}

register_kernel!(P2pIntraKernel, P2pIntraSpec);

#[cfg(test)]
mod tests {
    use super::{P2pIntraKernelConfig, P2pIntraKernelInput, P2pIntraSpec};
    use crate::common::Fabric;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn cfg() -> P2pIntraKernelConfig {
        P2pIntraKernelConfig {
            backends: vec!["nccl"],
            gpu_name: "H100".to_string(),
            fabric: Fabric::Nvlink,
        }
    }

    #[test]
    fn config_identity_includes_backend_and_shape() {
        let c = cfg();
        assert_eq!(c.backends, vec!["nccl"]);
        assert_eq!(c.fabric, Fabric::Nvlink);
    }

    #[test]
    fn describe_config_renders_tidy_field_list() {
        assert_eq!(
            cfg().describe_config(),
            serde_json::json!({
                "backends": ["nccl"], "gpu_name": "H100", "fabric": "nvlink",
            })
        );
    }

    #[test]
    fn input_sweep_coords_flatten_message_size_field() {
        let input = P2pIntraKernelInput {
            message_size_bytes: 1 << 20,
        };
        assert_eq!(&*input.coords(), &[(1u64 << 20) as f64]);
    }

    #[test]
    fn sweep_grid_is_1d_message_size_ladder() {
        let grid = P2pIntraSpec::sweep_grid(&cfg());
        assert_eq!(grid.axes().len(), 1);
        // 2^10 .. 2^28 inclusive = 19 points.
        assert_eq!(grid.axes()[0].len(), 19);
        assert_eq!(grid.axes()[0].first().copied(), Some(1024.0));
        assert_eq!(grid.axes()[0].last().copied(), Some((1u64 << 28) as f64));
        assert!(matches!(
            P2pIntraSpec::cache_kind("nccl"),
            CacheKind::Cache1DLinear
        ));
    }

    #[test]
    fn enumerate_emits_payload_with_all_wire_fields() {
        let c = cfg();
        let grid = P2pIntraSpec::sweep_grid(&c);
        let payloads = P2pIntraSpec::enumerate(&c, &grid, "nccl");

        assert!(!payloads.is_empty());
        let first = &payloads[0];
        let fields = first.fields();

        // Wire schema: { backend, message_size_bytes, dtype, fabric } — must
        // stay aligned with Python P2pIntraArgs in profiling/kernels/p2p_intra.py.
        assert_eq!(fields.len(), 4);
        assert_eq!(fields.get("backend"), Some(&Value::from("nccl")));
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("fabric"), Some(&Value::from("nvlink")));
        assert!(fields
            .get("message_size_bytes")
            .and_then(Value::as_u64)
            .is_some());
        assert_eq!(first.backend(), Some("nccl"));
    }
}
