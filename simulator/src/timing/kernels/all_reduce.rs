//! All-reduce collective kernel: one cached perf model per `(num_gpus, fabric)`
//! config, swept over `message_size_bytes`.
//!
//! Comm cost is size-keyed, NOT dtype-keyed: the collective's cost is dominated
//! by the `busbw`-bound bytes moved, so the curve is `time_ms` vs
//! `message_size_bytes` and dtype is not a cache axis. An fp8 payload is simply
//! fewer bytes on the same curve (the caller passes the fp8-width
//! `message_size_bytes`) — and NCCL has no fp8 reduction anyway. The kernel is
//! profiled once at bf16 (the `enumerate` `dtype` field is a fixed profiling
//! artifact, kept only to stay aligned with the Python `AllReduceArgs` schema).
//!
//! Everything generic (build / eval / the `Probe` impl / for-backend loops)
//! lives in `engine::Kernel<S>`. This file declares the all-reduce-specific
//! Config / Input, the `KIND` wire string, and the `enumerate` body that lifts
//! (config, sweep coord, backend) to the on-wire `ArgsPayload`. The Python
//! `AllReduceArgs` dataclass owns the schema.
//!
//! Comm vs the compute kernels: rows carry `algbw/busbw` measured metrics (no
//! `tflops`), so the cached metric reads `flops()==0` and `bytes()` derives from
//! `busbw × time` — the real per-GPU fabric traffic (send/recv) — instead of a
//! memory-bandwidth rate (see `bridge::payload`). `message_size_bytes` is an
//! args/cache-key axis only, NOT a result column. The cost is still just
//! `time_ms` interpolated over message size.
//!
//! Shape split (L1 design §8.2): static config is `(num_gpus, fabric, dtype)`;
//! the per-rank `message_size_bytes` is the runtime sweep axis. The sweep is a
//! `pow2(12, 30)` byte ladder (4 KB .. 1 GB), matching
//! `ref/profile/network/allreduce_nccl.py`; `Cache1DLinear` interpolates
//! between adjacent powers of two.

use crate::common::Fabric;
use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AllReduceKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_gpus: u32,
    pub fabric: Fabric,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct AllReduceKernelInput {
    /// Size of the FULL buffer each rank contributes to / receives from the
    /// collective — i.e. the whole tensor handed to `dist.all_reduce`, NOT a
    /// reduce-scatter shard (`÷ num_gpus`). The ring algorithm's 2(N-1)/N data
    /// movement is already baked into the measured time, so callers pass the
    /// complete output size. Example: a TP layer's row-parallel projection
    /// produces a full `[tokens × hidden]` partial-sum on every rank, so the
    /// caller passes `tokens × hidden × dtype_bytes` (full hidden, NOT
    /// `hidden/tp_size`).
    pub message_size_bytes: u64,
}

pub struct AllReduceSpec;

impl KernelSpec for AllReduceSpec {
    type Config = AllReduceKernelConfig;
    type Input = AllReduceKernelInput;

    const KIND: KernelKind = "all_reduce";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // 4 KB .. 1 GB byte ladder (2^12 .. 2^30).
        SweepGrid::new(vec![Axis::pow2(12, 30)])
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
                reason = "message_size comes from the pow2(12, 30) byte-ladder axis: a small, non-negative, pre-curated power of two far below u64::MAX"
            )]
            let message_size_bytes = message_size as u64;
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_gpus", config.num_gpus)
                .with("message_size_bytes", message_size_bytes)
                // Fixed profiling dtype: comm is size-keyed (see module doc), the
                // curve is measured once at bf16 for every logical payload dtype.
                .with("dtype", DType::Bf16.as_str())
                .with("fabric", config.fabric.as_str())
        })
    }
}

register_kernel!(AllReduceKernel, AllReduceSpec);

#[cfg(test)]
mod tests {
    use super::{AllReduceKernelConfig, AllReduceKernelInput, AllReduceSpec};
    use crate::common::Fabric;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn cfg() -> AllReduceKernelConfig {
        AllReduceKernelConfig {
            backends: vec!["nccl"],
            gpu_name: "H100".to_string(),
            num_gpus: 8,
            fabric: Fabric::Nvlink,
        }
    }

    #[test]
    fn config_identity_includes_backend_and_shape() {
        let c = cfg();
        assert_eq!(c.backends, vec!["nccl"]);
        assert_eq!(c.num_gpus, 8);
        assert_eq!(c.fabric, Fabric::Nvlink);
    }

    #[test]
    fn describe_config_renders_tidy_field_list() {
        assert_eq!(
            cfg().describe_config(),
            serde_json::json!({
                "backends": ["nccl"], "gpu_name": "H100",
                "num_gpus": 8, "fabric": "nvlink",
            })
        );
    }

    #[test]
    fn input_sweep_coords_flatten_message_size_field() {
        let input = AllReduceKernelInput {
            message_size_bytes: 1 << 20,
        };
        #[allow(
            clippy::cast_precision_loss,
            reason = "1u64 << 20 is a fixed test message size, far under f64's exact integer range"
        )]
        let expected = (1u64 << 20) as f64;
        assert_eq!(&*input.coords(), &[expected]);
    }

    #[test]
    fn sweep_grid_is_1d_message_size_ladder() {
        let grid = AllReduceSpec::sweep_grid(&cfg());
        assert_eq!(grid.axes().len(), 1);
        // 2^12 .. 2^30 inclusive = 19 points.
        assert_eq!(grid.axes()[0].len(), 19);
        assert_eq!(grid.axes()[0].first().copied(), Some(4096.0));
        #[allow(
            clippy::cast_precision_loss,
            reason = "1u64 << 30 is a fixed test message size, far under f64's exact integer range"
        )]
        let expected_last = (1u64 << 30) as f64;
        assert_eq!(grid.axes()[0].last().copied(), Some(expected_last));
        assert!(matches!(
            AllReduceSpec::cache_kind("nccl"),
            CacheKind::Cache1DLinear
        ));
    }

    #[test]
    fn enumerate_emits_payload_with_all_wire_fields() {
        let c = cfg();
        let grid = AllReduceSpec::sweep_grid(&c);
        let payloads = AllReduceSpec::enumerate(&c, &grid, "nccl");

        assert!(
            !payloads.is_empty(),
            "message-size sweep axis must yield at least one point"
        );
        let first = &payloads[0];
        let fields = first.fields();

        // Wire schema: { backend, num_gpus, message_size_bytes, dtype, fabric }
        // — must stay aligned with Python AllReduceArgs in
        // profiling/kernels/all_reduce.py.
        assert_eq!(fields.len(), 5);
        assert_eq!(fields.get("backend"), Some(&Value::from("nccl")));
        assert_eq!(fields.get("num_gpus"), Some(&Value::from(8_u32)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("fabric"), Some(&Value::from("nvlink")));
        assert!(fields
            .get("message_size_bytes")
            .and_then(Value::as_u64)
            .is_some());
        assert_eq!(first.backend(), Some("nccl"));
    }
}
