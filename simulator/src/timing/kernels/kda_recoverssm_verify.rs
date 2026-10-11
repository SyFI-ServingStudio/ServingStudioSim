//! KDA RecoverSSM speculative-verify `kda_recoverssm_verify` call (vLLM fork,
//! Kimi-K3 ops; the GLM-5.3-Flash KDA layer is the same op).
//!
//! One semantic call, one Triton launch per layer: read each request's fp32
//! checkpoint once, run its full `spec_query_len` window through the KDA
//! recurrence in registers, write the outputs and the per-token correction and
//! key/gate records. No state is written; `kda_recoverssm_commit` folds the
//! accepted records later. Head geometry, window length and dtype are static
//! identity; the number of verified requests is the only runtime axis (every
//! request verifies a full window, `batch_size * spec_query_len` tokens).

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct KdaRecoverSSMVerifyKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub spec_query_len: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct KdaRecoverSSMVerifyKernelInput {
    /// Spec-decode requests verified together.
    pub batch_size: u32,
}

/// Programs past which a launch of these kernels is several waves deep. On
/// B200 the verify and commit kernels hold about 740 programs at once (148 SMs,
/// five resident blocks each), and their time steps by one wave each time the
/// program count crosses a multiple of that: 8 heads verify 16 requests in
/// 22.9 us and 24 in 41.3 us. Linear interpolation between powers of two
/// misses those steps by up to 21% (B200 job 5569). Past ~5 waves a missed step
/// is under 10%, so powers of two suffice there.
const DENSE_PROGRAMS: u32 = 4096;

/// Value blocks per (request, head): the ops tile V by `min(next_pow2(V), 32)`.
pub(crate) fn value_blocks(head_dim: u32) -> u32 {
    let block = head_dim.next_power_of_two().min(32);
    head_dim.div_ceil(block)
}

/// The batch axis of a launch with `programs_per_request` programs per
/// request: every batch up to [`DENSE_PROGRAMS`] programs, then powers of two
/// up to 512.
pub(crate) fn batch_axis(programs_per_request: u32) -> Vec<f64> {
    let dense = (DENSE_PROGRAMS / programs_per_request.max(1)).clamp(1, 512);
    let mut batches: Vec<f64> = (1..=dense).map(f64::from).collect();
    batches.extend(
        Axis::pow2(0, 9)
            .into_iter()
            .filter(|&b| b > f64::from(dense)),
    );
    batches
}

pub struct KdaRecoverSSMVerifySpec;

impl KernelSpec for KdaRecoverSSMVerifySpec {
    type Config = KdaRecoverSSMVerifyKernelConfig;
    type Input = KdaRecoverSSMVerifyKernelInput;

    const KIND: KernelKind = "kda_recoverssm_verify";

    /// Every batch while the launch is a few waves, then powers of two to 512
    /// (see [`batch_axis`]).
    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let programs = value_blocks(config.head_dim.get()) * config.num_heads.get();
        SweepGrid::new(vec![batch_axis(programs)])
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
                .with("head_dim", config.head_dim.get())
                .with("spec_query_len", config.spec_query_len.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(KdaRecoverSSMVerifyKernel, KdaRecoverSSMVerifySpec);

#[cfg(test)]
mod tests {
    use super::{
        KdaRecoverSSMVerifyKernelConfig, KdaRecoverSSMVerifyKernelInput, KdaRecoverSSMVerifySpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> KdaRecoverSSMVerifyKernelConfig {
        KdaRecoverSSMVerifyKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads: 8.into(),
            head_dim: 128.into(),
            spec_query_len: 8.into(),
            dtype: DType::Bf16,
        }
    }

    /// Catches a dtype tag drift that would hide the kind from launcher
    /// backend selection (Python BackendSupport is compute bf16, no KV axis).
    #[test]
    fn dtype_tag_is_compute_only() {
        let cfg = config();
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
    }

    /// Catches payload keys that would miss the Python KernelArgs DB key, or a
    /// window length that is not forwarded from the config.
    #[test]
    fn enumerate_emits_exact_python_fields() {
        let cfg = config();
        let grid = KdaRecoverSSMVerifySpec::sweep_grid(&cfg);
        let payloads = KdaRecoverSSMVerifySpec::enumerate(&cfg, &grid, "vllm_triton");
        assert_eq!(payloads.len(), 130);
        for payload in &payloads {
            let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
            assert_eq!(
                names,
                [
                    "backend",
                    "batch_size",
                    "dtype",
                    "head_dim",
                    "num_heads",
                    "spec_query_len"
                ]
            );
        }
        let last = payloads[129].fields();
        assert_eq!(last["batch_size"], Value::from(512_u32));
        assert_eq!(last["spec_query_len"], Value::from(8_u32));
        assert_eq!(last["num_heads"], Value::from(8_u32));
    }

    /// Catches a batch axis that skips the few-wave batches where the time
    /// steps (8 heads: every batch to 128; 64 heads: to 16), or that loses the
    /// power-of-two tail to 512.
    #[test]
    fn batch_axis_is_dense_while_the_launch_is_shallow() {
        let grid = KdaRecoverSSMVerifySpec::sweep_grid(&config());
        let axis = &grid.axes()[0];
        assert_eq!(axis.len(), 130);
        assert_eq!(&axis[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&axis[127..], &[128.0, 256.0, 512.0]);
        let wide = KdaRecoverSSMVerifyKernelConfig {
            num_heads: 64.into(),
            ..config()
        };
        let grid = KdaRecoverSSMVerifySpec::sweep_grid(&wide);
        let axis = &grid.axes()[0];
        assert_eq!(axis.len(), 21);
        assert_eq!(&axis[15..], &[16.0, 32.0, 64.0, 128.0, 256.0, 512.0]);
    }

    /// Catches an Input that does not project batch_size onto the cache axis
    /// or does not reach the cost-log slot payload.
    #[test]
    fn input_projects_batch_size_and_logs_it() {
        let input = KdaRecoverSSMVerifyKernelInput { batch_size: 24 };
        assert_eq!(&*input.coords(), &[24.0]);
        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"batch_size": 24})
        );
    }
}
