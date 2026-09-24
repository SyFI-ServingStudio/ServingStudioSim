//! GLM-5.3-Flash KDA plain-decode `fused_recurrent_kda` call.
//!
//! One semantic call: 4 `.contiguous()` copies of q/k/v/beta plus the KDA-mode
//! fused recurrent kernel (5 launches) for batch >= 2, and a single launch at
//! batch 1 where every view is already contiguous. Head geometry and dtype are
//! static identity; decode batch size is the only runtime interpolation axis.
//! The fp32 recurrent state is fixed by the model, so it is not a key.

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
    pub head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
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

    /// Batch 1..512 in powers of two. Batch 1 stays a measured point because
    /// it has no copy launches and so sits off the batch >= 2 line.
    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::pow2(0, 9)])
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
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(KdaRecurrentDecodeKernel, KdaRecurrentDecodeSpec);

#[cfg(test)]
mod tests {
    use super::{
        KdaRecurrentDecodeKernelConfig, KdaRecurrentDecodeKernelInput, KdaRecurrentDecodeSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> KdaRecurrentDecodeKernelConfig {
        KdaRecurrentDecodeKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads: 16.into(),
            head_dim: 128.into(),
            dtype: DType::Bf16,
        }
    }

    /// Catches a dtype tag drift that would hide the kind from launcher
    /// backend selection (Python BackendSupport is compute bf16, no KV axis).
    #[test]
    fn dtype_tag_is_compute_only() {
        let cfg = config();
        assert_eq!(KdaRecurrentDecodeSpec::KIND, "kda_recurrent_decode");
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);
    }

    /// Catches a grid that drops the batch-1 (no-copy) point or exceeds the
    /// frozen 10-point feasible budget.
    #[test]
    fn grid_is_pow2_1_to_512_with_batch_one_and_ten_feasible_points() {
        let cfg = config();
        let grid = KdaRecurrentDecodeSpec::sweep_grid(&cfg);
        assert_eq!(grid.axes().len(), 1);
        assert_eq!(
            grid.axes()[0],
            [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]
        );
        assert!(KdaRecurrentDecodeSpec::infeasible_mask(&cfg, &grid).is_empty());
        assert_eq!(
            KdaRecurrentDecodeSpec::enumerate(&cfg, &grid, "vllm_triton").len(),
            10
        );
        assert_eq!(
            KdaRecurrentDecodeSpec::cache_kind("vllm_triton"),
            CacheKind::Cache1DLinear
        );
    }

    /// Catches payload keys that would miss the Python KernelArgs DB key.
    #[test]
    fn enumerate_emits_exact_python_fields() {
        let cfg = config();
        let grid = KdaRecurrentDecodeSpec::sweep_grid(&cfg);
        let payloads = KdaRecurrentDecodeSpec::enumerate(&cfg, &grid, "vllm_triton");
        for payload in &payloads {
            let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
            assert_eq!(
                names,
                ["backend", "batch_size", "dtype", "head_dim", "num_heads"]
            );
            assert_eq!(payload.backend(), Some("vllm_triton"));
        }
        let first = payloads[0].fields();
        assert_eq!(first["batch_size"], Value::from(1_u32));
        assert_eq!(first["num_heads"], Value::from(16_u32));
        assert_eq!(first["head_dim"], Value::from(128_u32));
        assert_eq!(first["dtype"], Value::from("bf16"));
        assert_eq!(payloads[9].fields()["batch_size"], Value::from(512_u32));
    }

    /// Catches an Input that does not project batch_size onto the cache axis
    /// or does not reach the cost-log slot payload.
    #[test]
    fn input_projects_batch_size_and_logs_it() {
        let input = KdaRecurrentDecodeKernelInput { batch_size: 24 };
        assert_eq!(&*input.coords(), &[24.0]);
        assert_eq!(
            KdaRecurrentDecodeKernelInput::coord_field_names(),
            &["batch_size"]
        );
        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"batch_size": 24})
        );
    }
}
