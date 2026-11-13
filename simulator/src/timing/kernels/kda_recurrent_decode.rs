//! KDA `fused_recurrent_kda` call: plain decode, or a speculative verify.
//!
//! One semantic call: 4 `.contiguous()` copies of q/k/v/beta plus the KDA-mode
//! fused recurrent kernel (5 launches) for batch >= 2, and a single launch at
//! batch 1 where every view is already contiguous. Head geometry, dtype, the
//! state pool dtype and a verify's tokens per request are static identity;
//! the request count is the only runtime interpolation axis. A verify
//! (`tokens_per_request` q > 1) is the same call with `[B, q]` state slots and
//! `num_accepted_tokens`: it reads each request's last accepted state once and
//! writes a full state after every token (profile arg `num_tokens = q · B`).

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::gdn_recurrent_decode::{verify_batch_axis, verify_tokens_per_request};
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
    /// Element type of the recurrent state pool (fp32 or bf16). Not a KV
    /// capability axis; the Python arg treats fp32 as its absent value, so the
    /// fp32 rows keep their keys.
    #[serde(default = "fp32_state")]
    pub state_dtype: DType,
    /// Tokens per request of a speculative verify. Unset, or 1, is one-token
    /// decode; q > 1 adds the profile arg `num_tokens = q · batch_size`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_per_request: Option<Dim>,
}

fn fp32_state() -> DType {
    DType::Fp32
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
    /// it has no copy launches and so sits off the batch >= 2 line. A verify
    /// takes the verify axis (1..8, then pow2 to 256).
    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        if verify_tokens_per_request(config.tokens_per_request.as_ref()).is_some() {
            return SweepGrid::new(vec![verify_batch_axis()]);
        }
        // Powers of two plus 12 and 24: the batch-8..32 curve is concave (the
        // four copy launches and the recurrent kernel fill the SMs at different
        // batch sizes; B200 job 1138), so pow2 alone misses batch 12/24 by
        // 11-16%.
        let mut batches = Axis::pow2(0, 9);
        batches.extend([12.0, 24.0]);
        batches.sort_by(f64::total_cmp);
        SweepGrid::new(vec![batches])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        let per_request = verify_tokens_per_request(config.tokens_per_request.as_ref());
        grid.expand_1d(|batch_size| {
            let payload = ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", batch_size as u32)
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("dtype", config.dtype.as_str())
                .with("state_dtype", config.state_dtype.as_str());
            match per_request {
                Some(q) => payload.with("num_tokens", batch_size as u32 * q),
                None => payload,
            }
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
            state_dtype: DType::Fp32,
            tokens_per_request: None,
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

    /// Catches a grid that drops the batch-1 (no-copy) point or the measured
    /// 12/24 curvature points, or grows past its 12-point feasible budget.
    #[test]
    fn grid_is_pow2_1_to_512_plus_12_and_24_with_twelve_feasible_points() {
        let cfg = config();
        let grid = KdaRecurrentDecodeSpec::sweep_grid(&cfg);
        assert_eq!(grid.axes().len(), 1);
        assert_eq!(
            grid.axes()[0],
            [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0, 128.0, 256.0, 512.0]
        );
        assert!(KdaRecurrentDecodeSpec::infeasible_mask(&cfg, &grid).is_empty());
        assert_eq!(
            KdaRecurrentDecodeSpec::enumerate(&cfg, &grid, "vllm_triton").len(),
            12
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
                [
                    "backend",
                    "batch_size",
                    "dtype",
                    "head_dim",
                    "num_heads",
                    "state_dtype"
                ]
            );
            assert_eq!(payload.backend(), Some("vllm_triton"));
        }
        let first = payloads[0].fields();
        assert_eq!(first["batch_size"], Value::from(1_u32));
        assert_eq!(first["num_heads"], Value::from(16_u32));
        assert_eq!(first["head_dim"], Value::from(128_u32));
        assert_eq!(first["dtype"], Value::from("bf16"));
        assert_eq!(payloads[11].fields()["batch_size"], Value::from(512_u32));
    }

    /// Catches a verify keyed as one-token decode rows (no num_tokens, the
    /// pow2 decode axis) or a bf16 pool keyed as the fp32 rows.
    #[test]
    fn a_verify_keys_tokens_per_request_and_the_pool_dtype() {
        let mut cfg = config();
        cfg.tokens_per_request = Some(8.into());
        cfg.state_dtype = DType::Bf16;
        let grid = KdaRecurrentDecodeSpec::sweep_grid(&cfg);
        assert_eq!(
            grid.axes()[0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
        );
        let payloads = KdaRecurrentDecodeSpec::enumerate(&cfg, &grid, "vllm_triton");
        let row = payloads[9].fields();
        assert_eq!(row["batch_size"], Value::from(32_u32));
        assert_eq!(row["num_tokens"], Value::from(256_u32));
        assert_eq!(row["state_dtype"], Value::from("bf16"));
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
