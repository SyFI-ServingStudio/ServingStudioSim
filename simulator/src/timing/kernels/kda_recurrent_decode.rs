//! GLM-5.3-Flash KDA `fused_recurrent_kda` call: plain decode or spec verify.
//!
//! One semantic call. Head geometry and dtype are static identity, and so is a
//! speculative step's tokens per request (`tokens_per_request`, q = accepted
//! token + drafts: the spec branch of `kda.py`, with `[B, q]` state slots and
//! `num_accepted_tokens`). Decode batch size (requests) is the only runtime
//! interpolation axis. The fp32 recurrent state is fixed by the model, so it is
//! not a key.

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
    /// Tokens per request of a speculative verify step. Unset, or 1, is
    /// one-token decode and keys the rows it keyed before the field existed;
    /// q > 1 adds the profile arg `num_tokens = q · batch_size`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_per_request: Option<Dim>,
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

    /// One-token decode: batch 1..512 in powers of two plus 12 and 24. Batch 1
    /// stays a measured point because it sits off the batch >= 2 line. A verify
    /// step (q > 1): the shared verify axis (1..8 dense, then pow2 to 256, the
    /// deployment's max_num_seqs) plus 12 and 24.
    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        if verify_tokens_per_request(config.tokens_per_request.as_ref()).is_some() {
            let mut batches = verify_batch_axis();
            batches.extend([12.0, 24.0]);
            batches.sort_by(f64::total_cmp);
            return SweepGrid::new(vec![batches]);
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
                .with("dtype", config.dtype.as_str());
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
                ["backend", "batch_size", "dtype", "head_dim", "num_heads"]
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

    /// A verify config must reach the profiler as `num_tokens = q · batch_size`
    /// over the verify axis; q = 1 must keep the one-token rows' keys and grid.
    #[test]
    fn verify_tokens_per_request_emits_num_tokens_and_q1_keeps_old_keys() {
        let mut cfg = config();
        cfg.tokens_per_request = Some(4.into());
        let grid = KdaRecurrentDecodeSpec::sweep_grid(&cfg);
        assert_eq!(
            grid.axes()[0],
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0, 128.0,
                256.0
            ]
        );
        let payloads = KdaRecurrentDecodeSpec::enumerate(&cfg, &grid, "vllm_triton");
        assert_eq!(payloads.len(), 15);
        assert_eq!(payloads[2].fields()["batch_size"], Value::from(3_u32));
        assert_eq!(payloads[2].fields()["num_tokens"], Value::from(12_u32));

        cfg.tokens_per_request = Some(1.into());
        let one_grid = KdaRecurrentDecodeSpec::sweep_grid(&cfg);
        assert_eq!(one_grid.axes(), KdaRecurrentDecodeSpec::sweep_grid(&config()).axes());
        let one = KdaRecurrentDecodeSpec::enumerate(&cfg, &one_grid, "vllm_triton");
        assert!(one.iter().all(|p| !p.fields().contains_key("num_tokens")));
        // Unset serializes without the field, so the q = 1 config's identity
        // (and its cache key) is the one it had before the field existed.
        let encoded = serde_json::to_value(config()).unwrap();
        assert!(encoded.get("tokens_per_request").is_none());
    }
}
