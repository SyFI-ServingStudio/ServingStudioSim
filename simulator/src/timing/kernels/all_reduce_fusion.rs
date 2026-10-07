//! FlashInfer TRT-LLM standalone all-reduce used by vLLM's TP boundaries.
//!
//! This is shape-keyed rather than byte-keyed: vLLM passes a contiguous
//! `[num_tokens, hidden_dim]` tensor and changes the PDL completion policy at
//! 16 tokens. The generic `all_reduce` kind cannot represent either property.
//!
//! The `flashinfer_mnnvl` backend (FlashInfer >= 0.6.18 MNNVL all-reduce, the
//! vLLM `auto` choice on B200) switches from one-shot to two-shot inside the
//! call once `num_tokens * hidden_dim * num_gpus * elem_size` exceeds 1 MiB, so
//! its grid carries the last one-shot and the first two-shot token count.
//!
//! Above the workspace cap vLLM routes to a different all-reduce, so the grid
//! stops at the cap. No mask or clamp is applied: a query above the cap
//! extrapolates linearly from the last two-shot segment and is flagged
//! `EXTRAPOLATED` rather than silently pinned to the cap time.
//!
//! BARRIER (Infinity Fabric / MI300X only). The `rocm_fabric_roofline` row
//! (`profiling/runners/comm/fabric_roofline.py`) is a pure bandwidth roofline:
//! `2(N-1)/N · bytes / 896 GB/s`, the data-movement floor. A bandwidth roofline
//! cannot represent the RCCL/PYNCCL cross-rank sync barrier — the real
//! `disable_custom_all_reduce` path's per-collective launch + link + sync
//! latency, which is independent of message size. `adjust_metrics` adds that
//! barrier on top of the cached roofline for `Fabric::InfinityFabric`
//! (see `INFINITY_FABRIC_BARRIER_US_PER_RING_STEP`). Keeping the barrier in the
//! cost model, not in the profiled row, keeps the row a clean roofline and
//! avoids double-counting if it is regenerated. NVLink/B200 is untouched: its
//! `flashinfer_mnnvl` row is a real measurement that already includes sync.

use crate::common::Fabric;
use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::cache::interp::LeafMetrics;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

/// Infinity-Fabric all-reduce cross-rank sync-barrier latency, in microseconds
/// per ring step, added on top of the bandwidth roofline (see the module doc's
/// BARRIER note). A ring all-reduce runs `2(num_gpus - 1)` sequential
/// send/recv steps; under vLLM-ROCm's `disable_custom_all_reduce` the collective
/// runs on PYNCCL/RCCL, whose per-step kernel-launch + link + sync latency does
/// not scale with message size. Calibrated to the first real measured MI300X
/// TP4/EP4 Check-1 (`servingstudio-mi300x-decisions.md` decision #103): the
/// measured prefill all-reduce (2048 tokens) and decode all-reduce (32 tokens)
/// are within ~20% of each other despite a 64x byte difference, i.e. the cost is
/// barrier- not bandwidth-bound, which a bandwidth roofline alone cannot show.
/// At TP4 the 6 ring steps sum to ~94.2 us/collective, reproducing the measured
/// 11,126 us prefill all-reduce (91 collectives) to within 0.02%; the same
/// constant predicts the measured 9,135 us decode all-reduce to within ~6%.
const INFINITY_FABRIC_BARRIER_US_PER_RING_STEP: f32 = 15.70;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AllReduceFusionKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_gpus: u32,
    pub hidden_dim: u32,
    #[compute_dtype]
    pub dtype: DType,
    pub fabric: Fabric,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct AllReduceFusionKernelInput {
    pub num_tokens: u32,
}

pub struct AllReduceFusionSpec;

impl AllReduceFusionSpec {
    /// vLLM's default FlashInfer workspace cap on SM100.
    fn max_fused_bytes(num_gpus: u32) -> u64 {
        let mib = match num_gpus {
            2 => 64,
            4 => 32,
            8 => 1,
            _ => panic!("FlashInfer all-reduce fusion supports TP 2/4/8"),
        };
        mib * 1024 * 1024
    }

    /// Largest token count for which vLLM selects FlashInfer on B200.
    ///
    /// The cap formula (`max_fused_bytes(num_gpus) / bytes_per_token`) is
    /// hardware-independent, so the grid is well-defined for any target; the
    /// gate only records which GPUs this fusion has actually been profiled on.
    /// On MI300X the backend is `rocm_fabric_roofline`, the analytic
    /// Infinity-Fabric ring all-reduce (`profiling/runners/comm/fabric_roofline.py`,
    /// decision #42), not FlashInfer MNNVL. That roofline is linear in num_tokens,
    /// so this FlashInfer-derived cap only bounds the sweep's upper token count;
    /// `Cache1DLinear` interpolates within it and extrapolates linearly above,
    /// which is exact for the roofline. B200 is unchanged.
    pub fn max_fused_tokens(config: &AllReduceFusionKernelConfig) -> u32 {
        assert!(
            config.gpu_name.contains("B200") || config.gpu_name.contains("MI300"),
            "all_reduce_fusion is currently profiled only on B200 and MI300X"
        );
        let bytes_per_token = u64::from(config.hidden_dim)
            .checked_mul(config.dtype.size_bytes() as u64)
            .expect("all-reduce bytes per token overflow");
        (Self::max_fused_bytes(config.num_gpus) / bytes_per_token) as u32
    }

    /// FlashInfer `MNNVL_ONE_SHOT_THRESHOLD` (`trtllm_mnnvl_ar.py`): one-shot
    /// iff the all-gathered payload `T * H * N * elem` is at most 1 MiB.
    const MNNVL_ONE_SHOT_BYTES: u64 = 1024 * 1024;

    /// Largest token count for which the MNNVL backend runs one-shot.
    pub fn mnnvl_max_oneshot_tokens(config: &AllReduceFusionKernelConfig) -> u32 {
        let bytes_per_token = u64::from(config.hidden_dim)
            * u64::from(config.num_gpus)
            * config.dtype.size_bytes() as u64;
        (Self::MNNVL_ONE_SHOT_BYTES / bytes_per_token) as u32
    }
}

impl KernelSpec for AllReduceFusionSpec {
    type Config = AllReduceFusionKernelConfig;
    type Input = AllReduceFusionKernelInput;

    const KIND: KernelKind = "all_reduce_fusion";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let max_fused_tokens = Self::max_fused_tokens(config);
        let mut boundaries = vec![max_fused_tokens];
        // Only MNNVL has the in-call one-shot/two-shot switch; keeping it off
        // the trtllm grid avoids resampling that backend's existing rows.
        if config.backends.contains(&"flashinfer_mnnvl") {
            let last_oneshot = Self::mnnvl_max_oneshot_tokens(config);
            if last_oneshot > 0 {
                boundaries.extend([last_oneshot, last_oneshot + 1]);
            }
        }
        let mut tokens = Axis::chain([
            Axis::values([1, 2, 4, 8, 16, 17]),
            Axis::token_axis(),
            Axis::values(boundaries),
        ]);
        tokens.retain(|num_tokens| *num_tokens <= f64::from(max_fused_tokens));
        tokens.sort_by(f64::total_cmp);
        SweepGrid::new(vec![tokens])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    /// Add the Infinity-Fabric cross-rank sync-barrier latency on top of the
    /// cached bandwidth roofline. Gated to `Fabric::InfinityFabric` with more
    /// than one rank, so every NVLink/B200 all-reduce is bit-identical (its row
    /// is a real FlashInfer MNNVL measurement that already includes launch/sync,
    /// so no correction applies). A single rank is a no-op collective (0 steps).
    fn adjust_metrics(
        config: &Self::Config,
        _input: &Self::Input,
        mut metrics: LeafMetrics,
    ) -> LeafMetrics {
        if config.fabric == Fabric::InfinityFabric && config.num_gpus > 1 {
            let ring_steps = 2 * (config.num_gpus - 1);
            let barrier_ms =
                INFINITY_FABRIC_BARRIER_US_PER_RING_STEP * ring_steps as f32 / 1000.0;
            metrics.m.time_ms += barrier_ms;
        }
        metrics
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_gpus", config.num_gpus)
                .with("num_tokens", num_tokens as u32)
                .with("hidden_dim", config.hidden_dim)
                .with("dtype", config.dtype.as_str())
                .with("fabric", config.fabric.as_str())
        })
    }
}

register_kernel!(AllReduceFusionKernel, AllReduceFusionSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config(num_gpus: u32) -> AllReduceFusionKernelConfig {
        AllReduceFusionKernelConfig {
            backends: vec!["flashinfer_trtllm"],
            gpu_name: "NVIDIA B200".to_string(),
            num_gpus,
            hidden_dim: 6144,
            dtype: DType::Bf16,
            fabric: Fabric::Nvlink,
        }
    }

    #[test]
    fn config_and_input_identity_are_explicit() {
        let config = config(4);
        assert_eq!(config.compute_dtype(), Some(DType::Bf16));
        assert_eq!(config.backends(), &["flashinfer_trtllm"]);
        assert_eq!(config.describe_config()["hidden_dim"], 6144);
        let input = AllReduceFusionKernelInput { num_tokens: 8 };
        assert_eq!(&*input.coords(), &[8.0]);
        assert_eq!(
            AllReduceFusionKernelInput::coord_field_names(),
            &["num_tokens"]
        );
    }

    #[test]
    fn sweep_keeps_completion_boundary_and_workspace_cap() {
        let tp4 = AllReduceFusionSpec::sweep_grid(&config(4));
        assert!(tp4.axes()[0].contains(&16.0));
        assert!(tp4.axes()[0].contains(&17.0));
        assert_eq!(tp4.axes()[0].last(), Some(&2730.0));

        let tp8 = AllReduceFusionSpec::sweep_grid(&config(8));
        assert_eq!(tp8.axes()[0].last(), Some(&85.0));
    }

    fn mnnvl_config(num_gpus: u32) -> AllReduceFusionKernelConfig {
        AllReduceFusionKernelConfig {
            backends: vec!["flashinfer_mnnvl"],
            hidden_dim: 4096,
            ..config(num_gpus)
        }
    }

    #[test]
    fn trtllm_grid_has_no_mnnvl_strategy_points() {
        let tp4 = AllReduceFusionSpec::sweep_grid(&config(4));
        assert!(!tp4.axes()[0].contains(&21.0));
        assert!(!tp4.axes()[0].contains(&22.0));
    }

    #[test]
    fn mnnvl_sweep_brackets_oneshot_switch_and_stops_at_cap() {
        let config = mnnvl_config(4);
        assert_eq!(AllReduceFusionSpec::mnnvl_max_oneshot_tokens(&config), 32);
        assert_eq!(AllReduceFusionSpec::max_fused_tokens(&config), 4096);

        let grid = AllReduceFusionSpec::sweep_grid(&config);
        let tokens = &grid.axes()[0];
        for t in [16.0, 17.0, 32.0, 33.0] {
            assert!(tokens.contains(&t), "missing {t}");
        }
        assert_eq!(tokens.last(), Some(&4096.0));
        assert!(tokens.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(tokens.len(), 41);

        let tp8 = AllReduceFusionSpec::sweep_grid(&mnnvl_config(8));
        assert!(tp8.axes()[0].contains(&16.0) && tp8.axes()[0].contains(&17.0));
        assert_eq!(tp8.axes()[0].last(), Some(&128.0));
    }

    #[test]
    fn mnnvl_enumerate_matches_python_schema() {
        let config = mnnvl_config(4);
        let grid = AllReduceFusionSpec::sweep_grid(&config);
        let payloads = AllReduceFusionSpec::enumerate(&config, &grid, "flashinfer_mnnvl");
        assert_eq!(payloads.len(), grid.axes()[0].len());
        let fields = payloads[0].fields();
        assert_eq!(fields.len(), 6);
        assert_eq!(
            fields.get("backend"),
            Some(&Value::from("flashinfer_mnnvl"))
        );
        assert_eq!(fields.get("hidden_dim"), Some(&Value::from(4096_u32)));
        assert_eq!(fields.get("fabric"), Some(&Value::from("nvlink")));
    }

    fn roofline_config(num_gpus: u32, fabric: Fabric) -> AllReduceFusionKernelConfig {
        AllReduceFusionKernelConfig {
            backends: vec!["rocm_fabric_roofline"],
            gpu_name: "MI300X".to_string(),
            num_gpus,
            hidden_dim: 4096,
            dtype: DType::Bf16,
            fabric,
        }
    }

    #[test]
    fn infinity_fabric_adds_sync_barrier_on_top_of_roofline() {
        use crate::timing::cache::interp::LeafMetrics;
        // Bandwidth roofline at 2048 tokens, TP4 (matches the profiled row).
        let roofline_ms = 0.028_086_857_f32;
        let mut m = LeafMetrics::ZERO;
        m.m.time_ms = roofline_ms;
        let input = AllReduceFusionKernelInput { num_tokens: 2048 };

        // TP4 Infinity Fabric: 6 ring steps add ~94.2 us → ~122.3 us/collective.
        let adjusted =
            AllReduceFusionSpec::adjust_metrics(&roofline_config(4, Fabric::InfinityFabric), &input, m);
        let expected =
            roofline_ms + INFINITY_FABRIC_BARRIER_US_PER_RING_STEP * 6.0 / 1000.0;
        assert!((adjusted.m.time_ms - expected).abs() < 1e-6);
        // 91 collectives reproduce the measured 11,126 us prefill all-reduce.
        let total_us = f64::from(adjusted.m.time_ms) * 1000.0 * 91.0;
        assert!((total_us - 11_125.9).abs() < 50.0, "got {total_us} us");
    }

    #[test]
    fn nvlink_and_single_rank_are_untouched() {
        use crate::timing::cache::interp::LeafMetrics;
        let mut m = LeafMetrics::ZERO;
        m.m.time_ms = 0.05;
        let input = AllReduceFusionKernelInput { num_tokens: 2048 };

        // NVLink (B200 path) gets no barrier — bit-identical.
        let nvlink =
            AllReduceFusionSpec::adjust_metrics(&roofline_config(4, Fabric::Nvlink), &input, m);
        assert_eq!(nvlink.m.time_ms, 0.05);

        // A single rank is a no-op collective: no barrier even on Infinity Fabric.
        let one_rank =
            AllReduceFusionSpec::adjust_metrics(&roofline_config(1, Fabric::InfinityFabric), &input, m);
        assert_eq!(one_rank.m.time_ms, 0.05);
    }

    #[test]
    fn enumerate_matches_python_schema() {
        let config = config(4);
        let grid = AllReduceFusionSpec::sweep_grid(&config);
        let payloads = AllReduceFusionSpec::enumerate(&config, &grid, "flashinfer_trtllm");
        let fields = payloads[0].fields();

        assert_eq!(fields.len(), 6);
        assert_eq!(
            fields.get("backend"),
            Some(&Value::from("flashinfer_trtllm"))
        );
        assert_eq!(fields.get("num_gpus"), Some(&Value::from(4_u32)));
        assert_eq!(fields.get("num_tokens"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("hidden_dim"), Some(&Value::from(6144_u32)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("fabric"), Some(&Value::from("nvlink")));
    }
}
