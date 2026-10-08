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

use crate::common::gpu::compute_capability;
use crate::common::Fabric;
use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

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
    /// An engine's own fused-token limit, replacing vLLM's workspace-size
    /// rule when set: SGLang fuses any batch of at most 2048 tokens whatever
    /// its byte size (`layers/communicator.py:162-179`). `None` (vLLM) is left
    /// out of the config identity, so vLLM configs keep their identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused_token_limit: Option<u32>,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct AllReduceFusionKernelInput {
    pub num_tokens: u32,
}

pub struct AllReduceFusionSpec;

/// vLLM's default FlashInfer fused all-reduce workspace, in bytes, by the GPU's
/// compute capability and TP (`FI_ALLREDUCE_FUSION_MAX_SIZE_MB` in
/// `vllm/compilation/passes/fusion/allreduce_rms_fusion.py`). vLLM keys it by
/// the exact capability, so SM103 (B300) is not SM100 (B200, GB200). `None`
/// where vLLM does not fuse: a TP or GPU with no row, or a GPU with no compute
/// capability in gpu/spec.json. Both fused all-reduce kinds read it. The TP16
/// rows are vLLM's multi-node MNNVL, which no runner measures
/// ([`runner_tp_sizes`]).
fn flashinfer_fusion_workspace(gpu_name: &str, num_gpus: u32) -> Option<u64> {
    let kib: u64 = match (compute_capability(gpu_name)?, num_gpus) {
        ((9, 0), 2) => 64 * 1024,
        ((9, 0), 4) => 2 * 1024,
        ((9, 0), 8) => 512,
        ((10, 0), 2) => 64 * 1024,
        ((10, 0), 4) => 32 * 1024,
        ((10, 0), 8) => 1024,
        ((10, 0), 16) => 64 * 1024,
        ((10, 3), 2 | 4 | 16) => 64 * 1024,
        ((10, 3), 8) => 4 * 1024,
        ((10, 7), 2 | 4) => 64 * 1024,
        ((10, 7), 8) => 2 * 1024,
        _ => return None,
    };
    Some(kib * 1024)
}

/// The TPs a FlashInfer fused all-reduce backend's runner measures, one node's
/// NVLink domain: `SUPPORTED_TP_SIZES` in
/// `profiling/runners/comm/flashinfer_trtllm.py` (both fused kinds) and the
/// keys of `SM100_MAX_SIZE_MB` in `flashinfer_mnnvl.py`. A test reads both.
fn runner_tp_sizes(backend: &str) -> &'static [u32] {
    match backend {
        "flashinfer_trtllm" | "flashinfer_mnnvl" => &[2, 4, 8],
        other => panic!("{other} is not a FlashInfer fused all-reduce backend"),
    }
}

/// Why a FlashInfer fused all-reduce with `backends` over `num_gpus` GPUs of
/// `gpu_name` cannot be modeled, if it cannot: a TP a backend's runner does
/// not measure, or one vLLM has no FlashInfer budget for on that GPU. An arch
/// that builds a fused all-reduce from its TP param checks this first, so it
/// refuses the TP instead of building a grid no runner can fill.
pub fn fused_all_reduce_refusal(
    backends: &[&str],
    gpu_name: &str,
    num_gpus: u32,
) -> Option<String> {
    if let Some(backend) = backends
        .iter()
        .find(|backend| !runner_tp_sizes(backend).contains(&num_gpus))
    {
        return Some(format!(
            "the {backend} fused all-reduce is measured at TP {:?}, not {num_gpus}",
            runner_tp_sizes(backend)
        ));
    }
    flashinfer_fusion_workspace(gpu_name, num_gpus)
        .is_none()
        .then(|| {
            format!("vLLM has no FlashInfer all-reduce budget for {gpu_name} at TP {num_gpus}")
        })
}

/// [`flashinfer_fusion_workspace`] for a fused kernel config, which exists
/// only where vLLM fuses.
pub(crate) fn flashinfer_fusion_max_bytes(gpu_name: &str, num_gpus: u32) -> u64 {
    flashinfer_fusion_workspace(gpu_name, num_gpus).unwrap_or_else(|| {
        panic!(
            "vLLM has no FlashInfer all-reduce fusion workspace for {gpu_name} at TP \
             {num_gpus} (or gpu/spec.json has no compute capability for it)"
        )
    })
}

/// The fused all-reduce workspace cap in bytes for an AMD Infinity-Fabric
/// target, keyed only by TP group size. MI300X runs the analytic Infinity-Fabric
/// ring all-reduce roofline (`rocm_fabric_roofline`, decision #42), which has no
/// FlashInfer workspace and no CUDA compute capability, so it cannot read
/// [`flashinfer_fusion_workspace`]. The roofline is linear in `num_tokens`, so
/// this cap only bounds the sweep's upper token count: `Cache1DLinear`
/// interpolates within it and extrapolates linearly above, which is exact for
/// the roofline. The TP->MiB values are vLLM's SM100 caps, kept so the MI300X
/// sweep's token range matches the measured rows.
fn infinity_fabric_fused_max_bytes(num_gpus: u32) -> u64 {
    let mib: u64 = match num_gpus {
        2 => 64,
        4 => 32,
        8 => 1,
        _ => panic!("fused all-reduce supports TP 2/4/8, not {num_gpus}"),
    };
    mib * 1024 * 1024
}

/// The fused all-reduce workspace cap in bytes for a config, vendor-aware by its
/// `fabric`. NVIDIA parts (NVLink) read vLLM's capability-keyed FlashInfer
/// budget; AMD Infinity Fabric uses the capability-free cap above, so a target
/// with no CUDA compute capability does not fall into the FlashInfer panic.
/// Branching on the fabric (not the GPU name) keeps every NVLink config
/// byte-identical.
pub(crate) fn fused_workspace_max_bytes(gpu_name: &str, num_gpus: u32, fabric: Fabric) -> u64 {
    match fabric {
        Fabric::InfinityFabric => infinity_fabric_fused_max_bytes(num_gpus),
        _ => flashinfer_fusion_max_bytes(gpu_name, num_gpus),
    }
}

impl AllReduceFusionSpec {
    /// Largest token count the engine runs through the fused all-reduce:
    /// `fused_token_limit` when set, else the vendor's workspace cap
    /// ([`fused_workspace_max_bytes`]).
    pub fn max_fused_tokens(config: &AllReduceFusionKernelConfig) -> u32 {
        if let Some(limit) = config.fused_token_limit {
            return limit;
        }
        let bytes_per_token = u64::from(config.hidden_dim)
            .checked_mul(config.dtype.size_bytes() as u64)
            .expect("all-reduce bytes per token overflow");
        (fused_workspace_max_bytes(&config.gpu_name, config.num_gpus, config.fabric)
            / bytes_per_token) as u32
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
            fused_token_limit: None,
        }
    }

    #[test]
    fn an_engine_token_limit_replaces_the_workspace_cap() {
        let sglang = AllReduceFusionKernelConfig {
            fused_token_limit: Some(2048),
            ..config(8)
        };
        assert_eq!(AllReduceFusionSpec::max_fused_tokens(&sglang), 2048);
        assert_eq!(
            AllReduceFusionSpec::sweep_grid(&sglang).axes()[0].last(),
            Some(&2048.0)
        );
        // vLLM configs keep their identity: the unset limit is not serialized.
        assert!(config(8).identity().get("fused_token_limit").is_none());
        assert_eq!(sglang.identity()["fused_token_limit"], 2048);
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
    fn workspace_follows_vllm_by_exact_compute_capability() {
        const MIB: u64 = 1024 * 1024;
        // SM100 (B200, GB200) and SM103 (B300) are separate vLLM rows.
        assert_eq!(flashinfer_fusion_max_bytes("NVIDIA B200", 4), 32 * MIB);
        assert_eq!(flashinfer_fusion_max_bytes("NVIDIA GB200", 8), MIB);
        assert_eq!(flashinfer_fusion_max_bytes("NVIDIA B300", 4), 64 * MIB);
        assert_eq!(flashinfer_fusion_max_bytes("NVIDIA B300", 8), 4 * MIB);
        assert_eq!(flashinfer_fusion_max_bytes("NVIDIA H200", 8), MIB / 2);
    }

    /// The TPs and the SM100 budget this module models are the ones the
    /// runners measure, read from the runners themselves.
    #[test]
    fn the_runners_measure_the_tps_and_sm100_budget_modeled_here() {
        use pyo3::prelude::*;
        use std::collections::BTreeMap;
        const MIB: f64 = 1024.0 * 1024.0;
        Python::with_gil(|py| {
            // The repository root, where `profiling` lives.
            py.import("sys")
                .unwrap()
                .getattr("path")
                .unwrap()
                .call_method1("insert", (0, env!("CARGO_MANIFEST_DIR")))
                .unwrap();
            let trtllm = PyModule::import(py, "profiling.runners.comm.flashinfer_trtllm").unwrap();
            let sorted = py.import("builtins").unwrap().getattr("sorted").unwrap();
            let tps: Vec<u32> = sorted
                .call1((trtllm.getattr("SUPPORTED_TP_SIZES").unwrap(),))
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(tps, runner_tp_sizes("flashinfer_trtllm"));
            let mnnvl = PyModule::import(py, "profiling.runners.comm.flashinfer_mnnvl").unwrap();
            let budget: BTreeMap<u32, f64> = mnnvl
                .getattr("SM100_MAX_SIZE_MB")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(
                budget.keys().copied().collect::<Vec<_>>(),
                runner_tp_sizes("flashinfer_mnnvl")
            );
            for (tp, mb) in budget {
                assert_eq!(
                    flashinfer_fusion_workspace("NVIDIA B200", tp),
                    Some((mb * MIB) as u64),
                    "TP{tp}"
                );
            }
        });
    }

    #[test]
    fn a_tp_no_runner_measures_is_refused_with_its_reason() {
        let trtllm = ["flashinfer_trtllm"];
        assert_eq!(fused_all_reduce_refusal(&trtllm, "NVIDIA B200", 8), None);
        // vLLM budgets SM100 TP16 (multi-node MNNVL); no runner launches it.
        assert!(fused_all_reduce_refusal(&trtllm, "NVIDIA B200", 16)
            .unwrap()
            .contains("measured at TP [2, 4, 8]"));
        assert!(fused_all_reduce_refusal(&["flashinfer_mnnvl"], "NVIDIA B200", 1).is_some());
        assert!(fused_all_reduce_refusal(&trtllm, "NVIDIA A100", 4)
            .unwrap()
            .contains("no FlashInfer all-reduce budget"));
    }

    #[test]
    #[should_panic(expected = "no FlashInfer all-reduce fusion workspace")]
    fn a_gpu_vllm_does_not_fuse_on_has_no_workspace() {
        flashinfer_fusion_max_bytes("NVIDIA A100", 4);
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

    #[test]
    fn hopper_uses_the_sm90_workspace_cap() {
        let h200 = |num_gpus| AllReduceFusionKernelConfig {
            gpu_name: "NVIDIA H200".to_string(),
            ..config(num_gpus)
        };
        // 2 MiB / (6144 * 2 B) and 0.5 MiB / (6144 * 2 B).
        assert_eq!(AllReduceFusionSpec::max_fused_tokens(&h200(4)), 170);
        assert_eq!(AllReduceFusionSpec::max_fused_tokens(&h200(8)), 42);
        assert_eq!(AllReduceFusionSpec::max_fused_tokens(&h200(2)), 5461);
    }

    #[test]
    fn mi300x_infinity_fabric_uses_the_capability_free_cap_without_panic() {
        // MI300X has no CUDA compute capability, so the FlashInfer workspace path
        // would panic. The Infinity-Fabric roofline (rocm_fabric_roofline) uses
        // the capability-free SM100 cap as the sweep upper bound: at TP4,
        // 32 MiB / (6144 * 2 B) = 2730 (identical to B200 TP4); at TP8,
        // 1 MiB / (6144 * 2 B) = 85.
        let amd = |num_gpus| AllReduceFusionKernelConfig {
            backends: vec!["rocm_fabric_roofline"],
            gpu_name: "MI300X".to_string(),
            fabric: Fabric::InfinityFabric,
            ..config(num_gpus)
        };
        assert_eq!(AllReduceFusionSpec::max_fused_tokens(&amd(4)), 2730);
        assert_eq!(
            AllReduceFusionSpec::sweep_grid(&amd(4)).axes()[0].last(),
            Some(&2730.0)
        );
        assert_eq!(AllReduceFusionSpec::max_fused_tokens(&amd(8)), 85);
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

    /// DeepSeek-V4.1-Flash TP4 (H=5120): the switch lands off the power-of-two
    /// grid (25/26) and the cap (3276) is not a token-axis point.
    #[test]
    fn mnnvl_sweep_at_hidden_5120_brackets_switch_and_cap() {
        let config = AllReduceFusionKernelConfig {
            hidden_dim: 5120,
            ..mnnvl_config(4)
        };
        assert_eq!(AllReduceFusionSpec::mnnvl_max_oneshot_tokens(&config), 25);
        assert_eq!(AllReduceFusionSpec::max_fused_tokens(&config), 3276);

        let grid = AllReduceFusionSpec::sweep_grid(&config);
        let tokens = &grid.axes()[0];
        for t in [16.0, 17.0, 25.0, 26.0, 3200.0, 3276.0] {
            assert!(tokens.contains(&t), "missing {t}");
        }
        assert_eq!(tokens.last(), Some(&3276.0));
        assert!(tokens.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(tokens.len(), 36);
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
