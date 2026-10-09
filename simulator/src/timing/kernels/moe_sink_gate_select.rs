//! Sink-renormalized MoE gate select: one `inkling_gate_select` call per MoE layer.
//!
//! On the router GEMM's fp32 logits (`num_routed_experts` routed columns, then
//! `num_shared_experts` sink columns, then GEMM padding up to
//! `logits_row_stride`) each token scores the routed experts as
//! `sigmoid(l) + bias`, keeps the `top_k` best, and renormalizes the selected
//! raw logits jointly with the sink logits (`sigmoid(l) / Σ sigmoid(l)`, times
//! `route_scale` and the global scale). Output: ids and weights
//! `[T, top_k + num_shared_experts]`, sinks last. One Triton program per token,
//! one launch, so `num_tokens` is the only input.
//!
//! The grid holds every CUDA-graph capture size vLLM pads a batch to up to the
//! 2048-token `max_num_batched_tokens` (1, 2, 4, then steps of 8 to 256 and 16
//! to 2048), so graph-padded calls read a profiled row, plus a few sizes past
//! it for extrapolation.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeSinkGateSelectKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Experts top-k selects from.
    pub num_routed_experts: u32,
    /// Sink experts every token activates; their logits follow the routed ones.
    pub num_shared_experts: u32,
    /// Routed experts selected per token.
    pub top_k: u32,
    /// Row stride of the logits: routed + sink columns plus the router GEMM's
    /// padding (e.g. 258 padded to 264).
    pub logits_row_stride: u32,
    /// Per-expert selection bias added to the sigmoid scores.
    pub has_bias: bool,
    /// Checkpoint global scale multiplied into the weights.
    pub has_global_scale: bool,
    #[compute_dtype]
    pub logits_dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeSinkGateSelectKernelInput {
    pub num_tokens: u32,
}

pub struct MoeSinkGateSelectSpec;

fn validate_config(config: &MoeSinkGateSelectKernelConfig) {
    assert!(
        config.top_k >= 1 && config.top_k <= config.num_routed_experts,
        "top_k must be in [1, num_routed_experts], got {}/{}",
        config.top_k,
        config.num_routed_experts
    );
    // The kernel spans the selected set with tl.arange(0, K + S).
    assert!(
        (config.top_k + config.num_shared_experts).is_power_of_two(),
        "top_k + num_shared_experts must be a power of two, got {}",
        config.top_k + config.num_shared_experts
    );
    assert!(
        config.logits_row_stride >= config.num_routed_experts + config.num_shared_experts,
        "logits_row_stride {} does not cover the routed and sink columns",
        config.logits_row_stride
    );
    // inkling_gate_select asserts fp32 logits.
    assert_eq!(config.logits_dtype, DType::Fp32, "gate select needs fp32 logits");
}

impl KernelSpec for MoeSinkGateSelectSpec {
    type Config = MoeSinkGateSelectKernelConfig;
    type Input = MoeSinkGateSelectKernelInput;
    const KIND: KernelKind = "moe_sink_gate_select";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        validate_config(config);
        SweepGrid::new(vec![Axis::chain([
            Axis::values([1, 2, 4]),
            Axis::arithmetic(8, 256, 8),
            Axis::arithmetic(272, 2048, 16),
            Axis::values([3072, 4096, 6144, 8192, 16384]),
        ])])
    }

    fn cache_kind(_: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", tokens as u32)
                .with("num_routed_experts", config.num_routed_experts)
                .with("num_shared_experts", config.num_shared_experts)
                .with("top_k", config.top_k)
                .with("logits_row_stride", config.logits_row_stride)
                .with("has_bias", config.has_bias)
                .with("has_global_scale", config.has_global_scale)
                .with("logits_dtype", config.logits_dtype.as_str())
        })
    }
}

register_kernel!(MoeSinkGateSelectKernel, MoeSinkGateSelectSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn inkling_small() -> MoeSinkGateSelectKernelConfig {
        MoeSinkGateSelectKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            num_routed_experts: 256,
            num_shared_experts: 2,
            top_k: 6,
            logits_row_stride: 264,
            has_bias: true,
            has_global_scale: true,
            logits_dtype: DType::Fp32,
        }
    }

    /// A graph-padded call must land on a profiled row: every vLLM CUDA-graph
    /// capture size up to 2048 is a grid point, and the grid extends past it.
    #[test]
    fn grid_holds_every_cuda_graph_capture_size() {
        let grid = MoeSinkGateSelectSpec::sweep_grid(&inkling_small());
        let axis = &grid.axes()[0];
        let capture = [1u32, 2, 4]
            .into_iter()
            .chain((8..=256).step_by(8))
            .chain((272..=2048).step_by(16));
        for size in capture {
            assert!(axis.contains(&(size as f64)), "capture size {size} missing");
        }
        assert_eq!(*axis.last().unwrap(), 16384.0);
        assert!(axis.len() <= 500);
    }

    /// Catches a payload that drops or renames a field the Python
    /// `MoeSinkGateSelectArgs` schema requires, or leaks an extra one.
    #[test]
    fn payload_matches_python_args() {
        let cfg = inkling_small();
        let grid = MoeSinkGateSelectSpec::sweep_grid(&cfg);
        let payloads = MoeSinkGateSelectSpec::enumerate(&cfg, &grid, "vllm_triton");
        let fields = payloads[3].fields();
        let keys: Vec<_> = fields.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "has_bias",
                "has_global_scale",
                "logits_dtype",
                "logits_row_stride",
                "num_routed_experts",
                "num_shared_experts",
                "num_tokens",
                "top_k"
            ]
        );
        assert_eq!(fields["num_tokens"], 8);
        assert_eq!(fields["logits_row_stride"], 264);
        assert_eq!(fields["logits_dtype"], "fp32");
    }

    /// The kernel cannot compile a selected set that is not a power of two; a
    /// config that slipped past would JIT-profile a shape vLLM never runs.
    #[test]
    #[should_panic(expected = "power of two")]
    fn rejects_a_selected_set_the_kernel_cannot_span() {
        let mut cfg = inkling_small();
        cfg.top_k = 5;
        MoeSinkGateSelectSpec::sweep_grid(&cfg);
    }
}
