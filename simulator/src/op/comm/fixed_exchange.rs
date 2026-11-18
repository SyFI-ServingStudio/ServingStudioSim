//! The fixed-pattern MNNVL exchange: collectives whose every rank stores into,
//! or reads from, its peers' fabric-mapped rows with Lamport-style arrival
//! (`profiling/runners/comm/csrc/mnnvl_fixed_exchange.cuh`). No engine ships
//! it; it is a design measured kernel by kernel, and both users here read
//! those measured rows at each rank's own row count.
//!
//! * [`FixedExpertExchange`]: the expert-parallel dispatch and combine around
//!   one MoE layer's routed experts, in place of vLLM's pynccl all-gatherv /
//!   reduce-scatterv. The dispatch gathers the already-quantized NVFP4 rows
//!   (the dispatch quantization stays its own launch in the caller); the
//!   combine pushes every rank's bf16 partial sums to the owners and sums them
//!   inside the kernel, so no separate sum follows.
//! * [`GroupGatherScatter`]: the pair a row-parallel group runs around a
//!   sharded sublayer (the KDA attention-TP group): an all-gather of the
//!   members' bf16 inputs before it and a reduce-scatter of the partial
//!   outputs after it. The combine has a pushed (`mnnvl_lamport`) and an
//!   owner-read (`mnnvl_pull`) variant, both measured; each leg is billed at
//!   the faster one.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    MoeEpCollectiveKernelInput, MoeEpQuantizedAllGatherKernel, MoeEpQuantizedAllGatherKernelConfig,
    MoeEpReduceScatterKernel, MoeEpReduceScatterKernelConfig, RowAllGatherKernel,
    RowAllGatherKernelConfig,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Evaluator, LeafMetrics, PerfApiBridge, Probe, SlotInput,
};

/// Every rank stores its rows into its peers' landing rows.
const PUSH_BACKENDS: &[&str] = &["mnnvl_lamport"];
/// Push, or each owner reading its peers' fabric-mapped buffers.
const PUSH_OR_PULL_BACKENDS: &[&str] = &["mnnvl_lamport", "mnnvl_pull"];
/// The only EP group the expert exchange's rows span.
pub const FIXED_EXPERT_EXCHANGE_EP_SIZE: u32 = 8;
const FABRIC: &str = "nvlink";

/// The expert exchange's two kernels for one EP group.
#[derive(Clone, Debug)]
pub struct FixedExpertExchangeConfig {
    /// NVFP4 activations, block scales, top-k ids and weights, gathered.
    pub dispatch: MoeEpQuantizedAllGatherKernelConfig,
    /// bf16 partial sums back to the owners.
    pub combine: MoeEpReduceScatterKernelConfig,
}

impl FixedExpertExchangeConfig {
    /// `max_total_tokens` is the largest row count summed over the group (the
    /// measured grids' total axis).
    pub fn new(
        gpu_name: &str,
        ep_size: u32,
        hidden: u32,
        top_k: u32,
        max_total_tokens: u32,
    ) -> Result<Self, String> {
        if ep_size != FIXED_EXPERT_EXCHANGE_EP_SIZE {
            return Err(format!(
                "ep_size {ep_size}: the fixed-pattern expert exchange has rows at \
                 {FIXED_EXPERT_EXCHANGE_EP_SIZE} GPUs only"
            ));
        }
        Ok(Self {
            dispatch: MoeEpQuantizedAllGatherKernelConfig {
                backends: PUSH_BACKENDS.to_vec(),
                gpu_name: gpu_name.to_string(),
                num_gpus: ep_size,
                hidden_size: hidden.into(),
                top_k: top_k.into(),
                activation_dtype: DType::Nvfp4E2m1,
                fabric: FABRIC.into(),
                max_total_tokens,
            },
            combine: MoeEpReduceScatterKernelConfig {
                backends: PUSH_BACKENDS.to_vec(),
                gpu_name: gpu_name.to_string(),
                num_gpus: ep_size,
                hidden_size: hidden.into(),
                dtype: DType::Bf16,
                fabric: FABRIC.into(),
                max_total_tokens,
            },
        })
    }
}

/// One MoE layer's expert exchange. The dispatch and the combine are compiled
/// and evaluated separately because the routed experts sit between them.
pub struct FixedExpertExchange {
    dispatch: Op<MoeEpQuantizedAllGatherKernel>,
    combine: Op<MoeEpReduceScatterKernel>,
}

impl FixedExpertExchange {
    /// The path, for the caller's cost-tree label.
    pub const LABEL: &'static str =
        "measured fixed-pattern MNNVL all-gather + reduce-scatter (mnnvl_lamport EP8 rows)";

    /// `prefix` is the MoE layer's name; the leaves are
    /// `<prefix>.dispatch_fast_all_gather` and
    /// `<prefix>.combine_fast_reduce_scatter`.
    pub fn build(
        prefix: &str,
        config: FixedExpertExchangeConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        Ok(Self {
            dispatch: atomic(
                format!("{prefix}.dispatch_fast_all_gather"),
                config.dispatch,
                MoeEpQuantizedAllGatherKernel::build,
                bridge,
            )?,
            combine: atomic(
                format!("{prefix}.combine_fast_reduce_scatter"),
                config.combine,
                MoeEpReduceScatterKernel::build,
                bridge,
            )?,
        })
    }

    pub fn compile_dispatch(&self, builder: &mut CostTreeBuilder) -> CostNode {
        self.dispatch.compile(builder)
    }

    pub fn compile_combine(&self, builder: &mut CostTreeBuilder) -> CostNode {
        self.combine.compile(builder)
    }

    /// `tokens` are each EP rank's rows in the exchange (all zero when no rank
    /// has work: nothing is launched).
    pub fn eval_dispatch(&self, tokens: &[u32], ev: &mut Evaluator) {
        push_unless_idle(&self.dispatch, tokens, ev);
    }

    /// Same `tokens` as [`Self::eval_dispatch`].
    pub fn eval_combine(&self, tokens: &[u32], ev: &mut Evaluator) {
        push_unless_idle(&self.combine, tokens, ev);
    }
}

/// The group pair's two kernels.
#[derive(Clone, Debug)]
pub struct GroupGatherScatterConfig {
    pub gather: RowAllGatherKernelConfig,
    pub scatter: MoeEpReduceScatterKernelConfig,
}

impl GroupGatherScatterConfig {
    /// Both legs move `hidden` bf16 values per row among `group_gpus` ranks;
    /// `max_total_tokens` is the largest row count summed over the group.
    pub fn new(gpu_name: &str, group_gpus: u32, hidden: u32, max_total_tokens: u32) -> Self {
        Self {
            gather: RowAllGatherKernelConfig {
                backends: PUSH_OR_PULL_BACKENDS.to_vec(),
                gpu_name: gpu_name.to_string(),
                num_gpus: group_gpus,
                hidden_size: hidden.into(),
                dtype: DType::Bf16,
                fabric: FABRIC.into(),
                max_total_tokens,
            },
            scatter: MoeEpReduceScatterKernelConfig {
                backends: PUSH_OR_PULL_BACKENDS.to_vec(),
                gpu_name: gpu_name.to_string(),
                num_gpus: group_gpus,
                hidden_size: hidden.into(),
                dtype: DType::Bf16,
                fabric: FABRIC.into(),
                max_total_tokens,
            },
        }
    }
}

/// A row-parallel group's input all-gather and output reduce-scatter, each
/// its own launch.
pub struct GroupGatherScatter {
    gather: Op<RowAllGatherKernel>,
    scatter: Op<MoeEpReduceScatterKernel>,
}

impl GroupGatherScatter {
    /// The pricing note, for the caller's cost-tree label.
    pub const PRICING: &'static str =
        "input all-gather + output reduce-scatter, two launches on the fixed-pattern rows \
         measured at the group size (faster of push and pull)";

    /// The leaves are `<prefix>.attn_tp_input_all_gather` and
    /// `<prefix>.attn_tp_output_reduce_scatter`.
    pub fn build(
        prefix: &str,
        config: GroupGatherScatterConfig,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        Ok(Self {
            gather: atomic(
                format!("{prefix}.attn_tp_input_all_gather"),
                config.gather,
                RowAllGatherKernel::build,
                bridge,
            )?,
            scatter: atomic(
                format!("{prefix}.attn_tp_output_reduce_scatter"),
                config.scatter,
                MoeEpReduceScatterKernel::build,
                bridge,
            )?,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            self.gather.compile(builder),
            self.scatter.compile(builder),
        ])
    }

    /// `member_rows` are each group rank's own rows.
    pub fn eval(&self, member_rows: &[u32], ev: &mut Evaluator) {
        let input = MoeEpCollectiveKernelInput {
            per_rank_tokens: member_rows.to_vec(),
        };
        self.gather.eval(&input, ev);
        self.scatter.eval(&input, ev);
    }
}

fn atomic<K, C, F>(
    name: String,
    config: C,
    build_kernel: F,
    bridge: &PerfApiBridge,
) -> Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError>,
{
    let kernel = build_kernel(name.clone(), config, bridge)?;
    Ok(Op::new(name, Arc::new(kernel)))
}

/// Push the collective at `tokens`, or zero when every rank is idle. The slot
/// exists either way.
fn push_unless_idle<K>(op: &Op<K>, tokens: &[u32], ev: &mut Evaluator)
where
    K: Probe<Input = MoeEpCollectiveKernelInput>,
    MoeEpCollectiveKernelInput: Into<SlotInput>,
{
    let input = MoeEpCollectiveKernelInput {
        per_rank_tokens: tokens.to_vec(),
    };
    let metrics = if tokens.iter().all(|&rows| rows == 0) {
        LeafMetrics::ZERO
    } else {
        op.kernel.eval(&input)
    };
    ev.push(metrics, || input.into());
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;

    const GPU: &str = "NVIDIA B200";

    fn bridge() -> PerfApiBridge {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        bridge
    }

    fn dim(value: u32) -> Value {
        json!({"bindings": {}, "expression": null, "value": value})
    }

    fn leaves(tree: &crate::timing::CostTree) -> Vec<(&str, &str, &Value)> {
        tree.slots
            .iter()
            .map(|slot| (slot.name.as_str(), slot.kind.as_str(), &slot.kernel_config))
            .collect()
    }

    /// The leaf names (presets key backend overrides by them), kinds and
    /// kernel identities of GLM-5.3-Flash at EP8: hidden 4096, top-8, 8192
    /// rows per rank.
    #[test]
    fn expert_exchange_is_a_pushed_gather_and_reduce_scatter() {
        let config = FixedExpertExchangeConfig::new(GPU, 8, 4096, 8, 65536).unwrap();
        let exchange = FixedExpertExchange::build("l0.moe", config, &bridge()).unwrap();
        let mut builder = CostTreeBuilder::new();
        let root = CostNode::Sum(vec![
            exchange.compile_dispatch(&mut builder),
            exchange.compile_combine(&mut builder),
        ]);
        let tree = builder.finish(root);
        let dispatch = json!({
            "activation_dtype": "nvfp4_e2m1",
            "backends": ["mnnvl_lamport"],
            "fabric": "nvlink",
            "gpu_name": GPU,
            "hidden_size": dim(4096),
            "max_total_tokens": 65536,
            "num_gpus": 8,
            "top_k": dim(8),
        });
        let combine = json!({
            "backends": ["mnnvl_lamport"],
            "dtype": "bf16",
            "fabric": "nvlink",
            "gpu_name": GPU,
            "hidden_size": dim(4096),
            "max_total_tokens": 65536,
            "num_gpus": 8,
        });
        assert_eq!(
            leaves(&tree),
            [
                (
                    "l0.moe.dispatch_fast_all_gather",
                    "moe_ep_quantized_all_gather",
                    &dispatch
                ),
                (
                    "l0.moe.combine_fast_reduce_scatter",
                    "moe_ep_reduce_scatter",
                    &combine
                ),
            ]
        );
    }

    /// The rows exist at EP8 only: another group must fail at config time, not
    /// extrapolate over the group size.
    #[test]
    fn expert_exchange_refuses_groups_without_rows() {
        for ep_size in [2, 4, 16] {
            let error = FixedExpertExchangeConfig::new(GPU, ep_size, 4096, 8, 65536).unwrap_err();
            assert!(error.contains(&format!("ep_size {ep_size}")), "{error}");
        }
    }

    /// A step in which no rank has rows launches nothing, but both slots are
    /// still filled so the layout does not depend on the batch.
    #[test]
    fn idle_expert_exchange_fills_both_slots_with_zero() {
        let config = FixedExpertExchangeConfig::new(GPU, 8, 4096, 8, 65536).unwrap();
        let exchange = FixedExpertExchange::build("l0.moe", config, &bridge()).unwrap();
        let mut buf = [LeafMetrics::MISS; 2];
        let mut inputs = Vec::new();
        let mut ev = Evaluator::with_inputs(&mut buf, &mut inputs);
        exchange.eval_dispatch(&[0; 8], &mut ev);
        exchange.eval_combine(&[0; 8], &mut ev);
        assert_eq!(ev.filled(), 2);
        assert_eq!(inputs.len(), 2);
        for slot in buf {
            assert_eq!(slot.m.time_ms, 0.0);
            assert!(slot.coverage == LeafMetrics::ZERO.coverage);
        }
    }

    /// Both group legs offer push and pull at the group's own size: a
    /// two-rank KDA group of the same model.
    #[test]
    fn group_pair_offers_push_and_pull_on_both_legs() {
        let config = GroupGatherScatterConfig::new(GPU, 2, 4096, 16384);
        let pair = GroupGatherScatter::build("l0", config, &bridge()).unwrap();
        let mut builder = CostTreeBuilder::new();
        let root = pair.compile(&mut builder);
        assert!(matches!(&root, CostNode::Sum(children) if children.len() == 2));
        let tree = builder.finish(root);
        let leg = json!({
            "backends": ["mnnvl_lamport", "mnnvl_pull"],
            "dtype": "bf16",
            "fabric": "nvlink",
            "gpu_name": GPU,
            "hidden_size": dim(4096),
            "max_total_tokens": 16384,
            "num_gpus": 2,
        });
        assert_eq!(
            leaves(&tree),
            [
                ("l0.attn_tp_input_all_gather", "row_all_gather", &leg),
                (
                    "l0.attn_tp_output_reduce_scatter",
                    "moe_ep_reduce_scatter",
                    &leg
                ),
            ]
        );
    }
}
