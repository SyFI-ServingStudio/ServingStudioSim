//! `qwen3_moe_dp_attn_ep_ffn` — L4 model_arch for Qwen3-MoE-style decoders with
//! **DP-attention + EP-FFN**: attention is sharded over `attn_tp_size` head-
//! parallel ranks and replicated as `num_dp_groups = ep_size / attn_tp_size`
//! independent DP shards (mirroring `llama3_dp_attn_tp_ffn`); the FFN is a
//! sparse mixture-of-experts whose experts are distributed across all `ep_size`
//! ranks of the replica via the L2 MoE dispatch / combine ops.
//!
//! Cost shape (iter-wise, mirrors `llama3_dp_attn_tp_ffn` but with the FFN
//! exploded into MoE sub-sections per L3 design.md §「sync-section invariant」):
//!
//! ```text
//! Sum(
//!   embed,
//!   Scale{num_layers}( Sum(
//!     Max{1.0}( attn_block_tp × num_dp_groups ),     // attn-DP fan-out
//!     Max{1.0}( moe_router_local × num_dp_groups ),  // post_norm + router, per-DP-shard (replicated)
//!     moe_dispatch,                                  // L2 op, 2 leaves (inter, intra)
//!     Max{1.0}( moe_expert_compute_local × ep_size ),// EP fan-out: per-rank expert compute
//!     Max{1.0}( moe_local_reduce × num_dp_groups ),  // home reduce, per-DP-shard
//!     moe_combine,                                   // L2 op, 4 leaves
//!   )),
//!   final_norm,
//!   lm_head,
//! )
//! ```
//!
//! The `× num_dp_groups` MAX nodes (attn_block, moe_router, moe_local_reduce)
//! are the DP-attention fan-out: distinct DP shards process distinct token
//! slices on the dense path, so their wallclocks are independent and each sync
//! section's wallclock is the slowest shard — DP load imbalance is modeled
//! exactly, every child eats its own shard's `batch_tokens` (never a pooled
//! average). post_norm + router and the post-combine home reduce run
//! *replicated* on every rank of a shard (the full hidden is replicated across
//! the shard's `attn_tp_size` ranks after the o_proj all-reduce), so each
//! shard's cost is its own token slice — NOT `m_total / ep_size`. The
//! `× ep_size` MAX on `expert_compute` is the EP fan-out: under skewed routing
//! each EP rank's `local_ppm` shard yields a distinct grouped-GEMM kernel cache
//! (L1 grouped gemm is distribution-sensitive). For v1 uniform routing the
//! fan-out children are identical and the MAX degenerates numerically — the
//! structure stays ready for the future skew (`MoeExpertComputeLocalWorklet`
//! instances can be minted with distinct `local_ppm` per EP rank without
//! touching the arch shape).
//!
//! v1 deviations (carried over from `llama3_dp_attn_tp_ffn`, plus MoE-specific):
//!   - embed / final_norm / lm_head are replicated full shapes on the pooled
//!     token total (no vocab-parallel split);
//!   - routing distribution is uniform across experts (`uniform_local_ppm`);
//!     hot-expert skew effects on grouped GEMM are NOT modeled at this layer.

use std::sync::Arc;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::moe_model_cfg::MoeModelCfg;
use crate::common::Fabric;
use crate::op::Op;
use crate::op::moe::{
    MoeCombineOp, MoeDispatchOp, MoeNetConfig, MoeNetInput, Placement,
};
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, RmsNormKernel,
    RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::routing::RoutingDistribution;
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::{
    AttnBlockTpWorklet, AttnBlockTpWorkletConfig, AttnBlockTpWorkletInput,
    AttnBlockTpWorkletResolved, MoeExpertComputeLocalWorklet,
    MoeExpertComputeLocalWorkletConfig, MoeExpertComputeLocalWorkletInput,
    MoeExpertComputeLocalWorkletResolved, MoeRouterLocalWorklet, MoeRouterLocalWorkletConfig,
    MoeRouterLocalWorkletInput, MoeRouterLocalWorkletResolved, uniform_local_ppm,
};

const NORM_BACKENDS: &[&str] = &["flashinfer"];
// Dense + grouped GEMM backends are dtype-driven (`MoeModelCfg::gemm_backends()`):
// fp8 runs select the FP8-only `deepgemm`, bf16 runs `torch`. A GEMM never lists a
// backend without rows for its dtype (that would abort the build at prewarm).
const ACT_BACKENDS: &[&str] = &["triton"];
// See llama3_dense: FlashInfer impls registered under fa2/fa3, not "flashinfer".
const ATTN_BACKENDS: &[&str] = &["fa2", "fa3"];
const ALLREDUCE_BACKENDS: &[&str] = &["nccl"];
// p2p backend for both MoE dispatch (inter+intra) and combine. nccl is the
// portable choice; nvshmem (DeepEP) can be plugged via this constant once its
// L1 kernel is wired everywhere this arch runs.
const P2P_BACKENDS: &[&str] = &["nccl"];
// v1 fabric choices: NVLink for the attn-TP allreduce AND the MoE intra leg;
// Infiniband for the MoE inter leg.
const TP_FABRIC: Fabric = Fabric::Nvlink;
const MOE_INTRA_FABRIC: Fabric = Fabric::Nvlink;
const MOE_INTER_FABRIC: Fabric = Fabric::Infiniband;

/// This arch's numeric parallel input. attention TP × num_dp_groups together
/// span the replica (= ep_size GPUs); `hp_size` controls the MoE
/// `ReplicatedHeadParallel` placement (the residing-set width on the FFN side);
/// `nvl_num_gpu` partitions the `ep_size` ranks into NVL domains for the MoE
/// dispatch/combine intra/inter split.
#[derive(Clone, Debug)]
pub struct Qwen3MoeParallel {
    pub attn_tp_size: u16,
    pub ep_size: u16,
    pub hp_size: u16,
    pub nvl_num_gpu: u16,
    pub gpu_name: String,
}

/// Raw worklet/op configs. `attn_block` bakes `attn_tp_size`; the MoE
/// worklets / ops bake their per-rank shape. The DP degree
/// (`num_dp_groups = ep_size / attn_tp_size`) is carried separately for the
/// L4 cost fan-out loop.
pub struct Qwen3MoeDpAttnEpFfnConfigs {
    pub attn_block: AttnBlockTpWorkletConfig,
    pub moe_router: MoeRouterLocalWorkletConfig,
    pub moe_dispatch: MoeNetConfig,
    pub moe_expert_compute: MoeExpertComputeLocalWorkletConfig,
    pub moe_local_reduce: ElementwiseKernelConfig,
    pub moe_combine: MoeNetConfig,
    pub embed: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub num_layers: u32,
    pub attn_tp_size: u16,
    pub ep_size: u16,
    pub num_dp_groups: u16,
    pub hp_size: u16,
    pub nvl_num_gpu: u16,
    pub top_k: u32,
    /// Total expert count across all EP ranks — carried for the model label only
    /// (the per-rank expert math already lives in `moe_expert_compute`).
    pub num_experts: u32,
}

/// Post-resolve aggregate; atomic ops (embed / final_norm / lm_head /
/// moe_local_reduce) carry their kernel config through unchanged.
pub struct Qwen3MoeDpAttnEpFfnResolved {
    pub attn_block: AttnBlockTpWorkletResolved,
    pub moe_router: MoeRouterLocalWorkletResolved,
    pub moe_dispatch: MoeNetConfig,
    pub moe_expert_compute: MoeExpertComputeLocalWorkletResolved,
    pub moe_local_reduce: ElementwiseKernelConfig,
    pub moe_combine: MoeNetConfig,
    pub embed: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub num_layers: u32,
    pub attn_tp_size: u16,
    pub ep_size: u16,
    pub num_dp_groups: u16,
    pub hp_size: u16,
    pub nvl_num_gpu: u16,
    pub top_k: u32,
    /// Total expert count across all EP ranks — carried for the model label.
    pub num_experts: u32,
}

pub struct Qwen3MoeDpAttnEpFfnModel {
    pub name: String,
    pub num_layers: u32,
    pub attn_tp_size: u16,
    pub ep_size: u16,
    pub num_dp_groups: u16,
    pub hp_size: u16,
    pub nvl_num_gpu: u16,
    pub top_k: u32,
    pub num_experts: u32,
    pub total_kv_bytes_per_token: u64,
    pub attn_block: AttnBlockTpWorklet,
    pub moe_router: MoeRouterLocalWorklet,
    pub moe_dispatch: MoeDispatchOp,
    pub moe_expert_compute: MoeExpertComputeLocalWorklet,
    pub moe_local_reduce: Op<ElementwiseKernel>,
    pub moe_combine: MoeCombineOp,
    pub embed: Op<ElementwiseKernel>,
    pub final_norm: Op<RmsNormKernel>,
    pub lm_head: Op<SingleGemmKernel>,
    /// CostTree structure compiled once at build (flattened) + its slot count,
    /// so per-iter `eval_iter` only evals leaves + aggregates.
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

/// The attention-block config for this arch, depending only on `attn_tp_size` +
/// model dims (input_norm / qkv / attention / o_proj / tp_allreduce partition).
fn attn_block_config(
    model: &MoeModelCfg,
    attn_tp_size: u16,
    gpu_name: &str,
) -> AttnBlockTpWorkletConfig {
    AttnBlockTpWorkletConfig {
        hidden: model.hidden,
        num_qo_heads: model.num_qo_heads,
        num_kv_heads: model.num_kv_heads,
        head_dim: model.head_dim,
        dtype: model.dtype,
        fp8: model.fp8,
        tp_size: attn_tp_size,
        allreduce_fabric: TP_FABRIC,
        gpu_name: gpu_name.to_string(),
        norm_backends: NORM_BACKENDS.to_vec(),
        gemm_backends: model.gemm_backends(),
        attn_backends: ATTN_BACKENDS.to_vec(),
        kv_cache_append_backends: vec!["vllm_cuda"],
        kv_cache_block_size: 16,
        kv_cache_layout: "NHD".to_string(),
        kv_scale_granularity: "tensor".to_string(),
        allreduce_backends: ALLREDUCE_BACKENDS.to_vec(),
    }
}

pub fn build_configs(
    model: &MoeModelCfg,
    parallel: &Qwen3MoeParallel,
    routing: &RoutingDistribution,
) -> Qwen3MoeDpAttnEpFfnConfigs {
    let gpu = &parallel.gpu_name;
    // Byte-transfer widths (local reduce, embed) use the compute dtype: fp8 halves
    // the wire size, bf16 leaves it unchanged. The RMSNorm ops keep `model.dtype`.
    let dtype_bytes = model.compute_dtype().size_bytes();
    assert!(
        parallel.attn_tp_size > 0 && parallel.ep_size > 0,
        "attn_tp_size / ep_size must be non-zero"
    );
    assert!(
        parallel.ep_size % parallel.attn_tp_size == 0,
        "ep_size {} must be a multiple of attn_tp_size {} (DP groups = ep / attn_tp)",
        parallel.ep_size,
        parallel.attn_tp_size,
    );
    let num_dp_groups = parallel.ep_size / parallel.attn_tp_size;
    assert!(
        parallel.hp_size > 0 && parallel.hp_size <= parallel.ep_size,
        "hp_size {} must be in 1..=ep_size {}",
        parallel.hp_size,
        parallel.ep_size,
    );
    assert!(
        parallel.ep_size % parallel.hp_size == 0,
        "ep_size {} must be a multiple of hp_size {}",
        parallel.ep_size,
        parallel.hp_size,
    );
    assert!(parallel.nvl_num_gpu > 0, "nvl_num_gpu must be non-zero");
    assert!(
        model.num_experts % u32::from(parallel.ep_size) == 0,
        "num_experts {} not divisible by ep_size {}",
        model.num_experts,
        parallel.ep_size,
    );
    assert_eq!(
        routing.num_experts(),
        model.num_experts,
        "routing distribution has {} experts, model has {}",
        routing.num_experts(),
        model.num_experts,
    );
    // `routing` drives the L2 MoE dispatch/combine `BottleneckCurve` simulation
    // — non-uniform routing produces different per-rank send/recv profiles and
    // hence different curves. The L3 grouped-GEMM `local_ppm` stays uniform
    // (v1 deviation, see agent-trace): per-rank `local_ppm` skew on the FFN
    // compute side is a deferred follow-up.
    let local_ppm = uniform_local_ppm(model.num_experts, parallel.ep_size);
    let moe_net = MoeNetConfig {
        backends: P2P_BACKENDS.to_vec(),
        gpu_name: gpu.clone(),
        // dispatch/combine ship the hidden activation → compute dtype (fp8 halves
        // the all-to-all payload).
        dtype: model.compute_dtype(),
        intra_fabric: MOE_INTRA_FABRIC,
        inter_fabric: MOE_INTER_FABRIC,
        ep_size: u32::from(parallel.ep_size),
        nvl_num_gpu: u32::from(parallel.nvl_num_gpu),
        h: model.hidden,
        top_k: model.top_k,
        routing: routing.clone(),
        placement: Placement::ReplicatedHeadParallel {
            hp_size: u32::from(parallel.hp_size),
        },
    };
    Qwen3MoeDpAttnEpFfnConfigs {
        attn_block: attn_block_config(model, parallel.attn_tp_size, gpu),
        moe_router: MoeRouterLocalWorkletConfig {
            hidden: model.hidden,
            num_experts: model.num_experts,
            dtype: model.dtype,
            compute_dtype: model.compute_dtype(),
            gpu_name: gpu.clone(),
            norm_backends: NORM_BACKENDS.to_vec(),
            gemm_backends: model.gemm_backends(),
        },
        moe_dispatch: moe_net.clone(),
        // MoE expert compute is all-fp8 in an fp8 run (grouped GEMMs + SwiGLU act);
        // no RMSNorm inside, so the whole worklet takes the compute dtype.
        moe_expert_compute: MoeExpertComputeLocalWorkletConfig {
            hidden: model.hidden,
            moe_intermediate: model.moe_intermediate,
            num_experts: model.num_experts,
            ep_size: parallel.ep_size,
            dtype: model.compute_dtype(),
            gpu_name: gpu.clone(),
            act_backends: ACT_BACKENDS.to_vec(),
            grouped_gemm_backends: model.gemm_backends(),
            local_ppm,
        },
        // Conservative v1 model: per home-rank token, read+write one full hidden
        // vector of partials. Refinement to top_k-weighted partial reads is
        // deferred until we model expert-redundancy at home-rank reduction.
        moe_local_reduce: ElementwiseKernelConfig {
            backends: ACT_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            input_bytes_per_token: model.hidden * dtype_bytes,
            output_bytes_per_token: model.hidden * dtype_bytes,
        },
        moe_combine: moe_net,
        // Embedding gather placeholder: read one hidden-wide row, write one out
        // (replicated; hidden NOT sharded).
        embed: ElementwiseKernelConfig {
            backends: ACT_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            input_bytes_per_token: model.hidden * dtype_bytes,
            output_bytes_per_token: model.hidden * dtype_bytes,
        },
        final_norm: RmsNormKernelConfig {
            backends: NORM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden: model.hidden,
            dtype: model.dtype,
        },
        // Replicated full lm_head for v1 (vocab-parallel split deferred).
        lm_head: SingleGemmKernelConfig {
            backends: model.gemm_backends(),
            gpu_name: gpu.clone(),
            n: model.vocab,
            k: model.hidden,
            dtype: model.compute_dtype(),
        },
        num_layers: model.num_layers,
        attn_tp_size: parallel.attn_tp_size,
        ep_size: parallel.ep_size,
        num_dp_groups,
        hp_size: parallel.hp_size,
        nvl_num_gpu: parallel.nvl_num_gpu,
        top_k: model.top_k,
        num_experts: model.num_experts,
    }
}

/// **Total** KV-cache bytes one token occupies — summed across all
/// `attn_tp_size` ranks of one DP shard, all layers, all KV heads. Same
/// definition as `llama3_dp_attn_tp_ffn`: the wire size of a token's KV for a
/// PD handoff (each DP shard owns one full copy of every KV head).
fn total_kv_bytes_per_token(resolved: &Qwen3MoeDpAttnEpFfnResolved) -> u64 {
    let raw = &resolved.attn_block.raw_cfg;
    2 * raw.num_kv_heads as u64
        * raw.head_dim as u64
        * raw.kv_dtype().size_bytes() as u64
        * resolved.num_layers as u64
}

pub fn resolve_configs(cfgs: &Qwen3MoeDpAttnEpFfnConfigs) -> Qwen3MoeDpAttnEpFfnResolved {
    Qwen3MoeDpAttnEpFfnResolved {
        attn_block: AttnBlockTpWorklet::resolve_config(&cfgs.attn_block),
        moe_router: MoeRouterLocalWorklet::resolve_config(&cfgs.moe_router),
        moe_dispatch: cfgs.moe_dispatch.clone(),
        moe_expert_compute: MoeExpertComputeLocalWorklet::resolve_config(&cfgs.moe_expert_compute),
        moe_local_reduce: cfgs.moe_local_reduce.clone(),
        moe_combine: cfgs.moe_combine.clone(),
        embed: cfgs.embed.clone(),
        final_norm: cfgs.final_norm.clone(),
        lm_head: cfgs.lm_head.clone(),
        num_layers: cfgs.num_layers,
        attn_tp_size: cfgs.attn_tp_size,
        ep_size: cfgs.ep_size,
        num_dp_groups: cfgs.num_dp_groups,
        hp_size: cfgs.hp_size,
        nvl_num_gpu: cfgs.nvl_num_gpu,
        top_k: cfgs.top_k,
        num_experts: cfgs.num_experts,
    }
}

pub fn build(
    model_name: String,
    resolved: Qwen3MoeDpAttnEpFfnResolved,
    bridge: &PerfApiBridge,
) -> Result<Qwen3MoeDpAttnEpFfnModel, BuildError> {
    let num_layers = resolved.num_layers;
    let attn_tp_size = resolved.attn_tp_size;
    let ep_size = resolved.ep_size;
    let num_dp_groups = resolved.num_dp_groups;
    let hp_size = resolved.hp_size;
    let nvl_num_gpu = resolved.nvl_num_gpu;
    let top_k = resolved.top_k;
    let num_experts = resolved.num_experts;
    let total_kv_bytes_per_token = total_kv_bytes_per_token(&resolved);

    let embed_name = format!("{model_name}.embedding");
    let final_norm_name = format!("{model_name}.final_norm");
    let lm_head_name = format!("{model_name}.lm_head");
    let local_reduce_name = format!("{model_name}.moe_local_reduce");

    let embed = Op::new(
        embed_name.clone(),
        Arc::new(ElementwiseKernel::build(embed_name, resolved.embed, bridge)?),
    );

    let attn_block = AttnBlockTpWorklet::build(
        format!("{model_name}.attn_block"),
        resolved.attn_block,
        bridge,
    )?;

    let moe_router = MoeRouterLocalWorklet::build(
        format!("{model_name}.moe_router"),
        resolved.moe_router,
        bridge,
    )?;

    let moe_dispatch = MoeDispatchOp::build(
        format!("{model_name}.moe_dispatch"),
        resolved.moe_dispatch,
        bridge,
    )?;

    let moe_expert_compute = MoeExpertComputeLocalWorklet::build(
        format!("{model_name}.moe_expert_compute"),
        resolved.moe_expert_compute,
        bridge,
    )?;

    let moe_local_reduce = Op::new(
        local_reduce_name.clone(),
        Arc::new(ElementwiseKernel::build(
            local_reduce_name,
            resolved.moe_local_reduce,
            bridge,
        )?),
    );

    let moe_combine = MoeCombineOp::build(
        format!("{model_name}.moe_combine"),
        resolved.moe_combine,
        bridge,
    )?;

    let final_norm = Op::new(
        final_norm_name.clone(),
        Arc::new(RmsNormKernel::build(
            final_norm_name,
            resolved.final_norm,
            bridge,
        )?),
    );

    let lm_head = Op::new(
        lm_head_name.clone(),
        Arc::new(SingleGemmKernel::build(
            lm_head_name,
            resolved.lm_head,
            bridge,
        )?),
    );

    let mut model = Qwen3MoeDpAttnEpFfnModel {
        name: model_name,
        num_layers,
        attn_tp_size,
        ep_size,
        num_dp_groups,
        hp_size,
        nvl_num_gpu,
        top_k,
        num_experts,
        total_kv_bytes_per_token,
        attn_block,
        moe_router,
        moe_dispatch,
        moe_expert_compute,
        moe_local_reduce,
        moe_combine,
        embed,
        final_norm,
        lm_head,
        cost_flat: Vec::new(),
        n_slots: 0,
    };

    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();

    tracing::info!(
        "[build] cost tree ({} leaf slots):\n{}",
        tree.n_slots(),
        tree.describe()
    );

    Ok(model)
}

impl Qwen3MoeDpAttnEpFfnModel {
    /// Compile the per-iteration cost STRUCTURE once: Sum(embed,
    /// Scale{num_layers}(Sum(attn_max, moe_router, dispatch, expert_max,
    /// local_reduce, combine)), final_norm, lm_head). Both `Max` nodes mint
    /// their children by repeated `compile` calls on the same worklet — each
    /// call mints fresh slots, so `eval_into` fills them per child in the same
    /// order.
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let embed = self.embed.compile(&mut b);

        // attn-DP fan-out — one attn_block subtree per DP shard, identical to
        // llama3_dp_attn_tp_ffn.
        let attn_groups: Vec<CostNode> = (0..self.num_dp_groups)
            .map(|_| self.attn_block.compile(&mut b))
            .collect();
        let attn_fanout = CostNode::Max {
            overlap: 1.0,
            children: attn_groups,
        };

        // Router fan-out — post_norm + router run replicated per DP shard (the
        // shard's hidden is replicated across its attn_tp ranks after the o_proj
        // all-reduce), so each shard routes its OWN token slice. Max-fanned like
        // the attn block: the section wallclock is the slowest shard.
        let router_groups: Vec<CostNode> = (0..self.num_dp_groups)
            .map(|_| self.moe_router.compile(&mut b))
            .collect();
        let router_fanout = CostNode::Max {
            overlap: 1.0,
            children: router_groups,
        };
        let moe_dispatch = self.moe_dispatch.compile(&mut b);

        // EP fan-out — one expert_compute subtree per EP rank. Identical local_ppm
        // under v1 uniform → numerically degenerate Max; ready for skew.
        let expert_groups: Vec<CostNode> = (0..self.ep_size)
            .map(|_| self.moe_expert_compute.compile(&mut b))
            .collect();
        let expert_fanout = CostNode::Max {
            overlap: 1.0,
            children: expert_groups,
        };

        // Home-reduce fan-out — the combine returns each token to its home DP
        // shard, so the reduce runs per shard on the shard's own tokens.
        let reduce_groups: Vec<CostNode> = (0..self.num_dp_groups)
            .map(|_| self.moe_local_reduce.compile(&mut b))
            .collect();
        let reduce_fanout = CostNode::Max {
            overlap: 1.0,
            children: reduce_groups,
        };
        let moe_combine = self.moe_combine.compile(&mut b);

        let layer = CostNode::Labeled {
            label: "layer".to_string(),
            child: Box::new(CostNode::Scale {
                n: self.num_layers,
                child: Box::new(CostNode::Sum(vec![
                    attn_fanout,
                    router_fanout,
                    moe_dispatch,
                    expert_fanout,
                    reduce_fanout,
                    moe_combine,
                ])),
            }),
        };
        let final_norm = self.final_norm.compile(&mut b);
        let lm_head = self.lm_head.compile(&mut b);
        let root = CostNode::Labeled {
            label: format!(
                "{} [DP attn (attn_tp={}, dp={}) × EP FFN (ep={}, hp={}, nvl={}), \
                 num_experts={} top_k={}, {} layers]",
                self.name,
                self.attn_tp_size,
                self.num_dp_groups,
                self.ep_size,
                self.hp_size,
                self.nvl_num_gpu,
                self.num_experts,
                self.top_k,
                self.num_layers
            ),
            child: Box::new(CostNode::Sum(vec![embed, layer, final_norm, lm_head])),
        };
        b.finish(root)
    }

    /// CostTree eval: stream this iteration's per-leaf [`LeafMetrics`] through
    /// `ev` in the EXACT slot order [`cost_tree`](Self::cost_tree) minted them:
    /// embed (pooled), then ONE layer's `num_dp_groups` attn_block evals (one
    /// per DP shard, each with its own batch_tokens), then `num_dp_groups`
    /// moe_router evals (post_norm + router, replicated per DP shard on the
    /// shard's own batch_tokens), then moe_dispatch (2 leaves on global token
    /// count), then `ep_size` moe_expert_compute evals (one per EP rank, uniform
    /// v1 → same input), then `num_dp_groups` moe_local_reduce evals (home
    /// reduce, per DP shard), then moe_combine (4 leaves on global token count).
    /// The `Scale{num_layers}` fold multiplies one layer body. Finally
    /// final_norm + lm_head on pooled tokens.
    fn eval_into(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        let m_total: u32 = batch.groups.iter().map(|g| g.batch_tokens).sum();
        let global_expert_selections = m_total * self.top_k;
        let tokens_for_comm = u64::from(m_total);

        self.embed.eval(
            &ElementwiseKernelInput {
                num_tokens: m_total,
            },
            ev,
        );

        // attn-DP fan-out: one eval per group, each with its own batch_tokens.
        for g in &batch.groups {
            self.attn_block.eval(
                &AttnBlockTpWorkletInput {
                    batch_tokens: g.batch_tokens,
                    prefill_chunk_pairs: g.prefill_chunk_pairs.clone(),
                    decode_kv_lens: g.decode_kv_lens.clone(),
                },
                ev,
            );
        }

        // Router fan-out: replicated per DP shard on the shard's own tokens.
        for g in &batch.groups {
            self.moe_router.eval(
                &MoeRouterLocalWorkletInput {
                    batch_tokens: g.batch_tokens,
                },
                ev,
            );
        }

        self.moe_dispatch.eval(
            &MoeNetInput {
                tokens: tokens_for_comm,
            },
            ev,
        );

        // EP fan-out: identical input per child under v1 uniform routing.
        for _ in 0..self.ep_size {
            self.moe_expert_compute.eval(
                &MoeExpertComputeLocalWorkletInput {
                    global_expert_selections,
                },
                ev,
            );
        }

        // Home-reduce fan-out: per DP shard on the shard's own tokens.
        for g in &batch.groups {
            self.moe_local_reduce.eval(
                &ElementwiseKernelInput {
                    num_tokens: g.batch_tokens,
                },
                ev,
            );
        }

        self.moe_combine.eval(
            &MoeNetInput {
                tokens: tokens_for_comm,
            },
            ev,
        );

        self.final_norm.eval(&RmsNormKernelInput { m: m_total }, ev);
        self.lm_head.eval(&SingleGemmKernelInput { m: m_total }, ev);
    }
}

impl IterwiseUnifiedModel for Qwen3MoeDpAttnEpFfnModel {
    fn total_kv_bytes_per_token(&self) -> u64 {
        self.total_kv_bytes_per_token
    }

    /// One replica spans the EP group — `ep_size` GPUs, with the DP-attention
    /// shards (`attn_tp_size` ranks each) nested inside it.
    fn gpus_per_replica(&self) -> u16 {
        self.ep_size
    }

    fn num_attn_dp_groups(&self) -> u16 {
        self.num_dp_groups
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        assert_eq!(
            batch.groups.len(),
            self.num_dp_groups as usize,
            "Qwen3-MoE DP-attn arch expects one group per DP shard (num_dp_groups)"
        );
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut ev = Evaluator::new(slots);
        self.eval_into(batch, &mut ev);
        debug_assert_eq!(
            ev.filled(),
            self.n_slots,
            "eval cursor must fill every slot"
        );
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        assert_eq!(
            batch.groups.len(),
            self.num_dp_groups as usize,
            "Qwen3-MoE DP-attn arch expects one group per DP shard (num_dp_groups)"
        );
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut ev = Evaluator::with_inputs(slots, inputs);
        self.eval_into(batch, &mut ev);
        debug_assert_eq!(
            ev.filled(),
            self.n_slots,
            "eval cursor must fill every slot"
        );
        let agg = CostTree::aggregate(&self.cost_flat, slots, scratch);
        debug_assert_eq!(inputs.len(), self.n_slots, "slot_input must align to slots");
        agg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parallel(attn_tp: u16, ep: u16, hp: u16, nvl: u16) -> Qwen3MoeParallel {
        Qwen3MoeParallel {
            attn_tp_size: attn_tp,
            ep_size: ep,
            hp_size: hp,
            nvl_num_gpu: nvl,
            gpu_name: "H100".to_string(),
        }
    }

    fn uniform_routing_for(model: &MoeModelCfg) -> RoutingDistribution {
        RoutingDistribution::uniform(model.num_experts)
    }

    #[test]
    fn build_configs_threads_split_attn_tp_ep_and_derives_dp() {
        let model = MoeModelCfg::qwen3_235b();
        let cfgs = build_configs(&model, &parallel(4, 8, 2, 8), &uniform_routing_for(&model));
        assert_eq!(cfgs.attn_tp_size, 4);
        assert_eq!(cfgs.ep_size, 8);
        // dp = ep / attn_tp.
        assert_eq!(cfgs.num_dp_groups, 2);
        assert_eq!(cfgs.hp_size, 2);
        // attn worklet sees attn_tp; expert_compute sees ep_size.
        assert_eq!(cfgs.attn_block.tp_size, 4);
        assert_eq!(cfgs.moe_expert_compute.ep_size, 8);
        // router gemm: n=num_experts, k=hidden.
        assert_eq!(cfgs.moe_router.num_experts, 128);
        assert_eq!(cfgs.moe_router.hidden, 4096);
        // MoE L2 net config carries placement + ep + nvl.
        assert_eq!(cfgs.moe_dispatch.ep_size, 8);
        assert_eq!(cfgs.moe_dispatch.nvl_num_gpu, 8);
        assert_eq!(cfgs.moe_dispatch.top_k, 8);
        // Both legs share the same MoeNetConfig (dispatch == combine).
        assert_eq!(cfgs.moe_dispatch.ep_size, cfgs.moe_combine.ep_size);
        assert_eq!(cfgs.moe_dispatch.h, cfgs.moe_combine.h);
        // lm_head replicated full (vocab-parallel deferred).
        assert_eq!(cfgs.lm_head.n, 152064);
        assert_eq!(cfgs.lm_head.k, 4096);
    }

    #[test]
    fn resolve_shards_attn_heads_by_attn_tp_and_moe_intermediate_unchanged() {
        let model = MoeModelCfg::qwen3_235b();
        let r = resolve_configs(&build_configs(
            &model,
            &parallel(4, 8, 2, 8),
            &uniform_routing_for(&model),
        ));
        // attention per-rank under attn_tp=4: qo 64/4=16, kv 4/4=1.
        assert_eq!(r.attn_block.num_qo_heads_per_rank, 16);
        assert_eq!(r.attn_block.num_kv_heads_per_rank, 1);
        // MoE expert compute: 128 experts / ep=8 → 16 experts per GPU.
        assert_eq!(r.moe_expert_compute.experts_per_gpu, 16);
        // grouped_gemm dims unchanged by EP (moe_intermediate NOT sharded across EP).
        assert_eq!(r.moe_expert_compute.gate_up.n, 2 * 3072);
        assert_eq!(r.moe_expert_compute.gate_up.k, 4096);
        assert_eq!(r.moe_expert_compute.down.n, 4096);
        assert_eq!(r.moe_expert_compute.down.k, 3072);
    }

    #[test]
    fn total_kv_bytes_per_token_sums_across_attn_ranks_all_layers() {
        let model = MoeModelCfg::qwen3_235b();
        let r = resolve_configs(&build_configs(
            &model,
            &parallel(4, 8, 2, 8),
            &uniform_routing_for(&model),
        ));
        // 2 (kv) × 4 kv_heads × 128 head_dim × 2 (bf16) × 94 layers.
        assert_eq!(total_kv_bytes_per_token(&r), 2 * 4 * 128 * 2 * 94);
    }

    #[test]
    fn attn_tp_equal_ep_is_single_dp_group() {
        // attn_tp == ep → no attn-DP replication (one group). Qwen3-235B has
        // num_kv_heads=4, so attn_tp must be ≤ 4: pick attn_tp=ep=4.
        let model = MoeModelCfg::qwen3_235b();
        let cfgs = build_configs(&model, &parallel(4, 4, 2, 4), &uniform_routing_for(&model));
        assert_eq!(cfgs.num_dp_groups, 1);
    }

    #[test]
    fn non_uniform_routing_threads_into_moe_net_config() {
        // A skewed profile (hot top-2 experts, cold tail) reaches MoeNetConfig
        // verbatim; the L2 dispatch/combine sim consumes it. The L3 worklet's
        // `local_ppm` stays uniform regardless (v1 limitation).
        let model = MoeModelCfg::qwen3_235b();
        let mut weights = vec![0.1f32; model.num_experts as usize];
        weights[0] = 5.0;
        weights[1] = 3.0;
        let skewed = RoutingDistribution::from_profile(&weights);
        let cfgs = build_configs(&model, &parallel(4, 8, 1, 8), &skewed);
        assert_eq!(cfgs.moe_dispatch.routing, skewed);
        assert_eq!(cfgs.moe_combine.routing, skewed);
        // local_ppm still uniform: every entry equals TOTAL_PPM / num_experts.
        let per_expert = RoutingDistribution::TOTAL_PPM / model.num_experts;
        assert!(
            cfgs.moe_expert_compute
                .local_ppm
                .iter()
                .all(|&v| v == per_expert),
            "local_ppm must stay uniform in v1"
        );
    }

    #[test]
    #[should_panic(expected = "must be a multiple of attn_tp_size")]
    fn ep_not_multiple_of_attn_tp_panics() {
        let model = MoeModelCfg::qwen3_235b();
        let _ = build_configs(&model, &parallel(3, 8, 2, 8), &uniform_routing_for(&model));
    }

    #[test]
    #[should_panic(expected = "hp_size")]
    fn hp_not_dividing_ep_panics() {
        let model = MoeModelCfg::qwen3_235b();
        let _ = build_configs(&model, &parallel(4, 8, 3, 8), &uniform_routing_for(&model));
    }

    #[test]
    #[should_panic(expected = "num_experts")]
    fn ep_not_dividing_num_experts_panics() {
        // num_experts=128, ep=7 → 128 % 7 != 0.
        let model = MoeModelCfg::qwen3_235b();
        let _ = build_configs(&model, &parallel(7, 7, 1, 7), &uniform_routing_for(&model));
    }

    #[test]
    #[should_panic(expected = "routing distribution has")]
    fn routing_distribution_expert_count_mismatch_panics() {
        let model = MoeModelCfg::qwen3_235b();
        let wrong = RoutingDistribution::uniform(model.num_experts + 1);
        let _ = build_configs(&model, &parallel(4, 8, 2, 8), &wrong);
    }
}
