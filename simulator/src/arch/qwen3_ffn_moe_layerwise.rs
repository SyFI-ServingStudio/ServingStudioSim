//! `qwen3_ffn_moe_layerwise` — L4 ffn-side model_arch for AFD (attention-FFN
//! disaggregation) of a Qwen3-MoE decoder. The ffn pool's half of the
//! `qwen3_moe_dp_attn_ep_ffn` split: it owns everything EXCEPT the attention
//! kernel. Composed of two TP worklets plus the EP-MoE ops and the iteration
//! embed / final_norm / lm_head:
//!   - [`PreAttnProjTpWorklet`]  = input_norm + qkv          (pre-attn)
//!   - [`PostAttnRouterTpWorklet`] = o_proj + [tp_allreduce] + post_norm + router
//!     (post-attn dense tail + MoE gate; the post_norm + router are a composed
//!     [`MoeRouterLocalWorklet`], shared with the unified arch)
//! The attention itself is the attn side (`qwen3_attn_layerwise`).
//!
//! Per-layer cost is split at the attn boundary into two groups plus an iteration
//! prologue/epilogue, with the Bridge/Bootstrap/Terminal fused-kernel convention
//! (L4 design.md §4.1):
//!   - `pre_attn_cost(0)`        = `Max{1.0}( pre_attn × num_dp )`  (Bootstrap)
//!   - `pre_attn_cost(L > 0)`    = ZERO  (the Bridge bills pre(L) inside post(L-1))
//!   - `post_attn_cost(L)`       = post_attn + MoE, and for `L < last` additionally
//!                                 the fused pre(L+1) (Bridge); `L == last` is post-only (Terminal)
//!   - `prologue_cost`           = embed
//!   - `epilogue_cost`           = Sum(final_norm, lm_head)
//!
//! where
//!   post_attn = `Max{1.0}( PostAttnRouterTpWorklet × num_dp )`,
//!   MoE       = `Sum( dispatch, Max{1.0}(expert × ep), Max{1.0}(local_reduce × num_dp), combine )`.
//!
//! The `× num_dp_groups` MAX nodes (pre_attn, post_attn, local_reduce) are the
//! DP-attention fan-out: each shard runs the dense + routing + home-reduce path on
//! its OWN token slice (`g.batch_tokens`), so DP load imbalance is modeled exactly
//! (the section wallclock is the slowest shard, never a pooled average). post_norm
//! + router run replicated across the shard's `attn_tp_size` ranks after the o_proj
//! all-reduce — same per-shard token count as o_proj, which is why they live in one
//! worklet.
//!
//! So `Σ_L pre + Σ_L post + prologue + epilogue` totals exactly one (norm+qkv) +
//! one (o_proj+ar+post_norm+router) + one MoE (dispatch+expert+reduce+combine) per
//! layer + embed + final_norm + lm_head — i.e. the full iteration minus the
//! attention kernel (which the attn side carries). This arch is self-contained: it
//! builds its own configs from the model dims (its own `build_configs` /
//! `resolve_configs`), independent of the unified `qwen3_moe_dp_attn_ep_ffn` arch.
//!
//! Comm: this side emits `ffn_to_attn_bytes_per_token` (the QKV projection output,
//! `(q_dim + 2·kv_dim) · bpe`) per token to the attn side.
//!
//! [`MoeRouterLocalWorklet`]: crate::worklet::MoeRouterLocalWorklet

use std::sync::Arc;

use crate::arch::contract::{FfnArchInput, FfnLayerwiseModel};
use crate::arch::moe_model_cfg::MoeModelCfg;
use crate::common::Fabric;
use crate::op::moe::{MoeCombineOp, MoeDispatchOp, MoeNetConfig, MoeNetInput, Placement};
use crate::op::Op;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, RmsNormKernel,
    RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::routing::RoutingDistribution;
use crate::timing::{
    BuildError, CostManifestDoc, CostNode, CostTree, CostTreeBuilder, Dim, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::{
    uniform_local_ppm, MoeExpertComputeLocalWorklet, MoeExpertComputeLocalWorkletConfig,
    MoeExpertComputeLocalWorkletInput, MoeExpertComputeLocalWorkletResolved,
    PostAttnRouterTpWorklet, PostAttnRouterTpWorkletConfig, PostAttnRouterTpWorkletInput,
    PostAttnRouterTpWorkletResolved, PreAttnProjTpWorklet, PreAttnProjTpWorkletConfig,
    PreAttnProjTpWorkletInput, PreAttnProjTpWorkletResolved,
};

pub struct Qwen3FfnMoeLayerwiseModel {
    pub name: String,
    pub num_layers: u32,
    pub ep_size: u16,
    /// Derived at build (`ep_size / attn_tp_size`), cached for the cost-tree fan-out.
    pub num_dp_groups: u16,
    pub top_k: u32,
    pub ffn_to_attn_bytes_per_token: Dim,
    // Dense attn-adjacent worklets, both sharded over `attn_tp_size` (the head
    // split is internal to each worklet's resolve_config):
    //   pre_attn  = input_norm + qkv
    //   post_attn = o_proj + [tp_allreduce] + post_norm + router
    pre_attn: PreAttnProjTpWorklet,
    post_attn: PostAttnRouterTpWorklet,
    // MoE + iteration ops — identical to the iter-wise arch (minus the router,
    // which lives inside `post_attn`).
    moe_dispatch: MoeDispatchOp,
    moe_expert_compute: MoeExpertComputeLocalWorklet,
    moe_local_reduce: Op<ElementwiseKernel>,
    moe_combine: MoeCombineOp,
    embed: Op<ElementwiseKernel>,
    final_norm: Op<RmsNormKernel>,
    lm_head: Op<SingleGemmKernel>,
    // Compiled cost trees (one per cost method; `post` has a mid-layer variant
    // that bills the fused pre(L+1) and a last-layer variant that does not).
    pre_flat: Vec<FlatCostNode>,
    pre_n_slots: usize,
    post_mid_flat: Vec<FlatCostNode>,
    post_mid_n_slots: usize,
    post_last_flat: Vec<FlatCostNode>,
    post_last_n_slots: usize,
    prologue_flat: Vec<FlatCostNode>,
    prologue_n_slots: usize,
    epilogue_flat: Vec<FlatCostNode>,
    epilogue_n_slots: usize,
}

// Backend / fabric policy for this arch. Local copy — the AFD archs are
// self-contained (no shared arch-level config).
const NORM_BACKENDS: &[&str] = &["flashinfer"];
// GEMM backends are dtype-driven, not a flat const: fp8 uses DeepGEMM; bf16
// dense GEMMs compare both Torch layouts while grouped experts stay `torch`.
const ACT_BACKENDS: &[&str] = &["triton"];
// nccl + nvshmem for the collective/p2p ops: the cost engine evals both and
// keeps the faster per op (best-of-N). MoE dispatch/combine (p2p_intra) and the
// router TP all-reduce both benefit — real EP deployments run these over NVSHMEM.
const ALLREDUCE_BACKENDS: &[&str] = &["nccl", "nvshmem"];
const P2P_BACKENDS: &[&str] = &["nccl", "nvshmem"];
const TP_FABRIC: Fabric = Fabric::Nvlink;
const MOE_INTRA_FABRIC: Fabric = Fabric::Nvlink;
const MOE_INTER_FABRIC: Fabric = Fabric::Infiniband;

/// FFN-side numeric parallel input. `attn_tp_size` drives the dense qkv / o_proj
/// per-rank shape (must match the paired attn arch) AND the MoE combine residing-
/// group width — a token resides on its qkv/o_proj TP group (post-allreduce every
/// `attn_tp_size` rank holds it), so combine fans the reduced output back to all of
/// them; `ep_size` is the expert-parallel width; `nvl_num_gpu` partitions the EP
/// ranks into NVL domains.
#[derive(Clone, Debug)]
pub struct Qwen3FfnMoeParallel {
    pub attn_tp_size: u16,
    pub ep_size: u16,
    pub nvl_num_gpu: u16,
    pub gpu_name: String,
}

/// Raw worklet/op configs for the ffn side. `num_dp_groups = ep_size / attn_tp_size`
/// is carried for the per-DP-shard fan-out (pre_attn / post_attn / local_reduce).
pub struct Qwen3FfnMoeConfigs {
    pub pre_attn: PreAttnProjTpWorkletConfig,
    pub post_attn: PostAttnRouterTpWorkletConfig,
    pub moe_dispatch: MoeNetConfig,
    pub moe_expert_compute: MoeExpertComputeLocalWorkletConfig,
    pub moe_local_reduce: ElementwiseKernelConfig,
    pub moe_combine: MoeNetConfig,
    pub embed: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub num_layers: u32,
    pub ep_size: u16,
    pub num_dp_groups: u16,
    pub top_k: u32,
    pub ffn_to_attn_bytes_per_token: Dim,
}

/// Post-resolve aggregate; the atomic ops (embed / final_norm / lm_head /
/// moe_local_reduce) carry their kernel config through unchanged (only the
/// worklets have a resolve step).
pub struct Qwen3FfnMoeResolved {
    pub pre_attn: PreAttnProjTpWorkletResolved,
    pub post_attn: PostAttnRouterTpWorkletResolved,
    pub moe_dispatch: MoeNetConfig,
    pub moe_expert_compute: MoeExpertComputeLocalWorkletResolved,
    pub moe_local_reduce: ElementwiseKernelConfig,
    pub moe_combine: MoeNetConfig,
    pub embed: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub num_layers: u32,
    pub ep_size: u16,
    pub num_dp_groups: u16,
    pub top_k: u32,
    pub ffn_to_attn_bytes_per_token: Dim,
}

pub fn build_configs(
    model: &MoeModelCfg,
    parallel: &Qwen3FfnMoeParallel,
    routing: &RoutingDistribution,
) -> Qwen3FfnMoeConfigs {
    let gpu = &parallel.gpu_name;
    // Byte-transfer widths (handoffs, dispatch/combine, local reduce, embed) use
    // the compute dtype: fp8 halves the wire size, bf16 leaves it unchanged. The
    // RMSNorm ops below keep the base `model.dtype`.
    let dtype_bytes = model.compute_dtype().size_bytes();
    let bytes = Dim::param("bytes", dtype_bytes);
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
    // Derived, not configured: DP shards the ffn side pools.
    let num_dp_groups = parallel.ep_size / parallel.attn_tp_size;
    assert!(parallel.nvl_num_gpu > 0, "nvl_num_gpu must be non-zero");
    assert!(
        model.num_experts.get() % u32::from(parallel.ep_size) == 0,
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
    let local_ppm = uniform_local_ppm(model.num_experts.get(), parallel.ep_size);
    let moe_net = MoeNetConfig {
        backends: P2P_BACKENDS.to_vec(),
        gpu_name: gpu.clone(),
        // dispatch/combine ship the hidden activation → compute dtype (fp8 in an
        // fp8 run halves the all-to-all payload).
        dtype: model.compute_dtype(),
        intra_fabric: MOE_INTRA_FABRIC,
        inter_fabric: MOE_INTER_FABRIC,
        ep_size: u32::from(parallel.ep_size),
        nvl_num_gpu: u32::from(parallel.nvl_num_gpu),
        // Comm-sizing seam: the MoE all-to-all payload is byte-keyed, so fold the
        // symbolic hidden to a concrete width here.
        h: model.hidden.get(),
        top_k: model.top_k,
        routing: routing.clone(),
        // A token resides on its qkv/o_proj TP group (post-allreduce every attn_tp
        // rank holds it), so the MoE combine fans the reduced output back to all
        // `attn_tp_size` ranks.
        placement: Placement::ReplicatedHeadParallel {
            hp_size: u32::from(parallel.attn_tp_size),
        },
    };
    // ffn → attn handoff: the QKV projection output, (q + 2·kv)·head_dim·bpe per
    // token (full, un-sharded wire size).
    let ffn_to_attn_bytes_per_token = (model.num_qo_heads.clone() + 2 * model.num_kv_heads.clone())
        * model.head_dim.clone()
        * bytes.clone();
    Qwen3FfnMoeConfigs {
        // pre-attn: input_norm + column-parallel qkv (head split over attn_tp).
        pre_attn: PreAttnProjTpWorkletConfig {
            hidden: model.hidden.clone(),
            num_qo_heads: model.num_qo_heads.clone(),
            num_kv_heads: model.num_kv_heads.clone(),
            head_dim: model.head_dim.clone(),
            dtype: model.dtype,
            compute_dtype: model.compute_dtype(),
            tp_size: parallel.attn_tp_size,
            tp_name: "attn_tp",
            gpu_name: gpu.clone(),
            norm_backends: NORM_BACKENDS.to_vec(),
            gemm_backends: model.single_gemm_backends(),
        },
        // post-attn: row-parallel o_proj + optional tp_allreduce + post_norm + router.
        post_attn: PostAttnRouterTpWorkletConfig {
            hidden: model.hidden.clone(),
            num_qo_heads: model.num_qo_heads.clone(),
            head_dim: model.head_dim.clone(),
            num_experts: model.num_experts.clone(),
            dtype: model.dtype,
            compute_dtype: model.compute_dtype(),
            tp_size: parallel.attn_tp_size,
            tp_name: "attn_tp",
            allreduce_fabric: TP_FABRIC,
            gpu_name: gpu.clone(),
            norm_backends: NORM_BACKENDS.to_vec(),
            gemm_backends: model.single_gemm_backends(),
            allreduce_backends: ALLREDUCE_BACKENDS.to_vec(),
        },
        moe_dispatch: moe_net.clone(),
        // MoE expert compute is all-fp8 in an fp8 run (grouped GEMMs + SwiGLU act);
        // no RMSNorm inside, so the whole worklet takes the compute dtype.
        moe_expert_compute: MoeExpertComputeLocalWorkletConfig {
            hidden: model.hidden.clone(),
            moe_intermediate: model.moe_intermediate.clone(),
            num_experts: model.num_experts.clone(),
            ep_size: parallel.ep_size,
            dtype: model.compute_dtype(),
            gpu_name: gpu.clone(),
            act_backends: ACT_BACKENDS.to_vec(),
            grouped_gemm_backends: model.grouped_gemm_backends(),
            local_ppm,
        },
        moe_local_reduce: ElementwiseKernelConfig {
            backends: ACT_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            input_bytes_per_token: model.hidden.clone() * bytes.clone(),
            output_bytes_per_token: model.hidden.clone() * bytes.clone(),
        },
        moe_combine: moe_net,
        embed: ElementwiseKernelConfig {
            backends: ACT_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            input_bytes_per_token: model.hidden.clone() * bytes.clone(),
            output_bytes_per_token: model.hidden.clone() * bytes.clone(),
        },
        final_norm: RmsNormKernelConfig {
            backends: NORM_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden: model.hidden.clone(),
            dtype: model.dtype,
        },
        lm_head: SingleGemmKernelConfig {
            backends: model.single_gemm_backends(),
            gpu_name: gpu.clone(),
            n: model.vocab.clone(),
            k: model.hidden.clone(),
            dtype: model.compute_dtype(),
        },
        num_layers: model.num_layers,
        ep_size: parallel.ep_size,
        num_dp_groups,
        top_k: model.top_k,
        ffn_to_attn_bytes_per_token,
    }
}

pub fn resolve_configs(cfgs: &Qwen3FfnMoeConfigs) -> Qwen3FfnMoeResolved {
    Qwen3FfnMoeResolved {
        pre_attn: PreAttnProjTpWorklet::resolve_config(&cfgs.pre_attn),
        post_attn: PostAttnRouterTpWorklet::resolve_config(&cfgs.post_attn),
        moe_dispatch: cfgs.moe_dispatch.clone(),
        moe_expert_compute: MoeExpertComputeLocalWorklet::resolve_config(&cfgs.moe_expert_compute),
        moe_local_reduce: cfgs.moe_local_reduce.clone(),
        moe_combine: cfgs.moe_combine.clone(),
        embed: cfgs.embed.clone(),
        final_norm: cfgs.final_norm.clone(),
        lm_head: cfgs.lm_head.clone(),
        num_layers: cfgs.num_layers,
        ep_size: cfgs.ep_size,
        num_dp_groups: cfgs.num_dp_groups,
        top_k: cfgs.top_k,
        ffn_to_attn_bytes_per_token: cfgs.ffn_to_attn_bytes_per_token.clone(),
    }
}

/// Build the ffn-side model from its own resolved configs ([`resolve_configs`]):
/// the pre/post-attn worklets supply the dense projections + routing, and the MoE
/// / iteration ops are built from their per-rank configs. Self-contained — no
/// dependency on the unified `qwen3_moe_dp_attn_ep_ffn` arch.
pub fn build(
    model_name: String,
    resolved: Qwen3FfnMoeResolved,
    bridge: &PerfApiBridge,
) -> Result<Qwen3FfnMoeLayerwiseModel, BuildError> {
    let num_layers = resolved.num_layers;
    let ep_size = resolved.ep_size;
    let num_dp_groups = resolved.num_dp_groups;
    let top_k = resolved.top_k;
    let ffn_to_attn_bytes_per_token = resolved.ffn_to_attn_bytes_per_token;

    // Dense attn-adjacent worklets.
    let pre_attn =
        PreAttnProjTpWorklet::build(format!("{model_name}.pre_attn"), resolved.pre_attn, bridge)?;
    let post_attn = PostAttnRouterTpWorklet::build(
        format!("{model_name}.post_attn"),
        resolved.post_attn,
        bridge,
    )?;

    // MoE + embed / final_norm / lm_head.
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
    let local_reduce_name = format!("{model_name}.moe_local_reduce");
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
    let embed_name = format!("{model_name}.embedding");
    let embed = Op::new(
        embed_name.clone(),
        Arc::new(ElementwiseKernel::build(
            embed_name,
            resolved.embed,
            bridge,
        )?),
    );
    let final_norm_name = format!("{model_name}.final_norm");
    let final_norm = Op::new(
        final_norm_name.clone(),
        Arc::new(RmsNormKernel::build(
            final_norm_name,
            resolved.final_norm,
            bridge,
        )?),
    );
    let lm_head_name = format!("{model_name}.lm_head");
    let lm_head = Op::new(
        lm_head_name.clone(),
        Arc::new(SingleGemmKernel::build(
            lm_head_name,
            resolved.lm_head,
            bridge,
        )?),
    );

    let mut model = Qwen3FfnMoeLayerwiseModel {
        name: model_name,
        num_layers,
        ep_size,
        num_dp_groups,
        top_k,
        ffn_to_attn_bytes_per_token,
        pre_attn,
        post_attn,
        moe_dispatch,
        moe_expert_compute,
        moe_local_reduce,
        moe_combine,
        embed,
        final_norm,
        lm_head,
        pre_flat: Vec::new(),
        pre_n_slots: 0,
        post_mid_flat: Vec::new(),
        post_mid_n_slots: 0,
        post_last_flat: Vec::new(),
        post_last_n_slots: 0,
        prologue_flat: Vec::new(),
        prologue_n_slots: 0,
        epilogue_flat: Vec::new(),
        epilogue_n_slots: 0,
    };

    let pre = model.compile_pre_tree();
    model.pre_flat = pre.flatten();
    model.pre_n_slots = pre.n_slots();
    let post_mid = model.compile_post_tree(true);
    model.post_mid_flat = post_mid.flatten();
    model.post_mid_n_slots = post_mid.n_slots();
    let post_last = model.compile_post_tree(false);
    model.post_last_flat = post_last.flatten();
    model.post_last_n_slots = post_last.n_slots();
    let prologue = model.compile_prologue_tree();
    model.prologue_flat = prologue.flatten();
    model.prologue_n_slots = prologue.n_slots();
    let epilogue = model.compile_epilogue_tree();
    model.epilogue_flat = epilogue.flatten();
    model.epilogue_n_slots = epilogue.n_slots();

    // Build-time cost-tree printout, mirroring the iter-wise archs' `[build] cost
    // tree` line. Layer-wise cost is split into five sub-trees (pre / post_mid /
    // post_last / prologue / epilogue), so each prints under its own label.
    for (label, tree) in [
        ("pre", &pre),
        ("post_mid", &post_mid),
        ("post_last", &post_last),
        ("prologue", &prologue),
        ("epilogue", &epilogue),
    ] {
        tracing::info!(
            "[build] {} cost tree [{}] ({} leaf slots):\n{}",
            model.name,
            label,
            tree.n_slots(),
            tree.describe()
        );
    }

    Ok(model)
}

impl Qwen3FfnMoeLayerwiseModel {
    // ── compile: each node mints slots in the order its eval helper fills them ──

    /// Pre-attn section: `Max{1.0}( pre_attn × num_dp_groups )` — one input_norm +
    /// qkv per DP shard, on the shard's own tokens.
    fn compile_pre_node(&self, b: &mut CostTreeBuilder) -> CostNode {
        let groups: Vec<CostNode> = (0..self.num_dp_groups)
            .map(|_| self.pre_attn.compile(b))
            .collect();
        CostNode::Max {
            overlap: 1.0,
            children: groups,
        }
    }

    /// Post-attn section: `Max{1.0}( post_attn × num_dp_groups )` — o_proj +
    /// [tp_allreduce] + post_norm + router per DP shard, on the shard's own tokens.
    fn compile_post_attn_node(&self, b: &mut CostTreeBuilder) -> CostNode {
        let groups: Vec<CostNode> = (0..self.num_dp_groups)
            .map(|_| self.post_attn.compile(b))
            .collect();
        CostNode::Max {
            overlap: 1.0,
            children: groups,
        }
    }

    /// MoE section: `Sum( dispatch, Max{1.0}(expert × ep_size),
    /// Max{1.0}(local_reduce × num_dp_groups), combine )`. The router has moved
    /// into `post_attn`; the home reduce is per DP shard (combine returns each
    /// token to its home shard).
    fn compile_moe_node(&self, b: &mut CostTreeBuilder) -> CostNode {
        let moe_dispatch = self.moe_dispatch.compile(b);
        let expert_groups: Vec<CostNode> = (0..self.ep_size)
            .map(|_| self.moe_expert_compute.compile(b))
            .collect();
        let expert_fanout = CostNode::Max {
            overlap: 1.0,
            children: expert_groups,
        };
        let reduce_groups: Vec<CostNode> = (0..self.num_dp_groups)
            .map(|_| self.moe_local_reduce.compile(b))
            .collect();
        let reduce_fanout = CostNode::Max {
            overlap: 1.0,
            children: reduce_groups,
        };
        let moe_combine = self.moe_combine.compile(b);
        CostNode::Sum(vec![
            moe_dispatch,
            expert_fanout,
            reduce_fanout,
            moe_combine,
        ])
    }

    /// One-line arch identity for the `cost_log` manifest root label (the
    /// breakdown/trace header), mirroring the iter-wise archs' root-label style.
    /// Every section's root carries it, so each AFD building block names its arch.
    fn arch_label(&self) -> String {
        format!(
            "{} [AFD ffn (ep={}, dp={}) MoE top_k={}, {} layers]",
            self.name, self.ep_size, self.num_dp_groups, self.top_k, self.num_layers
        )
    }

    fn compile_pre_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let root = self.compile_pre_node(&mut b);
        b.finish(CostNode::Labeled {
            label: self.arch_label(),
            child: Box::new(root),
        })
    }

    /// Post-attn cost tree. `with_fused_pre` (mid-layer Bridge) appends the fused
    /// pre-attn of the next layer; the last layer (Terminal) omits it.
    fn compile_post_tree(&self, with_fused_pre: bool) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let mut parts = vec![
            self.compile_post_attn_node(&mut b),
            self.compile_moe_node(&mut b),
        ];
        if with_fused_pre {
            parts.push(self.compile_pre_node(&mut b));
        }
        b.finish(CostNode::Labeled {
            label: self.arch_label(),
            child: Box::new(CostNode::Sum(parts)),
        })
    }

    fn compile_prologue_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        // Per DP shard: each group embeds its own tokens in parallel → Max over
        // num_dp_groups (mirrors pre_attn/post_attn/local_reduce).
        let groups: Vec<CostNode> = (0..self.num_dp_groups)
            .map(|_| self.embed.compile(&mut b))
            .collect();
        let root = CostNode::Max {
            overlap: 1.0,
            children: groups,
        };
        b.finish(CostNode::Labeled {
            label: self.arch_label(),
            child: Box::new(root),
        })
    }

    fn compile_epilogue_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        // Per DP shard: each group runs final_norm + lm_head on its own tokens in
        // parallel → Max over num_dp_groups (the epilogue is not vocab-TP sharded,
        // so each shard computes the full vocab for its local tokens).
        let groups: Vec<CostNode> = (0..self.num_dp_groups)
            .map(|_| {
                CostNode::Sum(vec![
                    self.final_norm.compile(&mut b),
                    self.lm_head.compile(&mut b),
                ])
            })
            .collect();
        let root = CostNode::Max {
            overlap: 1.0,
            children: groups,
        };
        b.finish(CostNode::Labeled {
            label: self.arch_label(),
            child: Box::new(root),
        })
    }

    // ── eval helpers: fill slots in the EXACT child order `compile` minted them ──

    fn eval_pre(&self, batch: &FfnArchInput, ev: &mut Evaluator) {
        for &batch_tokens in &batch.tokens_per_group {
            self.pre_attn
                .eval(&PreAttnProjTpWorkletInput { batch_tokens }, ev);
        }
    }

    fn eval_post_attn(&self, batch: &FfnArchInput, ev: &mut Evaluator) {
        for &batch_tokens in &batch.tokens_per_group {
            self.post_attn
                .eval(&PostAttnRouterTpWorkletInput { batch_tokens }, ev);
        }
    }

    fn eval_moe(&self, batch: &FfnArchInput, ev: &mut Evaluator) {
        let m_total: u32 = batch.tokens_per_group.iter().sum();
        let global_expert_selections = m_total * self.top_k;
        let tokens_for_comm = u64::from(m_total);
        self.moe_dispatch.eval(
            &MoeNetInput {
                tokens: tokens_for_comm,
            },
            ev,
        );
        for _ in 0..self.ep_size {
            self.moe_expert_compute.eval(
                &MoeExpertComputeLocalWorkletInput {
                    global_expert_selections,
                },
                ev,
            );
        }
        // Home reduce: per DP shard on the shard's own tokens.
        for &num_tokens in &batch.tokens_per_group {
            self.moe_local_reduce
                .eval(&ElementwiseKernelInput { num_tokens }, ev);
        }
        self.moe_combine.eval(
            &MoeNetInput {
                tokens: tokens_for_comm,
            },
            ev,
        );
    }

    // ── shared cost bodies (used by both `*_cost` and `*_cost_with_inputs`) ──────

    /// Build an evaluator over `slots`, capturing each leaf's typed input into
    /// `inputs` when `Some` (cleared first so it stays slot-aligned). Collapses the
    /// iter-wise `eval_iter` / `eval_iter_with_inputs` split into one body.
    fn evaluator<'a>(
        slots: &'a mut [LeafMetrics],
        inputs: Option<&'a mut Vec<SlotInput>>,
    ) -> Evaluator<'a> {
        match inputs {
            Some(inp) => {
                inp.clear();
                Evaluator::with_inputs(slots, inp)
            }
            None => Evaluator::new(slots),
        }
    }

    fn pre_attn_eval(
        &self,
        layer_idx: usize,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: Option<&mut Vec<SlotInput>>,
    ) -> LeafMetrics {
        slots.clear();
        if layer_idx == 0 {
            // Bootstrap: the real qkv of layer 0.
            slots.resize(self.pre_n_slots, LeafMetrics::ZERO);
            let mut ev = Self::evaluator(slots, inputs);
            self.eval_pre(batch, &mut ev);
            debug_assert_eq!(
                ev.filled(),
                self.pre_n_slots,
                "eval cursor must fill every slot"
            );
            CostTree::aggregate(&self.pre_flat, slots, scratch)
        } else {
            // Bridge bills pre(L) inside post(L-1); standalone pre_attn(L>0) is zero
            // (no slots, so the captured inputs are empty too).
            if let Some(i) = inputs {
                i.clear();
            }
            LeafMetrics::ZERO
        }
    }

    fn post_attn_eval(
        &self,
        layer_idx: usize,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: Option<&mut Vec<SlotInput>>,
    ) -> LeafMetrics {
        let last = self.num_layers.saturating_sub(1) as usize;
        slots.clear();
        if layer_idx < last {
            // Bridge: post(L) + the fused pre(L+1).
            slots.resize(self.post_mid_n_slots, LeafMetrics::ZERO);
            let mut ev = Self::evaluator(slots, inputs);
            self.eval_post_attn(batch, &mut ev);
            self.eval_moe(batch, &mut ev);
            self.eval_pre(batch, &mut ev);
            debug_assert_eq!(
                ev.filled(),
                self.post_mid_n_slots,
                "eval cursor must fill every slot"
            );
            CostTree::aggregate(&self.post_mid_flat, slots, scratch)
        } else {
            // Terminal: post-only.
            slots.resize(self.post_last_n_slots, LeafMetrics::ZERO);
            let mut ev = Self::evaluator(slots, inputs);
            self.eval_post_attn(batch, &mut ev);
            self.eval_moe(batch, &mut ev);
            debug_assert_eq!(
                ev.filled(),
                self.post_last_n_slots,
                "eval cursor must fill every slot"
            );
            CostTree::aggregate(&self.post_last_flat, slots, scratch)
        }
    }

    fn prologue_eval(
        &self,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: Option<&mut Vec<SlotInput>>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.prologue_n_slots, LeafMetrics::ZERO);
        let mut ev = Self::evaluator(slots, inputs);
        // One embed branch per DP shard, on the shard's own tokens (Max collapses
        // to the slowest shard). Order matches `compile_prologue_tree`.
        for &num_tokens in &batch.tokens_per_group {
            self.embed
                .eval(&ElementwiseKernelInput { num_tokens }, &mut ev);
        }
        debug_assert_eq!(
            ev.filled(),
            self.prologue_n_slots,
            "eval cursor must fill every slot"
        );
        CostTree::aggregate(&self.prologue_flat, slots, scratch)
    }

    fn epilogue_eval(
        &self,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: Option<&mut Vec<SlotInput>>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.epilogue_n_slots, LeafMetrics::ZERO);
        let mut ev = Self::evaluator(slots, inputs);
        // One (final_norm, lm_head) branch per DP shard, on the shard's own tokens
        // (Max collapses to the slowest shard). Order matches `compile_epilogue_tree`.
        for &m in &batch.tokens_per_group {
            self.final_norm.eval(&RmsNormKernelInput { m }, &mut ev);
            self.lm_head.eval(&SingleGemmKernelInput { m }, &mut ev);
        }
        debug_assert_eq!(
            ev.filled(),
            self.epilogue_n_slots,
            "eval cursor must fill every slot"
        );
        CostTree::aggregate(&self.epilogue_flat, slots, scratch)
    }
}

impl FfnLayerwiseModel for Qwen3FfnMoeLayerwiseModel {
    fn num_layers(&self) -> u32 {
        self.num_layers
    }

    /// One ffn replica spans the EP group — `ep_size` GPUs.
    fn gpus_per_replica(&self) -> u16 {
        self.ep_size
    }

    /// DP shards pooled by the ffn side (derived `ep_size / attn_tp_size` at build).
    fn num_dp_groups(&self) -> u16 {
        self.num_dp_groups
    }

    fn ffn_to_attn_bytes_per_token(&self) -> u64 {
        self.ffn_to_attn_bytes_per_token.get() as u64
    }

    fn pre_attn_cost(
        &self,
        layer_idx: usize,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        self.pre_attn_eval(layer_idx, batch, slots, scratch, None)
    }

    fn pre_attn_cost_with_inputs(
        &self,
        layer_idx: usize,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        self.pre_attn_eval(layer_idx, batch, slots, scratch, Some(inputs))
    }

    fn post_attn_cost(
        &self,
        layer_idx: usize,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        self.post_attn_eval(layer_idx, batch, slots, scratch, None)
    }

    fn post_attn_cost_with_inputs(
        &self,
        layer_idx: usize,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        self.post_attn_eval(layer_idx, batch, slots, scratch, Some(inputs))
    }

    fn prologue_cost(
        &self,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        self.prologue_eval(batch, slots, scratch, None)
    }

    fn prologue_cost_with_inputs(
        &self,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        self.prologue_eval(batch, slots, scratch, Some(inputs))
    }

    fn epilogue_cost(
        &self,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        self.epilogue_eval(batch, slots, scratch, None)
    }

    fn epilogue_cost_with_inputs(
        &self,
        batch: &FfnArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        self.epilogue_eval(batch, slots, scratch, Some(inputs))
    }

    /// One section per distinct CostTree. `post_attn` (the mid-layer Bridge, with
    /// the fused pre of the next layer) and `post_attn_last` (the Terminal, post
    /// only) are separate sections because they have different slot sets — a
    /// `cost_log` row tags itself with whichever it cost. Recompiled here (only at
    /// logger open), not on the per-layer cost path.
    fn cost_log_manifest(&self) -> CostManifestDoc {
        let mut doc = CostManifestDoc::empty();
        doc.push("prologue", self.compile_prologue_tree().manifest());
        doc.push("pre_attn", self.compile_pre_tree().manifest());
        doc.push("post_attn", self.compile_post_tree(true).manifest());
        doc.push("post_attn_last", self.compile_post_tree(false).manifest());
        doc.push("epilogue", self.compile_epilogue_tree().manifest());
        doc
    }
}
