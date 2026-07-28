//! `kimi_k3_kda_mla` — L4 model_arch for Kimi-K3-style **heterogeneous**
//! decoders: 24 MLA (compressed-KV full-attention) layers + 69 KDA (gated
//! linear-attention) layers, every layer carrying a 896-expert top-16 MoE FFN
//! plus 2 always-on shared experts. Attention is pure **DP** (`attn_tp_size`
//! fixed at 1 — the MLA absorbed-weight decode has ONE shared compressed KV
//! head, which cannot be TP-split), with `num_dp_groups = ep_size` shards; the
//! MoE experts are distributed across all `ep_size` ranks via the L2 MoE
//! dispatch / combine ops (mirroring `qwen3_moe_dp_attn_ep_ffn`).
//!
//! Cost shape (iter-wise; two heterogeneous `Scale` groups in the outer Sum):
//!
//! ```text
//! Sum(
//!   embed,
//!   Scale{mla_layers}( Sum(                     // 24 MLA layers
//!     Max{1.0}( mla_block_local × dp ),
//!     <moe body>                                // shared with the KDA template
//!   )),
//!   Scale{kda_layers}( Sum(                     // 69 KDA layers
//!     Max{1.0}( kda_block_local × dp ),
//!     <moe body>
//!   )),
//!   final_norm,
//!   lm_head,
//! )
//! <moe body> = Sum(
//!   Max{1.0}( moe_router_local × dp ),
//!   moe_dispatch,                               // L2 op, 2 leaves
//!   Max{1.0}( moe_expert_compute_local × ep ),
//!   Max{1.0}( moe_local_reduce × dp ),
//!   moe_combine,                                // L2 op, 4 leaves
//!   Max{1.0}( Sum(shared_gate_up, shared_act, shared_down) × dp ),
//! )
//! ```
//!
//! The MoE worklet/op INSTANCES are shared between the two layer templates —
//! each template's `compile` mints its own fresh slots, and `eval_into` streams
//! both bodies (one MLA, one KDA) per iteration; the two `Scale` folds supply
//! ×24 / ×69.
//!
//! v1 deviations / approximations (beyond the qwen3 arch's carried-over set):
//!   - **Layer 0's dense FFN** (`first_k_dense_replace = 1`, intermediate
//!     33792) is approximated as one more MoE layer — the layer split stays
//!     24 MLA + 69 KDA with a homogeneous MoE FFN everywhere. Exact layer-0
//!     handling would need a third Scale group for one layer; deferred.
//!   - **Routed experts run at the reduced `expert_hidden` (3584)**: the expert
//!     grouped GEMMs are `k=3584, n=3072` / `k=3072, n=3584` and the MoE
//!     dispatch/combine payload ships the reduced 3584-wide activation. The
//!     hidden(7168)↔expert_hidden(3584) boundary mapping is assumed folded
//!     into the expert GEMM dims (no separate projection op) until the
//!     reference implementation is public.
//!   - Shared experts (2 × dense FFN at `moe_intermediate` on the FULL hidden)
//!     are fused into single gate_up/act/down leaves per DP shard.
//!   - KDA per-request recurrent state (`num_heads·head_dim²` per layer) is
//!     constant per request, NOT per-token — it is excluded from
//!     `total_kv_bytes_per_token()` (and from KvPool sizing).

use std::sync::Arc;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::kimi_model_cfg::KimiModelCfg;
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
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::{
    uniform_local_ppm, KdaBlockLocalWorklet, KdaBlockLocalWorkletConfig, KdaBlockLocalWorkletInput,
    KdaBlockLocalWorkletResolved, MlaBlockLocalWorklet, MlaBlockLocalWorkletConfig,
    MlaBlockLocalWorkletInput, MlaBlockLocalWorkletResolved, MoeExpertComputeLocalWorklet,
    MoeExpertComputeLocalWorkletConfig, MoeExpertComputeLocalWorkletInput,
    MoeExpertComputeLocalWorkletResolved, MoeRouterLocalWorklet, MoeRouterLocalWorkletConfig,
    MoeRouterLocalWorkletInput, MoeRouterLocalWorkletResolved,
};

const NORM_BACKENDS: &[&str] = &["flashinfer"];
const ACT_BACKENDS: &[&str] = &["triton"];
// See llama3_dense: FlashInfer impls registered under fa2/fa3, not "flashinfer".
const ATTN_BACKENDS: &[&str] = &["fa2", "fa3"];
// The KDA chunked scan — registered Rust-side, profiled in a later phase.
const KDA_SCAN_BACKENDS: &[&str] = &["triton"];
const P2P_BACKENDS: &[&str] = &["nccl"];
const MOE_INTRA_FABRIC: Fabric = Fabric::Nvlink;
const MOE_INTER_FABRIC: Fabric = Fabric::Infiniband;

/// This arch's numeric parallel input. `attn_tp_size` MUST be 1 (MLA MQA
/// decode cannot TP-split its single KV head) — kept as an explicit field so a
/// misconfiguration fails loudly instead of silently ignoring the knob;
/// `num_dp_groups = ep_size / attn_tp_size = ep_size`. `hp_size` / `nvl_num_gpu`
/// mirror `Qwen3MoeParallel` (MoE placement / NVL-domain split).
#[derive(Clone, Debug)]
pub struct KimiK3Parallel {
    pub attn_tp_size: u16,
    pub ep_size: u16,
    pub hp_size: u16,
    pub nvl_num_gpu: u16,
    pub gpu_name: String,
}

/// Raw worklet/op configs — one field per slot + the layer split.
pub struct KimiK3KdaMlaConfigs {
    pub mla_block: MlaBlockLocalWorkletConfig,
    pub kda_block: KdaBlockLocalWorkletConfig,
    pub moe_router: MoeRouterLocalWorkletConfig,
    pub moe_dispatch: MoeNetConfig,
    pub moe_expert_compute: MoeExpertComputeLocalWorkletConfig,
    pub moe_local_reduce: ElementwiseKernelConfig,
    pub moe_combine: MoeNetConfig,
    pub shared_gate_up: SingleGemmKernelConfig,
    pub shared_act: ElementwiseKernelConfig,
    pub shared_down: SingleGemmKernelConfig,
    pub embed: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub mla_layers: u32,
    pub kda_layers: u32,
    pub attn_tp_size: u16,
    pub ep_size: u16,
    pub num_dp_groups: u16,
    pub hp_size: u16,
    pub nvl_num_gpu: u16,
    pub top_k: u32,
    pub num_experts: u32,
}

/// Post-resolve aggregate; atomic ops carry their kernel config unchanged.
pub struct KimiK3KdaMlaResolved {
    pub mla_block: MlaBlockLocalWorkletResolved,
    pub kda_block: KdaBlockLocalWorkletResolved,
    pub moe_router: MoeRouterLocalWorkletResolved,
    pub moe_dispatch: MoeNetConfig,
    pub moe_expert_compute: MoeExpertComputeLocalWorkletResolved,
    pub moe_local_reduce: ElementwiseKernelConfig,
    pub moe_combine: MoeNetConfig,
    pub shared_gate_up: SingleGemmKernelConfig,
    pub shared_act: ElementwiseKernelConfig,
    pub shared_down: SingleGemmKernelConfig,
    pub embed: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
    pub mla_layers: u32,
    pub kda_layers: u32,
    pub attn_tp_size: u16,
    pub ep_size: u16,
    pub num_dp_groups: u16,
    pub hp_size: u16,
    pub nvl_num_gpu: u16,
    pub top_k: u32,
    pub num_experts: u32,
}

pub struct KimiK3KdaMlaModel {
    pub name: String,
    pub mla_layers: u32,
    pub kda_layers: u32,
    pub attn_tp_size: u16,
    pub ep_size: u16,
    pub num_dp_groups: u16,
    pub hp_size: u16,
    pub nvl_num_gpu: u16,
    pub top_k: u32,
    pub num_experts: u32,
    pub total_kv_bytes_per_token: u64,
    pub mla_block: MlaBlockLocalWorklet,
    pub kda_block: KdaBlockLocalWorklet,
    pub moe_router: MoeRouterLocalWorklet,
    pub moe_dispatch: MoeDispatchOp,
    pub moe_expert_compute: MoeExpertComputeLocalWorklet,
    pub moe_local_reduce: Op<ElementwiseKernel>,
    pub moe_combine: MoeCombineOp,
    pub shared_gate_up: Op<SingleGemmKernel>,
    pub shared_act: Op<ElementwiseKernel>,
    pub shared_down: Op<SingleGemmKernel>,
    pub embed: Op<ElementwiseKernel>,
    pub final_norm: Op<RmsNormKernel>,
    pub lm_head: Op<SingleGemmKernel>,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build_configs(
    model: &KimiModelCfg,
    parallel: &KimiK3Parallel,
    routing: &RoutingDistribution,
) -> KimiK3KdaMlaConfigs {
    let gpu = &parallel.gpu_name;
    let dtype_bytes = model.compute_dtype().size_bytes();
    assert!(
        parallel.attn_tp_size == 1,
        "kimi_k3_kda_mla: attn_tp_size must be 1 — MLA absorbed-weight decode \
         has ONE shared compressed KV head (MQA), which cannot be TP-split; \
         use DP attention (got attn_tp_size={})",
        parallel.attn_tp_size,
    );
    assert!(parallel.ep_size > 0, "ep_size must be non-zero");
    let num_dp_groups = parallel.ep_size; // attn_tp = 1 → one shard per rank.
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
    let local_ppm = uniform_local_ppm(model.num_experts, parallel.ep_size);
    let moe_net = MoeNetConfig {
        backends: P2P_BACKENDS.to_vec(),
        gpu_name: gpu.clone(),
        dtype: model.compute_dtype(),
        intra_fabric: MOE_INTRA_FABRIC,
        inter_fabric: MOE_INTER_FABRIC,
        ep_size: u32::from(parallel.ep_size),
        nvl_num_gpu: u32::from(parallel.nvl_num_gpu),
        // Routed activations ship in the REDUCED expert hidden space (3584),
        // not the model hidden (see module-header approximation note).
        h: model.expert_hidden,
        top_k: model.top_k,
        routing: routing.clone(),
        placement: Placement::ReplicatedHeadParallel {
            hp_size: u32::from(parallel.hp_size),
        },
    };
    // Shared experts: `n_shared` dense FFNs at `moe_intermediate` on the FULL
    // hidden, always-on for every token — fused into one gate_up/act/down trio.
    let shared_inter = model.n_shared_experts * model.moe_intermediate;
    KimiK3KdaMlaConfigs {
        mla_block: MlaBlockLocalWorkletConfig {
            hidden: model.hidden,
            num_heads: model.num_heads,
            head_dim: model.head_dim,
            qk_head_dim: model.qk_head_dim(),
            q_lora_rank: model.q_lora_rank,
            kv_compressed_dim: model.kv_compressed_dim(),
            dtype: model.dtype,
            fp8: model.fp8,
            gpu_name: gpu.clone(),
            norm_backends: NORM_BACKENDS.to_vec(),
            gemm_backends: model.single_gemm_backends(),
            attn_backends: ATTN_BACKENDS.to_vec(),
            kv_cache_append_backends: vec!["vllm_cuda"],
            kv_cache_block_size: 16,
            kv_cache_layout: "NHD".to_string(),
            kv_scale_granularity: "tensor".to_string(),
        },
        kda_block: KdaBlockLocalWorkletConfig {
            hidden: model.hidden,
            num_heads: model.num_heads,
            head_dim: model.head_dim,
            short_conv_kernel_size: model.short_conv_kernel_size,
            dtype: model.dtype,
            fp8: model.fp8,
            gpu_name: gpu.clone(),
            norm_backends: NORM_BACKENDS.to_vec(),
            gemm_backends: model.single_gemm_backends(),
            act_backends: ACT_BACKENDS.to_vec(),
            kda_scan_backends: KDA_SCAN_BACKENDS.to_vec(),
        },
        moe_router: MoeRouterLocalWorkletConfig {
            hidden: model.hidden,
            num_experts: model.num_experts,
            dtype: model.dtype,
            compute_dtype: model.compute_dtype(),
            gpu_name: gpu.clone(),
            norm_backends: NORM_BACKENDS.to_vec(),
            gemm_backends: model.single_gemm_backends(),
        },
        moe_dispatch: moe_net.clone(),
        // Routed expert GEMMs at the REDUCED hidden: gate_up k=3584 n=2·3072,
        // down k=3072 n=3584.
        moe_expert_compute: MoeExpertComputeLocalWorkletConfig {
            hidden: model.expert_hidden,
            moe_intermediate: model.moe_intermediate,
            num_experts: model.num_experts,
            ep_size: parallel.ep_size,
            dtype: model.compute_dtype(),
            gpu_name: gpu.clone(),
            act_backends: ACT_BACKENDS.to_vec(),
            grouped_gemm_backends: model.grouped_gemm_backends(),
            local_ppm,
        },
        // Home-rank reduce over the FULL hidden residual (conservative v1, as
        // qwen3_moe).
        moe_local_reduce: ElementwiseKernelConfig {
            backends: ACT_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            input_bytes_per_token: model.hidden * dtype_bytes,
            output_bytes_per_token: model.hidden * dtype_bytes,
        },
        moe_combine: moe_net,
        shared_gate_up: SingleGemmKernelConfig {
            backends: model.single_gemm_backends(),
            gpu_name: gpu.clone(),
            n: 2 * shared_inter,
            k: model.hidden,
            dtype: model.compute_dtype(),
        },
        shared_act: ElementwiseKernelConfig {
            backends: ACT_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            input_bytes_per_token: 2 * shared_inter * dtype_bytes,
            output_bytes_per_token: shared_inter * dtype_bytes,
        },
        shared_down: SingleGemmKernelConfig {
            backends: model.single_gemm_backends(),
            gpu_name: gpu.clone(),
            n: model.hidden,
            k: shared_inter,
            dtype: model.compute_dtype(),
        },
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
            backends: model.single_gemm_backends(),
            gpu_name: gpu.clone(),
            n: model.vocab,
            k: model.hidden,
            dtype: model.compute_dtype(),
        },
        mla_layers: model.mla_layers,
        kda_layers: model.kda_layers(),
        attn_tp_size: parallel.attn_tp_size,
        ep_size: parallel.ep_size,
        num_dp_groups,
        hp_size: parallel.hp_size,
        nvl_num_gpu: parallel.nvl_num_gpu,
        top_k: model.top_k,
        num_experts: model.num_experts,
    }
}

/// **Total** KV-cache bytes one token occupies. Only the MLA layers cache
/// per-token state: ONE compressed `kv_compressed_dim`-wide vector per token
/// per layer (K and V share it — no factor 2, no per-head multiplicity). For
/// the real bf16 config: 24 layers × 576 × 2 B = **27_648 B**. KDA layers hold
/// a constant-size per-REQUEST recurrent state (not per-token), so they
/// contribute nothing here (deliberately excluded; see module header).
fn total_kv_bytes_per_token(resolved: &KimiK3KdaMlaResolved) -> u64 {
    let raw = &resolved.mla_block.raw_cfg;
    u64::from(resolved.mla_layers)
        * u64::from(raw.kv_compressed_dim)
        * u64::from(raw.kv_dtype().size_bytes())
}

pub fn resolve_configs(cfgs: &KimiK3KdaMlaConfigs) -> KimiK3KdaMlaResolved {
    KimiK3KdaMlaResolved {
        mla_block: MlaBlockLocalWorklet::resolve_config(&cfgs.mla_block),
        kda_block: KdaBlockLocalWorklet::resolve_config(&cfgs.kda_block),
        moe_router: MoeRouterLocalWorklet::resolve_config(&cfgs.moe_router),
        moe_dispatch: cfgs.moe_dispatch.clone(),
        moe_expert_compute: MoeExpertComputeLocalWorklet::resolve_config(&cfgs.moe_expert_compute),
        moe_local_reduce: cfgs.moe_local_reduce.clone(),
        moe_combine: cfgs.moe_combine.clone(),
        shared_gate_up: cfgs.shared_gate_up.clone(),
        shared_act: cfgs.shared_act.clone(),
        shared_down: cfgs.shared_down.clone(),
        embed: cfgs.embed.clone(),
        final_norm: cfgs.final_norm.clone(),
        lm_head: cfgs.lm_head.clone(),
        mla_layers: cfgs.mla_layers,
        kda_layers: cfgs.kda_layers,
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
    resolved: KimiK3KdaMlaResolved,
    bridge: &PerfApiBridge,
) -> Result<KimiK3KdaMlaModel, BuildError> {
    let total_kv_bytes_per_token = total_kv_bytes_per_token(&resolved);

    let embed_name = format!("{model_name}.embedding");
    let final_norm_name = format!("{model_name}.final_norm");
    let lm_head_name = format!("{model_name}.lm_head");
    let local_reduce_name = format!("{model_name}.moe_local_reduce");
    let sgu_name = format!("{model_name}.shared_experts.gate_up");
    let sact_name = format!("{model_name}.shared_experts.activation");
    let sdn_name = format!("{model_name}.shared_experts.down");

    let embed = Op::new(
        embed_name.clone(),
        Arc::new(ElementwiseKernel::build(
            embed_name,
            resolved.embed.clone(),
            bridge,
        )?),
    );
    let mla_block = MlaBlockLocalWorklet::build(
        format!("{model_name}.mla_block"),
        resolved.mla_block.clone(),
        bridge,
    )?;
    let kda_block = KdaBlockLocalWorklet::build(
        format!("{model_name}.kda_block"),
        resolved.kda_block.clone(),
        bridge,
    )?;
    let moe_router = MoeRouterLocalWorklet::build(
        format!("{model_name}.moe_router"),
        resolved.moe_router.clone(),
        bridge,
    )?;
    let moe_dispatch = MoeDispatchOp::build(
        format!("{model_name}.moe_dispatch"),
        resolved.moe_dispatch.clone(),
        bridge,
    )?;
    let moe_expert_compute = MoeExpertComputeLocalWorklet::build(
        format!("{model_name}.moe_expert_compute"),
        resolved.moe_expert_compute.clone(),
        bridge,
    )?;
    let moe_local_reduce = Op::new(
        local_reduce_name.clone(),
        Arc::new(ElementwiseKernel::build(
            local_reduce_name,
            resolved.moe_local_reduce.clone(),
            bridge,
        )?),
    );
    let moe_combine = MoeCombineOp::build(
        format!("{model_name}.moe_combine"),
        resolved.moe_combine.clone(),
        bridge,
    )?;
    let shared_gate_up = Op::new(
        sgu_name.clone(),
        Arc::new(SingleGemmKernel::build(
            sgu_name,
            resolved.shared_gate_up.clone(),
            bridge,
        )?),
    );
    let shared_act = Op::new(
        sact_name.clone(),
        Arc::new(ElementwiseKernel::build(
            sact_name,
            resolved.shared_act.clone(),
            bridge,
        )?),
    );
    let shared_down = Op::new(
        sdn_name.clone(),
        Arc::new(SingleGemmKernel::build(
            sdn_name,
            resolved.shared_down.clone(),
            bridge,
        )?),
    );
    let final_norm = Op::new(
        final_norm_name.clone(),
        Arc::new(RmsNormKernel::build(
            final_norm_name,
            resolved.final_norm.clone(),
            bridge,
        )?),
    );
    let lm_head = Op::new(
        lm_head_name.clone(),
        Arc::new(SingleGemmKernel::build(
            lm_head_name,
            resolved.lm_head.clone(),
            bridge,
        )?),
    );

    let mut model = KimiK3KdaMlaModel {
        name: model_name,
        mla_layers: resolved.mla_layers,
        kda_layers: resolved.kda_layers,
        attn_tp_size: resolved.attn_tp_size,
        ep_size: resolved.ep_size,
        num_dp_groups: resolved.num_dp_groups,
        hp_size: resolved.hp_size,
        nvl_num_gpu: resolved.nvl_num_gpu,
        top_k: resolved.top_k,
        num_experts: resolved.num_experts,
        total_kv_bytes_per_token,
        mla_block,
        kda_block,
        moe_router,
        moe_dispatch,
        moe_expert_compute,
        moe_local_reduce,
        moe_combine,
        shared_gate_up,
        shared_act,
        shared_down,
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

impl KimiK3KdaMlaModel {
    /// Compile the shared MoE FFN body (router → dispatch → experts → reduce →
    /// combine → shared experts) — called once per layer template, so each
    /// template owns fresh slots.
    fn compile_moe_body(&self, b: &mut CostTreeBuilder, parts: &mut Vec<CostNode>) {
        let router_groups: Vec<CostNode> = (0..self.num_dp_groups)
            .map(|_| self.moe_router.compile(b))
            .collect();
        parts.push(CostNode::Max {
            overlap: 1.0,
            children: router_groups,
        });
        parts.push(self.moe_dispatch.compile(b));
        let expert_groups: Vec<CostNode> = (0..self.ep_size)
            .map(|_| self.moe_expert_compute.compile(b))
            .collect();
        parts.push(CostNode::Max {
            overlap: 1.0,
            children: expert_groups,
        });
        let reduce_groups: Vec<CostNode> = (0..self.num_dp_groups)
            .map(|_| self.moe_local_reduce.compile(b))
            .collect();
        parts.push(CostNode::Max {
            overlap: 1.0,
            children: reduce_groups,
        });
        parts.push(self.moe_combine.compile(b));
        // Shared experts: dense, replicated per DP shard on the shard's own
        // tokens (Max fan-out like the router).
        let shared_groups: Vec<CostNode> = (0..self.num_dp_groups)
            .map(|_| {
                CostNode::Sum(vec![
                    self.shared_gate_up.compile(b),
                    self.shared_act.compile(b),
                    self.shared_down.compile(b),
                ])
            })
            .collect();
        parts.push(CostNode::Max {
            overlap: 1.0,
            children: shared_groups,
        });
    }

    /// Eval counterpart of [`Self::compile_moe_body`] — same slot order.
    fn eval_moe_body(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        let m_total: u32 = batch.groups.iter().map(|g| g.batch_tokens).sum();
        let global_expert_selections = m_total * self.top_k;
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
                tokens: u64::from(m_total),
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
                tokens: u64::from(m_total),
            },
            ev,
        );
        for g in &batch.groups {
            let m = g.batch_tokens;
            self.shared_gate_up.eval(&SingleGemmKernelInput { m }, ev);
            self.shared_act
                .eval(&ElementwiseKernelInput { num_tokens: m }, ev);
            self.shared_down.eval(&SingleGemmKernelInput { m }, ev);
        }
    }

    /// Compile the per-iteration cost STRUCTURE once: Sum(embed,
    /// Scale{mla_layers}(mla layer), Scale{kda_layers}(kda layer), final_norm,
    /// lm_head). Each layer template = Max-fanned attention block + the shared
    /// MoE body (freshly minted per template).
    pub fn cost_tree(&self) -> CostTree {
        let mut b = CostTreeBuilder::new();
        let embed = self.embed.compile(&mut b);

        // ── MLA layer template ──
        let mla_groups: Vec<CostNode> = (0..self.num_dp_groups)
            .map(|_| self.mla_block.compile(&mut b))
            .collect();
        let mut mla_parts = vec![CostNode::Max {
            overlap: 1.0,
            children: mla_groups,
        }];
        self.compile_moe_body(&mut b, &mut mla_parts);
        let mla_layer = CostNode::Labeled {
            label: "mla_layer".to_string(),
            child: Box::new(CostNode::Scale {
                n: self.mla_layers,
                child: Box::new(CostNode::Sum(mla_parts)),
            }),
        };

        // ── KDA layer template ──
        let kda_groups: Vec<CostNode> = (0..self.num_dp_groups)
            .map(|_| self.kda_block.compile(&mut b))
            .collect();
        let mut kda_parts = vec![CostNode::Max {
            overlap: 1.0,
            children: kda_groups,
        }];
        self.compile_moe_body(&mut b, &mut kda_parts);
        let kda_layer = CostNode::Labeled {
            label: "kda_layer".to_string(),
            child: Box::new(CostNode::Scale {
                n: self.kda_layers,
                child: Box::new(CostNode::Sum(kda_parts)),
            }),
        };

        let final_norm = self.final_norm.compile(&mut b);
        let lm_head = self.lm_head.compile(&mut b);
        let root = CostNode::Labeled {
            label: format!(
                "{} [KDA+MLA hybrid: {} MLA + {} KDA layers; DP attn (attn_tp=1, dp={}) × \
                 EP FFN (ep={}, hp={}, nvl={}), num_experts={} top_k={} (+{} shared)]",
                self.name,
                self.mla_layers,
                self.kda_layers,
                self.num_dp_groups,
                self.ep_size,
                self.hp_size,
                self.nvl_num_gpu,
                self.num_experts,
                self.top_k,
                2,
            ),
            child: Box::new(CostNode::Sum(vec![
                embed, mla_layer, kda_layer, final_norm, lm_head,
            ])),
        };
        b.finish(root)
    }

    /// CostTree eval: stream leaves in the EXACT `cost_tree` mint order —
    /// embed, ONE MLA layer body (dp × mla_block, then the MoE body), ONE KDA
    /// layer body (dp × kda_block, then the MoE body), final_norm, lm_head.
    /// The two `Scale` folds multiply the layer bodies by 24 / 69.
    fn eval_into(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        let m_total: u32 = batch.groups.iter().map(|g| g.batch_tokens).sum();
        let request_count: u32 = batch.groups.iter().map(|g| g.request_count()).sum();

        self.embed.eval(
            &ElementwiseKernelInput {
                num_tokens: m_total,
            },
            ev,
        );

        // MLA layer body.
        for g in &batch.groups {
            self.mla_block.eval(
                &MlaBlockLocalWorkletInput {
                    batch_tokens: g.batch_tokens,
                    prefill_chunk_pairs: g.prefill_chunk_pairs.clone(),
                    decode_kv_lens: g.decode_kv_lens.clone(),
                },
                ev,
            );
        }
        self.eval_moe_body(batch, ev);

        // KDA layer body.
        for g in &batch.groups {
            self.kda_block.eval(
                &KdaBlockLocalWorkletInput {
                    batch_tokens: g.batch_tokens,
                },
                ev,
            );
        }
        self.eval_moe_body(batch, ev);

        self.final_norm.eval(&RmsNormKernelInput { m: m_total }, ev);
        self.lm_head
            .eval(&SingleGemmKernelInput { m: request_count }, ev);
    }
}

impl IterwiseUnifiedModel for KimiK3KdaMlaModel {
    fn total_kv_bytes_per_token(&self) -> u64 {
        self.total_kv_bytes_per_token
    }

    /// One replica spans the EP group (= the DP-attention shards, attn_tp=1).
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
            "Kimi-K3 KDA+MLA arch expects one group per DP shard (num_dp_groups)"
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
            "Kimi-K3 KDA+MLA arch expects one group per DP shard (num_dp_groups)"
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

    fn parallel(ep: u16) -> KimiK3Parallel {
        KimiK3Parallel {
            attn_tp_size: 1,
            ep_size: ep,
            hp_size: 1,
            nvl_num_gpu: 8,
            gpu_name: "NVIDIA H200".to_string(),
        }
    }

    fn routing(model: &KimiModelCfg) -> RoutingDistribution {
        RoutingDistribution::uniform(model.num_experts)
    }

    #[test]
    fn build_configs_threads_the_real_kimi_k3_dims() {
        let model = KimiModelCfg::kimi_k3();
        let cfgs = build_configs(&model, &parallel(8), &routing(&model));
        // Layer split 24 MLA + 69 KDA; DP groups = ep (attn_tp = 1).
        assert_eq!(cfgs.mla_layers, 24);
        assert_eq!(cfgs.kda_layers, 69);
        assert_eq!(cfgs.num_dp_groups, 8);
        // MLA dims.
        assert_eq!(cfgs.mla_block.q_lora_rank, 1536);
        assert_eq!(cfgs.mla_block.kv_compressed_dim, 576);
        assert_eq!(cfgs.mla_block.qk_head_dim, 192);
        // KDA dims.
        assert_eq!(cfgs.kda_block.num_heads, 96);
        assert_eq!(cfgs.kda_block.short_conv_kernel_size, 4);
        // Router on the FULL hidden; experts at the REDUCED hidden.
        assert_eq!(cfgs.moe_router.hidden, 7168);
        assert_eq!(cfgs.moe_router.num_experts, 896);
        assert_eq!(cfgs.moe_expert_compute.hidden, 3584);
        assert_eq!(cfgs.moe_expert_compute.moe_intermediate, 3072);
        // Dispatch/combine ship the reduced activation; top-16 routing.
        assert_eq!(cfgs.moe_dispatch.h, 3584);
        assert_eq!(cfgs.moe_dispatch.top_k, 16);
        // Shared experts fused: gate_up n = 2·(2·3072), down k = 2·3072.
        assert_eq!(cfgs.shared_gate_up.n, 2 * 2 * 3072);
        assert_eq!(cfgs.shared_gate_up.k, 7168);
        assert_eq!(cfgs.shared_down.k, 2 * 3072);
        // lm_head replicated full.
        assert_eq!(cfgs.lm_head.n, 163840);
        assert_eq!(cfgs.lm_head.k, 7168);
    }

    #[test]
    fn resolve_bakes_expert_gemms_at_reduced_hidden() {
        let model = KimiModelCfg::kimi_k3();
        let r = resolve_configs(&build_configs(&model, &parallel(8), &routing(&model)));
        // 896 experts / ep=8 → 112 experts per GPU.
        assert_eq!(r.moe_expert_compute.experts_per_gpu, 112);
        assert_eq!(r.moe_expert_compute.gate_up.n, 2 * 3072);
        assert_eq!(r.moe_expert_compute.gate_up.k, 3584);
        assert_eq!(r.moe_expert_compute.down.n, 3584);
        assert_eq!(r.moe_expert_compute.down.k, 3072);
        // MLA attention shapes.
        assert_eq!(r.mla_block.attn.num_heads, 96);
        assert_eq!(r.mla_block.attn.head_dim, 128);
        assert_eq!(r.mla_block.attn.kv_compressed_dim, 576);
    }

    #[test]
    fn total_kv_bytes_per_token_is_mla_compressed_only() {
        let model = KimiModelCfg::kimi_k3();
        let r = resolve_configs(&build_configs(&model, &parallel(8), &routing(&model)));
        // 24 MLA layers × 576 compressed dim × 2 B (bf16), K/V shared → 27648.
        assert_eq!(total_kv_bytes_per_token(&r), 27_648);
        assert_eq!(total_kv_bytes_per_token(&r), 24 * 576 * 2);
    }

    #[test]
    #[should_panic(expected = "attn_tp_size must be 1")]
    fn attn_tp_greater_than_one_panics() {
        let model = KimiModelCfg::kimi_k3();
        let p = KimiK3Parallel {
            attn_tp_size: 2,
            ep_size: 8,
            hp_size: 1,
            nvl_num_gpu: 8,
            gpu_name: "NVIDIA H200".to_string(),
        };
        let _ = build_configs(&model, &p, &routing(&model));
    }

    #[test]
    #[should_panic(expected = "num_experts")]
    fn ep_not_dividing_num_experts_panics() {
        // 896 % 5 != 0.
        let model = KimiModelCfg::kimi_k3();
        let _ = build_configs(&model, &parallel(5), &routing(&model));
    }

    #[test]
    #[should_panic(expected = "hp_size")]
    fn hp_not_dividing_ep_panics() {
        let model = KimiModelCfg::kimi_k3();
        let p = KimiK3Parallel {
            attn_tp_size: 1,
            ep_size: 8,
            hp_size: 3,
            nvl_num_gpu: 8,
            gpu_name: "NVIDIA H200".to_string(),
        };
        let _ = build_configs(&model, &p, &routing(&model));
    }
}
