//! `qwen3_vllm_moe_dp_attn_ep_ffn` — L4 model_arch for Qwen3-MoE-style decoders with
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
//! `× ep_size` MAX on `expert_compute` is the EP fan-out: each EP rank owns one
//! `local_ppm` shard, so a measured or synthetic skew yields a distinct
//! grouped-GEMM kernel cache (L1 grouped GEMM is distribution-sensitive). The
//! uniform baseline still produces identical children, but the rank fan-out is
//! kept explicit so the same CostTree shape handles a profile without a second
//! arch implementation.
//!
//! v1 deviations (carried over from `llama3_dp_attn_tp_ffn`, plus MoE-specific):
//!   - embed / final_norm / lm_head are replicated full shapes on the pooled
//!     token total (no vocab-parallel split);
//!   - one routing snapshot is used for all homogeneous MoE layers. A profile
//!     JSON is therefore reduced to its all-layer expert totals at the L4 build
//!     seam; layer-by-layer CostTree materialisation is intentionally deferred.

use std::sync::Arc;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::moe_model_cfg::MoeModelCfg;
use crate::common::Fabric;
use crate::op::gemm::{
    SingleFp8GemmWithQuantConfig, SingleFp8GemmWithQuantInput, SingleFp8GemmWithQuantOp,
};
use crate::op::moe::{MoeDispatchOp, MoeNetConfig, MoeNetInput, Placement};
use crate::op::Op;
use crate::timing::kernels::{
    AllReduceKernel, AllReduceKernelConfig, AllReduceKernelInput, AllReduceResidualRmsNormKernel,
    AllReduceResidualRmsNormKernelConfig, AllReduceResidualRmsNormKernelInput,
    AllReduceResidualRmsNormSpec, ElementwiseKernel, ElementwiseKernelConfig,
    ElementwiseKernelInput, Fp8PerTokenGroupQuantKernelConfig, MoeFinalizeRoutingKernel,
    MoeFinalizeRoutingKernelConfig, MoeFinalizeRoutingKernelInput, RmsNormKernel,
    RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernelConfig,
};
use crate::timing::routing::RoutingDistribution;
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, DType, Dim, Evaluator,
    FlatCostNode, LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::{
    MoeExpertComputeLocalWorklet, MoeExpertComputeLocalWorkletConfig,
    MoeExpertComputeLocalWorkletInput, MoeExpertComputeLocalWorkletResolved,
    VllmFp8AttnBlockTpWorklet, VllmFp8AttnBlockTpWorkletConfig, VllmFp8AttnBlockTpWorkletInput,
    VllmFp8AttnBlockTpWorkletResolved, VllmFp8MoeRouterLocalWorklet,
    VllmFp8MoeRouterLocalWorkletConfig, VllmFp8MoeRouterLocalWorkletInput,
    VllmFp8MoeRouterLocalWorkletResolved,
};

const NORM_BACKENDS: &[&str] = &["flashinfer"];
// GEMM backends are dtype-driven by `MoeModelCfg`: fp8 uses DeepGEMM; bf16
// dense GEMMs compare both Torch weight layouts while grouped experts stay on
// their one registered Torch implementation.
const ACT_BACKENDS: &[&str] = &["triton"];
const ROUTED_FP8_QUANT_BACKENDS: &[&str] = &["flashinfer_trtllm"];
const ROUTED_FP8_GROUPED_GEMM_BACKENDS: &[&str] = &["flashinfer_trtllm"];
const DENSE_FP8_QUANT_BACKENDS: &[&str] = &["vllm_cuda"];
// See llama3_dense: FlashInfer impls registered under fa2/fa3, not "flashinfer".
const ATTN_BACKENDS: &[&str] = &["fa2", "fa3"];
// This alignment target pins every network leaf to NVSHMEM. Keep the TP
// all-reduce and MoE p2p policies together so a future backend change cannot
// silently produce a mixed-network model.
const ALLREDUCE_BACKENDS: &[&str] = &["nvshmem"];
const FUSED_ALLREDUCE_BACKENDS: &[&str] = &["flashinfer_trtllm"];
const P2P_BACKENDS: &[&str] = &["nvshmem"];
const VLLM_MOE_ALLREDUCE_BACKENDS: &[&str] = &["nccl"];
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
pub struct Qwen3VllmMoeParallel {
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
pub struct Qwen3VllmMoeDpAttnEpFfnConfigs {
    pub attn_block: VllmFp8AttnBlockTpWorkletConfig,
    pub moe_router: VllmFp8MoeRouterLocalWorkletConfig,
    pub moe_dispatch: MoeNetConfig,
    /// One distribution-sensitive expert worklet per EP rank. Keeping the
    /// vector at L4 makes the rank `Max` explicit and preserves each rank's
    /// raw `local_ppm` cache identity.
    pub moe_expert_compute: Vec<MoeExpertComputeLocalWorkletConfig>,
    pub moe_finalize: Vec<MoeFinalizeRoutingKernelConfig>,
    pub moe_combine_allreduce: AllReduceKernelConfig,
    pub moe_combine_allreduce_fused: AllReduceResidualRmsNormKernelConfig,
    pub embed: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleFp8GemmWithQuantConfig,
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
pub struct Qwen3VllmMoeDpAttnEpFfnResolved {
    pub attn_block: VllmFp8AttnBlockTpWorkletResolved,
    pub moe_router: VllmFp8MoeRouterLocalWorkletResolved,
    pub moe_dispatch: MoeNetConfig,
    pub moe_expert_compute: Vec<MoeExpertComputeLocalWorkletResolved>,
    pub moe_finalize: Vec<MoeFinalizeRoutingKernelConfig>,
    pub moe_combine_allreduce: AllReduceKernelConfig,
    pub moe_combine_allreduce_fused: AllReduceResidualRmsNormKernelConfig,
    pub moe_combine_max_fused_tokens: u32,
    pub embed: ElementwiseKernelConfig,
    pub final_norm: RmsNormKernelConfig,
    pub lm_head: SingleFp8GemmWithQuantConfig,
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

pub struct Qwen3VllmMoeDpAttnEpFfnModel {
    pub name: String,
    pub num_layers: u32,
    pub attn_tp_size: u16,
    pub ep_size: u16,
    pub num_dp_groups: u16,
    pub hp_size: u16,
    pub nvl_num_gpu: u16,
    pub top_k: u32,
    pub num_experts: u32,
    /// Symbolic KV footprint per token (folds to bytes at the worker seam).
    pub total_kv_bytes_per_token: Dim,
    pub attn_block: VllmFp8AttnBlockTpWorklet,
    pub moe_router: VllmFp8MoeRouterLocalWorklet,
    pub moe_dispatch: MoeDispatchOp,
    pub moe_expert_compute: Vec<MoeExpertComputeLocalWorklet>,
    pub moe_finalize: Vec<Op<MoeFinalizeRoutingKernel>>,
    pub moe_combine_allreduce: Op<AllReduceKernel>,
    pub moe_combine_allreduce_fused: Op<AllReduceResidualRmsNormKernel>,
    pub moe_combine_max_fused_tokens: u32,
    pub moe_combine_message_bytes_per_token: u64,
    pub embed: Op<ElementwiseKernel>,
    pub final_norm: Op<RmsNormKernel>,
    pub lm_head: SingleFp8GemmWithQuantOp,
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
) -> VllmFp8AttnBlockTpWorkletConfig {
    VllmFp8AttnBlockTpWorkletConfig {
        hidden: model.hidden.clone(),
        num_qo_heads: model.num_qo_heads.clone(),
        num_kv_heads: model.num_kv_heads.clone(),
        head_dim: model.head_dim.clone(),
        dtype: model.dtype,
        tp_size: attn_tp_size,
        tp_name: "attn_tp",
        allreduce_fabric: TP_FABRIC,
        allreduce_dtype: DType::Bf16,
        gpu_name: gpu_name.to_string(),
        norm_backends: NORM_BACKENDS.to_vec(),
        gemm_backends: model.single_gemm_backends(),
        fp8_quant_backends: DENSE_FP8_QUANT_BACKENDS.to_vec(),
        attn_backends: ATTN_BACKENDS.to_vec(),
        kv_cache_append_backends: vec!["vllm_cuda"],
        kv_cache_block_size: 16,
        kv_cache_layout: "NHD".to_string(),
        kv_scale_granularity: "tensor".to_string(),
        allreduce_backends: ALLREDUCE_BACKENDS.to_vec(),
        fused_allreduce_backends: FUSED_ALLREDUCE_BACKENDS.to_vec(),
    }
}

pub fn build_configs(
    model: &MoeModelCfg,
    parallel: &Qwen3VllmMoeParallel,
    routing: &RoutingDistribution,
) -> Qwen3VllmMoeDpAttnEpFfnConfigs {
    assert!(model.fp8, "vLLM-aligned Qwen L4 requires fp8=true");
    let gpu = &parallel.gpu_name;
    // Byte-transfer widths (local reduce, embed) use the compute dtype: fp8 halves
    // the wire size, bf16 leaves it unchanged. The RMSNorm ops keep `model.dtype`.
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
        model.num_experts.get() % u32::from(parallel.ep_size) == 0,
        "num_experts {} not divisible by ep_size {}",
        model.num_experts,
        parallel.ep_size,
    );
    assert_eq!(
        routing.num_experts(),
        model.num_experts.get(),
        "routing distribution has {} experts, model has {}",
        routing.num_experts(),
        model.num_experts,
    );
    // `routing` drives both the L2 MoE dispatch/combine `BottleneckCurve` and
    // the L3 grouped-GEMM cache identities. The L3 helper owns the full-ppm →
    // contiguous rank-shard partition; L4 only places those rank worklets in
    // the EP `Max` fan-out below.
    let moe_net = MoeNetConfig {
        backends: P2P_BACKENDS.to_vec(),
        gpu_name: gpu.clone(),
        // Dispatch/combine move activation/output tensors, not packed expert
        // weights. Pin the wire payload to BF16 even when expert GEMMs use FP8.
        dtype: DType::Bf16,
        intra_fabric: MOE_INTRA_FABRIC,
        inter_fabric: MOE_INTER_FABRIC,
        ep_size: u32::from(parallel.ep_size),
        nvl_num_gpu: u32::from(parallel.nvl_num_gpu),
        // Comm-sizing seam: the MoE all-to-all payload is byte-keyed.
        h: model.hidden.get(),
        top_k: model.top_k,
        routing: routing.clone(),
        placement: Placement::ReplicatedHeadParallel {
            hp_size: u32::from(parallel.hp_size),
        },
    };
    let lm_head_gemm = SingleGemmKernelConfig {
        backends: model.single_gemm_backends(),
        gpu_name: gpu.clone(),
        n: model.vocab.clone(),
        k: model.hidden.clone(),
        dtype: model.compute_dtype(),
    };
    assert_eq!(
        parallel.attn_tp_size, parallel.ep_size,
        "vLLM no-DP EP alignment requires one attention group spanning the EP group"
    );
    assert_eq!(
        parallel.hp_size, 1,
        "vLLM no-DP EP alignment requires hp_size=1"
    );
    // This EP path receives expert-partitioned rows between dispatch and
    // finalize, so keep the production FlashInfer/TRT-LLM block-quant +
    // blockscale-grouped-GEMM realization. The vLLM sibling worklet models the
    // distinct EP1 local path measured for Qwen3.6; it must not silently change
    // this arch's routed gate/up or down GEMMs.
    let moe_expert_compute = MoeExpertComputeLocalWorkletConfig::split_for_ep(
        MoeExpertComputeLocalWorkletConfig {
            hidden: model.hidden.clone(),
            moe_intermediate: model.moe_intermediate.clone(),
            num_experts: model.num_experts.clone(),
            ep_size: parallel.ep_size,
            top_k: model.top_k,
            dtype: DType::Fp8E4m3,
            activation_dtype: model.dtype,
            gpu_name: gpu.clone(),
            act_backends: ACT_BACKENDS.to_vec(),
            fp8_quant_backends: ROUTED_FP8_QUANT_BACKENDS.to_vec(),
            grouped_gemm_backends: model.grouped_gemm_backends(),
            fp8_grouped_gemm_backends: ROUTED_FP8_GROUPED_GEMM_BACKENDS.to_vec(),
            use_fp8_blockscale_grouped_gemm: true,
            local_ppm: Vec::new(),
        },
        routing.ppm(),
    );
    let experts_per_rank = model.num_experts.get() / u32::from(parallel.ep_size);
    let moe_finalize = {
        moe_expert_compute
            .iter()
            .map(|expert_config| MoeFinalizeRoutingKernelConfig {
                backends: ROUTED_FP8_GROUPED_GEMM_BACKENDS.to_vec(),
                gpu_name: gpu.clone(),
                hidden_size: model.hidden.clone(),
                top_k: model.top_k,
                num_experts_per_rank: experts_per_rank,
                local_ppm: expert_config.local_ppm.clone(),
                dtype: DType::Bf16,
            })
            .collect()
    };
    let vllm_fused_combine = AllReduceResidualRmsNormKernelConfig {
        backends: FUSED_ALLREDUCE_BACKENDS.to_vec(),
        gpu_name: gpu.clone(),
        num_gpus: u32::from(parallel.ep_size),
        hidden_dim: model.hidden.get(),
        dtype: DType::Bf16,
        fabric: MOE_INTRA_FABRIC,
        strategy: "auto".to_string(),
        launch_with_pdl: true,
        fp32_acc: true,
    };
    Qwen3VllmMoeDpAttnEpFfnConfigs {
        attn_block: attn_block_config(model, parallel.attn_tp_size, gpu),
        moe_router: VllmFp8MoeRouterLocalWorkletConfig {
            hidden: model.hidden.clone(),
            num_experts: model.num_experts.clone(),
            activation_dtype: model.dtype,
            gpu_name: gpu.clone(),
            gemm_backends: model.single_gemm_backends(),
            fp8_quant_backends: DENSE_FP8_QUANT_BACKENDS.to_vec(),
        },
        moe_dispatch: moe_net.clone(),
        // FP8 runs quantize each BF16 activation immediately before the two
        // grouped GEMMs. Their outputs and the SwiGLU intermediate remain in
        // the model activation dtype.
        moe_expert_compute,
        // Conservative v1 model: per home-rank token, read+write one full hidden
        // vector of partials. Refinement to top_k-weighted partial reads is
        // deferred until we model expert-redundancy at home-rank reduction.
        moe_finalize,
        moe_combine_allreduce: AllReduceKernelConfig {
            backends: VLLM_MOE_ALLREDUCE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            num_gpus: u32::from(parallel.ep_size),
            fabric: MOE_INTRA_FABRIC,
        },
        moe_combine_allreduce_fused: vllm_fused_combine,
        // Embedding gather placeholder: read one hidden-wide row, write one out
        // (replicated; hidden NOT sharded).
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
        // Replicated full lm_head for v1 (vocab-parallel split deferred). FP8
        // consumes the same L2 quant+GEMM compound op as other dense linears.
        lm_head: SingleFp8GemmWithQuantConfig {
            quant: Fp8PerTokenGroupQuantKernelConfig {
                backends: DENSE_FP8_QUANT_BACKENDS.to_vec(),
                gpu_name: gpu.clone(),
                hidden_size: model.hidden.clone(),
                group_size: 128,
                input_dtype: model.dtype,
                scale_format: "ue8m0_column_major".to_string(),
            },
            gemm: lm_head_gemm,
        },
        num_layers: model.num_layers,
        attn_tp_size: parallel.attn_tp_size,
        ep_size: parallel.ep_size,
        num_dp_groups,
        hp_size: parallel.hp_size,
        nvl_num_gpu: parallel.nvl_num_gpu,
        top_k: model.top_k,
        // Arch-level scalar kept for the model label only (a count, not a shape).
        num_experts: model.num_experts.get(),
    }
}

/// **Total** KV-cache bytes one token occupies — summed across all
/// `attn_tp_size` ranks of one DP shard, all layers, all KV heads. Same
/// definition as `llama3_dp_attn_tp_ffn`: the wire size of a token's KV for a
/// PD handoff (each DP shard owns one full copy of every KV head).
fn total_kv_bytes_per_token(resolved: &Qwen3VllmMoeDpAttnEpFfnResolved) -> Dim {
    let raw = &resolved.attn_block.raw_cfg;
    2 * raw.num_kv_heads.clone()
        * raw.head_dim.clone()
        * Dim::param("bytes", raw.kv_dtype().size_bytes())
        * Dim::param("num_layers", resolved.num_layers)
}

pub fn resolve_configs(cfgs: &Qwen3VllmMoeDpAttnEpFfnConfigs) -> Qwen3VllmMoeDpAttnEpFfnResolved {
    Qwen3VllmMoeDpAttnEpFfnResolved {
        attn_block: VllmFp8AttnBlockTpWorklet::resolve_config(&cfgs.attn_block),
        moe_router: VllmFp8MoeRouterLocalWorklet::resolve_config(&cfgs.moe_router),
        moe_dispatch: cfgs.moe_dispatch.clone(),
        moe_expert_compute: cfgs
            .moe_expert_compute
            .iter()
            .map(MoeExpertComputeLocalWorklet::resolve_config)
            .collect(),
        moe_finalize: cfgs.moe_finalize.clone(),
        moe_combine_allreduce: cfgs.moe_combine_allreduce.clone(),
        moe_combine_max_fused_tokens: AllReduceResidualRmsNormSpec::max_fused_tokens(
            &cfgs.moe_combine_allreduce_fused,
        ),
        moe_combine_allreduce_fused: cfgs.moe_combine_allreduce_fused.clone(),
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
    resolved: Qwen3VllmMoeDpAttnEpFfnResolved,
    bridge: &PerfApiBridge,
) -> Result<Qwen3VllmMoeDpAttnEpFfnModel, BuildError> {
    let num_layers = resolved.num_layers;
    let attn_tp_size = resolved.attn_tp_size;
    let ep_size = resolved.ep_size;
    let num_dp_groups = resolved.num_dp_groups;
    let hp_size = resolved.hp_size;
    let nvl_num_gpu = resolved.nvl_num_gpu;
    let top_k = resolved.top_k;
    let num_experts = resolved.num_experts;
    let total_kv_bytes_per_token = total_kv_bytes_per_token(&resolved);
    let moe_combine_message_bytes_per_token =
        u64::from(resolved.final_norm.hidden.get()) * u64::from(DType::Bf16.size_bytes());
    let moe_combine_max_fused_tokens = resolved.moe_combine_max_fused_tokens;

    let embed_name = format!("{model_name}.embedding");
    let final_norm_name = format!("{model_name}.final_norm");
    let lm_head_name = format!("{model_name}.lm_head");

    let embed = Op::new(
        embed_name.clone(),
        Arc::new(ElementwiseKernel::build(
            embed_name,
            resolved.embed,
            bridge,
        )?),
    );

    let attn_block = VllmFp8AttnBlockTpWorklet::build(
        format!("{model_name}.attn_block"),
        resolved.attn_block,
        bridge,
    )?;

    let moe_router = VllmFp8MoeRouterLocalWorklet::build(
        format!("{model_name}.moe_router"),
        resolved.moe_router,
        bridge,
    )?;

    let moe_dispatch = MoeDispatchOp::build(
        format!("{model_name}.moe_dispatch"),
        resolved.moe_dispatch,
        bridge,
    )?;

    let moe_expert_compute = resolved
        .moe_expert_compute
        .into_iter()
        .map(|expert_compute| {
            // Keep the same dotted slot names for every rank. The repeated
            // slots are the EP fan-out positions in the CostTree and existing
            // analyzer label mappings intentionally identify them by operation
            // name rather than synthetic rank suffixes.
            MoeExpertComputeLocalWorklet::build(
                format!("{model_name}.moe_expert_compute"),
                expert_compute,
                bridge,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    let moe_finalize = resolved
        .moe_finalize
        .into_iter()
        .map(|config| {
            let name = format!("{model_name}.moe_finalize");
            MoeFinalizeRoutingKernel::build(name.clone(), config, bridge)
                .map(|kernel| Op::new(name, Arc::new(kernel)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let allreduce_name = format!("{model_name}.moe_combine_allreduce");
    let moe_combine_allreduce = Op::new(
        allreduce_name.clone(),
        Arc::new(AllReduceKernel::build(
            allreduce_name,
            resolved.moe_combine_allreduce,
            bridge,
        )?),
    );
    let fused_name = format!("{model_name}.moe_combine_allreduce_residual_norm");
    let moe_combine_allreduce_fused = Op::new(
        fused_name.clone(),
        Arc::new(AllReduceResidualRmsNormKernel::build(
            fused_name,
            resolved.moe_combine_allreduce_fused,
            bridge,
        )?),
    );

    let final_norm = Op::new(
        final_norm_name.clone(),
        Arc::new(RmsNormKernel::build(
            final_norm_name,
            resolved.final_norm,
            bridge,
        )?),
    );

    let lm_head = SingleFp8GemmWithQuantOp::build(lm_head_name, resolved.lm_head, bridge)?;

    let mut model = Qwen3VllmMoeDpAttnEpFfnModel {
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
        moe_finalize,
        moe_combine_allreduce,
        moe_combine_allreduce_fused,
        moe_combine_max_fused_tokens,
        moe_combine_message_bytes_per_token,
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

impl Qwen3VllmMoeDpAttnEpFfnModel {
    /// Compile the per-iteration cost STRUCTURE once: Sum(embed,
    /// Scale{num_layers}(Sum(attn_max, moe_router, dispatch, expert_max,
    /// local_reduce, combine)), final_norm, lm_head). The DP `Max` children
    /// reuse one worklet because their shapes are identical; the EP `Max`
    /// children use the rank-specific worklets because their `local_ppm` cache
    /// identities may differ. Every compile call mints fresh slots, so
    /// `eval_into` fills them in the same order.
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

        // EP fan-out — one expert_compute subtree per EP rank. A uniform
        // snapshot makes this Max numerically degenerate; popularity gives each
        // child a distinct grouped-GEMM cache/config and exposes imbalance.
        assert_eq!(
            self.moe_expert_compute.len(),
            usize::from(self.ep_size),
            "expert worklet count must equal ep_size"
        );
        let expert_groups: Vec<CostNode> = self
            .moe_expert_compute
            .iter()
            .map(|expert_compute| expert_compute.compile(&mut b))
            .collect();
        let expert_fanout = CostNode::Max {
            overlap: 1.0,
            children: expert_groups,
        };

        assert_eq!(self.moe_finalize.len(), usize::from(self.ep_size));
        let combine_nodes = vec![
            CostNode::Max {
                overlap: 1.0,
                children: self
                    .moe_finalize
                    .iter()
                    .map(|finalize| finalize.compile(&mut b))
                    .collect(),
            },
            self.moe_combine_allreduce.compile(&mut b),
            self.moe_combine_allreduce_fused.compile(&mut b),
        ];

        let layer = CostNode::Labeled {
            label: "layer".to_string(),
            child: Box::new(CostNode::Scale {
                n: self.num_layers,
                child: Box::new(CostNode::Sum(vec![
                    attn_fanout,
                    router_fanout,
                    moe_dispatch,
                    expert_fanout,
                    CostNode::Sum(combine_nodes),
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
    /// count), then `ep_size` moe_expert_compute evals (one per EP rank, each
    /// with its own local routing shard), then `num_dp_groups` moe_local_reduce
    /// evals (home
    /// reduce, per DP shard), then moe_combine (4 leaves on global token count).
    /// The `Scale{num_layers}` fold multiplies one layer body. Finally,
    /// final_norm runs on pooled tokens and lm_head on pooled requests.
    fn eval_into(&self, batch: &UnifiedArchInput, ev: &mut Evaluator) {
        let m_total: u32 = batch.groups.iter().map(|g| g.batch_tokens).sum();
        let request_count: u32 = batch.groups.iter().map(|g| g.request_count()).sum();
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
                &VllmFp8AttnBlockTpWorkletInput {
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
                &VllmFp8MoeRouterLocalWorkletInput {
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

        // EP fan-out: the global selection count is shared, while each worklet
        // derives its local routed count from its own `local_ppm` shard.
        for expert_compute in &self.moe_expert_compute {
            expert_compute.eval(
                &MoeExpertComputeLocalWorkletInput {
                    global_expert_selections,
                },
                ev,
            );
        }

        for finalize in &self.moe_finalize {
            finalize.eval(
                &MoeFinalizeRoutingKernelInput {
                    token_count: m_total,
                },
                ev,
            );
        }

        let use_fused_combine = m_total > 0 && m_total <= self.moe_combine_max_fused_tokens;
        let allreduce_input = AllReduceKernelInput {
            message_size_bytes: u64::from(m_total) * self.moe_combine_message_bytes_per_token,
        };
        if use_fused_combine || m_total == 0 {
            ev.push(LeafMetrics::ZERO, || allreduce_input.clone().into());
        } else {
            self.moe_combine_allreduce.eval(&allreduce_input, ev);
        }
        let fused_input = AllReduceResidualRmsNormKernelInput {
            num_tokens: m_total,
        };
        if use_fused_combine {
            self.moe_combine_allreduce_fused.eval(&fused_input, ev);
        } else {
            ev.push(LeafMetrics::ZERO, || fused_input.into());
        }

        self.final_norm.eval(&RmsNormKernelInput { m: m_total }, ev);
        self.lm_head.eval(
            &SingleFp8GemmWithQuantInput {
                num_tokens: request_count,
            },
            ev,
        );
    }
}

impl IterwiseUnifiedModel for Qwen3VllmMoeDpAttnEpFfnModel {
    fn total_kv_bytes_per_token(&self) -> u64 {
        self.total_kv_bytes_per_token.get() as u64
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
    use crate::op::moe::{GroupedGemmConfig, GroupedQuantConfig};

    #[test]
    fn vllm_fp8_graph_has_finalize_and_ep_allreduce_only() {
        let model = MoeModelCfg::qwen3_235b().with_fp8(true);
        let parallel = Qwen3VllmMoeParallel {
            attn_tp_size: 4,
            ep_size: 4,
            hp_size: 1,
            nvl_num_gpu: 4,
            gpu_name: "NVIDIA H200".into(),
        };
        let cfg = build_configs(&model, &parallel, &RoutingDistribution::uniform(128));
        assert_eq!(cfg.moe_finalize.len(), 4);
        assert_eq!(cfg.moe_combine_allreduce.num_gpus, 4);
        assert_eq!(cfg.lm_head.gemm.dtype, DType::Fp8E4m3);
        assert!(cfg.moe_expert_compute[0].use_fp8_blockscale_grouped_gemm);
        assert_eq!(
            cfg.moe_expert_compute[0].fp8_grouped_gemm_backends,
            ROUTED_FP8_GROUPED_GEMM_BACKENDS
        );

        let resolved = MoeExpertComputeLocalWorklet::resolve_config(&cfg.moe_expert_compute[0]);
        let gate_up = resolved
            .gate_up_fp8
            .expect("vLLM EP expert compute must use the FP8 compound op");
        assert!(matches!(gate_up.quant, GroupedQuantConfig::Block(_)));
        assert!(matches!(
            gate_up.gemm,
            GroupedGemmConfig::TrtllmBlockscale(_)
        ));
    }
}
