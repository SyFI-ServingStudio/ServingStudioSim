//! Qwen4Exp hybrid decoders (`Qwen4ExpForConditionalGeneration`, text only)
//! with ModelOpt mixed-precision weights on one GPU under the vLLM fork,
//! drafting with the checkpoint's native MTP layer. Measured against the
//! diverse_100 capture of `nvidia/Qwen3.8-Flash-Next-NVFP4` at TP1 on B200
//! (Model Runner V2, breakable CUDA graphs, FULL_AND_PIECEWISE, MTP k = 3).
//!
//! ```text
//! prologue     embed, hidden.repeat (hc_count streams)
//! per layer    [PLE] attn HC mix, GDN or QSA, mlp HC mix, MoE block
//!                                                   (Scale per layer group)
//! epilogue     final HC mixer, MTP hidden-buffer copy, target head (logits
//!              rows), rejection-sampling placeholders
//! draft        MTP step 0 over every target row: FC prelude, attn HC mix,
//!              QSA, mlp HC mix, FP8-block MoE block, final mixer, greedy head
//!              on the request endpoints; then one endpoint pass folded
//!              `draft_tokens - 1` times
//! prep         engine preprocess placeholders
//! ```
//!
//! Every linear outside the experts is BF16 `F.linear`; there is no Inductor,
//! so every norm is the hyper-connection module's own entry norm
//! (`hc_combine_norm`, or `grouped_rms_norm` where no combine is pending: layer
//! 0 and the module right after a PLE block, whose `hc_combine` runs inside the
//! PLE worklet). The final mixer is the last norm.
//!
//! A step of at most the largest capture size replays a CUDA graph: the graph
//! pieces run on the token count padded to the next captured size. A uniform
//! verify step (no prefill) replays a FULL graph, so its eager breaks (QSA's
//! `_run_qsa`, the GDN core, the PLE lookup and conv) run padded too; a step
//! with a prefill runs them over the real rows. Larger steps run eagerly.
//! The draft recurrent pass replays the drafter's FULL graph at the padded
//! request count.
//!
//! Not modeled: the post-step bookkeeping launches (decision D8) and host
//! overhead.

use anyhow::Result;
use serde::Deserialize;

use crate::arch::contract::{SpeculativeArchInput, SpeculativeUnifiedModel};
use crate::arch::glm53_flash_vllm_fp8_kda_dsa_moe::graph_padded_tokens;
use crate::arch::kv_layout::{
    gqa_kv_bytes_per_token, hybrid_attention_block_tokens, EngineOverhead, KvCacheLayout,
    KvCachePart,
};
use crate::arch::qwen4_exp_model_cfg::{Qwen4ExpMixer, Qwen4ExpModelCfg};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::expert_demand::ExpertDemand;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, EmbeddingRowGatherKernel,
    EmbeddingRowGatherKernelConfig, EmbeddingRowGatherKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::bf16_gated_shared_expert_local::Bf16GatedSharedExpertLocalWorkletConfig;
use crate::worklet::engine_prep_local::{
    EnginePrepLocalWorklet, EnginePrepLocalWorkletConfig, EnginePrepLocalWorkletInput,
    EnginePrepLocalWorkletResolved, UnitBytes,
};
use crate::worklet::glm53_common::{atomic, elementwise, push_or_zero, Glm53WeightPrecision};
use crate::worklet::glm53_moe_local::Glm53RoutedMoeLocalWorkletConfig;
use crate::worklet::hc_mix_local::{
    HcMixEntry, HcMixLocalWorklet, HcMixLocalWorkletConfig, HcMixLocalWorkletInput,
    HcMixLocalWorkletResolved,
};
use crate::worklet::mtp_fc_prelude_local::{
    MtpFcPreludeLocalWorklet, MtpFcPreludeLocalWorkletConfig, MtpFcPreludeLocalWorkletInput,
    MtpFcPreludeLocalWorkletResolved,
};
use crate::worklet::nvfp4_moe_local::Nvfp4MoeLocalWorkletConfig;
use crate::worklet::ple_local::{
    PleLocalWorklet, PleLocalWorkletConfig, PleLocalWorkletInput, PleLocalWorkletResolved,
};
use crate::worklet::spec_head_local::{
    SpecHeadLocalWorklet, SpecHeadLocalWorkletConfig, SpecHeadLocalWorkletInput,
    SpecHeadLocalWorkletResolved,
};
use crate::worklet::spec_sample_local::{
    SpecSampleLocalWorklet, SpecSampleLocalWorkletConfig, SpecSampleLocalWorkletInput,
    SpecSampleLocalWorkletResolved,
};
use crate::worklet::vllm_bf16_gdn_spec_local::{
    VllmBf16GdnSpecLocalWorklet, VllmBf16GdnSpecLocalWorkletConfig,
    VllmBf16GdnSpecLocalWorkletInput, VllmBf16GdnSpecLocalWorkletResolved,
};
use crate::worklet::vllm_fp8_moe_shared_stream_local::{
    VllmFp8MoeSharedStreamLocalWorklet, VllmFp8MoeSharedStreamLocalWorkletConfig,
    VllmFp8MoeSharedStreamLocalWorkletInput, VllmFp8MoeSharedStreamLocalWorkletResolved,
};
use crate::worklet::vllm_moe_shared_stream_local::{
    VllmMoeSharedStreamLocalWorklet, VllmMoeSharedStreamLocalWorkletConfig,
    VllmMoeSharedStreamLocalWorkletInput, VllmMoeSharedStreamLocalWorkletResolved,
};
use crate::worklet::vllm_qsa_attn_local::{
    VllmQsaAttnLocalWorklet, VllmQsaAttnLocalWorkletConfig, VllmQsaAttnLocalWorkletInput,
    VllmQsaAttnLocalWorkletResolved,
};

const ARCH_KIND: &str = "qwen4_exp_vllm_nvfp4_gdn_qsa_moe_speculative";
const ACTIVATION_DTYPE: DType = DType::Bf16;
/// The attention kernel block the hybrid block is a multiple of
/// (`MultipleOf(16)`, `qsa.py:434-442`).
const KERNEL_BLOCK_TOKENS: u64 = 16;
/// `VocabParallelEmbedding` keeps the vocabulary in HBM.
const DEVICE: &str = "device";

/// BF16 `F.linear` in the serving fork's torch.
const GEMM_BACKENDS: &[&str] = &["torch_linear_vllm"];
/// The fork's Triton kernels (HC, PLE, QSA, GDN conv / post-conv / norm).
const VLLM_TRITON_BACKENDS: &[&str] = &["vllm_triton"];
/// Eager torch launches, measured in the serving fork's venv (glue_map.md).
const TORCH_BACKENDS: &[&str] = &["torch_vllm"];
const STRIDED_COPY_BACKENDS: &[&str] = &["torch_strided_copy_vllm"];
const CAT_BACKENDS: &[&str] = &["torch_cat_vllm"];
const INDEXED_ROW_COPY_BACKENDS: &[&str] = &["torch_vllm"];
/// Triton launches priced as byte placeholders (sampler, engine prep).
const TRITON_ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
/// `F.embedding` over the BF16 vocabulary.
const EMBEDDING_BACKENDS: &[&str] = &["torch"];
const GEMMA_NORM_BACKENDS: &[&str] = &["vllm_gemma_native"];
const CHUNK_DELTA_RULE_BACKENDS: &[&str] = &["flashinfer"];
const SIGMOID_GATING_DECODE_BACKENDS: &[&str] = &["vllm_triton_sigmoid_gating"];
/// `fused_gdn_decode_post_conv_mtp`.
const FUSED_VERIFY_BACKENDS: &[&str] = &["vllm_cuda"];
const TOPK_BACKENDS: &[&str] = &["vllm_cuda"];
const COOPERATIVE_TOPK_BACKENDS: &[&str] = &["vllm_cuda_cooperative"];
/// `reshape_and_cache_flash`.
const KV_APPEND_BACKENDS: &[&str] = &["vllm_cuda"];
const QUANT_BACKENDS: &[&str] = &["vllm_cuda"];
const NVFP4_MOE_BACKENDS: &[&str] = &["flashinfer_trtllm_sm100"];
const FP8_BLOCK_MOE_BACKENDS: &[&str] = &["flashinfer_trtllm_fp8_block_sm100"];
/// The body experts' NVFP4 scale group.
const NVFP4_GROUP_SIZE: u32 = 16;
/// `RoutingMethodType.Renormalize` with `norm_topk_prob`.
const ROUTING_METHOD: &str = "renormalize_naive";
const INT32_BYTES: u32 = 4;
const INT64_BYTES: u32 = 8;

/// vLLM `--kv-cache-dtype` / the QSA indexer's `indexer_kv_dtype`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen4ExpCacheDtype {
    #[default]
    Fp8,
    Bf16,
}

impl Qwen4ExpCacheDtype {
    pub const CHOICES: [&'static str; 2] = ["fp8", "bf16"];

    pub fn dtype(self) -> DType {
        match self {
            Self::Fp8 => DType::Fp8E4m3,
            Self::Bf16 => DType::Bf16,
        }
    }
}

/// Where the PLE n-gram table lives: pinned host memory read over UVA
/// (`--engram-config '{"cpu_offload": true}'`, the fork's default) or HBM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PleTableResidency {
    #[default]
    HostUva,
    Device,
}

impl PleTableResidency {
    pub const CHOICES: [&'static str; 2] = ["host_uva", "device"];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::HostUva => "host_uva",
            Self::Device => DEVICE,
        }
    }
}

/// Deployment layout: one GPU, MTP drafting.
#[derive(Clone, Debug)]
pub struct Qwen4ExpVllmParallel {
    pub max_model_len: u32,
    pub gpu_name: String,
    /// vLLM `--cudagraph-capture-sizes`; empty runs eager (no padding).
    pub cudagraph_capture_sizes: Vec<u32>,
    /// MTP `num_speculative_tokens`.
    pub draft_tokens: u32,
    pub kv_cache_dtype: Qwen4ExpCacheDtype,
    pub indexer_kv_dtype: Qwen4ExpCacheDtype,
    pub ple_table_residency: PleTableResidency,
}

/// A run of layers with one identical shape, folded by `Scale{n}`.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen4ExpLayerGroup {
    pub label: String,
    pub layers: Vec<u32>,
    mixer: Qwen4ExpMixer,
    /// `GroupedNorm` where no combine is pending: layer 0 and a PLE layer.
    attn_entry: HcMixEntry,
    /// The layer starts with a PLE block; `Some(pending combine)`.
    ple: Option<bool>,
}

pub fn layer_groups(model: &Qwen4ExpModelCfg) -> Vec<Qwen4ExpLayerGroup> {
    let mut groups: Vec<Qwen4ExpLayerGroup> = Vec::new();
    for (layer, &mixer) in model.mixers.iter().enumerate() {
        let layer = layer as u32;
        let ple = model.ple_layers.contains(&layer).then_some(layer > 0);
        let attn_entry = if layer == 0 || ple.is_some() {
            HcMixEntry::GroupedNorm
        } else {
            HcMixEntry::CombineNorm
        };
        if let Some(group) = groups
            .iter_mut()
            .find(|g| g.mixer == mixer && g.attn_entry == attn_entry && g.ple == ple)
        {
            group.layers.push(layer);
            continue;
        }
        let kind = match mixer {
            Qwen4ExpMixer::Gdn => "gdn",
            Qwen4ExpMixer::Qsa => "qsa",
        };
        let label = match (layer, ple) {
            (_, Some(_)) => format!("{kind}_ple"),
            (0, None) => format!("{kind}_first"),
            _ => kind.to_string(),
        };
        groups.push(Qwen4ExpLayerGroup {
            label,
            layers: vec![layer],
            mixer,
            attn_entry,
            ple,
        });
    }
    groups
}

/// Every configuration the model is built from.
#[derive(Clone, Debug)]
pub struct Qwen4ExpVllmConfigs {
    pub model: Qwen4ExpModelCfg,
    pub parallel: Qwen4ExpVllmParallel,
    pub groups: Vec<Qwen4ExpLayerGroup>,
    pub kv_cache_layout: KvCacheLayout,
    pub embedding: EmbeddingRowGatherKernelConfig,
    /// `hidden.repeat(1, hc_count)`.
    pub hidden_repeat: ElementwiseKernelConfig,
    pub hc_grouped: HcMixLocalWorkletConfig,
    pub hc_combine: HcMixLocalWorkletConfig,
    pub final_mixer: HcMixLocalWorkletConfig,
    pub gdn: VllmBf16GdnSpecLocalWorkletConfig,
    /// QSA at the verify width (target layers and MTP step 0).
    pub qsa: VllmQsaAttnLocalWorkletConfig,
    /// QSA at one query row (MTP recurrent passes).
    pub qsa_recurrent: VllmQsaAttnLocalWorkletConfig,
    /// PLE after layer 0 (a combine is pending) and in layer 0.
    pub ple: PleLocalWorkletConfig,
    pub ple_first: PleLocalWorkletConfig,
    pub moe: VllmMoeSharedStreamLocalWorkletConfig,
    /// `_mtp_hidden_buffer[:T].copy_(multi_hidden)`.
    pub mtp_hidden_copy: ElementwiseKernelConfig,
    /// The target's lm_head, which the drafter reuses.
    pub head: SpecHeadLocalWorkletConfig,
    pub sample: SpecSampleLocalWorkletConfig,
    pub mtp_prelude: MtpFcPreludeLocalWorkletConfig,
    pub mtp_moe: VllmFp8MoeSharedStreamLocalWorkletConfig,
    pub mtp_moe_recurrent: VllmFp8MoeSharedStreamLocalWorkletConfig,
    pub prep: EnginePrepLocalWorkletConfig,
}

/// Routed demand per MoE callable: the body at the verify width, the MTP
/// layer at the verify width (step 0) and at one row (recurrent passes).
pub struct Qwen4ExpDemand {
    pub body: ExpertDemand,
    pub mtp: ExpertDemand,
    pub mtp_recurrent: ExpertDemand,
}

pub fn build_configs(
    model: &Qwen4ExpModelCfg,
    parallel: &Qwen4ExpVllmParallel,
    demand: Qwen4ExpDemand,
) -> std::result::Result<Qwen4ExpVllmConfigs, BuildError> {
    if parallel.draft_tokens == 0 {
        return Err(fit_failed("an MTP deployment drafts at least one token"));
    }
    if model.mtp_layers != 1 {
        return Err(fit_failed(format!(
            "{} MTP layers; the modeled drafter cycles one",
            model.mtp_layers
        )));
    }
    let gpu = parallel.gpu_name.clone();
    let act = ACTIVATION_DTYPE.size_bytes();
    let streams = model.hc_count * model.hidden;
    let (kv_cache_layout, kv_block_size) = kv_cache_layout(model, parallel);
    let hc = |entry, use_combine| HcMixLocalWorkletConfig {
        hidden_size: model.hidden,
        hc_count: model.hc_count,
        hc_lowrank: model.hc_lowrank,
        entry,
        use_combine,
        dtype: ACTIVATION_DTYPE,
        gpu_name: gpu.clone(),
        hc_backends: VLLM_TRITON_BACKENDS.to_vec(),
        gemm_backends: GEMM_BACKENDS.to_vec(),
    };
    let qsa = |decode_query_len| VllmQsaAttnLocalWorkletConfig {
        hidden: model.hidden,
        num_q_heads: model.qsa_heads,
        num_kv_heads: model.qsa_kv_heads,
        head_dim: model.qsa_head_dim,
        rotary_dim: model.rotary_dim,
        // `--language-model-only` strips the multimodal RoPE sections.
        mrope: false,
        indexer_num_heads: model.indexer_heads,
        indexer_num_kv_heads: model.indexer_kv_heads,
        indexer_head_dim: model.indexer_head_dim,
        indexer_compress_ratio: model.indexer_compress_ratio,
        indexer_token_topk: model.indexer_token_topk,
        max_position_embeddings: model.max_position_embeddings,
        max_model_len: parallel.max_model_len,
        kv_block_size,
        num_speculative_tokens: parallel.draft_tokens,
        decode_query_len,
        activation_dtype: ACTIVATION_DTYPE,
        proj_dtype: ACTIVATION_DTYPE,
        kv_dtype: parallel.kv_cache_dtype.dtype(),
        indexer_kv_dtype: parallel.indexer_kv_dtype.dtype(),
        gpu_name: gpu.clone(),
        proj_gemm_backends: GEMM_BACKENDS.to_vec(),
        qk_norm_rope_backends: VLLM_TRITON_BACKENDS.to_vec(),
        indexer_prepare_backends: VLLM_TRITON_BACKENDS.to_vec(),
        indexer_logits_backends: VLLM_TRITON_BACKENDS.to_vec(),
        topk_backends: TOPK_BACKENDS.to_vec(),
        cooperative_topk_backends: COOPERATIVE_TOPK_BACKENDS.to_vec(),
        index_expand_backends: VLLM_TRITON_BACKENDS.to_vec(),
        kv_append_backends: KV_APPEND_BACKENDS.to_vec(),
        zero_fill_backends: TORCH_BACKENDS.to_vec(),
        attention_backends: VLLM_TRITON_BACKENDS.to_vec(),
    };
    let ple = |pending_combine| PleLocalWorkletConfig {
        hidden_size: model.hidden,
        hc_count: model.hc_count,
        ngram_context_len: model.ngram_size - 1,
        heads_per_ngram: model.heads_per_ngram,
        ple_embed_dim: model.ple_embed_dim,
        table_rows: model.ple_table_rows,
        residency: parallel.ple_table_residency.as_str().to_string(),
        table_dtype: model.ple_table_dtype,
        conv_kernel_size: model.ple_conv_kernel,
        conv_dilation: model.ple_conv_dilation(),
        pending_combine,
        dtype: ACTIVATION_DTYPE,
        gpu_name: gpu.clone(),
        triton_backends: VLLM_TRITON_BACKENDS.to_vec(),
        gather_backends: VLLM_TRITON_BACKENDS.to_vec(),
        elementwise_backends: TORCH_BACKENDS.to_vec(),
        gemm_backends: GEMM_BACKENDS.to_vec(),
    };
    let shared = Bf16GatedSharedExpertLocalWorkletConfig {
        hidden: model.hidden.into(),
        intermediate: model.shared_expert_intermediate.into(),
        activation_dtype: ACTIVATION_DTYPE,
        gpu_name: gpu.clone(),
        gemm_backends: GEMM_BACKENDS.to_vec(),
        act_backends: TORCH_BACKENDS.to_vec(),
        sigmoid_backends: TORCH_BACKENDS.to_vec(),
        gate_mul_backends: STRIDED_COPY_BACKENDS.to_vec(),
    };
    let mtp_moe = |expert_demand| VllmFp8MoeSharedStreamLocalWorkletConfig {
        hidden: model.hidden.into(),
        num_experts: model.num_experts.into(),
        activation_dtype: ACTIVATION_DTYPE,
        gpu_name: gpu.clone(),
        router_backends: GEMM_BACKENDS.to_vec(),
        scale_transpose_backends: STRIDED_COPY_BACKENDS.to_vec(),
        add_backends: TORCH_BACKENDS.to_vec(),
        shared: shared.clone(),
        routed: Glm53RoutedMoeLocalWorkletConfig {
            hidden: model.hidden.into(),
            moe_intermediate: model.moe_intermediate.into(),
            num_experts: model.num_experts.into(),
            ep_size: 1,
            top_k: model.experts_per_token,
            n_group: 0,
            topk_group: 0,
            routed_scaling_numerator: 1,
            routed_scaling_denominator: 1,
            precision: Glm53WeightPrecision::Fp8Block,
            routing_method: ROUTING_METHOD.into(),
            activation_dtype: ACTIVATION_DTYPE,
            gpu_name: gpu.clone(),
            quant_backends: QUANT_BACKENDS.to_vec(),
            fused_moe_backends: FP8_BLOCK_MOE_BACKENDS.to_vec(),
            expert_demand,
            folded_rank_position: 0,
        },
    };
    let blocks_per_request = parallel.max_model_len.div_ceil(kv_block_size);
    Ok(Qwen4ExpVllmConfigs {
        groups: layer_groups(model),
        embedding: EmbeddingRowGatherKernelConfig {
            backends: EMBEDDING_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            num_heads: 1,
            head_dim: model.hidden,
            table_rows: model.vocab_size.into(),
            residency: DEVICE.to_string(),
            weight_dtype: ACTIVATION_DTYPE,
        },
        // A strided copy priced at its output bytes (glue_map.md A3).
        hidden_repeat: elementwise(STRIDED_COPY_BACKENDS, &gpu, streams * act, streams * act),
        hc_grouped: hc(HcMixEntry::GroupedNorm, true),
        hc_combine: hc(HcMixEntry::CombineNorm, true),
        final_mixer: hc(HcMixEntry::CombineNorm, false),
        gdn: VllmBf16GdnSpecLocalWorkletConfig {
            hidden: model.hidden,
            num_key_heads: model.gdn_key_heads,
            num_value_heads: model.gdn_value_heads,
            key_head_dim: model.gdn_key_head_dim,
            value_head_dim: model.gdn_value_head_dim,
            conv_kernel_size: model.gdn_conv_kernel,
            tokens_per_request: parallel.draft_tokens + 1,
            activation_dtype: ACTIVATION_DTYPE,
            conv_state_dtype: ACTIVATION_DTYPE,
            ssm_state_dtype: model.gdn_state_dtype,
            gpu_name: gpu.clone(),
            gemm_backends: GEMM_BACKENDS.to_vec(),
            torch_elementwise_backends: TORCH_BACKENDS.to_vec(),
            strided_copy_backends: STRIDED_COPY_BACKENDS.to_vec(),
            cat_backends: CAT_BACKENDS.to_vec(),
            indexed_row_copy_backends: INDEXED_ROW_COPY_BACKENDS.to_vec(),
            causal_conv_prefill_backends: VLLM_TRITON_BACKENDS.to_vec(),
            prefill_post_conv_backends: VLLM_TRITON_BACKENDS.to_vec(),
            chunk_delta_rule_backends: CHUNK_DELTA_RULE_BACKENDS.to_vec(),
            causal_conv_decode_backends: VLLM_TRITON_BACKENDS.to_vec(),
            sigmoid_gating_decode_backends: SIGMOID_GATING_DECODE_BACKENDS.to_vec(),
            fused_verify_backends: FUSED_VERIFY_BACKENDS.to_vec(),
            gated_rms_norm_backends: VLLM_TRITON_BACKENDS.to_vec(),
        },
        qsa: qsa(parallel.draft_tokens + 1),
        qsa_recurrent: qsa(1),
        ple: ple(true),
        ple_first: ple(false),
        moe: VllmMoeSharedStreamLocalWorkletConfig {
            hidden: model.hidden.into(),
            num_experts: model.num_experts.into(),
            activation_dtype: ACTIVATION_DTYPE,
            gpu_name: gpu.clone(),
            router_backends: GEMM_BACKENDS.to_vec(),
            add_backends: TORCH_BACKENDS.to_vec(),
            shared: shared.clone(),
            routed: Nvfp4MoeLocalWorkletConfig {
                hidden: model.hidden.into(),
                moe_intermediate: model.moe_intermediate.into(),
                num_experts: model.num_experts.into(),
                ep_size: 1,
                tp_size: 1,
                top_k: model.experts_per_token,
                activation_dtype: ACTIVATION_DTYPE,
                gpu_name: gpu.clone(),
                quant_backends: QUANT_BACKENDS.to_vec(),
                moe_backends: NVFP4_MOE_BACKENDS.to_vec(),
                weight_format: DType::Nvfp4E2m1,
                activation_format: None,
                group_size: NVFP4_GROUP_SIZE,
                routing_method: ROUTING_METHOD.into(),
                n_group: 0,
                topk_group: 0,
                routed_scaling_numerator: 1,
                routed_scaling_denominator: 1,
                expert_demand: demand.body,
                folded_rank_position: 0,
            },
        },
        mtp_hidden_copy: elementwise(TORCH_BACKENDS, &gpu, streams * act, streams * act),
        head: SpecHeadLocalWorkletConfig {
            gpu_name: gpu.clone(),
            hidden: model.hidden.into(),
            vocab_size: model.vocab_size.into(),
            activation_dtype: ACTIVATION_DTYPE,
            lm_head_dtype: ACTIVATION_DTYPE,
            // `compute_logits` returns BF16 (the capture's `ArgMaxOps<BFloat16>`).
            logits_dtype: ACTIVATION_DTYPE,
            elementwise_backends: TORCH_BACKENDS.to_vec(),
            gemm_backends: GEMM_BACKENDS.to_vec(),
        },
        sample: SpecSampleLocalWorkletConfig {
            gpu_name: gpu.clone(),
            vocab_size: model.vocab_size.into(),
            logits_dtype: ACTIVATION_DTYPE,
            elementwise_backends: TRITON_ELEMENTWISE_BACKENDS.to_vec(),
        },
        mtp_prelude: MtpFcPreludeLocalWorkletConfig {
            gpu_name: gpu.clone(),
            hidden: model.hidden.into(),
            hc_count: model.hc_count,
            vocab_size: model.vocab_size.into(),
            activation_dtype: ACTIVATION_DTYPE,
            fc_dtype: ACTIVATION_DTYPE,
            embedding_backends: EMBEDDING_BACKENDS.to_vec(),
            rms_norm_backends: GEMMA_NORM_BACKENDS.to_vec(),
            gemm_backends: GEMM_BACKENDS.to_vec(),
        },
        mtp_moe: mtp_moe(demand.mtp),
        mtp_moe_recurrent: mtp_moe(demand.mtp_recurrent),
        // Index rows the input build moves: per request a block-table row and
        // four int64 scalars (token, position, sequence length, query start);
        // per token its slot and its sparse-attention work slot; per request
        // the align pre-copy's state index pair.
        prep: EnginePrepLocalWorkletConfig {
            gpu_name: gpu.clone(),
            input_prep_bytes_per_request: UnitBytes {
                input: blocks_per_request * INT32_BYTES + 4 * INT64_BYTES,
                output: blocks_per_request * INT32_BYTES + 4 * INT64_BYTES,
            },
            qsa_metadata_bytes_per_token: UnitBytes {
                input: INT64_BYTES,
                output: 2 * INT64_BYTES,
            },
            mamba_align_precopy_bytes_per_request: UnitBytes {
                input: 2 * INT32_BYTES,
                output: 2 * INT32_BYTES,
            },
            elementwise_backends: TRITON_ELEMENTWISE_BACKENDS.to_vec(),
        },
        kv_cache_layout,
        model: model.clone(),
        parallel: parallel.clone(),
    })
}

/// One request's cache in vLLM's hybrid allocator, and the hybrid block.
///
/// The block is the smallest multiple of the attention kernel block whose QSA
/// KV page holds the largest mamba page (the GDN state, `conv_kernel - 1 + k`
/// conv taps plus the SSM state). Every KV-cache group shares one block-id
/// pool whose stride is the widest group's page: the QSA layers' KV and
/// compressed index keys (the target's and the MTP layer's). The GDN layers
/// split into groups of at most that many layers, each spending one stride
/// per block; the PLE conv state and the raw-key ring (never prefix-cached)
/// take a stride each. A recurrent state keeps `1 + k` blocks (the running
/// state and one per speculative token); L5 adds align mode's previous-step
/// block to vLLM's concurrency estimate only, since a running request frees
/// it the step after it crosses a block. For Flash-Next at FP8: a 3200-token
/// block and a 43,929,600 B stride.
fn kv_cache_layout(
    model: &Qwen4ExpModelCfg,
    parallel: &Qwen4ExpVllmParallel,
) -> (KvCacheLayout, u32) {
    let act = u64::from(ACTIVATION_DTYPE.size_bytes());
    let k = u64::from(parallel.draft_tokens);
    let attn_layers = u64::from(model.count(Qwen4ExpMixer::Qsa) + model.mtp_layers);
    let gdn_layers = u64::from(model.count(Qwen4ExpMixer::Gdn));
    let ple_layers = model.ple_layers.len() as u64;
    let ratio = u64::from(model.indexer_compress_ratio);
    let kv_page = gqa_kv_bytes_per_token(
        1,
        u64::from(model.qsa_kv_heads),
        u64::from(model.qsa_head_dim),
        u64::from(model.qsa_head_dim),
        u64::from(parallel.kv_cache_dtype.dtype().size_bytes()),
        1,
    );
    // FP8 index keys carry no scales (`indexer_qsa.py:147-175`).
    let index_entry = u64::from(model.indexer_kv_heads * model.indexer_head_dim)
        * u64::from(parallel.indexer_kv_dtype.dtype().size_bytes());
    let gdn_state =
        u64::from(model.gdn_conv_channels()) * (u64::from(model.gdn_conv_kernel - 1) + k) * act
            + u64::from(model.gdn_value_heads * model.gdn_key_head_dim * model.gdn_value_head_dim)
                * u64::from(model.gdn_state_dtype.size_bytes());
    let ple_state = u64::from(model.hc_count * model.hidden)
        * (u64::from(model.ple_conv_state_len()) + k)
        * act;
    let block_tokens =
        hybrid_attention_block_tokens(KERNEL_BLOCK_TOKENS, gdn_state.max(ple_state), kv_page);
    let block = u64::from(block_tokens);
    let stride = attn_layers * (block * kv_page + block.div_ceil(ratio) * index_entry);
    // The raw-key ring holds the open group plus every speculative row in
    // whole groups, BF16 (`common/qsa_cache.py` `CircularBufferSpec`).
    let ring = ratio * (ratio + k).div_ceil(ratio) * u64::from(model.indexer_head_dim) * act;
    let copies = 1 + parallel.draft_tokens;
    let mut layout = KvCacheLayout::new()
        .with_part(KvCachePart::per_token("qsa_kv", attn_layers * kv_page))
        .with_part(KvCachePart::compressed(
            "qsa_index",
            model.indexer_compress_ratio,
            attn_layers * index_entry,
        ))
        .with_part(
            KvCachePart::per_request("qsa_ring", attn_layers * ring)
                .padded_to(stride)
                .not_prefix_reusable(),
        )
        .with_part(
            KvCachePart::per_request("gdn_state", gdn_layers * gdn_state)
                .padded_to(gdn_layers.div_ceil(attn_layers) * stride)
                .with_copies(copies),
        );
    if ple_layers > 0 {
        layout = layout.with_part(
            KvCachePart::per_request("ple_conv", ple_layers * ple_state)
                .padded_to(ple_layers.div_ceil(attn_layers) * stride)
                .with_copies(copies),
        );
    }
    let layout = layout.with_engine(EngineOverhead {
        block_tokens: Some(block_tokens),
        fixed_device_bytes: 0,
    });
    (layout, block_tokens)
}

#[derive(Clone, Debug)]
pub struct Qwen4ExpVllmResolved {
    pub raw_cfg: Qwen4ExpVllmConfigs,
    pub hc_grouped: HcMixLocalWorkletResolved,
    pub hc_combine: HcMixLocalWorkletResolved,
    pub final_mixer: HcMixLocalWorkletResolved,
    pub gdn: VllmBf16GdnSpecLocalWorkletResolved,
    pub qsa: VllmQsaAttnLocalWorkletResolved,
    pub qsa_recurrent: VllmQsaAttnLocalWorkletResolved,
    pub ple: PleLocalWorkletResolved,
    pub ple_first: PleLocalWorkletResolved,
    pub moe: VllmMoeSharedStreamLocalWorkletResolved,
    pub head: SpecHeadLocalWorkletResolved,
    pub sample: SpecSampleLocalWorkletResolved,
    pub mtp_prelude: MtpFcPreludeLocalWorkletResolved,
    pub mtp_moe: VllmFp8MoeSharedStreamLocalWorkletResolved,
    pub mtp_moe_recurrent: VllmFp8MoeSharedStreamLocalWorkletResolved,
    pub prep: EnginePrepLocalWorkletResolved,
}

pub fn resolve_configs(cfgs: &Qwen4ExpVllmConfigs) -> Qwen4ExpVllmResolved {
    Qwen4ExpVllmResolved {
        hc_grouped: HcMixLocalWorklet::resolve_config(&cfgs.hc_grouped),
        hc_combine: HcMixLocalWorklet::resolve_config(&cfgs.hc_combine),
        final_mixer: HcMixLocalWorklet::resolve_config(&cfgs.final_mixer),
        gdn: VllmBf16GdnSpecLocalWorklet::resolve_config(&cfgs.gdn),
        qsa: VllmQsaAttnLocalWorklet::resolve_config(&cfgs.qsa),
        qsa_recurrent: VllmQsaAttnLocalWorklet::resolve_config(&cfgs.qsa_recurrent),
        ple: PleLocalWorklet::resolve_config(&cfgs.ple),
        ple_first: PleLocalWorklet::resolve_config(&cfgs.ple_first),
        moe: VllmMoeSharedStreamLocalWorklet::resolve_config(&cfgs.moe),
        head: SpecHeadLocalWorklet::resolve_config(&cfgs.head),
        sample: SpecSampleLocalWorklet::resolve_config(&cfgs.sample),
        mtp_prelude: MtpFcPreludeLocalWorklet::resolve_config(&cfgs.mtp_prelude),
        mtp_moe: VllmFp8MoeSharedStreamLocalWorklet::resolve_config(&cfgs.mtp_moe),
        mtp_moe_recurrent: VllmFp8MoeSharedStreamLocalWorklet::resolve_config(
            &cfgs.mtp_moe_recurrent,
        ),
        prep: EnginePrepLocalWorklet::resolve_config(&cfgs.prep),
        raw_cfg: cfgs.clone(),
    }
}

/// One forward pass as every layer sees it: the target pass, MTP step 0 (the
/// same rows) or a recurrent endpoint pass.
#[derive(Clone, Debug)]
struct Pass {
    /// Rows the graph pieces run: the token count padded to the next captured
    /// size, or the real count above the largest.
    padded: u32,
    /// A FULL-graph replay: the eager breaks run at `padded` too.
    full_graph: bool,
    /// `(cached prefix, new tokens)` of each prefill request.
    prefills: Vec<(u32, u32)>,
    /// Final context of each verify (or recurrent decode) request.
    decode_kv_lens: Vec<u32>,
    /// Query rows of each decode request.
    query_len: u32,
}

impl Pass {
    fn new(
        sizes: &[u32],
        prefills: Vec<(u32, u32)>,
        decode_kv_lens: Vec<u32>,
        query_len: u32,
    ) -> Self {
        let tokens = prefills.iter().map(|&(_, new)| new).sum::<u32>()
            + query_len * decode_kv_lens.len() as u32;
        let padded = graph_padded_tokens(sizes, tokens);
        let full_graph = prefills.is_empty()
            && sizes.last().is_some_and(|&largest| padded <= largest)
            && padded.is_multiple_of(query_len);
        Self {
            padded,
            full_graph,
            prefills,
            decode_kv_lens,
            query_len,
        }
    }

    fn tokens(&self) -> u32 {
        self.prefill_tokens() + self.query_len * self.decode_kv_lens.len() as u32
    }

    fn prefill_tokens(&self) -> u32 {
        self.prefills.iter().map(|&(_, new)| new).sum()
    }

    /// Rows of an eager break.
    fn eager_rows(&self) -> u32 {
        if self.full_graph {
            self.padded
        } else {
            self.tokens()
        }
    }

    fn qsa_input(&self) -> VllmQsaAttnLocalWorkletInput {
        VllmQsaAttnLocalWorkletInput {
            padded_tokens: self.padded,
            full_graph: self.full_graph,
            prefill_chunks: self.prefills.clone(),
            decode_kv_lens: self.decode_kv_lens.clone(),
        }
    }
}

enum Mixer {
    Gdn(VllmBf16GdnSpecLocalWorklet),
    Qsa(VllmQsaAttnLocalWorklet),
}

/// One built layer group: the leaves of a single layer of its shape.
struct LayerGroup {
    label: String,
    layers: Vec<u32>,
    ple: Option<PleLocalWorklet>,
    attn_mix: HcMixLocalWorklet,
    mixer: Mixer,
    mlp_mix: HcMixLocalWorklet,
    moe: VllmMoeSharedStreamLocalWorklet,
}

impl LayerGroup {
    fn build(
        name: &str,
        group: &Qwen4ExpLayerGroup,
        r: &Qwen4ExpVllmResolved,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        let p = format!("{name}.{}", group.label);
        Ok(Self {
            label: group.label.clone(),
            layers: group.layers.clone(),
            ple: group
                .ple
                .map(|pending| {
                    let resolved = if pending { &r.ple } else { &r.ple_first };
                    PleLocalWorklet::build(format!("{p}.ple"), resolved.clone(), bridge)
                })
                .transpose()?,
            attn_mix: HcMixLocalWorklet::build(
                format!("{p}.attn_hc"),
                match group.attn_entry {
                    HcMixEntry::GroupedNorm => r.hc_grouped.clone(),
                    HcMixEntry::CombineNorm => r.hc_combine.clone(),
                },
                bridge,
            )?,
            mixer: match group.mixer {
                Qwen4ExpMixer::Gdn => Mixer::Gdn(VllmBf16GdnSpecLocalWorklet::build(
                    format!("{p}.gdn"),
                    r.gdn.clone(),
                    bridge,
                )?),
                Qwen4ExpMixer::Qsa => Mixer::Qsa(VllmQsaAttnLocalWorklet::build(
                    format!("{p}.qsa"),
                    r.qsa.clone(),
                    bridge,
                )?),
            },
            mlp_mix: HcMixLocalWorklet::build(format!("{p}.mlp_hc"), r.hc_combine.clone(), bridge)?,
            moe: VllmMoeSharedStreamLocalWorklet::build(format!("{p}.moe"), r.moe.clone(), bridge)?,
        })
    }

    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let mut layer = Vec::with_capacity(5);
        if let Some(ple) = &self.ple {
            layer.push(ple.compile(builder));
        }
        layer.push(self.attn_mix.compile(builder));
        layer.push(match &self.mixer {
            Mixer::Gdn(gdn) => gdn.compile(builder),
            Mixer::Qsa(qsa) => qsa.compile(builder),
        });
        layer.push(self.mlp_mix.compile(builder));
        layer.push(self.moe.compile(builder));
        CostNode::Labeled {
            label: format!("{} layers {:?}", self.label, self.layers),
            child: Box::new(CostNode::Scale {
                n: self.layers.len() as u32,
                child: Box::new(CostNode::Sum(layer)),
            }),
        }
    }

    fn eval(&self, pass: &Pass, ev: &mut Evaluator) {
        let rows = HcMixLocalWorkletInput {
            num_tokens: pass.padded,
        };
        let spec_requests = pass.decode_kv_lens.len() as u32;
        if let Some(ple) = &self.ple {
            ple.eval(
                &PleLocalWorkletInput {
                    graph_rows: pass.padded,
                    eager_rows: pass.eager_rows(),
                    decode_requests: 0,
                    spec_requests,
                    spec_query_len: pass.query_len,
                    prefill_requests: pass.prefills.len() as u32,
                    prefill_tokens: pass.prefill_tokens(),
                },
                ev,
            );
        }
        self.attn_mix.eval(&rows, ev);
        match &self.mixer {
            Mixer::Gdn(gdn) => gdn.eval(
                &VllmBf16GdnSpecLocalWorkletInput {
                    padded_tokens: pass.padded,
                    prefill_lengths: pass.prefills.iter().map(|&(_, new)| new).collect(),
                    spec_decodes: spec_requests,
                    non_spec_decodes: 0,
                },
                ev,
            ),
            Mixer::Qsa(qsa) => qsa.eval(&pass.qsa_input(), ev),
        }
        self.mlp_mix.eval(&rows, ev);
        self.moe.eval(
            &VllmMoeSharedStreamLocalWorkletInput {
                num_tokens: pass.padded,
            },
            ev,
        );
    }
}

/// The 48 target layers plus their head and sampler: the forward pass every
/// Qwen4Exp vLLM deployment runs, speculating or not. Its own struct so an
/// ordinary (non-MTP) model can share it by composition.
pub(crate) struct Qwen4ExpTargetForward {
    embedding: Op<EmbeddingRowGatherKernel>,
    hidden_repeat: Op<ElementwiseKernel>,
    groups: Vec<LayerGroup>,
    final_mixer: HcMixLocalWorklet,
    mtp_hidden_copy: Op<ElementwiseKernel>,
    head: SpecHeadLocalWorklet,
    sample: SpecSampleLocalWorklet,
}

impl Qwen4ExpTargetForward {
    fn build(
        name: &str,
        r: &Qwen4ExpVllmResolved,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        let cfg = &r.raw_cfg;
        Ok(Self {
            embedding: atomic(
                name,
                "embedding",
                cfg.embedding.clone(),
                EmbeddingRowGatherKernel::build,
                bridge,
            )?,
            hidden_repeat: atomic(
                name,
                "hidden_repeat",
                cfg.hidden_repeat.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            groups: cfg
                .groups
                .iter()
                .map(|group| LayerGroup::build(name, group, r, bridge))
                .collect::<std::result::Result<Vec<_>, _>>()?,
            final_mixer: HcMixLocalWorklet::build(
                format!("{name}.final_mixer"),
                r.final_mixer.clone(),
                bridge,
            )?,
            mtp_hidden_copy: atomic(
                name,
                "mtp_hidden_copy",
                cfg.mtp_hidden_copy.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            head: SpecHeadLocalWorklet::build(format!("{name}.head"), r.head.clone(), bridge)?,
            sample: SpecSampleLocalWorklet::build(
                format!("{name}.sample"),
                r.sample.clone(),
                bridge,
            )?,
        })
    }

    fn compile_children(&self, builder: &mut CostTreeBuilder) -> Vec<CostNode> {
        let mut children = vec![
            self.embedding.compile(builder),
            self.hidden_repeat.compile(builder),
        ];
        children.extend(self.groups.iter().map(|group| group.compile(builder)));
        children.push(self.final_mixer.compile(builder));
        children.push(self.mtp_hidden_copy.compile(builder));
        children.push(self.head.compile(builder));
        children.push(self.sample.compile(builder));
        children
    }

    fn eval(&self, batch: &NormalizedBatch, ev: &mut Evaluator) {
        let pass = &batch.target;
        let rows = pass.padded;
        push_or_zero(
            &self.embedding,
            EmbeddingRowGatherKernelInput { num_tokens: rows },
            false,
            ev,
        );
        push_or_zero(
            &self.hidden_repeat,
            ElementwiseKernelInput { num_tokens: rows },
            false,
            ev,
        );
        for group in &self.groups {
            group.eval(pass, ev);
        }
        self.final_mixer
            .eval(&HcMixLocalWorkletInput { num_tokens: rows }, ev);
        push_or_zero(
            &self.mtp_hidden_copy,
            ElementwiseKernelInput { num_tokens: rows },
            false,
            ev,
        );
        // The rejection sampler reads the logits unreduced.
        self.head.eval(
            &SpecHeadLocalWorkletInput {
                rows: batch.logits_rows,
                greedy_argmax: false,
            },
            ev,
        );
        self.sample.eval(
            &SpecSampleLocalWorkletInput {
                logits_rows: batch.logits_rows,
                requests: batch.request_count,
            },
            ev,
        );
    }
}

/// One MTP forward pass: prelude, the QSA decoder layer with its own indexer
/// and FP8-block experts, the final mixer and the greedy head (the target's
/// lm_head) on the request endpoints.
struct MtpPass {
    prelude: MtpFcPreludeLocalWorklet,
    /// Its pending combine injects the embedding with unit weight.
    attn_mix: HcMixLocalWorklet,
    qsa: VllmQsaAttnLocalWorklet,
    mlp_mix: HcMixLocalWorklet,
    moe: VllmFp8MoeSharedStreamLocalWorklet,
    final_mixer: HcMixLocalWorklet,
    head: SpecHeadLocalWorklet,
}

impl MtpPass {
    fn build(
        p: &str,
        r: &Qwen4ExpVllmResolved,
        qsa: &VllmQsaAttnLocalWorkletResolved,
        moe: &VllmFp8MoeSharedStreamLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        Ok(Self {
            prelude: MtpFcPreludeLocalWorklet::build(
                format!("{p}.prelude"),
                r.mtp_prelude.clone(),
                bridge,
            )?,
            attn_mix: HcMixLocalWorklet::build(
                format!("{p}.attn_hc"),
                r.hc_combine.clone(),
                bridge,
            )?,
            qsa: VllmQsaAttnLocalWorklet::build(format!("{p}.qsa"), qsa.clone(), bridge)?,
            mlp_mix: HcMixLocalWorklet::build(format!("{p}.mlp_hc"), r.hc_combine.clone(), bridge)?,
            moe: VllmFp8MoeSharedStreamLocalWorklet::build(
                format!("{p}.moe"),
                moe.clone(),
                bridge,
            )?,
            final_mixer: HcMixLocalWorklet::build(
                format!("{p}.final_mixer"),
                r.final_mixer.clone(),
                bridge,
            )?,
            head: SpecHeadLocalWorklet::build(format!("{p}.head"), r.head.clone(), bridge)?,
        })
    }

    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            self.prelude.compile(builder),
            self.attn_mix.compile(builder),
            self.qsa.compile(builder),
            self.mlp_mix.compile(builder),
            self.moe.compile(builder),
            self.final_mixer.compile(builder),
            self.head.compile(builder),
        ])
    }

    fn eval(&self, pass: &Pass, endpoints: u32, ev: &mut Evaluator) {
        let rows = pass.padded;
        self.prelude
            .eval(&MtpFcPreludeLocalWorkletInput { batch_tokens: rows }, ev);
        let mix = HcMixLocalWorkletInput { num_tokens: rows };
        self.attn_mix.eval(&mix, ev);
        self.qsa.eval(&pass.qsa_input(), ev);
        self.mlp_mix.eval(&mix, ev);
        self.moe.eval(
            &VllmFp8MoeSharedStreamLocalWorkletInput { num_tokens: rows },
            ev,
        );
        self.final_mixer.eval(&mix, ev);
        self.head.eval(
            &SpecHeadLocalWorkletInput {
                rows: endpoints,
                greedy_argmax: true,
            },
            ev,
        );
    }
}

/// The MTP proposer: one pass over the target's whole scheduled batch, then
/// one pass per further drafted position over the request endpoints, compiled
/// once and folded `draft_tokens - 1` times (the capture's `REPEAT x2`).
struct MtpDraftStage {
    step_0: MtpPass,
    recurrent: Option<MtpPass>,
    recurrent_count: u32,
}

impl MtpDraftStage {
    fn compile(&self, builder: &mut CostTreeBuilder, name: &str) -> CostNode {
        let mut passes = vec![CostNode::Labeled {
            label: format!("{name}.mtp.step_0 [whole-batch pass]"),
            child: Box::new(self.step_0.compile(builder)),
        }];
        if let Some(recurrent) = &self.recurrent {
            passes.push(CostNode::Labeled {
                label: format!(
                    "{name}.mtp.recurrent [request-endpoint pass; {} steps]",
                    self.recurrent_count
                ),
                child: Box::new(CostNode::Scale {
                    n: self.recurrent_count,
                    child: Box::new(recurrent.compile(builder)),
                }),
            });
        }
        CostNode::Sum(passes)
    }

    fn eval(&self, batch: &NormalizedBatch, ev: &mut Evaluator) {
        self.step_0.eval(&batch.target, batch.request_count, ev);
        if let Some(recurrent) = &self.recurrent {
            recurrent.eval(&batch.recurrent, batch.request_count, ev);
        }
    }
}

/// Qwen4Exp on one GPU with its MTP layer as a real drafter.
pub struct Qwen4ExpVllmSpeculativeModel {
    pub name: String,
    target: Qwen4ExpTargetForward,
    draft: MtpDraftStage,
    prep: EnginePrepLocalWorklet,
    draft_tokens: u32,
    max_model_len: u32,
    /// Sorted, deduplicated capture sizes.
    cudagraph_capture_sizes: Vec<u32>,
    kv_cache_layout: KvCacheLayout,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: Qwen4ExpVllmResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<Qwen4ExpVllmSpeculativeModel, BuildError> {
    let cfg = &resolved.raw_cfg;
    let n = name.as_str();
    let draft_tokens = cfg.parallel.draft_tokens;
    let recurrent_count = draft_tokens - 1;
    let mut model = Qwen4ExpVllmSpeculativeModel {
        target: Qwen4ExpTargetForward::build(n, &resolved, bridge)?,
        draft: MtpDraftStage {
            step_0: MtpPass::build(
                &format!("{n}.mtp.step_0"),
                &resolved,
                &resolved.qsa,
                &resolved.mtp_moe,
                bridge,
            )?,
            recurrent: (recurrent_count > 0)
                .then(|| {
                    MtpPass::build(
                        &format!("{n}.mtp.recurrent"),
                        &resolved,
                        &resolved.qsa_recurrent,
                        &resolved.mtp_moe_recurrent,
                        bridge,
                    )
                })
                .transpose()?,
            recurrent_count,
        },
        prep: EnginePrepLocalWorklet::build(format!("{n}.prep"), resolved.prep.clone(), bridge)?,
        draft_tokens,
        max_model_len: cfg.parallel.max_model_len,
        cudagraph_capture_sizes: {
            let mut sizes = cfg.parallel.cudagraph_capture_sizes.clone();
            sizes.sort_unstable();
            sizes.dedup();
            sizes
        },
        kv_cache_layout: cfg.kv_cache_layout.clone(),
        cost_flat: Vec::new(),
        n_slots: 0,
        name,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

impl Qwen4ExpVllmSpeculativeModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let mut children = self.target.compile_children(&mut builder);
        children.push(CostNode::Labeled {
            label: format!("MTP layer [{}-deep draft]", self.draft_tokens),
            child: Box::new(self.draft.compile(&mut builder, &self.name)),
        });
        children.push(self.prep.compile(&mut builder));
        let root = CostNode::Labeled {
            label: format!(
                "{} (Qwen4ExpVllmSpeculativeModel) [local (1 GPU); MTP draft tokens={}; \
                 verify width={}; timing_context<={}]",
                self.name,
                self.draft_tokens,
                self.draft_tokens + 1,
                self.max_model_len
            ),
            child: Box::new(CostNode::Sum(children)),
        };
        builder.finish(root)
    }

    fn normalize(
        &self,
        input: &SpeculativeArchInput,
    ) -> std::result::Result<NormalizedBatch, String> {
        normalize_speculative_input(
            input,
            self.draft_tokens,
            self.max_model_len,
            &self.cudagraph_capture_sizes,
        )
    }

    fn eval_checked(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: Option<&mut Vec<SlotInput>>,
    ) -> LeafMetrics {
        let batch = self.normalize(batch).unwrap_or_else(|reason| {
            panic!("invalid Qwen4ExpVllmSpeculativeModel input: {reason}")
        });
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut evaluator = match inputs {
            Some(inputs) => Evaluator::with_inputs(slots, inputs),
            None => Evaluator::new(slots),
        };
        self.target.eval(&batch, &mut evaluator);
        self.draft.eval(&batch, &mut evaluator);
        self.prep.eval(
            &EnginePrepLocalWorkletInput {
                requests: batch.request_count,
                tokens: batch.target.tokens(),
            },
            &mut evaluator,
        );
        assert_eq!(
            evaluator.filled(),
            self.n_slots,
            "eval must fill every compiled slot"
        );
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }
}

impl SpeculativeUnifiedModel for Qwen4ExpVllmSpeculativeModel {
    fn check_input(&self, batch: &SpeculativeArchInput) -> Result<(), String> {
        self.normalize(batch).map(|_| ())
    }

    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout.clone()
    }

    fn max_model_len(&self) -> u32 {
        self.max_model_len
    }

    fn gpus_per_replica(&self) -> u16 {
        1
    }

    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }

    fn eval_speculative_iter(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        self.eval_checked(batch, slots, scratch, None)
    }

    fn eval_speculative_iter_with_inputs(
        &self,
        batch: &SpeculativeArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        let total = self.eval_checked(batch, slots, scratch, Some(inputs));
        assert_eq!(
            inputs.len(),
            self.n_slots,
            "slot inputs must align with compiled slots"
        );
        total
    }
}

/// One speculative iteration, lowered to the passes it runs.
#[derive(Clone, Debug)]
struct NormalizedBatch {
    /// The target verify pass; MTP step 0 runs the same rows.
    target: Pass,
    /// Prefill and verify requests alike.
    request_count: u32,
    /// Rows the target head scores: one per prefill request and every verify row.
    logits_rows: u32,
    /// The recurrent passes' shape: one row per request at its context
    /// advanced by the mean drafted position of the folded passes.
    recurrent: Pass,
}

fn normalize_speculative_input(
    input: &SpeculativeArchInput,
    draft_tokens: u32,
    max_model_len: u32,
    sorted_capture_sizes: &[u32],
) -> std::result::Result<NormalizedBatch, String> {
    // The compiled tree assumes one depth; another would bill the wrong rows.
    if input.draft_tokens != draft_tokens {
        return Err(format!(
            "input draft_tokens {} must match model draft_tokens {draft_tokens}",
            input.draft_tokens
        ));
    }
    let [group] = input.groups.as_slice() else {
        return Err(format!(
            "expected exactly one attention group, got {}",
            input.groups.len()
        ));
    };
    let verify_width = draft_tokens + 1;
    let mut prefill_tokens = 0_u32;
    let mut endpoints =
        Vec::with_capacity(group.prefill_chunk_pairs.len() + group.decode_requests.len());
    for (request, &(prefix, append)) in group.prefill_chunk_pairs.iter().enumerate() {
        if append == 0 {
            return Err(format!("prefill request {request} append must be nonzero"));
        }
        let context = prefix
            .checked_add(append)
            .ok_or_else(|| format!("prefill request {request} context overflows u32"))?;
        if context > max_model_len {
            return Err(format!(
                "prefill request {request} context {context} exceeds max_model_len {max_model_len}"
            ));
        }
        prefill_tokens = prefill_tokens
            .checked_add(append)
            .ok_or("prefill token sum overflows u32")?;
        endpoints.push(context);
    }
    if group.prefill_tokens != prefill_tokens {
        return Err(format!(
            "prefill_tokens {} must equal append sum {prefill_tokens}",
            group.prefill_tokens
        ));
    }
    let mut decode_kv_lens = Vec::with_capacity(group.decode_requests.len());
    for (request, decode) in group.decode_requests.iter().enumerate() {
        // The scheduler only submits whole verify groups; acceptance truncates
        // the output after this compute, not the compute.
        if decode.query_len != verify_width {
            return Err(format!(
                "decode request {request} query_len {} must equal verify width {verify_width}",
                decode.query_len
            ));
        }
        if !(verify_width..=max_model_len).contains(&decode.kv_len) {
            return Err(format!(
                "decode request {request} kv_len {} must be in {verify_width}..={max_model_len}",
                decode.kv_len
            ));
        }
        decode_kv_lens.push(decode.kv_len);
        endpoints.push(decode.kv_len);
    }
    let decode_tokens = verify_width
        .checked_mul(decode_kv_lens.len() as u32)
        .ok_or("decode token sum overflows u32")?;
    if group.decode_tokens != decode_tokens {
        return Err(format!(
            "decode_tokens {} must equal query_len sum {decode_tokens}",
            group.decode_tokens
        ));
    }
    let total = prefill_tokens
        .checked_add(decode_tokens)
        .ok_or("batch token sum overflows u32")?;
    if group.batch_tokens != total {
        return Err(format!(
            "batch_tokens {} must equal prefill+decode {total}",
            group.batch_tokens
        ));
    }
    if total == 0 {
        return Err("an iteration must carry at least one token".into());
    }
    let request_count = group.request_count();
    // Recurrent pass i runs after i drafted positions; the fold bills one
    // evaluation at the mean advance of passes 1..=draft_tokens-1.
    let mean_advance = draft_tokens.div_ceil(2);
    let recurrent_contexts = endpoints
        .iter()
        .map(|&context| context.saturating_add(mean_advance).min(max_model_len))
        .collect();
    Ok(NormalizedBatch {
        target: Pass::new(
            sorted_capture_sizes,
            group.prefill_chunk_pairs.clone(),
            decode_kv_lens,
            verify_width,
        ),
        request_count,
        logits_rows: group.prefill_chunk_pairs.len() as u32 + decode_tokens,
        recurrent: Pass::new(sorted_capture_sizes, Vec::new(), recurrent_contexts, 1),
    })
}

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::{SpeculativeArchGroupInput, SpeculativeDecodeInput};
    use crate::timing::routing::RoutingDistribution;
    use std::path::Path;

    fn model_cfg() -> Qwen4ExpModelCfg {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("model/config/qwen3_8_flash_next_nvfp4.json");
        Qwen4ExpModelCfg::from_json(&path).unwrap()
    }

    /// The capture's FULL_AND_PIECEWISE sizes at chunk 2048: 1, 2, 4, 8..256
    /// by 8, then 272..2048 by 16.
    fn capture_sizes() -> Vec<u32> {
        let mut sizes = vec![1, 2, 4];
        sizes.extend((8..=256).step_by(8));
        sizes.extend((272..=2048).step_by(16));
        sizes
    }

    fn parallel() -> Qwen4ExpVllmParallel {
        Qwen4ExpVllmParallel {
            max_model_len: 131072,
            gpu_name: "NVIDIA B200".into(),
            cudagraph_capture_sizes: capture_sizes(),
            draft_tokens: 3,
            kv_cache_dtype: Qwen4ExpCacheDtype::Fp8,
            indexer_kv_dtype: Qwen4ExpCacheDtype::Fp8,
            ple_table_residency: PleTableResidency::HostUva,
        }
    }

    fn demand() -> Qwen4ExpDemand {
        let uniform = || ExpertDemand::popularity(&RoutingDistribution::uniform(512), 1);
        Qwen4ExpDemand {
            body: uniform(),
            mtp: uniform(),
            mtp_recurrent: uniform(),
        }
    }

    fn built() -> Qwen4ExpVllmSpeculativeModel {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let configs = build_configs(&model_cfg(), &parallel(), demand()).unwrap();
        build("unified".into(), resolve_configs(&configs), &bridge).unwrap()
    }

    fn step(prefills: Vec<(u32, u32)>, decodes: Vec<u32>) -> SpeculativeArchInput {
        let prefill_tokens = prefills.iter().map(|&(_, a)| a).sum::<u32>();
        let decode_tokens = 4 * decodes.len() as u32;
        SpeculativeArchInput {
            draft_tokens: 3,
            groups: vec![SpeculativeArchGroupInput {
                batch_tokens: prefill_tokens + decode_tokens,
                prefill_tokens,
                decode_tokens,
                prefill_chunk_pairs: prefills,
                decode_requests: decodes
                    .into_iter()
                    .map(|kv_len| SpeculativeDecodeInput {
                        kv_len,
                        query_len: 4,
                    })
                    .collect(),
                total_kv_len: 0,
            }],
            tokens_per_source_rank: Vec::new(),
        }
    }

    fn normalize(input: &SpeculativeArchInput) -> std::result::Result<NormalizedBatch, String> {
        normalize_speculative_input(input, 3, 131072, &capture_sizes())
    }

    /// Layer 0, the PLE layer 1, 34 more GDN layers and 12 QSA layers.
    #[test]
    fn layers_fold_one_one_thirty_four_twelve() {
        let groups = layer_groups(&model_cfg());
        let summary: Vec<(String, usize, HcMixEntry, Option<bool>)> = groups
            .into_iter()
            .map(|g| (g.label, g.layers.len(), g.attn_entry, g.ple))
            .collect();
        use HcMixEntry::*;
        assert_eq!(
            summary,
            [
                ("gdn_first".to_string(), 1, GroupedNorm, None),
                ("gdn_ple".to_string(), 1, GroupedNorm, Some(true)),
                ("gdn".to_string(), 34, CombineNorm, None),
                ("qsa".to_string(), 12, CombineNorm, None),
            ]
        );
    }

    #[test]
    fn configs_derive_the_profiled_shapes() {
        let c = build_configs(&model_cfg(), &parallel(), demand()).unwrap();
        let r = resolve_configs(&c);
        assert_eq!(r.hc_combine.down.n.get(), 336);
        assert_eq!(r.final_mixer.down.n.get(), 320);
        assert_eq!((r.gdn.qkvz.n.get(), r.gdn.ba.n.get()), (16384, 96));
        assert!(r.gdn.fused_verify);
        assert_eq!(r.qsa.qkv_proj.n.get(), 13312);
        assert_eq!(c.qsa.kv_block_size, 3200);
        assert_eq!(
            (c.qsa.decode_query_len, c.qsa_recurrent.decode_query_len),
            (4, 1)
        );
        assert_eq!((r.ple.lookup.num_heads, r.ple.lookup.head_dim), (16, 160));
        assert_eq!(r.ple.lookup.table_rows, 320_001_536);
        assert_eq!(r.ple.raw_cfg.conv_dilation, 3);
        assert!(r.ple.combine.is_some() && r.ple_first.combine.is_none());
        assert_eq!(r.head.lm_head.n.get(), 248320);
        assert_eq!(r.moe.routed.fused_moe.routing_method, "renormalize_naive");
        assert_eq!(c.hidden_repeat.output_bytes_per_token.get(), 20480);
    }

    #[test]
    fn padding_follows_the_graph_mode() {
        let shape = |input: SpeculativeArchInput| {
            let b = normalize(&input).unwrap();
            (
                b.target.padded,
                b.target.full_graph,
                b.target.eager_rows(),
                b.recurrent.padded,
                b.recurrent.full_graph,
            )
        };
        // 5 verify requests: 20 rows replay the 24-row FULL graph.
        assert_eq!(shape(step(vec![], vec![4096; 5])), (24, true, 24, 8, true));
        // A prefill makes the step piecewise: eager breaks at the real rows.
        assert_eq!(
            shape(step(vec![(0, 1536)], vec![4096; 128])),
            (2048, false, 2048, 136, true)
        );
        assert_eq!(
            shape(step(vec![(0, 1500)], vec![4096; 2])),
            (1520, false, 1508, 4, true)
        );
        // Past the largest captured size a step runs eagerly, unpadded.
        assert_eq!(
            shape(step(vec![(0, 8192)], vec![])),
            (8192, false, 8192, 1, true)
        );
        let b = normalize(&step(vec![(100, 28)], vec![131072, 64])).unwrap();
        assert_eq!(b.recurrent.decode_kv_lens, [130, 131072, 66]);
        assert_eq!((b.request_count, b.logits_rows), (3, 9));
    }

    #[test]
    fn normalization_rejects_malformed_batches() {
        let mut wrong_depth = step(vec![], vec![64]);
        wrong_depth.draft_tokens = 2;
        assert!(normalize(&wrong_depth).is_err());
        let mut ragged = step(vec![], vec![64]);
        ragged.groups[0].decode_requests[0].query_len = 3;
        assert!(normalize(&ragged).is_err());
        assert!(normalize(&step(vec![], vec![3])).is_err());
        assert!(normalize(&step(vec![], vec![131073])).is_err());
        assert!(normalize(&step(vec![(131000, 100)], vec![])).is_err());
        assert!(normalize(&step(vec![(0, 0)], vec![])).is_err());
        assert!(normalize(&step(vec![], vec![])).is_err());
        let mut miscounted = step(vec![], vec![64]);
        miscounted.groups[0].decode_tokens = 1;
        assert!(normalize(&miscounted).is_err());
    }

    #[test]
    fn the_manifest_names_every_component() {
        let model = built();
        for input in [
            step(vec![], vec![4096; 5]),
            step(vec![(0, 1536)], vec![4096; 128]),
            step(vec![(0, 8192)], vec![]),
            step(vec![(126976, 2048)], vec![]),
        ] {
            model.check_input(&input).unwrap();
        }
        let names: Vec<String> = model
            .cost_log_manifest()
            .slots
            .into_iter()
            .map(|slot| slot.name)
            .collect();
        assert_eq!(names.len(), model.n_slots);
        for name in [
            "unified.embedding",
            "unified.gdn_first.attn_hc.grouped_norm",
            "unified.gdn_ple.ple.hc_combine",
            "unified.gdn_ple.ple.lookup",
            "unified.gdn.gdn.verify_recurrent_norm",
            "unified.qsa.qsa.o_proj",
            "unified.qsa.moe.routed.fused_moe",
            "unified.final_mixer.down",
            "unified.head.lm_head",
            "unified.sample.logits_stats",
            "unified.mtp.step_0.prelude.fc_hidden",
            "unified.mtp.step_0.moe.routed.fused_moe",
            "unified.mtp.recurrent.qsa.o_proj",
            "unified.mtp.recurrent.head.logits_reduce",
            "unified.prep.input_prep",
        ] {
            assert!(names.iter().any(|n| n == name), "{name}");
        }
        assert!(!names.iter().any(|n| n.starts_with("unified.gdn_first.ple")));
    }

    /// The necessary-work maps name exactly the compiled locations (one GPU: no
    /// collectives). A one-token draft compiles no recurrent MTP pass, so it
    /// takes its own map.
    #[test]
    fn necessary_work_maps_cover_the_compiled_locations() {
        for (draft_tokens, file, slots) in [
            (
                3,
                include_str!(
                    "../../../model/work/location_maps/qwen4_exp_vllm_nvfp4_gdn_qsa_moe_speculative_unified.json"
                ),
                340,
            ),
            (
                1,
                include_str!(
                    "../../../model/work/location_maps/qwen4_exp_vllm_nvfp4_gdn_qsa_moe_speculative_single_draft_unified.json"
                ),
                286,
            ),
        ] {
            let mut layout = parallel();
            layout.draft_tokens = draft_tokens;
            let bridge = PerfApiBridge::new_uninit_for_test();
            bridge.enable_enumerate();
            let configs = build_configs(&model_cfg(), &layout, demand()).unwrap();
            let model = build("unified".into(), resolve_configs(&configs), &bridge).unwrap();
            let actual: std::collections::BTreeSet<String> = model
                .cost_log_manifest()
                .slots
                .into_iter()
                .map(|slot| slot.name)
                .collect();
            assert_eq!(actual.len(), slots);
            let map: serde_json::Value = serde_json::from_str(file).unwrap();
            assert_eq!(map["arch_types"][0], ARCH_KIND);
            let mapped: std::collections::BTreeSet<String> = map["locations"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row["location"].as_str().unwrap().to_string())
                .collect();
            assert_eq!(actual, mapped, "draft_tokens = {draft_tokens}");
        }
    }

    #[test]
    fn kv_cache_layout_reproduces_the_logged_hybrid_block() {
        let (layout, block) = kv_cache_layout(&model_cfg(), &parallel());
        assert_eq!(block, 3200);
        // 13 QSA layers (12 + MTP) at 1024 B/token FP8 KV plus 32 B/token index.
        assert_eq!(layout.per_token_bytes(), 13 * (1024 + 32));
        let gdn = layout.parts.iter().find(|p| p.name == "gdn_state").unwrap();
        assert_eq!(gdn.bytes_per_entry, 36 * 3_268_608);
        assert_eq!(gdn.overhead.padded_bytes_per_entry, Some(3 * 43_929_600));
        let ring = layout.parts.iter().find(|p| p.name == "qsa_ring").unwrap();
        assert_eq!(ring.bytes_per_entry, 26_624);
        assert!(!ring.prefix_reusable);
        let ple = layout.parts.iter().find(|p| p.name == "ple_conv").unwrap();
        assert_eq!(ple.bytes_per_entry, 245_760);
        // "Padding mamba page size by 0.25%".
        let padding = 100.0 * (f64::from(3200 * 1024) / 3_268_608.0 - 1.0);
        assert!((padding - 0.25).abs() < 0.005, "{padding}");
    }

    /// vLLM's logged pools at chunk 2048, prefix caching on: 79.80 GiB gives
    /// 1950 blocks and "31.45x" (4,122,425 tokens) at 131,072 tokens; the
    /// smoke's 78.57 GiB gives 30.97x. The estimate charges one request 41 KV
    /// blocks, 3 x 5 GDN, 5 PLE and 1 ring block (62); a running request holds
    /// 3 x 4 GDN and 4 PLE, so a short one holds 18 of the 1950.
    #[test]
    fn kv_cache_layout_reproduces_vllms_logged_concurrency() {
        use crate::worker::kv::{KvAccounting, KvAccountingOptions};
        let (layout, _) = kv_cache_layout(&model_cfg(), &parallel());
        for (gib, logged) in [(79.80, 4_122_425.0 / 131_072.0), (78.57, 30.97)] {
            let options = KvAccountingOptions {
                max_in_flight_tokens: Some(2 * 2048),
                prefix_caching: true,
                ..Default::default()
            };
            let budget = (gib * f64::from(1u32 << 30)) as u64;
            let accounting = KvAccounting::resolve(&layout, options, budget).unwrap();
            let concurrency = accounting.concurrency_at(131072);
            assert!(
                (concurrency / logged - 1.0).abs() < 0.005,
                "{gib} GiB: {concurrency} vs {logged}"
            );
        }
        // At 79.80 GiB the pool is 1950 strides, and a running request under
        // one block holds 18 of them: 108 at once, the captures' peak.
        let stride = 43_929_600;
        let budget = (79.80 * f64::from(1u32 << 30)) as u64;
        let options = KvAccountingOptions {
            max_in_flight_tokens: Some(2 * 2048),
            prefix_caching: true,
            ..Default::default()
        };
        let accounting = KvAccounting::resolve(&layout, options, budget).unwrap();
        let unit = accounting.bytes_per_unit();
        assert_eq!(accounting.capacity_units() * unit / stride, 1950);
        assert_eq!(accounting.held_while_running(256, 1) * unit, 18 * stride);
        assert_eq!(accounting.needed_to_admit(256) * unit, 18 * stride);
    }
}
