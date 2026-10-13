//! Qwen3.5-family hybrid decoders (`Qwen3_5ForConditionalGeneration`, text
//! only) with ModelOpt mixed-precision weights on one GPU under vLLM: Gated
//! DeltaNet and gated-GQA token mixers, each followed by a dense SwiGLU MLP.
//! FP8 per-tensor attention projections, BF16 `in_proj_ba`, NVFP4 MLP and
//! lm_head, FP8 KV cache. Measured against the vLLM capture of
//! `nvidia/Qwen3.8-27B-NVFP4` at TP1 on B200 (diverse_100).
//!
//! ```text
//! per layer      input norm, GDN or gated GQA, post-attention norm, MLP
//!                                                       (Scale per group)
//! head           final norm; logits row select, NVFP4 quant, lm_head
//! ```
//!
//! Every RMSNorm is one Inductor kernel that also adds the pending residual
//! terms (`multi_add_rms_norm`): layer 0's input norm reads the embedding
//! (fused in), a later input norm adds the residual and the previous layer's
//! attention and MLP outputs and stores the new residual, the post-attention
//! norm re-adds the attention output without storing it, and the final norm
//! adds all three. An input norm writes the BF16 row a BF16 consumer reads
//! (GDN's `in_proj_ba`) and the FP8 row the FP8 projection reads.
//!
//! A step of at most the largest capture size replays a CUDA graph
//! (FULL_AND_PIECEWISE): the compiled pieces run on the token count padded to
//! the next captured size; a uniform decode replays a FULL graph, so the GDN
//! core and decode attention run padded too, while a step with a prefill runs
//! them eagerly over the real tokens. Larger steps run eagerly.
//!
//! Sampling, the GDN state post-processing (`postprocess_mamba`) and host
//! overhead are not modeled.

use anyhow::Result;

use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::arch::glm53_flash_vllm_fp8_kda_dsa_moe::graph_padded_tokens;
use crate::arch::kv_layout::{hybrid_attention_state_layout, KvCacheLayout};
use crate::arch::qwen35_model_cfg::{
    Qwen35LinearPrecision, Qwen35Mixer, Qwen35ModelCfg, Qwen35Precisions,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, MultiAddRmsNormKernel,
    MultiAddRmsNormKernelConfig, MultiAddRmsNormKernelInput, Nvfp4QuantKernel,
    Nvfp4QuantKernelConfig, Nvfp4QuantKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, Evaluator, FlatCostNode,
    LeafMetrics, PerfApiBridge, SlotInput,
};
use crate::worklet::glm53_common::{atomic, elementwise, push_or_zero};
use crate::worklet::glm53_mlp_local::{NVFP4_GROUP_SIZE, NVFP4_SWIZZLED_SCALE_FORMAT};
use crate::worklet::{
    VllmNvfp4SwigluMlpLocalWorklet, VllmNvfp4SwigluMlpLocalWorkletConfig,
    VllmNvfp4SwigluMlpLocalWorkletInput, VllmNvfp4SwigluMlpLocalWorkletResolved,
    VllmQwen35GatedGqaLocalWorklet, VllmQwen35GatedGqaLocalWorkletConfig,
    VllmQwen35GatedGqaLocalWorkletInput, VllmQwen35GatedGqaLocalWorkletResolved,
    VllmQwen35GdnLocalWorklet, VllmQwen35GdnLocalWorkletConfig, VllmQwen35GdnLocalWorkletInput,
    VllmQwen35GdnLocalWorkletResolved,
};

const ARCH_KIND: &str = "qwen35_vllm_nvfp4_gdn_dense";
const ACTIVATION_DTYPE: DType = DType::Bf16;
/// vLLM `--kv-cache-dtype fp8` (the capture's). The query runs FP8 with it.
const KV_DTYPE: DType = DType::Fp8E4m3;
/// vLLM `--block-size` (the default): the attention kernel block the hybrid
/// block is a multiple of, and the page the KV append writes.
const KV_BLOCK_SIZE: u32 = 16;

const NORM_BACKENDS: &[&str] = &["vllm_inductor"];
/// ModelOpt FP8 per-tensor linears: FlashInfer's FP8 GEMM over the FP8 row
/// the producer already wrote (norm or static quant), no quant of its own.
const FP8_GEMM_BACKENDS: &[&str] = &["flashinfer_fp8_tensor"];
/// `F.linear` in vLLM's torch (`in_proj_ba`).
const BF16_GEMM_BACKENDS: &[&str] = &["torch_linear_vllm"];
const NVFP4_GEMM_BACKENDS: &[&str] = &["flashinfer_cutedsl"];
const NVFP4_QUANT_BACKENDS: &[&str] = &["vllm_cuda"];
const INDUCTOR_BACKENDS: &[&str] = &["vllm_inductor"];
const TRITON_ELEMENTWISE_BACKENDS: &[&str] = &["triton"];
const TORCH_ELEMENTWISE_BACKENDS: &[&str] = &["torch"];
const STRIDED_COPY_BACKENDS: &[&str] = &["torch_strided_copy"];
const CAT_BACKENDS: &[&str] = &["torch_cat"];
const VLLM_TRITON_BACKENDS: &[&str] = &["vllm_triton"];
/// FlashInfer's CUTLASS chunked delta rule (`auto` on SM100).
const CHUNK_DELTA_RULE_BACKENDS: &[&str] = &["flashinfer"];
const SIGMOID_GATING_DECODE_BACKENDS: &[&str] = &["vllm_triton_sigmoid_gating"];
/// `reshape_and_cache_flash`.
const KV_APPEND_BACKENDS: &[&str] = &["vllm_cuda"];
/// trtllm-gen: FA4 has no FP8-KV path at head dim 256.
const ATTENTION_BACKENDS: &[&str] = &["flashinfer_trtllm_gen"];
/// `hidden_states[logits_indices]`: an int64 index per selected row.
const INT64_INDEX_BYTES: u32 = 8;

/// The per-module precision this arch's leaves are built for.
const MODELED_PRECISIONS: Qwen35Precisions = Qwen35Precisions {
    gdn_in_proj_qkvz: Qwen35LinearPrecision::Fp8Tensor,
    gdn_in_proj_ba: Qwen35LinearPrecision::Bf16,
    gdn_out_proj: Qwen35LinearPrecision::Fp8Tensor,
    gqa_qkv_proj: Qwen35LinearPrecision::Fp8Tensor,
    gqa_o_proj: Qwen35LinearPrecision::Fp8Tensor,
    mlp: Qwen35LinearPrecision::Nvfp4,
    lm_head: Qwen35LinearPrecision::Nvfp4,
};

/// Deployment layout: one GPU.
#[derive(Clone, Debug)]
pub struct Qwen35VllmParallel {
    pub max_model_len: u32,
    pub gpu_name: String,
    /// vLLM `--cudagraph-capture-sizes`; empty runs eager (no padding).
    pub cudagraph_capture_sizes: Vec<u32>,
}

/// A run of layers with one identical shape, folded by `Scale{n}`. Layer 0
/// is its own group: its input norm reads the embedding.
#[derive(Clone, Debug)]
pub struct Qwen35LayerGroup {
    pub label: String,
    pub layers: Vec<u32>,
    mixer: Qwen35Mixer,
    first: bool,
}

fn layer_groups(model: &Qwen35ModelCfg) -> Vec<Qwen35LayerGroup> {
    let mut groups: Vec<Qwen35LayerGroup> = Vec::new();
    for (layer, &mixer) in model.mixers.iter().enumerate() {
        let first = layer == 0;
        if let Some(group) = groups
            .iter_mut()
            .find(|g| g.mixer == mixer && g.first == first)
        {
            group.layers.push(layer as u32);
            continue;
        }
        let kind = match mixer {
            Qwen35Mixer::Gdn => "gdn",
            Qwen35Mixer::Gqa => "gqa",
        };
        groups.push(Qwen35LayerGroup {
            label: if first {
                format!("{kind}_first")
            } else {
                kind.to_string()
            },
            layers: vec![layer as u32],
            mixer,
            first,
        });
    }
    groups
}

/// Every configuration the model is built from.
#[derive(Clone, Debug)]
pub struct Qwen35VllmConfigs {
    pub model: Qwen35ModelCfg,
    pub parallel: Qwen35VllmParallel,
    pub groups: Vec<Qwen35LayerGroup>,
    /// Layer 0's input norm, over the embedding.
    pub first_norm: MultiAddRmsNormKernelConfig,
    pub gdn_input_norm: MultiAddRmsNormKernelConfig,
    pub gqa_input_norm: MultiAddRmsNormKernelConfig,
    pub gdn: VllmQwen35GdnLocalWorkletConfig,
    pub gqa: VllmQwen35GatedGqaLocalWorkletConfig,
    pub post_attention_norm: MultiAddRmsNormKernelConfig,
    pub mlp: VllmNvfp4SwigluMlpLocalWorkletConfig,
    pub final_norm: MultiAddRmsNormKernelConfig,
    pub logits_select: ElementwiseKernelConfig,
    pub lm_head_quant: Nvfp4QuantKernelConfig,
    pub lm_head: SingleGemmKernelConfig,
}

pub fn build_configs(
    model: &Qwen35ModelCfg,
    parallel: &Qwen35VllmParallel,
) -> std::result::Result<Qwen35VllmConfigs, BuildError> {
    if model.precisions != MODELED_PRECISIONS {
        return Err(fit_failed(format!(
            "linear precisions {:?} are not the modeled mix {MODELED_PRECISIONS:?}",
            model.precisions
        )));
    }
    let gpu = parallel.gpu_name.clone();
    let p = &model.precisions;
    // What the norm in front of a mixer writes: the BF16 row when a BF16
    // projection reads it, the FP8 row for the FP8 projection.
    let input_norm = |addends: u32, store_sum: bool, mixer: Qwen35Mixer| {
        let (bf16_reader, fp8_reader) = match mixer {
            Qwen35Mixer::Gdn => (
                p.gdn_in_proj_ba == Qwen35LinearPrecision::Bf16,
                p.gdn_in_proj_qkvz == Qwen35LinearPrecision::Fp8Tensor,
            ),
            Qwen35Mixer::Gqa => (false, p.gqa_qkv_proj == Qwen35LinearPrecision::Fp8Tensor),
        };
        norm(
            &gpu,
            model.hidden,
            addends,
            store_sum,
            bf16_reader,
            fp8_reader,
        )
    };
    let first_mixer = *model
        .mixers
        .first()
        .ok_or_else(|| fit_failed("the model has no layers"))?;
    let hidden_bytes = model.hidden * ACTIVATION_DTYPE.size_bytes();
    Ok(Qwen35VllmConfigs {
        groups: layer_groups(model),
        first_norm: input_norm(1, false, first_mixer),
        gdn_input_norm: input_norm(3, true, Qwen35Mixer::Gdn),
        gqa_input_norm: input_norm(3, true, Qwen35Mixer::Gqa),
        gdn: VllmQwen35GdnLocalWorkletConfig {
            hidden: model.hidden,
            num_key_heads: model.gdn_key_heads,
            num_value_heads: model.gdn_value_heads,
            key_head_dim: model.gdn_key_head_dim,
            value_head_dim: model.gdn_value_head_dim,
            conv_kernel_size: model.gdn_conv_kernel,
            activation_dtype: ACTIVATION_DTYPE,
            conv_state_dtype: ACTIVATION_DTYPE,
            ssm_state_dtype: model.gdn_state_dtype,
            proj_dtype: p.gdn_in_proj_qkvz.gemm_dtype(),
            gpu_name: gpu.clone(),
            proj_gemm_backends: FP8_GEMM_BACKENDS.to_vec(),
            ba_gemm_backends: BF16_GEMM_BACKENDS.to_vec(),
            static_quant_backends: INDUCTOR_BACKENDS.to_vec(),
            triton_elementwise_backends: TRITON_ELEMENTWISE_BACKENDS.to_vec(),
            torch_elementwise_backends: TORCH_ELEMENTWISE_BACKENDS.to_vec(),
            strided_copy_backends: STRIDED_COPY_BACKENDS.to_vec(),
            cat_backends: CAT_BACKENDS.to_vec(),
            state_copy_backends: TORCH_ELEMENTWISE_BACKENDS.to_vec(),
            causal_conv_prefill_backends: VLLM_TRITON_BACKENDS.to_vec(),
            prefill_post_conv_backends: VLLM_TRITON_BACKENDS.to_vec(),
            chunk_delta_rule_backends: CHUNK_DELTA_RULE_BACKENDS.to_vec(),
            causal_conv_decode_backends: VLLM_TRITON_BACKENDS.to_vec(),
            packed_decode_backends: VLLM_TRITON_BACKENDS.to_vec(),
            sigmoid_gating_decode_backends: SIGMOID_GATING_DECODE_BACKENDS.to_vec(),
            gated_rms_norm_backends: VLLM_TRITON_BACKENDS.to_vec(),
        },
        gqa: VllmQwen35GatedGqaLocalWorkletConfig {
            hidden: model.hidden,
            num_q_heads: model.gqa_heads,
            num_kv_heads: model.gqa_kv_heads,
            head_dim: model.gqa_head_dim,
            rotary_dim: model.rotary_dim,
            mrope: model.mrope,
            kv_block_size: KV_BLOCK_SIZE,
            activation_dtype: ACTIVATION_DTYPE,
            kv_dtype: KV_DTYPE,
            proj_dtype: p.gqa_qkv_proj.gemm_dtype(),
            gpu_name: gpu.clone(),
            proj_gemm_backends: FP8_GEMM_BACKENDS.to_vec(),
            qk_norm_rope_backends: VLLM_TRITON_BACKENDS.to_vec(),
            static_quant_backends: INDUCTOR_BACKENDS.to_vec(),
            kv_append_backends: KV_APPEND_BACKENDS.to_vec(),
            attention_backends: ATTENTION_BACKENDS.to_vec(),
        },
        // The MLP's NVFP4 quant reads the BF16 normed row.
        post_attention_norm: norm(&gpu, model.hidden, 2, false, true, false),
        mlp: VllmNvfp4SwigluMlpLocalWorkletConfig {
            hidden: model.hidden,
            intermediate: model.intermediate,
            activation_dtype: ACTIVATION_DTYPE,
            gpu_name: gpu.clone(),
            quant_backends: NVFP4_QUANT_BACKENDS.to_vec(),
            gemm_backends: NVFP4_GEMM_BACKENDS.to_vec(),
            activation_backends: INDUCTOR_BACKENDS.to_vec(),
        },
        final_norm: norm(&gpu, model.hidden, 3, false, true, false),
        // One hidden row and its int64 index in, the row out.
        logits_select: elementwise(
            TORCH_ELEMENTWISE_BACKENDS,
            &gpu,
            hidden_bytes + INT64_INDEX_BYTES,
            hidden_bytes,
        ),
        lm_head_quant: Nvfp4QuantKernelConfig {
            backends: NVFP4_QUANT_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            hidden_size: model.hidden.into(),
            group_size: NVFP4_GROUP_SIZE,
            input_dtype: ACTIVATION_DTYPE,
            scale_format: NVFP4_SWIZZLED_SCALE_FORMAT.to_string(),
        },
        lm_head: SingleGemmKernelConfig {
            backends: NVFP4_GEMM_BACKENDS.to_vec(),
            gpu_name: gpu,
            n: model.vocab_size.into(),
            k: model.hidden.into(),
            dtype: p.lm_head.gemm_dtype(),
        },
        model: model.clone(),
        parallel: parallel.clone(),
    })
}

fn norm(
    gpu: &str,
    hidden: u32,
    num_addends: u32,
    store_sum: bool,
    store_normed: bool,
    store_fp8: bool,
) -> MultiAddRmsNormKernelConfig {
    MultiAddRmsNormKernelConfig {
        backends: NORM_BACKENDS.to_vec(),
        gpu_name: gpu.to_string(),
        hidden_dim: hidden.into(),
        num_addends,
        store_sum,
        store_normed,
        store_fp8,
        dtype: ACTIVATION_DTYPE,
    }
}

#[derive(Clone, Debug)]
pub struct Qwen35VllmResolved {
    pub raw_cfg: Qwen35VllmConfigs,
    pub gdn: VllmQwen35GdnLocalWorkletResolved,
    pub gqa: VllmQwen35GatedGqaLocalWorkletResolved,
    pub mlp: VllmNvfp4SwigluMlpLocalWorkletResolved,
}

pub fn resolve_configs(cfgs: &Qwen35VllmConfigs) -> Qwen35VllmResolved {
    Qwen35VllmResolved {
        gdn: VllmQwen35GdnLocalWorklet::resolve_config(&cfgs.gdn),
        gqa: VllmQwen35GatedGqaLocalWorklet::resolve_config(&cfgs.gqa),
        mlp: VllmNvfp4SwigluMlpLocalWorklet::resolve_config(&cfgs.mlp),
        raw_cfg: cfgs.clone(),
    }
}

enum Mixer {
    Gdn(VllmQwen35GdnLocalWorklet),
    Gqa(VllmQwen35GatedGqaLocalWorklet),
}

/// One built layer group: the leaves of a single layer of its shape.
struct LayerGroup {
    label: String,
    layers: Vec<u32>,
    input_norm: Op<MultiAddRmsNormKernel>,
    mixer: Mixer,
    post_attention_norm: Op<MultiAddRmsNormKernel>,
    mlp: VllmNvfp4SwigluMlpLocalWorklet,
}

impl LayerGroup {
    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let layer = CostNode::Sum(vec![
            self.input_norm.compile(builder),
            match &self.mixer {
                Mixer::Gdn(gdn) => gdn.compile(builder),
                Mixer::Gqa(gqa) => gqa.compile(builder),
            },
            self.post_attention_norm.compile(builder),
            self.mlp.compile(builder),
        ]);
        CostNode::Labeled {
            label: format!("{} layers {:?}", self.label, self.layers),
            child: Box::new(CostNode::Scale {
                n: self.layers.len() as u32,
                child: Box::new(layer),
            }),
        }
    }

    fn eval(&self, step: &Step, ev: &mut Evaluator) {
        let norm = MultiAddRmsNormKernelInput {
            num_tokens: step.padded_tokens,
        };
        push_or_zero(&self.input_norm, norm.clone(), false, ev);
        match &self.mixer {
            Mixer::Gdn(gdn) => gdn.eval(
                &VllmQwen35GdnLocalWorkletInput {
                    padded_tokens: step.padded_tokens,
                    full_graph_rows: step.full_graph_rows,
                    prefill_lengths: step.prefills.iter().map(|&(_, new)| new).collect(),
                    decode_requests: step.decode_kv_lens.len() as u32,
                },
                ev,
            ),
            Mixer::Gqa(gqa) => gqa.eval(
                &VllmQwen35GatedGqaLocalWorkletInput {
                    padded_tokens: step.padded_tokens,
                    full_graph_rows: step.full_graph_rows,
                    prefill_requests: step.prefills.clone(),
                    decode_kv_lens: step.decode_kv_lens.clone(),
                },
                ev,
            ),
        }
        push_or_zero(&self.post_attention_norm, norm, false, ev);
        self.mlp.eval(
            &VllmNvfp4SwigluMlpLocalWorkletInput {
                num_tokens: step.padded_tokens,
            },
            ev,
        );
    }
}

fn build_layer_group(
    name: &str,
    group: &Qwen35LayerGroup,
    resolved: &Qwen35VllmResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<LayerGroup, BuildError> {
    let cfg = &resolved.raw_cfg;
    let prefix = format!("{name}.{}", group.label);
    let p = prefix.as_str();
    let input_norm = if group.first {
        cfg.first_norm.clone()
    } else {
        match group.mixer {
            Qwen35Mixer::Gdn => cfg.gdn_input_norm.clone(),
            Qwen35Mixer::Gqa => cfg.gqa_input_norm.clone(),
        }
    };
    Ok(LayerGroup {
        label: group.label.clone(),
        layers: group.layers.clone(),
        input_norm: atomic(
            p,
            "input_norm",
            input_norm,
            MultiAddRmsNormKernel::build,
            bridge,
        )?,
        mixer: match group.mixer {
            Qwen35Mixer::Gdn => Mixer::Gdn(VllmQwen35GdnLocalWorklet::build(
                format!("{p}.gdn"),
                resolved.gdn.clone(),
                bridge,
            )?),
            Qwen35Mixer::Gqa => Mixer::Gqa(VllmQwen35GatedGqaLocalWorklet::build(
                format!("{p}.gqa"),
                resolved.gqa.clone(),
                bridge,
            )?),
        },
        post_attention_norm: atomic(
            p,
            "post_attention_norm",
            cfg.post_attention_norm.clone(),
            MultiAddRmsNormKernel::build,
            bridge,
        )?,
        mlp: VllmNvfp4SwigluMlpLocalWorklet::build(
            format!("{p}.mlp"),
            resolved.mlp.clone(),
            bridge,
        )?,
    })
}

pub struct Qwen35VllmModel {
    pub name: String,
    pub max_model_len: u32,
    /// Sorted, deduplicated capture sizes.
    cudagraph_capture_sizes: Vec<u32>,
    groups: Vec<LayerGroup>,
    final_norm: Op<MultiAddRmsNormKernel>,
    logits_select: Op<ElementwiseKernel>,
    lm_head_quant: Op<Nvfp4QuantKernel>,
    lm_head: Op<SingleGemmKernel>,
    kv_cache_layout: KvCacheLayout,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: Qwen35VllmResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<Qwen35VllmModel, BuildError> {
    let cfg = &resolved.raw_cfg;
    let n = name.as_str();
    let groups = cfg
        .groups
        .iter()
        .map(|group| build_layer_group(n, group, &resolved, bridge))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut model = Qwen35VllmModel {
        groups,
        final_norm: atomic(
            n,
            "final_norm",
            cfg.final_norm.clone(),
            MultiAddRmsNormKernel::build,
            bridge,
        )?,
        logits_select: atomic(
            n,
            "logits_select",
            cfg.logits_select.clone(),
            ElementwiseKernel::build,
            bridge,
        )?,
        lm_head_quant: atomic(
            n,
            "lm_head_input_quant",
            cfg.lm_head_quant.clone(),
            Nvfp4QuantKernel::build,
            bridge,
        )?,
        lm_head: atomic(
            n,
            "lm_head",
            cfg.lm_head.clone(),
            SingleGemmKernel::build,
            bridge,
        )?,
        kv_cache_layout: kv_cache_layout(&cfg.model),
        max_model_len: cfg.parallel.max_model_len,
        cudagraph_capture_sizes: {
            let mut sizes = cfg.parallel.cudagraph_capture_sizes.clone();
            sizes.sort_unstable();
            sizes.dedup();
            sizes
        },
        cost_flat: Vec::new(),
        n_slots: 0,
        name,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

/// One request's cache in vLLM's hybrid allocator: FP8 K and V of every GQA
/// layer per token, and per GDN layer the FP32 SSM state plus the BF16
/// `conv_kernel - 1`-tap conv window, padded to the attention page of one
/// hybrid block. For Qwen3.8-27B: 2,048 B per token per GQA layer and a
/// 3,207,168 B mamba page give the logged 1568-token block and its 0.13%
/// padding.
fn kv_cache_layout(model: &Qwen35ModelCfg) -> KvCacheLayout {
    let attn_page_per_token = 2
        * u64::from(model.gqa_kv_heads)
        * u64::from(model.gqa_head_dim)
        * u64::from(KV_DTYPE.size_bytes());
    let ssm_state = u64::from(model.gdn_value_heads)
        * u64::from(model.gdn_key_head_dim)
        * u64::from(model.gdn_value_head_dim)
        * u64::from(model.gdn_state_dtype.size_bytes());
    let conv_state = u64::from(model.gdn_conv_channels())
        * u64::from(model.gdn_conv_kernel - 1)
        * u64::from(ACTIVATION_DTYPE.size_bytes());
    let count = |mixer| model.mixers.iter().filter(|&&m| m == mixer).count() as u64;
    hybrid_attention_state_layout(
        u64::from(KV_BLOCK_SIZE),
        attn_page_per_token,
        count(Qwen35Mixer::Gqa),
        ssm_state + conv_state,
        count(Qwen35Mixer::Gdn),
    )
}

impl Qwen35VllmModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let mut children: Vec<CostNode> = self
            .groups
            .iter()
            .map(|group| group.compile(&mut builder))
            .collect();
        children.push(self.final_norm.compile(&mut builder));
        children.push(self.logits_select.compile(&mut builder));
        children.push(self.lm_head_quant.compile(&mut builder));
        children.push(self.lm_head.compile(&mut builder));
        let root = CostNode::Labeled {
            label: format!(
                "{} (Qwen35VllmModel) [local (1 GPU); timing_context<={}]",
                self.name, self.max_model_len
            ),
            child: Box::new(CostNode::Sum(children)),
        };
        builder.finish(root)
    }

    fn eval_into(&self, input: &UnifiedArchInput, ev: &mut Evaluator) {
        let step = normalize_input(input, self.max_model_len, &self.cudagraph_capture_sizes)
            .unwrap_or_else(|reason| panic!("invalid Qwen35VllmModel input: {reason}"));
        for group in &self.groups {
            group.eval(&step, ev);
        }
        push_or_zero(
            &self.final_norm,
            MultiAddRmsNormKernelInput {
                num_tokens: step.padded_tokens,
            },
            false,
            ev,
        );
        let logits = step.request_count;
        push_or_zero(
            &self.logits_select,
            ElementwiseKernelInput { num_tokens: logits },
            false,
            ev,
        );
        push_or_zero(
            &self.lm_head_quant,
            Nvfp4QuantKernelInput { num_tokens: logits },
            false,
            ev,
        );
        push_or_zero(
            &self.lm_head,
            SingleGemmKernelInput { m: logits },
            false,
            ev,
        );
    }

    fn eval_checked(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: Option<&mut Vec<SlotInput>>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut evaluator = match inputs {
            Some(inputs) => Evaluator::with_inputs(slots, inputs),
            None => Evaluator::new(slots),
        };
        self.eval_into(batch, &mut evaluator);
        assert_eq!(
            evaluator.filled(),
            self.n_slots,
            "eval must fill every compiled slot"
        );
        drop(evaluator);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }
}

impl IterwiseUnifiedModel for Qwen35VllmModel {
    fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
        normalize_input(batch, self.max_model_len, &self.cudagraph_capture_sizes).map(|_| ())
    }

    fn kv_cache_layout(&self) -> KvCacheLayout {
        self.kv_cache_layout.clone()
    }

    fn gpus_per_replica(&self) -> u16 {
        1
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
        self.eval_checked(batch, slots, scratch, None)
    }

    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
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

/// One iteration as every layer sees it.
struct Step {
    /// The rows the compiled pieces run: the token count padded to the next
    /// captured size, or the real count above the largest.
    padded_tokens: u32,
    /// A uniform decode replays a FULL graph, whose GDN core and decode
    /// attention run on the padded rows; 0 otherwise.
    full_graph_rows: u32,
    request_count: u32,
    /// `(cached prefix, new tokens)` of each prefill request.
    prefills: Vec<(u32, u32)>,
    decode_kv_lens: Vec<u32>,
}

fn normalize_input(
    input: &UnifiedArchInput,
    max_model_len: u32,
    sorted_capture_sizes: &[u32],
) -> std::result::Result<Step, String> {
    let [group] = input.groups.as_slice() else {
        return Err(format!(
            "expected exactly one attention group, got {}",
            input.groups.len()
        ));
    };
    let mut prefill_tokens = 0_u32;
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
    }
    if group.prefill_tokens != prefill_tokens {
        return Err(format!(
            "prefill_tokens {} must equal append sum {prefill_tokens}",
            group.prefill_tokens
        ));
    }
    for (request, &context) in group.decode_kv_lens.iter().enumerate() {
        if !(1..=max_model_len).contains(&context) {
            return Err(format!(
                "decode request {request} context {context} must be in 1..={max_model_len}"
            ));
        }
    }
    let decode_tokens = group.decode_kv_lens.len() as u32;
    if group.decode_tokens != decode_tokens {
        return Err(format!(
            "decode_tokens {} must equal decode_kv_lens length {decode_tokens}",
            group.decode_tokens
        ));
    }
    let total_tokens = prefill_tokens + decode_tokens;
    if group.batch_tokens != total_tokens {
        return Err(format!(
            "batch_tokens {} must equal prefill+decode {total_tokens}",
            group.batch_tokens
        ));
    }
    if total_tokens == 0 {
        return Err("an iteration must carry at least one token".into());
    }
    let padded_tokens = graph_padded_tokens(sorted_capture_sizes, total_tokens);
    Ok(Step {
        padded_tokens,
        full_graph_rows: if group.prefill_chunk_pairs.is_empty() {
            padded_tokens
        } else {
            0
        },
        request_count: group.request_count(),
        prefills: group.prefill_chunk_pairs.clone(),
        decode_kv_lens: group.decode_kv_lens.clone(),
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
    use crate::arch::contract::ArchGroupInput;
    use std::path::Path;

    fn model_cfg() -> Qwen35ModelCfg {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/qwen3_8_27b_nvfp4.json");
        Qwen35ModelCfg::from_json(&path).unwrap()
    }

    /// The capture's FULL_AND_PIECEWISE sizes: 1, 2, 4, 8..256 by 8, then
    /// 272..2048 by 16.
    fn capture_sizes() -> Vec<u32> {
        let mut sizes = vec![1, 2, 4];
        sizes.extend((8..=256).step_by(8));
        sizes.extend((272..=2048).step_by(16));
        sizes
    }

    fn parallel() -> Qwen35VllmParallel {
        Qwen35VllmParallel {
            max_model_len: 131072,
            gpu_name: "NVIDIA B200".into(),
            cudagraph_capture_sizes: capture_sizes(),
        }
    }

    fn built() -> Qwen35VllmModel {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let configs = build_configs(&model_cfg(), &parallel()).unwrap();
        build("unified".into(), resolve_configs(&configs), &bridge).unwrap()
    }

    fn step(prefill_chunk_pairs: Vec<(u32, u32)>, decode_kv_lens: Vec<u32>) -> UnifiedArchInput {
        let prefill_tokens = prefill_chunk_pairs
            .iter()
            .map(|&(_, append)| append)
            .sum::<u32>();
        let decode_tokens = decode_kv_lens.len() as u32;
        UnifiedArchInput {
            groups: vec![ArchGroupInput {
                batch_tokens: prefill_tokens + decode_tokens,
                prefill_tokens,
                decode_tokens,
                prefill_chunk_pairs,
                decode_kv_lens,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn layers_fold_into_the_first_gdn_layer_and_two_repeated_groups() {
        let summary: Vec<(String, usize)> = layer_groups(&model_cfg())
            .into_iter()
            .map(|g| (g.label, g.layers.len()))
            .collect();
        assert_eq!(
            summary,
            [
                ("gdn_first".to_string(), 1),
                ("gdn".to_string(), 47),
                ("gqa".to_string(), 16)
            ]
        );
    }

    /// The five norm variants the capture runs (profiled as such).
    #[test]
    fn norms_store_what_their_readers_read() {
        let c = build_configs(&model_cfg(), &parallel()).unwrap();
        let shape = |n: &MultiAddRmsNormKernelConfig| {
            (n.num_addends, n.store_sum, n.store_normed, n.store_fp8)
        };
        assert_eq!(shape(&c.first_norm), (1, false, true, true));
        assert_eq!(shape(&c.gdn_input_norm), (3, true, true, true));
        assert_eq!(shape(&c.gqa_input_norm), (3, true, false, true));
        assert_eq!(shape(&c.post_attention_norm), (2, false, true, false));
        assert_eq!(shape(&c.final_norm), (3, false, true, false));
        assert_eq!((c.lm_head.n.get(), c.lm_head.k.get()), (248320, 5120));
        assert_eq!(c.lm_head.dtype, DType::Nvfp4E2m1);
    }

    #[test]
    fn an_unmodeled_precision_mix_is_refused() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/qwen3_6_27b.json");
        let bf16 = Qwen35ModelCfg::from_json(&path).unwrap();
        assert!(build_configs(&bf16, &parallel()).is_err());
    }

    #[test]
    fn padding_follows_the_graph_mode() {
        let sizes = {
            let mut sizes = capture_sizes();
            sizes.sort_unstable();
            sizes
        };
        let shape = |input: UnifiedArchInput| {
            let step = normalize_input(&input, 131072, &sizes).unwrap();
            (step.padded_tokens, step.full_graph_rows)
        };
        assert_eq!(shape(step(vec![], vec![400; 5])), (8, 8));
        assert_eq!(
            shape(step(vec![(0, 629), (87808, 805)], vec![9; 4])),
            (1440, 0)
        );
        // Past the largest captured size a step runs eagerly, unpadded.
        assert_eq!(shape(step(vec![], vec![400; 2049])), (2049, 2049));
        assert_eq!(shape(step(vec![(0, 4096)], vec![])), (4096, 0));
        assert!(normalize_input(&step(vec![], vec![131073]), 131072, &sizes).is_err());
    }

    #[test]
    fn the_manifest_folds_three_layer_groups_and_the_head() {
        let model = built();
        for input in [
            step(vec![], vec![400; 5]),
            step(vec![(0, 629), (87808, 805)], vec![9; 4]),
            step(vec![(0, 8192)], vec![]),
        ] {
            model.check_input(&input).unwrap();
        }
        let names: Vec<String> = model
            .cost_log_manifest()
            .slots
            .into_iter()
            .map(|slot| slot.name)
            .collect();
        // Per group: 2 norms, the mixer (GDN 23, GQA 8) and the MLP (5).
        assert_eq!(names.len(), 2 * (2 + 23 + 5) + (2 + 8 + 5) + 4);
        for name in [
            "unified.gdn_first.input_norm",
            "unified.gdn.gdn.mixed_decode_recurrent",
            "unified.gqa.gqa.decode_attention",
            "unified.gqa.mlp.silu_and_mul",
            "unified.lm_head",
        ] {
            assert!(names.iter().any(|n| n == name), "{name}");
        }
    }

    #[test]
    fn kv_cache_layout_reproduces_the_logged_hybrid_block() {
        let layout = kv_cache_layout(&model_cfg());
        assert_eq!(layout.per_token_bytes(), 16 * 2048);
        assert_eq!(layout.engine.block_tokens, Some(1568));
        let gdn = layout.parts.iter().find(|part| part.name == "gdn").unwrap();
        assert_eq!(gdn.bytes_per_entry, 48 * 3_207_168);
        assert_eq!(gdn.overhead.padded_bytes_per_entry, Some(48 * 1568 * 2048));
        // "Padding mamba page size by 0.13%".
        let padding = 100.0 * (f64::from(1568 * 2048) / 3_207_168.0 - 1.0);
        assert!((padding - 0.13).abs() < 0.005, "{padding}");
    }

    /// "Maximum concurrency for 131,072 tokens per request: 29.14x" from the
    /// capture's 125.52 GiB pool at chunk 2048, prefix caching on (align mode
    /// holds one more state block per request).
    #[test]
    fn kv_cache_layout_reproduces_vllms_logged_concurrency() {
        use crate::worker::kv::{KvAccounting, KvAccountingOptions};
        let layout = kv_cache_layout(&model_cfg());
        let options = KvAccountingOptions {
            max_in_flight_tokens: Some(2 * 2048),
            prefix_caching: true,
            ..Default::default()
        };
        let budget = (125.52 * f64::from(1u32 << 30)) as u64;
        let accounting = KvAccounting::resolve(&layout, options, budget).unwrap();
        let concurrency = accounting.concurrency_at(131072);
        assert!(
            (concurrency / 29.14 - 1.0).abs() < 0.005,
            "{concurrency} vs 29.14"
        );
    }
}
