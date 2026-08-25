//! DeepSeek-V4-Flash-0731 in vLLM's measured kernel granularity.
//!
//! The graph keeps seven unique layer bodies and folds the final nineteen
//! C128/C4 pairs. Attention uses the busiest DP rank, while the two EP
//! collectives retain the exact ragged four-rank token vector. Routed expert
//! compute uses one contiguous, highest-mass expert shard; popularity never
//! changes collective bytes.

use std::path::Path;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

use crate::arch::config::ModelSpec;
use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
use crate::op::Op;
use crate::timing::kernels::{
    DeepseekV4TerminalMhcHeadKernel, ElementwiseKernel, ElementwiseKernelConfig,
    ElementwiseKernelInput, MhcRmsNormKernelConfig, MhcRmsNormKernelInput, MoeEpAllGatherKernel,
    MoeEpAllGatherKernelConfig, MoeEpCollectiveKernelInput, MoeEpReduceScatterKernel,
    MoeEpReduceScatterKernelConfig, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput,
};
use crate::timing::routing::RoutingDistribution;
use crate::timing::{
    BuildError, CostManifest, CostNode, CostTree, CostTreeBuilder, DType, Dim, Evaluator,
    FlatCostNode, LeafMetrics, PerfApiBridge, Probe, SlotInput,
};
use crate::worklet::{
    DeepseekV4AttentionEntry, DeepseekV4AttentionLocalWorklet,
    DeepseekV4AttentionLocalWorkletConfig, DeepseekV4AttentionLocalWorkletInput,
    DeepseekV4AttentionLocalWorkletResolved, DeepseekV4MoeExpertComputeLocalWorklet,
    DeepseekV4MoeExpertComputeLocalWorkletConfig, DeepseekV4MoeExpertComputeLocalWorkletInput,
    DeepseekV4MoeExpertComputeLocalWorkletResolved, DeepseekV4MoeRouterLocalWorklet,
    DeepseekV4MoeRouterLocalWorkletConfig, DeepseekV4MoeRouterLocalWorkletInput,
    DeepseekV4MoeRouterLocalWorkletResolved, DeepseekV4SharedExpertLocalWorklet,
    DeepseekV4SharedExpertLocalWorkletConfig, DeepseekV4SharedExpertLocalWorkletInput,
    DeepseekV4SharedExpertLocalWorkletResolved,
};

const ARCH_KIND: &str = "deepseek_v4_vllm";
const CHECKPOINT_LAYERS: u32 = 43;
const NUM_HASH_LAYERS: u32 = 3;
const EP_SIZE: u32 = 4;
const HIDDEN: u32 = 4096;
const NUM_EXPERTS: u32 = 256;
const TOP_K: u32 = 6;
const EXPERT_WIDTH: u32 = 2048;
const VOCAB_SIZE: u32 = 129_280;
const MAX_BATCHED_TOKENS: u32 = 8192;
const CYCLE_COUNT: u32 = 19;

const MHC_BACKENDS: &[&str] = &["vllm_tilelang"];
const QUANT_BACKENDS: &[&str] = &["vllm_cuda"];
const GEMM_BACKENDS: &[&str] = &["deepgemm"];
const FP32_GEMM_BACKENDS: &[&str] = &["torch_cublas"];
const CLAMPED_SWIGLU_BACKENDS: &[&str] = &["vllm_inductor"];
const COLLECTIVE_BACKENDS: &[&str] = &["vllm_pynccl"];
const MARLIN_BACKENDS: &[&str] = &["vllm_marlin"];

#[derive(Clone, Debug)]
pub struct DeepseekV4ModelCfg {
    pub num_layers: u32,
    pub hidden_size: Dim,
    pub hc_mult: u32,
    pub num_attention_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub rope_dim: Dim,
    pub index_num_heads: Dim,
    pub index_head_dim: Dim,
    pub selected_k: u32,
    pub q_lora_rank: Dim,
    pub o_lora_rank: Dim,
    pub o_groups: Dim,
    pub num_experts: Dim,
    pub top_k: u32,
    pub expert_intermediate: Dim,
    pub num_shared_experts: u32,
    pub vocab_size: Dim,
    pub max_model_len: u32,
    pub compress_ratios: Vec<u32>,
}

#[derive(Deserialize)]
struct JsonDeepseekV4Config {
    architectures: Vec<String>,
    model_type: String,
    torch_dtype: String,
    hidden_size: u32,
    hc_mult: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    qk_rope_head_dim: u32,
    index_n_heads: u32,
    index_head_dim: u32,
    index_topk: u32,
    q_lora_rank: u32,
    o_lora_rank: u32,
    o_groups: u32,
    n_routed_experts: u32,
    num_experts_per_tok: u32,
    moe_intermediate_size: u32,
    n_shared_experts: u32,
    vocab_size: u32,
    num_hash_layers: u32,
    max_position_embeddings: u32,
    compress_ratios: Vec<u32>,
    expert_dtype: String,
    scoring_func: String,
    topk_method: String,
    norm_topk_prob: bool,
}

impl DeepseekV4ModelCfg {
    pub fn from_json(path: &Path, spec: &ModelSpec) -> Result<Self> {
        ensure!(spec.fp8, "{ARCH_KIND} requires fp8=true");
        ensure!(
            spec.num_layers.is_none() && spec.sim_num_layers.is_none(),
            "{ARCH_KIND} requires the exact heterogeneous 43-layer schedule"
        );
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading DeepSeek V4 config {}", path.display()))?;
        let raw: JsonDeepseekV4Config = serde_json::from_str(&text).context("parsing JSON")?;
        ensure!(raw.architectures == ["DeepseekV4ForCausalLM"]);
        ensure!(raw.model_type == "deepseek_v4");
        ensure!(raw.torch_dtype == "bfloat16");
        ensure!(raw.expert_dtype == "fp4");
        ensure!(raw.scoring_func == "sqrtsoftplus");
        ensure!(raw.topk_method == "noaux_tc" && raw.norm_topk_prob);
        for (name, actual, expected) in [
            ("hidden_size", raw.hidden_size, HIDDEN),
            ("hc_mult", raw.hc_mult, 4),
            (
                "num_hidden_layers",
                raw.num_hidden_layers,
                CHECKPOINT_LAYERS,
            ),
            ("num_attention_heads", raw.num_attention_heads, 64),
            ("num_key_value_heads", raw.num_key_value_heads, 1),
            ("head_dim", raw.head_dim, 512),
            ("qk_rope_head_dim", raw.qk_rope_head_dim, 64),
            ("index_n_heads", raw.index_n_heads, 64),
            ("index_head_dim", raw.index_head_dim, 128),
            ("index_topk", raw.index_topk, 512),
            ("q_lora_rank", raw.q_lora_rank, 1024),
            ("o_lora_rank", raw.o_lora_rank, 1024),
            ("o_groups", raw.o_groups, 8),
            ("n_routed_experts", raw.n_routed_experts, NUM_EXPERTS),
            ("num_experts_per_tok", raw.num_experts_per_tok, TOP_K),
            (
                "moe_intermediate_size",
                raw.moe_intermediate_size,
                EXPERT_WIDTH,
            ),
            ("n_shared_experts", raw.n_shared_experts, 1),
            ("vocab_size", raw.vocab_size, VOCAB_SIZE),
            ("num_hash_layers", raw.num_hash_layers, NUM_HASH_LAYERS),
            (
                "max_position_embeddings",
                raw.max_position_embeddings,
                1_048_576,
            ),
        ] {
            ensure!(
                actual == expected,
                "{name} must be {expected}, got {actual}"
            );
        }
        ensure!(
            raw.compress_ratios.len() == CHECKPOINT_LAYERS as usize + 3
                && raw.compress_ratios[CHECKPOINT_LAYERS as usize..] == [0, 0, 0]
        );
        let normalized = raw.compress_ratios[..CHECKPOINT_LAYERS as usize]
            .iter()
            .map(|&ratio| if ratio == 0 { 1 } else { ratio })
            .collect::<Vec<_>>();
        ensure!(normalized[..5] == [1, 1, 4, 128, 4]);
        ensure!(normalized[5..]
            .iter()
            .enumerate()
            .all(|(index, &ratio)| ratio == if index % 2 == 0 { 128 } else { 4 }));
        Ok(Self {
            num_layers: raw.num_hidden_layers,
            hidden_size: raw.hidden_size.into(),
            hc_mult: raw.hc_mult,
            num_attention_heads: raw.num_attention_heads.into(),
            num_kv_heads: raw.num_key_value_heads.into(),
            head_dim: raw.head_dim.into(),
            rope_dim: raw.qk_rope_head_dim.into(),
            index_num_heads: raw.index_n_heads.into(),
            index_head_dim: raw.index_head_dim.into(),
            selected_k: raw.index_topk,
            q_lora_rank: raw.q_lora_rank.into(),
            o_lora_rank: raw.o_lora_rank.into(),
            o_groups: raw.o_groups.into(),
            num_experts: raw.n_routed_experts.into(),
            top_k: raw.num_experts_per_tok,
            expert_intermediate: raw.moe_intermediate_size.into(),
            num_shared_experts: raw.n_shared_experts,
            vocab_size: raw.vocab_size.into(),
            max_model_len: raw.max_position_embeddings,
            compress_ratios: normalized,
        })
    }
}

#[derive(Clone, Debug)]
pub struct DeepseekV4VllmParallel {
    pub ep_size: u16,
    pub nvl_num_gpu: u16,
    pub gpu_name: String,
    pub serialize_streams: bool,
}

#[derive(Clone, Copy)]
struct LayerRecipe {
    name: &'static str,
    ratio: u32,
    planner: &'static str,
    entry: DeepseekV4AttentionEntry,
    selection: &'static str,
}

const LAYER_RECIPES: [LayerRecipe; 7] = [
    LayerRecipe {
        name: "layer0_c1_planned_hash",
        ratio: 1,
        planner: "planned",
        entry: DeepseekV4AttentionEntry::Layer0Pre,
        selection: "hash",
    },
    LayerRecipe {
        name: "layer1_c1_planned_hash",
        ratio: 1,
        planner: "planned",
        entry: DeepseekV4AttentionEntry::LaterLayerFusedPostPre,
        selection: "hash",
    },
    LayerRecipe {
        name: "layer2_c4_planned_hash",
        ratio: 4,
        planner: "planned",
        entry: DeepseekV4AttentionEntry::LaterLayerFusedPostPre,
        selection: "hash",
    },
    LayerRecipe {
        name: "layer3_c128_planned_learned",
        ratio: 128,
        planner: "planned",
        entry: DeepseekV4AttentionEntry::LaterLayerFusedPostPre,
        selection: "learned",
    },
    LayerRecipe {
        name: "layer4_c4_reused_learned",
        ratio: 4,
        planner: "reused",
        entry: DeepseekV4AttentionEntry::LaterLayerFusedPostPre,
        selection: "learned",
    },
    LayerRecipe {
        name: "cycle_c128_reused_learned",
        ratio: 128,
        planner: "reused",
        entry: DeepseekV4AttentionEntry::LaterLayerFusedPostPre,
        selection: "learned",
    },
    LayerRecipe {
        name: "cycle_c4_reused_learned",
        ratio: 4,
        planner: "reused",
        entry: DeepseekV4AttentionEntry::LaterLayerFusedPostPre,
        selection: "learned",
    },
];

#[derive(Clone)]
pub struct DeepseekV4LayerConfig {
    attention: DeepseekV4AttentionLocalWorkletConfig,
    router: DeepseekV4MoeRouterLocalWorkletConfig,
    dispatch: MoeEpAllGatherKernelConfig,
    shared: DeepseekV4SharedExpertLocalWorkletConfig,
    expert: DeepseekV4MoeExpertComputeLocalWorkletConfig,
    combine: MoeEpReduceScatterKernelConfig,
    finalize: ElementwiseKernelConfig,
}

pub struct DeepseekV4VllmConfigs {
    layers: Vec<DeepseekV4LayerConfig>,
    lm_head: SingleGemmKernelConfig,
    terminal_mhc_head: MhcRmsNormKernelConfig,
    total_kv_bytes_per_token: u64,
}

pub struct DeepseekV4LayerResolved {
    attention: DeepseekV4AttentionLocalWorkletResolved,
    router: DeepseekV4MoeRouterLocalWorkletResolved,
    dispatch: MoeEpAllGatherKernelConfig,
    shared: DeepseekV4SharedExpertLocalWorkletResolved,
    expert: DeepseekV4MoeExpertComputeLocalWorkletResolved,
    combine: MoeEpReduceScatterKernelConfig,
    finalize: ElementwiseKernelConfig,
}

pub struct DeepseekV4VllmResolved {
    layers: Vec<DeepseekV4LayerResolved>,
    lm_head: SingleGemmKernelConfig,
    terminal_mhc_head: MhcRmsNormKernelConfig,
    total_kv_bytes_per_token: u64,
}

fn fit_failed(reason: impl Into<String>) -> BuildError {
    BuildError::FitFailed {
        kind: ARCH_KIND,
        reason: reason.into(),
    }
}

pub fn build_configs(
    model: &DeepseekV4ModelCfg,
    parallel: &DeepseekV4VllmParallel,
    routing: &RoutingDistribution,
) -> std::result::Result<DeepseekV4VllmConfigs, BuildError> {
    if parallel.ep_size != EP_SIZE as u16
        || parallel.nvl_num_gpu != EP_SIZE as u16
        || parallel.gpu_name != "NVIDIA H200"
    {
        return Err(fit_failed(
            "DeepSeek V4 Flash requires H200 EP4 within one NVLink domain",
        ));
    }
    if routing.num_experts() != NUM_EXPERTS {
        return Err(fit_failed("routing distribution must contain 256 experts"));
    }
    let local_ppm = critical_contiguous_shard(routing.ppm(), EP_SIZE as usize);
    let layers = LAYER_RECIPES
        .iter()
        .map(|recipe| layer_config(model, parallel, recipe, local_ppm.clone()))
        .collect();
    let total_kv_bytes_per_token = model
        .compress_ratios
        .iter()
        .try_fold(0_u64, |sum, ratio| {
            sum.checked_add(
                584 + match ratio {
                    1 => 0,
                    4 => 146,
                    128 => 5,
                    _ => unreachable!(),
                },
            )
        })
        .ok_or_else(|| fit_failed("KV byte accounting overflow"))?;
    Ok(DeepseekV4VllmConfigs {
        layers,
        lm_head: SingleGemmKernelConfig {
            backends: vec!["torch_linear"],
            gpu_name: parallel.gpu_name.clone(),
            n: model.vocab_size.clone(),
            k: model.hidden_size.clone(),
            dtype: DType::Bf16,
        },
        terminal_mhc_head: MhcRmsNormKernelConfig {
            backends: MHC_BACKENDS.to_vec(),
            gpu_name: parallel.gpu_name.clone(),
            hidden_size: model.hidden_size.clone(),
            hc_mult: model.hc_mult,
            hidden_dtype: DType::Bf16,
        },
        total_kv_bytes_per_token,
    })
}

fn critical_contiguous_shard(ppm: &[u32], partitions: usize) -> Vec<u32> {
    assert_eq!(ppm.len() % partitions, 0);
    ppm.chunks_exact(ppm.len() / partitions)
        .max_by_key(|shard| shard.iter().map(|&mass| u64::from(mass)).sum::<u64>())
        .unwrap()
        .to_vec()
}

fn layer_config(
    model: &DeepseekV4ModelCfg,
    parallel: &DeepseekV4VllmParallel,
    recipe: &LayerRecipe,
    local_ppm: Vec<u32>,
) -> DeepseekV4LayerConfig {
    let gpu = parallel.gpu_name.clone();
    DeepseekV4LayerConfig {
        attention: DeepseekV4AttentionLocalWorkletConfig {
            entry: recipe.entry,
            compress_ratio: recipe.ratio,
            planner_mode: recipe.planner.into(),
            serialize_streams: parallel.serialize_streams,
            max_model_len: model.max_model_len,
            max_num_batched_tokens: MAX_BATCHED_TOKENS,
            hidden_size: model.hidden_size.clone(),
            hc_mult: model.hc_mult,
            num_attention_heads: model.num_attention_heads.clone(),
            num_kv_heads: model.num_kv_heads.clone(),
            head_dim: model.head_dim.clone(),
            rope_dim: model.rope_dim.clone(),
            q_lora_rank: model.q_lora_rank.clone(),
            o_lora_rank: model.o_lora_rank.clone(),
            o_groups: model.o_groups.clone(),
            index_num_heads: model.index_num_heads.clone(),
            index_head_dim: model.index_head_dim.clone(),
            selected_k: model.selected_k,
            activation_dtype: DType::Bf16,
            projection_dtype: DType::Fp8E4m3,
            cache_dtype: DType::Fp8E4m3,
            gpu_name: gpu.clone(),
            mhc_backends: MHC_BACKENDS.to_vec(),
            quant_backends: QUANT_BACKENDS.to_vec(),
            gemm_backends: GEMM_BACKENDS.to_vec(),
            fp32_gemm_backends: FP32_GEMM_BACKENDS.to_vec(),
            fused_q_kv_rmsnorm_backends: vec!["vllm_triton"],
            qnorm_rope_kv_insert_backends: vec!["vllm_cuda"],
            compressor_store_backends: vec!["vllm_deepseek_v4_cutedsl"],
            indexer_compressor_store_backends: vec!["vllm_deepseek_v4_triton"],
            sparse_prefill_backends: vec!["vllm_flashmla_bf16"],
            sparse_decode_backends: vec!["vllm_flashmla_fp8_cudagraph"],
            inverse_rope_quant_backends: vec!["vllm_triton"],
            indexer_q_rope_quant_backends: vec!["vllm_cutedsl_fp8"],
            indexer_prefill_logits_backends: vec!["vllm_deepgemm_fp8"],
            indexer_prefill_topk_backends: vec!["vllm_cuda"],
            indexer_decode_logits_backends: vec!["vllm_deepgemm_fp8"],
            indexer_decode_topk_backends: vec!["vllm_cuda"],
        },
        router: DeepseekV4MoeRouterLocalWorkletConfig {
            hidden_size: model.hidden_size.clone(),
            num_experts: model.num_experts.clone(),
            hidden_dtype: DType::Bf16,
            gpu_name: gpu.clone(),
            hc_mult: model.hc_mult,
            mhc_backends: MHC_BACKENDS.to_vec(),
            gate_backends: FP32_GEMM_BACKENDS.to_vec(),
        },
        dispatch: MoeEpAllGatherKernelConfig {
            backends: COLLECTIVE_BACKENDS.to_vec(),
            gpu_name: gpu.clone(),
            num_gpus: EP_SIZE,
            hidden_size: model.hidden_size.clone(),
            num_experts: model.num_experts.clone(),
            hidden_dtype: DType::Bf16,
            router_dtype: DType::Fp32,
            fabric: "nvlink".into(),
        },
        shared: DeepseekV4SharedExpertLocalWorkletConfig {
            hidden_size: model.hidden_size.clone(),
            intermediate_size: model.expert_intermediate.clone(),
            num_shared_experts: model.num_shared_experts,
            hidden_dtype: DType::Bf16,
            gpu_name: gpu.clone(),
            quant_backends: QUANT_BACKENDS.to_vec(),
            gemm_backends: GEMM_BACKENDS.to_vec(),
            activation_backends: CLAMPED_SWIGLU_BACKENDS.to_vec(),
        },
        expert: DeepseekV4MoeExpertComputeLocalWorkletConfig {
            hidden_size: model.hidden_size.clone(),
            intermediate_size: model.expert_intermediate.clone(),
            selection_mode: recipe.selection.into(),
            num_experts: model.num_experts.clone(),
            top_k: model.top_k,
            hash_vocab_size: if recipe.selection == "hash" {
                VOCAB_SIZE
            } else {
                0
            },
            hidden_dtype: DType::Bf16,
            local_ppm,
            gpu_name: gpu.clone(),
            selection_backends: vec!["vllm_cuda"],
            align_backends: vec!["vllm_cuda"],
            marlin_backends: MARLIN_BACKENDS.to_vec(),
            activation_backends: vec!["vllm_inductor"],
            zero_fill_backends: vec!["torch"],
            sum_backends: vec!["vllm_cuda"],
        },
        combine: MoeEpReduceScatterKernelConfig {
            backends: COLLECTIVE_BACKENDS.to_vec(),
            gpu_name: gpu,
            num_gpus: EP_SIZE,
            hidden_size: model.hidden_size.clone(),
            dtype: DType::Bf16,
            fabric: "nvlink".into(),
        },
        finalize: ElementwiseKernelConfig {
            backends: vec!["torch"],
            gpu_name: parallel.gpu_name.clone(),
            input_bytes_per_token: (2 * HIDDEN * 2).into(),
            output_bytes_per_token: (HIDDEN * 2).into(),
        },
    }
}

pub fn resolve_configs(configs: &DeepseekV4VllmConfigs) -> DeepseekV4VllmResolved {
    DeepseekV4VllmResolved {
        layers: configs
            .layers
            .iter()
            .map(|config| DeepseekV4LayerResolved {
                attention: DeepseekV4AttentionLocalWorklet::resolve_config(&config.attention),
                router: DeepseekV4MoeRouterLocalWorklet::resolve_config(&config.router),
                dispatch: config.dispatch.clone(),
                shared: DeepseekV4SharedExpertLocalWorklet::resolve_config(&config.shared),
                expert: DeepseekV4MoeExpertComputeLocalWorklet::resolve_config(&config.expert),
                combine: config.combine.clone(),
                finalize: config.finalize.clone(),
            })
            .collect(),
        lm_head: configs.lm_head.clone(),
        terminal_mhc_head: configs.terminal_mhc_head.clone(),
        total_kv_bytes_per_token: configs.total_kv_bytes_per_token,
    }
}

struct DeepseekV4LayerBody {
    name: String,
    attention: DeepseekV4AttentionLocalWorklet,
    router: DeepseekV4MoeRouterLocalWorklet,
    dispatch: Op<MoeEpAllGatherKernel>,
    shared: DeepseekV4SharedExpertLocalWorklet,
    expert: DeepseekV4MoeExpertComputeLocalWorklet,
    combine: Op<MoeEpReduceScatterKernel>,
    finalize: Op<ElementwiseKernel>,
}

impl DeepseekV4LayerBody {
    fn build(
        name: String,
        resolved: DeepseekV4LayerResolved,
        bridge: &PerfApiBridge,
    ) -> std::result::Result<Self, BuildError> {
        let dispatch_name = format!("{name}.moe.dispatch_ep_all_gather");
        let combine_name = format!("{name}.moe.combine_ep_reduce_scatter");
        let finalize_name = format!("{name}.moe.shared_routed_add");
        Ok(Self {
            name: name.clone(),
            attention: DeepseekV4AttentionLocalWorklet::build(
                format!("{name}.attention"),
                resolved.attention,
                bridge,
            )?,
            router: DeepseekV4MoeRouterLocalWorklet::build(
                format!("{name}.moe.router"),
                resolved.router,
                bridge,
            )?,
            dispatch: Op::new(
                dispatch_name.clone(),
                Arc::new(MoeEpAllGatherKernel::build(
                    dispatch_name,
                    resolved.dispatch,
                    bridge,
                )?),
            ),
            shared: DeepseekV4SharedExpertLocalWorklet::build(
                format!("{name}.moe.shared_expert"),
                resolved.shared,
                bridge,
            )?,
            expert: DeepseekV4MoeExpertComputeLocalWorklet::build(
                format!("{name}.moe.routed_expert"),
                resolved.expert,
                bridge,
            )?,
            combine: Op::new(
                combine_name.clone(),
                Arc::new(MoeEpReduceScatterKernel::build(
                    combine_name,
                    resolved.combine,
                    bridge,
                )?),
            ),
            finalize: Op::new(
                finalize_name.clone(),
                Arc::new(ElementwiseKernel::build(
                    finalize_name,
                    resolved.finalize,
                    bridge,
                )?),
            ),
        })
    }

    fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: self.name.clone(),
            child: Box::new(CostNode::Sum(vec![
                self.attention.compile(builder),
                self.router.compile(builder),
                self.dispatch.compile(builder),
                self.shared.compile(builder),
                self.expert.compile(builder),
                self.combine.compile(builder),
                self.finalize.compile(builder),
            ])),
        }
    }

    fn eval(&self, input: &NormalizedInput, evaluator: &mut Evaluator) {
        self.attention.eval(&input.attention, evaluator);
        self.router.eval(
            &DeepseekV4MoeRouterLocalWorkletInput {
                num_local_tokens: input.local_tokens,
            },
            evaluator,
        );
        eval_or_zero(
            &self.dispatch,
            MoeEpCollectiveKernelInput {
                per_rank_tokens: input.per_rank_tokens.clone(),
            },
            input.gathered_tokens == 0,
            evaluator,
        );
        self.shared.eval(
            &DeepseekV4SharedExpertLocalWorkletInput {
                num_local_tokens: input.local_tokens,
            },
            evaluator,
        );
        self.expert.eval(
            &DeepseekV4MoeExpertComputeLocalWorkletInput {
                num_gathered_tokens: input.gathered_tokens,
            },
            evaluator,
        );
        eval_or_zero(
            &self.combine,
            MoeEpCollectiveKernelInput {
                per_rank_tokens: input.per_rank_tokens.clone(),
            },
            input.gathered_tokens == 0,
            evaluator,
        );
        eval_or_zero(
            &self.finalize,
            ElementwiseKernelInput {
                num_tokens: input.local_tokens,
            },
            input.local_tokens == 0,
            evaluator,
        );
    }
}

pub struct DeepseekV4VllmModel {
    name: String,
    layers: Vec<DeepseekV4LayerBody>,
    terminal_mhc_head: Op<DeepseekV4TerminalMhcHeadKernel>,
    lm_head: Op<SingleGemmKernel>,
    total_kv_bytes_per_token: u64,
    cost_flat: Vec<FlatCostNode>,
    n_slots: usize,
}

pub fn build(
    name: String,
    resolved: DeepseekV4VllmResolved,
    bridge: &PerfApiBridge,
) -> std::result::Result<DeepseekV4VllmModel, BuildError> {
    let layers = LAYER_RECIPES
        .iter()
        .zip(resolved.layers)
        .map(|(recipe, layer)| {
            DeepseekV4LayerBody::build(format!("{name}.{}", recipe.name), layer, bridge)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let head_name = format!("{name}.terminal_lm_head");
    let terminal_name = format!("{name}.terminal_mhc_head");
    let mut model = DeepseekV4VllmModel {
        name,
        layers,
        terminal_mhc_head: Op::new(
            terminal_name.clone(),
            Arc::new(DeepseekV4TerminalMhcHeadKernel::build(
                terminal_name,
                resolved.terminal_mhc_head,
                bridge,
            )?),
        ),
        lm_head: Op::new(
            head_name.clone(),
            Arc::new(SingleGemmKernel::build(
                head_name,
                resolved.lm_head,
                bridge,
            )?),
        ),
        total_kv_bytes_per_token: resolved.total_kv_bytes_per_token,
        cost_flat: Vec::new(),
        n_slots: 0,
    };
    let tree = model.cost_tree();
    model.cost_flat = tree.flatten();
    model.n_slots = tree.n_slots();
    Ok(model)
}

impl DeepseekV4VllmModel {
    pub fn cost_tree(&self) -> CostTree {
        let mut builder = CostTreeBuilder::new();
        let mut prefix = self.layers[..5]
            .iter()
            .map(|layer| layer.compile(&mut builder))
            .collect::<Vec<_>>();
        let cycle = CostNode::Scale {
            n: CYCLE_COUNT,
            child: Box::new(CostNode::Sum(vec![
                self.layers[5].compile(&mut builder),
                self.layers[6].compile(&mut builder),
            ])),
        };
        prefix.push(cycle);
        prefix.push(self.terminal_mhc_head.compile(&mut builder));
        prefix.push(self.lm_head.compile(&mut builder));
        let root = CostNode::Labeled {
            label: format!(
                "{} (DeepseekV4VllmModel) [EP4; 5 unique layers + 19 C128/C4 cycles]",
                self.name
            ),
            child: Box::new(CostNode::Sum(prefix)),
        };
        builder.finish(root)
    }

    fn eval_into(&self, batch: &UnifiedArchInput, evaluator: &mut Evaluator) {
        let input = normalize_input(batch)
            .unwrap_or_else(|reason| panic!("invalid DeepSeek V4 input: {reason}"));
        for layer in &self.layers {
            layer.eval(&input, evaluator);
        }
        eval_or_zero(
            &self.terminal_mhc_head,
            MhcRmsNormKernelInput {
                num_tokens: input.logits_tokens,
            },
            input.logits_tokens == 0,
            evaluator,
        );
        eval_or_zero(
            &self.lm_head,
            SingleGemmKernelInput {
                m: input.logits_tokens,
            },
            input.logits_tokens == 0,
            evaluator,
        );
    }
}

impl IterwiseUnifiedModel for DeepseekV4VllmModel {
    fn eval_iter(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut evaluator = Evaluator::new(slots);
        self.eval_into(batch, &mut evaluator);
        debug_assert_eq!(evaluator.filled(), self.n_slots);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }
    fn eval_iter_with_inputs(
        &self,
        batch: &UnifiedArchInput,
        slots: &mut Vec<LeafMetrics>,
        scratch: &mut Vec<LeafMetrics>,
        inputs: &mut Vec<SlotInput>,
    ) -> LeafMetrics {
        slots.clear();
        slots.resize(self.n_slots, LeafMetrics::ZERO);
        let mut evaluator = Evaluator::with_inputs(slots, inputs);
        self.eval_into(batch, &mut evaluator);
        debug_assert_eq!(evaluator.filled(), self.n_slots);
        CostTree::aggregate(&self.cost_flat, slots, scratch)
    }
    fn cost_log_manifest(&self) -> CostManifest {
        self.cost_tree().manifest()
    }
    fn total_kv_bytes_per_token(&self) -> u64 {
        self.total_kv_bytes_per_token
    }
    fn gpus_per_replica(&self) -> u16 {
        EP_SIZE as u16
    }
    fn num_attn_dp_groups(&self) -> u16 {
        EP_SIZE as u16
    }
    fn num_attn_shards(&self) -> u16 {
        1
    }
}

struct NormalizedInput {
    attention: DeepseekV4AttentionLocalWorkletInput,
    per_rank_tokens: Vec<u32>,
    local_tokens: u32,
    gathered_tokens: u32,
    logits_tokens: u32,
}

fn normalize_input(input: &UnifiedArchInput) -> std::result::Result<NormalizedInput, String> {
    if input.groups.len() != EP_SIZE as usize
        || input.tokens_per_source_rank.len() != EP_SIZE as usize
    {
        return Err("DeepSeek V4 requires exactly four DP/EP rank inputs".into());
    }
    for (rank, group) in input.groups.iter().enumerate() {
        let append_sum = group
            .prefill_chunk_pairs
            .iter()
            .try_fold(0_u32, |sum, &(_, append)| {
                sum.checked_add(append).ok_or("prefill token sum overflow")
            })?;
        if append_sum != group.prefill_tokens
            || group.decode_tokens != group.decode_kv_lens.len() as u32
            || group.batch_tokens
                != group
                    .prefill_tokens
                    .checked_add(group.decode_tokens)
                    .ok_or("batch token sum overflow")?
            || input.tokens_per_source_rank[rank] != group.batch_tokens
        {
            return Err(format!("rank {rank} token accounting is inconsistent"));
        }
    }
    let critical_rank = input
        .groups
        .iter()
        .enumerate()
        .max_by_key(|(_, group)| group.batch_tokens)
        .map(|(rank, _)| rank)
        .unwrap();
    let group = &input.groups[critical_rank];
    let prefill_query_context_pairs = group
        .prefill_chunk_pairs
        .iter()
        .map(|&(prefix, append)| {
            prefix
                .checked_add(append)
                .map(|context| (append, context))
                .ok_or("prefill context overflow")
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let gathered_tokens = input
        .tokens_per_source_rank
        .iter()
        .try_fold(0_u32, |sum, &tokens| {
            sum.checked_add(tokens).ok_or("gathered token sum overflow")
        })?;
    let logits_tokens = input
        .groups
        .iter()
        .map(|group| group.request_count())
        .max()
        .unwrap_or(0);
    Ok(NormalizedInput {
        attention: DeepseekV4AttentionLocalWorkletInput {
            num_tokens: group.batch_tokens,
            num_insert_tokens: group.batch_tokens,
            prefill_query_context_pairs,
            decode_kv_lens: group.decode_kv_lens.clone(),
        },
        per_rank_tokens: input.tokens_per_source_rank.clone(),
        local_tokens: group.batch_tokens,
        gathered_tokens,
        logits_tokens,
    })
}

fn eval_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, evaluator: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    if zero {
        evaluator.push(LeafMetrics::ZERO, || input.into());
    } else {
        op.eval(&input, evaluator);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::ArchGroupInput;

    #[test]
    fn ragged_dp_input_preserves_collective_topology_and_lowers_only_the_busiest_attention_rank() {
        let groups = vec![
            ArchGroupInput {
                batch_tokens: 4,
                prefill_tokens: 4,
                decode_tokens: 0,
                prefill_chunk_pairs: vec![(8, 4)],
                decode_kv_lens: vec![],
                total_kv_len: 0,
            },
            ArchGroupInput {
                batch_tokens: 2,
                prefill_tokens: 0,
                decode_tokens: 2,
                prefill_chunk_pairs: vec![],
                decode_kv_lens: vec![31, 63],
                total_kv_len: 94,
            },
            ArchGroupInput::default(),
            ArchGroupInput {
                batch_tokens: 3,
                prefill_tokens: 3,
                decode_tokens: 0,
                prefill_chunk_pairs: vec![(0, 3)],
                decode_kv_lens: vec![],
                total_kv_len: 0,
            },
        ];
        let normalized = normalize_input(&UnifiedArchInput {
            groups,
            tokens_per_source_rank: vec![4, 2, 0, 3],
        })
        .unwrap();
        assert_eq!(normalized.per_rank_tokens, [4, 2, 0, 3]);
        assert_eq!(normalized.gathered_tokens, 9);
        assert_eq!(normalized.local_tokens, 4);
        assert_eq!(normalized.attention.prefill_query_context_pairs, [(4, 12)]);
    }

    #[test]
    fn popularity_selects_one_real_contiguous_shard_without_reordering_experts() {
        let mut ppm = vec![1; 256];
        ppm[128] = 100;
        ppm[191] = 50;
        assert_eq!(critical_contiguous_shard(&ppm, 4), ppm[128..192]);
    }
}
