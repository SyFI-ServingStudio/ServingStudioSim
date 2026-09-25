//! Kimi-K3 rank-local absorbed MLA worklet for the SGLang decode path.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    BatchedGemmKernel, BatchedGemmKernelConfig, BatchedGemmKernelInput, ElementwiseKernel,
    ElementwiseKernelConfig, ElementwiseKernelInput, K3AttnResPrefillKernel,
    K3AttnResPrefillKernelConfig, K3AttnResPrefillKernelInput, MlaCacheAppendKernel,
    MlaCacheAppendKernelConfig, MlaCacheAppendKernelInput, MlaDecodeAttentionKernel,
    MlaDecodeAttentionKernelConfig, MlaDecodeAttentionKernelInput, MlaMergeStateKernel,
    MlaMergeStateKernelConfig, MlaMergeStateKernelInput, MlaPrefillAttentionKernel,
    MlaPrefillAttentionKernelConfig, MlaPrefillAttentionKernelInput, MlaPrefixGatherKernel,
    MlaPrefixGatherKernelConfig, MlaPrefixGatherKernelInput, ResidualRmsNormKernel,
    ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput, RmsNormKernel, RmsNormKernelConfig,
    RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::kimi_k3_common::{
    build_atomic, eval_atomic_or_zero, zero_all_reduce, ZeroAllReduceProbe,
};

const HIDDEN: u32 = 7_168;
const Q_LORA_RANK: u32 = 1_536;
const KV_LORA_RANK: u32 = 512;
const QK_NOPE: u32 = 128;
const ROPE_DIM: u32 = 64;
const V_HEAD_DIM: u32 = 128;
const PAGE_SIZE: u32 = 64;

#[cfg(test)]
const SOURCE_ORDER: [&str; 23] = [
    "input_layernorm",
    "attn_res_prefill",
    "fused_qkv_a_proj",
    "q_a_layernorm",
    "q_b_proj",
    "q_b_proj_prefill",
    "kv_a_layernorm",
    "q_absorb",
    "mla_cache_append",
    "mla_decode_attention",
    "v_up",
    "mla_prefix_gather",
    "mla_kv_b_proj_prefill",
    "mla_prefill_attention_prefix",
    "mla_prefill_attention_causal",
    "mla_merge_state",
    "output_gate",
    "output_gate_prefill",
    "sigmoid_mul",
    "o_proj",
    "o_proj_prefill",
    "tp_allreduce_zero",
    "post_attention_layernorm",
];

#[derive(Clone, Debug)]
pub struct KimiK3MlaLocalWorkletConfig {
    pub gpu_name: String,
    pub hidden: Dim,
    pub heads: Dim,
    pub q_lora_rank: Dim,
    pub kv_lora_rank: Dim,
    pub qk_nope: Dim,
    pub rope_dim: Dim,
    pub v_head_dim: Dim,
    pub page_size: Dim,
    pub dtype: DType,
    pub cache_dtype: DType,
    pub residual_norm_backends: Vec<&'static str>,
    pub rms_norm_backends: Vec<&'static str>,
    pub fused_qkv_a_backends: Vec<&'static str>,
    pub projection_backends: Vec<&'static str>,
    pub prefill_projection_backends: Vec<&'static str>,
    pub absorb_backends: Vec<&'static str>,
    pub cache_append_backends: Vec<&'static str>,
    pub attention_backends: Vec<&'static str>,
    pub prefill_attention_backends: Vec<&'static str>,
    pub prefill_aux_backends: Vec<&'static str>,
    pub prefill_attn_res_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub tp_size: u16,
}

#[derive(Clone, Debug)]
pub struct KimiK3MlaLocalWorkletResolved {
    pub raw_cfg: KimiK3MlaLocalWorkletConfig,
    pub input_layernorm: ResidualRmsNormKernelConfig,
    pub prefill_attn_res: K3AttnResPrefillKernelConfig,
    pub fused_qkv_a_proj: SingleGemmKernelConfig,
    pub q_a_layernorm: RmsNormKernelConfig,
    pub q_b_proj: SingleGemmKernelConfig,
    pub q_b_proj_prefill: SingleGemmKernelConfig,
    pub kv_a_layernorm: RmsNormKernelConfig,
    pub q_absorb: BatchedGemmKernelConfig,
    pub cache_append: MlaCacheAppendKernelConfig,
    pub decode_attention: MlaDecodeAttentionKernelConfig,
    pub prefill_attention_prefix: MlaPrefillAttentionKernelConfig,
    pub prefill_attention_causal: MlaPrefillAttentionKernelConfig,
    pub prefill_prefix_gather: MlaPrefixGatherKernelConfig,
    pub prefill_kv_b_proj: BatchedGemmKernelConfig,
    pub prefill_merge_state: MlaMergeStateKernelConfig,
    pub v_up: BatchedGemmKernelConfig,
    pub output_gate: SingleGemmKernelConfig,
    pub output_gate_prefill: SingleGemmKernelConfig,
    pub sigmoid_mul: ElementwiseKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
    pub o_proj_prefill: SingleGemmKernelConfig,
    pub post_attention_layernorm: ResidualRmsNormKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct KimiK3MlaLocalWorkletInput {
    pub batch_tokens: u32,
    pub decode_kv_lens: Vec<u32>,
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
}

pub struct KimiK3MlaLocalWorklet {
    pub name: String,
    pub input_layernorm: Op<ResidualRmsNormKernel>,
    pub prefill_attn_res: Op<K3AttnResPrefillKernel>,
    pub fused_qkv_a_proj: Op<SingleGemmKernel>,
    pub q_a_layernorm: Op<RmsNormKernel>,
    pub q_b_proj: Op<SingleGemmKernel>,
    pub q_b_proj_prefill: Op<SingleGemmKernel>,
    pub kv_a_layernorm: Op<RmsNormKernel>,
    pub q_absorb: Op<BatchedGemmKernel>,
    pub cache_append: Op<MlaCacheAppendKernel>,
    pub decode_attention: Op<MlaDecodeAttentionKernel>,
    pub prefill_attention_prefix: Op<MlaPrefillAttentionKernel>,
    pub prefill_attention_causal: Op<MlaPrefillAttentionKernel>,
    pub prefill_prefix_gather: Op<MlaPrefixGatherKernel>,
    pub prefill_kv_b_proj: Op<BatchedGemmKernel>,
    pub prefill_merge_state: Op<MlaMergeStateKernel>,
    pub v_up: Op<BatchedGemmKernel>,
    pub output_gate: Op<SingleGemmKernel>,
    pub output_gate_prefill: Op<SingleGemmKernel>,
    pub sigmoid_mul: Op<ElementwiseKernel>,
    pub o_proj: Op<SingleGemmKernel>,
    pub o_proj_prefill: Op<SingleGemmKernel>,
    pub tp_allreduce_zero: Op<ZeroAllReduceProbe>,
    pub post_attention_layernorm: Op<ResidualRmsNormKernel>,
    resolved: KimiK3MlaLocalWorkletResolved,
}

impl KimiK3MlaLocalWorklet {
    pub fn resolve_config(cfg: &KimiK3MlaLocalWorkletConfig) -> KimiK3MlaLocalWorkletResolved {
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid KimiK3MlaLocalWorkletConfig: {reason}"));
        let heads = cfg.heads.get();
        let q_b_n = heads * (cfg.qk_nope.get() + cfg.rope_dim.get());
        KimiK3MlaLocalWorkletResolved {
            input_layernorm: ResidualRmsNormKernelConfig {
                backends: cfg.residual_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            prefill_attn_res: K3AttnResPrefillKernelConfig {
                backends: cfg.prefill_attn_res_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden_size: cfg.hidden.clone(),
                num_valid_blocks: 1,
                num_launches: 2,
                write_prefix: false,
                dtype: cfg.dtype,
            },
            fused_qkv_a_proj: SingleGemmKernelConfig {
                backends: cfg.fused_qkv_a_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: (cfg.q_lora_rank.get() + cfg.kv_lora_rank.get() + cfg.rope_dim.get()).into(),
                k: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            q_a_layernorm: RmsNormKernelConfig {
                backends: cfg.rms_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.q_lora_rank.clone(),
                dtype: cfg.dtype,
            },
            q_b_proj: SingleGemmKernelConfig {
                backends: cfg.projection_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: q_b_n.into(),
                k: cfg.q_lora_rank.clone(),
                dtype: cfg.dtype,
            },
            q_b_proj_prefill: SingleGemmKernelConfig {
                backends: cfg.prefill_projection_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: q_b_n.into(),
                k: cfg.q_lora_rank.clone(),
                dtype: cfg.dtype,
            },
            kv_a_layernorm: RmsNormKernelConfig {
                backends: cfg.rms_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.kv_lora_rank.clone(),
                dtype: cfg.dtype,
            },
            q_absorb: BatchedGemmKernelConfig {
                backends: cfg.absorb_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_batches: cfg.heads.clone(),
                n: cfg.kv_lora_rank.clone(),
                k: cfg.qk_nope.clone(),
                dtype: cfg.dtype,
            },
            cache_append: MlaCacheAppendKernelConfig {
                backends: cfg.cache_append_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                kv_lora_rank: cfg.kv_lora_rank.clone(),
                rope_dim: cfg.rope_dim.clone(),
                block_size: PAGE_SIZE,
                // The latent+rope vector arrives in the activation dtype (bf16) and is
                // quantized to the fp8 cache inside set_mla_kv_concat_q_fp8.
                input_dtype: cfg.dtype,
                kv_dtype: cfg.cache_dtype,
                cache_format: "page_planar_fp8".to_string(),
            },
            decode_attention: MlaDecodeAttentionKernelConfig {
                backends: cfg.attention_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.heads.clone(),
                kv_lora_rank: cfg.kv_lora_rank.clone(),
                rope_dim: cfg.rope_dim.clone(),
                q_dtype: cfg.dtype,
                kv_dtype: cfg.cache_dtype,
                page_size: cfg.page_size.clone(),
            },
            prefill_attention_prefix: MlaPrefillAttentionKernelConfig {
                backends: cfg.prefill_attention_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.heads.clone(),
                qk_head_dim: (cfg.qk_nope.get() + cfg.rope_dim.get()).into(),
                v_head_dim: cfg.v_head_dim.clone(),
                q_dtype: DType::Fp8E4m3,
                kv_dtype: cfg.cache_dtype,
                o_dtype: cfg.dtype,
                causal: false,
            },
            prefill_attention_causal: MlaPrefillAttentionKernelConfig {
                backends: cfg.prefill_attention_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.heads.clone(),
                qk_head_dim: (cfg.qk_nope.get() + cfg.rope_dim.get()).into(),
                v_head_dim: cfg.v_head_dim.clone(),
                q_dtype: DType::Fp8E4m3,
                kv_dtype: cfg.cache_dtype,
                o_dtype: cfg.dtype,
                causal: true,
            },
            prefill_prefix_gather: MlaPrefixGatherKernelConfig {
                backends: cfg.prefill_aux_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                kv_lora_rank: cfg.kv_lora_rank.clone(),
                rope_dim: cfg.rope_dim.clone(),
                dtype: cfg.dtype,
                cache_dtype: cfg.cache_dtype,
            },
            prefill_kv_b_proj: BatchedGemmKernelConfig {
                backends: cfg.absorb_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_batches: cfg.heads.clone(),
                n: (cfg.qk_nope.get() + cfg.v_head_dim.get()).into(),
                k: cfg.kv_lora_rank.clone(),
                dtype: cfg.dtype,
            },
            prefill_merge_state: MlaMergeStateKernelConfig {
                backends: cfg.prefill_aux_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.heads.clone(),
                value_dim: cfg.v_head_dim.clone(),
                dtype: cfg.dtype,
            },
            v_up: BatchedGemmKernelConfig {
                backends: cfg.absorb_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_batches: cfg.heads.clone(),
                n: cfg.v_head_dim.clone(),
                k: cfg.kv_lora_rank.clone(),
                dtype: cfg.dtype,
            },
            output_gate: SingleGemmKernelConfig {
                backends: cfg.projection_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: (heads * cfg.v_head_dim.get()).into(),
                k: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            output_gate_prefill: SingleGemmKernelConfig {
                backends: cfg.prefill_projection_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: (heads * cfg.v_head_dim.get()).into(),
                k: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            sigmoid_mul: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: (2 * heads * cfg.v_head_dim.get() * 2).into(),
                output_bytes_per_token: (heads * cfg.v_head_dim.get() * 2).into(),
            },
            o_proj: SingleGemmKernelConfig {
                backends: cfg.projection_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden.clone(),
                k: (heads * cfg.v_head_dim.get()).into(),
                dtype: cfg.dtype,
            },
            o_proj_prefill: SingleGemmKernelConfig {
                backends: cfg.prefill_projection_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden.clone(),
                k: (heads * cfg.v_head_dim.get()).into(),
                dtype: cfg.dtype,
            },
            post_attention_layernorm: ResidualRmsNormKernelConfig {
                backends: cfg.residual_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden.clone(),
                dtype: cfg.dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: KimiK3MlaLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, crate::timing::BuildError> {
        let tp_allreduce_zero = zero_all_reduce(
            format!("{name}.tp_allreduce_zero"),
            serde_json::json!({
                "gpu_name": resolved.raw_cfg.gpu_name,
                "num_gpus": resolved.raw_cfg.tp_size,
                "fabric": "nvlink",
                "zero_time": true,
            }),
        );
        Ok(Self {
            input_layernorm: build_atomic(
                &name,
                "input_layernorm",
                resolved.input_layernorm.clone(),
                ResidualRmsNormKernel::build,
                bridge,
            )?,
            prefill_attn_res: build_atomic(
                &name,
                "attn_res_prefill",
                resolved.prefill_attn_res.clone(),
                K3AttnResPrefillKernel::build,
                bridge,
            )?,
            fused_qkv_a_proj: build_atomic(
                &name,
                "fused_qkv_a_proj",
                resolved.fused_qkv_a_proj.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            q_a_layernorm: build_atomic(
                &name,
                "q_a_layernorm",
                resolved.q_a_layernorm.clone(),
                RmsNormKernel::build,
                bridge,
            )?,
            q_b_proj: build_atomic(
                &name,
                "q_b_proj",
                resolved.q_b_proj.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            q_b_proj_prefill: build_atomic(
                &name,
                "q_b_proj_prefill",
                resolved.q_b_proj_prefill.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            kv_a_layernorm: build_atomic(
                &name,
                "kv_a_layernorm",
                resolved.kv_a_layernorm.clone(),
                RmsNormKernel::build,
                bridge,
            )?,
            q_absorb: build_atomic(
                &name,
                "q_absorb",
                resolved.q_absorb.clone(),
                BatchedGemmKernel::build,
                bridge,
            )?,
            cache_append: build_atomic(
                &name,
                "mla_cache_append",
                resolved.cache_append.clone(),
                MlaCacheAppendKernel::build,
                bridge,
            )?,
            decode_attention: build_atomic(
                &name,
                "mla_decode_attention",
                resolved.decode_attention.clone(),
                MlaDecodeAttentionKernel::build,
                bridge,
            )?,
            prefill_attention_prefix: build_atomic(
                &name,
                "mla_prefill_attention_prefix",
                resolved.prefill_attention_prefix.clone(),
                MlaPrefillAttentionKernel::build,
                bridge,
            )?,
            prefill_attention_causal: build_atomic(
                &name,
                "mla_prefill_attention_causal",
                resolved.prefill_attention_causal.clone(),
                MlaPrefillAttentionKernel::build,
                bridge,
            )?,
            prefill_prefix_gather: build_atomic(
                &name,
                "mla_prefix_gather",
                resolved.prefill_prefix_gather.clone(),
                MlaPrefixGatherKernel::build,
                bridge,
            )?,
            prefill_kv_b_proj: build_atomic(
                &name,
                "mla_kv_b_proj_prefill",
                resolved.prefill_kv_b_proj.clone(),
                BatchedGemmKernel::build,
                bridge,
            )?,
            prefill_merge_state: build_atomic(
                &name,
                "mla_merge_state",
                resolved.prefill_merge_state.clone(),
                MlaMergeStateKernel::build,
                bridge,
            )?,
            v_up: build_atomic(
                &name,
                "v_up",
                resolved.v_up.clone(),
                BatchedGemmKernel::build,
                bridge,
            )?,
            output_gate: build_atomic(
                &name,
                "output_gate",
                resolved.output_gate.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            output_gate_prefill: build_atomic(
                &name,
                "output_gate_prefill",
                resolved.output_gate_prefill.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            sigmoid_mul: build_atomic(
                &name,
                "sigmoid_mul",
                resolved.sigmoid_mul.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            o_proj: build_atomic(
                &name,
                "o_proj",
                resolved.o_proj.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            o_proj_prefill: build_atomic(
                &name,
                "o_proj_prefill",
                resolved.o_proj_prefill.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            tp_allreduce_zero,
            post_attention_layernorm: build_atomic(
                &name,
                "post_attention_layernorm",
                resolved.post_attention_layernorm.clone(),
                ResidualRmsNormKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        // The q and KV latent paths fan out after fused_qkv_a_proj. The gate
        // GEMM is launched on SGLang's alternate stream and joins before the
        // sigmoid/multiply barrier, so both stream boundaries are explicit.
        let input_layernorm = self.input_layernorm.compile(builder);
        let prefill_attn_res = self.prefill_attn_res.compile(builder);
        let fused_qkv_a_proj = self.fused_qkv_a_proj.compile(builder);
        let q_a_layernorm = self.q_a_layernorm.compile(builder);
        let q_b_proj = self.q_b_proj.compile(builder);
        let q_b_proj_prefill = self.q_b_proj_prefill.compile(builder);
        let kv_a_layernorm = self.kv_a_layernorm.compile(builder);
        let q_absorb = self.q_absorb.compile(builder);
        let cache_append = self.cache_append.compile(builder);
        let decode_attention = self.decode_attention.compile(builder);
        let v_up = self.v_up.compile(builder);
        let prefill_prefix_gather = self.prefill_prefix_gather.compile(builder);
        let prefill_kv_b_proj = self.prefill_kv_b_proj.compile(builder);
        let prefill_attention_prefix = self.prefill_attention_prefix.compile(builder);
        let prefill_attention_causal = self.prefill_attention_causal.compile(builder);
        let prefill_merge_state = self.prefill_merge_state.compile(builder);
        let output_gate = self.output_gate.compile(builder);
        let output_gate_prefill = self.output_gate_prefill.compile(builder);
        let sigmoid_mul = self.sigmoid_mul.compile(builder);
        let o_proj = self.o_proj.compile(builder);
        let o_proj_prefill = self.o_proj_prefill.compile(builder);
        let tp_allreduce_zero = self.tp_allreduce_zero.compile(builder);
        let post_attention_layernorm = self.post_attention_layernorm.compile(builder);

        let latent_paths = CostNode::Labeled {
            label: format!("{}.latent q/KV paths [concurrent streams]", self.name),
            child: Box::new(CostNode::Max {
                overlap: 1.0,
                children: vec![
                    CostNode::Sum(vec![q_a_layernorm, q_b_proj, q_b_proj_prefill, q_absorb]),
                    CostNode::Sum(vec![kv_a_layernorm, cache_append]),
                ],
            }),
        };
        let prefill_core = CostNode::Sum(vec![
            prefill_prefix_gather,
            prefill_kv_b_proj,
            prefill_attention_prefix,
            prefill_attention_causal,
            prefill_merge_state,
        ]);
        let attention_core = CostNode::Sum(vec![
            fused_qkv_a_proj,
            latent_paths,
            decode_attention,
            v_up,
            prefill_core,
        ]);
        let attention_and_gate = CostNode::Labeled {
            label: format!("{}.attention + output gate [concurrent streams]", self.name),
            child: Box::new(CostNode::Max {
                overlap: 1.0,
                children: vec![attention_core, output_gate, output_gate_prefill],
            }),
        };
        CostNode::Labeled {
            label: format!(
                "{} (KimiK3MlaLocalWorklet) [TP{}; heads={}]",
                self.name, self.resolved.raw_cfg.tp_size, self.resolved.raw_cfg.heads
            ),
            child: Box::new(CostNode::Sum(vec![
                input_layernorm,
                prefill_attn_res,
                attention_and_gate,
                sigmoid_mul,
                o_proj,
                o_proj_prefill,
                tp_allreduce_zero,
                post_attention_layernorm,
            ])),
        }
    }

    pub fn eval(&self, input: &KimiK3MlaLocalWorkletInput, evaluator: &mut Evaluator) {
        let rows = phase_token_count(input.batch_tokens, &input.prefill_chunk_pairs);
        let is_prefill = !input.prefill_chunk_pairs.is_empty();
        eval_atomic_or_zero(
            &self.input_layernorm,
            ResidualRmsNormKernelInput { m: rows },
            rows == 0 || is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.prefill_attn_res,
            K3AttnResPrefillKernelInput { num_tokens: rows },
            rows == 0 || !is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.fused_qkv_a_proj,
            SingleGemmKernelInput { m: rows },
            rows == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.q_a_layernorm,
            RmsNormKernelInput { m: rows },
            rows == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.q_b_proj,
            SingleGemmKernelInput { m: rows },
            rows == 0 || is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.q_b_proj_prefill,
            SingleGemmKernelInput { m: rows },
            rows == 0 || !is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.kv_a_layernorm,
            RmsNormKernelInput { m: rows },
            rows == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.q_absorb,
            BatchedGemmKernelInput { m: rows },
            rows == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.cache_append,
            MlaCacheAppendKernelInput { num_tokens: rows },
            rows == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.decode_attention,
            decode_attention_input(input),
            input.decode_kv_lens.is_empty(),
            evaluator,
        );
        eval_atomic_or_zero(
            &self.v_up,
            BatchedGemmKernelInput { m: rows },
            rows == 0,
            evaluator,
        );
        eval_prefill(self, input, evaluator);
        eval_atomic_or_zero(
            &self.output_gate,
            SingleGemmKernelInput { m: rows },
            rows == 0 || is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.output_gate_prefill,
            SingleGemmKernelInput { m: rows },
            rows == 0 || !is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.sigmoid_mul,
            ElementwiseKernelInput { num_tokens: rows },
            rows == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.o_proj,
            SingleGemmKernelInput { m: rows },
            rows == 0 || is_prefill,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.o_proj_prefill,
            SingleGemmKernelInput { m: rows },
            rows == 0 || !is_prefill,
            evaluator,
        );
        self.tp_allreduce_zero.eval(
            &crate::timing::kernels::AllReduceKernelInput {
                message_size_bytes: u64::from(rows) * u64::from(HIDDEN) * 2,
            },
            evaluator,
        );
        eval_atomic_or_zero(
            &self.post_attention_layernorm,
            ResidualRmsNormKernelInput { m: rows },
            rows == 0 || is_prefill,
            evaluator,
        );
    }
}

const MAX_KV_CHUNK_CAPACITY: u32 = 131_072;

#[derive(Clone, Copy)]
struct PrefillSummary {
    batch_size: u32,
    num_tokens: u32,
    q_len: u32,
    prefix_len: u32,
    prefix_chunk_tokens: u32,
}

fn prefill_summary(input: &KimiK3MlaLocalWorkletInput) -> Option<PrefillSummary> {
    if input.prefill_chunk_pairs.is_empty() {
        return None;
    }
    let batch_size = input.prefill_chunk_pairs.len() as u32;
    let num_tokens = input
        .prefill_chunk_pairs
        .iter()
        .try_fold(0_u32, |total, &(_prefix, append)| total.checked_add(append))
        .expect("MLA prefill token count must fit u32");
    let q_len = input
        .prefill_chunk_pairs
        .iter()
        .map(|&(_prefix, append)| append)
        .max()
        .expect("non-empty MLA prefill pairs have an append length");
    let prefix_len = input
        .prefill_chunk_pairs
        .iter()
        .map(|&(prefix, _append)| prefix)
        .max()
        .expect("non-empty MLA prefill pairs have a prefix length");
    let prefix_chunk_tokens = if prefix_len == 0 {
        0
    } else {
        prefix_len.min(MAX_KV_CHUNK_CAPACITY / batch_size.max(1))
    };
    Some(PrefillSummary {
        batch_size,
        num_tokens,
        q_len,
        prefix_len,
        prefix_chunk_tokens,
    })
}

fn phase_token_count(batch_tokens: u32, pairs: &[(u32, u32)]) -> u32 {
    if pairs.is_empty() {
        return batch_tokens;
    }
    pairs
        .iter()
        .try_fold(0_u32, |total, &(_prefix, append)| total.checked_add(append))
        .expect("MLA prefill token count must fit u32")
}

fn eval_prefill(
    worklet: &KimiK3MlaLocalWorklet,
    input: &KimiK3MlaLocalWorkletInput,
    evaluator: &mut Evaluator,
) {
    let Some(summary) = prefill_summary(input) else {
        eval_atomic_or_zero(
            &worklet.prefill_prefix_gather,
            MlaPrefixGatherKernelInput {
                batch_size: 0,
                num_tokens: 0,
            },
            true,
            evaluator,
        );
        eval_atomic_or_zero(
            &worklet.prefill_kv_b_proj,
            BatchedGemmKernelInput { m: 0 },
            true,
            evaluator,
        );
        eval_atomic_or_zero(
            &worklet.prefill_attention_prefix,
            MlaPrefillAttentionKernelInput {
                batch_size: 0,
                q_len: 0,
                kv_len: 0,
            },
            true,
            evaluator,
        );
        eval_atomic_or_zero(
            &worklet.prefill_attention_causal,
            MlaPrefillAttentionKernelInput {
                batch_size: 0,
                q_len: 0,
                kv_len: 0,
            },
            true,
            evaluator,
        );
        eval_atomic_or_zero(
            &worklet.prefill_merge_state,
            MlaMergeStateKernelInput { num_tokens: 0 },
            true,
            evaluator,
        );
        return;
    };

    let prefix_tokens = summary.prefix_chunk_tokens * summary.batch_size;
    eval_atomic_or_zero(
        &worklet.prefill_prefix_gather,
        MlaPrefixGatherKernelInput {
            batch_size: summary.batch_size,
            num_tokens: prefix_tokens,
        },
        prefix_tokens == 0,
        evaluator,
    );
    eval_atomic_or_zero(
        &worklet.prefill_kv_b_proj,
        BatchedGemmKernelInput {
            m: summary.prefix_chunk_tokens,
        },
        summary.prefix_chunk_tokens == 0,
        evaluator,
    );
    eval_atomic_or_zero(
        &worklet.prefill_attention_prefix,
        MlaPrefillAttentionKernelInput {
            batch_size: summary.batch_size,
            q_len: summary.q_len,
            kv_len: summary.prefix_chunk_tokens,
        },
        summary.prefix_chunk_tokens == 0,
        evaluator,
    );
    eval_atomic_or_zero(
        &worklet.prefill_attention_causal,
        MlaPrefillAttentionKernelInput {
            batch_size: summary.batch_size,
            q_len: summary.q_len,
            kv_len: summary.q_len,
        },
        false,
        evaluator,
    );
    eval_atomic_or_zero(
        &worklet.prefill_merge_state,
        MlaMergeStateKernelInput {
            num_tokens: summary.num_tokens,
        },
        summary.prefix_len == 0,
        evaluator,
    );
}

fn decode_attention_input(input: &KimiK3MlaLocalWorkletInput) -> MlaDecodeAttentionKernelInput {
    MlaDecodeAttentionKernelInput {
        // MLA receives one query per decode request. `decode_kv_total` is a
        // workload aggregate; the callable expects the per-request context
        // length and batch cardinality separately.
        batch_size: input.decode_kv_lens.len() as u32,
        kv_len: input.decode_kv_lens.iter().copied().max().unwrap_or(0),
    }
}

fn validate_config(cfg: &KimiK3MlaLocalWorkletConfig) -> Result<(), String> {
    for (name, actual, required) in [
        ("hidden", cfg.hidden.get(), HIDDEN),
        ("q_lora_rank", cfg.q_lora_rank.get(), Q_LORA_RANK),
        ("kv_lora_rank", cfg.kv_lora_rank.get(), KV_LORA_RANK),
        ("qk_nope", cfg.qk_nope.get(), QK_NOPE),
        ("rope_dim", cfg.rope_dim.get(), ROPE_DIM),
        ("v_head_dim", cfg.v_head_dim.get(), V_HEAD_DIM),
        ("page_size", cfg.page_size.get(), PAGE_SIZE),
    ] {
        if actual != required {
            return Err(format!("{name} must be {required}, got {actual}"));
        }
    }
    if cfg.heads.get() == 0 {
        return Err("heads must be positive".to_string());
    }
    if cfg.prefill_attention_backends.is_empty() || cfg.prefill_aux_backends.is_empty() {
        return Err("K3 MLA requires non-empty prefill backend lists".to_string());
    }
    if cfg.prefill_attn_res_backends.is_empty() {
        return Err("K3 MLA requires a non-empty attention-residual backend list".to_string());
    }
    if cfg.dtype != DType::Bf16 || cfg.cache_dtype != DType::Fp8E4m3 {
        return Err("K3 MLA uses bf16 compute and fp8_e4m3 cache".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> KimiK3MlaLocalWorkletConfig {
        KimiK3MlaLocalWorkletConfig {
            gpu_name: "NVIDIA B200".to_string(),
            hidden: HIDDEN.into(),
            heads: 12.into(),
            q_lora_rank: Q_LORA_RANK.into(),
            kv_lora_rank: KV_LORA_RANK.into(),
            qk_nope: QK_NOPE.into(),
            rope_dim: ROPE_DIM.into(),
            v_head_dim: V_HEAD_DIM.into(),
            page_size: PAGE_SIZE.into(),
            dtype: DType::Bf16,
            cache_dtype: DType::Fp8E4m3,
            residual_norm_backends: vec!["vllm_cuda"],
            rms_norm_backends: vec!["flashinfer"],
            fused_qkv_a_backends: vec!["sglang_fused_a_auto"],
            projection_backends: vec!["sglang_bf16_auto"],
            prefill_projection_backends: vec!["sglang_k3_raw_bf16"],
            absorb_backends: vec!["sglang_k3_absorb"],
            cache_append_backends: vec!["sglang_cuda"],
            attention_backends: vec!["sglang_cutedsl_mla", "sglang_trtllm_mla"],
            prefill_attention_backends: vec!["sglang_trtllm_mla"],
            prefill_aux_backends: vec!["sglang_triton"],
            prefill_attn_res_backends: vec!["sglang_k3"],
            elementwise_backends: vec!["triton"],
            tp_size: 8,
        }
    }

    #[test]
    fn resolved_shapes_match_the_rank_local_mla_recipe() {
        assert_eq!(SOURCE_ORDER.len(), 23);
        let resolved = KimiK3MlaLocalWorklet::resolve_config(&config());
        assert_eq!(resolved.fused_qkv_a_proj.n, 2_112);
        assert_eq!(resolved.q_b_proj.n, 12 * 192);
        assert_eq!(resolved.q_absorb.num_batches, 12);
        assert_eq!(resolved.q_absorb.n, 512);
        assert_eq!(resolved.q_absorb.k, 128);
        assert_eq!(resolved.v_up.k, 512);
        assert_eq!(resolved.v_up.n, 128);
        assert_eq!(resolved.prefill_kv_b_proj.num_batches, 12);
        assert_eq!(resolved.prefill_kv_b_proj.k, 512);
        assert_eq!(resolved.prefill_kv_b_proj.n, 256);
        assert_eq!(resolved.decode_attention.kv_dtype, DType::Fp8E4m3);
        assert_eq!(resolved.prefill_attention_causal.qk_head_dim, 192);
        assert!(!resolved.prefill_attention_prefix.causal);
        assert!(resolved.prefill_attention_causal.causal);
        assert_eq!(resolved.cache_append.kv_lora_rank, 512);
        assert_eq!(resolved.cache_append.rope_dim, 64);
    }

    #[test]
    fn kv_bytes_are_576_fp8_bytes_per_token_per_mla_layer() {
        let kv_width = KV_LORA_RANK + ROPE_DIM;
        assert_eq!(kv_width, 576);
        assert_eq!(u64::from(kv_width) * 1 * 24, 13_824);
    }

    #[test]
    fn ragged_decode_input_uses_the_busiest_context_for_one_cache_axis() {
        let input = KimiK3MlaLocalWorkletInput {
            batch_tokens: 3,
            decode_kv_lens: vec![64, 8192],
            prefill_chunk_pairs: Vec::new(),
        };
        let attention = decode_attention_input(&input);
        assert_eq!(attention.batch_size, 2);
        assert_eq!(attention.kv_len, 8192);
    }

    #[test]
    fn large_uniform_decode_group_keeps_request_count_separate_from_kv_total() {
        let input = KimiK3MlaLocalWorkletInput {
            batch_tokens: 512,
            decode_kv_lens: vec![8192; 512],
            prefill_chunk_pairs: Vec::new(),
        };
        let attention = decode_attention_input(&input);
        assert_eq!(attention.batch_size, 512);
        assert_eq!(attention.kv_len, 8192);
    }

    #[test]
    fn prefill_summary_selects_prefix_chunk_and_causal_paths() {
        let input = KimiK3MlaLocalWorkletInput {
            batch_tokens: 16_384,
            decode_kv_lens: Vec::new(),
            prefill_chunk_pairs: vec![(49_152, 16_384)],
        };
        let summary = prefill_summary(&input).unwrap();
        assert_eq!(summary.batch_size, 1);
        assert_eq!(summary.num_tokens, 16_384);
        assert_eq!(summary.q_len, 16_384);
        assert_eq!(summary.prefix_chunk_tokens, 49_152);

        let input = KimiK3MlaLocalWorkletInput {
            batch_tokens: 16_384,
            decode_kv_lens: Vec::new(),
            prefill_chunk_pairs: vec![(0, 4_096); 4],
        };
        let summary = prefill_summary(&input).unwrap();
        assert_eq!(summary.batch_size, 4);
        assert_eq!(summary.num_tokens, 16_384);
        assert_eq!(summary.q_len, 4_096);
        assert_eq!(summary.prefix_chunk_tokens, 0);
    }
}
