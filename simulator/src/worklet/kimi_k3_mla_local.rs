//! Kimi-K3 rank-local absorbed MLA worklet for the SGLang decode path.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    BatchedGemmKernel, BatchedGemmKernelConfig, BatchedGemmKernelInput, ElementwiseKernel,
    ElementwiseKernelConfig, ElementwiseKernelInput, MlaCacheAppendKernel,
    MlaCacheAppendKernelConfig, MlaCacheAppendKernelInput, MlaDecodeAttentionKernel,
    MlaDecodeAttentionKernelConfig, MlaDecodeAttentionKernelInput, ResidualRmsNormKernel,
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
const SOURCE_ORDER: [&str; 14] = [
    "input_layernorm",
    "fused_qkv_a_proj",
    "q_a_layernorm",
    "q_b_proj",
    "kv_a_layernorm",
    "q_absorb",
    "mla_cache_append",
    "mla_decode_attention",
    "v_up",
    "output_gate",
    "sigmoid_mul",
    "o_proj",
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
    pub absorb_backends: Vec<&'static str>,
    pub cache_append_backends: Vec<&'static str>,
    pub attention_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub tp_size: u16,
}

#[derive(Clone, Debug)]
pub struct KimiK3MlaLocalWorkletResolved {
    pub raw_cfg: KimiK3MlaLocalWorkletConfig,
    pub input_layernorm: ResidualRmsNormKernelConfig,
    pub fused_qkv_a_proj: SingleGemmKernelConfig,
    pub q_a_layernorm: RmsNormKernelConfig,
    pub q_b_proj: SingleGemmKernelConfig,
    pub kv_a_layernorm: RmsNormKernelConfig,
    pub q_absorb: BatchedGemmKernelConfig,
    pub cache_append: MlaCacheAppendKernelConfig,
    pub decode_attention: MlaDecodeAttentionKernelConfig,
    pub v_up: BatchedGemmKernelConfig,
    pub output_gate: SingleGemmKernelConfig,
    pub sigmoid_mul: ElementwiseKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
    pub post_attention_layernorm: ResidualRmsNormKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct KimiK3MlaLocalWorkletInput {
    pub batch_tokens: u32,
    pub decode_kv_lens: Vec<u32>,
}

pub struct KimiK3MlaLocalWorklet {
    pub name: String,
    pub input_layernorm: Op<ResidualRmsNormKernel>,
    pub fused_qkv_a_proj: Op<SingleGemmKernel>,
    pub q_a_layernorm: Op<RmsNormKernel>,
    pub q_b_proj: Op<SingleGemmKernel>,
    pub kv_a_layernorm: Op<RmsNormKernel>,
    pub q_absorb: Op<BatchedGemmKernel>,
    pub cache_append: Op<MlaCacheAppendKernel>,
    pub decode_attention: Op<MlaDecodeAttentionKernel>,
    pub v_up: Op<BatchedGemmKernel>,
    pub output_gate: Op<SingleGemmKernel>,
    pub sigmoid_mul: Op<ElementwiseKernel>,
    pub o_proj: Op<SingleGemmKernel>,
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
        CostNode::Labeled {
            label: format!(
                "{} (KimiK3MlaLocalWorklet) [TP{}; heads={}]",
                self.name, self.resolved.raw_cfg.tp_size, self.resolved.raw_cfg.heads
            ),
            child: Box::new(CostNode::Sum(vec![
                self.input_layernorm.compile(builder),
                self.fused_qkv_a_proj.compile(builder),
                self.q_a_layernorm.compile(builder),
                self.q_b_proj.compile(builder),
                self.kv_a_layernorm.compile(builder),
                self.q_absorb.compile(builder),
                self.cache_append.compile(builder),
                self.decode_attention.compile(builder),
                self.v_up.compile(builder),
                self.output_gate.compile(builder),
                self.sigmoid_mul.compile(builder),
                self.o_proj.compile(builder),
                self.tp_allreduce_zero.compile(builder),
                self.post_attention_layernorm.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &KimiK3MlaLocalWorkletInput, evaluator: &mut Evaluator) {
        let rows = input.batch_tokens;
        let decode_batch = input.decode_kv_lens.len() as u32;
        let kv_len = input.decode_kv_lens.iter().copied().max().unwrap_or(0);
        eval_atomic_or_zero(
            &self.input_layernorm,
            ResidualRmsNormKernelInput { m: rows },
            rows == 0,
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
            rows == 0,
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
            MlaDecodeAttentionKernelInput {
                batch_size: decode_batch,
                kv_len,
            },
            decode_batch == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.v_up,
            BatchedGemmKernelInput { m: rows },
            rows == 0,
            evaluator,
        );
        eval_atomic_or_zero(
            &self.output_gate,
            SingleGemmKernelInput { m: rows },
            rows == 0,
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
            rows == 0,
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
            rows == 0,
            evaluator,
        );
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
            absorb_backends: vec!["sglang_k3_absorb"],
            cache_append_backends: vec!["sglang_cuda"],
            attention_backends: vec!["sglang_cutedsl_mla", "sglang_trtllm_mla"],
            elementwise_backends: vec!["triton"],
            tp_size: 8,
        }
    }

    #[test]
    fn resolved_shapes_match_the_rank_local_mla_recipe() {
        assert_eq!(SOURCE_ORDER.len(), 14);
        let resolved = KimiK3MlaLocalWorklet::resolve_config(&config());
        assert_eq!(resolved.fused_qkv_a_proj.n, 2_112);
        assert_eq!(resolved.q_b_proj.n, 12 * 192);
        assert_eq!(resolved.q_absorb.num_batches, 12);
        assert_eq!(resolved.q_absorb.n, 512);
        assert_eq!(resolved.q_absorb.k, 128);
        assert_eq!(resolved.v_up.k, 512);
        assert_eq!(resolved.v_up.n, 128);
        assert_eq!(resolved.decode_attention.kv_dtype, DType::Fp8E4m3);
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
        };
        assert_eq!(input.decode_kv_lens.iter().copied().max(), Some(8192));
        assert_eq!(input.decode_kv_lens.len(), 2);
    }
}
