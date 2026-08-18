//! Qwen3.6 TP1 gated-GQA attention section in vLLM launch granularity.
//!
//! This local, self-synchronizing section starts at the decoder's delayed
//! residual-add + input RMSNorm boundary and ends at the post-attention
//! residual-add + RMSNorm boundary. The following MoE router consumes the
//! normalized hidden states and therefore owns no norm. There are no TP, EP,
//! collective, or network children.

use std::sync::Arc;

use crate::op::attention::{
    FlashInferAttentionConfig, FlashInferAttentionInput, FlashInferAttentionOp,
};
use crate::op::gemm::{
    SingleFp8GemmWithQuantConfig, SingleFp8GemmWithQuantInput, SingleFp8GemmWithQuantOp,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    Fp8PerTokenGroupQuantKernelConfig, ResidualRmsNormKernel,
    ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput, RmsNormKernel,
    RmsNormKernelConfig, RmsNormKernelInput, SingleGemmKernelConfig,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge,
};

const HIDDEN: u32 = 2048;
const NUM_QO_HEADS: u32 = 16;
const NUM_KV_HEADS: u32 = 2;
const HEAD_DIM: u32 = 256;
const ROPE_DIM: u32 = 64;
const Q_WIDTH: u32 = NUM_QO_HEADS * HEAD_DIM;
const OUTPUT_GATE_WIDTH: u32 = Q_WIDTH;
const K_WIDTH: u32 = NUM_KV_HEADS * HEAD_DIM;
const V_WIDTH: u32 = K_WIDTH;
const QKV_GATE_WIDTH: u32 = Q_WIDTH + OUTPUT_GATE_WIDTH + K_WIDTH + V_WIDTH;
const FP8_GROUP_SIZE: u32 = 128;
const SCALE_FORMAT: &str = "ue8m0_column_major";
const PARTIAL_ROPE_BYTES_PER_TOKEN: u32 = (NUM_QO_HEADS + NUM_KV_HEADS) * ROPE_DIM * 2;
const OUTPUT_GATE_INPUT_BYTES_PER_TOKEN: u32 = 2 * Q_WIDTH * 2;
const OUTPUT_GATE_OUTPUT_BYTES_PER_TOKEN: u32 = Q_WIDTH * 2;

#[derive(Clone, Debug)]
pub struct Qwen36GatedGqaLocalWorkletConfig {
    pub hidden: Dim,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub rope_dim: Dim,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub kv_cache_block_size: u32,
    pub kv_cache_layout: String,
    pub kv_scale_granularity: String,
    pub residual_rms_norm_backends: Vec<&'static str>,
    pub fp8_quant_backends: Vec<&'static str>,
    pub fp8_gemm_backends: Vec<&'static str>,
    pub qk_rms_norm_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub attention_backends: Vec<&'static str>,
    pub kv_cache_append_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Qwen36GatedGqaLocalWorkletResolved {
    pub raw_cfg: Qwen36GatedGqaLocalWorkletConfig,
    pub input_add_rms_norm: ResidualRmsNormKernelConfig,
    pub qkv_gate: SingleFp8GemmWithQuantConfig,
    pub q_norm: RmsNormKernelConfig,
    pub k_norm: RmsNormKernelConfig,
    pub partial_rope: ElementwiseKernelConfig,
    pub attention: FlashInferAttentionConfig,
    pub output_gate: ElementwiseKernelConfig,
    pub out_proj: SingleFp8GemmWithQuantConfig,
    pub post_attention_add_rms_norm: ResidualRmsNormKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Qwen36GatedGqaLocalWorkletInput {
    pub batch_tokens: u32,
    pub prefill_chunk_pairs: Vec<(u32, u32)>,
    pub decode_kv_lens: Vec<u32>,
}

pub struct Qwen36GatedGqaLocalWorklet {
    pub name: String,
    pub input_add_rms_norm: Op<ResidualRmsNormKernel>,
    pub qkv_gate: SingleFp8GemmWithQuantOp,
    pub q_norm: Op<RmsNormKernel>,
    pub k_norm: Op<RmsNormKernel>,
    pub partial_rope: Op<ElementwiseKernel>,
    pub attention: FlashInferAttentionOp,
    pub output_gate: Op<ElementwiseKernel>,
    pub out_proj: SingleFp8GemmWithQuantOp,
    pub post_attention_add_rms_norm: Op<ResidualRmsNormKernel>,
    resolved: Qwen36GatedGqaLocalWorkletResolved,
}

impl Qwen36GatedGqaLocalWorklet {
    pub fn resolve_config(
        cfg: &Qwen36GatedGqaLocalWorkletConfig,
    ) -> Qwen36GatedGqaLocalWorkletResolved {
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid Qwen36GatedGqaLocalWorkletConfig: {reason}"));

        let quant = |hidden_size: Dim| Fp8PerTokenGroupQuantKernelConfig {
            backends: cfg.fp8_quant_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden_size,
            group_size: FP8_GROUP_SIZE,
            input_dtype: cfg.activation_dtype,
            scale_format: SCALE_FORMAT.to_string(),
        };
        let residual_norm = || ResidualRmsNormKernelConfig {
            backends: cfg.residual_rms_norm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden: cfg.hidden.clone(),
            dtype: cfg.activation_dtype,
        };
        let qk_norm = || RmsNormKernelConfig {
            backends: cfg.qk_rms_norm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden: cfg.head_dim.clone(),
            dtype: cfg.activation_dtype,
        };
        let elementwise = |input_bytes: u32, output_bytes: u32| ElementwiseKernelConfig {
            backends: cfg.elementwise_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            input_bytes_per_token: input_bytes.into(),
            output_bytes_per_token: output_bytes.into(),
        };

        Qwen36GatedGqaLocalWorkletResolved {
            input_add_rms_norm: residual_norm(),
            qkv_gate: SingleFp8GemmWithQuantConfig {
                quant: quant(cfg.hidden.clone()),
                gemm: SingleGemmKernelConfig {
                    backends: cfg.fp8_gemm_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    n: QKV_GATE_WIDTH.into(),
                    k: cfg.hidden.clone(),
                    dtype: DType::Fp8E4m3,
                },
            },
            q_norm: qk_norm(),
            k_norm: qk_norm(),
            partial_rope: elementwise(
                PARTIAL_ROPE_BYTES_PER_TOKEN,
                PARTIAL_ROPE_BYTES_PER_TOKEN,
            ),
            attention: FlashInferAttentionConfig {
                backends: cfg.attention_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_qo_heads: cfg.num_qo_heads.clone(),
                num_kv_heads: cfg.num_kv_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                dtype: cfg.activation_dtype,
                fp8: false,
                kv_cache_append_backends: cfg.kv_cache_append_backends.clone(),
                kv_cache_block_size: cfg.kv_cache_block_size,
                kv_cache_layout: cfg.kv_cache_layout.clone(),
                kv_scale_granularity: cfg.kv_scale_granularity.clone(),
            },
            output_gate: elementwise(
                OUTPUT_GATE_INPUT_BYTES_PER_TOKEN,
                OUTPUT_GATE_OUTPUT_BYTES_PER_TOKEN,
            ),
            out_proj: SingleFp8GemmWithQuantConfig {
                quant: quant(Q_WIDTH.into()),
                gemm: SingleGemmKernelConfig {
                    backends: cfg.fp8_gemm_backends.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    n: cfg.hidden.clone(),
                    k: Q_WIDTH.into(),
                    dtype: DType::Fp8E4m3,
                },
            },
            post_attention_add_rms_norm: residual_norm(),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Qwen36GatedGqaLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let input_add_rms_norm = build_atomic(
            &name,
            "input_add_rms_norm",
            resolved.input_add_rms_norm.clone(),
            ResidualRmsNormKernel::build,
            bridge,
        )?;
        let qkv_gate = SingleFp8GemmWithQuantOp::build(
            format!("{name}.qkv_gate"),
            resolved.qkv_gate.clone(),
            bridge,
        )?;
        let q_norm = build_atomic(
            &name,
            "q_norm",
            resolved.q_norm.clone(),
            RmsNormKernel::build,
            bridge,
        )?;
        let k_norm = build_atomic(
            &name,
            "k_norm",
            resolved.k_norm.clone(),
            RmsNormKernel::build,
            bridge,
        )?;
        let partial_rope = build_atomic(
            &name,
            "partial_rope",
            resolved.partial_rope.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        let attention = FlashInferAttentionOp::build(
            format!("{name}.attention"),
            resolved.attention.clone(),
            bridge,
        )?;
        let output_gate = build_atomic(
            &name,
            "output_gate",
            resolved.output_gate.clone(),
            ElementwiseKernel::build,
            bridge,
        )?;
        let out_proj = SingleFp8GemmWithQuantOp::build(
            format!("{name}.out_proj"),
            resolved.out_proj.clone(),
            bridge,
        )?;
        let post_attention_add_rms_norm = build_atomic(
            &name,
            "post_attention_add_rms_norm",
            resolved.post_attention_add_rms_norm.clone(),
            ResidualRmsNormKernel::build,
            bridge,
        )?;

        Ok(Self {
            name,
            input_add_rms_norm,
            qkv_gate,
            q_norm,
            k_norm,
            partial_rope,
            attention,
            output_gate,
            out_proj,
            post_attention_add_rms_norm,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (Qwen36GatedGqaLocalWorklet) [local (1 GPU); Hq={}, Hkv={}, D={}, rope={}]",
                self.name, cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rope_dim,
            ),
            child: Box::new(CostNode::Sum(vec![
                self.input_add_rms_norm.compile(builder),
                self.qkv_gate.compile(builder),
                self.q_norm.compile(builder),
                self.k_norm.compile(builder),
                self.partial_rope.compile(builder),
                self.attention.compile(builder),
                self.output_gate.compile(builder),
                self.out_proj.compile(builder),
                self.post_attention_add_rms_norm.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Qwen36GatedGqaLocalWorkletInput, ev: &mut Evaluator) {
        let work = derive_work(input).unwrap_or_else(|reason| {
            panic!("invalid Qwen36GatedGqaLocalWorkletInput: {reason}")
        });
        self.input_add_rms_norm.eval(&work.input_add_rms_norm, ev);
        self.qkv_gate.eval(&work.qkv_gate, ev);
        self.q_norm.eval(&work.q_norm, ev);
        self.k_norm.eval(&work.k_norm, ev);
        self.partial_rope.eval(&work.partial_rope, ev);
        self.attention.eval(&work.attention, ev);
        self.output_gate.eval(&work.output_gate, ev);
        self.out_proj.eval(&work.out_proj, ev);
        self.post_attention_add_rms_norm
            .eval(&work.post_attention_add_rms_norm, ev);
    }
}

struct WorkInputs {
    input_add_rms_norm: ResidualRmsNormKernelInput,
    qkv_gate: SingleFp8GemmWithQuantInput,
    q_norm: RmsNormKernelInput,
    k_norm: RmsNormKernelInput,
    partial_rope: ElementwiseKernelInput,
    attention: FlashInferAttentionInput,
    output_gate: ElementwiseKernelInput,
    out_proj: SingleFp8GemmWithQuantInput,
    post_attention_add_rms_norm: ResidualRmsNormKernelInput,
}

fn derive_work(input: &Qwen36GatedGqaLocalWorkletInput) -> Result<WorkInputs, String> {
    if input.batch_tokens == 0 {
        return Err("batch_tokens must be positive".into());
    }

    let mut prefill_tokens = 0_u64;
    for &(prefix_len, append_len) in &input.prefill_chunk_pairs {
        if append_len == 0 {
            return Err("every prefill append length must be positive".into());
        }
        prefix_len
            .checked_add(append_len)
            .ok_or("prefill KV length overflow")?;
        prefill_tokens = prefill_tokens
            .checked_add(u64::from(append_len))
            .ok_or("prefill token sum overflow")?;
    }
    let mut decode_total_kv = 0_u64;
    for &kv_len in &input.decode_kv_lens {
        if kv_len == 0 {
            return Err("every decode KV length must be positive".into());
        }
        decode_total_kv = decode_total_kv
            .checked_add(u64::from(kv_len))
            .ok_or("decode KV length sum overflow")?;
    }
    u32::try_from(decode_total_kv).map_err(|_| "decode KV length sum exceeds u32")?;
    let decode_tokens = u64::try_from(input.decode_kv_lens.len())
        .map_err(|_| "decode request count exceeds u64")?;
    let expected_tokens = prefill_tokens
        .checked_add(decode_tokens)
        .ok_or("batch token accounting overflow")?;
    if expected_tokens != u64::from(input.batch_tokens) {
        return Err(format!(
            "batch_tokens {} does not equal prefill appends + decode count {}",
            input.batch_tokens, expected_tokens
        ));
    }

    let q_rows = input
        .batch_tokens
        .checked_mul(NUM_QO_HEADS)
        .ok_or("Q RMSNorm row count overflow")?;
    let k_rows = input
        .batch_tokens
        .checked_mul(NUM_KV_HEADS)
        .ok_or("K RMSNorm row count overflow")?;

    Ok(WorkInputs {
        input_add_rms_norm: ResidualRmsNormKernelInput {
            m: input.batch_tokens,
        },
        qkv_gate: SingleFp8GemmWithQuantInput {
            num_tokens: input.batch_tokens,
        },
        q_norm: RmsNormKernelInput { m: q_rows },
        k_norm: RmsNormKernelInput { m: k_rows },
        partial_rope: ElementwiseKernelInput {
            num_tokens: input.batch_tokens,
        },
        attention: FlashInferAttentionInput {
            prefill_chunk_pairs: input.prefill_chunk_pairs.clone(),
            decode_kv_lens: input.decode_kv_lens.clone(),
        },
        output_gate: ElementwiseKernelInput {
            num_tokens: input.batch_tokens,
        },
        out_proj: SingleFp8GemmWithQuantInput {
            num_tokens: input.batch_tokens,
        },
        post_attention_add_rms_norm: ResidualRmsNormKernelInput {
            m: input.batch_tokens,
        },
    })
}

fn validate_config(cfg: &Qwen36GatedGqaLocalWorkletConfig) -> Result<(), String> {
    for (name, actual, expected) in [
        ("hidden", cfg.hidden.get(), HIDDEN),
        ("num_qo_heads", cfg.num_qo_heads.get(), NUM_QO_HEADS),
        ("num_kv_heads", cfg.num_kv_heads.get(), NUM_KV_HEADS),
        ("head_dim", cfg.head_dim.get(), HEAD_DIM),
        ("rope_dim", cfg.rope_dim.get(), ROPE_DIM),
    ] {
        if actual != expected {
            return Err(format!("{name} must be {expected}, got {actual}"));
        }
    }
    if cfg.activation_dtype != DType::Bf16 {
        return Err("activation_dtype must be BF16".into());
    }
    if cfg.kv_cache_block_size == 0 {
        return Err("kv_cache_block_size must be positive".into());
    }
    if cfg.kv_cache_layout.is_empty() {
        return Err("kv_cache_layout must not be empty".into());
    }
    if cfg.kv_scale_granularity.is_empty() {
        return Err("kv_scale_granularity must not be empty".into());
    }
    Ok(())
}

fn build_atomic<K, C, F>(
    parent: &str,
    suffix: &str,
    config: C,
    build: F,
    bridge: &PerfApiBridge,
) -> Result<Op<K>, BuildError>
where
    K: crate::timing::Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError>,
{
    let name = format!("{parent}.{suffix}");
    Ok(Op::new(
        name.clone(),
        Arc::new(build(name, config, bridge)?),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::{CostTreeBuilder, PerfApiBridge};

    fn cfg() -> Qwen36GatedGqaLocalWorkletConfig {
        Qwen36GatedGqaLocalWorkletConfig {
            hidden: HIDDEN.into(),
            num_qo_heads: NUM_QO_HEADS.into(),
            num_kv_heads: NUM_KV_HEADS.into(),
            head_dim: HEAD_DIM.into(),
            rope_dim: ROPE_DIM.into(),
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA H200".into(),
            kv_cache_block_size: 16,
            kv_cache_layout: "NHD".into(),
            kv_scale_granularity: "tensor".into(),
            residual_rms_norm_backends: vec!["vllm_cuda"],
            fp8_quant_backends: vec!["vllm_cuda"],
            fp8_gemm_backends: vec!["deepgemm"],
            qk_rms_norm_backends: vec!["vllm_cuda"],
            elementwise_backends: vec!["triton"],
            attention_backends: vec!["fa2", "fa3"],
            kv_cache_append_backends: vec!["vllm_cuda"],
        }
    }

    fn enumerate_worklet() -> Qwen36GatedGqaLocalWorklet {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        Qwen36GatedGqaLocalWorklet::build(
            "model.gated_gqa".into(),
            Qwen36GatedGqaLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap()
    }

    #[test]
    fn resolution_freezes_qwen_identity_projection_partitions_and_precision() {
        assert_eq!(Q_WIDTH, 4096);
        assert_eq!(OUTPUT_GATE_WIDTH, 4096);
        assert_eq!(K_WIDTH, 512);
        assert_eq!(V_WIDTH, 512);
        assert_eq!(QKV_GATE_WIDTH, 9216);

        let resolved = Qwen36GatedGqaLocalWorklet::resolve_config(&cfg());
        assert_eq!(
            (
                resolved.qkv_gate.quant.hidden_size.get(),
                resolved.qkv_gate.gemm.k.get(),
                resolved.qkv_gate.gemm.n.get(),
            ),
            (2048, 2048, 9216)
        );
        assert_eq!(resolved.qkv_gate.quant.group_size, 128);
        assert_eq!(resolved.qkv_gate.quant.input_dtype, DType::Bf16);
        assert_eq!(resolved.qkv_gate.quant.scale_format, SCALE_FORMAT);
        assert_eq!(resolved.qkv_gate.gemm.dtype, DType::Fp8E4m3);
        assert_eq!(
            (
                resolved.out_proj.quant.hidden_size.get(),
                resolved.out_proj.gemm.k.get(),
                resolved.out_proj.gemm.n.get(),
            ),
            (4096, 4096, 2048)
        );
        assert_eq!(resolved.out_proj.gemm.dtype, DType::Fp8E4m3);
        assert_eq!(resolved.attention.dtype, DType::Bf16);
        assert!(!resolved.attention.fp8);
        assert_eq!(resolved.attention.kv_dtype(), DType::Bf16);
    }

    #[test]
    fn resolution_freezes_norms_attention_cache_and_byte_rates() {
        let resolved = Qwen36GatedGqaLocalWorklet::resolve_config(&cfg());
        assert_eq!(resolved.input_add_rms_norm.hidden.get(), 2048);
        assert_eq!(resolved.post_attention_add_rms_norm.hidden.get(), 2048);
        assert_eq!(resolved.q_norm.hidden.get(), 256);
        assert_eq!(resolved.k_norm.hidden.get(), 256);
        assert_eq!(
            (
                resolved.partial_rope.input_bytes_per_token.get(),
                resolved.partial_rope.output_bytes_per_token.get(),
            ),
            (2304, 2304)
        );
        assert_eq!(
            (
                resolved.output_gate.input_bytes_per_token.get(),
                resolved.output_gate.output_bytes_per_token.get(),
            ),
            (16384, 8192)
        );
        assert_eq!(
            (
                resolved.attention.num_qo_heads.get(),
                resolved.attention.num_kv_heads.get(),
                resolved.attention.head_dim.get(),
            ),
            (16, 2, 256)
        );
        assert_eq!(resolved.attention.kv_cache_block_size, 16);
        assert_eq!(resolved.attention.kv_cache_layout, "NHD");
        assert_eq!(resolved.attention.kv_scale_granularity, "tensor");
    }

    #[test]
    fn invalid_qwen_identity_dtype_and_cache_contract_are_rejected() {
        for mutate in [
            |cfg: &mut Qwen36GatedGqaLocalWorkletConfig| cfg.hidden = 4096.into(),
            |cfg: &mut Qwen36GatedGqaLocalWorkletConfig| cfg.num_qo_heads = 32.into(),
            |cfg: &mut Qwen36GatedGqaLocalWorkletConfig| cfg.num_kv_heads = 4.into(),
            |cfg: &mut Qwen36GatedGqaLocalWorkletConfig| cfg.head_dim = 128.into(),
            |cfg: &mut Qwen36GatedGqaLocalWorkletConfig| cfg.rope_dim = 128.into(),
        ] {
            let mut bad = cfg();
            mutate(&mut bad);
            assert!(std::panic::catch_unwind(|| {
                Qwen36GatedGqaLocalWorklet::resolve_config(&bad)
            })
            .is_err());
        }
        let mut bad = cfg();
        bad.activation_dtype = DType::Fp16;
        assert!(std::panic::catch_unwind(|| {
            Qwen36GatedGqaLocalWorklet::resolve_config(&bad)
        })
        .is_err());
        let mut bad = cfg();
        bad.kv_cache_block_size = 0;
        assert!(std::panic::catch_unwind(|| {
            Qwen36GatedGqaLocalWorklet::resolve_config(&bad)
        })
        .is_err());
    }

    #[test]
    fn token_accounting_and_norm_rows_cover_prefill_decode_and_mixed() {
        let prefill = derive_work(&Qwen36GatedGqaLocalWorkletInput {
            batch_tokens: 5,
            prefill_chunk_pairs: vec![(0, 3), (20, 2)],
            decode_kv_lens: vec![],
        })
        .unwrap();
        assert_eq!((prefill.q_norm.m, prefill.k_norm.m), (80, 10));
        assert_eq!(prefill.attention.prefill_chunk_pairs, [(0, 3), (20, 2)]);

        let decode = derive_work(&Qwen36GatedGqaLocalWorkletInput {
            batch_tokens: 3,
            prefill_chunk_pairs: vec![],
            decode_kv_lens: vec![1, 64, 129],
        })
        .unwrap();
        assert_eq!((decode.q_norm.m, decode.k_norm.m), (48, 6));
        assert_eq!(decode.attention.decode_kv_lens, [1, 64, 129]);

        let mixed = derive_work(&Qwen36GatedGqaLocalWorkletInput {
            batch_tokens: 6,
            prefill_chunk_pairs: vec![(7, 4)],
            decode_kv_lens: vec![8, 9],
        })
        .unwrap();
        assert_eq!((mixed.q_norm.m, mixed.k_norm.m), (96, 12));
        assert_eq!(mixed.qkv_gate.num_tokens, 6);
        assert_eq!(mixed.output_gate.num_tokens, 6);
        assert_eq!(mixed.out_proj.num_tokens, 6);
    }

    #[test]
    fn token_accounting_rejects_empty_zero_lengths_mismatch_and_overflow() {
        assert!(derive_work(&Qwen36GatedGqaLocalWorkletInput::default()).is_err());
        assert!(derive_work(&Qwen36GatedGqaLocalWorkletInput {
            batch_tokens: 1,
            prefill_chunk_pairs: vec![(0, 0)],
            decode_kv_lens: vec![1],
        })
        .is_err());
        assert!(derive_work(&Qwen36GatedGqaLocalWorkletInput {
            batch_tokens: 1,
            prefill_chunk_pairs: vec![],
            decode_kv_lens: vec![0],
        })
        .is_err());
        assert!(derive_work(&Qwen36GatedGqaLocalWorkletInput {
            batch_tokens: 2,
            prefill_chunk_pairs: vec![(0, 1)],
            decode_kv_lens: vec![],
        })
        .is_err());
        assert!(derive_work(&Qwen36GatedGqaLocalWorkletInput {
            batch_tokens: u32::MAX,
            prefill_chunk_pairs: vec![(u32::MAX, 1)],
            decode_kv_lens: vec![],
        })
        .is_err());
        assert!(derive_work(&Qwen36GatedGqaLocalWorkletInput {
            batch_tokens: u32::MAX,
            prefill_chunk_pairs: vec![(0, u32::MAX)],
            decode_kv_lens: vec![],
        })
        .is_err());
    }

    #[test]
    fn compile_has_exact_nine_children_and_thirteen_flattened_leaves() {
        let worklet = enumerate_worklet();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        let expected = [
            "input_add_rms_norm",
            "qkv_gate.input_quant",
            "qkv_gate.gemm",
            "q_norm",
            "k_norm",
            "partial_rope",
            "attention.kv_cache_append",
            "attention.prefill",
            "attention.decode",
            "output_gate",
            "out_proj.input_quant",
            "out_proj.gemm",
            "post_attention_add_rms_norm",
        ];
        assert_eq!(tree.n_slots(), 13);
        assert_eq!(
            tree.slots
                .iter()
                .map(|slot| slot.name.strip_prefix("model.gated_gqa.").unwrap())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(tree.slots[0].kind, "residual_rms_norm");
        assert_eq!(tree.slots[3].kind, "rms_norm");
        assert_eq!(tree.slots[5].kind, "elementwise");
        assert_eq!(tree.slots[6].kind, "kv_cache_append");
        assert_eq!(tree.slots[7].kind, "flashinfer_attn_prefill");
        assert_eq!(tree.slots[8].kind, "flashinfer_attn_decode");
        assert_eq!(tree.slots[12].kind, "residual_rms_norm");
        assert!(!tree.slots.iter().any(|slot| {
            matches!(
                slot.kind.as_str(),
                "all_reduce" | "all_to_all" | "send_recv"
            )
        }));
        match tree.root {
            CostNode::Labeled { child, .. } => match *child {
                CostNode::Sum(children) => assert_eq!(children.len(), 9),
                _ => panic!("expected Sum"),
            },
            _ => panic!("expected Labeled"),
        }
    }

    #[test]
    fn compiled_children_preserve_backend_roles_and_non_fp8_attention() {
        let worklet = enumerate_worklet();
        assert_eq!(worklet.input_add_rms_norm.kernel.config.backends, ["vllm_cuda"]);
        assert_eq!(worklet.q_norm.kernel.config.backends, ["vllm_cuda"]);
        assert_eq!(worklet.partial_rope.kernel.config.backends, ["triton"]);
        assert_eq!(worklet.attention.prefill.config.backends, ["fa2", "fa3"]);
        assert_eq!(worklet.attention.prefill.config.q_dtype, DType::Bf16);
        assert_eq!(worklet.attention.prefill.config.kv_dtype, DType::Bf16);
        assert_eq!(worklet.attention.decode.config.kv_dtype, DType::Bf16);
        assert_eq!(
            worklet.attention.kv_cache_append.config.kv_dtype,
            DType::Bf16
        );
    }

    #[test]
    fn flashinfer_compile_topology_is_fixed_when_prefill_or_decode_is_absent() {
        let worklet = enumerate_worklet();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        assert_eq!(tree.n_slots(), 13);

        for input in [
            Qwen36GatedGqaLocalWorkletInput {
                batch_tokens: 2,
                prefill_chunk_pairs: vec![(0, 2)],
                decode_kv_lens: vec![],
            },
            Qwen36GatedGqaLocalWorkletInput {
                batch_tokens: 2,
                prefill_chunk_pairs: vec![],
                decode_kv_lens: vec![8, 9],
            },
        ] {
            let work = derive_work(&input).unwrap();
            assert_eq!(work.attention.prefill_chunk_pairs.is_empty(), input.prefill_chunk_pairs.is_empty());
            assert_eq!(work.attention.decode_kv_lens.is_empty(), input.decode_kv_lens.is_empty());
        }
    }
}
