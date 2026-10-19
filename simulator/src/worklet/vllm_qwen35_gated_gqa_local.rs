//! One Qwen3.5-family gated full-attention mixer on one GPU, in vLLM's launch
//! granularity (`qwen3_5.py` attention, ModelOpt FP8 per-tensor projections,
//! FP8 KV cache, FlashInfer trtllm-gen attention).
//!
//! The section starts after the input RMSNorm (the arch owns it; it already
//! wrote the FP8 copy `qkv_proj` reads) and ends at the `o_proj` output.
//!
//! ```text
//! compiled piece   qkv_proj GEMM (q with its gate half, k, v)
//!                  fused q/k RMSNorm + partial (m)RoPE + gate split
//!                  query FP8 quant
//! attention        reshape_and_cache_flash (FP8 KV), trtllm-gen context
//!                  attention of the prefill requests, trtllm-gen decode
//!                  attention of the decode requests
//! compiled piece   sigmoid gate x output, FP8 quant; o_proj GEMM
//! ```
//!
//! The compiled pieces run at the graph-padded row count. Attention is a
//! splitting op: a piecewise or eager step runs it over the real requests; a
//! FULL-graph (uniform decode) replay launches decode attention over the
//! padded request count. The KV append writes the real tokens
//! (`slot_mapping`). The two `fmha` workspace memsets (about 1.2 us each) are
//! not modeled.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    KvCacheAppendKernel, KvCacheAppendKernelConfig, KvCacheAppendKernelInput,
    PagedAttnBatchDecodeKernel, PagedAttnBatchDecodeKernelConfig, PagedAttnBatchDecodeKernelInput,
    PagedAttnBatchPrefillKernel, PagedAttnBatchPrefillKernelConfig,
    PagedAttnBatchPrefillKernelInput, QkRmsNormRopeGateKernel, QkRmsNormRopeGateKernelConfig,
    QkRmsNormRopeGateKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
    StaticFp8QuantKernel, StaticFp8QuantKernelConfig, StaticFp8QuantKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, push_or_zero};

/// vLLM's KV cache tensor layout for FlashInfer.
const CACHE_LAYOUT: &str = "NHD";
/// One static scale for the whole FP8 KV cache.
const KV_SCALE_GRANULARITY: &str = "tensor";

#[derive(Clone, Debug)]
pub struct VllmQwen35GatedGqaLocalWorkletConfig {
    pub hidden: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub mrope: bool,
    /// Tokens per KV page the append writes into.
    pub kv_block_size: u32,
    pub activation_dtype: DType,
    pub kv_dtype: DType,
    pub proj_dtype: DType,
    pub gpu_name: String,
    pub proj_gemm_backends: Vec<&'static str>,
    pub qk_norm_rope_backends: Vec<&'static str>,
    pub static_quant_backends: Vec<&'static str>,
    pub kv_append_backends: Vec<&'static str>,
    pub attention_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct VllmQwen35GatedGqaLocalWorkletResolved {
    pub raw_cfg: VllmQwen35GatedGqaLocalWorkletConfig,
    pub qkv_proj: SingleGemmKernelConfig,
    pub qk_norm_rope_gate: QkRmsNormRopeGateKernelConfig,
    pub query_quant: StaticFp8QuantKernelConfig,
    pub kv_append: KvCacheAppendKernelConfig,
    pub prefill_attention: PagedAttnBatchPrefillKernelConfig,
    pub decode_attention: PagedAttnBatchDecodeKernelConfig,
    pub gate_quant: StaticFp8QuantKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
}

/// One step as the attention layer sees it.
#[derive(Clone, Debug, Default)]
pub struct VllmQwen35GatedGqaLocalWorkletInput {
    /// Rows the compiled pieces run: the graph-padded token count.
    pub padded_tokens: u32,
    /// The request count a FULL-graph decode replay launches attention over,
    /// or 0 when attention runs over the real requests.
    pub full_graph_rows: u32,
    /// `(cached prefix, new tokens)` of each prefill request.
    pub prefill_requests: Vec<(u32, u32)>,
    /// Context length (cached tokens plus the one being decoded) of each
    /// decode request.
    pub decode_kv_lens: Vec<u32>,
}

pub struct VllmQwen35GatedGqaLocalWorklet {
    pub name: String,
    qkv_proj: Op<SingleGemmKernel>,
    qk_norm_rope_gate: Op<QkRmsNormRopeGateKernel>,
    query_quant: Op<StaticFp8QuantKernel>,
    kv_append: Op<KvCacheAppendKernel>,
    prefill_attention: Op<PagedAttnBatchPrefillKernel>,
    decode_attention: Op<PagedAttnBatchDecodeKernel>,
    gate_quant: Op<StaticFp8QuantKernel>,
    o_proj: Op<SingleGemmKernel>,
    resolved: VllmQwen35GatedGqaLocalWorkletResolved,
}

impl VllmQwen35GatedGqaLocalWorklet {
    pub fn resolve_config(
        cfg: &VllmQwen35GatedGqaLocalWorkletConfig,
    ) -> VllmQwen35GatedGqaLocalWorkletResolved {
        let gpu = &cfg.gpu_name;
        let q_width = cfg.num_q_heads * cfg.head_dim;
        let gemm = |n: u32, k: u32| SingleGemmKernelConfig {
            backends: cfg.proj_gemm_backends.clone(),
            gpu_name: gpu.clone(),
            n: n.into(),
            k: k.into(),
            dtype: cfg.proj_dtype,
        };
        let quant = |gated: bool| StaticFp8QuantKernelConfig {
            backends: cfg.static_quant_backends.clone(),
            gpu_name: gpu.clone(),
            hidden_dim: q_width.into(),
            gated,
            dtype: cfg.activation_dtype,
        };
        // The query runs FP8 to match the FP8 KV cache.
        let q_dtype = cfg.kv_dtype;
        VllmQwen35GatedGqaLocalWorkletResolved {
            // q and its gate half, then k and v.
            qkv_proj: gemm(
                2 * q_width + 2 * cfg.num_kv_heads * cfg.head_dim,
                cfg.hidden,
            ),
            qk_norm_rope_gate: QkRmsNormRopeGateKernelConfig {
                backends: cfg.qk_norm_rope_backends.clone(),
                gpu_name: gpu.clone(),
                num_q_heads: cfg.num_q_heads.into(),
                num_kv_heads: cfg.num_kv_heads.into(),
                head_dim: cfg.head_dim.into(),
                rotary_dim: cfg.rotary_dim.into(),
                mrope: cfg.mrope,
                dtype: cfg.activation_dtype,
            },
            query_quant: quant(false),
            kv_append: KvCacheAppendKernelConfig {
                backends: cfg.kv_append_backends.clone(),
                gpu_name: gpu.clone(),
                num_kv_heads: cfg.num_kv_heads.into(),
                head_dim: cfg.head_dim.into(),
                head_dim_v: None,
                block_size: cfg.kv_block_size,
                input_dtype: cfg.activation_dtype,
                kv_dtype: cfg.kv_dtype,
                cache_layout: CACHE_LAYOUT.to_string(),
                scale_granularity: KV_SCALE_GRANULARITY.to_string(),
            },
            prefill_attention: PagedAttnBatchPrefillKernelConfig {
                backends: cfg.attention_backends.clone(),
                gpu_name: gpu.clone(),
                num_qo_heads: cfg.num_q_heads.into(),
                num_kv_heads: cfg.num_kv_heads.into(),
                head_dim: cfg.head_dim.into(),
                q_dtype,
                kv_dtype: cfg.kv_dtype,
                o_dtype: cfg.activation_dtype,
            },
            decode_attention: PagedAttnBatchDecodeKernelConfig {
                backends: cfg.attention_backends.clone(),
                gpu_name: gpu.clone(),
                num_qo_heads: cfg.num_q_heads.into(),
                num_kv_heads: cfg.num_kv_heads.into(),
                head_dim: cfg.head_dim.into(),
                q_dtype,
                kv_dtype: cfg.kv_dtype,
                o_dtype: cfg.activation_dtype,
            },
            gate_quant: quant(true),
            o_proj: gemm(cfg.hidden, q_width),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: VllmQwen35GatedGqaLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            qkv_proj: atomic(n, "qkv_proj", r.qkv_proj, SingleGemmKernel::build, bridge)?,
            qk_norm_rope_gate: atomic(
                n,
                "qk_norm_rope_gate",
                r.qk_norm_rope_gate,
                QkRmsNormRopeGateKernel::build,
                bridge,
            )?,
            query_quant: atomic(
                n,
                "query_quant",
                r.query_quant,
                StaticFp8QuantKernel::build,
                bridge,
            )?,
            kv_append: atomic(
                n,
                "kv_append",
                r.kv_append,
                KvCacheAppendKernel::build,
                bridge,
            )?,
            prefill_attention: atomic(
                n,
                "prefill_attention",
                r.prefill_attention,
                PagedAttnBatchPrefillKernel::build,
                bridge,
            )?,
            decode_attention: atomic(
                n,
                "decode_attention",
                r.decode_attention,
                PagedAttnBatchDecodeKernel::build,
                bridge,
            )?,
            gate_quant: atomic(
                n,
                "gate_quant",
                r.gate_quant,
                StaticFp8QuantKernel::build,
                bridge,
            )?,
            o_proj: atomic(n, "o_proj", r.o_proj, SingleGemmKernel::build, bridge)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (VllmQwen35GatedGqaLocalWorklet) [local (1 GPU); Hq={}, Hkv={}, D={}]",
                self.name, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim
            ),
            child: Box::new(CostNode::Sum(vec![
                self.qkv_proj.compile(builder),
                self.qk_norm_rope_gate.compile(builder),
                self.query_quant.compile(builder),
                self.kv_append.compile(builder),
                self.prefill_attention.compile(builder),
                self.decode_attention.compile(builder),
                self.gate_quant.compile(builder),
                self.o_proj.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &VllmQwen35GatedGqaLocalWorkletInput, ev: &mut Evaluator) {
        let padded = input.padded_tokens;
        let prefill_tokens: u32 = input.prefill_requests.iter().map(|&(_, new)| new).sum();
        let decodes = input.decode_kv_lens.len() as u32;
        let rows = SingleGemmKernelInput { m: padded };
        let quant = StaticFp8QuantKernelInput { num_tokens: padded };
        push_or_zero(&self.qkv_proj, rows.clone(), false, ev);
        push_or_zero(
            &self.qk_norm_rope_gate,
            QkRmsNormRopeGateKernelInput { num_tokens: padded },
            false,
            ev,
        );
        push_or_zero(&self.query_quant, quant.clone(), false, ev);
        push_or_zero(
            &self.kv_append,
            KvCacheAppendKernelInput {
                num_tokens: prefill_tokens + decodes,
            },
            false,
            ev,
        );
        push_or_zero(
            &self.prefill_attention,
            PagedAttnBatchPrefillKernelInput {
                requests: input.prefill_requests.clone(),
            },
            input.prefill_requests.is_empty(),
            ev,
        );
        let batch_size = if input.full_graph_rows > 0 {
            input.full_graph_rows
        } else {
            decodes
        };
        push_or_zero(
            &self.decode_attention,
            PagedAttnBatchDecodeKernelInput {
                batch_size,
                max_kv_len: input.decode_kv_lens.iter().copied().max().unwrap_or(0),
                total_kv_len: input.decode_kv_lens.iter().map(|&l| u64::from(l)).sum(),
            },
            decodes == 0,
            ev,
        );
        push_or_zero(&self.gate_quant, quant, false, ev);
        push_or_zero(&self.o_proj, rows, false, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_are_the_profiled_qwen38_rows() {
        let r =
            VllmQwen35GatedGqaLocalWorklet::resolve_config(&VllmQwen35GatedGqaLocalWorkletConfig {
                hidden: 5120,
                num_q_heads: 24,
                num_kv_heads: 4,
                head_dim: 256,
                rotary_dim: 64,
                mrope: true,
                kv_block_size: 16,
                activation_dtype: DType::Bf16,
                kv_dtype: DType::Fp8E4m3,
                proj_dtype: DType::Fp8E4m3,
                gpu_name: "NVIDIA B200".into(),
                proj_gemm_backends: vec!["flashinfer_fp8_tensor"],
                qk_norm_rope_backends: vec!["vllm_triton"],
                static_quant_backends: vec!["vllm_inductor"],
                kv_append_backends: vec!["vllm_cuda"],
                attention_backends: vec!["flashinfer_trtllm_gen"],
            });
        assert_eq!((r.qkv_proj.n.get(), r.qkv_proj.k.get()), (14336, 5120));
        assert_eq!((r.o_proj.n.get(), r.o_proj.k.get()), (5120, 6144));
        assert_eq!(r.query_quant.hidden_dim.get(), 6144);
        assert!(r.gate_quant.gated && !r.query_quant.gated);
        assert_eq!(r.decode_attention.q_dtype, DType::Fp8E4m3);
    }
}
