//! One sparse-GQA (QSA) attention mixer on one GPU, in vLLM's launch
//! granularity: a gated GQA whose rows attend only the token groups a
//! weight-free indexer selected (`models/qwen4_exp/nvidia/qsa.py`; projection,
//! KV and index-key dtypes from the config).
//!
//! The section starts after the arch's pre-attention norm and ends at the
//! `o_proj` output.
//!
//! ```text
//! graph piece   qkv_proj GEMM (q with its gate half, k, v)
//!               fused q/k RMSNorm + partial RoPE + gate split
//!               index_qk_proj GEMM (indexer q heads + one key head)
//! _run_qsa      indexer: prepare, decode / prefill logits, top-k, expand
//!               attention: KV append, output memset, sparse paged attention
//! graph piece   o_proj GEMM
//! ```
//!
//! The graph pieces run at the graph-padded row count. `_run_qsa` is an eager
//! break: under a piecewise or eager step it runs over the real rows (the
//! memset still clears the whole padded output tensor). A FULL-graph replay,
//! which only a uniform verify step takes, runs everything at the padded rows,
//! with the padded rows forming whole verify requests (UNIFORM_BATCH metadata).
//! Decode, mixed and prefill steps all go through the same two compound ops; a
//! mixed step runs both the decode and the prefill selection.

use crate::op::attention::{
    ring_tokens, SparseGqaAttentionOp, SparseGqaAttentionOpConfig, SparseGqaAttentionOpInput,
    SparseGqaAttentionOpResolved, SparseGqaIndexerOp, SparseGqaIndexerOpConfig,
    SparseGqaIndexerOpInput, SparseGqaIndexerOpResolved,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    QkRmsNormRopeGateKernel, QkRmsNormRopeGateKernelConfig, QkRmsNormRopeGateKernelInput,
    SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, push_or_zero};

#[derive(Clone, Debug)]
pub struct VllmQsaAttnLocalWorkletConfig {
    pub hidden: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub mrope: bool,
    pub indexer_num_heads: u32,
    pub indexer_num_kv_heads: u32,
    pub indexer_head_dim: u32,
    pub indexer_compress_ratio: u32,
    /// Tokens selected per row (`indexer_budget`).
    pub indexer_token_topk: u32,
    pub max_position_embeddings: u32,
    /// Deployment `max_model_len`; it sets the padded decode logits row.
    pub max_model_len: u32,
    /// Tokens per KV page.
    pub kv_block_size: u32,
    pub num_speculative_tokens: u32,
    /// Query rows of every decode request this instance prices: `1 +
    /// num_speculative_tokens` for a verify step.
    pub decode_query_len: u32,
    pub activation_dtype: DType,
    pub proj_dtype: DType,
    pub kv_dtype: DType,
    pub indexer_kv_dtype: DType,
    pub gpu_name: String,
    pub proj_gemm_backends: Vec<&'static str>,
    pub qk_norm_rope_backends: Vec<&'static str>,
    pub indexer_prepare_backends: Vec<&'static str>,
    pub indexer_logits_backends: Vec<&'static str>,
    pub topk_backends: Vec<&'static str>,
    /// Empty where the platform has no cooperative top-k.
    pub cooperative_topk_backends: Vec<&'static str>,
    pub index_expand_backends: Vec<&'static str>,
    pub kv_append_backends: Vec<&'static str>,
    pub zero_fill_backends: Vec<&'static str>,
    pub attention_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct VllmQsaAttnLocalWorkletResolved {
    pub raw_cfg: VllmQsaAttnLocalWorkletConfig,
    pub qkv_proj: SingleGemmKernelConfig,
    pub qk_norm_rope_gate: QkRmsNormRopeGateKernelConfig,
    pub index_qk_proj: SingleGemmKernelConfig,
    pub indexer: SparseGqaIndexerOpResolved,
    pub attention: SparseGqaAttentionOpResolved,
    pub o_proj: SingleGemmKernelConfig,
}

/// One step as the attention layer sees it.
#[derive(Clone, Debug, Default)]
pub struct VllmQsaAttnLocalWorkletInput {
    /// Rows the graph pieces run: the graph-padded token count.
    pub padded_tokens: u32,
    /// A FULL-graph replay of a uniform verify step: `_run_qsa` runs at the
    /// padded rows too.
    pub full_graph: bool,
    /// `(cached prefix, new tokens)` of each prefill chunk.
    pub prefill_chunks: Vec<(u32, u32)>,
    /// Context of each decode (verify) request, its query rows included.
    pub decode_kv_lens: Vec<u32>,
}

/// The compound ops' inputs and the graph-piece rows of one step.
#[derive(Clone, Debug)]
pub struct VllmQsaAttnStep {
    pub padded_rows: u32,
    pub indexer: SparseGqaIndexerOpInput,
    pub attention: SparseGqaAttentionOpInput,
}

pub struct VllmQsaAttnLocalWorklet {
    pub name: String,
    qkv_proj: Op<SingleGemmKernel>,
    qk_norm_rope_gate: Op<QkRmsNormRopeGateKernel>,
    index_qk_proj: Op<SingleGemmKernel>,
    indexer: SparseGqaIndexerOp,
    attention: SparseGqaAttentionOp,
    o_proj: Op<SingleGemmKernel>,
    resolved: VllmQsaAttnLocalWorkletResolved,
}

impl VllmQsaAttnLocalWorklet {
    pub fn resolve_config(cfg: &VllmQsaAttnLocalWorkletConfig) -> VllmQsaAttnLocalWorkletResolved {
        assert_eq!(
            cfg.indexer_num_kv_heads, 1,
            "the sparse indexer kernels take one key head"
        );
        let max_decode_query_len = 1 + cfg.num_speculative_tokens;
        assert!(
            cfg.decode_query_len >= 1 && cfg.decode_query_len <= max_decode_query_len,
            "decode query length {} outside 1..={max_decode_query_len}",
            cfg.decode_query_len
        );
        let gpu = &cfg.gpu_name;
        let ratio = cfg.indexer_compress_ratio;
        let gemm = |n: u32, k: u32| SingleGemmKernelConfig {
            backends: cfg.proj_gemm_backends.clone(),
            gpu_name: gpu.clone(),
            n: n.into(),
            k: k.into(),
            dtype: cfg.proj_dtype,
        };
        let q_width = cfg.num_q_heads * cfg.head_dim;
        VllmQsaAttnLocalWorkletResolved {
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
            index_qk_proj: gemm(
                (cfg.indexer_num_heads + cfg.indexer_num_kv_heads) * cfg.indexer_head_dim,
                cfg.hidden,
            ),
            indexer: SparseGqaIndexerOp::resolve_config(&SparseGqaIndexerOpConfig {
                gpu_name: gpu.clone(),
                num_heads: cfg.indexer_num_heads,
                head_dim: cfg.indexer_head_dim,
                compress_ratio: ratio,
                token_topk: cfg.indexer_token_topk,
                ring_size: ring_tokens(ratio, cfg.num_speculative_tokens),
                decode_query_len: cfg.decode_query_len,
                max_position_embeddings: cfg.max_position_embeddings,
                max_model_len: cfg.max_model_len,
                kv_block_size: cfg.kv_block_size,
                input_dtype: cfg.activation_dtype,
                cache_dtype: cfg.indexer_kv_dtype,
                prepare_backends: cfg.indexer_prepare_backends.clone(),
                logits_decode_backends: cfg.indexer_logits_backends.clone(),
                logits_prefill_backends: cfg.indexer_logits_backends.clone(),
                topk_backends: cfg.topk_backends.clone(),
                cooperative_topk_backends: cfg.cooperative_topk_backends.clone(),
                expand_backends: cfg.index_expand_backends.clone(),
            }),
            attention: SparseGqaAttentionOp::resolve_config(&SparseGqaAttentionOpConfig {
                gpu_name: gpu.clone(),
                num_q_heads: cfg.num_q_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                token_topk: cfg.indexer_token_topk,
                compress_ratio: ratio,
                kv_block_size: cfg.kv_block_size,
                decode_query_len: cfg.decode_query_len,
                max_decode_query_len,
                activation_dtype: cfg.activation_dtype,
                kv_dtype: cfg.kv_dtype,
                kv_append_backends: cfg.kv_append_backends.clone(),
                zero_fill_backends: cfg.zero_fill_backends.clone(),
                attention_backends: cfg.attention_backends.clone(),
            }),
            o_proj: gemm(cfg.hidden, q_width),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: VllmQsaAttnLocalWorkletResolved,
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
            index_qk_proj: atomic(
                n,
                "index_qk_proj",
                r.index_qk_proj,
                SingleGemmKernel::build,
                bridge,
            )?,
            indexer: SparseGqaIndexerOp::build(format!("{n}.indexer"), r.indexer, bridge)?,
            attention: SparseGqaAttentionOp::build(format!("{n}.attention"), r.attention, bridge)?,
            o_proj: atomic(n, "o_proj", r.o_proj, SingleGemmKernel::build, bridge)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        CostNode::Labeled {
            label: format!(
                "{} (VllmQsaAttnLocalWorklet) [local (1 GPU); Hq={}, Hkv={}, D={}, topk={}/{}]",
                self.name,
                cfg.num_q_heads,
                cfg.num_kv_heads,
                cfg.head_dim,
                cfg.indexer_token_topk,
                cfg.indexer_compress_ratio
            ),
            child: Box::new(CostNode::Sum(vec![
                self.qkv_proj.compile(builder),
                self.qk_norm_rope_gate.compile(builder),
                self.index_qk_proj.compile(builder),
                self.indexer.compile(builder),
                self.attention.compile(builder),
                self.o_proj.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &VllmQsaAttnLocalWorkletInput, ev: &mut Evaluator) {
        let step = step(&self.resolved, input);
        let rows = SingleGemmKernelInput {
            m: step.padded_rows,
        };
        let empty = step.padded_rows == 0;
        push_or_zero(&self.qkv_proj, rows.clone(), empty, ev);
        push_or_zero(
            &self.qk_norm_rope_gate,
            QkRmsNormRopeGateKernelInput {
                num_tokens: step.padded_rows,
            },
            empty,
            ev,
        );
        push_or_zero(&self.index_qk_proj, rows.clone(), empty, ev);
        self.indexer.eval(&step.indexer, ev);
        self.attention.eval(&step.attention, ev);
        push_or_zero(&self.o_proj, rows, empty, ev);
    }
}

/// Map one step onto the compound ops' inputs. Pure.
pub fn step(
    r: &VllmQsaAttnLocalWorkletResolved,
    input: &VllmQsaAttnLocalWorkletInput,
) -> VllmQsaAttnStep {
    let q = r.raw_cfg.decode_query_len;
    let decodes = input.decode_kv_lens.len() as u32;
    let real_rows: u32 = input
        .prefill_chunks
        .iter()
        .map(|&(_, new)| new)
        .sum::<u32>()
        + q * decodes;
    assert!(
        input.padded_tokens >= real_rows,
        "padded rows {} below the step's {real_rows} rows",
        input.padded_tokens
    );
    let (launch_rows, decode_requests) = if input.full_graph {
        assert!(
            input.prefill_chunks.is_empty() && input.padded_tokens.is_multiple_of(q),
            "a FULL graph replays only a uniform verify batch of whole requests"
        );
        (input.padded_tokens, input.padded_tokens / q)
    } else {
        (real_rows, decodes)
    };
    VllmQsaAttnStep {
        padded_rows: input.padded_tokens,
        indexer: SparseGqaIndexerOpInput {
            num_rows: launch_rows,
            decode_requests,
            decode_kv_lens: input.decode_kv_lens.clone(),
            prefill_chunks: input.prefill_chunks.clone(),
        },
        attention: SparseGqaAttentionOpInput {
            output_rows: input.padded_tokens,
            append_rows: launch_rows,
            attention_rows: launch_rows,
            prefill_chunks: input.prefill_chunks.clone(),
            decode_kv_lens: input.decode_kv_lens.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::attention::sparse_gqa_indexer::TopkBackend;
    use crate::op::attention::{sparse_gqa, sparse_gqa_indexer};

    fn resolved() -> VllmQsaAttnLocalWorkletResolved {
        VllmQsaAttnLocalWorklet::resolve_config(&VllmQsaAttnLocalWorkletConfig {
            hidden: 2560,
            num_q_heads: 24,
            num_kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            mrope: false,
            indexer_num_heads: 4,
            indexer_num_kv_heads: 1,
            indexer_head_dim: 128,
            indexer_compress_ratio: 4,
            indexer_token_topk: 2048,
            max_position_embeddings: 262144,
            max_model_len: 131072,
            kv_block_size: 3200,
            num_speculative_tokens: 3,
            decode_query_len: 4,
            activation_dtype: DType::Bf16,
            proj_dtype: DType::Bf16,
            kv_dtype: DType::Fp8E4m3,
            indexer_kv_dtype: DType::Fp8E4m3,
            gpu_name: "NVIDIA B200".into(),
            proj_gemm_backends: vec!["torch_linear_vllm"],
            qk_norm_rope_backends: vec!["vllm_triton"],
            indexer_prepare_backends: vec!["vllm_triton"],
            indexer_logits_backends: vec!["vllm_triton"],
            topk_backends: vec!["vllm_cuda"],
            cooperative_topk_backends: vec!["vllm_cuda_cooperative"],
            index_expand_backends: vec!["vllm_triton"],
            kv_append_backends: vec!["vllm_cuda"],
            zero_fill_backends: vec!["torch"],
            attention_backends: vec!["vllm_triton"],
        })
    }

    #[test]
    fn shapes_are_the_profiled_rows() {
        let r = resolved();
        assert_eq!((r.qkv_proj.n.get(), r.qkv_proj.k.get()), (13312, 2560));
        assert_eq!(
            (r.index_qk_proj.n.get(), r.index_qk_proj.k.get()),
            (640, 2560)
        );
        assert_eq!((r.o_proj.n.get(), r.o_proj.k.get()), (2560, 6144));
        assert_eq!(r.indexer.prepare.ring_size, 8);
        assert_eq!(r.indexer.logits_width, 32800);
        assert_eq!(r.attention.max_decode_query_len, 4);
    }

    /// Pure verify under a FULL graph: everything at the padded rows, whole
    /// padded requests; the uniform attention table.
    #[test]
    fn pure_verify_full_graph_runs_at_padded_rows() {
        let r = resolved();
        let s = step(
            &r,
            &VllmQsaAttnLocalWorkletInput {
                padded_tokens: 24,
                full_graph: true,
                prefill_chunks: vec![],
                decode_kv_lens: vec![4096; 5],
            },
        );
        assert_eq!((s.indexer.num_rows, s.indexer.decode_requests), (24, 6));
        assert_eq!(
            (s.attention.attention_rows, s.attention.output_rows),
            (24, 24)
        );
        let idx = sparse_gqa_indexer::plan(&r.indexer, &s.indexer, true);
        assert_eq!(idx.decode_topk.unwrap().0, TopkBackend::Cooperative);
        assert!(idx.prefill_topk.is_none() && idx.prefill_chunks.is_empty());
        let attn = sparse_gqa::plan(&r.attention, &s.attention);
        assert!(!attn.prefill_config);
        assert_eq!(attn.append_rows, 24);
    }

    /// Mixed under a piecewise graph: `_run_qsa` at the real rows, both
    /// selection branches, the prefill attention table.
    #[test]
    fn mixed_step_runs_both_branches_at_real_rows() {
        let r = resolved();
        let s = step(
            &r,
            &VllmQsaAttnLocalWorkletInput {
                padded_tokens: 2048,
                full_graph: false,
                prefill_chunks: vec![(0, 1536)],
                decode_kv_lens: vec![4096; 120],
            },
        );
        assert_eq!(s.padded_rows, 2048);
        assert_eq!((s.indexer.num_rows, s.indexer.decode_requests), (2016, 120));
        assert_eq!(
            (
                s.attention.attention_rows,
                s.attention.append_rows,
                s.attention.output_rows
            ),
            (2016, 2016, 2048)
        );
        let idx = sparse_gqa_indexer::plan(&r.indexer, &s.indexer, true);
        assert!(idx.logits_decode.is_some() && idx.prefill_topk.is_some());
        assert!(sparse_gqa::plan(&r.attention, &s.attention).prefill_config);
    }

    #[test]
    fn prefill_only_step_has_no_decode_branch() {
        let r = resolved();
        let s = step(
            &r,
            &VllmQsaAttnLocalWorkletInput {
                padded_tokens: 8192,
                full_graph: false,
                prefill_chunks: vec![(0, 8192)],
                decode_kv_lens: vec![],
            },
        );
        let idx = sparse_gqa_indexer::plan(&r.indexer, &s.indexer, true);
        assert!(idx.logits_decode.is_none() && idx.decode_topk.is_none());
        let (backend, cells) = idx.prefill_topk.unwrap();
        assert_eq!(backend, TopkBackend::Persistent);
        assert_eq!(cells.len(), 2);
    }

    #[test]
    #[should_panic(expected = "uniform verify")]
    fn full_graph_rejects_a_prefill_row() {
        step(
            &resolved(),
            &VllmQsaAttnLocalWorkletInput {
                padded_tokens: 16,
                full_graph: true,
                prefill_chunks: vec![(0, 8)],
                decode_kv_lens: vec![64],
            },
        );
    }
}
