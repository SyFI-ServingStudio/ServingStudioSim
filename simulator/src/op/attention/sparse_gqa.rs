//! Sparse GQA attention compound op: the cache write and the sparse attention
//! call that reads the indexer's selection, for one step.
//!
//! Four fixed leaves, in launch order (`models/qwen4_exp/nvidia/qsa.py`
//! `_run_qsa` / `forward_qsa`):
//!
//! ```text
//! kv_append         kv_cache_append            the step's new tokens (slot_mapping)
//! out_zero          elementwise memset         output.zero_() over the output tensor rows
//! attn_uniform      sparse_gqa_paged_attention prefill_config = false
//! attn_prefill_cfg  sparse_gqa_paged_attention prefill_config = true
//! ```
//!
//! The attention is one launch over every row of the step, decode and prefill
//! alike (a capture shows one split-K call per layer even in mixed steps), at
//! the cell `(num_rows, row-weighted mean context)`. The framework switches its
//! launch table to the prefill one when the longest query of the step exceeds
//! the verify length `1 + num_speculative_tokens`; exactly one of the two
//! attention leaves is non-zero.

use std::sync::Arc;

use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, KvCacheAppendKernel,
    KvCacheAppendKernelConfig, KvCacheAppendKernelInput, SparseGqaPagedAttentionKernel,
    SparseGqaPagedAttentionKernelConfig, SparseGqaPagedAttentionKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Evaluator, LeafMetrics, PerfApiBridge, Probe,
};

/// vLLM's paged KV layout and its one static FP8 scale.
const CACHE_LAYOUT: &str = "NHD";
const KV_SCALE_GRANULARITY: &str = "tensor";

#[derive(Clone, Debug)]
pub struct SparseGqaAttentionOpConfig {
    pub gpu_name: String,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    /// Tokens the indexer selects per row.
    pub token_topk: u32,
    pub compress_ratio: u32,
    pub kv_block_size: u32,
    /// Query rows of every decode request.
    pub decode_query_len: u32,
    /// The framework's longest decode query, `1 + num_speculative_tokens`; a
    /// longer query selects the prefill launch table.
    pub max_decode_query_len: u32,
    pub activation_dtype: DType,
    pub kv_dtype: DType,
    pub kv_append_backends: Vec<&'static str>,
    pub zero_fill_backends: Vec<&'static str>,
    pub attention_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct SparseGqaAttentionOpResolved {
    pub kv_append: KvCacheAppendKernelConfig,
    pub out_zero: ElementwiseKernelConfig,
    pub attn_uniform: SparseGqaPagedAttentionKernelConfig,
    pub attn_prefill_cfg: SparseGqaPagedAttentionKernelConfig,
    pub decode_query_len: u32,
    pub max_decode_query_len: u32,
}

/// One step as the attention call sees it.
#[derive(Clone, Debug, Default)]
pub struct SparseGqaAttentionOpInput {
    /// Rows of the output tensor the memset clears (the graph-padded rows).
    pub output_rows: u32,
    /// Rows the attention launch covers (graph-padded under a FULL graph).
    pub attention_rows: u32,
    /// `(cached prefix, new tokens)` of each prefill chunk.
    pub prefill_chunks: Vec<(u32, u32)>,
    /// Context of each decode request, its query rows included.
    pub decode_kv_lens: Vec<u32>,
}

/// The leaf inputs of one step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseGqaAttentionPlan {
    pub append_tokens: u32,
    pub output_rows: u32,
    pub attention_rows: u32,
    pub context_len: u32,
    pub prefill_config: bool,
}

pub struct SparseGqaAttentionOp {
    pub name: String,
    kv_append: Arc<KvCacheAppendKernel>,
    out_zero: Arc<ElementwiseKernel>,
    attn_uniform: Arc<SparseGqaPagedAttentionKernel>,
    attn_prefill_cfg: Arc<SparseGqaPagedAttentionKernel>,
    resolved: SparseGqaAttentionOpResolved,
}

impl SparseGqaAttentionOp {
    /// Expand the op config into its kernel configs. Pure.
    pub fn resolve_config(cfg: &SparseGqaAttentionOpConfig) -> SparseGqaAttentionOpResolved {
        assert!(
            cfg.decode_query_len >= 1 && cfg.decode_query_len <= cfg.max_decode_query_len,
            "decode query length {} outside 1..={}",
            cfg.decode_query_len,
            cfg.max_decode_query_len
        );
        let gpu = &cfg.gpu_name;
        let attention = |prefill_config: bool| SparseGqaPagedAttentionKernelConfig {
            backends: cfg.attention_backends.clone(),
            gpu_name: gpu.clone(),
            num_q_heads: cfg.num_q_heads.into(),
            num_kv_heads: cfg.num_kv_heads.into(),
            head_dim: cfg.head_dim.into(),
            token_topk: cfg.token_topk,
            compress_ratio: cfg.compress_ratio,
            q_dtype: cfg.activation_dtype,
            kv_dtype: cfg.kv_dtype,
            prefill_config,
        };
        SparseGqaAttentionOpResolved {
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
            // A pure write of one [heads x head_dim] output row per token.
            out_zero: ElementwiseKernelConfig {
                backends: cfg.zero_fill_backends.clone(),
                gpu_name: gpu.clone(),
                input_bytes_per_token: 0.into(),
                output_bytes_per_token: (cfg.num_q_heads
                    * cfg.head_dim
                    * cfg.activation_dtype.size_bytes())
                .into(),
            },
            attn_uniform: attention(false),
            attn_prefill_cfg: attention(true),
            decode_query_len: cfg.decode_query_len,
            max_decode_query_len: cfg.max_decode_query_len,
        }
    }

    pub fn build(
        name: String,
        resolved: SparseGqaAttentionOpResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        Ok(Self {
            kv_append: Arc::new(KvCacheAppendKernel::build(
                format!("{name}.kv_append"),
                r.kv_append,
                bridge,
            )?),
            out_zero: Arc::new(ElementwiseKernel::build(
                format!("{name}.out_zero"),
                r.out_zero,
                bridge,
            )?),
            attn_uniform: Arc::new(SparseGqaPagedAttentionKernel::build(
                format!("{name}.attn_uniform"),
                r.attn_uniform,
                bridge,
            )?),
            attn_prefill_cfg: Arc::new(SparseGqaPagedAttentionKernel::build(
                format!("{name}.attn_prefill_cfg"),
                r.attn_prefill_cfg,
                bridge,
            )?),
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Sum(vec![
            leaf(builder, &self.name, "kv_append", &*self.kv_append),
            leaf(builder, &self.name, "out_zero", &*self.out_zero),
            leaf(builder, &self.name, "attn_uniform", &*self.attn_uniform),
            leaf(
                builder,
                &self.name,
                "attn_prefill_cfg",
                &*self.attn_prefill_cfg,
            ),
        ])
    }

    pub fn eval(&self, input: &SparseGqaAttentionOpInput, ev: &mut Evaluator) {
        let p = plan(&self.resolved, input);

        let append = KvCacheAppendKernelInput {
            num_tokens: p.append_tokens,
        };
        let metrics = if p.append_tokens == 0 {
            LeafMetrics::ZERO
        } else {
            self.kv_append.eval(&append)
        };
        ev.push(metrics, || append.clone().into());

        let zero = ElementwiseKernelInput {
            num_tokens: p.output_rows,
        };
        let metrics = if p.output_rows == 0 {
            LeafMetrics::ZERO
        } else {
            self.out_zero.eval(&zero)
        };
        ev.push(metrics, || zero.clone().into());

        let cell = SparseGqaPagedAttentionKernelInput {
            num_rows: p.attention_rows,
            context_len: p.context_len,
        };
        for (kernel, prefill_config) in
            [(&self.attn_uniform, false), (&self.attn_prefill_cfg, true)]
        {
            let metrics = if p.attention_rows == 0 || p.prefill_config != prefill_config {
                LeafMetrics::ZERO
            } else {
                kernel.eval(&cell)
            };
            ev.push(metrics, || cell.clone().into());
        }
    }
}

fn leaf<K: Probe>(builder: &mut CostTreeBuilder, name: &str, slot: &str, kernel: &K) -> CostNode {
    builder.leaf(
        format!("{name}.{slot}"),
        kernel.kind(),
        kernel.describe_config(),
    )
}

/// Every leaf input of one step. Pure.
pub fn plan(
    r: &SparseGqaAttentionOpResolved,
    input: &SparseGqaAttentionOpInput,
) -> SparseGqaAttentionPlan {
    let q = u64::from(r.decode_query_len);
    let mut rows = 0u64;
    let mut context_sum = 0u64;
    let mut longest_query = 0u32;
    for &(prefix, new) in &input.prefill_chunks {
        let (p, n) = (u64::from(prefix), u64::from(new));
        // A row at position prefix + i sees prefix + i + 1 tokens.
        rows += n;
        context_sum += n * p + n * (n + 1) / 2;
        longest_query = longest_query.max(new);
    }
    for &kv in &input.decode_kv_lens {
        let kv = u64::from(kv);
        assert!(
            kv >= q,
            "decode context {kv} shorter than its {q} query rows"
        );
        // Rows at positions kv - q .. kv - 1 see kv - q + 1 .. kv tokens.
        rows += q;
        context_sum += q * kv - q * (q - 1) / 2;
    }
    if !input.decode_kv_lens.is_empty() {
        longest_query = longest_query.max(r.decode_query_len);
    }
    let context_len = if rows == 0 {
        0
    } else {
        ((context_sum + rows / 2) / rows) as u32
    };
    SparseGqaAttentionPlan {
        append_tokens: rows as u32,
        output_rows: input.output_rows,
        attention_rows: input.attention_rows,
        context_len,
        prefill_config: longest_query > r.max_decode_query_len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved() -> SparseGqaAttentionOpResolved {
        SparseGqaAttentionOp::resolve_config(&SparseGqaAttentionOpConfig {
            gpu_name: "NVIDIA B200".into(),
            num_q_heads: 24,
            num_kv_heads: 2,
            head_dim: 256,
            token_topk: 2048,
            compress_ratio: 4,
            kv_block_size: 3200,
            decode_query_len: 4,
            max_decode_query_len: 4,
            activation_dtype: DType::Bf16,
            kv_dtype: DType::Fp8E4m3,
            kv_append_backends: vec!["vllm_cuda"],
            zero_fill_backends: vec!["torch"],
            attention_backends: vec!["vllm_triton"],
        })
    }

    #[test]
    fn configs_key_the_profiled_rows() {
        let r = resolved();
        assert_eq!(r.out_zero.output_bytes_per_token.get(), 12288);
        assert_eq!(r.out_zero.input_bytes_per_token.get(), 0);
        assert!(!r.attn_uniform.prefill_config && r.attn_prefill_cfg.prefill_config);
        assert_eq!(r.kv_append.block_size, 3200);
    }

    /// A pure verify step keeps the uniform table; any prefill chunk longer
    /// than the verify width switches it.
    #[test]
    fn prefill_config_follows_the_longest_query() {
        let r = resolved();
        let verify = SparseGqaAttentionOpInput {
            output_rows: 16,
            attention_rows: 16,
            prefill_chunks: vec![],
            decode_kv_lens: vec![100, 200, 300, 400],
        };
        let p = plan(&r, &verify);
        assert!(!p.prefill_config);
        assert_eq!(p.append_tokens, 16);
        // Mean over rows 97..=100, 197..=200, ...: 248.5 rounds to 249.
        assert_eq!(p.context_len, 249);

        let short_prefill = SparseGqaAttentionOpInput {
            prefill_chunks: vec![(0, 4)],
            ..verify.clone()
        };
        assert!(!plan(&r, &short_prefill).prefill_config);
        let mixed = SparseGqaAttentionOpInput {
            prefill_chunks: vec![(0, 5)],
            ..verify
        };
        let p = plan(&r, &mixed);
        assert!(p.prefill_config);
        assert_eq!(p.append_tokens, 21);
    }

    #[test]
    fn prefill_context_is_the_causal_row_mean() {
        let p = plan(
            &resolved(),
            &SparseGqaAttentionOpInput {
                output_rows: 2048,
                attention_rows: 2048,
                prefill_chunks: vec![(126976, 2048)],
                decode_kv_lens: vec![],
            },
        );
        assert!(p.prefill_config);
        assert_eq!(p.context_len, 126976 + 1025);
    }
}
