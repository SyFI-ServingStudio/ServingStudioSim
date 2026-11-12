//! Inkling attention on one TP rank, up to the reduction
//! (`vllm/models/inkling/nvidia/attention.py`).
//!
//! Four launches in order: the merged QKVR projection (`F.linear`, q | k | v |
//! the relative-position branch r), the fused QKVR prep (k/v short conv, q/k
//! head norms, log-scaling tau on global layers, KV insert, rel-logit
//! projection), the relative-position FA4 call over the whole step, and the
//! O projection (`RowParallelLinear(reduce_results=False)`). The reduction is
//! the arch's Lamport RS + sconv + AG + norm.
//!
//! Every launch but attention runs on the graph-padded rows; attention reads
//! the real requests.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    RelBiasAttentionKernel, RelBiasAttentionKernelConfig, RelBiasAttentionKernelInput,
    SconvQkvrPrepKernel, SconvQkvrPrepKernelConfig, SconvQkvrPrepKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, push_or_zero};

/// The fork's paged KV layout: K and V of one (token, head) side by side.
const INKLING_KV_LAYOUT: &str = "lbnhc";

#[derive(Clone, Debug)]
pub struct InklingAttnLocalWorkletConfig {
    pub hidden: Dim,
    /// Query heads on this rank.
    pub num_q_heads: u32,
    /// KV heads on this rank.
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub d_rel: u32,
    /// Sliding-window layer.
    pub is_local: bool,
    /// Relative distances covered: the window on local layers, the model's
    /// `rel_extent` on global layers.
    pub rel_extent: u32,
    pub sconv_kernel_size: u32,
    /// Tokens per KV cache page.
    pub page_size: u32,
    pub max_model_len: u32,
    pub dtype: DType,
    pub gpu_name: String,
    pub gemm_backends: Vec<&'static str>,
    pub prep_backends: Vec<&'static str>,
    pub attention_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct InklingAttnLocalWorkletResolved {
    pub raw_cfg: InklingAttnLocalWorkletConfig,
    pub qkvr_proj: SingleGemmKernelConfig,
    pub qkvr_prep: SconvQkvrPrepKernelConfig,
    pub rel_attention: RelBiasAttentionKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct InklingAttnLocalWorkletInput {
    /// Graph-padded rows the projections and the prep run on.
    pub num_tokens: u32,
    /// The step's real requests as `(query_len, context_len)`.
    pub query_context_pairs: Vec<(u32, u32)>,
    /// Requests attention is launched over when a FULL CUDA graph pads the
    /// batch with zero-length requests; 0 = unpadded.
    pub padded_requests: u32,
}

pub struct InklingAttnLocalWorklet {
    pub name: String,
    pub qkvr_proj: Op<SingleGemmKernel>,
    pub qkvr_prep: Op<SconvQkvrPrepKernel>,
    pub rel_attention: Op<RelBiasAttentionKernel>,
    pub o_proj: Op<SingleGemmKernel>,
    resolved: InklingAttnLocalWorkletResolved,
}

impl InklingAttnLocalWorklet {
    pub fn resolve_config(cfg: &InklingAttnLocalWorkletConfig) -> InklingAttnLocalWorkletResolved {
        assert!(
            cfg.num_kv_heads > 0 && cfg.num_q_heads.is_multiple_of(cfg.num_kv_heads),
            "query heads must group evenly over KV heads"
        );
        let q_width = cfg.num_q_heads * cfg.head_dim;
        let kv_width = cfg.num_kv_heads * cfg.head_dim;
        // q | k | v | r, r being d_rel per query head.
        let qkvr_width = q_width + 2 * kv_width + cfg.num_q_heads * cfg.d_rel;
        let gemm = |n: u32, k: Dim| SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n: n.into(),
            k,
            dtype: cfg.dtype,
        };
        InklingAttnLocalWorkletResolved {
            qkvr_proj: gemm(qkvr_width, cfg.hidden.clone()),
            qkvr_prep: SconvQkvrPrepKernelConfig {
                backends: cfg.prep_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_q_heads: cfg.num_q_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                d_rel: cfg.d_rel,
                rel_extent: cfg.rel_extent,
                // tau scales q on global layers only.
                apply_log_scaling: !cfg.is_local,
                sconv_kernel_size: cfg.sconv_kernel_size,
                // vLLM sizes a conv-state block to the conv window.
                conv_block_size: cfg.sconv_kernel_size,
                page_size: cfg.page_size,
                dtype: cfg.dtype,
            },
            rel_attention: RelBiasAttentionKernelConfig {
                backends: cfg.attention_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_qo_heads: cfg.num_q_heads.into(),
                num_kv_heads: cfg.num_kv_heads.into(),
                head_dim: cfg.head_dim.into(),
                is_local: cfg.is_local,
                rel_extent: cfg.rel_extent,
                page_size: cfg.page_size,
                kv_layout: INKLING_KV_LAYOUT.to_string(),
                q_dtype: cfg.dtype,
                kv_dtype: cfg.dtype,
                max_model_len: cfg.max_model_len,
            },
            o_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden.clone(),
                k: q_width.into(),
                dtype: cfg.dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: InklingAttnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            qkvr_proj: atomic(n, "qkvr_proj", r.qkvr_proj, SingleGemmKernel::build, bridge)?,
            qkvr_prep: atomic(
                n,
                "qkvr_prep",
                r.qkvr_prep,
                SconvQkvrPrepKernel::build,
                bridge,
            )?,
            rel_attention: atomic(
                n,
                "rel_attention",
                r.rel_attention,
                RelBiasAttentionKernel::build,
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
                "{} (InklingAttnLocalWorklet) [TP rank; {}; q/kv heads {}/{}; rel_extent {}]",
                self.name,
                if cfg.is_local { "sliding window" } else { "global" },
                cfg.num_q_heads,
                cfg.num_kv_heads,
                cfg.rel_extent
            ),
            child: Box::new(CostNode::Sum(vec![
                self.qkvr_proj.compile(builder),
                self.qkvr_prep.compile(builder),
                self.rel_attention.compile(builder),
                self.o_proj.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &InklingAttnLocalWorkletInput, ev: &mut Evaluator) {
        let rows = SingleGemmKernelInput {
            m: input.num_tokens,
        };
        push_or_zero(&self.qkvr_proj, rows.clone(), false, ev);
        push_or_zero(
            &self.qkvr_prep,
            SconvQkvrPrepKernelInput {
                num_tokens: input.num_tokens,
            },
            false,
            ev,
        );
        push_or_zero(
            &self.rel_attention,
            RelBiasAttentionKernelInput {
                query_context_pairs: input.query_context_pairs.clone(),
                padded_requests: input.padded_requests,
            },
            false,
            ev,
        );
        push_or_zero(&self.o_proj, rows, false, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(is_local: bool) -> InklingAttnLocalWorkletConfig {
        InklingAttnLocalWorkletConfig {
            hidden: 4096.into(),
            num_q_heads: 8,
            num_kv_heads: 2,
            head_dim: 128,
            d_rel: 16,
            is_local,
            rel_extent: if is_local { 512 } else { 1024 },
            sconv_kernel_size: 4,
            page_size: 32,
            max_model_len: 8192,
            dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".into(),
            gemm_backends: vec!["torch_linear_vllm"],
            prep_backends: vec!["vllm_triton"],
            attention_backends: vec!["tml_fa4"],
        }
    }

    #[test]
    fn tp4_shapes_match_the_capture() {
        // Inkling-Small at TP4: q 1024 | k 256 | v 256 | r 128 = 1664, and an
        // O projection of 1024 -> 4096 with no reduction.
        let local = InklingAttnLocalWorklet::resolve_config(&cfg(true));
        assert_eq!((local.qkvr_proj.n.get(), local.qkvr_proj.k.get()), (1664, 4096));
        assert_eq!((local.o_proj.n.get(), local.o_proj.k.get()), (4096, 1024));
        assert!(!local.qkvr_prep.apply_log_scaling);
        let global = InklingAttnLocalWorklet::resolve_config(&cfg(false));
        assert!(global.qkvr_prep.apply_log_scaling && !global.rel_attention.is_local);
        assert_eq!(global.rel_attention.rel_extent, 1024);
    }
}
