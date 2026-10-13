//! One Gated DeltaNet token mixer with BF16 projections on one GPU, in vLLM's
//! launch granularity for a speculative (MTP verify) deployment
//! (`qwen_gdn_linear_attn.py` `forward_cuda` with the fused-norm packed core).
//!
//! The section starts after the arch's input norm and ends at the `out_proj`
//! output. Every projection is a plain BF16 GEMM; nothing is quantized.
//!
//! ```text
//! compiled piece   qkvz GEMM, ba GEMM, core_attn_out zeros
//! core (eager)     pure verify, fused: spec causal_conv1d_update, then
//!                  fused_gdn_decode_post_conv_mtp (recurrence + gated norm)
//!                  otherwise: b / a contiguous, then
//!                    mixed: index_select of the spec and non-spec rows
//!                    spec rows: causal_conv1d_update + sigmoid-gating recurrence
//!                      (q tokens per request), rearrange copies + cat
//!                    prefill rows: causal_conv1d_fn, post-conv, state gather /
//!                      zero-fresh / scatter around the FlashInfer chunked
//!                      delta rule, cu_seqlens cast, exp(g)
//!                    mixed: index_copy_ of both outputs into core_attn_out
//!                    output_gate reshape copy, gated RMSNorm (sigmoid gate)
//! compiled piece   out_proj GEMM
//! ```
//!
//! Path selection follows `_forward_core_fused_norm` and the GDN metadata
//! builder (`gdn_attn.py:307-314`): a step with spec decodes reclassifies any
//! one-token non-spec decode as a one-token prefill, so it runs the chunk path.
//! A step with no prefill takes the fused verify kernel when it supports the
//! geometry (`_can_use_fused_gdn_mtp_decode`: at most 8 tokens per request and
//! a value/key head ratio of 1, 2, 3, 4 or 8); otherwise the verify rows run the
//! unfused spec path and the spec output is copied into `core_attn_out`.
//!
//! The core is an eager break (`qwen_gdn_attention_core_fused_norm_packed`), so
//! a piecewise step runs it over the real tokens. A FULL-graph verify replay
//! launches the two verify kernels over the padded requests, but padded rows
//! carry `PAD_SLOT_ID` and return before touching state, so only real requests
//! cost. The zero fill and the GEMMs run at the graph-padded row count.
//!
//! Glue pricing follows the diverse_100 glue attribution: zeros as a torch fill,
//! b / a contiguous and the output_gate copy as strided copies, row splits and
//! merges as `indexed_row_copy` `index_select` / `index_copy`, state movement as
//! `indexed_row_copy` gather / masked_zero / scatter (their int64 index casts
//! are inside those rows; only the cu_seqlens cast is its own launch).

use crate::op::ssm::{
    GdnDecodeOp, GdnDecodeOpConfig, GdnDecodeOpInput, GdnPrefillOp, GdnPrefillOpConfig,
    GdnPrefillOpInput,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, GdnCausalConvDecodeKernel,
    GdnCausalConvDecodeKernelConfig, GdnCausalConvDecodeKernelInput,
    GdnCausalConvPrefillKernelConfig, GdnChunkDeltaRuleKernelConfig, GdnGatedRmsNormKernel,
    GdnGatedRmsNormKernelConfig, GdnGatedRmsNormKernelInput, GdnPrefillPostConvKernelConfig,
    GdnRecurrentDecodeGatedNormKernel, GdnRecurrentDecodeGatedNormKernelConfig,
    GdnRecurrentDecodeGatedNormKernelInput, GdnRecurrentDecodeKernelConfig, IndexedRowCopyKernel,
    IndexedRowCopyKernelConfig, IndexedRowCopyKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, elementwise, push_or_zero, repeated};

/// An int64 index element.
const INDEX_BYTES: u32 = 8;
/// `MAX_FUSED_GDN_MTP_TOKENS`: the fused verify kernel's tokens-per-request cap.
const MAX_FUSED_VERIFY_TOKENS: u32 = 8;
/// Value/key head ratios the fused verify kernel is instantiated for.
const FUSED_VERIFY_HEAD_RATIOS: [u32; 5] = [1, 2, 3, 4, 8];

#[derive(Clone, Debug)]
pub struct VllmBf16GdnSpecLocalWorkletConfig {
    pub hidden: u32,
    pub num_key_heads: u32,
    pub num_value_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel_size: u32,
    /// Tokens each verify request submits: the accepted token plus the drafts.
    pub tokens_per_request: u32,
    pub activation_dtype: DType,
    pub conv_state_dtype: DType,
    pub ssm_state_dtype: DType,
    pub gpu_name: String,
    /// BF16 `in_proj_qkvz`, `in_proj_ba` and `out_proj`.
    pub gemm_backends: Vec<&'static str>,
    /// `torch.zeros` fills and the contiguous index casts.
    pub torch_elementwise_backends: Vec<&'static str>,
    /// Copies whose source is a non-contiguous view (b / a, the spec q/k/v
    /// slices, the output gate).
    pub strided_copy_backends: Vec<&'static str>,
    pub cat_backends: Vec<&'static str>,
    /// State-pool gather / masked zero / scatter and token-row index_select /
    /// index_copy.
    pub indexed_row_copy_backends: Vec<&'static str>,
    pub causal_conv_prefill_backends: Vec<&'static str>,
    pub prefill_post_conv_backends: Vec<&'static str>,
    pub chunk_delta_rule_backends: Vec<&'static str>,
    pub causal_conv_decode_backends: Vec<&'static str>,
    /// `fused_sigmoid_gating_delta_rule_update` over the spec rows.
    pub sigmoid_gating_decode_backends: Vec<&'static str>,
    /// `fused_gdn_decode_post_conv_mtp`.
    pub fused_verify_backends: Vec<&'static str>,
    pub gated_rms_norm_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct VllmBf16GdnSpecLocalWorkletResolved {
    pub raw_cfg: VllmBf16GdnSpecLocalWorkletConfig,
    /// Whether a verify-only step takes the fused verify kernel.
    pub fused_verify: bool,
    pub qkvz: SingleGemmKernelConfig,
    pub ba: SingleGemmKernelConfig,
    pub core_output_zero: ElementwiseKernelConfig,
    pub verify_conv: GdnCausalConvDecodeKernelConfig,
    pub verify_recurrent_norm: GdnRecurrentDecodeGatedNormKernelConfig,
    /// `b.contiguous()` and `a.contiguous()`, one leaf folded twice.
    pub ba_contiguous: ElementwiseKernelConfig,
    pub qkv_select: IndexedRowCopyKernelConfig,
    /// `a` and `b` index_select, one leaf folded twice per row set.
    pub ab_select: IndexedRowCopyKernelConfig,
    pub spec_decode: GdnDecodeOpConfig,
    pub spec_qk_copy: ElementwiseKernelConfig,
    pub spec_v_copy: ElementwiseKernelConfig,
    pub spec_qkv_cat: ElementwiseKernelConfig,
    pub prefill: GdnPrefillOpConfig,
    pub state_gather: IndexedRowCopyKernelConfig,
    pub state_zero: IndexedRowCopyKernelConfig,
    pub state_scatter: IndexedRowCopyKernelConfig,
    pub cu_seqlens_cast: ElementwiseKernelConfig,
    pub g_exp: ElementwiseKernelConfig,
    pub output_index_copy: IndexedRowCopyKernelConfig,
    /// `core_attn_out[:n] = spec output` of an unfused verify-only step.
    pub spec_output_copy: ElementwiseKernelConfig,
    pub output_gate_copy: ElementwiseKernelConfig,
    pub gated_norm: GdnGatedRmsNormKernelConfig,
    pub out_proj: SingleGemmKernelConfig,
}

/// One step as the GDN layer sees it.
#[derive(Clone, Debug, Default)]
pub struct VllmBf16GdnSpecLocalWorkletInput {
    /// Rows the compiled pieces run: the graph-padded token count.
    pub padded_tokens: u32,
    /// Prefill chunk lengths of the step's prefill requests.
    pub prefill_lengths: Vec<u32>,
    /// Verify requests, each `tokens_per_request` rows.
    pub spec_decodes: u32,
    /// One-token decodes without drafts. Beside spec decodes they run as
    /// one-token prefills; without spec decodes they take the plain decode path,
    /// which this worklet does not model.
    pub non_spec_decodes: u32,
}

pub struct VllmBf16GdnSpecLocalWorklet {
    pub name: String,
    qkvz: Op<SingleGemmKernel>,
    ba: Op<SingleGemmKernel>,
    core_output_zero: Op<ElementwiseKernel>,
    verify_conv: Op<GdnCausalConvDecodeKernel>,
    verify_recurrent_norm: Op<GdnRecurrentDecodeGatedNormKernel>,
    ba_contiguous: Op<ElementwiseKernel>,
    spec_qkv_select: Op<IndexedRowCopyKernel>,
    spec_ab_select: Op<IndexedRowCopyKernel>,
    non_spec_qkv_select: Op<IndexedRowCopyKernel>,
    non_spec_ab_select: Op<IndexedRowCopyKernel>,
    spec_decode: GdnDecodeOp,
    spec_qk_copy: Op<ElementwiseKernel>,
    spec_v_copy: Op<ElementwiseKernel>,
    spec_qkv_cat: Op<ElementwiseKernel>,
    prefill: GdnPrefillOp,
    state_gather: Op<IndexedRowCopyKernel>,
    state_zero: Op<IndexedRowCopyKernel>,
    state_scatter: Op<IndexedRowCopyKernel>,
    cu_seqlens_cast: Op<ElementwiseKernel>,
    g_exp: Op<ElementwiseKernel>,
    spec_output_index_copy: Op<IndexedRowCopyKernel>,
    non_spec_output_index_copy: Op<IndexedRowCopyKernel>,
    spec_output_copy: Op<ElementwiseKernel>,
    output_gate_copy: Op<ElementwiseKernel>,
    gated_norm: Op<GdnGatedRmsNormKernel>,
    out_proj: Op<SingleGemmKernel>,
    resolved: VllmBf16GdnSpecLocalWorkletResolved,
}

impl VllmBf16GdnSpecLocalWorklet {
    pub fn resolve_config(
        cfg: &VllmBf16GdnSpecLocalWorkletConfig,
    ) -> VllmBf16GdnSpecLocalWorkletResolved {
        assert!(
            cfg.tokens_per_request >= 1,
            "a verify request has at least one token"
        );
        assert_eq!(
            cfg.num_value_heads % cfg.num_key_heads,
            0,
            "GDN value heads must be a multiple of key heads"
        );
        let gpu = &cfg.gpu_name;
        let act = cfg.activation_dtype.size_bytes();
        let key_width = cfg.num_key_heads * cfg.key_head_dim;
        let value_width = cfg.num_value_heads * cfg.value_head_dim;
        let conv_channels = 2 * key_width + value_width;
        let value_row = value_width * act;
        let q = cfg.tokens_per_request;
        let fused_verify = q <= MAX_FUSED_VERIFY_TOKENS
            && FUSED_VERIFY_HEAD_RATIOS.contains(&(cfg.num_value_heads / cfg.num_key_heads));
        let gemm = |n: u32, k: u32| SingleGemmKernelConfig {
            backends: cfg.gemm_backends.clone(),
            gpu_name: gpu.clone(),
            n: n.into(),
            k: k.into(),
            dtype: cfg.activation_dtype,
        };
        let torch = |input: u32, output: u32| {
            elementwise(&cfg.torch_elementwise_backends, gpu, input, output)
        };
        let strided_copy = |bytes: u32| elementwise(&cfg.strided_copy_backends, gpu, bytes, bytes);
        let rows = |name: &'static str, elements: u32, dtype: DType, operation: &str| {
            IndexedRowCopyKernelConfig {
                backends: cfg.indexed_row_copy_backends.clone(),
                gpu_name: gpu.clone(),
                row_elements: Dim::param(name, elements),
                dtype,
                operation: operation.to_string(),
            }
        };
        let ssm_state_elements = cfg.num_value_heads * cfg.key_head_dim * cfg.value_head_dim;
        let state = |operation: &str| {
            rows(
                "ssm_state_elements",
                ssm_state_elements,
                cfg.ssm_state_dtype,
                operation,
            )
        };
        let spec_conv = GdnCausalConvDecodeKernelConfig {
            backends: cfg.causal_conv_decode_backends.clone(),
            gpu_name: gpu.clone(),
            channels: conv_channels.into(),
            kernel_size: cfg.conv_kernel_size.into(),
            dtype: cfg.activation_dtype,
            state_dtype: cfg.conv_state_dtype,
            tokens_per_request: Some(q.into()),
        };
        VllmBf16GdnSpecLocalWorkletResolved {
            fused_verify,
            qkvz: gemm(conv_channels + value_width, cfg.hidden),
            ba: gemm(2 * cfg.num_value_heads, cfg.hidden),
            core_output_zero: torch(0, value_row),
            verify_conv: spec_conv.clone(),
            verify_recurrent_norm: GdnRecurrentDecodeGatedNormKernelConfig {
                backends: cfg.fused_verify_backends.clone(),
                gpu_name: gpu.clone(),
                tokens_per_request: q.into(),
                num_qk_heads: cfg.num_key_heads.into(),
                num_value_heads: cfg.num_value_heads.into(),
                key_head_dim: cfg.key_head_dim.into(),
                value_head_dim: cfg.value_head_dim.into(),
                dtype: cfg.activation_dtype,
                state_dtype: cfg.ssm_state_dtype,
            },
            ba_contiguous: strided_copy(cfg.num_value_heads * act),
            qkv_select: rows(
                "conv_channels",
                conv_channels,
                cfg.activation_dtype,
                "index_select",
            ),
            ab_select: rows(
                "num_value_heads",
                cfg.num_value_heads,
                cfg.activation_dtype,
                "index_select",
            ),
            spec_decode: GdnDecodeOpConfig {
                causal_conv: spec_conv,
                recurrent: GdnRecurrentDecodeKernelConfig {
                    backends: cfg.sigmoid_gating_decode_backends.clone(),
                    gpu_name: gpu.clone(),
                    num_qk_heads: cfg.num_key_heads.into(),
                    num_value_heads: cfg.num_value_heads.into(),
                    key_head_dim: cfg.key_head_dim.into(),
                    value_head_dim: cfg.value_head_dim.into(),
                    dtype: cfg.activation_dtype,
                    state_dtype: cfg.ssm_state_dtype,
                    tokens_per_request: Some(q.into()),
                },
            },
            spec_qk_copy: strided_copy(key_width * act),
            spec_v_copy: strided_copy(value_row),
            spec_qkv_cat: elementwise(
                &cfg.cat_backends,
                gpu,
                conv_channels * act,
                conv_channels * act,
            ),
            prefill: GdnPrefillOpConfig {
                causal_conv: GdnCausalConvPrefillKernelConfig {
                    backends: cfg.causal_conv_prefill_backends.clone(),
                    gpu_name: gpu.clone(),
                    channels: conv_channels.into(),
                    kernel_size: cfg.conv_kernel_size.into(),
                    dtype: cfg.activation_dtype,
                    state_dtype: cfg.conv_state_dtype,
                },
                post_conv: GdnPrefillPostConvKernelConfig {
                    backends: cfg.prefill_post_conv_backends.clone(),
                    gpu_name: gpu.clone(),
                    num_qk_heads: cfg.num_key_heads.into(),
                    num_value_heads: cfg.num_value_heads.into(),
                    key_head_dim: cfg.key_head_dim.into(),
                    value_head_dim: cfg.value_head_dim.into(),
                    dtype: cfg.activation_dtype,
                },
                chunk_delta_rule: GdnChunkDeltaRuleKernelConfig {
                    backends: cfg.chunk_delta_rule_backends.clone(),
                    gpu_name: gpu.clone(),
                    num_key_heads: cfg.num_key_heads.into(),
                    num_heads: cfg.num_value_heads.into(),
                    key_head_dim: cfg.key_head_dim.into(),
                    value_head_dim: cfg.value_head_dim.into(),
                    dtype: cfg.activation_dtype,
                },
            },
            state_gather: state("gather"),
            state_zero: state("masked_zero"),
            state_scatter: state("scatter"),
            cu_seqlens_cast: torch(INDEX_BYTES, INDEX_BYTES),
            g_exp: torch(
                cfg.num_value_heads * DType::Fp32.size_bytes(),
                cfg.num_value_heads * DType::Fp32.size_bytes(),
            ),
            output_index_copy: rows(
                "value_width",
                value_width,
                cfg.activation_dtype,
                "index_copy",
            ),
            spec_output_copy: torch(value_row, value_row),
            output_gate_copy: strided_copy(value_row),
            gated_norm: GdnGatedRmsNormKernelConfig {
                backends: cfg.gated_rms_norm_backends.clone(),
                gpu_name: gpu.clone(),
                hidden: cfg.value_head_dim.into(),
                dtype: cfg.activation_dtype,
            },
            out_proj: gemm(cfg.hidden, value_width),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: VllmBf16GdnSpecLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        let gemm = |suffix: &str, config: SingleGemmKernelConfig| {
            atomic(n, suffix, config, SingleGemmKernel::build, bridge)
        };
        let ew = |suffix: &str, config: ElementwiseKernelConfig| {
            atomic(n, suffix, config, ElementwiseKernel::build, bridge)
        };
        let rows = |suffix: &str, config: IndexedRowCopyKernelConfig| {
            atomic(n, suffix, config, IndexedRowCopyKernel::build, bridge)
        };
        Ok(Self {
            qkvz: gemm("qkvz", r.qkvz)?,
            ba: gemm("ba", r.ba)?,
            core_output_zero: ew("core_output_zero", r.core_output_zero)?,
            verify_conv: atomic(
                n,
                "verify_conv",
                r.verify_conv,
                GdnCausalConvDecodeKernel::build,
                bridge,
            )?,
            verify_recurrent_norm: atomic(
                n,
                "verify_recurrent_norm",
                r.verify_recurrent_norm,
                GdnRecurrentDecodeGatedNormKernel::build,
                bridge,
            )?,
            ba_contiguous: ew("ba_contiguous", r.ba_contiguous)?,
            spec_qkv_select: rows("spec_qkv_select", r.qkv_select.clone())?,
            spec_ab_select: rows("spec_ab_select", r.ab_select.clone())?,
            non_spec_qkv_select: rows("non_spec_qkv_select", r.qkv_select)?,
            non_spec_ab_select: rows("non_spec_ab_select", r.ab_select)?,
            spec_decode: GdnDecodeOp::build(format!("{n}.spec_decode"), r.spec_decode, bridge)?,
            spec_qk_copy: ew("spec_qk_copy", r.spec_qk_copy)?,
            spec_v_copy: ew("spec_v_copy", r.spec_v_copy)?,
            spec_qkv_cat: ew("spec_qkv_cat", r.spec_qkv_cat)?,
            prefill: GdnPrefillOp::build(format!("{n}.prefill"), r.prefill, bridge)?,
            state_gather: rows("state_gather", r.state_gather)?,
            state_zero: rows("state_zero", r.state_zero)?,
            state_scatter: rows("state_scatter", r.state_scatter)?,
            cu_seqlens_cast: ew("cu_seqlens_cast", r.cu_seqlens_cast)?,
            g_exp: ew("g_exp", r.g_exp)?,
            spec_output_index_copy: rows("spec_output_index_copy", r.output_index_copy.clone())?,
            non_spec_output_index_copy: rows("non_spec_output_index_copy", r.output_index_copy)?,
            spec_output_copy: ew("spec_output_copy", r.spec_output_copy)?,
            output_gate_copy: ew("output_gate_copy", r.output_gate_copy)?,
            gated_norm: atomic(
                n,
                "gated_norm",
                r.gated_norm,
                GdnGatedRmsNormKernel::build,
                bridge,
            )?,
            out_proj: gemm("out_proj", r.out_proj)?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        let children = vec![
            self.qkvz.compile(builder),
            self.ba.compile(builder),
            self.core_output_zero.compile(builder),
            self.verify_conv.compile(builder),
            self.verify_recurrent_norm.compile(builder),
            repeated(&self.ba_contiguous, 2, builder),
            self.spec_qkv_select.compile(builder),
            repeated(&self.spec_ab_select, 2, builder),
            self.non_spec_qkv_select.compile(builder),
            repeated(&self.non_spec_ab_select, 2, builder),
            self.spec_decode.compile(builder),
            repeated(&self.spec_qk_copy, 2, builder),
            self.spec_v_copy.compile(builder),
            self.spec_qkv_cat.compile(builder),
            self.prefill.compile(builder),
            self.state_gather.compile(builder),
            self.state_zero.compile(builder),
            self.state_scatter.compile(builder),
            self.cu_seqlens_cast.compile(builder),
            self.g_exp.compile(builder),
            self.spec_output_index_copy.compile(builder),
            self.non_spec_output_index_copy.compile(builder),
            self.spec_output_copy.compile(builder),
            self.output_gate_copy.compile(builder),
            self.gated_norm.compile(builder),
            self.out_proj.compile(builder),
        ];
        CostNode::Labeled {
            label: format!(
                "{} (VllmBf16GdnSpecLocalWorklet) [local (1 GPU); Hk={}, Hv={}, K={}, V={}, q={}]",
                self.name,
                cfg.num_key_heads,
                cfg.num_value_heads,
                cfg.key_head_dim,
                cfg.value_head_dim,
                cfg.tokens_per_request
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &VllmBf16GdnSpecLocalWorkletInput, ev: &mut Evaluator) {
        let cfg = &self.resolved.raw_cfg;
        let w = Work::of(input, cfg.tokens_per_request, self.resolved.fused_verify);
        let tokens = |num_tokens: u32| ElementwiseKernelInput { num_tokens };
        let rows = |num_rows: u32| IndexedRowCopyKernelInput { num_rows };
        let fused = w.path == Path::FusedVerify;
        let unfused = matches!(w.path, Path::Verify | Path::Mixed | Path::Prefill);
        let mixed = w.path == Path::Mixed;
        let spec = matches!(w.path, Path::Verify | Path::Mixed);
        let prefill = matches!(w.path, Path::Mixed | Path::Prefill);
        let gemm = |m: u32| SingleGemmKernelInput { m };

        push_or_zero(&self.qkvz, gemm(w.padded), w.padded == 0, ev);
        push_or_zero(&self.ba, gemm(w.padded), w.padded == 0, ev);
        push_or_zero(&self.core_output_zero, tokens(w.padded), w.padded == 0, ev);
        push_or_zero(
            &self.verify_conv,
            GdnCausalConvDecodeKernelInput {
                batch_size: w.spec_decodes,
            },
            !fused,
            ev,
        );
        push_or_zero(
            &self.verify_recurrent_norm,
            GdnRecurrentDecodeGatedNormKernelInput {
                batch_size: w.spec_decodes,
            },
            !fused,
            ev,
        );
        push_or_zero(&self.ba_contiguous, tokens(w.real), !unfused, ev);
        push_or_zero(&self.spec_qkv_select, rows(w.spec_tokens), !mixed, ev);
        push_or_zero(&self.spec_ab_select, rows(w.spec_tokens), !mixed, ev);
        push_or_zero(
            &self.non_spec_qkv_select,
            rows(w.prefill_tokens),
            !mixed,
            ev,
        );
        push_or_zero(&self.non_spec_ab_select, rows(w.prefill_tokens), !mixed, ev);
        self.spec_decode.eval(
            &GdnDecodeOpInput {
                batch_size: if spec { w.spec_decodes } else { 0 },
            },
            ev,
        );
        push_or_zero(&self.spec_qk_copy, tokens(w.spec_tokens), !spec, ev);
        push_or_zero(&self.spec_v_copy, tokens(w.spec_tokens), !spec, ev);
        push_or_zero(&self.spec_qkv_cat, tokens(w.spec_tokens), !spec, ev);
        self.prefill.eval(
            &GdnPrefillOpInput {
                sequence_lengths: w.prefill_lengths.clone(),
                decode_rows: 0,
            },
            ev,
        );
        push_or_zero(&self.state_gather, rows(w.prefills), !prefill, ev);
        push_or_zero(&self.state_zero, rows(w.prefills), !prefill, ev);
        push_or_zero(&self.state_scatter, rows(w.prefills), !prefill, ev);
        push_or_zero(&self.cu_seqlens_cast, tokens(w.prefills + 1), !prefill, ev);
        push_or_zero(&self.g_exp, tokens(w.prefill_tokens), !prefill, ev);
        push_or_zero(
            &self.spec_output_index_copy,
            rows(w.spec_tokens),
            !mixed,
            ev,
        );
        push_or_zero(
            &self.non_spec_output_index_copy,
            rows(w.prefill_tokens),
            !mixed,
            ev,
        );
        push_or_zero(
            &self.spec_output_copy,
            tokens(w.spec_tokens),
            w.path != Path::Verify,
            ev,
        );
        push_or_zero(&self.output_gate_copy, tokens(w.real), !unfused, ev);
        push_or_zero(
            &self.gated_norm,
            GdnGatedRmsNormKernelInput {
                m: w.real * cfg.num_value_heads,
            },
            !unfused,
            ev,
        );
        push_or_zero(&self.out_proj, gemm(w.padded), w.padded == 0, ev);
    }
}

/// The core path one step takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    /// No tokens.
    Empty,
    /// Verify rows only, through `fused_gdn_decode_post_conv_mtp`.
    FusedVerify,
    /// Verify rows only, through the unfused spec path.
    Verify,
    /// Verify rows beside prefill rows.
    Mixed,
    /// Prefill rows only.
    Prefill,
}

/// The row counts of one step's launches.
#[derive(Debug)]
struct Work {
    path: Path,
    padded: u32,
    /// Real tokens: the eager core's rows.
    real: u32,
    spec_decodes: u32,
    spec_tokens: u32,
    /// Chunk-path sequences: the prefills, plus one-token non-spec decodes
    /// reclassified as prefills beside spec decodes.
    prefill_lengths: Vec<u32>,
    prefills: u32,
    prefill_tokens: u32,
}

impl Work {
    fn of(input: &VllmBf16GdnSpecLocalWorkletInput, q: u32, fused_verify: bool) -> Self {
        debug_assert!(
            input.non_spec_decodes == 0 || input.spec_decodes > 0,
            "non-spec decodes without spec decodes take the plain decode path, not modeled here"
        );
        let mut prefill_lengths = input.prefill_lengths.clone();
        // gdn_attn.py:307-314: with spec decodes present, non-spec decodes are
        // one-token prefills.
        if input.spec_decodes > 0 {
            prefill_lengths.extend(std::iter::repeat(1).take(input.non_spec_decodes as usize));
        }
        let prefills = prefill_lengths.len() as u32;
        let prefill_tokens: u32 = prefill_lengths.iter().sum();
        let spec_tokens = input.spec_decodes * q;
        let path = match (input.spec_decodes > 0, prefills > 0) {
            (false, false) => Path::Empty,
            (true, false) if fused_verify => Path::FusedVerify,
            (true, false) => Path::Verify,
            (true, true) => Path::Mixed,
            (false, true) => Path::Prefill,
        };
        Self {
            path,
            padded: input.padded_tokens,
            real: spec_tokens + prefill_tokens,
            spec_decodes: input.spec_decodes,
            spec_tokens,
            prefill_lengths,
            prefills,
            prefill_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VllmBf16GdnSpecLocalWorkletConfig {
        VllmBf16GdnSpecLocalWorkletConfig {
            hidden: 2560,
            num_key_heads: 16,
            num_value_heads: 48,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_size: 4,
            tokens_per_request: 4,
            activation_dtype: DType::Bf16,
            conv_state_dtype: DType::Bf16,
            ssm_state_dtype: DType::Fp32,
            gpu_name: "NVIDIA B200".into(),
            gemm_backends: vec!["torch_linear_vllm"],
            torch_elementwise_backends: vec!["torch"],
            strided_copy_backends: vec!["torch_strided_copy"],
            cat_backends: vec!["torch_cat"],
            indexed_row_copy_backends: vec!["torch"],
            causal_conv_prefill_backends: vec!["vllm_triton"],
            prefill_post_conv_backends: vec!["vllm_triton"],
            chunk_delta_rule_backends: vec!["flashinfer"],
            causal_conv_decode_backends: vec!["vllm_triton"],
            sigmoid_gating_decode_backends: vec!["vllm_triton_sigmoid_gating"],
            fused_verify_backends: vec!["vllm_cuda"],
            gated_rms_norm_backends: vec!["vllm_triton"],
        }
    }

    fn input(prefills: &[u32], spec: u32, non_spec: u32) -> VllmBf16GdnSpecLocalWorkletInput {
        VllmBf16GdnSpecLocalWorkletInput {
            padded_tokens: prefills.iter().sum::<u32>() + 4 * spec + non_spec,
            prefill_lengths: prefills.to_vec(),
            spec_decodes: spec,
            non_spec_decodes: non_spec,
        }
    }

    #[test]
    fn shapes_are_the_profiled_bf16_rows() {
        let r = VllmBf16GdnSpecLocalWorklet::resolve_config(&cfg());
        assert_eq!((r.qkvz.n.get(), r.qkvz.k.get()), (16384, 2560));
        assert_eq!((r.ba.n.get(), r.ba.k.get()), (96, 2560));
        assert_eq!((r.out_proj.n.get(), r.out_proj.k.get()), (2560, 6144));
        assert_eq!(r.qkvz.dtype, DType::Bf16);
        assert_eq!(r.qkv_select.row_elements.get(), 10240);
        assert_eq!(r.ab_select.row_elements.get(), 48);
        assert_eq!(r.output_index_copy.row_elements.get(), 6144);
        assert_eq!(r.state_gather.row_elements.get(), 786_432);
        assert_eq!(r.output_gate_copy.input_bytes_per_token.get(), 12288);
        assert_eq!(r.verify_recurrent_norm.tokens_per_request.get(), 4);
        assert!(r.fused_verify);
    }

    #[test]
    fn a_verify_only_step_takes_the_fused_kernel_at_the_real_requests() {
        let mut step = input(&[], 5, 0);
        step.padded_tokens = 24;
        let w = Work::of(&step, 4, true);
        assert_eq!(w.path, Path::FusedVerify);
        assert_eq!(
            (w.spec_decodes, w.spec_tokens, w.real, w.padded),
            (5, 20, 20, 24)
        );
        assert_eq!(w.prefills, 0);
    }

    #[test]
    fn verify_past_the_fused_kernel_limits_takes_the_unfused_spec_path() {
        let long = VllmBf16GdnSpecLocalWorkletConfig {
            tokens_per_request: 9,
            ..cfg()
        };
        assert!(!VllmBf16GdnSpecLocalWorklet::resolve_config(&long).fused_verify);
        let odd_ratio = VllmBf16GdnSpecLocalWorkletConfig {
            num_value_heads: 96,
            ..cfg()
        };
        assert!(!VllmBf16GdnSpecLocalWorklet::resolve_config(&odd_ratio).fused_verify);
        assert_eq!(Work::of(&input(&[], 3, 0), 9, false).path, Path::Verify);
    }

    #[test]
    fn a_mixed_step_splits_spec_and_prefill_rows() {
        let w = Work::of(&input(&[1536, 100], 128, 0), 4, true);
        assert_eq!(w.path, Path::Mixed);
        assert_eq!(
            (w.spec_tokens, w.prefills, w.prefill_tokens),
            (512, 2, 1636)
        );
        assert_eq!(w.real, 2148);
    }

    #[test]
    fn non_spec_decodes_beside_spec_decodes_run_as_one_token_prefills() {
        // gdn_attn.py:307-314.
        let w = Work::of(&input(&[], 2, 3), 4, true);
        assert_eq!(w.path, Path::Mixed);
        assert_eq!(w.prefill_lengths, [1, 1, 1]);
        assert_eq!((w.prefills, w.prefill_tokens, w.real), (3, 3, 11));
    }

    #[test]
    fn a_prefill_only_step_takes_the_chunk_path() {
        let w = Work::of(&input(&[2048], 0, 0), 4, true);
        assert_eq!(w.path, Path::Prefill);
        assert_eq!((w.spec_tokens, w.prefill_tokens, w.real), (0, 2048, 2048));
        assert_eq!(Work::of(&input(&[], 0, 0), 4, true).path, Path::Empty);
    }

    #[test]
    fn compile_has_one_slot_per_leaf() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let worklet = VllmBf16GdnSpecLocalWorklet::build(
            "gdn".into(),
            VllmBf16GdnSpecLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        // 24 atomic leaves plus the spec decode (2) and prefill (3) ops.
        assert_eq!(tree.n_slots(), 29);
    }
}
