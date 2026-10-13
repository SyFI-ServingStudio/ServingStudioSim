//! One Qwen3.5-family Gated DeltaNet token mixer on one GPU, in vLLM's launch
//! granularity (`qwen_gdn_linear_attn.py`, ModelOpt mixed-precision
//! checkpoints: FP8 per-tensor `in_proj_qkvz` / `out_proj`, BF16 `in_proj_ba`).
//!
//! The section starts after the input RMSNorm (the arch owns it; it already
//! wrote the FP8 copy `in_proj_qkvz` reads) and ends at the `out_proj` output.
//!
//! ```text
//! compiled piece   qkvz GEMM, ba GEMM, core_attn_out zeros, b / a contiguous
//! core (eager)     decode-only: causal_conv1d_update + packed recurrent decode
//!                  with a prefill: varlen conv over every token, post-conv over
//!                  the prefill tokens, the prefills' state gather / zero-fresh /
//!                  scatter around the chunked delta rule, exp(g); a mixed step
//!                  also rearranges the decode rows' q/k/v and runs the
//!                  sigmoid-gating recurrent update on them, then concatenates
//!                  decode and prefill outputs
//!                  core_attn_out copy
//! compiled piece   gated RMSNorm, static FP8 quant, out_proj GEMM
//! ```
//!
//! Under FULL_AND_PIECEWISE graphs the compiled pieces run at the graph-padded
//! row count, and so does the core in a FULL-graph (uniform decode) replay; a
//! piecewise replay runs the core (`vllm::qwen_gdn_attention_core`, a splitting
//! op) eagerly over the real tokens.
//!
//! The glue launches (TensorIterator copies, cat, exp, the index_put / gather /
//! masked_fill state movement) are 4-6% of a measured step's forward time
//! (diverse_100 capture, B200), so each is a leaf.

use crate::op::ssm::{
    GdnDecodeOp, GdnDecodeOpConfig, GdnDecodeOpInput, GdnPrefillOp, GdnPrefillOpConfig,
    GdnPrefillOpInput,
};
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    GdnCausalConvDecodeKernelConfig, GdnCausalConvPrefillKernelConfig,
    GdnChunkDeltaRuleKernelConfig, GdnGatedRmsNormKernel, GdnGatedRmsNormKernelConfig,
    GdnGatedRmsNormKernelInput, GdnPrefillPostConvKernelConfig, GdnRecurrentDecodeKernel,
    GdnRecurrentDecodeKernelConfig, GdnRecurrentDecodeKernelInput, IndexedRowCopyKernel,
    IndexedRowCopyKernelConfig, IndexedRowCopyKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput, StaticFp8QuantKernel,
    StaticFp8QuantKernelConfig, StaticFp8QuantKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, elementwise, push_or_zero, repeated};

/// An int64 index or bool mask element.
const INDEX_BYTES: u32 = 8;

#[derive(Clone, Debug)]
pub struct VllmQwen35GdnLocalWorkletConfig {
    pub hidden: u32,
    pub num_key_heads: u32,
    pub num_value_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel_size: u32,
    pub activation_dtype: DType,
    pub conv_state_dtype: DType,
    pub ssm_state_dtype: DType,
    /// `in_proj_qkvz` and `out_proj` weights.
    pub proj_dtype: DType,
    pub gpu_name: String,
    pub proj_gemm_backends: Vec<&'static str>,
    pub ba_gemm_backends: Vec<&'static str>,
    pub static_quant_backends: Vec<&'static str>,
    /// Compiled (Triton) elementwise launches.
    pub triton_elementwise_backends: Vec<&'static str>,
    /// Eager TensorIterator maps over contiguous tensors (`b.contiguous()`, index
    /// copies, `g.exp()`).
    pub torch_elementwise_backends: Vec<&'static str>,
    /// Eager copies whose source is a non-contiguous view; TensorIterator cannot
    /// vectorize them (the decode q/k/v slices and the core-output copy-back).
    pub strided_copy_backends: Vec<&'static str>,
    /// Eager `torch.cat`.
    pub cat_backends: Vec<&'static str>,
    pub state_copy_backends: Vec<&'static str>,
    pub causal_conv_prefill_backends: Vec<&'static str>,
    pub prefill_post_conv_backends: Vec<&'static str>,
    pub chunk_delta_rule_backends: Vec<&'static str>,
    pub causal_conv_decode_backends: Vec<&'static str>,
    /// `fused_recurrent_gated_delta_rule_packed_decode` (decode-only steps).
    pub packed_decode_backends: Vec<&'static str>,
    /// `fused_sigmoid_gating_delta_rule_update` (a mixed step's decode rows).
    pub sigmoid_gating_decode_backends: Vec<&'static str>,
    pub gated_rms_norm_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct VllmQwen35GdnLocalWorkletResolved {
    pub raw_cfg: VllmQwen35GdnLocalWorkletConfig,
    pub qkvz: SingleGemmKernelConfig,
    pub ba: SingleGemmKernelConfig,
    pub core_output_zero: ElementwiseKernelConfig,
    /// `b.contiguous()` and `a.contiguous()`, one leaf folded twice.
    pub ba_contiguous: ElementwiseKernelConfig,
    pub decode: GdnDecodeOpConfig,
    pub prefill: GdnPrefillOpConfig,
    /// `rearrange_mixed_qkv` of the decode rows: q and k copies (folded twice),
    /// the v copy, and the cat into one buffer.
    pub decode_qk_copy: ElementwiseKernelConfig,
    pub decode_v_copy: ElementwiseKernelConfig,
    pub decode_qkv_cat: ElementwiseKernelConfig,
    pub mixed_decode_recurrent: GdnRecurrentDecodeKernelConfig,
    /// The prefill state-index / initial-state-mask copies, three launches.
    pub prefill_index_copy: ElementwiseKernelConfig,
    pub state_gather: IndexedRowCopyKernelConfig,
    pub state_zero: IndexedRowCopyKernelConfig,
    pub state_scatter: IndexedRowCopyKernelConfig,
    /// `g.exp()` over the prefill tokens' FP32 decay.
    pub g_exp: ElementwiseKernelConfig,
    pub output_cat: ElementwiseKernelConfig,
    pub core_output_copy: ElementwiseKernelConfig,
    pub gated_norm: GdnGatedRmsNormKernelConfig,
    pub out_quant: StaticFp8QuantKernelConfig,
    pub out_proj: SingleGemmKernelConfig,
}

/// One step as the GDN layer sees it.
#[derive(Clone, Debug, Default)]
pub struct VllmQwen35GdnLocalWorkletInput {
    /// Rows the compiled pieces run: the graph-padded token count.
    pub padded_tokens: u32,
    /// Rows a FULL-graph replay runs the core at (the padded count), or 0
    /// when the core runs eagerly over the real tokens.
    pub full_graph_rows: u32,
    /// Prefill chunk lengths of the step's prefill requests.
    pub prefill_lengths: Vec<u32>,
    pub decode_requests: u32,
}

pub struct VllmQwen35GdnLocalWorklet {
    pub name: String,
    qkvz: Op<SingleGemmKernel>,
    ba: Op<SingleGemmKernel>,
    core_output_zero: Op<ElementwiseKernel>,
    ba_contiguous: Op<ElementwiseKernel>,
    decode: GdnDecodeOp,
    prefill: GdnPrefillOp,
    decode_qk_copy: Op<ElementwiseKernel>,
    decode_v_copy: Op<ElementwiseKernel>,
    decode_qkv_cat: Op<ElementwiseKernel>,
    mixed_decode_recurrent: Op<GdnRecurrentDecodeKernel>,
    prefill_index_copy: Op<ElementwiseKernel>,
    state_gather: Op<IndexedRowCopyKernel>,
    state_zero: Op<IndexedRowCopyKernel>,
    state_scatter: Op<IndexedRowCopyKernel>,
    g_exp: Op<ElementwiseKernel>,
    output_cat: Op<ElementwiseKernel>,
    core_output_copy: Op<ElementwiseKernel>,
    gated_norm: Op<GdnGatedRmsNormKernel>,
    out_quant: Op<StaticFp8QuantKernel>,
    out_proj: Op<SingleGemmKernel>,
    resolved: VllmQwen35GdnLocalWorkletResolved,
}

impl VllmQwen35GdnLocalWorklet {
    pub fn resolve_config(
        cfg: &VllmQwen35GdnLocalWorkletConfig,
    ) -> VllmQwen35GdnLocalWorkletResolved {
        let gpu = &cfg.gpu_name;
        let act = cfg.activation_dtype.size_bytes();
        let key_width = cfg.num_key_heads * cfg.key_head_dim;
        let value_width = cfg.num_value_heads * cfg.value_head_dim;
        let conv_channels = 2 * key_width + value_width;
        let gemm =
            |backends: &[&'static str], n: u32, k: u32, dtype: DType| SingleGemmKernelConfig {
                backends: backends.to_vec(),
                gpu_name: gpu.clone(),
                n: n.into(),
                k: k.into(),
                dtype,
            };
        let triton = |input: u32, output: u32| {
            elementwise(&cfg.triton_elementwise_backends, gpu, input, output)
        };
        let torch = |input: u32, output: u32| {
            elementwise(&cfg.torch_elementwise_backends, gpu, input, output)
        };
        let strided_copy =
            |bytes: u32| elementwise(&cfg.strided_copy_backends, gpu, bytes, bytes);
        let cat = |bytes: u32| elementwise(&cfg.cat_backends, gpu, bytes, bytes);
        let state = |operation: &str| IndexedRowCopyKernelConfig {
            backends: cfg.state_copy_backends.clone(),
            gpu_name: gpu.clone(),
            row_elements: Dim::param(
                "ssm_state_elements",
                cfg.num_value_heads * cfg.key_head_dim * cfg.value_head_dim,
            ),
            dtype: cfg.ssm_state_dtype,
            operation: operation.to_string(),
        };
        let recurrent = |backends: &[&'static str]| GdnRecurrentDecodeKernelConfig {
            backends: backends.to_vec(),
            gpu_name: gpu.clone(),
            num_qk_heads: cfg.num_key_heads.into(),
            num_value_heads: cfg.num_value_heads.into(),
            key_head_dim: cfg.key_head_dim.into(),
            value_head_dim: cfg.value_head_dim.into(),
            dtype: cfg.activation_dtype,
            state_dtype: cfg.ssm_state_dtype,
        };
        let value_row = value_width * act;
        VllmQwen35GdnLocalWorkletResolved {
            qkvz: gemm(
                &cfg.proj_gemm_backends,
                conv_channels + value_width,
                cfg.hidden,
                cfg.proj_dtype,
            ),
            ba: gemm(
                &cfg.ba_gemm_backends,
                2 * cfg.num_value_heads,
                cfg.hidden,
                cfg.activation_dtype,
            ),
            core_output_zero: triton(0, value_row),
            ba_contiguous: torch(cfg.num_value_heads * act, cfg.num_value_heads * act),
            decode: GdnDecodeOpConfig {
                causal_conv: GdnCausalConvDecodeKernelConfig {
                    backends: cfg.causal_conv_decode_backends.clone(),
                    gpu_name: gpu.clone(),
                    channels: conv_channels.into(),
                    kernel_size: cfg.conv_kernel_size.into(),
                    dtype: cfg.activation_dtype,
                    state_dtype: cfg.conv_state_dtype,
                },
                recurrent: recurrent(&cfg.packed_decode_backends),
            },
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
            decode_qk_copy: strided_copy(key_width * act),
            decode_v_copy: strided_copy(value_row),
            decode_qkv_cat: cat(conv_channels * act),
            mixed_decode_recurrent: recurrent(&cfg.sigmoid_gating_decode_backends),
            prefill_index_copy: torch(INDEX_BYTES, INDEX_BYTES),
            state_gather: state("gather"),
            state_zero: state("masked_zero"),
            state_scatter: state("scatter"),
            g_exp: torch(
                cfg.num_value_heads * DType::Fp32.size_bytes(),
                cfg.num_value_heads * DType::Fp32.size_bytes(),
            ),
            output_cat: cat(value_row),
            core_output_copy: strided_copy(value_row),
            gated_norm: GdnGatedRmsNormKernelConfig {
                backends: cfg.gated_rms_norm_backends.clone(),
                gpu_name: gpu.clone(),
                hidden: cfg.value_head_dim.into(),
                dtype: cfg.activation_dtype,
            },
            out_quant: StaticFp8QuantKernelConfig {
                backends: cfg.static_quant_backends.clone(),
                gpu_name: gpu.clone(),
                hidden_dim: value_width.into(),
                gated: false,
                dtype: cfg.activation_dtype,
            },
            out_proj: gemm(
                &cfg.proj_gemm_backends,
                cfg.hidden,
                value_width,
                cfg.proj_dtype,
            ),
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: VllmQwen35GdnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        let ew = |suffix: &str, config: ElementwiseKernelConfig| {
            atomic(n, suffix, config, ElementwiseKernel::build, bridge)
        };
        let state = |suffix: &str, config: IndexedRowCopyKernelConfig| {
            atomic(n, suffix, config, IndexedRowCopyKernel::build, bridge)
        };
        Ok(Self {
            qkvz: atomic(n, "qkvz", r.qkvz, SingleGemmKernel::build, bridge)?,
            ba: atomic(n, "ba", r.ba, SingleGemmKernel::build, bridge)?,
            core_output_zero: ew("core_output_zero", r.core_output_zero)?,
            ba_contiguous: ew("ba_contiguous", r.ba_contiguous)?,
            decode: GdnDecodeOp::build(format!("{n}.decode"), r.decode, bridge)?,
            prefill: GdnPrefillOp::build(format!("{n}.prefill"), r.prefill, bridge)?,
            decode_qk_copy: ew("decode_qk_copy", r.decode_qk_copy)?,
            decode_v_copy: ew("decode_v_copy", r.decode_v_copy)?,
            decode_qkv_cat: ew("decode_qkv_cat", r.decode_qkv_cat)?,
            mixed_decode_recurrent: atomic(
                n,
                "mixed_decode_recurrent",
                r.mixed_decode_recurrent,
                GdnRecurrentDecodeKernel::build,
                bridge,
            )?,
            prefill_index_copy: ew("prefill_index_copy", r.prefill_index_copy)?,
            state_gather: state("state_gather", r.state_gather)?,
            state_zero: state("state_zero", r.state_zero)?,
            state_scatter: state("state_scatter", r.state_scatter)?,
            g_exp: ew("g_exp", r.g_exp)?,
            output_cat: ew("output_cat", r.output_cat)?,
            core_output_copy: ew("core_output_copy", r.core_output_copy)?,
            gated_norm: atomic(
                n,
                "gated_norm",
                r.gated_norm,
                GdnGatedRmsNormKernel::build,
                bridge,
            )?,
            out_quant: atomic(
                n,
                "out_quant",
                r.out_quant,
                StaticFp8QuantKernel::build,
                bridge,
            )?,
            out_proj: atomic(n, "out_proj", r.out_proj, SingleGemmKernel::build, bridge)?,
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
            repeated(&self.ba_contiguous, 2, builder),
            self.decode.compile(builder),
            self.prefill.compile(builder),
            repeated(&self.decode_qk_copy, 2, builder),
            self.decode_v_copy.compile(builder),
            self.decode_qkv_cat.compile(builder),
            self.mixed_decode_recurrent.compile(builder),
            repeated(&self.prefill_index_copy, 3, builder),
            self.state_gather.compile(builder),
            self.state_zero.compile(builder),
            self.state_scatter.compile(builder),
            self.g_exp.compile(builder),
            self.output_cat.compile(builder),
            self.core_output_copy.compile(builder),
            self.gated_norm.compile(builder),
            self.out_quant.compile(builder),
            self.out_proj.compile(builder),
        ];
        CostNode::Labeled {
            label: format!(
                "{} (VllmQwen35GdnLocalWorklet) [local (1 GPU); Hk={}, Hv={}, K={}, V={}]",
                self.name,
                cfg.num_key_heads,
                cfg.num_value_heads,
                cfg.key_head_dim,
                cfg.value_head_dim
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &VllmQwen35GdnLocalWorkletInput, ev: &mut Evaluator) {
        let w = Work::of(input);
        let tokens = |num_tokens: u32| ElementwiseKernelInput { num_tokens };
        let rows = |num_rows: u32| IndexedRowCopyKernelInput { num_rows };
        let decode_only = w.prefills == 0;
        let mixed = w.mixed_decodes > 0;
        push_or_zero(&self.qkvz, SingleGemmKernelInput { m: w.padded }, false, ev);
        push_or_zero(&self.ba, SingleGemmKernelInput { m: w.padded }, false, ev);
        push_or_zero(&self.core_output_zero, tokens(w.padded), false, ev);
        push_or_zero(&self.ba_contiguous, tokens(w.padded), false, ev);
        self.decode.eval(
            &GdnDecodeOpInput {
                batch_size: w.packed_decode_rows,
            },
            ev,
        );
        self.prefill.eval(&w.prefill, ev);
        push_or_zero(&self.decode_qk_copy, tokens(w.mixed_decodes), !mixed, ev);
        push_or_zero(&self.decode_v_copy, tokens(w.mixed_decodes), !mixed, ev);
        push_or_zero(&self.decode_qkv_cat, tokens(w.mixed_decodes), !mixed, ev);
        push_or_zero(
            &self.mixed_decode_recurrent,
            GdnRecurrentDecodeKernelInput {
                batch_size: w.mixed_decodes,
            },
            !mixed,
            ev,
        );
        push_or_zero(
            &self.prefill_index_copy,
            tokens(w.prefills),
            decode_only,
            ev,
        );
        push_or_zero(&self.state_gather, rows(w.prefills), decode_only, ev);
        push_or_zero(&self.state_zero, rows(w.prefills), decode_only, ev);
        push_or_zero(&self.state_scatter, rows(w.prefills), decode_only, ev);
        push_or_zero(&self.g_exp, tokens(w.prefill_tokens), decode_only, ev);
        push_or_zero(&self.output_cat, tokens(w.output_cat_rows), !mixed, ev);
        push_or_zero(&self.core_output_copy, tokens(w.core), false, ev);
        push_or_zero(
            &self.gated_norm,
            GdnGatedRmsNormKernelInput {
                m: w.padded * self.resolved.raw_cfg.num_value_heads,
            },
            false,
            ev,
        );
        push_or_zero(
            &self.out_quant,
            StaticFp8QuantKernelInput {
                num_tokens: w.padded,
            },
            false,
            ev,
        );
        push_or_zero(
            &self.out_proj,
            SingleGemmKernelInput { m: w.padded },
            false,
            ev,
        );
    }
}

/// The row counts of one step's launches.
#[derive(Debug)]
struct Work {
    padded: u32,
    /// Rows the core runs: the FULL-graph padded rows, else the real tokens.
    core: u32,
    prefills: u32,
    prefill_tokens: u32,
    /// The packed decode path's rows (a decode-only step).
    packed_decode_rows: u32,
    /// A mixed step's decode rows (sigmoid-gating path), else 0.
    mixed_decodes: u32,
    output_cat_rows: u32,
    prefill: GdnPrefillOpInput,
}

impl Work {
    fn of(input: &VllmQwen35GdnLocalWorkletInput) -> Self {
        let prefill_tokens: u32 = input.prefill_lengths.iter().sum();
        let real = prefill_tokens + input.decode_requests;
        let core = if input.full_graph_rows > 0 {
            input.full_graph_rows
        } else {
            real
        };
        let prefills = input.prefill_lengths.len() as u32;
        let mixed = prefills > 0 && input.decode_requests > 0;
        Self {
            padded: input.padded_tokens,
            core,
            prefills,
            prefill_tokens,
            packed_decode_rows: if prefills == 0 { core } else { 0 },
            mixed_decodes: if mixed { input.decode_requests } else { 0 },
            output_cat_rows: if mixed { real } else { 0 },
            prefill: GdnPrefillOpInput {
                sequence_lengths: input.prefill_lengths.clone(),
                decode_rows: input.decode_requests,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VllmQwen35GdnLocalWorkletConfig {
        VllmQwen35GdnLocalWorkletConfig {
            hidden: 5120,
            num_key_heads: 16,
            num_value_heads: 48,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_size: 4,
            activation_dtype: DType::Bf16,
            conv_state_dtype: DType::Bf16,
            ssm_state_dtype: DType::Fp32,
            proj_dtype: DType::Fp8E4m3,
            gpu_name: "NVIDIA B200".into(),
            proj_gemm_backends: vec!["flashinfer_fp8_tensor"],
            ba_gemm_backends: vec!["torch_linear_vllm"],
            static_quant_backends: vec!["vllm_inductor"],
            triton_elementwise_backends: vec!["triton"],
            torch_elementwise_backends: vec!["torch"],
            strided_copy_backends: vec!["torch_strided_copy"],
            cat_backends: vec!["torch_cat"],
            state_copy_backends: vec!["torch"],
            causal_conv_prefill_backends: vec!["vllm_triton"],
            prefill_post_conv_backends: vec!["vllm_triton"],
            chunk_delta_rule_backends: vec!["flashinfer"],
            causal_conv_decode_backends: vec!["vllm_triton"],
            packed_decode_backends: vec!["vllm_triton"],
            sigmoid_gating_decode_backends: vec!["vllm_triton_sigmoid_gating"],
            gated_rms_norm_backends: vec!["vllm_triton"],
        }
    }

    #[test]
    fn shapes_are_the_profiled_qwen38_rows() {
        let r = VllmQwen35GdnLocalWorklet::resolve_config(&cfg());
        assert_eq!((r.qkvz.n.get(), r.qkvz.k.get()), (16384, 5120));
        assert_eq!((r.ba.n.get(), r.ba.k.get()), (96, 5120));
        assert_eq!((r.out_proj.n.get(), r.out_proj.k.get()), (5120, 6144));
        assert_eq!(r.prefill.causal_conv.channels.get(), 10240);
        assert_eq!(r.state_gather.row_elements.get(), 786_432);
        assert_eq!(r.out_quant.hidden_dim.get(), 6144);
        assert_eq!(r.core_output_zero.output_bytes_per_token.get(), 12288);
        assert_eq!(r.decode_qkv_cat.input_bytes_per_token.get(), 20480);
    }

    #[test]
    fn a_full_graph_decode_runs_the_packed_core_at_the_padded_rows() {
        let w = Work::of(&VllmQwen35GdnLocalWorkletInput {
            padded_tokens: 8,
            full_graph_rows: 8,
            prefill_lengths: vec![],
            decode_requests: 5,
        });
        assert_eq!((w.packed_decode_rows, w.core, w.mixed_decodes), (8, 8, 0));
        assert_eq!(w.prefills, 0);
    }

    #[test]
    fn a_mixed_step_convolves_the_decode_rows_with_the_prefills() {
        let w = Work::of(&VllmQwen35GdnLocalWorkletInput {
            padded_tokens: 1440,
            full_graph_rows: 0,
            prefill_lengths: vec![629, 805],
            decode_requests: 4,
        });
        assert_eq!(w.prefill.sequence_lengths, [629, 805]);
        assert_eq!(w.prefill.decode_rows, 4);
        assert_eq!((w.packed_decode_rows, w.mixed_decodes), (0, 4));
        assert_eq!((w.prefills, w.prefill_tokens), (2, 1434));
        assert_eq!((w.output_cat_rows, w.core, w.padded), (1438, 1438, 1440));
    }

    #[test]
    fn compile_has_one_slot_per_leaf() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let worklet = VllmQwen35GdnLocalWorklet::build(
            "gdn".into(),
            VllmQwen35GdnLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        // 18 atomic leaves plus the decode (2) and prefill (3) ops.
        assert_eq!(tree.n_slots(), 23);
    }
}
