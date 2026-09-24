//! DFlash2 draft decoder layer — the decode-shaped half of a draft call.
//!
//! Split into two rank-local sections because the architecture owns the two
//! tensor-parallel collectives inside a layer: one after `o_proj`, one after
//! the FFN's down projection.
//!
//! What makes this layer unlike the GLM-5.2 MTP layer it replaces:
//!
//! - **The whole draft runs once, not `draft_tokens` times.** DFlash drafts in
//!   parallel, so a request submits `1 + num_speculative_tokens` query rows
//!   (one bonus token plus the mask tokens) in a single forward pass. There is
//!   no recurrence and therefore no folded recurrent pass.
//! - **Attention is non-causal.** Every query row in the block attends the whole
//!   context, so the shape is a rectangle `(q_len, kv_len)` rather than a causal
//!   prefix/append split — which is exactly what `flashinfer_attn_rect` is
//!   parametrized by.
//! - **Four grouped convolutions per layer.** `attention_conv` and `mlp_conv`
//!   each wrap their section with a `prepare` (projection + convolution) and a
//!   `finish` (convolution reusing the prepared coefficients).
//!
//! Measured against `logs/20260920_0_glm53_dflash2_phase0` (B200, tp=4): the
//! grouped convolutions are **not** fused by inductor — they run as plain ATen
//! binary element-wise kernels, three of which together are 22% of the `draft`
//! phase across 97 launches per iteration. Pricing them with the `elementwise`
//! leaf is therefore the faithful identity, not an approximation: the trap
//! `vllm_mla_rope` documents (an inductor fusion that rewrites a whole tensor to
//! change a slice of it) is a different situation.
//!
//! Over the 296 pure-decode iterations of the Phase 0 capture, the draft phase
//! is 1.675 ms of a 22.296 ms decode iteration (7.5%), and attention within it
//! is 0.131 ms — 0.59% of the iteration, over six launches.
//!
//! # The attention leaf is measured in bf16, not fp8
//!
//! The engine attends **fp8** KV with an fp8 query. No `flashinfer_attn_rect`
//! backend can measure that on B200 today: `fa2` has no fp8 tensor-core path,
//! `cudnn` is bf16-only, `fa3`'s ragged kernels are built for SM90 and have no
//! Blackwell image, and `trt` (Blackwell-only) has no ragged kernel at all. The
//! leaf is therefore measured at bf16/bf16, which reads twice the KV bytes the
//! engine does, so it **overstates** this term and errs slow rather than fast.
//!
//! The capture measures the overstatement rather than leaving it to argument:
//! `dflash2.layer.attn.attention` is 0.131 ms measured against 0.199 ms
//! simulated per decode iteration, **+52%**, or 0.069 ms — **0.31% of a decode
//! iteration**. Two things differ at once and the split between them is not
//! separable here: the dtype, and the implementation (the engine runs vLLM's
//! CuTe SM100 FlashAttention, the leaf is profiled against FlashInfer `fa2`).
//! A pure-bandwidth argument would predict +100% from the dtype alone, so the
//! implementation difference is plainly working the other way.
//!
//! This is second in line behind `selector.candidate_topk`, which overstates by
//! 0.357 ms per decode iteration — 5.2x more. It is also a different kind of
//! repair: closing it needs a Blackwell ragged fp8 rect kernel, or a CuTe SM100
//! identity, at L1. Recomposing this worklet cannot reach it.

use std::sync::Arc;

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput, FlashinferAttnRectKernel,
    FlashinferAttnRectKernelConfig, FlashinferAttnRectKernelInput, ResidualRmsNormKernel,
    ResidualRmsNormKernelConfig, ResidualRmsNormKernelInput, RmsNormKernel, RmsNormKernelConfig,
    RmsNormKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dflash2DraftAttnLog, Dim, Evaluator, LeafMetrics,
    PerfApiBridge, Probe, SlotInput,
};

const HIDDEN_DIM: u32 = 6144;
const INTERMEDIATE_DIM: u32 = 12288;
const HEAD_DIM: u32 = 128;
const NUM_QO_HEADS: u32 = 64;
const NUM_KV_HEADS: u32 = 8;
/// `dflash_config.conv_kernel_size` — the convolution reaches back one position.
const CONV_TAPS: u32 = 2;
/// `dflash_config.conv_group_size`.
const CONV_GROUP_SIZE: u32 = 16;

/// Shared identity for both halves of one draft layer.
#[derive(Clone, Debug)]
pub struct Dflash2DraftLayerLocalWorkletConfig {
    pub residual_norm_backends: Vec<&'static str>,
    pub norm_backends: Vec<&'static str>,
    pub gemm_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
    pub attention_backends: Vec<&'static str>,
    pub tp_size: u16,
    pub gpu_name: String,
    pub hidden_dim: Dim,
    pub intermediate_dim: Dim,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub conv_taps: u32,
    pub conv_group_size: u32,
    pub dtype: DType,
    pub gemm_dtype: DType,
    /// Dtypes the attention leaf is *measured* at, which are not always the
    /// dtypes the engine runs. The engine attends fp8 KV with an fp8 query (the
    /// capture's server log notes the missing q scale being set from k_scale,
    /// which only matters to fp8 attention backends), but no B200 rect backend
    /// can measure that today -- see the module note. Kept as explicit fields
    /// rather than derived from the cache dtype so the substitution is visible
    /// at the call site instead of hidden in a default.
    pub attn_q_dtype: DType,
    pub attn_kv_dtype: DType,
}

/// One grouped convolution's byte rates. `prepare` also runs a projection; both
/// halves run the convolution itself, which reads the block and its one-position
/// shift and writes the block back.
fn conv_elementwise(
    cfg: &Dflash2DraftLayerLocalWorkletConfig,
    width: u32,
) -> Result<ElementwiseKernelConfig, String> {
    let dtype_bytes = cfg.dtype.size_bytes();
    let input_bytes = checked_product(
        "grouped_conv.input_bytes_per_token",
        &[cfg.conv_taps, width, dtype_bytes],
    )?;
    let output_bytes =
        checked_product("grouped_conv.output_bytes_per_token", &[width, dtype_bytes])?;
    Ok(ElementwiseKernelConfig {
        backends: cfg.elementwise_backends.clone(),
        gpu_name: cfg.gpu_name.clone(),
        input_bytes_per_token: input_bytes.into(),
        output_bytes_per_token: output_bytes.into(),
    })
}

/// `kernel_projection` is a `ReplicatedLinear`: `hidden -> 2 * taps * num_groups`,
/// unsharded, one per grouped convolution pair.
fn conv_projection(
    cfg: &Dflash2DraftLayerLocalWorkletConfig,
) -> Result<SingleGemmKernelConfig, String> {
    let num_groups = cfg.hidden_dim.get() / cfg.conv_group_size;
    let n = checked_product("conv_projection.n", &[2, cfg.conv_taps, num_groups])?;
    Ok(SingleGemmKernelConfig {
        backends: cfg.gemm_backends.clone(),
        gpu_name: cfg.gpu_name.clone(),
        n: n.into(),
        k: cfg.hidden_dim.clone(),
        dtype: cfg.gemm_dtype,
    })
}

// ---------------------------------------------------------------- attention --

#[derive(Clone, Debug)]
pub struct Dflash2DraftAttnLocalWorkletResolved {
    pub raw_cfg: Dflash2DraftLayerLocalWorkletConfig,
    pub input_add_rms_norm: ResidualRmsNormKernelConfig,
    pub attn_conv_projection: SingleGemmKernelConfig,
    pub attn_conv_prepare: ElementwiseKernelConfig,
    pub qkv_proj: SingleGemmKernelConfig,
    pub q_norm: RmsNormKernelConfig,
    pub k_norm: RmsNormKernelConfig,
    pub rope: ElementwiseKernelConfig,
    pub attention: FlashinferAttnRectKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Dflash2DraftAttnLocalWorkletInput {
    /// Query rows this section forwards: `requests * (1 + draft_tokens)`.
    pub query_tokens: u32,
    /// One `(q_len, kv_len)` rectangle per request. `q_len` is the whole query
    /// block, `kv_len` the context it attends — there is no causal split.
    pub rectangles: Vec<(u32, u32)>,
}

pub struct Dflash2DraftAttnLocalWorklet {
    pub name: String,
    pub input_add_rms_norm: Op<ResidualRmsNormKernel>,
    pub attn_conv_projection: Op<SingleGemmKernel>,
    pub attn_conv_prepare: Op<ElementwiseKernel>,
    pub qkv_proj: Op<SingleGemmKernel>,
    pub q_norm: Op<RmsNormKernel>,
    pub k_norm: Op<RmsNormKernel>,
    pub rope: Op<ElementwiseKernel>,
    pub attention: Op<FlashinferAttnRectKernel>,
    pub o_proj: Op<SingleGemmKernel>,
    resolved: Dflash2DraftAttnLocalWorkletResolved,
}

impl Dflash2DraftAttnLocalWorklet {
    pub fn resolve_config(
        cfg: &Dflash2DraftLayerLocalWorkletConfig,
    ) -> Dflash2DraftAttnLocalWorkletResolved {
        validate_config(cfg).unwrap_or_else(|reason| {
            panic!("invalid Dflash2DraftLayerLocalWorkletConfig: {reason}")
        });

        let tp = Dim::param("attn_tp", u32::from(cfg.tp_size));
        let qo_heads_per_rank = cfg.num_qo_heads.clone() / tp.clone();
        let kv_heads_per_rank = cfg.num_kv_heads.clone() / tp;
        let dtype_bytes = cfg.dtype.size_bytes();
        let q_width = checked_product(
            "qkv_proj.q_width",
            &[qo_heads_per_rank.get(), cfg.head_dim.get()],
        )
        .expect("validated DFlash2 query width must fit u32");
        let kv_width = checked_product(
            "qkv_proj.kv_width",
            &[kv_heads_per_rank.get(), cfg.head_dim.get()],
        )
        .expect("validated DFlash2 KV width must fit u32");
        let qkv_n = q_width + 2 * kv_width;
        // RoPE rotates Q and K in place and gathers one cos/sin row per token.
        let rope_width = q_width + kv_width;
        let rope_input_bytes =
            checked_product("rope.input_bytes_per_token", &[2, rope_width, dtype_bytes])
                .expect("validated DFlash2 RoPE input byte rate must fit u32");
        let rope_output_bytes =
            checked_product("rope.output_bytes_per_token", &[rope_width, dtype_bytes])
                .expect("validated DFlash2 RoPE output byte rate must fit u32");

        Dflash2DraftAttnLocalWorkletResolved {
            input_add_rms_norm: ResidualRmsNormKernelConfig {
                backends: cfg.residual_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden_dim.clone(),
                dtype: cfg.dtype,
            },
            attn_conv_projection: conv_projection(cfg)
                .expect("validated DFlash2 conv projection must fit u32"),
            attn_conv_prepare: conv_elementwise(cfg, cfg.hidden_dim.get())
                .expect("validated DFlash2 conv byte rate must fit u32"),
            qkv_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: qkv_n.into(),
                k: cfg.hidden_dim.clone(),
                dtype: cfg.gemm_dtype,
            },
            q_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.head_dim.clone(),
                dtype: cfg.dtype,
            },
            k_norm: RmsNormKernelConfig {
                backends: cfg.norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.head_dim.clone(),
                dtype: cfg.dtype,
            },
            rope: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: rope_input_bytes.into(),
                output_bytes_per_token: rope_output_bytes.into(),
            },
            attention: FlashinferAttnRectKernelConfig {
                backends: cfg.attention_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_qo_heads: qo_heads_per_rank,
                num_kv_heads: kv_heads_per_rank,
                head_dim: cfg.head_dim.clone(),
                q_dtype: cfg.attn_q_dtype,
                kv_dtype: cfg.attn_kv_dtype,
                o_dtype: cfg.dtype,
            },
            o_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden_dim.clone(),
                k: q_width.into(),
                dtype: cfg.gemm_dtype,
            },
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Dflash2DraftAttnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        Ok(Self {
            input_add_rms_norm: build_atomic(
                &name,
                "input_add_rms_norm",
                resolved.input_add_rms_norm.clone(),
                ResidualRmsNormKernel::build,
                bridge,
            )?,
            attn_conv_projection: build_atomic(
                &name,
                "attn_conv_projection",
                resolved.attn_conv_projection.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            attn_conv_prepare: build_atomic(
                &name,
                "attn_conv_prepare",
                resolved.attn_conv_prepare.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            qkv_proj: build_atomic(
                &name,
                "qkv_proj",
                resolved.qkv_proj.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            q_norm: build_atomic(
                &name,
                "q_norm",
                resolved.q_norm.clone(),
                RmsNormKernel::build,
                bridge,
            )?,
            k_norm: build_atomic(
                &name,
                "k_norm",
                resolved.k_norm.clone(),
                RmsNormKernel::build,
                bridge,
            )?,
            rope: build_atomic(
                &name,
                "rope",
                resolved.rope.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            attention: build_atomic(
                &name,
                "attention",
                resolved.attention.clone(),
                FlashinferAttnRectKernel::build,
                bridge,
            )?,
            o_proj: build_atomic(
                &name,
                "o_proj",
                resolved.o_proj.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: attn_label(&self.name, &self.resolved.raw_cfg),
            child: Box::new(CostNode::Sum(vec![
                self.input_add_rms_norm.compile(builder),
                self.attn_conv_projection.compile(builder),
                self.attn_conv_prepare.compile(builder),
                self.qkv_proj.compile(builder),
                self.q_norm.compile(builder),
                self.k_norm.compile(builder),
                self.rope.compile(builder),
                self.attention.compile(builder),
                self.o_proj.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Dflash2DraftAttnLocalWorkletInput, ev: &mut Evaluator) {
        let tokens = input.query_tokens;
        let zero = tokens == 0;
        let qo_heads = self.resolved.attention.num_qo_heads.get();
        let kv_heads = self.resolved.attention.num_kv_heads.get();

        eval_atomic_or_zero(
            &self.input_add_rms_norm,
            ResidualRmsNormKernelInput { m: tokens },
            zero,
            ev,
        );
        eval_atomic_or_zero(
            &self.attn_conv_projection,
            SingleGemmKernelInput { m: tokens },
            zero,
            ev,
        );
        eval_atomic_or_zero(
            &self.attn_conv_prepare,
            ElementwiseKernelInput { num_tokens: tokens },
            zero,
            ev,
        );
        eval_atomic_or_zero(
            &self.qkv_proj,
            SingleGemmKernelInput { m: tokens },
            zero,
            ev,
        );
        eval_atomic_or_zero(
            &self.q_norm,
            RmsNormKernelInput {
                m: tokens.saturating_mul(qo_heads),
            },
            zero,
            ev,
        );
        eval_atomic_or_zero(
            &self.k_norm,
            RmsNormKernelInput {
                m: tokens.saturating_mul(kv_heads),
            },
            zero,
            ev,
        );
        eval_atomic_or_zero(
            &self.rope,
            ElementwiseKernelInput { num_tokens: tokens },
            zero,
            ev,
        );

        // One aggregating slot, priced as ONE launch over every request's
        // rectangle: the engine runs a single varlen FlashAttention call per
        // layer, not one per request. See `batched_rectangle`. `add_fanin` (not
        // `add`) so the slot carries the selected backend rather than the ZERO
        // accumulator's sentinel, which would read as "never executed".
        let mut attention = LeafMetrics::ZERO;
        if let Some((q_len, kv_len)) = batched_rectangle(&input.rectangles) {
            attention.add_fanin(
                self.attention
                    .kernel
                    .eval(&FlashinferAttnRectKernelInput { q_len, kv_len }),
            );
        }
        ev.push(attention, || {
            Dflash2DraftAttnLog {
                rectangles: input.rectangles.clone(),
            }
            .into()
        });

        eval_atomic_or_zero(&self.o_proj, SingleGemmKernelInput { m: tokens }, zero, ev);
    }
}

// ---------------------------------------------------------------------- FFN --

#[derive(Clone, Debug)]
pub struct Dflash2DraftFfnLocalWorkletResolved {
    pub raw_cfg: Dflash2DraftLayerLocalWorkletConfig,
    pub attn_conv_finish: ElementwiseKernelConfig,
    pub post_attn_add_rms_norm: ResidualRmsNormKernelConfig,
    pub mlp_conv_projection: SingleGemmKernelConfig,
    pub mlp_conv_prepare: ElementwiseKernelConfig,
    pub gate_up_proj: SingleGemmKernelConfig,
    pub silu_and_mul: ElementwiseKernelConfig,
    pub down_proj: SingleGemmKernelConfig,
    pub mlp_conv_finish: ElementwiseKernelConfig,
}

#[derive(Clone, Debug, Default)]
pub struct Dflash2DraftFfnLocalWorkletInput {
    pub query_tokens: u32,
}

pub struct Dflash2DraftFfnLocalWorklet {
    pub name: String,
    pub attn_conv_finish: Op<ElementwiseKernel>,
    pub post_attn_add_rms_norm: Op<ResidualRmsNormKernel>,
    pub mlp_conv_projection: Op<SingleGemmKernel>,
    pub mlp_conv_prepare: Op<ElementwiseKernel>,
    pub gate_up_proj: Op<SingleGemmKernel>,
    pub silu_and_mul: Op<ElementwiseKernel>,
    pub down_proj: Op<SingleGemmKernel>,
    pub mlp_conv_finish: Op<ElementwiseKernel>,
    resolved: Dflash2DraftFfnLocalWorkletResolved,
}

impl Dflash2DraftFfnLocalWorklet {
    pub fn resolve_config(
        cfg: &Dflash2DraftLayerLocalWorkletConfig,
    ) -> Dflash2DraftFfnLocalWorkletResolved {
        validate_config(cfg).unwrap_or_else(|reason| {
            panic!("invalid Dflash2DraftLayerLocalWorkletConfig: {reason}")
        });

        let intermediate_per_rank =
            cfg.intermediate_dim.clone() / Dim::param("ffn_tp", u32::from(cfg.tp_size));
        let dtype_bytes = cfg.dtype.size_bytes();
        let gate_up_n = checked_product("gate_up_proj.n", &[2, intermediate_per_rank.get()])
            .expect("validated DFlash2 gate/up width must fit u32");
        let silu_input_bytes = checked_product(
            "silu_and_mul.input_bytes_per_token",
            &[2, intermediate_per_rank.get(), dtype_bytes],
        )
        .expect("validated DFlash2 SiLU input byte rate must fit u32");
        let silu_output_bytes = checked_product(
            "silu_and_mul.output_bytes_per_token",
            &[intermediate_per_rank.get(), dtype_bytes],
        )
        .expect("validated DFlash2 SiLU output byte rate must fit u32");
        let conv = conv_elementwise(cfg, cfg.hidden_dim.get())
            .expect("validated DFlash2 conv byte rate must fit u32");

        Dflash2DraftFfnLocalWorkletResolved {
            attn_conv_finish: conv.clone(),
            post_attn_add_rms_norm: ResidualRmsNormKernelConfig {
                backends: cfg.residual_norm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                hidden: cfg.hidden_dim.clone(),
                dtype: cfg.dtype,
            },
            mlp_conv_projection: conv_projection(cfg)
                .expect("validated DFlash2 conv projection must fit u32"),
            mlp_conv_prepare: conv.clone(),
            gate_up_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: gate_up_n.into(),
                k: cfg.hidden_dim.clone(),
                dtype: cfg.gemm_dtype,
            },
            silu_and_mul: ElementwiseKernelConfig {
                backends: cfg.elementwise_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                input_bytes_per_token: silu_input_bytes.into(),
                output_bytes_per_token: silu_output_bytes.into(),
            },
            down_proj: SingleGemmKernelConfig {
                backends: cfg.gemm_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                n: cfg.hidden_dim.clone(),
                k: intermediate_per_rank,
                dtype: cfg.gemm_dtype,
            },
            mlp_conv_finish: conv,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Dflash2DraftFfnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        Ok(Self {
            attn_conv_finish: build_atomic(
                &name,
                "attn_conv_finish",
                resolved.attn_conv_finish.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            post_attn_add_rms_norm: build_atomic(
                &name,
                "post_attn_add_rms_norm",
                resolved.post_attn_add_rms_norm.clone(),
                ResidualRmsNormKernel::build,
                bridge,
            )?,
            mlp_conv_projection: build_atomic(
                &name,
                "mlp_conv_projection",
                resolved.mlp_conv_projection.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            mlp_conv_prepare: build_atomic(
                &name,
                "mlp_conv_prepare",
                resolved.mlp_conv_prepare.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            gate_up_proj: build_atomic(
                &name,
                "gate_up_proj",
                resolved.gate_up_proj.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            silu_and_mul: build_atomic(
                &name,
                "silu_and_mul",
                resolved.silu_and_mul.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            down_proj: build_atomic(
                &name,
                "down_proj",
                resolved.down_proj.clone(),
                SingleGemmKernel::build,
                bridge,
            )?,
            mlp_conv_finish: build_atomic(
                &name,
                "mlp_conv_finish",
                resolved.mlp_conv_finish.clone(),
                ElementwiseKernel::build,
                bridge,
            )?,
            name,
            resolved,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: ffn_label(&self.name, &self.resolved.raw_cfg),
            child: Box::new(CostNode::Sum(vec![
                self.attn_conv_finish.compile(builder),
                self.post_attn_add_rms_norm.compile(builder),
                self.mlp_conv_projection.compile(builder),
                self.mlp_conv_prepare.compile(builder),
                self.gate_up_proj.compile(builder),
                self.silu_and_mul.compile(builder),
                self.down_proj.compile(builder),
                self.mlp_conv_finish.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Dflash2DraftFfnLocalWorkletInput, ev: &mut Evaluator) {
        let tokens = input.query_tokens;
        let zero = tokens == 0;
        let whole = ElementwiseKernelInput { num_tokens: tokens };

        eval_atomic_or_zero(&self.attn_conv_finish, whole.clone(), zero, ev);
        eval_atomic_or_zero(
            &self.post_attn_add_rms_norm,
            ResidualRmsNormKernelInput { m: tokens },
            zero,
            ev,
        );
        eval_atomic_or_zero(
            &self.mlp_conv_projection,
            SingleGemmKernelInput { m: tokens },
            zero,
            ev,
        );
        eval_atomic_or_zero(&self.mlp_conv_prepare, whole.clone(), zero, ev);
        eval_atomic_or_zero(
            &self.gate_up_proj,
            SingleGemmKernelInput { m: tokens },
            zero,
            ev,
        );
        eval_atomic_or_zero(&self.silu_and_mul, whole.clone(), zero, ev);
        eval_atomic_or_zero(
            &self.down_proj,
            SingleGemmKernelInput { m: tokens },
            zero,
            ev,
        );
        // Billed in this section although it runs after the architecture's
        // collective. `CostNode::Sum` is additive, so the total is unchanged;
        // only the cost-log narrative places it one boundary early.
        eval_atomic_or_zero(&self.mlp_conv_finish, whole, zero, ev);
    }
}

// ------------------------------------------------------------------ shared --

fn validate_config(cfg: &Dflash2DraftLayerLocalWorkletConfig) -> Result<(), String> {
    for (name, actual, required) in [
        ("hidden_dim", cfg.hidden_dim.get(), HIDDEN_DIM),
        (
            "intermediate_dim",
            cfg.intermediate_dim.get(),
            INTERMEDIATE_DIM,
        ),
        ("head_dim", cfg.head_dim.get(), HEAD_DIM),
        ("num_qo_heads", cfg.num_qo_heads.get(), NUM_QO_HEADS),
        ("num_kv_heads", cfg.num_kv_heads.get(), NUM_KV_HEADS),
        ("conv_taps", cfg.conv_taps, CONV_TAPS),
        ("conv_group_size", cfg.conv_group_size, CONV_GROUP_SIZE),
    ] {
        if actual != required {
            return Err(format!("{name} must be {required}, got {actual}"));
        }
    }
    if cfg.dtype != DType::Bf16 {
        return Err(format!(
            "dtype must be {}, got {}",
            DType::Bf16.as_str(),
            cfg.dtype.as_str()
        ));
    }
    if cfg.tp_size == 0 {
        return Err("tp_size must be positive".to_string());
    }
    // The ragged rect path requires one compute dtype for both operands.
    if cfg.attn_q_dtype != cfg.attn_kv_dtype {
        return Err(format!(
            "ragged rect attention requires attn_q_dtype == attn_kv_dtype, got {} and {}",
            cfg.attn_q_dtype.as_str(),
            cfg.attn_kv_dtype.as_str()
        ));
    }
    for (name, value) in [
        ("num_qo_heads", cfg.num_qo_heads.get()),
        ("num_kv_heads", cfg.num_kv_heads.get()),
        ("intermediate_dim", cfg.intermediate_dim.get()),
    ] {
        if value % u32::from(cfg.tp_size) != 0 {
            return Err(format!(
                "{name} {value} must be divisible by tp_size {}",
                cfg.tp_size
            ));
        }
    }
    Ok(())
}

fn checked_product(name: &str, factors: &[u32]) -> Result<u32, String> {
    factors.iter().try_fold(1_u32, |value, &factor| {
        value
            .checked_mul(factor)
            .ok_or_else(|| format!("{name} overflows u32"))
    })
}

fn attn_label(name: &str, cfg: &Dflash2DraftLayerLocalWorkletConfig) -> String {
    format!(
        "{name} (Dflash2DraftAttnLocalWorklet) \
         [rank-local; tp={}; non-causal; qo/rank={}; kv/rank={}; head_dim={}]",
        cfg.tp_size,
        cfg.num_qo_heads.get() / u32::from(cfg.tp_size),
        cfg.num_kv_heads.get() / u32::from(cfg.tp_size),
        cfg.head_dim
    )
}

fn ffn_label(name: &str, cfg: &Dflash2DraftLayerLocalWorkletConfig) -> String {
    format!(
        "{name} (Dflash2DraftFfnLocalWorklet) \
         [rank-local; tp={}; intermediate/rank={}; grouped conv taps={}]",
        cfg.tp_size,
        cfg.intermediate_dim.get() / u32::from(cfg.tp_size),
        cfg.conv_taps
    )
}

fn build_atomic<K, C, F>(
    parent: &str,
    suffix: &str,
    config: C,
    build: F,
    bridge: &PerfApiBridge,
) -> Result<Op<K>, BuildError>
where
    K: Probe,
    F: FnOnce(String, C, &PerfApiBridge) -> Result<K, BuildError>,
{
    let name = format!("{parent}.{suffix}");
    Ok(Op::new(
        name.clone(),
        Arc::new(build(name, config, bridge)?),
    ))
}

/// The single rectangle one batched draft-attention launch is priced as:
/// `q_len = sum(q_i)` and `kv_len = sum(q_i * kv_i) / sum(q_i)`, rounded up.
///
/// The rect leaf is profiled one request at a time, so summing it per request
/// bills the launch and its fixed latency once per request. The engine makes one
/// varlen call per layer, and on the v0.28 captures
/// (`logs/20260923_4_glm53_dflash2_pack`) the per-request sum overstated
/// `dflash2.layer.attn.attention` 55x at 256 concurrent requests (811 ms simulated
/// vs 14.6 ms measured over the case) and 9x at 32; the error grew with the
/// request count, not the context. One rectangle keeps the attention work
/// (`sum(q_i * kv_i)`, every query row against its own keys) and bills one
/// launch. It reads the keys of one mean-length context rather than every
/// request's own, which errs fast on a KV-bound call; the leaf is measured at
/// bf16 against the engine's fp8, which errs slow (see the module note).
fn batched_rectangle(rectangles: &[(u32, u32)]) -> Option<(u32, u32)> {
    let q_total: u64 = rectangles.iter().map(|&(q, _)| u64::from(q)).sum();
    if q_total == 0 {
        return None;
    }
    let work: u64 = rectangles
        .iter()
        .map(|&(q, kv)| u64::from(q) * u64::from(kv))
        .sum();
    let q_len = u32::try_from(q_total).expect("draft query rows fit u32");
    let kv_len = u32::try_from(work.div_ceil(q_total)).expect("mean kv_len fits u32");
    Some((q_len, kv_len))
}

fn eval_atomic_or_zero<K>(op: &Op<K>, input: K::Input, zero: bool, ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    let metrics = if zero {
        LeafMetrics::ZERO
    } else {
        op.kernel.eval(&input)
    };
    ev.push(metrics, || input.clone().into());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Dflash2DraftLayerLocalWorkletConfig {
        Dflash2DraftLayerLocalWorkletConfig {
            residual_norm_backends: vec!["vllm_cuda"],
            norm_backends: vec!["vllm_cuda"],
            gemm_backends: vec!["torch"],
            elementwise_backends: vec!["triton"],
            attention_backends: vec!["flashinfer"],
            tp_size: 4,
            gpu_name: "NVIDIA B200".to_string(),
            hidden_dim: Dim::param("hidden_dim", HIDDEN_DIM),
            intermediate_dim: Dim::param("intermediate_dim", INTERMEDIATE_DIM),
            num_qo_heads: Dim::param("num_qo_heads", NUM_QO_HEADS),
            num_kv_heads: Dim::param("num_kv_heads", NUM_KV_HEADS),
            head_dim: Dim::param("head_dim", HEAD_DIM),
            conv_taps: CONV_TAPS,
            conv_group_size: CONV_GROUP_SIZE,
            dtype: DType::Bf16,
            gemm_dtype: DType::Bf16,
            attn_q_dtype: DType::Bf16,
            attn_kv_dtype: DType::Bf16,
        }
    }

    #[test]
    fn a_batch_of_rectangles_is_one_launch_with_the_same_work() {
        assert_eq!(batched_rectangle(&[]), None);
        assert_eq!(batched_rectangle(&[(0, 100)]), None);
        assert_eq!(batched_rectangle(&[(8, 100)]), Some((8, 100)));
        // 8*100 + 8*301 = 3208 query-key pairs over 16 rows: 200.5, rounded up.
        assert_eq!(batched_rectangle(&[(8, 100), (8, 301)]), Some((16, 201)));
    }

    #[test]
    fn attention_shards_heads_but_the_conv_projection_stays_whole() {
        let resolved = Dflash2DraftAttnLocalWorklet::resolve_config(&cfg());
        assert_eq!(resolved.attention.num_qo_heads.get(), 16);
        assert_eq!(resolved.attention.num_kv_heads.get(), 2);
        // `kernel_projection` is a ReplicatedLinear: 2 * taps * (6144 / 16).
        assert_eq!(resolved.attn_conv_projection.n.get(), 1536);
        assert_eq!(resolved.attn_conv_projection.k.get(), HIDDEN_DIM);
    }

    #[test]
    fn qkv_and_output_projections_follow_the_head_shard() {
        let resolved = Dflash2DraftAttnLocalWorklet::resolve_config(&cfg());
        // (16 q + 2 k + 2 v) heads * 128.
        assert_eq!(resolved.qkv_proj.n.get(), 2560);
        // o_proj reads only this rank's query heads.
        assert_eq!(resolved.o_proj.k.get(), 16 * HEAD_DIM);
        assert_eq!(resolved.o_proj.n.get(), HIDDEN_DIM);
    }

    #[test]
    fn the_ffn_shards_the_intermediate_width() {
        let resolved = Dflash2DraftFfnLocalWorklet::resolve_config(&cfg());
        assert_eq!(resolved.gate_up_proj.n.get(), 2 * INTERMEDIATE_DIM / 4);
        assert_eq!(resolved.down_proj.k.get(), INTERMEDIATE_DIM / 4);
    }

    #[test]
    fn a_conv_group_size_other_than_the_checkpoint_is_rejected() {
        let mut bad = cfg();
        bad.conv_group_size = 7;
        assert!(validate_config(&bad).is_err());
    }

    #[test]
    fn a_head_count_that_does_not_shard_is_rejected() {
        let mut bad = cfg();
        bad.tp_size = 16;
        assert!(validate_config(&bad).is_err());
    }
}
