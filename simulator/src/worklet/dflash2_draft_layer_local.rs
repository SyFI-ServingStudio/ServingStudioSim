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
//! The draft phase as a whole is 1.61 ms of a 16.90 ms decode iteration (9.5%),
//! and attention within it is 5.7% of that — 0.54% of the iteration, over six
//! launches.

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
    pub kv_dtype: DType,
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
    let output_bytes = checked_product("grouped_conv.output_bytes_per_token", &[width, dtype_bytes])?;
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

#[cfg(test)]
const ATTN_SOURCE_ORDER: [&str; 9] = [
    "input_add_rms_norm",
    "attn_conv_projection",
    "attn_conv_prepare",
    "qkv_proj",
    "q_norm",
    "k_norm",
    "rope",
    "attention",
    "o_proj",
];

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
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid Dflash2DraftLayerLocalWorkletConfig: {reason}"));

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
        let qkv_n = checked_product("qkv_proj.n", &[1, q_width + 2 * kv_width])
            .expect("validated DFlash2 QKV width must fit u32");
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
                q_dtype: cfg.dtype,
                kv_dtype: cfg.kv_dtype,
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
            ElementwiseKernelInput {
                num_tokens: tokens,
            },
            zero,
            ev,
        );
        eval_atomic_or_zero(&self.qkv_proj, SingleGemmKernelInput { m: tokens }, zero, ev);
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
            ElementwiseKernelInput {
                num_tokens: tokens,
            },
            zero,
            ev,
        );

        // One aggregating slot over every request's rectangle. `add_fanin` (not
        // `add`) so the slot carries the selected backend rather than the ZERO
        // accumulator's sentinel, which would read as "never executed".
        let mut attention = LeafMetrics::ZERO;
        for &(q_len, kv_len) in &input.rectangles {
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

#[cfg(test)]
const FFN_SOURCE_ORDER: [&str; 7] = [
    "attn_conv_finish",
    "post_attn_add_rms_norm",
    "mlp_conv_projection",
    "mlp_conv_prepare",
    "gate_up_proj",
    "silu_and_mul",
    "down_proj",
];

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
        validate_config(cfg)
            .unwrap_or_else(|reason| panic!("invalid Dflash2DraftLayerLocalWorkletConfig: {reason}"));

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
        let whole = ElementwiseKernelInput {
            num_tokens: tokens,
        };

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
        ("intermediate_dim", cfg.intermediate_dim.get(), INTERMEDIATE_DIM),
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
    if cfg.hidden_dim.get() % cfg.conv_group_size != 0 {
        return Err(format!(
            "conv_group_size {} must divide hidden_dim {}",
            cfg.conv_group_size, cfg.hidden_dim
        ));
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
            kv_dtype: DType::Fp8E4m3,
        }
    }

    #[test]
    fn the_attention_section_is_exactly_the_source_order() {
        assert_eq!(
            ATTN_SOURCE_ORDER,
            [
                "input_add_rms_norm",
                "attn_conv_projection",
                "attn_conv_prepare",
                "qkv_proj",
                "q_norm",
                "k_norm",
                "rope",
                "attention",
                "o_proj",
            ]
        );
    }

    #[test]
    fn the_ffn_section_carries_both_halves_of_both_convolutions() {
        // Four grouped convolutions per layer: attention prepare/finish and MLP
        // prepare/finish. `prepare` lives with its own section's projection, so
        // this half owns the attention finish and both MLP halves.
        assert_eq!(FFN_SOURCE_ORDER.len(), 7);
        assert!(FFN_SOURCE_ORDER.contains(&"attn_conv_finish"));
        assert!(FFN_SOURCE_ORDER.contains(&"mlp_conv_prepare"));
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
    fn a_conv_group_size_that_does_not_divide_hidden_is_rejected() {
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
