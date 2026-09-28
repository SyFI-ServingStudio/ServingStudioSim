//! GLM-5.3-Flash Kimi Delta Attention (KDA) sublayer on one TP rank.
//!
//! Starts after the mHC pre boundary and ends before the TP all-reduce, which
//! the arch owns. Launch order follows the vLLM fork (`glm5next/nvidia/kda.py`):
//! BF16 `in_proj_qkvbfg_a`, the `f_b` / `g_b` low-rank gate projections, the
//! merged q/k/v short conv, split/copy glue, the KDA core, the sigmoid-gated
//! RMSNorm, and the BF16 `o_proj`.
//!
//! The decode core has no glue leaf of its own: `kda_recurrent_decode` times
//! the whole `fused_recurrent_kda` call, whose four q/k/v/beta `.contiguous()`
//! copies precede the recurrent kernel inside that call.
//!
//! Core selection is the fork's, not GDN's: an iteration with any prefill sends
//! ALL of its tokens, decodes included, through one `chunk_kda_with_fused_gate`
//! call (`kda_chunk_prefill`, `D` = decode count) and one varlen conv; a
//! decode-only iteration uses `causal_conv1d_update` and `fused_recurrent_kda`.
//! Only prefill-bearing iterations gather and scatter the fp32 SSM state.
//!
//! Placeholders (elementwise, byte-sized): the split/copy glue, the gated
//! RMSNorm (`gdn_gated_rms_norm` is H200-only and a different entry point),
//! and the state gather/scatter. State movement is expressed in 4-KiB transfer
//! units, as in `qwen36_gdn_local`, so one 1-MiB state is 256 units.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelInput, GdnCausalConvDecodeKernel,
    GdnCausalConvDecodeKernelConfig, GdnCausalConvDecodeKernelInput, GdnCausalConvPrefillKernel,
    GdnCausalConvPrefillKernelConfig, GdnCausalConvPrefillKernelInput, KdaChunkPrefillKernel,
    KdaChunkPrefillKernelConfig, KdaChunkPrefillKernelInput, KdaRecurrentDecodeKernel,
    KdaRecurrentDecodeKernelConfig, KdaRecurrentDecodeKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, PerfApiBridge};

use super::glm53_common::{atomic, elementwise, push_or_zero, repeated};

const STATE_TRANSFER_UNIT_BYTES: u32 = 4096;
/// Prefill-bearing q/k/v `.contiguous()` copies ahead of the qk l2norm.
const PREFILL_COPY_LAUNCHES: u32 = 3;
/// Prefill-bearing small launches: beta sigmoid, index scans, gate views.
const PREFILL_SMALL_GLUE_LAUNCHES: u32 = 17;
const SMALL_GLUE_BYTES_PER_TOKEN: u32 = 64;

#[derive(Clone, Debug)]
pub struct Glm53KdaAttnLocalWorkletConfig {
    pub hidden: Dim,
    /// KDA heads on this rank.
    pub num_heads: Dim,
    pub head_dim: Dim,
    /// Width of the `f_a` / `g_a` low-rank gate factors.
    pub gate_rank: u32,
    pub conv_kernel_size: Dim,
    pub activation_dtype: DType,
    pub gpu_name: String,
    pub bf16_gemm_backends: Vec<&'static str>,
    pub conv_backends: Vec<&'static str>,
    pub core_backends: Vec<&'static str>,
    pub elementwise_backends: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct Glm53KdaAttnLocalWorkletResolved {
    pub raw_cfg: Glm53KdaAttnLocalWorkletConfig,
    pub in_proj: SingleGemmKernelConfig,
    pub f_b: SingleGemmKernelConfig,
    pub g_b: SingleGemmKernelConfig,
    pub conv_prefill: GdnCausalConvPrefillKernelConfig,
    pub conv_decode: GdnCausalConvDecodeKernelConfig,
    pub prefill_copy: crate::timing::kernels::ElementwiseKernelConfig,
    pub prefill_small_glue: crate::timing::kernels::ElementwiseKernelConfig,
    pub state_gather: crate::timing::kernels::ElementwiseKernelConfig,
    pub chunk_prefill: KdaChunkPrefillKernelConfig,
    pub recurrent_decode: KdaRecurrentDecodeKernelConfig,
    pub state_scatter: crate::timing::kernels::ElementwiseKernelConfig,
    pub gated_norm: crate::timing::kernels::ElementwiseKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
    /// fp32 `[H, K, V]` recurrent state of one request on this rank.
    pub ssm_state_bytes_per_request: u32,
    /// bf16 `(kernel_size - 1)`-token conv window of one request on this rank.
    pub conv_state_bytes_per_request: u32,
}

/// One iteration's KDA work on this rank.
#[derive(Clone, Debug, Default)]
pub struct Glm53KdaAttnLocalWorkletInput {
    /// Appended tokens of each prefill request.
    pub prefill_sequence_lengths: Vec<u32>,
    pub decode_batch_size: u32,
}

pub struct Glm53KdaAttnLocalWorklet {
    pub name: String,
    pub in_proj: Op<SingleGemmKernel>,
    pub f_b: Op<SingleGemmKernel>,
    pub g_b: Op<SingleGemmKernel>,
    pub conv_prefill: Op<GdnCausalConvPrefillKernel>,
    pub conv_decode: Op<GdnCausalConvDecodeKernel>,
    pub prefill_copy: Op<ElementwiseKernel>,
    pub prefill_small_glue: Op<ElementwiseKernel>,
    pub state_gather: Op<ElementwiseKernel>,
    pub chunk_prefill: Op<KdaChunkPrefillKernel>,
    pub recurrent_decode: Op<KdaRecurrentDecodeKernel>,
    pub state_scatter: Op<ElementwiseKernel>,
    pub gated_norm: Op<ElementwiseKernel>,
    pub o_proj: Op<SingleGemmKernel>,
    resolved: Glm53KdaAttnLocalWorkletResolved,
}

impl Glm53KdaAttnLocalWorklet {
    pub fn resolve_config(
        cfg: &Glm53KdaAttnLocalWorkletConfig,
    ) -> Glm53KdaAttnLocalWorkletResolved {
        let heads = cfg.num_heads.get();
        let head_dim = cfg.head_dim.get();
        let value_width = heads * head_dim;
        let bpe = cfg.activation_dtype.size_bytes();
        // q, k, v, one beta per head, then the f_a and g_a gate factors.
        let in_proj_width = 3 * value_width + heads + 2 * cfg.gate_rank;
        let conv_channels = 3 * value_width;
        let gemm = |n: Dim, k: Dim| SingleGemmKernelConfig {
            backends: cfg.bf16_gemm_backends.clone(),
            gpu_name: cfg.gpu_name.clone(),
            n,
            k,
            dtype: cfg.activation_dtype,
        };
        let ew = |input: u32, output: u32| {
            elementwise(&cfg.elementwise_backends, &cfg.gpu_name, input, output)
        };
        let per_head_bytes = value_width * bpe;
        let ssm_state_bytes_per_request = heads * head_dim * head_dim * DType::Fp32.size_bytes();
        assert_eq!(
            ssm_state_bytes_per_request % STATE_TRANSFER_UNIT_BYTES,
            0,
            "KDA state must be a whole number of transfer units"
        );
        Glm53KdaAttnLocalWorkletResolved {
            in_proj: gemm(in_proj_width.into(), cfg.hidden.clone()),
            f_b: gemm(value_width.into(), cfg.gate_rank.into()),
            g_b: gemm(value_width.into(), cfg.gate_rank.into()),
            conv_prefill: GdnCausalConvPrefillKernelConfig {
                backends: cfg.conv_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                channels: conv_channels.into(),
                kernel_size: cfg.conv_kernel_size.clone(),
                dtype: cfg.activation_dtype,
                state_dtype: cfg.activation_dtype,
            },
            conv_decode: GdnCausalConvDecodeKernelConfig {
                backends: cfg.conv_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                channels: conv_channels.into(),
                kernel_size: cfg.conv_kernel_size.clone(),
                dtype: cfg.activation_dtype,
                state_dtype: cfg.activation_dtype,
            },
            // One q/k/v-sized head-major copy per launch.
            prefill_copy: ew(per_head_bytes, per_head_bytes),
            prefill_small_glue: ew(SMALL_GLUE_BYTES_PER_TOKEN, SMALL_GLUE_BYTES_PER_TOKEN),
            state_gather: ew(STATE_TRANSFER_UNIT_BYTES, STATE_TRANSFER_UNIT_BYTES),
            chunk_prefill: KdaChunkPrefillKernelConfig {
                backends: cfg.core_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.num_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                dtype: cfg.activation_dtype,
            },
            recurrent_decode: KdaRecurrentDecodeKernelConfig {
                backends: cfg.core_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.num_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                dtype: cfg.activation_dtype,
            },
            state_scatter: ew(STATE_TRANSFER_UNIT_BYTES, STATE_TRANSFER_UNIT_BYTES),
            // rmsnorm(o) * w * sigmoid(g): reads o and g, writes the gated o.
            gated_norm: ew(2 * per_head_bytes, per_head_bytes),
            o_proj: gemm(cfg.hidden.clone(), value_width.into()),
            ssm_state_bytes_per_request,
            conv_state_bytes_per_request: conv_channels * (cfg.conv_kernel_size.get() - 1) * bpe,
            raw_cfg: cfg.clone(),
        }
    }

    pub fn build(
        name: String,
        resolved: Glm53KdaAttnLocalWorkletResolved,
        bridge: &PerfApiBridge,
    ) -> Result<Self, BuildError> {
        let r = resolved.clone();
        let n = name.as_str();
        Ok(Self {
            in_proj: atomic(n, "in_proj", r.in_proj, SingleGemmKernel::build, bridge)?,
            f_b: atomic(n, "f_b_proj", r.f_b, SingleGemmKernel::build, bridge)?,
            g_b: atomic(n, "g_b_proj", r.g_b, SingleGemmKernel::build, bridge)?,
            conv_prefill: atomic(
                n,
                "short_conv_prefill",
                r.conv_prefill,
                GdnCausalConvPrefillKernel::build,
                bridge,
            )?,
            conv_decode: atomic(
                n,
                "short_conv_decode",
                r.conv_decode,
                GdnCausalConvDecodeKernel::build,
                bridge,
            )?,
            prefill_copy: atomic(
                n,
                "prefill_qkv_copy",
                r.prefill_copy,
                ElementwiseKernel::build,
                bridge,
            )?,
            prefill_small_glue: atomic(
                n,
                "prefill_glue",
                r.prefill_small_glue,
                ElementwiseKernel::build,
                bridge,
            )?,
            state_gather: atomic(
                n,
                "state_gather",
                r.state_gather,
                ElementwiseKernel::build,
                bridge,
            )?,
            chunk_prefill: atomic(
                n,
                "chunk_prefill",
                r.chunk_prefill,
                KdaChunkPrefillKernel::build,
                bridge,
            )?,
            recurrent_decode: atomic(
                n,
                "recurrent_decode",
                r.recurrent_decode,
                KdaRecurrentDecodeKernel::build,
                bridge,
            )?,
            state_scatter: atomic(
                n,
                "state_scatter",
                r.state_scatter,
                ElementwiseKernel::build,
                bridge,
            )?,
            gated_norm: atomic(
                n,
                "gated_norm",
                r.gated_norm,
                ElementwiseKernel::build,
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
                "{} (Glm53KdaAttnLocalWorklet) [TP rank; H={}, D={}]",
                self.name, cfg.num_heads, cfg.head_dim
            ),
            child: Box::new(CostNode::Sum(vec![
                self.in_proj.compile(builder),
                self.f_b.compile(builder),
                self.g_b.compile(builder),
                self.conv_prefill.compile(builder),
                self.conv_decode.compile(builder),
                repeated(&self.prefill_copy, PREFILL_COPY_LAUNCHES, builder),
                repeated(
                    &self.prefill_small_glue,
                    PREFILL_SMALL_GLUE_LAUNCHES,
                    builder,
                ),
                self.state_gather.compile(builder),
                self.chunk_prefill.compile(builder),
                self.recurrent_decode.compile(builder),
                self.state_scatter.compile(builder),
                self.gated_norm.compile(builder),
                self.o_proj.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &Glm53KdaAttnLocalWorkletInput, ev: &mut Evaluator) {
        let w = derive_work(input, self.resolved.ssm_state_bytes_per_request)
            .unwrap_or_else(|reason| panic!("invalid Glm53KdaAttnLocalWorkletInput: {reason}"));
        let rows = SingleGemmKernelInput { m: w.total_tokens };
        let tokens = ElementwiseKernelInput {
            num_tokens: w.total_tokens,
        };
        let decode_only = !w.prefill_bearing;
        push_or_zero(&self.in_proj, rows.clone(), false, ev);
        push_or_zero(&self.f_b, rows.clone(), false, ev);
        push_or_zero(&self.g_b, rows.clone(), false, ev);
        push_or_zero(
            &self.conv_prefill,
            GdnCausalConvPrefillKernelInput {
                batch_size: 1,
                sequence_length: w.total_tokens,
            },
            decode_only,
            ev,
        );
        push_or_zero(
            &self.conv_decode,
            GdnCausalConvDecodeKernelInput {
                batch_size: input.decode_batch_size,
            },
            w.prefill_bearing,
            ev,
        );
        push_or_zero(&self.prefill_copy, tokens.clone(), decode_only, ev);
        push_or_zero(&self.prefill_small_glue, tokens.clone(), decode_only, ev);
        let state = ElementwiseKernelInput {
            num_tokens: w.state_units,
        };
        push_or_zero(&self.state_gather, state.clone(), decode_only, ev);
        push_or_zero(
            &self.chunk_prefill,
            KdaChunkPrefillKernelInput {
                num_tokens: w.total_tokens,
                max_sequence_length: w.max_prefill_length,
                num_decode_sequences: input.decode_batch_size,
            },
            decode_only,
            ev,
        );
        push_or_zero(
            &self.recurrent_decode,
            KdaRecurrentDecodeKernelInput {
                batch_size: input.decode_batch_size,
            },
            w.prefill_bearing,
            ev,
        );
        push_or_zero(&self.state_scatter, state, decode_only, ev);
        push_or_zero(&self.gated_norm, tokens, false, ev);
        push_or_zero(&self.o_proj, rows, false, ev);
    }
}

struct Work {
    total_tokens: u32,
    prefill_bearing: bool,
    max_prefill_length: u32,
    /// 4-KiB units of every chunked request's state (prefills and decodes).
    state_units: u32,
}

fn derive_work(input: &Glm53KdaAttnLocalWorkletInput, state_bytes: u32) -> Result<Work, String> {
    let mut prefill_tokens = 0_u32;
    let mut max_prefill_length = 0_u32;
    for (index, &length) in input.prefill_sequence_lengths.iter().enumerate() {
        if length == 0 {
            return Err(format!("prefill sequence {index} must be positive"));
        }
        prefill_tokens = prefill_tokens
            .checked_add(length)
            .ok_or("prefill token sum overflows u32")?;
        max_prefill_length = max_prefill_length.max(length);
    }
    let total_tokens = prefill_tokens
        .checked_add(input.decode_batch_size)
        .ok_or("token sum overflows u32")?;
    if total_tokens == 0 {
        return Err("an iteration must carry at least one token".into());
    }
    let requests = input.prefill_sequence_lengths.len() as u32 + input.decode_batch_size;
    let state_units = requests
        .checked_mul(state_bytes / STATE_TRANSFER_UNIT_BYTES)
        .ok_or("state transfer units overflow u32")?;
    Ok(Work {
        total_tokens,
        prefill_bearing: prefill_tokens > 0,
        max_prefill_length,
        state_units,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn cfg() -> Glm53KdaAttnLocalWorkletConfig {
        Glm53KdaAttnLocalWorkletConfig {
            hidden: 4096.into(),
            num_heads: 16.into(),
            head_dim: 128.into(),
            gate_rank: 128,
            conv_kernel_size: 4.into(),
            activation_dtype: DType::Bf16,
            gpu_name: "NVIDIA B200".into(),
            bf16_gemm_backends: vec!["torch_linear_vllm"],
            conv_backends: vec!["vllm_triton"],
            core_backends: vec!["vllm_triton"],
            elementwise_backends: vec!["triton"],
        }
    }

    #[test]
    fn projection_and_state_shapes_match_the_tp4_checkpoint() {
        let r = Glm53KdaAttnLocalWorklet::resolve_config(&cfg());
        assert_eq!((r.in_proj.n.get(), r.in_proj.k.get()), (6416, 4096));
        assert_eq!((r.f_b.n.get(), r.f_b.k.get()), (2048, 128));
        assert_eq!((r.o_proj.n.get(), r.o_proj.k.get()), (4096, 2048));
        assert_eq!(r.conv_prefill.channels.get(), 6144);
        assert_eq!(r.ssm_state_bytes_per_request, 1 << 20);
        assert_eq!(r.conv_state_bytes_per_request, 6144 * 3 * 2);
    }

    #[test]
    fn prefill_bearing_iteration_chunks_every_token() {
        let w = derive_work(
            &Glm53KdaAttnLocalWorkletInput {
                prefill_sequence_lengths: vec![2019],
                decode_batch_size: 29,
            },
            1 << 20,
        )
        .unwrap();
        assert!(w.prefill_bearing);
        assert_eq!((w.total_tokens, w.max_prefill_length), (2048, 2019));
        assert_eq!(w.state_units, 30 * 256);
        let decode = derive_work(
            &Glm53KdaAttnLocalWorkletInput {
                prefill_sequence_lengths: Vec::new(),
                decode_batch_size: 32,
            },
            1 << 20,
        )
        .unwrap();
        assert!(!decode.prefill_bearing);
        assert!(derive_work(&Glm53KdaAttnLocalWorkletInput::default(), 1 << 20).is_err());
    }

    #[test]
    fn compile_has_fixed_slots_for_both_core_paths() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let worklet = Glm53KdaAttnLocalWorklet::build(
            "m.kda".into(),
            Glm53KdaAttnLocalWorklet::resolve_config(&cfg()),
            &bridge,
        )
        .unwrap();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        assert_eq!(tree.slots.len(), 13);
        assert_eq!(tree.slots[8].name, "m.kda.chunk_prefill");
        assert_eq!(tree.slots[9].name, "m.kda.recurrent_decode");
    }
}
