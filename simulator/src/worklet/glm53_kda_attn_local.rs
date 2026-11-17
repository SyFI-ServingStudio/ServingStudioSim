//! GLM-5.3-Flash Kimi Delta Attention (KDA) sublayer on one TP rank.
//!
//! Starts after the mHC pre boundary and ends before the TP all-reduce, which
//! the arch owns. Launch order follows the vLLM fork (`glm5next/nvidia/kda.py`):
//! BF16 `in_proj_qkvbfg_a`, the `f_b` / `g_b` low-rank gate projections, the
//! merged q/k/v short conv, split/copy glue, the KDA core, the sigmoid-gated
//! RMSNorm, and the BF16 `o_proj`.
//!
//! The decode core has no glue leaf of its own: `kda_recurrent_decode` times
//! the whole `fused_recurrent_kda` call.
//!
//! Core selection is the fork's, not GDN's: an iteration with any prefill sends
//! ALL of its tokens, decodes included, through one `chunk_kda_with_fused_gate`
//! call (`kda_chunk_prefill`, `D` = decode count) and one varlen conv; a
//! decode-only iteration uses `causal_conv1d_update` and `fused_recurrent_kda`.
//! A per-sequence short-conv prefill backend (`conv_prefill_per_sequence`)
//! instead launches once per prefill sequence and runs the iteration's
//! decode tokens through `causal_conv1d_update`, so its short conv is the sum of
//! the `(1, L_i)` rows plus one decode launch.
//! Only prefill-bearing iterations gather and scatter the fp32 SSM state.
//!
//! The prefill core's q/k/v `.contiguous()` copies and the FLA chain's
//! scratch fills run inside the call `kda_chunk_prefill` times, so they have no
//! leaf here. FLA's chunk-index setup (about 11 small launches, ~17 us) runs
//! once per step and is cached across layers; it is left out.
//!
//! Placeholders (elementwise, byte-sized): the beta sigmoid, the gated
//! RMSNorm (`gdn_gated_rms_norm` is H200-only and a different entry point),
//! and the state gather/scatter. State movement is expressed in 4-KiB transfer
//! units, as in `qwen36_gdn_local`, so one 1-MiB state is 256 units.
//!
//! Speculative decoding (a build with verify width q = K + 1 > 1,
//! [`Glm53KdaAttnLocalWorklet::resolve_config_with_verify_width`]): every decode
//! request is a spec decode carrying q tokens (`decode_batch_size` counts
//! requests, not tokens). Its short conv is `causal_conv1d_update` in the varlen
//! verify form and its core `fused_recurrent_kda` with `[B, q]` state slots and
//! `num_accepted_tokens` (`kda.py` `use_spec`); both are priced at
//! `tokens_per_request = q`. Spec decodes never join the chunk call. A pure
//! verify step has no glue (the fork skips the identity `index_select`s and
//! writes the layer output in place). A mixed step runs the prefills alone
//! through the varlen conv, the beta sigmoid, the state gather, the chunk core
//! and the state scatter, and adds the split/merge glue: six `index_select`s of
//! qkv/g1/beta (spec and non-spec groups, every token) and two `index_copy_`s of
//! the core outputs back into token order, priced as 4-KiB copies spread evenly
//! over those launches. vLLM reclassifies a one-token non-spec decode in a step
//! with spec decodes as a length-1 prefill (`gdn_attn.py`), so the caller passes
//! it in `prefill_sequence_lengths`.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::gdn_causal_conv_prefill::launches_per_sequence;
use crate::timing::kernels::kda_chunk_prefill::takes_beta_logits;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelInput, GdnCausalConvDecodeKernel,
    GdnCausalConvDecodeKernelConfig, GdnCausalConvDecodeKernelInput, GdnCausalConvPrefillKernel,
    GdnCausalConvPrefillKernelConfig, GdnCausalConvPrefillKernelInput, KdaChunkPrefillKernel,
    KdaChunkPrefillKernelConfig, KdaChunkPrefillKernelInput, KdaRecurrentDecodeKernel,
    KdaRecurrentDecodeKernelConfig, KdaRecurrentDecodeKernelInput, SingleGemmKernel,
    SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, GdnCausalConvPrefillLog, LeafMetrics,
    PerfApiBridge,
};

use super::glm53_common::{atomic, backends_agree, elementwise, push_or_zero, repeated};

const STATE_TRANSFER_UNIT_BYTES: u32 = 4096;
/// Mixed spec step: `index_select` of qkv, g1 and beta for the spec and the
/// non-spec token groups (`kda.py` `_forward`, `use_spec` with non-spec tokens).
const SPEC_SPLIT_LAUNCHES: u32 = 6;
/// Mixed spec step: `core_attn_out.index_copy_` of the spec and non-spec outputs.
const SPEC_MERGE_LAUNCHES: u32 = 2;
/// Beta sigmoid ahead of a prefill core that takes beta already passed
/// through one: reads the bf16 logits, writes fp32 (capture 20260924_0).
const BETA_SIGMOID_IN_BYTES_PER_HEAD: u32 = 2;
const BETA_SIGMOID_OUT_BYTES_PER_HEAD: u32 = 4;

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
    /// Short-conv prefill backends (`gdn_causal_conv_prefill`).
    pub conv_prefill_backends: Vec<&'static str>,
    /// Short-conv decode backends (`gdn_causal_conv_decode`).
    pub conv_decode_backends: Vec<&'static str>,
    /// Recurrent-decode backends.
    pub core_backends: Vec<&'static str>,
    /// Chunked-prefill backends: the serving engine's KDA prefill kernel.
    pub chunk_prefill_backends: Vec<&'static str>,
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
    pub prefill_small_glue: crate::timing::kernels::ElementwiseKernelConfig,
    pub state_gather: crate::timing::kernels::ElementwiseKernelConfig,
    pub chunk_prefill: KdaChunkPrefillKernelConfig,
    pub recurrent_decode: KdaRecurrentDecodeKernelConfig,
    pub state_scatter: crate::timing::kernels::ElementwiseKernelConfig,
    pub gated_norm: crate::timing::kernels::ElementwiseKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
    /// Verify width q (tokens per decode request): 1 without speculative
    /// decoding, K + 1 with K drafts.
    pub verify_width: u32,
    /// Mixed spec step only (`Some` when `verify_width > 1`): the qkv/g1/beta
    /// `index_select`s, in 4-KiB copy units.
    pub spec_split: Option<crate::timing::kernels::ElementwiseKernelConfig>,
    /// Mixed spec step only: the two output `index_copy_`s, in 4-KiB units.
    pub spec_merge: Option<crate::timing::kernels::ElementwiseKernelConfig>,
    /// fp32 `[H, K, V]` recurrent state of one state slot (one mamba page) on
    /// this rank. Without spec decoding a request holds one slot.
    pub ssm_state_bytes_per_request: u32,
    /// bf16 conv window of one state slot on this rank:
    /// `kernel_size - 1 + (verify_width - 1)` tokens (`kda_state_shape`,
    /// `num_spec`), so `kernel_size - 1` without spec decoding.
    pub conv_state_bytes_per_request: u32,
    /// State slots (mamba pages, each one SSM state plus one conv window) vLLM
    /// gives a request per KDA layer: [`kda_state_slots_per_request`].
    pub state_slots_per_request: u32,
}

/// State slots (mamba pages) per request per KDA layer for verify width `q`:
/// `1 + num_speculative_blocks` with `num_speculative_blocks = q - 1`
/// (`MambaSpec.max_memory_usage_bytes`, `mamba_cache_mode = "none"`; the
/// `[B, q]` `spec_state_indices_tensor` the verify kernels write). Each page
/// holds the fp32 SSM state and a `kernel_size - 1 + q - 1` conv window. With
/// prefix caching vLLM switches to `mamba_cache_mode = "align"`, whose budget is
/// one page more (`2 + num_speculative_blocks`): add
/// [`KDA_ALIGN_MODE_EXTRA_SLOTS`] for that mode.
pub fn kda_state_slots_per_request(verify_width: u32) -> u32 {
    assert!(verify_width >= 1, "verify width must be >= 1");
    verify_width
}

/// Extra state page per request under `mamba_cache_mode = "align"` (prefix
/// caching): `MambaSpec.max_memory_usage_bytes` reserves `2 + num_spec` pages.
pub const KDA_ALIGN_MODE_EXTRA_SLOTS: u32 = 1;

/// One iteration's KDA work on this rank.
#[derive(Clone, Debug, Default)]
pub struct Glm53KdaAttnLocalWorkletInput {
    /// Appended tokens of each prefill request. In a spec build, also each
    /// one-token non-spec decode (vLLM chunks it as a length-1 prefill).
    pub prefill_sequence_lengths: Vec<u32>,
    /// Decode requests. Each carries one token, or the build's verify width q
    /// tokens in a spec build (`verify_width > 1`).
    pub decode_batch_size: u32,
}

pub struct Glm53KdaAttnLocalWorklet {
    pub name: String,
    pub in_proj: Op<SingleGemmKernel>,
    pub f_b: Op<SingleGemmKernel>,
    pub g_b: Op<SingleGemmKernel>,
    pub conv_prefill: Op<GdnCausalConvPrefillKernel>,
    pub conv_decode: Op<GdnCausalConvDecodeKernel>,
    pub prefill_small_glue: Op<ElementwiseKernel>,
    pub state_gather: Op<ElementwiseKernel>,
    pub chunk_prefill: Op<KdaChunkPrefillKernel>,
    pub recurrent_decode: Op<KdaRecurrentDecodeKernel>,
    pub state_scatter: Op<ElementwiseKernel>,
    pub gated_norm: Op<ElementwiseKernel>,
    pub o_proj: Op<SingleGemmKernel>,
    /// Spec builds only: the mixed step's split / merge glue.
    pub spec_split: Option<Op<ElementwiseKernel>>,
    pub spec_merge: Option<Op<ElementwiseKernel>>,
    /// The short-conv prefill backends launch once per sequence
    /// (`gdn_causal_conv_prefill::launches_per_sequence`) rather than once over
    /// every token of the iteration.
    conv_per_sequence: bool,
    /// The beta sigmoid's launches: none when the chunked-prefill backends
    /// apply it inside their call (`kda_chunk_prefill::takes_beta_logits`).
    beta_sigmoid_launches: u32,
    resolved: Glm53KdaAttnLocalWorkletResolved,
}

impl Glm53KdaAttnLocalWorklet {
    /// The no-spec build (one token per decode request).
    pub fn resolve_config(
        cfg: &Glm53KdaAttnLocalWorkletConfig,
    ) -> Glm53KdaAttnLocalWorkletResolved {
        Self::resolve_config_with_verify_width(cfg, 1)
    }

    /// The build for verify width `verify_width` = q = K + 1 (1 = no spec
    /// decoding, identical to [`Self::resolve_config`]). q is static identity:
    /// it keys the verify rows of the short-conv and recurrent decode kernels.
    pub fn resolve_config_with_verify_width(
        cfg: &Glm53KdaAttnLocalWorkletConfig,
        verify_width: u32,
    ) -> Glm53KdaAttnLocalWorkletResolved {
        assert!(verify_width >= 1, "verify width must be >= 1");
        let spec = verify_width > 1;
        let tokens_per_request = spec.then(|| Dim::from(verify_width));
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
                backends: cfg.conv_prefill_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                channels: conv_channels.into(),
                kernel_size: cfg.conv_kernel_size.clone(),
                dtype: cfg.activation_dtype,
                state_dtype: cfg.activation_dtype,
            },
            conv_decode: GdnCausalConvDecodeKernelConfig {
                backends: cfg.conv_decode_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                channels: conv_channels.into(),
                kernel_size: cfg.conv_kernel_size.clone(),
                dtype: cfg.activation_dtype,
                state_dtype: cfg.activation_dtype,
                tokens_per_request: tokens_per_request.clone(),
            },
            prefill_small_glue: ew(
                heads * BETA_SIGMOID_IN_BYTES_PER_HEAD,
                heads * BETA_SIGMOID_OUT_BYTES_PER_HEAD,
            ),
            state_gather: ew(STATE_TRANSFER_UNIT_BYTES, STATE_TRANSFER_UNIT_BYTES),
            chunk_prefill: KdaChunkPrefillKernelConfig {
                backends: cfg.chunk_prefill_backends.clone(),
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
                tokens_per_request,
            },
            state_scatter: ew(STATE_TRANSFER_UNIT_BYTES, STATE_TRANSFER_UNIT_BYTES),
            // rmsnorm(o) * w * sigmoid(g): reads o and g, writes the gated o.
            gated_norm: ew(2 * per_head_bytes, per_head_bytes),
            o_proj: gemm(cfg.hidden.clone(), value_width.into()),
            verify_width,
            spec_split: spec.then(|| ew(STATE_TRANSFER_UNIT_BYTES, STATE_TRANSFER_UNIT_BYTES)),
            spec_merge: spec.then(|| ew(STATE_TRANSFER_UNIT_BYTES, STATE_TRANSFER_UNIT_BYTES)),
            ssm_state_bytes_per_request,
            conv_state_bytes_per_request: conv_channels
                * (cfg.conv_kernel_size.get() - 1 + (verify_width - 1))
                * bpe,
            state_slots_per_request: kda_state_slots_per_request(verify_width),
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
        let mut worklet = Self {
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
            spec_split: r
                .spec_split
                .map(|c| atomic(n, "spec_split", c, ElementwiseKernel::build, bridge))
                .transpose()?,
            spec_merge: r
                .spec_merge
                .map(|c| atomic(n, "spec_merge", c, ElementwiseKernel::build, bridge))
                .transpose()?,
            conv_per_sequence: false,
            beta_sigmoid_launches: 0,
            name,
            resolved,
        };
        // The backends as built, after any run-config override, set the leaves
        // around the two cores.
        worklet.conv_per_sequence = backends_agree(
            &worklet.conv_prefill,
            "they launch once per sequence",
            launches_per_sequence,
        )?;
        worklet.beta_sigmoid_launches = u32::from(!backends_agree(
            &worklet.chunk_prefill,
            "they take beta logits",
            takes_beta_logits,
        )?);
        Ok(worklet)
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        let cfg = &self.resolved.raw_cfg;
        let verify = match self.resolved.verify_width {
            1 => String::new(),
            q => format!(", verify q={q}"),
        };
        let mut children = vec![
            self.in_proj.compile(builder),
            self.f_b.compile(builder),
            self.g_b.compile(builder),
        ];
        if let Some(split) = &self.spec_split {
            children.push(repeated(split, SPEC_SPLIT_LAUNCHES, builder));
        }
        children.extend([
                self.conv_prefill.compile(builder),
                self.conv_decode.compile(builder),
                repeated(
                    &self.prefill_small_glue,
                    self.beta_sigmoid_launches,
                    builder,
                ),
                self.state_gather.compile(builder),
                self.chunk_prefill.compile(builder),
                self.recurrent_decode.compile(builder),
                self.state_scatter.compile(builder),
        ]);
        if let Some(merge) = &self.spec_merge {
            children.push(repeated(merge, SPEC_MERGE_LAUNCHES, builder));
        }
        children.extend([self.gated_norm.compile(builder), self.o_proj.compile(builder)]);
        CostNode::Labeled {
            label: format!(
                "{} (Glm53KdaAttnLocalWorklet) [TP rank; H={}, D={}{verify}]",
                self.name, cfg.num_heads, cfg.head_dim
            ),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &Glm53KdaAttnLocalWorkletInput, ev: &mut Evaluator) {
        let cfg = &self.resolved.raw_cfg;
        let bpe = cfg.activation_dtype.size_bytes();
        let (heads, head_dim) = (cfg.num_heads.get(), cfg.head_dim.get());
        let w = derive_work_for(
            input,
            self.resolved.ssm_state_bytes_per_request,
            self.resolved.verify_width,
            spec_split_bytes_per_token(heads, head_dim, bpe),
            // The core output, H·D per token, copied back into token order.
            u64::from(heads * head_dim * bpe),
        )
        .unwrap_or_else(|reason| panic!("invalid Glm53KdaAttnLocalWorkletInput: {reason}"));
        let rows = SingleGemmKernelInput { m: w.total_tokens };
        let tokens = ElementwiseKernelInput {
            num_tokens: w.total_tokens,
        };
        let chunk_tokens = ElementwiseKernelInput {
            num_tokens: w.chunk_tokens,
        };
        let decode_only = !w.prefill_bearing;
        let per_sequence = self.conv_per_sequence;
        push_or_zero(&self.in_proj, rows.clone(), false, ev);
        push_or_zero(&self.f_b, rows.clone(), false, ev);
        push_or_zero(&self.g_b, rows.clone(), false, ev);
        if let Some(split) = &self.spec_split {
            push_or_zero(
                split,
                ElementwiseKernelInput {
                    num_tokens: w.split_units_per_launch,
                },
                !w.mixed_spec,
                ev,
            );
        }
        if per_sequence {
            push_conv_per_sequence(&self.conv_prefill, &input.prefill_sequence_lengths, ev);
        } else {
            push_or_zero(
                &self.conv_prefill,
                GdnCausalConvPrefillKernelInput {
                    batch_size: 1,
                    sequence_length: w.chunk_tokens,
                },
                decode_only,
                ev,
            );
        }
        push_or_zero(
            &self.conv_decode,
            GdnCausalConvDecodeKernelInput {
                batch_size: input.decode_batch_size,
            },
            !decode_conv_runs(
                per_sequence || w.spec,
                w.prefill_bearing,
                input.decode_batch_size,
            ),
            ev,
        );
        push_or_zero(&self.prefill_small_glue, chunk_tokens.clone(), decode_only, ev);
        let state = ElementwiseKernelInput {
            num_tokens: w.state_units,
        };
        push_or_zero(&self.state_gather, state.clone(), decode_only, ev);
        push_or_zero(
            &self.chunk_prefill,
            KdaChunkPrefillKernelInput {
                num_tokens: w.chunk_tokens,
                max_sequence_length: w.max_prefill_length,
                num_decode_sequences: w.chunk_decode_sequences,
            },
            decode_only,
            ev,
        );
        push_or_zero(
            &self.recurrent_decode,
            KdaRecurrentDecodeKernelInput {
                batch_size: input.decode_batch_size,
            },
            w.prefill_bearing && !w.spec,
            ev,
        );
        push_or_zero(&self.state_scatter, state, decode_only, ev);
        if let Some(merge) = &self.spec_merge {
            push_or_zero(
                merge,
                ElementwiseKernelInput {
                    num_tokens: w.merge_units_per_launch,
                },
                !w.mixed_spec,
                ev,
            );
        }
        push_or_zero(&self.gated_norm, tokens, false, ev);
        push_or_zero(&self.o_proj, rows, false, ev);
    }
}

/// Whether the decode short conv launches: in a decode-only iteration, and in
/// a mixed one only when the prefill backend launches per sequence, since
/// vLLM's varlen prefill launch covers the decode tokens too.
fn decode_conv_runs(per_sequence: bool, prefill_bearing: bool, decode_batch_size: u32) -> bool {
    decode_batch_size > 0 && (per_sequence || !prefill_bearing)
}

/// One `(1, L_i)` launch per prefill sequence, summed into the one slot. No
/// prefill sequence is zero work and skips the cache lookup.
fn push_conv_per_sequence(
    op: &Op<GdnCausalConvPrefillKernel>,
    sequence_lengths: &[u32],
    ev: &mut Evaluator,
) {
    let metrics = sum_per_sequence(sequence_lengths, |input| op.kernel.eval(input));
    ev.push(metrics, || {
        GdnCausalConvPrefillLog {
            sequence_lengths: sequence_lengths.to_vec(),
            decode_rows: 0,
        }
        .into()
    });
}

fn sum_per_sequence(
    sequence_lengths: &[u32],
    mut eval: impl FnMut(&GdnCausalConvPrefillKernelInput) -> LeafMetrics,
) -> LeafMetrics {
    let mut metrics = LeafMetrics::ZERO;
    for &sequence_length in sequence_lengths {
        metrics.add_fanin(eval(&GdnCausalConvPrefillKernelInput {
            batch_size: 1,
            sequence_length,
        }));
    }
    metrics
}

struct Work {
    total_tokens: u32,
    prefill_bearing: bool,
    max_prefill_length: u32,
    /// Spec decodes are present (a spec build with decode requests): they take
    /// the verify path and stay out of the chunk call.
    spec: bool,
    /// A spec step that also carries prefills: the split/merge glue runs.
    mixed_spec: bool,
    /// Tokens of the varlen conv / chunk call: every token, or only the
    /// prefill tokens when the decodes are spec decodes.
    chunk_tokens: u32,
    /// Decode sequences folded into the chunk call.
    chunk_decode_sequences: u32,
    /// 4-KiB units of every chunked request's state.
    state_units: u32,
    /// 4-KiB units per launch of the mixed spec step's split and merge glue.
    split_units_per_launch: u32,
    merge_units_per_launch: u32,
}

/// Bytes per token the split `index_select`s copy: qkv (3·H·D), g1 (H·D) and
/// beta (H) of the token's group.
fn spec_split_bytes_per_token(heads: u32, head_dim: u32, bpe: u32) -> u64 {
    u64::from((4 * heads * head_dim + heads) * bpe)
}

/// The no-spec work of `input` (tests).
#[cfg(test)]
fn derive_work(input: &Glm53KdaAttnLocalWorkletInput, state_bytes: u32) -> Result<Work, String> {
    derive_work_for(input, state_bytes, 1, 0, 0)
}

fn derive_work_for(
    input: &Glm53KdaAttnLocalWorkletInput,
    state_bytes: u32,
    verify_width: u32,
    split_bytes_per_token: u64,
    merge_bytes_per_token: u64,
) -> Result<Work, String> {
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
    let decode_tokens = input
        .decode_batch_size
        .checked_mul(verify_width)
        .ok_or("decode token count overflows u32")?;
    let total_tokens = prefill_tokens
        .checked_add(decode_tokens)
        .ok_or("token sum overflows u32")?;
    if total_tokens == 0 {
        return Err("an iteration must carry at least one token".into());
    }
    let prefill_bearing = prefill_tokens > 0;
    let spec = verify_width > 1 && input.decode_batch_size > 0;
    let mixed_spec = spec && prefill_bearing;
    let (chunk_tokens, chunk_decode_sequences) = if spec {
        (prefill_tokens, 0)
    } else {
        (total_tokens, input.decode_batch_size)
    };
    let requests = input.prefill_sequence_lengths.len() as u32 + chunk_decode_sequences;
    let state_units = requests
        .checked_mul(state_bytes / STATE_TRANSFER_UNIT_BYTES)
        .ok_or("state transfer units overflow u32")?;
    let units_per_launch = |bytes_per_token: u64, launches: u32| -> Result<u32, String> {
        if !mixed_spec {
            return Ok(0);
        }
        let per_launch = (u64::from(total_tokens) * bytes_per_token)
            .div_ceil(u64::from(launches) * u64::from(STATE_TRANSFER_UNIT_BYTES));
        u32::try_from(per_launch).map_err(|_| "spec glue units overflow u32".to_string())
    };
    Ok(Work {
        total_tokens,
        prefill_bearing,
        max_prefill_length,
        spec,
        mixed_spec,
        chunk_tokens,
        chunk_decode_sequences,
        state_units,
        split_units_per_launch: units_per_launch(split_bytes_per_token, SPEC_SPLIT_LAUNCHES)?,
        merge_units_per_launch: units_per_launch(merge_bytes_per_token, SPEC_MERGE_LAUNCHES)?,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

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
            conv_prefill_backends: vec!["vllm_triton"],
            conv_decode_backends: vec!["vllm_triton"],
            core_backends: vec!["vllm_triton"],
            chunk_prefill_backends: vec!["vllm_triton"],
            elementwise_backends: vec!["triton"],
        }
    }

    #[test]
    fn the_built_backends_set_the_sigmoid_and_conv_launches() {
        // An override replaces the arch's backends inside Kernel::build; the
        // leaves around the cores must follow the backends as built.
        let build = |overrides: &[(&str, &str)]| {
            let bridge = PerfApiBridge::new_uninit_for_test();
            bridge.enable_enumerate();
            let map: HashMap<String, Vec<String>> = overrides
                .iter()
                .map(|(role, backend)| (role.to_string(), vec![backend.to_string()]))
                .collect();
            let _scope = bridge.with_backend_overrides("main", Some(&map));
            Glm53KdaAttnLocalWorklet::build(
                "m.kda".into(),
                Glm53KdaAttnLocalWorklet::resolve_config(&cfg()),
                &bridge,
            )
        };
        let fla = build(&[]).unwrap();
        assert_eq!(
            (fla.beta_sigmoid_launches, fla.conv_per_sequence),
            (1, false)
        );
        let fast = build(&[
            ("m.kda.chunk_prefill", "flashinfer_cute_persistent"),
            ("m.kda.short_conv_prefill", "dao_channellast"),
        ])
        .unwrap();
        assert_eq!(
            (fast.beta_sigmoid_launches, fast.conv_per_sequence),
            (0, true)
        );
    }

    #[test]
    fn backends_with_different_beta_inputs_are_rejected() {
        let mut mixed = cfg();
        mixed.chunk_prefill_backends = vec!["vllm_triton", "flashkda"];
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let built = Glm53KdaAttnLocalWorklet::build(
            "m.kda".into(),
            Glm53KdaAttnLocalWorklet::resolve_config(&mixed),
            &bridge,
        );
        assert!(matches!(
            built,
            Err(BuildError::MixedBackendProperty { .. })
        ));
    }

    #[test]
    fn per_sequence_conv_sums_one_row_per_prefill_sequence() {
        // A per-sequence backend priced as one (1, T) row would hide a launch
        // per request; each length must reach the cache as its own B=1 row.
        let mut seen = Vec::new();
        let metrics = sum_per_sequence(&[3, 65, 2], |input| {
            seen.push((input.batch_size, input.sequence_length));
            let mut leaf = LeafMetrics::ZERO;
            leaf.m.time_ms = input.sequence_length as f32;
            leaf
        });
        assert_eq!(seen, [(1, 3), (1, 65), (1, 2)]);
        assert_eq!(metrics.m.time_ms, 70.0);
        assert_eq!(sum_per_sequence(&[], |_| unreachable!()).m.time_ms, 0.0);
    }

    #[test]
    fn decode_tokens_leave_the_varlen_launch_only_for_a_per_sequence_backend() {
        // vllm_triton: decodes ride in the one varlen prefill launch.
        assert!(!decode_conv_runs(false, true, 29));
        assert!(decode_conv_runs(false, false, 32));
        // dao_channellast: a mixed iteration still runs causal_conv1d_update.
        assert!(decode_conv_runs(true, true, 29));
        assert!(decode_conv_runs(true, false, 32));
        assert!(!decode_conv_runs(true, true, 0));
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
        assert_eq!(tree.slots.len(), 12);
        assert_eq!(tree.slots[7].name, "m.kda.chunk_prefill");
        assert_eq!(tree.slots[8].name, "m.kda.recurrent_decode");
    }

    fn spec_worklet(q: u32) -> (Glm53KdaAttnLocalWorklet, crate::timing::CostTree) {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let worklet = Glm53KdaAttnLocalWorklet::build(
            "m.kda".into(),
            Glm53KdaAttnLocalWorklet::resolve_config_with_verify_width(&cfg(), q),
            &bridge,
        )
        .unwrap();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        (worklet, tree)
    }

    #[test]
    fn verify_width_one_is_the_no_spec_build() {
        // Defect: a spec-capable build changing the no-spec (q = 1) rows' keys,
        // its slots, or its state footprint.
        let r = Glm53KdaAttnLocalWorklet::resolve_config_with_verify_width(&cfg(), 1);
        assert_eq!(r.conv_decode.tokens_per_request, None);
        assert_eq!(r.recurrent_decode.tokens_per_request, None);
        assert!(r.spec_split.is_none() && r.spec_merge.is_none());
        assert_eq!(r.state_slots_per_request, 1);
        assert_eq!(r.conv_state_bytes_per_request, 6144 * 3 * 2);
        let (_, tree) = spec_worklet(1);
        assert_eq!(tree.slots.len(), 12);
    }

    #[test]
    fn spec_build_keys_the_verify_rows_and_sizes_q_state_slots() {
        // Defect: pricing a verify step with the one-token rows, or a state
        // footprint without vLLM's q pages / num_spec-wide conv window.
        let r = Glm53KdaAttnLocalWorklet::resolve_config_with_verify_width(&cfg(), 4);
        assert_eq!(r.conv_decode.tokens_per_request, Some(4.into()));
        assert_eq!(r.recurrent_decode.tokens_per_request, Some(4.into()));
        assert_eq!(r.state_slots_per_request, 4);
        assert_eq!(r.ssm_state_bytes_per_request, 1 << 20);
        assert_eq!(r.conv_state_bytes_per_request, 6144 * (3 + 3) * 2);
        assert_eq!(kda_state_slots_per_request(8), 8);
        let (_, tree) = spec_worklet(4);
        let names: Vec<&str> = tree.slots.iter().map(|slot| slot.name.as_str()).collect();
        assert_eq!(names.len(), 14);
        assert_eq!(names[3], "m.kda.spec_split");
        assert_eq!(names[11], "m.kda.spec_merge");
        assert_eq!(names[9], "m.kda.recurrent_decode");
    }

    #[test]
    fn pure_verify_step_runs_no_chunk_path_and_no_glue() {
        // Defect: chunking spec decodes, or charging the identity index_selects
        // the fork skips on a pure verify step.
        let w = derive_work_for(
            &Glm53KdaAttnLocalWorkletInput {
                prefill_sequence_lengths: Vec::new(),
                decode_batch_size: 32,
            },
            1 << 20,
            4,
            16_416,
            4096,
        )
        .unwrap();
        assert!(w.spec && !w.mixed_spec && !w.prefill_bearing);
        assert_eq!(w.total_tokens, 128);
        assert_eq!((w.chunk_tokens, w.chunk_decode_sequences, w.state_units), (0, 0, 0));
        assert_eq!((w.split_units_per_launch, w.merge_units_per_launch), (0, 0));
    }

    #[test]
    fn mixed_spec_step_chunks_prefills_only_and_sizes_the_glue_over_every_token() {
        // Defect: folding spec decodes into the chunk call (the no-spec mixed
        // path), gathering their states, or sizing the split/merge from one
        // group only.
        let w = derive_work_for(
            &Glm53KdaAttnLocalWorkletInput {
                prefill_sequence_lengths: vec![1000, 1],
                decode_batch_size: 12,
            },
            1 << 20,
            4,
            spec_split_bytes_per_token(16, 128, 2),
            4096,
        )
        .unwrap();
        assert!(w.spec && w.mixed_spec);
        assert_eq!(w.total_tokens, 1049);
        assert_eq!((w.chunk_tokens, w.chunk_decode_sequences), (1001, 0));
        assert_eq!(w.state_units, 2 * 256);
        assert_eq!(spec_split_bytes_per_token(16, 128, 2), 16_416);
        assert_eq!(w.split_units_per_launch, (1049 * 16_416_u32).div_ceil(6 * 4096));
        assert_eq!(w.merge_units_per_launch, (1049 * 4096_u32).div_ceil(2 * 4096));
        // Without decodes a spec build is the no-spec prefill step.
        let prefill_only = derive_work_for(
            &Glm53KdaAttnLocalWorkletInput {
                prefill_sequence_lengths: vec![64],
                decode_batch_size: 0,
            },
            1 << 20,
            4,
            16_416,
            4096,
        )
        .unwrap();
        assert!(!prefill_only.spec && !prefill_only.mixed_spec);
        assert_eq!(prefill_only.chunk_tokens, 64);
    }
}
