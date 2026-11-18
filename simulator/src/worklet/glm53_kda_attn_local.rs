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
//! A per-sequence short-conv prefill backend (`conv_prefill_per_sequence`)
//! instead launches once per prefill sequence and runs the iteration's
//! decode tokens through `causal_conv1d_update`, so its short conv is the sum of
//! the `(1, L_i)` rows plus one decode launch.
//! Only prefill-bearing iterations gather and scatter the SSM state.
//!
//! The state pool is fp32 (vLLM's default) or bf16 (`--mamba-ssm-cache-dtype
//! bfloat16`). The pool dtype sizes the per-request state and its gather and
//! scatter, and keys the recurrent-decode and verify rows. The chunked-prefill
//! rows carry no state-dtype axis and stay fp32-pool measurements.
//!
//! A speculative verify graph (`decode_query_len` = `1 + k` rows per decode
//! request) keeps its decode requests off the chunk path: vLLM chunks only the
//! prefill requests and runs the verify core over the spec decodes. That core
//! is one of two ([`KdaSpecVerify`]). RecoverSSM reads each request's state
//! once and caches a correction and key/gate per row; a commit, which the arch
//! bills once per KDA cache group, folds the accepted rows into the one state
//! the request keeps. Its kernels are the ones the fork selects (`kda.py`
//! `can_use_cute_recoverssm`): where its CuTe kernels run, the short conv and
//! the verify in one launch and the CuTe commit fold; otherwise
//! `causal_conv1d_update` in its varlen form, the Triton verify and the Triton
//! commit. The per-position core is `fused_recurrent_kda` over `[B, 1 + k]`
//! state slots, which writes a full state at every position and so holds
//! `1 + k` state pages per request and needs no commit.
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

use crate::common::gpu::compute_capability;
use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::gdn_causal_conv_prefill::launches_per_sequence;
use crate::timing::kernels::kda_chunk_prefill::takes_beta_logits;
use crate::timing::kernels::{
    ElementwiseKernel, ElementwiseKernelInput, GdnCausalConvDecodeKernel,
    GdnCausalConvDecodeKernelConfig, GdnCausalConvDecodeKernelInput, GdnCausalConvPrefillKernel,
    GdnCausalConvPrefillKernelConfig, GdnCausalConvPrefillKernelInput, KdaChunkPrefillKernel,
    KdaChunkPrefillKernelConfig, KdaChunkPrefillKernelInput, KdaRecoverSSMCommitKernelConfig,
    KdaRecoverSSMConvVerifyKernel, KdaRecoverSSMConvVerifyKernelConfig,
    KdaRecoverSSMConvVerifyKernelInput, KdaRecoverSSMVerifyKernel, KdaRecoverSSMVerifyKernelConfig,
    KdaRecoverSSMVerifyKernelInput, KdaRecurrentDecodeKernel, KdaRecurrentDecodeKernelConfig,
    KdaRecurrentDecodeKernelInput, SingleGemmKernel, SingleGemmKernelConfig, SingleGemmKernelInput,
};
use crate::timing::{
    BuildError, CostNode, CostTreeBuilder, Dim, Evaluator, GdnCausalConvPrefillLog, LeafMetrics,
    PerfApiBridge,
};

use super::glm53_common::{atomic, backends_agree, elementwise, push_or_zero, repeated};

const STATE_TRANSFER_UNIT_BYTES: u32 = 4096;
/// The fork's RecoverSSM kernels where its CuTe ones cannot run: the Triton
/// verify after `causal_conv1d_update`, and the Triton commit.
const RECOVERSSM_TRITON_BACKENDS: &[&str] = &["vllm_triton"];
/// The fork's CuTe RecoverSSM kernels (`kda.py` `recoverssm_backend = "cute"`):
/// the conv and the verify in one launch on its `[state_len, dim]` conv pages,
/// and the commit fold.
const RECOVERSSM_CUTE_BACKENDS: &[&str] = &["vllm_cute"];
/// The fork creates the KDA conv taps in fp32 (`glm5next/common/kda.py`
/// `conv1d`, `params_dtype=torch.float32`).
const CONV_WEIGHT_DTYPE: DType = DType::Fp32;
/// `can_use_cute_recoverssm`: head dim 128, a width-4 conv and a 1..8-token
/// window (`recoverssm/kernels.py` `K`, `MAX_WINDOW`).
const CUTE_RECOVERSSM_HEAD_DIM: u32 = 128;
const CUTE_RECOVERSSM_CONV_WIDTH: u32 = 4;
const CUTE_RECOVERSSM_MAX_WINDOW: u32 = 8;
/// Beta sigmoid ahead of a prefill core that takes beta already passed
/// through one: reads the bf16 logits, writes fp32 (capture 20260924_0).
const BETA_SIGMOID_IN_BYTES_PER_HEAD: u32 = 2;
const BETA_SIGMOID_OUT_BYTES_PER_HEAD: u32 = 4;

/// How a speculative verify advances the KDA state.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum KdaSpecVerify {
    /// RecoverSSM: a verify that reads each request's state once, and a
    /// commit that folds the accepted rows into it.
    #[default]
    Recoverssm,
    /// The vLLM fork's GLM path (`kda.py` spec branch): `fused_recurrent_kda`
    /// reads the last accepted state and stores the state after every
    /// position into that position's slot.
    RecurrentPerPosition,
}

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
    /// Element type of the recurrent state pool (fp32 or bf16).
    pub ssm_state_dtype: DType,
    /// Query rows per decode request: 1 for plain decode, `1 + k` for a
    /// speculative verify, whose decode core `spec_verify` selects.
    pub decode_query_len: u32,
    /// The verify core of a graph with `decode_query_len > 1`.
    pub spec_verify: KdaSpecVerify,
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
    /// Plain decode's core, or a per-position verify's, keyed at
    /// `decode_query_len` tokens per request. Exactly one of this,
    /// `recoverssm_verify` and `recoverssm_conv_verify` is set.
    pub recurrent_decode: Option<KdaRecurrentDecodeKernelConfig>,
    /// A RecoverSSM verify graph's core after `conv_decode`, where the fork's
    /// CuTe kernels cannot run.
    pub recoverssm_verify: Option<KdaRecoverSSMVerifyKernelConfig>,
    /// A RecoverSSM verify graph's core with the short conv inside it (the
    /// fork's CuTe kernel); `conv_decode` then bills nothing for the decode requests.
    pub recoverssm_conv_verify: Option<KdaRecoverSSMConvVerifyKernelConfig>,
    /// The commit that goes with a RecoverSSM verify, for the arch to bill
    /// per KDA cache group: it sets `num_layers` to each group's layer count;
    /// here it is 1.
    pub recoverssm_commit: Option<KdaRecoverSSMCommitKernelConfig>,
    pub state_scatter: crate::timing::kernels::ElementwiseKernelConfig,
    pub gated_norm: crate::timing::kernels::ElementwiseKernelConfig,
    pub o_proj: SingleGemmKernelConfig,
    /// `[H, K, V]` recurrent state of one request on this rank, at the pool
    /// dtype.
    pub ssm_state_bytes_per_request: u32,
    /// Activation-dtype `(kernel_size - 1 + decode_query_len - 1)`-token conv
    /// window of one request on this rank.
    pub conv_state_bytes_per_request: u32,
    /// State pages (SSM state + conv window) one request holds per layer: 1,
    /// or `decode_query_len` for a per-position verify (vLLM `MambaSpec`
    /// `1 + num_speculative_blocks`).
    pub state_pages_per_request: u32,
    /// RecoverSSM's per-row verify records of one request on this rank: an
    /// fp32 correction `[H, rows, V]` and an activation-dtype key/gate
    /// `[H, rows, 2K]` (vLLM `append_kda_recoverssm_record`). Zero without a
    /// RecoverSSM verify.
    pub verify_cache_bytes_per_request: u32,
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
    pub prefill_small_glue: Op<ElementwiseKernel>,
    pub state_gather: Op<ElementwiseKernel>,
    pub chunk_prefill: Op<KdaChunkPrefillKernel>,
    pub recurrent_decode: Option<Op<KdaRecurrentDecodeKernel>>,
    pub recoverssm_verify: Option<Op<KdaRecoverSSMVerifyKernel>>,
    pub recoverssm_conv_verify: Option<Op<KdaRecoverSSMConvVerifyKernel>>,
    pub state_scatter: Op<ElementwiseKernel>,
    pub gated_norm: Op<ElementwiseKernel>,
    pub o_proj: Op<SingleGemmKernel>,
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
        assert!(
            matches!(cfg.ssm_state_dtype, DType::Fp32 | DType::Bf16),
            "KDA state pool must be fp32 or bf16, got {:?}",
            cfg.ssm_state_dtype
        );
        let ssm_state_bytes_per_request =
            heads * head_dim * head_dim * cfg.ssm_state_dtype.size_bytes();
        assert!(cfg.decode_query_len >= 1, "decode_query_len must be >= 1");
        let verify = cfg.decode_query_len > 1;
        let per_position = verify && cfg.spec_verify == KdaSpecVerify::RecurrentPerPosition;
        let recoverssm = verify && cfg.spec_verify == KdaSpecVerify::Recoverssm;
        let cute = recoverssm && cute_recoverssm(cfg);
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
                // A verify runs `causal_conv1d_update` in its varlen form over
                // `decode_query_len` rows per request.
                tokens_per_request: verify.then(|| cfg.decode_query_len.into()),
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
            recurrent_decode: (!recoverssm).then(|| KdaRecurrentDecodeKernelConfig {
                backends: cfg.core_backends.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.num_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                dtype: cfg.activation_dtype,
                state_dtype: cfg.ssm_state_dtype,
                tokens_per_request: per_position.then(|| cfg.decode_query_len.into()),
            }),
            recoverssm_verify: (recoverssm && !cute).then(|| KdaRecoverSSMVerifyKernelConfig {
                backends: RECOVERSSM_TRITON_BACKENDS.to_vec(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.num_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                spec_query_len: cfg.decode_query_len.into(),
                dtype: cfg.activation_dtype,
                state_dtype: cfg.ssm_state_dtype,
            }),
            recoverssm_conv_verify: cute.then(|| KdaRecoverSSMConvVerifyKernelConfig {
                backends: RECOVERSSM_CUTE_BACKENDS.to_vec(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.num_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                spec_query_len: cfg.decode_query_len.into(),
                dtype: cfg.activation_dtype,
                conv_weight_dtype: CONV_WEIGHT_DTYPE,
            }),
            recoverssm_commit: recoverssm.then(|| KdaRecoverSSMCommitKernelConfig {
                backends: if cute {
                    RECOVERSSM_CUTE_BACKENDS
                } else {
                    RECOVERSSM_TRITON_BACKENDS
                }
                .to_vec(),
                gpu_name: cfg.gpu_name.clone(),
                num_heads: cfg.num_heads.clone(),
                head_dim: cfg.head_dim.clone(),
                spec_query_len: cfg.decode_query_len.into(),
                num_layers: 1.into(),
                dtype: cfg.activation_dtype,
                state_dtype: cfg.ssm_state_dtype,
            }),
            state_scatter: ew(STATE_TRANSFER_UNIT_BYTES, STATE_TRANSFER_UNIT_BYTES),
            // rmsnorm(o) * w * sigmoid(g): reads o and g, writes the gated o.
            gated_norm: ew(2 * per_head_bytes, per_head_bytes),
            o_proj: gemm(cfg.hidden.clone(), value_width.into()),
            ssm_state_bytes_per_request,
            // vLLM `kda_state_shape`: the window grows by the drafted rows.
            conv_state_bytes_per_request: conv_channels
                * (cfg.conv_kernel_size.get() - 1 + cfg.decode_query_len - 1)
                * bpe,
            state_pages_per_request: if per_position {
                cfg.decode_query_len
            } else {
                1
            },
            verify_cache_bytes_per_request: if recoverssm {
                heads
                    * cfg.decode_query_len
                    * (head_dim * DType::Fp32.size_bytes() + 2 * head_dim * bpe)
            } else {
                0
            },
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
            recurrent_decode: r
                .recurrent_decode
                .map(|c| {
                    atomic(
                        n,
                        "recurrent_decode",
                        c,
                        KdaRecurrentDecodeKernel::build,
                        bridge,
                    )
                })
                .transpose()?,
            recoverssm_verify: r
                .recoverssm_verify
                .map(|c| {
                    atomic(
                        n,
                        "recoverssm_verify",
                        c,
                        KdaRecoverSSMVerifyKernel::build,
                        bridge,
                    )
                })
                .transpose()?,
            recoverssm_conv_verify: r
                .recoverssm_conv_verify
                .map(|c| {
                    atomic(
                        n,
                        "recoverssm_conv_verify",
                        c,
                        KdaRecoverSSMConvVerifyKernel::build,
                        bridge,
                    )
                })
                .transpose()?,
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
        CostNode::Labeled {
            label: format!(
                "{} (Glm53KdaAttnLocalWorklet) [TP rank; H={}, D={}]",
                self.name, cfg.num_heads, cfg.head_dim
            ),
            child: Box::new(CostNode::Sum(
                [
                    self.in_proj.compile(builder),
                    self.f_b.compile(builder),
                    self.g_b.compile(builder),
                    self.conv_prefill.compile(builder),
                    self.conv_decode.compile(builder),
                    repeated(
                        &self.prefill_small_glue,
                        self.beta_sigmoid_launches,
                        builder,
                    ),
                    self.state_gather.compile(builder),
                    self.chunk_prefill.compile(builder),
                ]
                .into_iter()
                // The decode core: exactly one of the three.
                .chain(self.recurrent_decode.as_ref().map(|op| op.compile(builder)))
                .chain(
                    self.recoverssm_verify
                        .as_ref()
                        .map(|op| op.compile(builder)),
                )
                .chain(
                    self.recoverssm_conv_verify
                        .as_ref()
                        .map(|op| op.compile(builder)),
                )
                .chain([
                    self.state_scatter.compile(builder),
                    self.gated_norm.compile(builder),
                    self.o_proj.compile(builder),
                ])
                .collect(),
            )),
        }
    }

    pub fn eval(&self, input: &Glm53KdaAttnLocalWorkletInput, ev: &mut Evaluator) {
        let w = derive_work(
            input,
            self.resolved.raw_cfg.decode_query_len,
            self.resolved.ssm_state_bytes_per_request,
        )
        .unwrap_or_else(|reason| panic!("invalid Glm53KdaAttnLocalWorkletInput: {reason}"));
        let rows = SingleGemmKernelInput { m: w.total_tokens };
        let tokens = ElementwiseKernelInput {
            num_tokens: w.total_tokens,
        };
        let no_decodes = input.decode_batch_size == 0;
        // The chunk path runs over every token of a prefill-bearing
        // plain-decode iteration, or over a verify's prefill requests alone.
        let decode_only = !w.prefill_bearing;
        let per_sequence = self.conv_per_sequence;
        push_or_zero(&self.in_proj, rows.clone(), false, ev);
        push_or_zero(&self.f_b, rows.clone(), false, ev);
        push_or_zero(&self.g_b, rows.clone(), false, ev);
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
            !if w.verify {
                // Beside any prefill; the fused conv + verify runs it inside.
                !no_decodes && self.recoverssm_conv_verify.is_none()
            } else {
                decode_conv_runs(per_sequence, w.prefill_bearing, input.decode_batch_size)
            },
            ev,
        );
        push_or_zero(
            &self.prefill_small_glue,
            ElementwiseKernelInput {
                num_tokens: w.chunk_tokens,
            },
            decode_only,
            ev,
        );
        let state = ElementwiseKernelInput {
            num_tokens: w.state_units,
        };
        push_or_zero(&self.state_gather, state.clone(), decode_only, ev);
        push_or_zero(
            &self.chunk_prefill,
            KdaChunkPrefillKernelInput {
                num_tokens: w.chunk_tokens,
                max_sequence_length: w.max_prefill_length,
                num_decode_sequences: w.chunk_decodes,
            },
            decode_only,
            ev,
        );
        // Plain decode: a prefill-bearing iteration chunks its decodes. A
        // verify keeps its spec decodes on its own kernel beside any prefill.
        if let Some(op) = &self.recurrent_decode {
            push_or_zero(
                op,
                KdaRecurrentDecodeKernelInput {
                    batch_size: input.decode_batch_size,
                },
                if w.verify {
                    no_decodes
                } else {
                    w.prefill_bearing
                },
                ev,
            );
        }
        if let Some(op) = &self.recoverssm_verify {
            push_or_zero(
                op,
                KdaRecoverSSMVerifyKernelInput {
                    batch_size: input.decode_batch_size,
                },
                no_decodes,
                ev,
            );
        }
        if let Some(op) = &self.recoverssm_conv_verify {
            push_or_zero(
                op,
                KdaRecoverSSMConvVerifyKernelInput {
                    batch_size: input.decode_batch_size,
                },
                no_decodes,
                ev,
            );
        }
        push_or_zero(&self.state_scatter, state, decode_only, ev);
        push_or_zero(&self.gated_norm, tokens, false, ev);
        push_or_zero(&self.o_proj, rows, false, ev);
    }
}

/// Whether the fork's CuTe RecoverSSM kernels serve this layer
/// (`can_use_cute_recoverssm`): an SM10x GPU, head dim 128, a width-4 conv, a
/// 1..8-token window, and bf16 activations, conv state and SSM state.
fn cute_recoverssm(cfg: &Glm53KdaAttnLocalWorkletConfig) -> bool {
    matches!(compute_capability(&cfg.gpu_name), Some((10, _)))
        && cfg.head_dim.get() == CUTE_RECOVERSSM_HEAD_DIM
        && cfg.conv_kernel_size.get() == CUTE_RECOVERSSM_CONV_WIDTH
        && cfg.decode_query_len <= CUTE_RECOVERSSM_MAX_WINDOW
        && cfg.activation_dtype == DType::Bf16
        && cfg.ssm_state_dtype == DType::Bf16
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
    /// Whether the chunk path runs this iteration.
    prefill_bearing: bool,
    max_prefill_length: u32,
    /// 4-KiB units of every chunked request's state.
    state_units: u32,
    /// A verify graph (`decode_query_len > 1`).
    verify: bool,
    /// Tokens and decode sequences the chunk path covers.
    chunk_tokens: u32,
    chunk_decodes: u32,
}

fn derive_work(
    input: &Glm53KdaAttnLocalWorkletInput,
    decode_query_len: u32,
    state_bytes: u32,
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
        .checked_mul(decode_query_len)
        .ok_or("decode token count overflows u32")?;
    let total_tokens = prefill_tokens
        .checked_add(decode_tokens)
        .ok_or("token sum overflows u32")?;
    if total_tokens == 0 {
        return Err("an iteration must carry at least one token".into());
    }
    // A plain-decode iteration with any prefill chunks every token; a verify
    // chunks only its prefill requests.
    let verify = decode_query_len > 1;
    let (chunk_tokens, chunk_decodes) = if verify {
        (prefill_tokens, 0)
    } else {
        (total_tokens, input.decode_batch_size)
    };
    let chunked_requests = input.prefill_sequence_lengths.len() as u32 + chunk_decodes;
    let state_units = chunked_requests
        .checked_mul(state_bytes / STATE_TRANSFER_UNIT_BYTES)
        .ok_or("state transfer units overflow u32")?;
    Ok(Work {
        total_tokens,
        prefill_bearing: prefill_tokens > 0,
        max_prefill_length,
        state_units,
        verify,
        chunk_tokens,
        chunk_decodes,
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
            ssm_state_dtype: DType::Fp32,
            decode_query_len: 1,
            spec_verify: KdaSpecVerify::Recoverssm,
        }
    }

    fn slot_names(cfg: &Glm53KdaAttnLocalWorkletConfig) -> Vec<String> {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let worklet = Glm53KdaAttnLocalWorklet::build(
            "m.kda".into(),
            Glm53KdaAttnLocalWorklet::resolve_config(cfg),
            &bridge,
        )
        .unwrap();
        let mut builder = CostTreeBuilder::new();
        let root = worklet.compile(&mut builder);
        let tree = builder.finish(root);
        tree.slots.iter().map(|slot| slot.name.clone()).collect()
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

    /// Catches a bf16 pool sized or keyed as the fp32 one: the state a
    /// prefill gathers and scatters halves, and the recurrent call reads the
    /// bf16-pool rows.
    #[test]
    fn a_bf16_pool_halves_the_state_and_keys_the_recurrent_rows() {
        let mut bf16 = cfg();
        bf16.ssm_state_dtype = DType::Bf16;
        let r = Glm53KdaAttnLocalWorklet::resolve_config(&bf16);
        assert_eq!(r.ssm_state_bytes_per_request, 1 << 19);
        assert_eq!(r.recurrent_decode.unwrap().state_dtype, DType::Bf16);
        let fp32 = Glm53KdaAttnLocalWorklet::resolve_config(&cfg());
        assert_eq!(fp32.recurrent_decode.unwrap().state_dtype, DType::Fp32);
    }

    #[test]
    fn prefill_bearing_iteration_chunks_every_token() {
        let w = derive_work(
            &Glm53KdaAttnLocalWorkletInput {
                prefill_sequence_lengths: vec![2019],
                decode_batch_size: 29,
            },
            1,
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
            1,
            1 << 20,
        )
        .unwrap();
        assert!(!decode.prefill_bearing);
        assert!(derive_work(&Glm53KdaAttnLocalWorkletInput::default(), 1, 1 << 20).is_err());
    }

    #[test]
    fn compile_has_fixed_slots_for_both_core_paths() {
        let slots = slot_names(&cfg());
        assert_eq!(slots.len(), 12);
        assert_eq!(slots[7], "m.kda.chunk_prefill");
        assert_eq!(slots[8], "m.kda.recurrent_decode");
    }

    #[test]
    fn a_verify_chunks_only_its_prefills_and_steps_every_decode_row() {
        // vLLM keeps spec decodes on their own kernel: the chunk path and its
        // state copies cover the prefill requests alone, while the
        // projections see every verify row.
        let w = derive_work(
            &Glm53KdaAttnLocalWorkletInput {
                prefill_sequence_lengths: vec![100],
                decode_batch_size: 4,
            },
            8,
            1 << 20,
        )
        .unwrap();
        assert!(w.verify && w.prefill_bearing);
        assert_eq!(w.total_tokens, 132);
        assert_eq!((w.chunk_tokens, w.chunk_decodes), (100, 0));
        assert_eq!(w.state_units, 256);
    }

    #[test]
    fn a_verify_graph_swaps_the_decode_core_for_recoverssm() {
        let mut verify = cfg();
        verify.decode_query_len = 8;
        let r = Glm53KdaAttnLocalWorklet::resolve_config(&verify);
        // Correction fp32 [16, 8, 128] plus key/gate bf16 [16, 8, 256].
        assert_eq!(
            r.verify_cache_bytes_per_request,
            16 * 8 * (128 * 4 + 256 * 2)
        );
        assert_eq!(r.conv_state_bytes_per_request, 6144 * (3 + 7) * 2);
        // The conv keys the varlen verify rows (R requests x 8 tokens), not
        // 8R one-token updates; plain decode keys the one-token rows.
        assert_eq!(r.conv_decode.tokens_per_request, Some(8.into()));
        let plain = Glm53KdaAttnLocalWorklet::resolve_config(&cfg());
        assert_eq!(plain.conv_decode.tokens_per_request, None);
        assert_eq!(plain.verify_cache_bytes_per_request, 0);
        assert!(plain.recoverssm_commit.is_none());
        // An fp32 pool: the Triton conv + verify, and the Triton commit.
        assert_eq!(r.recoverssm_commit.unwrap().backends, ["vllm_triton"]);
        let slots = slot_names(&verify);
        assert_eq!(slots.len(), 12);
        assert_eq!(slots[8], "m.kda.recoverssm_verify");
    }

    #[test]
    fn a_bf16_sm100_verify_runs_the_fused_conv_verify_and_others_fall_back() {
        // can_use_cute_recoverssm: SM10x, bf16 state and activations, a
        // window up to 8. Anything else keeps the Triton conv + verify.
        let mut fused = cfg();
        fused.decode_query_len = 8;
        fused.ssm_state_dtype = DType::Bf16;
        let r = Glm53KdaAttnLocalWorklet::resolve_config(&fused);
        assert!(r.recoverssm_verify.is_none());
        assert_eq!(r.recoverssm_conv_verify.unwrap().backends, ["vllm_cute"]);
        assert_eq!(r.recoverssm_commit.unwrap().backends, ["vllm_cute"]);
        assert_eq!(slot_names(&fused)[8], "m.kda.recoverssm_conv_verify");

        let mut fp32 = fused.clone();
        fp32.ssm_state_dtype = DType::Fp32;
        assert_eq!(slot_names(&fp32)[8], "m.kda.recoverssm_verify");
        let mut hopper = fused.clone();
        hopper.gpu_name = "NVIDIA H100 80GB HBM3".into();
        assert_eq!(slot_names(&hopper)[8], "m.kda.recoverssm_verify");
        let mut wide = fused.clone();
        wide.decode_query_len = 9;
        assert_eq!(slot_names(&wide)[8], "m.kda.recoverssm_verify");
    }

    #[test]
    fn a_per_position_verify_keeps_the_recurrent_call_and_one_page_per_position() {
        let mut fork = cfg();
        fork.decode_query_len = 8;
        fork.spec_verify = KdaSpecVerify::RecurrentPerPosition;
        fork.ssm_state_dtype = DType::Bf16;
        let r = Glm53KdaAttnLocalWorklet::resolve_config(&fork);
        assert!(r.recoverssm_verify.is_none() && r.recoverssm_conv_verify.is_none());
        let recurrent = r.recurrent_decode.as_ref().unwrap();
        assert_eq!(recurrent.tokens_per_request, Some(8.into()));
        assert_eq!(recurrent.state_dtype, DType::Bf16);
        // 1 + k pages, no RecoverSSM records and no commit.
        assert_eq!(r.state_pages_per_request, 8);
        assert_eq!(r.verify_cache_bytes_per_request, 0);
        assert!(r.recoverssm_commit.is_none());
        assert_eq!(slot_names(&fork)[8], "m.kda.recurrent_decode");
        // RecoverSSM keeps one page.
        fork.spec_verify = KdaSpecVerify::Recoverssm;
        let r = Glm53KdaAttnLocalWorklet::resolve_config(&fork);
        assert_eq!(r.state_pages_per_request, 1);
        assert!(r.recurrent_decode.is_none());
    }
}
