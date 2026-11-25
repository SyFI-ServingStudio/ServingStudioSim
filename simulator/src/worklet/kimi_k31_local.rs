//! Kimi-K3.1 sublayers on one tensor-parallel rank, in the launch order of the
//! vLLM Kimi-K3 fork (`vllm/models/kimi_k3/nvidia/{kda,mla,model}.py`) as the
//! GB200 TP16 and B200 TP8 x DP2 captures run it
//! (`logs/20261119_kimi_k31_gb200_tp16_phase1`, `.._b200x2_phase1`).
//!
//! Every sublayer starts after the block attention-residual combine
//! (`attn_res_fwd_online_v2`, billed by the arch) and ends before its TP
//! collective, which the arch owns because it differs by deployment
//! (all-reduce at TP16, sequence-parallel reduce-scatter at TP8 x DP2).
//!
//! - [`KimiKdaAttnLocal`]: MXFP8 `in_proj_qkvgfab` (activation quant inside the
//!   leaf), the BF16 `f_b` gate up-projection, a glue copy, then per request
//!   kind: spec-verify decodes run `causal_conv1d_update` (varlen, verify width
//!   rows) and the Triton RecoverSSM verify; prefills run the packed causal
//!   conv, the initial-state gather and FlashKDA (`_flash_kda_fwd_prepare` +
//!   `_flash_kda_fwd_recurrence`), the checkpoint store and two small glue
//!   launches. Both kinds share the gated output norm (FLA
//!   `layer_norm_gated_fwd`) and the MXFP8 `o_proj`.
//! - [`KimiMlaAttnLocal`]: the NoPE MLA with the DSA indexer, BF16 throughout
//!   except the FP8 latent cache and the MXFP4 indexer cache; the output gate
//!   (`g_proj`, sigmoid, multiply) precedes `o_proj`.
//! - [`KimiMlpLocal`]: SiTU-GLU MLP, BF16 (the dense layer 0) or MXFP8 (the
//!   shared experts).
//!
//! Proxies, each a measured kernel standing in for the fork's own until its
//! rows exist: the MXFP4 indexer logits use the FP8 DeepGEMM rows at the same
//! geometry; the packed causal-conv prefill uses `causal_conv1d_fn`; SiTU-and-
//! mul uses `silu_and_mul`. Glue copies are byte-sized elementwise placeholders.

use crate::op::Op;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    BatchedGemmKernel, BatchedGemmKernelConfig, BatchedGemmKernelInput, DsaIndexCacheAppendKernel,
    DsaIndexCacheAppendKernelConfig, DsaIndexCacheAppendKernelInput, DsaMqaLogitsPrefillKernel,
    DsaMqaLogitsPrefillKernelConfig, DsaMqaLogitsPrefillKernelInput,
    DsaPagedMqaLogitsDecodeKernel, DsaPagedMqaLogitsDecodeKernelConfig,
    DsaPagedMqaLogitsDecodeKernelInput, DsaPersistentTopkDecodeKernel,
    DsaPersistentTopkDecodeKernelConfig, DsaPersistentTopkDecodeKernelInput,
    DsaSparseMlaAttentionKernel, DsaSparseMlaAttentionKernelConfig,
    DsaSparseMlaAttentionKernelInput, DsaSparseMlaPrefillKernel, DsaSparseMlaPrefillKernelConfig,
    DsaSparseMlaPrefillKernelInput, DsaTopkPrefillKernel, DsaTopkPrefillKernelConfig,
    DsaTopkPrefillKernelInput, ElementwiseKernel, ElementwiseKernelConfig, ElementwiseKernelInput,
    GdnCausalConvDecodeKernel, GdnCausalConvDecodeKernelConfig, GdnCausalConvDecodeKernelInput,
    GdnCausalConvPrefillKernel, GdnCausalConvPrefillKernelConfig, GdnCausalConvPrefillKernelInput,
    KdaChunkPrefillKernel, KdaChunkPrefillKernelConfig, KdaChunkPrefillKernelInput,
    KdaRecoverSSMVerifyKernel, KdaRecoverSSMVerifyKernelConfig, KdaRecoverSSMVerifyKernelInput,
    MlaCacheAppendKernel, MlaCacheAppendKernelConfig, MlaCacheAppendKernelInput, RmsNormKernel,
    RmsNormKernelConfig, RmsNormKernelInput, SigmoidGatedRmsNormKernel,
    SigmoidGatedRmsNormKernelConfig, SigmoidGatedRmsNormKernelInput, SiluAndMulKernel,
    SiluAndMulKernelConfig, SiluAndMulKernelInput, SingleGemmKernel, SingleGemmKernelConfig,
    SingleGemmKernelInput, StaticFp8QuantKernel, StaticFp8QuantKernelConfig,
    StaticFp8QuantKernelInput, SparseMlaShardQKernel, SparseMlaShardQKernelConfig,
    SparseMlaShardQKernelInput,
};
use crate::timing::kernels::dsa_sparse_mla_attention::ValidCountsPattern;
use crate::timing::kernels::sparse_indexer_topk::COOPERATIVE_MAX_ROWS;
use crate::timing::{BuildError, CostNode, CostTreeBuilder, Evaluator, LeafMetrics, Probe, SlotInput};
use crate::timing::PerfApiBridge;

use super::glm53_common::{atomic, push_or_zero};

/// Pushes one slot that sums the leaf over `inputs` (a request-local fan-in in
/// one launch slot). An empty list pushes zero, logged at `empty`.
fn push_fanin<K>(op: &Op<K>, inputs: &[K::Input], empty: K::Input, ev: &mut Evaluator)
where
    K: Probe,
    K::Input: Clone + Into<SlotInput>,
{
    let mut metrics = LeafMetrics::ZERO;
    for input in inputs {
        metrics.add_fanin(op.kernel.eval(input));
    }
    let logged = inputs.first().cloned().unwrap_or(empty);
    ev.push(metrics, || logged.into());
}

fn ew(backends: &[&'static str], gpu: &str, input: u32, output: u32) -> ElementwiseKernelConfig {
    ElementwiseKernelConfig {
        backends: backends.to_vec(),
        gpu_name: gpu.to_string(),
        input_bytes_per_token: input.into(),
        output_bytes_per_token: output.into(),
    }
}

fn gemm(backends: &[&'static str], gpu: &str, n: u32, k: u32, dtype: DType) -> SingleGemmKernelConfig {
    SingleGemmKernelConfig {
        backends: backends.to_vec(),
        gpu_name: gpu.to_string(),
        n: n.into(),
        k: k.into(),
        dtype,
    }
}

fn rows(m: u32) -> SingleGemmKernelInput {
    SingleGemmKernelInput { m }
}

/// The backends every Kimi-K3.1 sublayer takes. A run config's `backends`
/// map may override any role.
#[derive(Clone, Debug)]
pub struct KimiK31Backends {
    pub bf16_gemm: Vec<&'static str>,
    pub mxfp8_gemm: Vec<&'static str>,
    pub elementwise: Vec<&'static str>,
    pub rms_norm: Vec<&'static str>,
    pub act: Vec<&'static str>,
    pub conv_decode: Vec<&'static str>,
    pub conv_prefill: Vec<&'static str>,
    pub kda_verify: Vec<&'static str>,
    pub kda_prefill: Vec<&'static str>,
    pub gated_norm: Vec<&'static str>,
    pub q_absorb: Vec<&'static str>,
    pub v_up: Vec<&'static str>,
    pub index_cache_append: Vec<&'static str>,
    pub index_decode_logits: Vec<&'static str>,
    pub index_decode_topk: Vec<&'static str>,
    /// Decode top-k above the cooperative kernel's row ceiling, run eagerly.
    pub index_decode_topk_persistent: Vec<&'static str>,
    /// Decode top-k above the cooperative ceiling, replayed from a full graph.
    pub index_decode_topk_graph: Vec<&'static str>,
    pub index_prefill_logits: Vec<&'static str>,
    pub index_prefill_topk: Vec<&'static str>,
    pub mla_cache_append: Vec<&'static str>,
    pub static_quant: Vec<&'static str>,
    pub sparse_attention: Vec<&'static str>,
    pub sparse_prefill: Vec<&'static str>,
    pub shard_q: Vec<&'static str>,
}

impl Default for KimiK31Backends {
    fn default() -> Self {
        Self {
            bf16_gemm: vec!["torch_linear_vllm"],
            mxfp8_gemm: vec!["flashinfer_mxfp8"],
            // Glue between compiled ops runs as Inductor-fused Triton kernels.
            elementwise: vec!["triton_vllm"],
            rms_norm: vec!["vllm_cuda"],
            act: vec!["vllm_cuda"],
            conv_decode: vec!["vllm_triton"],
            conv_prefill: vec!["vllm_triton"],
            kda_verify: vec!["vllm_triton"],
            kda_prefill: vec!["flashkda"],
            gated_norm: vec!["vllm_fla_triton"],
            q_absorb: vec!["torch_mla_q_absorb_nope128_rope64"],
            v_up: vec!["torch_mla_v_up_v128_unpadded"],
            index_cache_append: vec!["vllm_cuda"],
            index_decode_logits: vec!["deepgemm_fp8"],
            // vLLM `indexer_topk.py` picks the decode top-k by row count:
            // cooperative_topk up to 64 rows, persistent_topk above.
            index_decode_topk: vec!["vllm_cuda_cooperative"],
            index_decode_topk_persistent: vec!["vllm_cuda"],
            index_decode_topk_graph: vec!["vllm_cuda_graph"],
            index_prefill_logits: vec!["deepgemm_fp8"],
            index_prefill_topk: vec!["vllm_cuda"],
            mla_cache_append: vec!["vllm_cuda"],
            static_quant: vec!["vllm_cuda"],
            sparse_attention: vec!["flashinfer_trtllm_fp8"],
            sparse_prefill: vec!["flashinfer_trtllm_fp8"],
            shard_q: vec!["vllm_symm_mem"],
        }
    }
}

/// One attention sublayer's requests on this rank: every prefill chunk as
/// `(appended tokens, context after the chunk)`, and every spec-verify decode
/// as its context after the verify rows.
#[derive(Clone, Debug, Default)]
pub struct KimiAttnInput {
    pub prefill_chunks: Vec<(u32, u32)>,
    pub decode_kv_lens: Vec<u32>,
    /// Rows the step's graph runs: the scheduled tokens, padded to the
    /// captured graph size when the step replays a graph.
    pub graph_tokens: u32,
    /// Whether the step replays a full CUDA graph, attention included. Host
    /// arguments recorded at capture (the decode top-k's `max_seq_len`) then
    /// hold their capture-time values.
    pub full_graph: bool,
}

/// Which decode top-k kernel a step runs (vLLM `indexer_topk.py`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeTopk {
    /// `cooperative_topk`, up to [`COOPERATIVE_MAX_ROWS`] rows.
    Cooperative,
    /// `persistent_topk` launched eagerly with the batch's `max_seq_len`.
    Persistent,
    /// `persistent_topk` replayed from a full graph, `max_seq_len` the
    /// capture-time max_model_len.
    Graph,
}

impl DecodeTopk {
    /// The kernel for `rows` verify rows, `None` when there are none.
    fn route(rows: u32, full_graph: bool) -> Option<Self> {
        match (rows, full_graph) {
            (0, _) => None,
            (r, _) if r <= COOPERATIVE_MAX_ROWS => Some(Self::Cooperative),
            (_, false) => Some(Self::Persistent),
            (_, true) => Some(Self::Graph),
        }
    }
}

impl KimiAttnInput {
    fn prefill_tokens(&self) -> u32 {
        self.prefill_chunks.iter().map(|&(q, _)| q).sum()
    }
}

// ---------------------------------------------------------------------------
// KDA

#[derive(Clone, Debug)]
pub struct KimiKdaAttnLocalConfig {
    pub gpu_name: String,
    pub hidden: u32,
    /// KDA heads on this rank.
    pub num_heads: u32,
    pub head_dim: u32,
    /// `in_proj_qkvgfab` output width on this rank, padded as vLLM pads it.
    pub in_proj_width: u32,
    /// Width of the `f_a` low-rank gate factor.
    pub gate_rank: u32,
    pub conv_kernel_size: u32,
    /// Query rows per spec-verify decode (`1 + k`).
    pub verify_width: u32,
    /// Conv and recurrent state pool dtype (`--mamba-cache-dtype`).
    pub state_dtype: DType,
    pub backends: KimiK31Backends,
}

pub struct KimiKdaAttnLocal {
    name: String,
    num_heads: u32,
    verify_width: u32,
    in_proj: Op<SingleGemmKernel>,
    f_b: Op<SingleGemmKernel>,
    glue: Op<ElementwiseKernel>,
    conv_decode: Op<GdnCausalConvDecodeKernel>,
    verify: Op<KdaRecoverSSMVerifyKernel>,
    conv_prefill: Op<GdnCausalConvPrefillKernel>,
    state_io: Op<ElementwiseKernel>,
    chunk_prefill: Op<KdaChunkPrefillKernel>,
    prefill_glue: Op<ElementwiseKernel>,
    gated_norm: Op<SigmoidGatedRmsNormKernel>,
    o_proj: Op<SingleGemmKernel>,
}

impl KimiKdaAttnLocal {
    pub fn build(name: String, cfg: &KimiKdaAttnLocalConfig, bridge: &PerfApiBridge) -> Result<Self, BuildError> {
        let b = &cfg.backends;
        let g = cfg.gpu_name.as_str();
        let n = name.as_str();
        let value_width = cfg.num_heads * cfg.head_dim;
        let channels = 3 * value_width;
        let bf16 = DType::Bf16;
        Ok(Self {
            in_proj: atomic(n, "in_proj", gemm(&b.mxfp8_gemm, g, cfg.in_proj_width, cfg.hidden, DType::Mxfp8E4m3), SingleGemmKernel::build, bridge)?,
            f_b: atomic(n, "f_b", gemm(&b.bf16_gemm, g, value_width, cfg.gate_rank, bf16), SingleGemmKernel::build, bridge)?,
            // The fp32 copy of the per-head gate parameters ahead of the core.
            glue: atomic(n, "glue_copy", ew(&b.elementwise, g, 4 * value_width, 4 * value_width), ElementwiseKernel::build, bridge)?,
            conv_decode: atomic(
                n,
                "conv_decode",
                GdnCausalConvDecodeKernelConfig {
                    backends: b.conv_decode.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    channels: channels.into(),
                    kernel_size: cfg.conv_kernel_size.into(),
                    dtype: bf16,
                    state_dtype: cfg.state_dtype,
                    tokens_per_request: (cfg.verify_width > 1).then(|| cfg.verify_width.into()),
                },
                GdnCausalConvDecodeKernel::build,
                bridge,
            )?,
            verify: atomic(
                n,
                "recoverssm_verify",
                KdaRecoverSSMVerifyKernelConfig {
                    backends: b.kda_verify.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    num_heads: cfg.num_heads.into(),
                    head_dim: cfg.head_dim.into(),
                    spec_query_len: cfg.verify_width.into(),
                    dtype: bf16,
                    state_dtype: cfg.state_dtype,
                },
                KdaRecoverSSMVerifyKernel::build,
                bridge,
            )?,
            conv_prefill: atomic(
                n,
                "conv_prefill",
                GdnCausalConvPrefillKernelConfig {
                    backends: b.conv_prefill.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    channels: channels.into(),
                    kernel_size: cfg.conv_kernel_size.into(),
                    dtype: bf16,
                    state_dtype: bf16,
                },
                GdnCausalConvPrefillKernel::build,
                bridge,
            )?,
            // `_gather_initial_states` + `store_cache_checkpoints`: one state
            // read and one checkpoint write per prefill request, billed per
            // request in 4 KiB units of the rank's state.
            state_io: atomic(n, "state_io", ew(&b.elementwise, g, 4096, 4096), ElementwiseKernel::build, bridge)?,
            chunk_prefill: atomic(
                n,
                "chunk_prefill",
                KdaChunkPrefillKernelConfig {
                    backends: b.kda_prefill.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    num_heads: cfg.num_heads.into(),
                    head_dim: cfg.head_dim.into(),
                    dtype: bf16,
                },
                KdaChunkPrefillKernel::build,
                bridge,
            )?,
            // `unrolled_elementwise` + `index_elementwise` after the core.
            prefill_glue: atomic(n, "prefill_glue", ew(&b.elementwise, g, 2 * value_width, 2 * value_width), ElementwiseKernel::build, bridge)?,
            gated_norm: atomic(
                n,
                "gated_norm",
                SigmoidGatedRmsNormKernelConfig {
                    backends: b.gated_norm.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    hidden: cfg.head_dim.into(),
                    dtype: bf16,
                },
                SigmoidGatedRmsNormKernel::build,
                bridge,
            )?,
            o_proj: atomic(n, "o_proj", gemm(&b.mxfp8_gemm, g, cfg.hidden, value_width, DType::Mxfp8E4m3), SingleGemmKernel::build, bridge)?,
            num_heads: cfg.num_heads,
            verify_width: cfg.verify_width,
            name,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!("{} (KimiKdaAttnLocal) [{} heads]", self.name, self.num_heads),
            child: Box::new(CostNode::Sum(vec![
                self.in_proj.compile(builder),
                self.f_b.compile(builder),
                self.glue.compile(builder),
                self.conv_decode.compile(builder),
                self.verify.compile(builder),
                self.conv_prefill.compile(builder),
                self.state_io.compile(builder),
                self.chunk_prefill.compile(builder),
                self.prefill_glue.compile(builder),
                self.gated_norm.compile(builder),
                self.o_proj.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, input: &KimiAttnInput, ev: &mut Evaluator) {
        let m = input.graph_tokens;
        let decodes = input.decode_kv_lens.len() as u32;
        let prefill_tokens = input.prefill_tokens();
        let prefill_requests = input.prefill_chunks.len() as u32;
        push_or_zero(&self.in_proj, rows(m), m == 0, ev);
        push_or_zero(&self.f_b, rows(m), m == 0, ev);
        push_or_zero(&self.glue, ElementwiseKernelInput { num_tokens: m }, m == 0, ev);
        push_or_zero(&self.conv_decode, GdnCausalConvDecodeKernelInput { batch_size: decodes }, decodes == 0, ev);
        push_or_zero(&self.verify, KdaRecoverSSMVerifyKernelInput { batch_size: decodes }, decodes == 0, ev);
        // One varlen launch over every prefill token.
        push_or_zero(
            &self.conv_prefill,
            GdnCausalConvPrefillKernelInput { batch_size: 1, sequence_length: prefill_tokens },
            prefill_tokens == 0,
            ev,
        );
        push_or_zero(&self.state_io, ElementwiseKernelInput { num_tokens: prefill_requests }, prefill_requests == 0, ev);
        let max_len = input.prefill_chunks.iter().map(|&(q, _)| q).max().unwrap_or(0);
        push_or_zero(
            &self.chunk_prefill,
            KdaChunkPrefillKernelInput { num_tokens: prefill_tokens, max_sequence_length: max_len, num_decode_sequences: 0 },
            prefill_tokens == 0,
            ev,
        );
        push_or_zero(&self.prefill_glue, ElementwiseKernelInput { num_tokens: prefill_tokens }, prefill_tokens == 0, ev);
        let norm_rows = m * self.num_heads;
        push_or_zero(&self.gated_norm, SigmoidGatedRmsNormKernelInput { m: norm_rows }, m == 0, ev);
        push_or_zero(&self.o_proj, rows(m), m == 0, ev);
        let _ = self.verify_width;
    }
}

// ---------------------------------------------------------------------------
// MLA + DSA indexer

#[derive(Clone, Debug)]
pub struct KimiMlaAttnLocalConfig {
    pub gpu_name: String,
    pub hidden: u32,
    /// MLA heads on this rank.
    pub num_heads: u32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    /// Indexer heads: replicated on every rank.
    pub index_num_heads: u32,
    pub index_head_dim: u32,
    pub index_topk: u32,
    pub verify_width: u32,
    pub max_model_len: u32,
    pub cache_block_size: u32,
    /// Attention TP size; above 1, vLLM's ShardQ re-shards large prefill
    /// steps by token (`sparse_mla_shard_q`).
    pub tp_size: u32,
    /// `attention_config.sparse_mla_shard_q_min_tokens`.
    pub shard_q_min_tokens: u32,
    pub backends: KimiK31Backends,
}

/// ShardQ ops (TP > 1): the token exchanges and the all-head sparse prefill.
struct ShardQOps {
    tp: u32,
    min_tokens: u32,
    shard_q: Op<SparseMlaShardQKernel>,
    unshard_out: Op<SparseMlaShardQKernel>,
    pad_heads: Option<Op<ElementwiseKernel>>,
    sparse_prefill_all_heads: Op<DsaSparseMlaPrefillKernel>,
}

/// vLLM `trtllm_gen_query_heads`: trtllm-gen MLA takes no head count strictly
/// between 64 and 128, so those are zero-padded to 128.
fn trtllm_gen_query_heads(num_heads: u32) -> u32 {
    if 64 < num_heads && num_heads < 128 { 128 } else { num_heads }
}

pub struct KimiMlaAttnLocal {
    name: String,
    num_heads: u32,
    verify_width: u32,
    index_topk: u32,
    fused_qkv_a: Op<SingleGemmKernel>,
    q_a_norm: Op<RmsNormKernel>,
    kv_a_norm: Op<RmsNormKernel>,
    q_b: Op<SingleGemmKernel>,
    index_wk: Op<SingleGemmKernel>,
    index_wq: Op<SingleGemmKernel>,
    index_cache_append: Op<DsaIndexCacheAppendKernel>,
    index_q_quant: Op<ElementwiseKernel>,
    index_prefill_logits: Op<DsaMqaLogitsPrefillKernel>,
    index_prefill_topk: Op<DsaTopkPrefillKernel>,
    index_decode_logits: Op<DsaPagedMqaLogitsDecodeKernel>,
    index_decode_topk: Op<DsaPersistentTopkDecodeKernel>,
    index_decode_topk_persistent: Op<DsaPersistentTopkDecodeKernel>,
    index_decode_topk_graph: Op<DsaPersistentTopkDecodeKernel>,
    mla_cache_append: Op<MlaCacheAppendKernel>,
    q_absorb: Op<BatchedGemmKernel>,
    q_quant: Op<StaticFp8QuantKernel>,
    index_remap: Op<ElementwiseKernel>,
    sparse_prefill: Op<DsaSparseMlaPrefillKernel>,
    shardq: Option<ShardQOps>,
    sparse_decode: Op<DsaSparseMlaAttentionKernel>,
    v_up: Op<BatchedGemmKernel>,
    g_proj: Op<SingleGemmKernel>,
    gate_mul: Op<ElementwiseKernel>,
    o_proj: Op<SingleGemmKernel>,
}

impl KimiMlaAttnLocal {
    pub fn build(name: String, cfg: &KimiMlaAttnLocalConfig, bridge: &PerfApiBridge) -> Result<Self, BuildError> {
        let b = &cfg.backends;
        let g = cfg.gpu_name.as_str();
        let n = name.as_str();
        let bf16 = DType::Bf16;
        let heads = cfg.num_heads;
        let q_head = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
        let latent = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
        let value_width = heads * cfg.v_head_dim;
        let bmm = |backends: &[&'static str], nn: u32, kk: u32| BatchedGemmKernelConfig {
            backends: backends.to_vec(),
            gpu_name: cfg.gpu_name.clone(),
            num_batches: heads.into(),
            n: nn.into(),
            k: kk.into(),
            dtype: bf16,
        };
        let norm = |hidden: u32| RmsNormKernelConfig {
            backends: b.rms_norm.clone(),
            gpu_name: cfg.gpu_name.clone(),
            hidden: hidden.into(),
            dtype: bf16,
        };
        let softmax_scale_denominator = (q_head as f64).sqrt().round() as u32;
        let sparse_decode = DsaSparseMlaAttentionKernelConfig {
            backends: b.sparse_attention.clone(),
            gpu_name: cfg.gpu_name.clone(),
            num_heads: heads.into(),
            num_kv_heads: 1u32.into(),
            selected_k: cfg.index_topk,
            latent_dim: cfg.kv_lora_rank.into(),
            rope_dim: cfg.qk_rope_head_dim.into(),
            value_dim: cfg.kv_lora_rank.into(),
            softmax_scale_denominator,
            q_dtype: DType::Fp8E4m3,
            cache_dtype: DType::Fp8E4m3,
            index_dtype: "int32".into(),
            output_dtype: bf16,
            valid_counts_pattern: ValidCountsPattern::for_decode(cfg.verify_width)
                .unwrap_or(ValidCountsPattern::UniformFull),
            index_distribution: "recent_contiguous".into(),
            cache_layout: "hnd_paged_mqa_fp8_latent_rope".into(),
        };
        let sparse_prefill = DsaSparseMlaPrefillKernelConfig {
            backends: b.sparse_prefill.clone(),
            gpu_name: cfg.gpu_name.clone(),
            max_model_len: cfg.max_model_len,
            num_heads: heads.into(),
            num_kv_heads: 1u32.into(),
            selected_k: cfg.index_topk,
            latent_dim: cfg.kv_lora_rank.into(),
            rope_dim: cfg.qk_rope_head_dim.into(),
            value_dim: cfg.kv_lora_rank.into(),
            softmax_scale_denominator,
            q_dtype: DType::Fp8E4m3,
            cache_dtype: DType::Fp8E4m3,
            index_dtype: "int32".into(),
            output_dtype: bf16,
            index_distribution: "recent_contiguous".into(),
            cache_layout: "hnd_paged_mqa_fp8_latent_rope".into(),
        };
        let shardq = if cfg.tp_size > 1 {
            let exchange = |direction: &str, head_dim: u32, dtype: DType| SparseMlaShardQKernelConfig {
                backends: b.shard_q.clone(),
                gpu_name: cfg.gpu_name.clone(),
                num_gpus: cfg.tp_size,
                num_heads: heads,
                head_dim: head_dim.into(),
                dtype,
                direction: direction.into(),
            };
            let gathered = heads * cfg.tp_size;
            let padded = trtllm_gen_query_heads(gathered);
            let mut all_heads = sparse_prefill.clone();
            all_heads.num_heads = padded.into();
            Some(ShardQOps {
                tp: cfg.tp_size,
                min_tokens: cfg.shard_q_min_tokens,
                // fp8 query [T, H/tp, kv_lora_rank + rope] to [T/tp, H, ...].
                shard_q: atomic(n, "shard_q", exchange("shard_q", latent, DType::Fp8E4m3), SparseMlaShardQKernel::build, bridge)?,
                // bf16 latent output [T/tp, H, kv_lora_rank] back to head shards.
                unshard_out: atomic(n, "unshard_out", exchange("unshard_out", cfg.kv_lora_rank, bf16), SparseMlaShardQKernel::build, bridge)?,
                // `_pad_query_heads_kernel`: fp8 [T/tp, H, latent] into [T/tp, padded, latent].
                pad_heads: if padded != gathered {
                    Some(atomic(n, "shard_q_pad_heads", ew(&b.elementwise, g, gathered * latent, padded * latent), ElementwiseKernel::build, bridge)?)
                } else {
                    None
                },
                sparse_prefill_all_heads: atomic(n, "sparse_prefill_shard_q", all_heads, DsaSparseMlaPrefillKernel::build, bridge)?,
            })
        } else {
            None
        };
        Ok(Self {
            shardq,
            fused_qkv_a: atomic(n, "fused_qkv_a", gemm(&b.bf16_gemm, g, cfg.q_lora_rank + latent, cfg.hidden, bf16), SingleGemmKernel::build, bridge)?,
            q_a_norm: atomic(n, "q_a_norm", norm(cfg.q_lora_rank), RmsNormKernel::build, bridge)?,
            kv_a_norm: atomic(n, "kv_a_norm", norm(cfg.kv_lora_rank), RmsNormKernel::build, bridge)?,
            q_b: atomic(n, "q_b", gemm(&b.bf16_gemm, g, heads * q_head, cfg.q_lora_rank, bf16), SingleGemmKernel::build, bridge)?,
            // `wkv_index`: the index key plus one weight per index head.
            index_wk: atomic(n, "index_wk", gemm(&b.bf16_gemm, g, cfg.index_head_dim + cfg.index_num_heads, cfg.hidden, bf16), SingleGemmKernel::build, bridge)?,
            index_wq: atomic(n, "index_wq", gemm(&b.bf16_gemm, g, cfg.index_num_heads * cfg.index_head_dim, cfg.q_lora_rank, bf16), SingleGemmKernel::build, bridge)?,
            index_cache_append: atomic(
                n,
                "index_cache_append",
                DsaIndexCacheAppendKernelConfig {
                    backends: b.index_cache_append.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    index_dim: cfg.index_head_dim.into(),
                    block_size: cfg.cache_block_size,
                    quant_block_size: 128,
                    input_dtype: bf16,
                    cache_dtype: DType::Fp8E4m3,
                    scale_format: "ue8m0".into(),
                    cache_format: "page_planar_fp8_fp32_scale".into(),
                },
                DsaIndexCacheAppendKernel::build,
                bridge,
            )?,
            // `_mxfp4_q_quant`: the bf16 index query in, packed e2m1 with one
            // ue8m0 scale per 32 elements out.
            index_q_quant: atomic(
                n,
                "index_q_quant",
                ew(&b.elementwise, g, 2 * cfg.index_num_heads * cfg.index_head_dim, cfg.index_num_heads * (cfg.index_head_dim / 2 + cfg.index_head_dim / 32)),
                ElementwiseKernel::build,
                bridge,
            )?,
            index_prefill_logits: atomic(
                n,
                "index_prefill_logits",
                DsaMqaLogitsPrefillKernelConfig {
                    backends: b.index_prefill_logits.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    num_sequences: 1,
                    num_heads: cfg.index_num_heads.into(),
                    head_dim: cfg.index_head_dim.into(),
                    q_dtype: DType::Fp8E4m3,
                    k_dtype: DType::Fp8E4m3,
                    k_scale_dtype: DType::Fp32,
                    weight_dtype: DType::Fp32,
                    output_dtype: DType::Fp32,
                    span_mode: "single_causal_tail".into(),
                    clean_logits: false,
                },
                DsaMqaLogitsPrefillKernel::build,
                bridge,
            )?,
            index_prefill_topk: atomic(
                n,
                "index_prefill_topk",
                DsaTopkPrefillKernelConfig {
                    backends: b.index_prefill_topk.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    num_sequences: 1,
                    top_k: cfg.index_topk,
                    logits_dtype: DType::Fp32,
                    index_dtype: "int32".into(),
                    span_mode: "single_causal_tail".into(),
                },
                DsaTopkPrefillKernel::build,
                bridge,
            )?,
            index_decode_logits: atomic(
                n,
                "index_decode_logits",
                DsaPagedMqaLogitsDecodeKernelConfig {
                    backends: b.index_decode_logits.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    next_n: cfg.verify_width,
                    max_model_len: cfg.max_model_len.into(),
                    num_heads: cfg.index_num_heads.into(),
                    head_dim: cfg.index_head_dim.into(),
                    block_size: cfg.cache_block_size,
                    q_dtype: DType::Fp8E4m3,
                    cache_dtype: DType::Fp8E4m3,
                    scale_dtype: DType::Fp32,
                    weight_dtype: DType::Fp32,
                    output_dtype: DType::Fp32,
                    context_mode: "uniform".into(),
                    page_mapping: "unique_scattered".into(),
                    cache_format: "page_planar_fp8_fp32_scale".into(),
                    clean_logits: false,
                },
                DsaPagedMqaLogitsDecodeKernel::build,
                bridge,
            )?,
            index_decode_topk: atomic(
                n,
                "index_decode_topk",
                DsaPersistentTopkDecodeKernelConfig {
                    backends: b.index_decode_topk.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    next_n: cfg.verify_width,
                    max_model_len: cfg.max_model_len.into(),
                    top_k: cfg.index_topk,
                    logits_row_stride: cfg.max_model_len.into(),
                    logits_dtype: DType::Fp32,
                    index_dtype: "int32".into(),
                    context_mode: "uniform_measured".into(),
                },
                DsaPersistentTopkDecodeKernel::build,
                bridge,
            )?,
            index_decode_topk_persistent: atomic(
                n,
                "index_decode_topk_persistent",
                DsaPersistentTopkDecodeKernelConfig {
                    backends: b.index_decode_topk_persistent.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    next_n: cfg.verify_width,
                    max_model_len: cfg.max_model_len.into(),
                    top_k: cfg.index_topk,
                    logits_row_stride: cfg.max_model_len.into(),
                    logits_dtype: DType::Fp32,
                    index_dtype: "int32".into(),
                    context_mode: "uniform_measured".into(),
                },
                DsaPersistentTopkDecodeKernel::build,
                bridge,
            )?,
            index_decode_topk_graph: atomic(
                n,
                "index_decode_topk_graph",
                DsaPersistentTopkDecodeKernelConfig {
                    backends: b.index_decode_topk_graph.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    next_n: cfg.verify_width,
                    max_model_len: cfg.max_model_len.into(),
                    top_k: cfg.index_topk,
                    logits_row_stride: cfg.max_model_len.into(),
                    logits_dtype: DType::Fp32,
                    index_dtype: "int32".into(),
                    context_mode: "uniform_measured".into(),
                },
                DsaPersistentTopkDecodeKernel::build,
                bridge,
            )?,
            mla_cache_append: atomic(
                n,
                "mla_cache_append",
                MlaCacheAppendKernelConfig {
                    backends: b.mla_cache_append.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    kv_lora_rank: cfg.kv_lora_rank.into(),
                    rope_dim: cfg.qk_rope_head_dim.into(),
                    block_size: cfg.cache_block_size,
                    input_dtype: bf16,
                    kv_dtype: DType::Fp8E4m3,
                    cache_format: "plain".into(),
                },
                MlaCacheAppendKernel::build,
                bridge,
            )?,
            q_absorb: atomic(n, "q_absorb", bmm(&b.q_absorb, cfg.kv_lora_rank, cfg.qk_nope_head_dim), BatchedGemmKernel::build, bridge)?,
            q_quant: atomic(
                n,
                "q_quant",
                StaticFp8QuantKernelConfig {
                    backends: b.static_quant.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    hidden_dim: (heads * latent).into(),
                    gated: false,
                    dtype: bf16,
                },
                StaticFp8QuantKernel::build,
                bridge,
            )?,
            // `sparse_mla_index_remap` (aux stream): int32 indices in and out.
            index_remap: atomic(n, "index_remap", ew(&b.elementwise, g, 4 * cfg.index_topk, 4 * cfg.index_topk), ElementwiseKernel::build, bridge)?,
            sparse_prefill: atomic(n, "sparse_prefill", sparse_prefill, DsaSparseMlaPrefillKernel::build, bridge)?,
            sparse_decode: atomic(n, "sparse_decode", sparse_decode, DsaSparseMlaAttentionKernel::build, bridge)?,
            v_up: atomic(n, "v_up", bmm(&b.v_up, cfg.v_head_dim, cfg.kv_lora_rank), BatchedGemmKernel::build, bridge)?,
            g_proj: atomic(n, "g_proj", gemm(&b.bf16_gemm, g, value_width, cfg.hidden, bf16), SingleGemmKernel::build, bridge)?,
            // sigmoid(g) then the multiply: two bf16 launches over [T, H * V].
            gate_mul: atomic(n, "gate_mul", ew(&b.elementwise, g, 6 * value_width, 4 * value_width), ElementwiseKernel::build, bridge)?,
            o_proj: atomic(n, "o_proj", gemm(&b.bf16_gemm, g, cfg.hidden, value_width, bf16), SingleGemmKernel::build, bridge)?,
            num_heads: heads,
            verify_width: cfg.verify_width,
            index_topk: cfg.index_topk,
            name,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        // Leaves are compiled in eval's push order. An absent stage adds no
        // child: the alignment analyzer rejects composites with no children,
        // so `Sum(vec![])` is not a "nothing" node.
        let mut children = vec![
            self.fused_qkv_a.compile(builder),
            self.q_a_norm.compile(builder),
            self.kv_a_norm.compile(builder),
            self.q_b.compile(builder),
            self.index_wk.compile(builder),
            self.index_wq.compile(builder),
            self.index_cache_append.compile(builder),
            self.index_q_quant.compile(builder),
            self.index_prefill_logits.compile(builder),
            self.index_prefill_topk.compile(builder),
            self.index_decode_logits.compile(builder),
            self.index_decode_topk.compile(builder),
            self.index_decode_topk_persistent.compile(builder),
            self.index_decode_topk_graph.compile(builder),
            self.mla_cache_append.compile(builder),
            self.q_absorb.compile(builder),
            self.q_quant.compile(builder),
            self.index_remap.compile(builder),
            self.sparse_prefill.compile(builder),
        ];
        if let Some(sq) = &self.shardq {
            let mut shard = vec![sq.shard_q.compile(builder)];
            shard.extend(sq.pad_heads.as_ref().map(|op| op.compile(builder)));
            shard.push(sq.sparse_prefill_all_heads.compile(builder));
            shard.push(sq.unshard_out.compile(builder));
            children.push(CostNode::Sum(shard));
        }
        children.extend([
            self.sparse_decode.compile(builder),
            self.v_up.compile(builder),
            self.g_proj.compile(builder),
            self.gate_mul.compile(builder),
            self.o_proj.compile(builder),
        ]);
        CostNode::Labeled {
            label: format!("{} (KimiMlaAttnLocal) [{} heads, DSA top-{}]", self.name, self.num_heads, self.index_topk),
            child: Box::new(CostNode::Sum(children)),
        }
    }

    pub fn eval(&self, input: &KimiAttnInput, ev: &mut Evaluator) {
        let m = input.graph_tokens;
        let z = m == 0;
        let decodes = input.decode_kv_lens.len() as u32;
        let decode_rows = decodes * self.verify_width;
        let prefill_tokens = input.prefill_tokens();
        // vLLM `build_shardq_split`: TP > 1, enough prefill tokens, a prefill
        // longer than the top-k, and every rank owning a zigzag chunk.
        let shard_tp = self.shardq.as_ref().filter(|sq| {
            prefill_tokens > 0
                && prefill_tokens >= sq.min_tokens
                && input.prefill_chunks.iter().any(|&(_, kv)| kv > self.index_topk)
                && input.prefill_chunks.iter().map(|&(q, _)| q).max().unwrap_or(0) >= 2 * sq.tp
        }).map(|sq| sq.tp);
        // Under ShardQ a rank runs the prefill indexer and attention on its
        // zigzag 1/tp of each chunk's queries. The zigzag keeps the mean
        // causal key span at the chunk's (`kv - q/2`), so the rank's tail is
        // `q' = ceil(q/tp)` queries ending at `kv - q/2 + q'/2`.
        let rank_chunks: Vec<(u32, u32)> = match shard_tp {
            Some(tp) => input
                .prefill_chunks
                .iter()
                .map(|&(q, kv)| {
                    let qr = q.div_ceil(tp);
                    (qr, (kv - q / 2 + qr / 2).max(qr))
                })
                .collect(),
            None => input.prefill_chunks.clone(),
        };
        push_or_zero(&self.fused_qkv_a, rows(m), z, ev);
        push_or_zero(&self.q_a_norm, RmsNormKernelInput { m }, z, ev);
        push_or_zero(&self.kv_a_norm, RmsNormKernelInput { m }, z, ev);
        push_or_zero(&self.q_b, rows(m), z, ev);
        push_or_zero(&self.index_wk, rows(m), z, ev);
        push_or_zero(&self.index_wq, rows(m), z, ev);
        push_or_zero(&self.index_cache_append, DsaIndexCacheAppendKernelInput { num_tokens: m }, z, ev);
        push_or_zero(&self.index_q_quant, ElementwiseKernelInput { num_tokens: m }, z, ev);
        // Prefill indexer: one causal-tail call per chunk, summed.
        let prefill_logits: Vec<_> = rank_chunks
            .iter()
            .map(|&(q, kv)| DsaMqaLogitsPrefillKernelInput { num_queries: q, num_keys: kv })
            .collect();
        push_fanin(&self.index_prefill_logits, &prefill_logits, DsaMqaLogitsPrefillKernelInput { num_queries: 0, num_keys: 0 }, ev);
        let prefill_topk: Vec<_> = rank_chunks
            .iter()
            .map(|&(q, kv)| DsaTopkPrefillKernelInput { num_queries: q, num_keys: kv })
            .collect();
        push_fanin(&self.index_prefill_topk, &prefill_topk, DsaTopkPrefillKernelInput { num_queries: 0, num_keys: 0 }, ev);
        // Decode indexer logits at the requests' mean context.
        let mean_ctx = if decodes == 0 {
            0
        } else {
            let sum: u64 = input.decode_kv_lens.iter().map(|&k| u64::from(k)).sum();
            sum.div_ceil(u64::from(decodes)) as u32
        };
        push_or_zero(
            &self.index_decode_logits,
            DsaPagedMqaLogitsDecodeKernelInput { batch_size: decodes, context_len: mean_ctx },
            decodes == 0,
            ev,
        );
        // Decode top-k: one kernel over all rows, whose time is set by the
        // longest row, so it reads the batch's longest context. vLLM picks the
        // kernel by row count; the persistent kernel's path follows the host
        // `max_seq_len`, the batch maximum eagerly and the capture-time
        // max_model_len under a full graph (`vllm_cuda_graph`).
        let max_ctx = input.decode_kv_lens.iter().copied().max().unwrap_or(0);
        let topk = DsaPersistentTopkDecodeKernelInput { batch_size: decodes, context_len: max_ctx };
        let route = DecodeTopk::route(decode_rows, input.full_graph);
        push_or_zero(&self.index_decode_topk, topk.clone(), route != Some(DecodeTopk::Cooperative), ev);
        push_or_zero(
            &self.index_decode_topk_persistent,
            topk.clone(),
            route != Some(DecodeTopk::Persistent),
            ev,
        );
        push_or_zero(&self.index_decode_topk_graph, topk, route != Some(DecodeTopk::Graph), ev);
        push_or_zero(&self.mla_cache_append, MlaCacheAppendKernelInput { num_tokens: m }, z, ev);
        push_or_zero(&self.q_absorb, BatchedGemmKernelInput { m }, z, ev);
        push_or_zero(&self.q_quant, StaticFp8QuantKernelInput { num_tokens: m }, z, ev);
        push_or_zero(&self.index_remap, ElementwiseKernelInput { num_tokens: m }, z, ev);
        push_or_zero(
            &self.sparse_prefill,
            DsaSparseMlaPrefillKernelInput { query_context_pairs: input.prefill_chunks.clone() },
            prefill_tokens == 0 || shard_tp.is_some(),
            ev,
        );
        if let Some(sq) = &self.shardq {
            let off = shard_tp.is_none();
            push_or_zero(&sq.shard_q, SparseMlaShardQKernelInput { num_tokens: prefill_tokens }, off, ev);
            if let Some(pad) = &sq.pad_heads {
                let rank_tokens: u32 = rank_chunks.iter().map(|&(q, _)| q).sum();
                push_or_zero(pad, ElementwiseKernelInput { num_tokens: rank_tokens }, off, ev);
            }
            push_or_zero(
                &sq.sparse_prefill_all_heads,
                DsaSparseMlaPrefillKernelInput { query_context_pairs: rank_chunks.clone() },
                off,
                ev,
            );
            push_or_zero(&sq.unshard_out, SparseMlaShardQKernelInput { num_tokens: prefill_tokens }, off, ev);
        }
        let total_ctx: u64 = input.decode_kv_lens.iter().map(|&k| u64::from(k)).sum();
        push_or_zero(
            &self.sparse_decode,
            DsaSparseMlaAttentionKernelInput {
                num_queries: decode_rows,
                num_cache_tokens: u32::try_from(total_ctx).unwrap_or(u32::MAX),
            },
            decodes == 0,
            ev,
        );
        push_or_zero(&self.v_up, BatchedGemmKernelInput { m }, z, ev);
        push_or_zero(&self.g_proj, rows(m), z, ev);
        push_or_zero(&self.gate_mul, ElementwiseKernelInput { num_tokens: m }, z, ev);
        push_or_zero(&self.o_proj, rows(m), z, ev);
    }
}

// ---------------------------------------------------------------------------
// SiTU-GLU MLP

#[derive(Clone, Debug)]
pub struct KimiMlpLocalConfig {
    pub gpu_name: String,
    pub hidden: u32,
    /// Intermediate width on this rank.
    pub intermediate: u32,
    /// `Bf16` or `Mxfp8E4m3`.
    pub weight_dtype: DType,
    pub backends: KimiK31Backends,
}

pub struct KimiMlpLocal {
    name: String,
    gate_up: Op<SingleGemmKernel>,
    act: Op<SiluAndMulKernel>,
    down: Op<SingleGemmKernel>,
}

impl KimiMlpLocal {
    pub fn build(name: String, cfg: &KimiMlpLocalConfig, bridge: &PerfApiBridge) -> Result<Self, BuildError> {
        let b = &cfg.backends;
        let g = cfg.gpu_name.as_str();
        let n = name.as_str();
        let gemm_backends = if cfg.weight_dtype == DType::Mxfp8E4m3 { &b.mxfp8_gemm } else { &b.bf16_gemm };
        Ok(Self {
            gate_up: atomic(n, "gate_up", gemm(gemm_backends, g, 2 * cfg.intermediate, cfg.hidden, cfg.weight_dtype), SingleGemmKernel::build, bridge)?,
            act: atomic(
                n,
                "situ_and_mul",
                SiluAndMulKernelConfig {
                    backends: b.act.clone(),
                    gpu_name: cfg.gpu_name.clone(),
                    hidden_dim: cfg.intermediate.into(),
                    dtype: DType::Bf16,
                },
                SiluAndMulKernel::build,
                bridge,
            )?,
            down: atomic(n, "down", gemm(gemm_backends, g, cfg.hidden, cfg.intermediate, cfg.weight_dtype), SingleGemmKernel::build, bridge)?,
            name,
        })
    }

    pub fn compile(&self, builder: &mut CostTreeBuilder) -> CostNode {
        CostNode::Labeled {
            label: format!("{} (KimiMlpLocal)", self.name),
            child: Box::new(CostNode::Sum(vec![
                self.gate_up.compile(builder),
                self.act.compile(builder),
                self.down.compile(builder),
            ])),
        }
    }

    pub fn eval(&self, num_tokens: u32, ev: &mut Evaluator) {
        let z = num_tokens == 0;
        push_or_zero(&self.gate_up, rows(num_tokens), z, ev);
        push_or_zero(&self.act, SiluAndMulKernelInput { num_tokens }, z, ev);
        push_or_zero(&self.down, rows(num_tokens), z, ev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_topk_routes_by_rows_then_graph_mode() {
        assert_eq!(DecodeTopk::route(0, true), None);
        assert_eq!(DecodeTopk::route(0, false), None);
        // 8 requests x 8 verify rows is the cooperative ceiling.
        for graph in [false, true] {
            assert_eq!(DecodeTopk::route(8, graph), Some(DecodeTopk::Cooperative));
            assert_eq!(DecodeTopk::route(64, graph), Some(DecodeTopk::Cooperative));
        }
        assert_eq!(DecodeTopk::route(65, false), Some(DecodeTopk::Persistent));
        assert_eq!(DecodeTopk::route(72, true), Some(DecodeTopk::Graph));
        assert_eq!(DecodeTopk::route(256, true), Some(DecodeTopk::Graph));
    }
}
