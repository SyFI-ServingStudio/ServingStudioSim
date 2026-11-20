//! Arch (L4) config surface — the model-arch *selectors*, co-located with the L4
//! implementations they pick (L4 design).
//!
//! Each selector is a serde tagged enum (`#[serde(tag = "type")]`): choosing the
//! tag is the only way that variant's params appear — no global union,
//! provider-first. `model_config` + dims belong to the arch (the layer that
//! consumes them), so every arch variant flattens [`ModelSpec`]. arch and worker
//! are symmetric sibling providers (not nested).
//!
//! Iter-wise Llama selectors, three Qwen3-MoE execution graphs (native BF16,
//! native FP8, vLLM-aligned FP8), and two GLM-5.2 DSA/MoE graphs (native and
//! vLLM-aligned) build today. Each has a distinct selector and L4
//! implementation; precision/backend selection is therefore not hidden behind an
//! in-arch recipe branch, and a `_vllm_` variant differs from its sibling only in
//! how finely it cuts the same work into leaves, so a measured nsys timeline can
//! be reconciled leaf by leaf. They pair with the same `hp_unified` worker as the
//! DP-attn dense variant (one KV partition state per attn-DP shard).
//!
//! NOTE (serde): `#[serde(deny_unknown_fields)]` is silently ignored on
//! internally-tagged enum variants, so a typo inside an arch payload is NOT
//! caught here — the launcher's schema walk is the authoritative typo guard.
//!
//! Launcher schema is *derived*: `#[derive(ParamStruct)]` on [`ModelSpec`] emits
//! its `PARAMS` (the model fields every arch tag carries), and
//! `#[derive(ProviderSchema)]` on each selector emits a `SCHEMA` of
//! `(tag, params)` rows; `schema::dump::list_params` aggregates them. Defaults /
//! cache-key flags / descriptions live once, on the fields themselves.

use serde::Deserialize;

use schema_derive::{ParamStruct, ProviderSchema};

use super::glm52_model_cfg::Glm52MtpMode;
use super::qwen4_exp_model_cfg::{PleTableResidency, Qwen4ExpCacheDtype};

/// Model identity + layer controls. Flattened into every arch tag (§4), so it
/// carries no `deny_unknown_fields` (the flattened struct must let the arch's
/// own sharding fields through). `model_config` / `fp8` are cache-key (model
/// identity / dtype change which kernels are needed); `num_layers` /
/// `sim_num_layers` change layer COUNT, not per-layer shape, so they are not.
#[derive(Debug, Clone, Deserialize, ParamStruct)]
pub struct ModelSpec {
    /// Path to the model config JSON (or a known model name).
    #[param(cache_key)]
    pub model_config: String,
    /// Number of transformer layers (omit to use the model config's value).
    #[serde(default)]
    pub num_layers: Option<u32>,
    /// Simulate only this many layers with scaled timing (omit = all layers).
    #[serde(default)]
    pub sim_num_layers: Option<u32>,
    /// Use FP8 precision (DeepGEMM / fp8 prefill, halved transfers).
    #[param(cache_key)]
    pub fp8: bool,
}

impl ModelSpec {
    /// The checkpoint's `max_position_embeddings`, the longest request an arch
    /// without its own `max_model_len` serves (see
    /// [`model_cfg::max_position_embeddings`](super::model_cfg::max_position_embeddings)).
    pub fn max_position_embeddings(&self) -> anyhow::Result<u32> {
        super::model_cfg::max_position_embeddings(std::path::Path::new(&self.model_config))
    }
}

// ── iter-wise contract (unified, pd) ────────────────────────────────────────

/// Where a MoE arch's expert demand comes from: synthetic uniform/random
/// demand, or a measured distribution loaded from `expert_popularity_file`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingKind {
    /// Even load across all experts (the v1 default).
    #[default]
    Uniform,
    /// Deterministic pseudo-random skew seeded by `routing_seed`.
    Random,
    /// Measured per-expert marginal; requires `expert_popularity_file`.
    Popularity,
    /// Recorded per-token expert routes; requires `token_corpus_file`. Unlike a
    /// marginal it keeps whole tokens, so a batch's verify blocks can be
    /// sampled as the contiguous runs they are.
    Corpus,
}

/// The `routing` choices the launcher schema advertises (mirror of [`RoutingKind`]).
const ROUTING_KINDS: [&str; 4] = ["uniform", "random", "popularity", "corpus"];

// Only the GLM-5.2 NVFP4 and DeepSeek-V4.1 archs read a token corpus so far;
// the rest advertise the marginal-backed kinds.
const MARGINAL_ROUTING_KINDS: [&str; 3] = ["uniform", "random", "popularity"];

// AFD FFN selectors do not yet accept popularity files.
const SYNTHETIC_ROUTING_KINDS: [&str; 2] = ["uniform", "random"];

/// GLM-5.2 proposer modes exposed by the iter selector. This mirrors
/// [`Glm52MtpMode`]'s serde representation.
const GLM52_MTP_MODES: [&str; 3] = ["off", "full_index", "index_share"];

const fn default_deepseek_v41_dp_size() -> u32 {
    4
}

fn default_decoder_swa_bounded_replay() -> bool {
    true
}

/// The `engine` choices of `deepseek_v41_vllm_dp_attn_ep_moe` (mirror of
/// [`DeepseekV41Engine`]).
const DEEPSEEK_V41_ENGINES: [&str; 2] = ["vllm", "deepinfer"];

/// Which engine's kernels `deepseek_v41_vllm_dp_attn_ep_moe` composes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeepseekV41Engine {
    #[default]
    Vllm,
    Deepinfer,
}

/// The DeepInfer options of `deepseek_v41_vllm_dp_attn_ep_moe`, `None` under
/// vLLM; DeepInfer-only fields set under vLLM are an error.
pub fn deepseek_v41_dp_engine(
    engine: DeepseekV41Engine,
    num_redundant_experts: u32,
    yoco_keep_tokens: u32,
    engram_fetch: crate::arch::deepseek_v41_engram_fetch::EngramFetch,
    force_load_balance: bool,
    prefill_chunk_tokens: u32,
    step: DeepInferStepCfg,
) -> anyhow::Result<Option<super::deepseek_v41_vllm_dp_attn_ep_moe::DeepInferEngine>> {
    let engram_fetch = engram_fetch.fetches();
    match engine {
        DeepseekV41Engine::Vllm => {
            anyhow::ensure!(
                num_redundant_experts == 0
                    && yoco_keep_tokens == 0
                    && !force_load_balance
                    && prefill_chunk_tokens == 0
                    && step.host_step_us == 0
                    && step.step_gap_us == 0
                    && !step.host_first_stage_only,
                "num_redundant_experts, yoco_keep_tokens, force_load_balance, \
                 prefill_chunk_tokens, deepinfer_host_step_us, deepinfer_step_gap_us and \
                 deepinfer_host_first_stage_only are DeepInfer options; set engine: deepinfer"
            );
            Ok(None)
        }
        DeepseekV41Engine::Deepinfer => Ok(Some(
            super::deepseek_v41_vllm_dp_attn_ep_moe::DeepInferEngine {
                num_redundant_experts,
                yoco_keep_tokens: (yoco_keep_tokens > 0).then_some(yoco_keep_tokens),
                engram_fetch,
                force_load_balance,
                prefill_chunk_tokens: (prefill_chunk_tokens > 0).then_some(prefill_chunk_tokens),
                host_step_us: step.host_step_us,
                step_gap_us: step.step_gap_us,
                host_first_stage_only: step.host_first_stage_only,
                max_num_seqs: step.max_num_seqs,
            },
        )),
    }
}

/// The DeepInfer per-step fields of the DP selector, passed through
/// [`deepseek_v41_dp_engine`].
#[derive(Clone, Copy, Debug)]
pub struct DeepInferStepCfg {
    pub host_step_us: u32,
    pub step_gap_us: u32,
    pub host_first_stage_only: bool,
    pub max_num_seqs: u32,
}

const fn default_deepinfer_max_num_seqs() -> u32 {
    256
}

const fn default_engram_cpu_threads() -> u32 {
    8
}

const fn default_deepseek_v41_max_num_batched_tokens() -> u32 {
    crate::arch::deepseek_v41_vllm::MAX_BATCHED_TOKENS
}

fn default_deepseek_v41_cudagraph_capture_sizes() -> Vec<u32> {
    crate::arch::deepseek_v41_vllm::vllm_default_cudagraph_capture_sizes(
        crate::arch::deepseek_v41_vllm::MAX_BATCHED_TOKENS,
    )
}

const fn default_glm52_parallel_size() -> u16 {
    8
}

const fn default_glm52_nvfp4_parallel_size() -> u16 {
    4
}

const fn default_glm52_nvfp4_max_model_len() -> u32 {
    1_048_576
}

/// The captured GLM-5.3-Flash deployment's `--max-model-len`.
const fn default_glm53_flash_max_model_len() -> u32 {
    8192
}

/// The MiMo-V2.6-Flash capture's `--max-model-len` (its attention rows'
/// FA4 split count).
const fn default_mimo_v2_max_model_len() -> u32 {
    131_072
}

/// The Qwen3.8-27B capture's `--max-model-len`.
const fn default_qwen35_max_model_len() -> u32 {
    131_072
}

/// The Qwen3.8-Flash-Next capture's `--max-model-len`.
const fn default_qwen4_exp_max_model_len() -> u32 {
    131_072
}

/// The Qwen3.8-Flash-Next capture's MTP `num_speculative_tokens`.
const fn default_qwen4_exp_draft_tokens() -> u32 {
    3
}

/// [`Qwen4ExpCacheDtype`]'s serde representation.
const QWEN4_EXP_CACHE_DTYPES: [&str; 2] = Qwen4ExpCacheDtype::CHOICES;
/// [`PleTableResidency`]'s serde representation.
const PLE_TABLE_RESIDENCIES: [&str; 2] = PleTableResidency::CHOICES;

/// The captured GLM-5.3-Flash deployment ran `--enable-expert-parallel`.
const fn default_glm53_flash_enable_expert_parallel() -> bool {
    true
}

const fn default_glm53_flash_mla_layout_copies() -> bool {
    true
}

/// vLLM's default `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (`vllm/envs.py`).
const fn default_glm53_flash_indexer_max_logits_mb() -> u32 {
    512
}

/// A speculative GLM must actually run its MTP layer, so unlike the ordinary
/// selector it cannot default to `off`.
const fn default_glm52_speculative_mtp_mode() -> Glm52MtpMode {
    Glm52MtpMode::IndexShare
}

/// Default to the measured Spec5 workload; depth remains an execution choice.
const fn default_glm52_draft_tokens() -> u32 {
    5
}

/// The DFlash2 checkpoint's own block size is 8: one bonus token plus seven
/// drafted positions.
const fn default_dflash2_draft_tokens() -> u32 {
    7
}

/// DeepSeek-V4.1-Flash's `dspark_block_size`.
const fn default_dspark_draft_tokens() -> u32 {
    5
}

/// `sliding_window` in the DFlash2 draft checkpoint.
const fn default_dflash2_sliding_window() -> u32 {
    2048
}

/// KDA recurrent state pool dtype (vLLM `--mamba-ssm-cache-dtype`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KdaStateDtype {
    #[default]
    Fp32,
    Bf16,
}

impl KdaStateDtype {
    pub fn dtype(self) -> crate::timing::bridge::DType {
        match self {
            Self::Fp32 => crate::timing::bridge::DType::Fp32,
            Self::Bf16 => crate::timing::bridge::DType::Bf16,
        }
    }
}

/// [`KdaStateDtype`]'s serde representation.
const KDA_STATE_DTYPES: [&str; 2] = ["fp32", "bf16"];
/// [`crate::worklet::KdaSpecVerify`]'s serde representation.
const KDA_SPEC_VERIFIES: [&str; 2] = ["recoverssm", "recurrent_per_position"];

/// How the DSA layers' elementwise glue is priced (`dsa_glue`; see
/// [`crate::worklet::DsaGlue`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DsaGluePricing {
    /// 17 launches of 64 B per token.
    #[default]
    Placeholder,
    /// The vLLM fork's 7 decode launches at their real bytes.
    ForkLaunchList,
}

const DSA_GLUE_PRICINGS: [&str; 2] = ["placeholder", "fork_launch_list"];

/// DSA front cut A (`dsa_hidden_gemm`): the GEMMs reading the hidden state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DsaHiddenGemmMode {
    /// The fork's separate projections.
    #[default]
    Separate,
    /// [`crate::worklet::Glm53DsaFront::hidden_split_state_writer`], priced by
    /// `dsa_hidden_gemm_backend`.
    Fp32SplitStateWriter,
}

const DSA_HIDDEN_GEMM_MODES: [&str; 2] = ["separate", "fp32_split_state_writer"];

/// DSA front cut B (`dsa_q_lora_gemm`): the GEMMs reading the normed q latent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DsaQLoraGemmMode {
    /// The fork's separate `q_b_proj` and indexer `wq_b`.
    #[default]
    Separate,
    /// [`crate::worklet::Glm53DsaFront::q_lora_hadamard_epilogue`], priced by
    /// `dsa_q_lora_gemm_backend`.
    ConcatHadamardEpilogue,
}

const DSA_Q_LORA_GEMM_MODES: [&str; 2] = ["separate", "concat_hadamard_epilogue"];

const EXPERT_EXCHANGES: [&str; 2] =
    crate::arch::glm53_flash_vllm_fp8_dp_attn_ep_moe::EXPERT_EXCHANGES;
const MOE_SHARED_OVERLAPS: [&str; 2] =
    crate::arch::glm53_flash_vllm_fp8_dp_attn_ep_moe::MOE_SHARED_OVERLAPS;
const DRAFT_ATTENTIONS: [&str; 3] = crate::arch::glm53_vllm_nvfp4_dsa_moe_dflash2::DRAFT_ATTENTIONS;
const DRAFT_TOPKS: [&str; 2] = crate::arch::glm53_vllm_nvfp4_dsa_moe_dflash2::DRAFT_TOPKS;

const fn default_true() -> bool {
    true
}

const fn default_kda_tp_gpus() -> u16 {
    1
}

/// The KDA-TP design's one-node EP group.
const fn default_kda_tp_fused_ep_size() -> u16 {
    8
}

/// The vLLM fork's GLM-5.3-Flash speculative verify.
const fn default_recurrent_per_position() -> crate::worklet::KdaSpecVerify {
    crate::worklet::KdaSpecVerify::RecurrentPerPosition
}

/// The fork's DSA decode glue launches.
const fn default_fork_launch_list() -> DsaGluePricing {
    DsaGluePricing::ForkLaunchList
}

/// The measured shared-expert exposure.
const fn default_moe_shared_exposed(
) -> crate::arch::glm53_flash_vllm_fp8_dp_attn_ep_moe::MoeSharedOverlap {
    crate::arch::glm53_flash_vllm_fp8_dp_attn_ep_moe::MoeSharedOverlap::Exposed
}

/// The engine's DFlash2 draft attention call.
const fn default_fa4_fp8_split32() -> crate::arch::glm53_vllm_nvfp4_dsa_moe_dflash2::DraftAttention
{
    crate::arch::glm53_vllm_nvfp4_dsa_moe_dflash2::DraftAttention::Fa4Fp8Split32
}

/// Iteration-wise arch provider. Sharding parameters live only on the variants
/// that consume them (provider-first: select the arch, then it exposes its own
/// params).
#[derive(Debug, Clone, Deserialize, ProviderSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IterArchSel {
    /// Exact text-only Qwen3.6-35B-A3B FP8 execution graph on one local H200.
    /// TP1/EP1 are architectural invariants, so this selector exposes no
    /// sharding parameters — but routing is not a sharding parameter. Every
    /// expert lives on the one GPU, which removes dispatch/combine traffic, not
    /// the grouped GEMM's dependence on how many tokens each expert draws: skew
    /// leaves the total token-expert selections unchanged while redistributing
    /// them into fuller and emptier groups.
    Qwen36Local {
        #[serde(flatten)]
        model: ModelSpec,
        #[serde(default)]
        #[param(string, default = "uniform", choices = MARGINAL_ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
    },
    Llama3Dense {
        #[serde(flatten)]
        model: ModelSpec,
    },
    /// Megatron TP over dense Llama 3. `tp_size` must divide the KV heads (8).
    Llama3DenseTp {
        #[serde(flatten)]
        model: ModelSpec,
        /// Tensor parallelism size.
        #[param(default = 2, cache_key)]
        tp_size: u16,
    },
    Llama3DpAttnTpFfn {
        #[serde(flatten)]
        model: ModelSpec,
        /// Attention tensor-parallelism size (heads sharded across these ranks).
        /// DP groups = `ffn_tp_size / attn_tp_size`.
        #[param(default = 4, cache_key)]
        attn_tp_size: u16,
        /// FFN tensor-parallelism size (hidden/intermediate sharded; spans the
        /// whole replica).
        #[param(default = 8, cache_key)]
        ffn_tp_size: u16,
    },
    Qwen3MoeDpAttnEpFfn {
        #[serde(flatten)]
        model: ModelSpec,
        /// Attention tensor-parallelism size (heads sharded across these ranks).
        /// `num_dp_groups = ep_size / attn_tp_size`.
        #[param(default = 4, cache_key)]
        attn_tp_size: u16,
        /// Expert-parallelism size (FFN expert ranks; spans the whole replica).
        #[param(default = 8, cache_key)]
        ep_size: u16,
        /// `ReplicatedHeadParallel` residing-group size on the FFN side
        /// (drives whether the L2 MoE combine bcast/fanout legs are non-zero).
        #[param(default = 1, cache_key)]
        hp_size: u16,
        /// NVLink-domain size — partitions `ep_size` ranks into NVL domains
        /// for the intra/inter split of MoE dispatch/combine.
        #[param(default = 8, cache_key)]
        nvl_num_gpu: u16,
        /// Expert routing distribution: `uniform` (default) spreads load evenly;
        /// `random` draws a deterministic pseudo-random skew seeded by
        /// `routing_seed`; `popularity` loads `expert_popularity_file`.
        /// Drives the L2 MoE dispatch/combine `BottleneckCurve`
        /// and the L3 grouped-GEMM `local_ppm` shards. Omitted → `uniform`.
        #[serde(default)]
        #[param(string, default = "uniform", choices = MARGINAL_ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        /// Seed for `routing = random` (ignored for `uniform`). Fixed so a run is
        /// reproducible (the throughput golden is bit-identical); vary it to
        /// sample a different random skew.
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        /// Optional measured expert-popularity JSON. Its all-layer expert
        /// totals replace the synthetic routing distribution and are split
        /// into one raw `local_ppm` shard per EP rank. The homogeneous iter-wise
        /// arch intentionally uses one aggregate snapshot for every layer.
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
    },
    /// Native FP8 Qwen recipe: per-token/group quantized dense projections,
    /// DeepGEMM expert kernels, and ServingStudioSim's P2P EP dispatch/combine graph.
    Qwen3MoeFp8DpAttnEpFfn {
        #[serde(flatten)]
        model: ModelSpec,
        #[param(default = 4, cache_key)]
        attn_tp_size: u16,
        #[param(default = 8, cache_key)]
        ep_size: u16,
        #[param(default = 1, cache_key)]
        hp_size: u16,
        #[param(default = 8, cache_key)]
        nvl_num_gpu: u16,
        #[serde(default)]
        #[param(string, default = "uniform", choices = MARGINAL_ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
    },
    /// Alignment-only Qwen recipe matching the target vLLM execution path:
    /// FlashInfer/TensorRT-LLM block-scale expert GEMMs plus local finalize and
    /// an EP all-reduce. The generic Qwen tag above intentionally retains
    /// ServingStudioSim's original DeepGEMM + P2P communication recipe.
    Qwen3VllmMoeDpAttnEpFfn {
        #[serde(flatten)]
        model: ModelSpec,
        #[param(default = 4, cache_key)]
        attn_tp_size: u16,
        #[param(default = 8, cache_key)]
        ep_size: u16,
        #[param(default = 1, cache_key)]
        hp_size: u16,
        #[param(default = 8, cache_key)]
        nvl_num_gpu: u16,
        #[serde(default)]
        #[param(string, default = "uniform", choices = MARGINAL_ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
    },
    /// DeepSeek-V4-Flash-0731 at vLLM's physical kernel boundaries. EP4 and
    /// local attention DP4 are checkpoint/deployment invariants.
    DeepseekV4Vllm {
        #[serde(flatten)]
        model: ModelSpec,
        #[serde(default)]
        #[param(string, default = "uniform", choices = MARGINAL_ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
    },
    /// The same physical DeepSeek kernels with each source-level parallel
    /// region serialized. This is an explicit alignment counterfactual, not a
    /// hidden runtime knob on the production selector.
    DeepseekV4VllmSerialStreams {
        #[serde(flatten)]
        model: ModelSpec,
        #[serde(default)]
        #[param(string, default = "uniform", choices = MARGINAL_ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
    },
    /// DeepSeek-V4.1-Flash at vLLM's physical kernel boundaries on B200: TP4
    /// attention and EP4 experts over the same four ranks (deployment
    /// invariants). Routed demand normally comes from the capture's token
    /// corpus (`routing = corpus`), the demand its MoE rows were profiled on.
    DeepseekV41Vllm {
        #[serde(flatten)]
        model: ModelSpec,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as a
        /// manifest path. Requires `routing = corpus`. All 40 routed layers
        /// share one binding over the corpus's layer axis.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--max-model-len`: the longest request, and the extent the
        /// FlashMLA prefill planner sizes its compressed-KV workspace for.
        /// Omitted, the checkpoint's `max_position_embeddings` (1048576), as
        /// vLLM defaults it; anything larger is refused.
        #[serde(default)]
        #[param(cache_key)]
        max_model_len: Option<u32>,
        /// vLLM's decoder SWA bounded replay (`swa_bounded_replay`, on by
        /// default): on eager steps and large PIECEWISE steps the layers from
        /// the last KV source on run only each request's last
        /// `sliding_window` rows. `false` is the counterfactual without it.
        #[serde(default = "default_decoder_swa_bounded_replay")]
        #[param(default = true, cache_key)]
        decoder_swa_bounded_replay: bool,
        /// vLLM `--max-num-batched-tokens` (the chunk size): the FlashMLA
        /// prefill workspace identity of the mega attention.
        #[serde(default = "default_deepseek_v41_max_num_batched_tokens")]
        #[param(default = 2048, cache_key)]
        max_num_batched_tokens: u32,
        /// vLLM `--cudagraph-capture-sizes`: the PIECEWISE token sizes (the
        /// last one is the capture limit); the FULL decode graphs' request
        /// counts derive from them and `max_num_seqs` 64 as vLLM's MRV2
        /// runner does. Omitted: vLLM's default list up to 2048. Empty: eager.
        #[serde(default = "default_deepseek_v41_cudagraph_capture_sizes")]
        cudagraph_capture_sizes: Vec<u32>,
    },
    /// DeepSeek-V4.1-Flash under vLLM DP4 attention + EP4 experts on B200
    /// (`--data-parallel-size 4 --enable-expert-parallel`, TP1): every rank
    /// its own engine with its own batch and KV; attention, Engram, shared
    /// expert and the full-vocabulary lm_head at TP1, the routed experts
    /// behind an MXFP8 all-gatherv and a bf16 reduce-scatterv.
    DeepseekV41VllmDpAttnEpMoe {
        #[serde(flatten)]
        model: ModelSpec,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as a
        /// manifest path. Requires `routing = corpus`.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--max-model-len`, as for `deepseek_v41_vllm`.
        #[serde(default)]
        #[param(cache_key)]
        max_model_len: Option<u32>,
        /// vLLM's decoder SWA bounded replay, as for `deepseek_v41_vllm`;
        /// the DP ranks replay together.
        #[serde(default = "default_decoder_swa_bounded_replay")]
        #[param(default = true, cache_key)]
        decoder_swa_bounded_replay: bool,
        /// vLLM `--max-num-batched-tokens` per rank, as for `deepseek_v41_vllm`.
        #[serde(default = "default_deepseek_v41_max_num_batched_tokens")]
        #[param(default = 2048, cache_key)]
        max_num_batched_tokens: u32,
        /// vLLM `--cudagraph-capture-sizes` per rank, as for
        /// `deepseek_v41_vllm`. Omitted: vLLM's default list up to 2048.
        /// Empty: eager.
        #[serde(default = "default_deepseek_v41_cudagraph_capture_sizes")]
        cudagraph_capture_sizes: Vec<u32>,
        /// Serving engine whose kernels run the layers: `vllm` (the fork) or
        /// `deepinfer` (DeepGEMM mega MoE, DeepGEMM MXFP8 linears, YoCo
        /// prefill truncation; `DeepInferEngine`).
        #[serde(default)]
        #[param(string, default = "vllm", choices = DEEPSEEK_V41_ENGINES, cache_key)]
        engine: DeepseekV41Engine,
        /// DeepInfer only: EPLB redundant expert slots (`num_extra_experts`).
        #[serde(default)]
        #[param(default = 0, cache_key)]
        num_redundant_experts: u32,
        /// DeepInfer only: YoCo prefill truncation keeps this many rows per
        /// request past the cut layer (`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS`);
        /// 0 = off. `decoder_swa_bounded_replay` is vLLM's and is ignored.
        #[serde(default)]
        #[param(default = 0, cache_key)]
        yoco_keep_tokens: u32,
        /// How Engram rows reach the GPU: `uva` (vLLM's host-UVA lookup),
        /// `cpu_threads` (host gather on `engram_cpu_threads` threads per rank
        /// + one H2D copy per layer + device unpack;
        /// `arch::deepseek_v41_engram_fetch`), `prefetch` (the same, started
        /// during the previous step), `gpu` (the table in HBM, device lookup; a
        /// bound only, it OOMs on B200), or `off` (the fetch priced out, the
        /// Engram blocks' compute stays). `cpu_gather` / `cpu_gather_prefetch`
        /// are the earlier names; `true`/`false` read as uva/off.
        #[serde(default)]
        #[param(string, default = "uva", choices = crate::arch::deepseek_v41_engram_fetch::ENGRAM_FETCH_CHOICES, cache_key)]
        engram_fetch: crate::arch::deepseek_v41_engram_fetch::EngramFetch,
        /// Host gather threads per rank under `engram_fetch: cpu_threads` /
        /// `prefetch`; priced at the nearest measured count (1, 2, 4, 8, 14;
        /// `arch::deepseek_v41_engram_fetch`).
        #[serde(default = "default_engram_cpu_threads")]
        #[param(default = 8, cache_key)]
        engram_cpu_threads: u32,
        /// DeepInfer only: host time per engine step with the GPU idle, in
        /// microseconds (scheduling, input metadata, post-process between two
        /// prefill forwards). Measured, not tuned: set it from a capture and
        /// cite the capture in the preset. 0 = not modeled.
        #[serde(default)]
        #[param(default = 0, cache_key)]
        deepinfer_host_step_us: u32,
        /// DeepInfer only: GPU idle inside one engine step, in microseconds:
        /// the sum of the short (< 1 ms) gaps between the kernels of one eager
        /// prefill forward. Measured, not tuned: set it from a capture and cite
        /// the capture in the preset. 0 = not modeled.
        #[serde(default)]
        #[param(default = 0, cache_key)]
        deepinfer_step_gap_us: u32,
        /// DeepInfer only: `max_num_seqs` (EngineArgs default 256). It sizes
        /// the per-sequence state slots DeepInfer reserves out of the KV budget
        /// (`cache_engine.py`), so it moves the token capacity.
        #[serde(default = "default_deepinfer_max_num_seqs")]
        #[param(default = 256, cache_key)]
        deepinfer_max_num_seqs: u32,
        /// DP ranks per replica, which are also its EP group (2, 4 or 8; the
        /// captured deployment is 4). A pool of `replicas` such groups places
        /// sessions by prefix affinity like the PP replicas.
        #[serde(default = "default_deepseek_v41_dp_size")]
        #[param(default = 4, cache_key)]
        dp_size: u32,
        /// DeepInfer only: its benchmark's `DEEPINFER_FORCE_LOAD_BALANCE=1`
        /// (`examples/benchmark.py`): every token takes top-k distinct slots
        /// uniformly at random instead of the router's experts. For
        /// reproducing DeepInfer's own benchmark numbers only.
        #[serde(default)]
        #[param(default = false, cache_key)]
        force_load_balance: bool,
        /// DeepInfer only, `routing: corpus`: the tokens one request
        /// contributes to a prefill batch (its prompt chunk). The mega MoE
        /// draws a batch's routes as contiguous corpus segments of this
        /// length, because one request's tokens route alike; 0 draws
        /// independent tokens.
        #[serde(default)]
        #[param(default = 0, cache_key)]
        prefill_chunk_tokens: u32,
    },
    /// DeepSeek-V4.1-Flash under plain pipeline parallelism on B200: each of
    /// `pp_size` stages is one GPU running a contiguous layer range at TP1 /
    /// EP1 (all heads and all 384 experts local, no collective). Any partition
    /// is modeled; a cut inside a KV-sharing group keeps a copy of the
    /// upstream KV source on the reading stage and forwards its new entries,
    /// and bounded replay spans the stages after layer 20 (see
    /// `arch/deepseek_v41_vllm_pp.rs`). Runs only under deployment `pp`.
    DeepseekV41VllmPp {
        #[serde(flatten)]
        model: ModelSpec,
        /// Pipeline stages, one GPU each.
        #[param(cache_key)]
        pp_size: u16,
        /// vLLM `VLLM_PP_LAYER_PARTITION`: layers per stage, `pp_size`
        /// positive counts summing to 40. Empty: vLLM's `get_pp_indices`.
        #[serde(default)]
        layer_partition: Vec<u32>,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as a
        /// manifest path. Requires `routing = corpus`. Routes are per token,
        /// so a corpus captured at any EP size folds to EP1.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--max-model-len`, as for `deepseek_v41_vllm`.
        #[serde(default)]
        #[param(cache_key)]
        max_model_len: Option<u32>,
        /// vLLM's decoder SWA bounded replay, as for `deepseek_v41_vllm`.
        #[serde(default = "default_decoder_swa_bounded_replay")]
        #[param(default = true, cache_key)]
        decoder_swa_bounded_replay: bool,
        /// vLLM `--max-num-batched-tokens`, as for `deepseek_v41_vllm`.
        #[serde(default = "default_deepseek_v41_max_num_batched_tokens")]
        #[param(default = 2048, cache_key)]
        max_num_batched_tokens: u32,
        /// vLLM `--cudagraph-capture-sizes`, as for `deepseek_v41_vllm`.
        /// Omitted: vLLM's default list up to 2048. Empty: eager.
        #[serde(default = "default_deepseek_v41_cudagraph_capture_sizes")]
        cudagraph_capture_sizes: Vec<u32>,
        /// Per stage, its layer ranges as `start-end` (end exclusive), stages
        /// separated by `;` (a folded split: `0-10,20-30;10-20,30-40`).
        /// Overrides `layer_partition`; a fold needs a worker back pass
        /// (`back_flush`). Empty: `layer_partition`.
        #[serde(default)]
        #[param(string, cache_key)]
        stage_layers: String,
        /// How the replay cut layer's input reaches it: `v1` all rows' state,
        /// `v2` continuing rows' state + collapsed hidden, `v3` continuing
        /// rows' state + the cut source's KV entries (`write_kv` before the
        /// hop).
        #[serde(default)]
        #[param(string, default = "v1", choices = ["v1", "v2", "v3"], cache_key)]
        cut_handoff: crate::arch::deepseek_v41_vllm_pp::CutHandoff,
        /// Serving engine whose kernels run each stage, as for
        /// `deepseek_v41_vllm_dp_attn_ep_moe` (`deepinfer`: DeepGEMM mega MoE
        /// over the GPU's own 384 experts, DeepGEMM MXFP8 linears, YoCo).
        #[serde(default)]
        #[param(string, default = "vllm", choices = DEEPSEEK_V41_ENGINES, cache_key)]
        engine: DeepseekV41Engine,
        /// DeepInfer only: YoCo keep rows per request past the cut; 0 = off.
        #[serde(default)]
        #[param(default = 0, cache_key)]
        yoco_keep_tokens: u32,
        /// How Engram rows reach the GPU, as for `deepseek_v41_vllm_dp_attn_ep_moe`.
        #[serde(default)]
        #[param(string, default = "uva", choices = crate::arch::deepseek_v41_engram_fetch::ENGRAM_FETCH_CHOICES, cache_key)]
        engram_fetch: crate::arch::deepseek_v41_engram_fetch::EngramFetch,
        /// Host gather threads per rank under `engram_fetch: cpu_threads` /
        /// `prefetch`; priced at the nearest measured count (1, 2, 4, 8, 14;
        /// `arch::deepseek_v41_engram_fetch`).
        #[serde(default = "default_engram_cpu_threads")]
        #[param(default = 8, cache_key)]
        engram_cpu_threads: u32,
        /// DeepInfer only: host time per engine step with the GPU idle, in
        /// microseconds (scheduling, input metadata, post-process between two
        /// prefill forwards). Measured, not tuned: set it from a capture and
        /// cite the capture in the preset. 0 = not modeled.
        #[serde(default)]
        #[param(default = 0, cache_key)]
        deepinfer_host_step_us: u32,
        /// DeepInfer only: GPU idle inside one engine step, in microseconds:
        /// the sum of the short (< 1 ms) gaps between the kernels of one eager
        /// prefill forward. Measured, not tuned: set it from a capture and cite
        /// the capture in the preset. 0 = not modeled.
        #[serde(default)]
        #[param(default = 0, cache_key)]
        deepinfer_step_gap_us: u32,
        /// DeepInfer only: `max_num_seqs` (EngineArgs default 256). It sizes
        /// the per-sequence state slots DeepInfer reserves out of the KV budget
        /// (`cache_engine.py`), so it moves the token capacity.
        #[serde(default = "default_deepinfer_max_num_seqs")]
        #[param(default = 256, cache_key)]
        deepinfer_max_num_seqs: u32,
        /// DeepInfer only: price the host step and the in-step gaps on the
        /// first stage's step alone (one per pipeline round: the lower bound)
        /// instead of on every stage's step (false: the upper bound). DeepInfer
        /// has no PP, so neither is measured.
        #[serde(default)]
        #[param(default = false, cache_key)]
        deepinfer_host_first_stage_only: bool,
    },
    /// The same DeepSeek-V4.1 kernels with the gated side streams (stage-A
    /// input projections, the compressor aux stream, the shared expert)
    /// serialized; the Engram lookups keep racing the main path. An explicit
    /// alignment counterfactual, not a runtime knob on the production selector.
    DeepseekV41VllmSerialStreams {
        #[serde(flatten)]
        model: ModelSpec,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--max-model-len`: the longest request, and the extent the
        /// FlashMLA prefill planner sizes its compressed-KV workspace for.
        /// Omitted, the checkpoint's `max_position_embeddings` (1048576), as
        /// vLLM defaults it; anything larger is refused.
        #[serde(default)]
        #[param(cache_key)]
        max_model_len: Option<u32>,
        /// vLLM's decoder SWA bounded replay (`swa_bounded_replay`, on by
        /// default): on eager steps and large PIECEWISE steps the layers from
        /// the last KV source on run only each request's last
        /// `sliding_window` rows. `false` is the counterfactual without it.
        #[serde(default = "default_decoder_swa_bounded_replay")]
        #[param(default = true, cache_key)]
        decoder_swa_bounded_replay: bool,
        /// vLLM `--max-num-batched-tokens` (the chunk size): the FlashMLA
        /// prefill workspace identity of the mega attention.
        #[serde(default = "default_deepseek_v41_max_num_batched_tokens")]
        #[param(default = 2048, cache_key)]
        max_num_batched_tokens: u32,
        /// vLLM `--cudagraph-capture-sizes`: the PIECEWISE token sizes (the
        /// last one is the capture limit); the FULL decode graphs' request
        /// counts derive from them and `max_num_seqs` 64 as vLLM's MRV2
        /// runner does. Omitted: vLLM's default list up to 2048. Empty: eager.
        #[serde(default = "default_deepseek_v41_cudagraph_capture_sizes")]
        cudagraph_capture_sizes: Vec<u32>,
    },
    /// The DeepSeek-V4.1 target driven by its **DSpark** proposer
    /// (`--speculative-config {"method":"dspark"}`): three SWA-only draft
    /// layers drafting a `draft_tokens` block in one pass, then that many
    /// sequential Markov steps. TP4 attention + EP4 experts as the target.
    DeepseekV41VllmDspark {
        #[serde(flatten)]
        model: ModelSpec,
        /// Target routing; the draft MoE routes uniformly (no capture).
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Target per-token routes (`routing = corpus`), recorded at the
        /// verify width when available.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--max-model-len`, as for `deepseek_v41_vllm`.
        #[serde(default)]
        #[param(cache_key)]
        max_model_len: Option<u32>,
        /// Positions drafted per request: must equal `dspark_block_size`.
        #[serde(default = "default_dspark_draft_tokens")]
        #[param(default = 5, cache_key)]
        draft_tokens: u32,
        /// Decoder SWA bounded replay of the target, as for `deepseek_v41_vllm`.
        #[serde(default = "default_decoder_swa_bounded_replay")]
        #[param(default = true, cache_key)]
        decoder_swa_bounded_replay: bool,
        /// vLLM `--max-num-batched-tokens`, as for `deepseek_v41_vllm`.
        #[serde(default = "default_deepseek_v41_max_num_batched_tokens")]
        #[param(default = 2048, cache_key)]
        max_num_batched_tokens: u32,
        /// vLLM `--cudagraph-capture-sizes`, as for `deepseek_v41_vllm`; the
        /// FULL verify and draft graphs' request counts derive from it at the
        /// verify width. Omitted: vLLM's default list up to 2048. Empty: eager.
        #[serde(default = "default_deepseek_v41_cudagraph_capture_sizes")]
        cudagraph_capture_sizes: Vec<u32>,
    },
    /// [`DeepseekV41VllmDspark`] over the DP4 target
    /// (`deepseek_v41_vllm_dp_attn_ep_moe`): rank-local sampler and proposer
    /// edges, draft layers as DP layers with the EP-sharded draft MoE.
    /// The DeepSeek-V4.1 target driven by its **DSpark** proposer
    /// (`--speculative-config {"method":"dspark"}`): three SWA-only draft
    /// layers drafting a `draft_tokens` block in one pass, then that many
    /// sequential Markov steps. TP4 attention + EP4 experts as the target.
    DeepseekV41VllmDpAttnEpMoeDspark {
        #[serde(flatten)]
        model: ModelSpec,
        /// Target routing; the draft MoE routes uniformly (no capture).
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Target per-token routes (`routing = corpus`), recorded at the
        /// verify width when available.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--max-model-len`, as for `deepseek_v41_vllm`.
        #[serde(default)]
        #[param(cache_key)]
        max_model_len: Option<u32>,
        /// Positions drafted per request: must equal `dspark_block_size`.
        #[serde(default = "default_dspark_draft_tokens")]
        #[param(default = 5, cache_key)]
        draft_tokens: u32,
        /// Decoder SWA bounded replay of the target, as for `deepseek_v41_vllm`.
        #[serde(default = "default_decoder_swa_bounded_replay")]
        #[param(default = true, cache_key)]
        decoder_swa_bounded_replay: bool,
        /// vLLM `--max-num-batched-tokens`, as for `deepseek_v41_vllm`.
        #[serde(default = "default_deepseek_v41_max_num_batched_tokens")]
        #[param(default = 2048, cache_key)]
        max_num_batched_tokens: u32,
        /// vLLM `--cudagraph-capture-sizes`, as for `deepseek_v41_vllm`; the
        /// FULL verify and draft graphs' request counts derive from it at the
        /// verify width. Omitted: vLLM's default list up to 2048. Empty: eager.
        #[serde(default = "default_deepseek_v41_cudagraph_capture_sizes")]
        cudagraph_capture_sizes: Vec<u32>,
    },
    /// GLM-5.2's aligned vLLM execution graph with local TP1 attention and
    /// expert parallelism across the replica.
    Glm52VllmDsaMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Expert-parallel ranks and independent local-attention DP groups.
        #[serde(default = "default_glm52_parallel_size")]
        #[param(default = 8, cache_key)]
        ep_size: u16,
        /// NVLink-domain size for the MoE dispatch/combine split.
        #[serde(default = "default_glm52_parallel_size")]
        #[param(default = 8, cache_key)]
        nvl_num_gpu: u16,
        /// Expert routing distribution used by dispatch/combine.
        #[serde(default)]
        #[param(string, default = "uniform", choices = MARGINAL_ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        /// Seed for `routing = random`; ignored for uniform routing.
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        /// Optional MTP proposer work: off, full-index step 0, or a later
        /// IndexShare step.
        #[serde(default)]
        #[param(string, default = "off", choices = GLM52_MTP_MODES, cache_key)]
        mtp_mode: Glm52MtpMode,
        /// Measured per-expert popularity from a profile pass. Requires
        /// `routing = popularity`; uniform/random routing cannot carry a profile.
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
    },
    /// B200 execution graph for NVIDIA's GLM-5.2 NVFP4 checkpoint. Tensor and
    /// expert parallelism share one rank group, as observed in vLLM.
    Glm52VllmNvfp4DsaMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Shared tensor/expert-parallel rank count.
        #[serde(default = "default_glm52_nvfp4_parallel_size")]
        #[param(default = 4, cache_key)]
        ep_size: u16,
        #[serde(default = "default_glm52_nvfp4_parallel_size")]
        #[param(default = 4, cache_key)]
        nvl_num_gpu: u16,
        /// Configured context cap and padded DSA-logits row stride.
        #[serde(default = "default_glm52_nvfp4_max_model_len")]
        #[param(default = 1048576, cache_key)]
        max_model_len: u32,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(string, default = "off", choices = GLM52_MTP_MODES, cache_key)]
        mtp_mode: Glm52MtpMode,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as
        /// a manifest path. Requires `routing = corpus`. One artifact covers
        /// every routed layer, so the body MoE and the MTP MoE read slices
        /// of it rather than separate files.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
    },
    /// [`Self::Glm52VllmNvfp4DsaMoe`]'s kernels under data-parallel attention
    /// and expert-parallel MoE (vLLM `--data-parallel-size ep_size
    /// --enable-expert-parallel`, TP1). Every GPU is its own engine with its
    /// own batch and KV; attention, dense FFN, shared expert and lm_head run
    /// whole on each GPU's tokens, and only the routed experts are sharded,
    /// behind an NVFP4 all-gather and a bf16 reduce-scatter. MTP is not run.
    /// GLM-5.3 NVFP4 is the same graph.
    Glm52VllmNvfp4DpAttnDsaMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Data-parallel attention ranks, which are also the expert-parallel
        /// group: one GPU each.
        #[serde(default = "default_glm52_nvfp4_parallel_size")]
        #[param(default = 4, cache_key)]
        ep_size: u16,
        #[serde(default = "default_glm52_nvfp4_parallel_size")]
        #[param(default = 4, cache_key)]
        nvl_num_gpu: u16,
        /// Configured context cap and padded DSA-logits row stride.
        #[serde(default = "default_glm52_nvfp4_max_model_len")]
        #[param(default = 1048576, cache_key)]
        max_model_len: u32,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        /// Measured per-expert popularity captured at the same
        /// `expert_parallel_size`. Requires `routing = popularity`.
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as a
        /// manifest path. Requires `routing = corpus`. Only the body slice is
        /// read: no MTP layer runs.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM's CUDA-graph capture sizes. When the busiest DP rank's tokens
        /// fit a captured size, every rank pads to that graph and every
        /// kernel outside the attention graph break runs on the padded rows.
        /// Empty: every step runs eager, unpadded.
        #[serde(default)]
        cudagraph_capture_sizes: Vec<u32>,
    },
    /// [`Self::Glm52VllmNvfp4DsaMoe`]'s kernels under pure pipeline
    /// parallelism. Each of `pp_size` stages is one GPU running a contiguous
    /// layer range (vLLM's `get_pp_indices`) at EP1: every head and expert is
    /// local, so a stage has no collective. MTP is not run. GLM-5.3 NVFP4 is
    /// the same graph. Runs only under deployment `pp`.
    Glm52VllmNvfp4PpDsaMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Pipeline stages, one GPU each.
        #[param(cache_key)]
        pp_size: u16,
        /// Configured context cap and padded DSA-logits row stride.
        #[serde(default = "default_glm52_nvfp4_max_model_len")]
        #[param(default = 1048576, cache_key)]
        max_model_len: u32,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        /// Measured per-expert popularity captured at EP1 (one rank owns all
        /// 256 experts). Requires `routing = popularity`.
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as a
        /// manifest path. Requires `routing = corpus`. Only the body slice is
        /// read: a stage runs no MTP layer.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
    },
    /// [`Self::Glm52VllmNvfp4DsaMoe`] driving its MTP layer as a real drafter:
    /// one target verify pass over `draft_tokens + 1` rows per decode request,
    /// then `draft_tokens` draft passes.
    ///
    /// A separate selector because it builds a different model type, not the
    /// same one with speculation switched on — the two compile different cost
    /// trees, so nothing can read one's cost log as the other's. It pairs only
    /// with the `speculative` worker.
    Glm52VllmNvfp4DsaMoeSpeculative {
        #[serde(flatten)]
        model: ModelSpec,
        /// Shared tensor/expert-parallel rank count.
        #[serde(default = "default_glm52_nvfp4_parallel_size")]
        #[param(default = 4, cache_key)]
        ep_size: u16,
        #[serde(default = "default_glm52_nvfp4_parallel_size")]
        #[param(default = 4, cache_key)]
        nvl_num_gpu: u16,
        /// Configured context cap and padded DSA-logits row stride.
        #[serde(default = "default_glm52_nvfp4_max_model_len")]
        #[param(default = 1048576, cache_key)]
        max_model_len: u32,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        /// The draft layer's own execution mode. `off` is rejected here: a
        /// speculative arch with no MTP layer has nothing to draft with.
        #[serde(default = "default_glm52_speculative_mtp_mode")]
        #[param(string, default = "index_share", choices = GLM52_MTP_MODES, cache_key)]
        mtp_mode: Glm52MtpMode,
        /// Candidate positions drafted per request per iteration. Fixes the
        /// verify width at `draft_tokens + 1`, which selects a profiled kernel
        /// shape and so cannot vary per iteration.
        #[serde(default = "default_glm52_draft_tokens")]
        #[param(default = 5, cache_key)]
        draft_tokens: u32,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as
        /// a manifest path. Requires `routing = corpus`. One artifact covers
        /// every routed layer, so the body MoE and the MTP MoE read slices
        /// of it rather than separate files.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
    },
    /// The same GLM target graph driven by a **DFlash2** proposer instead of the
    /// MTP layer.
    ///
    /// The distinction that matters here is not the checkpoint but the drafting
    /// shape: DFlash2 proposes a whole block in one forward pass, so there is no
    /// recurrence to fold and the draft forwards `draft_tokens + 1` rows per
    /// request rather than one. It also carries its own dense checkpoint, so it
    /// routes nothing and has no `mtp_mode`.
    Glm53VllmNvfp4DsaMoeDflash2 {
        #[serde(flatten)]
        model: ModelSpec,
        /// Shared tensor/expert-parallel rank count.
        #[serde(default = "default_glm52_nvfp4_parallel_size")]
        #[param(default = 4, cache_key)]
        ep_size: u16,
        #[serde(default = "default_glm52_nvfp4_parallel_size")]
        #[param(default = 4, cache_key)]
        nvl_num_gpu: u16,
        /// Configured context cap and padded DSA-logits row stride.
        #[serde(default = "default_glm52_nvfp4_max_model_len")]
        #[param(default = 1048576, cache_key)]
        max_model_len: u32,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        /// Candidate positions drafted per request per iteration. Fixes the
        /// verify width at `draft_tokens + 1`, which selects a profiled kernel
        /// shape and so cannot vary per iteration.
        #[serde(default = "default_dflash2_draft_tokens")]
        #[param(default = 7, cache_key)]
        draft_tokens: u32,
        /// The draft's own attention window. It caps the context each draft
        /// query attends to, not the draft's KV: vLLM allocates those layers as
        /// full attention, so their cache still grows with every token.
        #[serde(default = "default_dflash2_sliding_window")]
        #[param(default = 2048, cache_key)]
        draft_sliding_window: u32,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token target expert routes from a `token_corpus` pass,
        /// as a manifest path. Requires `routing = corpus`. Only the body
        /// slice is read: this target runs no MTP layer.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
    },
    /// GLM-5.3-Flash FP8 block checkpoint on B200 through the vLLM fork: 34 KDA
    /// + 11 DSA (kpool indexer) layers, 3 dense + 42 MoE FFNs, 4-wide mHC. One
    /// tensor-parallel rank group whose routed experts are either expert- or
    /// tensor-parallel; MTP is not run.
    Glm53FlashVllmFp8KdaDsaMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Tensor-parallel rank count: attention heads, the dense FFN and the
        /// shared expert are split over it, and so are the routed experts.
        #[serde(default = "default_glm52_nvfp4_parallel_size")]
        #[param(default = 4, cache_key)]
        tp_size: u16,
        /// vLLM `--enable-expert-parallel`. On, each rank owns
        /// `n_routed_experts / tp_size` whole experts (EP = TP). Off, each rank
        /// owns every expert, sliced to `moe_intermediate_size / tp_size` on
        /// the intermediate axis (MoE TP).
        #[serde(default = "default_glm53_flash_enable_expert_parallel")]
        #[param(default = true, cache_key)]
        enable_expert_parallel: bool,
        /// Configured context cap: the indexer's logits row stride and the
        /// sparse-index page-table extent.
        #[serde(default = "default_glm53_flash_max_model_len")]
        #[param(default = 8192, cache_key)]
        max_model_len: u32,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as a
        /// manifest path. Requires `routing = corpus`.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--cudagraph-capture-sizes`. The engine pads an iteration's
        /// token count up to the next captured size, and every kernel outside
        /// the attention graph break runs on the padded rows. Empty: no
        /// padding (eager).
        #[serde(default)]
        cudagraph_capture_sizes: Vec<u32>,
        /// The two sparse-MLA layout copies of vLLM fork 3f667d7: the
        /// `q_concat` transpose copy of the absorbed query and the `masked_fill_`
        /// over the attention output. vLLM 04730e8 and later write the query
        /// token-major and fill only under DCP or HiSparse; false models that.
        #[serde(default = "default_glm53_flash_mla_layout_copies")]
        #[param(default = true, cache_key)]
        mla_layout_copies: bool,
        /// vLLM `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`: the prefill indexer splits
        /// its queries so each MQA-logits call's fp32 `[rows, pools]` matrix
        /// fits this many MiB, and runs logits and top-k once per piece.
        #[serde(default = "default_glm53_flash_indexer_max_logits_mb")]
        #[param(default = 512, cache_key)]
        indexer_max_logits_mb: u32,
    },
    /// [`Self::Glm53FlashVllmFp8KdaDsaMoe`]'s kernels under pure pipeline
    /// parallelism. Each of `pp_size` stages is one GPU running a contiguous
    /// layer range (vLLM's `get_pp_indices`) at TP1 / EP1: every head and
    /// expert is local, so a stage has no collective. Each stage caches only
    /// its own KDA (per request) and DSA (per token) layers; from PP12 a stage
    /// has KDA layers but no DSA layer, which vLLM's hybrid cache rejects. MTP
    /// is not run. Runs only under deployment `pp`.
    Glm53FlashVllmFp8PpKdaDsaMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Pipeline stages, one GPU each.
        #[param(cache_key)]
        pp_size: u16,
        /// Configured context cap: the indexer's logits row stride and the
        /// sparse-index page-table extent.
        #[serde(default = "default_glm53_flash_max_model_len")]
        #[param(default = 8192, cache_key)]
        max_model_len: u32,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        /// Measured per-expert popularity captured at EP1 (one rank owns all
        /// 288 experts). Requires `routing = popularity`.
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as a
        /// manifest path. Requires `routing = corpus`. Routes are per token, so
        /// a corpus captured at any EP size folds to EP1.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--cudagraph-capture-sizes`, as for the TP = EP graph. Empty:
        /// no padding (eager).
        #[serde(default)]
        cudagraph_capture_sizes: Vec<u32>,
        /// vLLM `VLLM_PP_LAYER_PARTITION`: layers per stage, `pp_size` counts
        /// summing to 45. Empty: vLLM's default `get_pp_indices` split.
        #[serde(default)]
        layer_partition: Vec<u32>,
        /// The two sparse-MLA layout copies of vLLM fork 3f667d7: the
        /// `q_concat` transpose copy of the absorbed query and the `masked_fill_`
        /// over the attention output. vLLM 04730e8 and later write the query
        /// token-major and fill only under DCP or HiSparse; false models that.
        #[serde(default = "default_glm53_flash_mla_layout_copies")]
        #[param(default = true, cache_key)]
        mla_layout_copies: bool,
        /// vLLM `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`: the prefill indexer splits
        /// its queries so each MQA-logits call's fp32 `[rows, pools]` matrix
        /// fits this many MiB, and runs logits and top-k once per piece.
        #[serde(default = "default_glm53_flash_indexer_max_logits_mb")]
        #[param(default = 512, cache_key)]
        indexer_max_logits_mb: u32,
    },
    /// [`Self::Glm53FlashVllmFp8KdaDsaMoe`]'s kernels under data-parallel
    /// attention and expert-parallel MoE (vLLM `--data-parallel-size ep_size
    /// --enable-expert-parallel`, TP1). Every GPU is its own engine with its
    /// own batch, KV and KDA state; attention, dense FFN, shared expert and
    /// lm_head run whole on each GPU's tokens, and only the routed experts are
    /// sharded, behind an FP8 all-gather and a bf16 reduce-scatter.
    Glm53FlashVllmFp8DpAttnEpMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Data-parallel attention ranks, which are also the expert-parallel
        /// group: one GPU each.
        #[serde(default = "default_glm52_nvfp4_parallel_size")]
        #[param(default = 4, cache_key)]
        ep_size: u16,
        /// Configured context cap: the indexer's logits row stride and the
        /// sparse-index page-table extent.
        #[serde(default = "default_glm53_flash_max_model_len")]
        #[param(default = 8192, cache_key)]
        max_model_len: u32,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        /// Measured per-expert popularity captured at the same
        /// `expert_parallel_size`. Requires `routing = popularity`.
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as a
        /// manifest path. Requires `routing = corpus`.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--cudagraph-capture-sizes`. When every rank's batch fits a
        /// captured size, all ranks pad to the busiest rank's graph; otherwise
        /// they run eager on their own rows. Empty: always eager.
        #[serde(default)]
        cudagraph_capture_sizes: Vec<u32>,
        /// The two sparse-MLA layout copies of vLLM fork 3f667d7: the
        /// `q_concat` transpose copy of the absorbed query and the `masked_fill_`
        /// over the attention output. vLLM 04730e8 and later write the query
        /// token-major and fill only under DCP or HiSparse; false models that.
        #[serde(default = "default_glm53_flash_mla_layout_copies")]
        #[param(default = true, cache_key)]
        mla_layout_copies: bool,
        /// vLLM `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`: the prefill indexer splits
        /// its queries so each MQA-logits call's fp32 `[rows, pools]` matrix
        /// fits this many MiB, and runs logits and top-k once per piece.
        #[serde(default = "default_glm53_flash_indexer_max_logits_mb")]
        #[param(default = 512, cache_key)]
        indexer_max_logits_mb: u32,
    },
    /// [`Self::Glm53FlashVllmFp8KdaDsaMoe`]'s graph for NVIDIA's ModelOpt NVFP4
    /// checkpoint (`nvidia/GLM-5.3-Flash-NVFP4`): NVFP4 routed experts and
    /// dense FFN, BF16 shared expert; attention and the router as in FP8.
    Glm53FlashVllmNvfp4KdaDsaMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Tensor-parallel rank count: attention heads, the dense FFN and the
        /// shared expert are split over it, and so are the routed experts.
        #[serde(default = "default_glm52_nvfp4_parallel_size")]
        #[param(default = 4, cache_key)]
        tp_size: u16,
        /// vLLM `--enable-expert-parallel`. On, each rank owns
        /// `n_routed_experts / tp_size` whole experts (EP = TP). Off, each rank
        /// owns every expert, sliced to `moe_intermediate_size / tp_size` on
        /// the intermediate axis (MoE TP).
        #[serde(default = "default_glm53_flash_enable_expert_parallel")]
        #[param(default = true, cache_key)]
        enable_expert_parallel: bool,
        /// Configured context cap: the indexer's logits row stride and the
        /// sparse-index page-table extent.
        #[serde(default = "default_glm53_flash_max_model_len")]
        #[param(default = 8192, cache_key)]
        max_model_len: u32,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as a
        /// manifest path. Requires `routing = corpus`.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--cudagraph-capture-sizes`. The engine pads an iteration's
        /// token count up to the next captured size, and every kernel outside
        /// the attention graph break runs on the padded rows. Empty: no
        /// padding (eager).
        #[serde(default)]
        cudagraph_capture_sizes: Vec<u32>,
        /// The two sparse-MLA layout copies of vLLM fork 3f667d7: the
        /// `q_concat` transpose copy of the absorbed query and the `masked_fill_`
        /// over the attention output. vLLM 04730e8 and later write the query
        /// token-major and fill only under DCP or HiSparse; false models that.
        #[serde(default = "default_glm53_flash_mla_layout_copies")]
        #[param(default = true, cache_key)]
        mla_layout_copies: bool,
        /// vLLM `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`: the prefill indexer splits
        /// its queries so each MQA-logits call's fp32 `[rows, pools]` matrix
        /// fits this many MiB, and runs logits and top-k once per piece.
        #[serde(default = "default_glm53_flash_indexer_max_logits_mb")]
        #[param(default = 512, cache_key)]
        indexer_max_logits_mb: u32,
    },
    /// [`Self::Glm53FlashVllmFp8PpKdaDsaMoe`] for the NVFP4 checkpoint (see
    /// [`Self::Glm53FlashVllmNvfp4KdaDsaMoe`]). Runs only under deployment `pp`.
    Glm53FlashVllmNvfp4PpKdaDsaMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Pipeline stages, one GPU each.
        #[param(cache_key)]
        pp_size: u16,
        /// Configured context cap: the indexer's logits row stride and the
        /// sparse-index page-table extent.
        #[serde(default = "default_glm53_flash_max_model_len")]
        #[param(default = 8192, cache_key)]
        max_model_len: u32,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        /// Measured per-expert popularity captured at EP1 (one rank owns all
        /// 288 experts). Requires `routing = popularity`.
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as a
        /// manifest path. Requires `routing = corpus`. Routes are per token, so
        /// a corpus captured at any EP size folds to EP1.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--cudagraph-capture-sizes`, as for the TP = EP graph. Empty:
        /// no padding (eager).
        #[serde(default)]
        cudagraph_capture_sizes: Vec<u32>,
        /// vLLM `VLLM_PP_LAYER_PARTITION`: layers per stage, `pp_size` counts
        /// summing to 45. Empty: vLLM's default `get_pp_indices` split.
        #[serde(default)]
        layer_partition: Vec<u32>,
        /// The two sparse-MLA layout copies of vLLM fork 3f667d7: the
        /// `q_concat` transpose copy of the absorbed query and the `masked_fill_`
        /// over the attention output. vLLM 04730e8 and later write the query
        /// token-major and fill only under DCP or HiSparse; false models that.
        #[serde(default = "default_glm53_flash_mla_layout_copies")]
        #[param(default = true, cache_key)]
        mla_layout_copies: bool,
        /// vLLM `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`: the prefill indexer splits
        /// its queries so each MQA-logits call's fp32 `[rows, pools]` matrix
        /// fits this many MiB, and runs logits and top-k once per piece.
        #[serde(default = "default_glm53_flash_indexer_max_logits_mb")]
        #[param(default = 512, cache_key)]
        indexer_max_logits_mb: u32,
    },
    /// [`Self::Glm53FlashVllmFp8DpAttnEpMoe`] for the NVFP4 checkpoint (see
    /// [`Self::Glm53FlashVllmNvfp4KdaDsaMoe`]): the routed experts sit behind
    /// an NVFP4 all-gather and a bf16 reduce-scatter.
    Glm53FlashVllmNvfp4DpAttnEpMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Data-parallel attention ranks, which are also the expert-parallel
        /// group: one GPU each.
        #[serde(default = "default_glm52_nvfp4_parallel_size")]
        #[param(default = 4, cache_key)]
        ep_size: u16,
        /// Configured context cap: the indexer's logits row stride and the
        /// sparse-index page-table extent.
        #[serde(default = "default_glm53_flash_max_model_len")]
        #[param(default = 8192, cache_key)]
        max_model_len: u32,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        /// Measured per-expert popularity captured at the same
        /// `expert_parallel_size`. Requires `routing = popularity`.
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as a
        /// manifest path. Requires `routing = corpus`.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--cudagraph-capture-sizes`. When every rank's batch fits a
        /// captured size, all ranks pad to the busiest rank's graph; otherwise
        /// they run eager on their own rows. Empty: always eager.
        #[serde(default)]
        cudagraph_capture_sizes: Vec<u32>,
        /// The two sparse-MLA layout copies of vLLM fork 3f667d7: the
        /// `q_concat` transpose copy of the absorbed query and the `masked_fill_`
        /// over the attention output. vLLM 04730e8 and later write the query
        /// token-major and fill only under DCP or HiSparse; false models that.
        #[serde(default = "default_glm53_flash_mla_layout_copies")]
        #[param(default = true, cache_key)]
        mla_layout_copies: bool,
        /// vLLM `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`: the prefill indexer splits
        /// its queries so each MQA-logits call's fp32 `[rows, pools]` matrix
        /// fits this many MiB, and runs logits and top-k once per piece.
        #[serde(default = "default_glm53_flash_indexer_max_logits_mb")]
        #[param(default = 512, cache_key)]
        indexer_max_logits_mb: u32,
    },
    /// [`Self::Glm53FlashVllmNvfp4DpAttnEpMoe`] verifying a DFlash2
    /// proposer's drafts (`arch::glm53_flash_vllm_nvfp4_dflash2`), one TP1
    /// proposer per attention DP rank, with the DP graph's options.
    Glm53FlashVllmNvfp4DpAttnEpMoeDflash2 {
        #[serde(flatten)]
        model: ModelSpec,
        #[serde(default = "default_glm52_nvfp4_parallel_size")]
        #[param(default = 4, cache_key)]
        ep_size: u16,
        /// KDA layers tensor-parallel over groups of this many owner ranks
        /// (the MLA layers stay on each request's owner rank); 1 is plain DP
        /// attention. Must divide `ep_size`.
        #[serde(default = "default_kda_tp_gpus")]
        #[param(default = 1, cache_key)]
        kda_tp_gpus: u16,
        /// Each KDA layer's two low-rank gate up-projections (`f_b`, `g_b`) as
        /// one batched GEMM. Not a vLLM path.
        #[serde(default)]
        #[param(cache_key)]
        kda_gate_b_merged: bool,
        #[serde(default = "default_glm53_flash_max_model_len")]
        #[param(default = 8192, cache_key)]
        max_model_len: u32,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        /// Measured per-expert popularity captured at the same EP size.
        /// Requires `routing = popularity`.
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes, as a manifest path. Requires
        /// `routing = corpus`.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--cudagraph-capture-sizes`, as for the DP graph.
        #[serde(default)]
        cudagraph_capture_sizes: Vec<u32>,
        /// The fork's two sparse-MLA layout copies; the DFlash2 fork is past
        /// vLLM 04730e8 and has none.
        #[serde(default)]
        #[param(default = false, cache_key)]
        mla_layout_copies: bool,
        #[serde(default = "default_glm53_flash_indexer_max_logits_mb")]
        #[param(default = 512, cache_key)]
        indexer_max_logits_mb: u32,
        /// Candidate positions drafted per request per iteration, at most the
        /// Flash draft checkpoint's `block_size - 1` (7). The speculative
        /// worker's `draft_tokens` must match.
        #[serde(default = "default_dflash2_draft_tokens")]
        #[param(default = 7, cache_key)]
        draft_tokens: u32,
        /// KDA recurrent state pool dtype (vLLM `--mamba-ssm-cache-dtype`).
        #[serde(default)]
        #[param(string, default = "fp32", choices = KDA_STATE_DTYPES, cache_key)]
        kda_state_dtype: KdaStateDtype,
        /// How the verify advances the KDA state: `recurrent_per_position` (the
        /// fork's GLM path, the default: a state per position, `1 + draft_tokens`
        /// state pages per request, no commit) or `recoverssm` (verify +
        /// commit, one state per request).
        #[serde(default = "default_recurrent_per_position")]
        #[param(string, default = "recurrent_per_position", choices = KDA_SPEC_VERIFIES, cache_key)]
        kda_spec_verify: crate::worklet::KdaSpecVerify,
        /// Expert exchange: `measured_allgather` (vLLM's all-gatherv and
        /// reduce-scatterv) or `fixed_pattern` (the fixed-pattern MNNVL
        /// exchange as measured; NVFP4 EP 8).
        #[serde(default)]
        #[param(string, default = "measured_allgather", choices = EXPERT_EXCHANGES, cache_key)]
        expert_exchange: crate::arch::glm53_flash_vllm_fp8_dp_attn_ep_moe::ExpertExchange,
        /// What-if (`extrapolated`): every fused mHC post/pre boundary costs
        /// vLLM's row less this many us. Absent: vLLM's rows.
        #[serde(default)]
        #[param(cache_key)]
        mhc_fused_saving_us: Option<f64>,
        /// DSA elementwise glue: `placeholder` (17 launches of 64 B per token)
        /// or `fork_launch_list` (the fork's 7 decode launches at their real
        /// bytes).
        #[serde(default = "default_fork_launch_list")]
        #[param(string, default = "fork_launch_list", choices = DSA_GLUE_PRICINGS, cache_key)]
        dsa_glue: DsaGluePricing,
        /// Not a fork path: DSA front cut A ([`DsaHiddenGemmMode`]).
        #[serde(default)]
        #[param(string, default = "separate", choices = DSA_HIDDEN_GEMM_MODES, cache_key)]
        dsa_hidden_gemm: DsaHiddenGemmMode,
        /// `dsa_hidden_gemm: fp32_split_state_writer` only (required there):
        /// the epilogue's backend.
        #[serde(default)]
        #[param(cache_key)]
        dsa_hidden_gemm_backend: Option<String>,
        /// Not a fork path: DSA front cut B ([`DsaQLoraGemmMode`]).
        #[serde(default)]
        #[param(string, default = "separate", choices = DSA_Q_LORA_GEMM_MODES, cache_key)]
        dsa_q_lora_gemm: DsaQLoraGemmMode,
        /// `dsa_q_lora_gemm: concat_hadamard_epilogue` only (required there):
        /// the epilogue's backend.
        #[serde(default)]
        #[param(cache_key)]
        dsa_q_lora_gemm_backend: Option<String>,
        /// Not a fork path: DSA front cut C, `q_absorb` with the fp8 quant in
        /// its epilogue ([`crate::worklet::Glm53DsaFront::q_absorb_fp8`]), on
        /// this backend.
        #[serde(default)]
        #[param(cache_key)]
        dsa_q_absorb_fp8_backend: Option<String>,
        /// Not a fork path: DSA indexer cut E, the decode rows' selected pools
        /// to the final page table in one launch
        /// ([`crate::worklet::Glm53DsaFront::topk_page_table`]), on this
        /// backend.
        #[serde(default)]
        #[param(cache_key)]
        dsa_topk_page_table_backend: Option<String>,
        /// Compute the MoE router gate once per layer (the fork since #55736)
        /// instead of twice (the 2026-09 capture's line).
        #[serde(default = "default_true")]
        #[param(default = true, cache_key)]
        router_gate_once: bool,
        /// Price a bf16 copy of the routed input ahead of the dispatch quant
        /// (`moe.input_glue`, from the 2026-09 FP8 TP4/EP4 capture). The
        /// fork's NVFP4 naive DP/EP path quantizes the hidden directly, so
        /// `false` drops the leaf.
        #[serde(default)]
        #[param(default = false, cache_key)]
        moe_input_copy: bool,
        /// Fold the MoE block's shared + routed add into the next mHC post
        /// (`mhc_fused_sum_post_pre_rms_norm` `vllm_tilelang_post_addend`);
        /// the add is not billed. Refused beside a nonzero
        /// `mhc_fused_saving_us`, whose after-MoE share already removes it.
        #[serde(default)]
        #[param(cache_key)]
        moe_combine_add_in_mhc_post: bool,
        /// How the aux-stream shared expert (local rows <= 256) overlaps the
        /// MoE chain: `exposed` (the default: the measured time it adds) or
        /// `chain` (an overlap factor).
        #[serde(default = "default_moe_shared_exposed")]
        #[param(string, default = "exposed", choices = MOE_SHARED_OVERLAPS, cache_key)]
        moe_shared_overlap: crate::arch::glm53_flash_vllm_fp8_dp_attn_ep_moe::MoeSharedOverlap,
        /// `moe_shared_overlap: exposed` only: a table `{"points": [{"rows",
        /// "<column>_us", ...}, ...]}` of the time the shared expert adds to the
        /// MoE block by rows per rank. Absent: the measured 8x B200 EP8 table in
        /// `model/`.
        #[serde(default)]
        #[param(cache_key)]
        moe_shared_pair_file: Option<String>,
        /// The `<column>_us` column of the table; absent: `production`, the
        /// vLLM fork's MoE block.
        #[serde(default)]
        #[param(cache_key)]
        moe_shared_pair_column: Option<String>,
        /// DFlash2 draft attention leaf
        /// ([`crate::arch::glm53_vllm_nvfp4_dsa_moe_dflash2::DraftAttention`]).
        #[serde(default = "default_fa4_fp8_split32")]
        #[param(string, default = "fa4_fp8_split32", choices = DRAFT_ATTENTIONS, cache_key)]
        draft_attention: crate::arch::glm53_vllm_nvfp4_dsa_moe_dflash2::DraftAttention,
        /// DFlash2 draft vocabulary top-k
        /// ([`crate::arch::glm53_vllm_nvfp4_dsa_moe_dflash2::DraftTopk`]).
        #[serde(default)]
        #[param(string, default = "flashinfer", choices = DRAFT_TOPKS, cache_key)]
        draft_topk: crate::arch::glm53_vllm_nvfp4_dsa_moe_dflash2::DraftTopk,
        /// DFlash2 context K/V: one launch normalizes, rotates and writes every
        /// draft layer's cache (`layered_kv_norm_rope_cache_write`).
        #[serde(default)]
        #[param(cache_key)]
        draft_context_kv_store: bool,
        /// DFlash2 draft layers: one launch for q/k norm, RoPE, the fp8 query
        /// quant and the block's K/V write. Needs an fp8 `draft_attention`.
        #[serde(default)]
        #[param(cache_key)]
        draft_qk_norm_rope_kv_write: bool,
        /// DFlash2 draft layers: each conv finish fused with the residual
        /// RMSNorm after it.
        #[serde(default)]
        #[param(cache_key)]
        draft_conv_residual_norm: bool,
    },
    /// GLM-5.3-Flash NVFP4 + DFlash2 under attention DP with the KDA layers
    /// tensor-parallel in groups of `kda_tp_gpus`, the fixed-pattern expert
    /// exchange and the design's kernels. Not a vLLM layout: a fixed
    /// configuration of [`Self::Glm53FlashVllmNvfp4DpAttnEpMoeDflash2`]
    /// (`arch::glm53_flash_nvfp4_dp_attn_kda_tp_fused_ep_moe_dflash2`); only
    /// the deployment's shape is a parameter.
    Glm53FlashNvfp4DpAttnKdaTpFusedEpMoeDflash2 {
        #[serde(flatten)]
        model: ModelSpec,
        /// Attention DP ranks, which are also the EP group: one GPU each.
        #[serde(default = "default_kda_tp_fused_ep_size")]
        #[param(default = 8, cache_key)]
        ep_size: u16,
        /// GPUs per KDA tensor-parallel group (1 = plain DP attention). Must
        /// divide `ep_size`; the fastest group size depends on the load.
        #[param(cache_key)]
        kda_tp_gpus: u16,
        #[serde(default = "default_glm53_flash_max_model_len")]
        #[param(default = 8192, cache_key)]
        max_model_len: u32,
        /// MoE routing; no default (a synthetic default would hide a missing
        /// corpus).
        #[param(string, choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        /// Measured per-expert popularity captured at the same EP size.
        /// Requires `routing = popularity`.
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes, as a manifest path. Requires
        /// `routing = corpus`.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--cudagraph-capture-sizes`, as for the DP graph.
        #[serde(default)]
        cudagraph_capture_sizes: Vec<u32>,
        #[serde(default = "default_glm53_flash_indexer_max_logits_mb")]
        #[param(default = 512, cache_key)]
        indexer_max_logits_mb: u32,
        /// Candidate positions drafted per request per iteration, at most the
        /// Flash draft checkpoint's `block_size - 1` (7). The speculative
        /// worker's `draft_tokens` must match.
        #[serde(default = "default_dflash2_draft_tokens")]
        #[param(default = 7, cache_key)]
        draft_tokens: u32,
    },
    /// SGLang's B200 NVFP4 launch graph under pure tensor parallelism. Every
    /// rank owns all experts (EP1) and shards the routed intermediate axis by
    /// TP, so there is no expert-parallel or NVLink-domain selector.
    Glm52SglangNvfp4TpDsaMoe {
        #[serde(flatten)]
        model: ModelSpec,
        #[param(cache_key)]
        tp_size: u16,
        /// Configured context cap and padded DSA-logits row stride.
        #[param(cache_key)]
        max_model_len: u32,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(string, default = "off", choices = GLM52_MTP_MODES, cache_key)]
        mtp_mode: Glm52MtpMode,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as
        /// a manifest path. Requires `routing = corpus`. One artifact covers
        /// every routed layer, so the body MoE and the MTP MoE read slices
        /// of it rather than separate files.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
    },
    /// Inkling's ModelOpt NVFP4 checkpoints (`thinkingmachines/Inkling-NVFP4`,
    /// `Inkling-Small-NVFP4`) under the vLLM fork's TP group: TP attention,
    /// dense FFN and sink experts, replicated router, routed experts EP = TP
    /// or MoE TP, and every sublayer closed by the Lamport RS + sconv + AG +
    /// norm call.
    InklingVllmNvfp4RelAttnMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Tensor-parallel rank count (at least 2: the Lamport boundary).
        #[serde(default = "default_glm52_nvfp4_parallel_size")]
        #[param(default = 4, cache_key)]
        tp_size: u16,
        /// vLLM `--enable-expert-parallel`. On, each rank owns
        /// `n_routed_experts / tp_size` whole experts (EP = TP). Off, each rank
        /// owns every expert, sliced on the intermediate axis (MoE TP).
        #[serde(default = "default_glm53_flash_enable_expert_parallel")]
        #[param(default = true, cache_key)]
        enable_expert_parallel: bool,
        /// Configured context cap: the global layers' FA4 split count. It is
        /// also part of every attention layer's kernel config, local layers
        /// included (their context axis stops at it), and each request's
        /// context is validated against it.
        #[serde(default = "default_glm53_flash_max_model_len")]
        #[param(default = 8192, cache_key)]
        max_model_len: u32,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as a
        /// manifest path. Requires `routing = corpus`. Its layer axis is the
        /// MoE layers; the BF16-expert layers and the NVFP4 ones each fold
        /// their own slice.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--cudagraph-capture-sizes`. Every kernel but attention runs
        /// on the token count padded up to the next captured size; attention
        /// on a uniform-decode step runs at the padded request count. The
        /// sizes also decide whether the sink experts overlap the routed call
        /// (FULL-graph decode or eager steps only). Empty: no padding (eager).
        #[serde(default)]
        cudagraph_capture_sizes: Vec<u32>,
    },
    /// MiMo-V2's FP8 checkpoints (`XiaomiMiMo/MiMo-V2.6-Flash-RL`) under
    /// vLLM's TP group: TP attention over global and sliding-window layers
    /// (FA4 DiffKV), replicated router, routed MXFP4 experts EP = TP with BF16
    /// activations, and every TP all-reduce fused with the residual add and
    /// RMSNorm (FlashInfer MNNVL).
    MimoV2VllmFp8SwaMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Tensor-parallel rank count; the routed experts are expert-parallel
        /// over the same ranks (`--enable-expert-parallel`).
        #[serde(default = "default_glm52_nvfp4_parallel_size")]
        #[param(default = 4, cache_key)]
        tp_size: u16,
        /// Configured context cap: FA4's split count on a FULL-graph decode
        /// and the context axis top of every attention layer's config. Each
        /// request's context is validated against it.
        #[serde(default = "default_mimo_v2_max_model_len")]
        #[param(default = 131072, cache_key)]
        max_model_len: u32,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as a
        /// manifest path. Requires `routing = corpus`. Its layer axis is the
        /// MoE layers (layer 0's dense MLP has none).
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--cudagraph-capture-sizes` (FULL_AND_PIECEWISE). Every kernel
        /// but attention runs on the token count padded up to the next
        /// captured size; attention on a uniform-decode step runs at the
        /// padded request count from a FULL graph. Empty: no padding, no graphs
        /// (eager).
        #[serde(default)]
        cudagraph_capture_sizes: Vec<u32>,
    },
    /// Qwen3.5-family hybrid decoders (`Qwen3_5ForConditionalGeneration`,
    /// text only) with ModelOpt mixed-precision weights (FP8 per-tensor
    /// attention projections, NVFP4 MLP and lm_head) on one GPU under vLLM:
    /// Gated DeltaNet and gated-GQA layers (FlashInfer trtllm-gen, FP8 KV),
    /// each with a dense SwiGLU MLP. Precisions come from the checkpoint's
    /// `quantization_config`; `fp8` is not read.
    Qwen35VllmNvfp4GdnDense {
        #[serde(flatten)]
        model: ModelSpec,
        /// Configured context cap; each request's context is validated
        /// against it.
        #[serde(default = "default_qwen35_max_model_len")]
        #[param(default = 131072, cache_key)]
        max_model_len: u32,
        /// vLLM `--cudagraph-capture-sizes` (FULL_AND_PIECEWISE). The compiled
        /// pieces run on the token count padded up to the next captured size;
        /// a uniform decode replays a FULL graph, so the GDN core and decode
        /// attention run padded too. Empty: no padding (eager).
        #[serde(default)]
        cudagraph_capture_sizes: Vec<u32>,
    },
    /// Qwen4Exp hybrid decoders (`Qwen4ExpForConditionalGeneration`, text
    /// only; Qwen3.8-Flash-Next) with ModelOpt mixed-precision weights on one
    /// GPU under the vLLM fork, drafting with the checkpoint's MTP layer:
    /// Gated DeltaNet and QSA (sparse GQA with a compressed indexer) layers
    /// joined by hyper-connections, PLE n-gram blocks, NVFP4 routed experts
    /// with a BF16 shared expert, and an FP8-block MTP layer. Precisions come
    /// from the checkpoint's `quantization_config`; `fp8` is not read.
    Qwen4ExpVllmNvfp4GdnQsaMoeSpeculative {
        #[serde(flatten)]
        model: ModelSpec,
        /// Configured context cap; each request's context is validated
        /// against it, and it sizes the QSA top-k and block-table rows.
        #[serde(default = "default_qwen4_exp_max_model_len")]
        #[param(default = 131072, cache_key)]
        max_model_len: u32,
        /// MTP `num_speculative_tokens`: fixes the verify width at
        /// `draft_tokens + 1`, a profiled kernel shape.
        #[serde(default = "default_qwen4_exp_draft_tokens")]
        #[param(default = 3, cache_key)]
        draft_tokens: u32,
        /// vLLM `--kv-cache-dtype` of the QSA KV.
        #[serde(default)]
        #[param(string, default = "fp8", choices = QWEN4_EXP_CACHE_DTYPES, cache_key)]
        kv_cache_dtype: Qwen4ExpCacheDtype,
        /// The QSA indexer's key-cache dtype.
        #[serde(default)]
        #[param(string, default = "fp8", choices = QWEN4_EXP_CACHE_DTYPES, cache_key)]
        indexer_kv_dtype: Qwen4ExpCacheDtype,
        /// Where the PLE n-gram table lives: pinned host memory over UVA
        /// (the fork's CPU offload) or HBM.
        #[serde(default)]
        #[param(string, default = "host_uva", choices = PLE_TABLE_RESIDENCIES, cache_key)]
        ple_table_residency: PleTableResidency,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
        /// Recorded per-token expert routes from a `token_corpus` pass, as a
        /// manifest path. Requires `routing = corpus`. Its layer axis is the
        /// body's MoE layers, then the MTP layer.
        #[serde(default)]
        #[param(cache_key)]
        token_corpus_file: Option<String>,
        /// vLLM `--cudagraph-capture-sizes` (FULL_AND_PIECEWISE). The graph
        /// pieces run on the token count padded up to the next captured size;
        /// a uniform verify step replays a FULL graph, so its eager breaks run
        /// padded too. Empty: no padding (eager).
        #[serde(default)]
        cudagraph_capture_sizes: Vec<u32>,
    },
}

impl IterArchSel {
    /// The model identity/dims this arch operates on (every variant carries it).
    pub fn model(&self) -> &ModelSpec {
        match self {
            Self::Qwen36Local { model, .. }
            | Self::Llama3Dense { model }
            | Self::Llama3DenseTp { model, .. }
            | Self::Llama3DpAttnTpFfn { model, .. }
            | Self::Qwen3MoeDpAttnEpFfn { model, .. }
            | Self::Qwen3MoeFp8DpAttnEpFfn { model, .. }
            | Self::Qwen3VllmMoeDpAttnEpFfn { model, .. }
            | Self::DeepseekV4Vllm { model, .. }
            | Self::DeepseekV4VllmSerialStreams { model, .. }
            | Self::DeepseekV41Vllm { model, .. }
            | Self::DeepseekV41VllmSerialStreams { model, .. }
            | Self::DeepseekV41VllmDspark { model, .. }
            | Self::DeepseekV41VllmDpAttnEpMoeDspark { model, .. }
            | Self::DeepseekV41VllmDpAttnEpMoe { model, .. }
            | Self::DeepseekV41VllmPp { model, .. }
            | Self::Glm52VllmDsaMoe { model, .. }
            | Self::Glm52VllmNvfp4DsaMoe { model, .. }
            | Self::Glm52VllmNvfp4PpDsaMoe { model, .. }
            | Self::Glm52VllmNvfp4DpAttnDsaMoe { model, .. }
            | Self::Glm52VllmNvfp4DsaMoeSpeculative { model, .. }
            | Self::Glm53VllmNvfp4DsaMoeDflash2 { model, .. }
            | Self::Glm53FlashVllmFp8KdaDsaMoe { model, .. }
            | Self::Glm53FlashVllmFp8PpKdaDsaMoe { model, .. }
            | Self::Glm53FlashVllmFp8DpAttnEpMoe { model, .. }
            | Self::Glm53FlashVllmNvfp4KdaDsaMoe { model, .. }
            | Self::Glm53FlashVllmNvfp4PpKdaDsaMoe { model, .. }
            | Self::Glm53FlashVllmNvfp4DpAttnEpMoe { model, .. }
            | Self::Glm53FlashVllmNvfp4DpAttnEpMoeDflash2 { model, .. }
            | Self::Glm53FlashNvfp4DpAttnKdaTpFusedEpMoeDflash2 { model, .. }
            | Self::Glm52SglangNvfp4TpDsaMoe { model, .. }
            | Self::InklingVllmNvfp4RelAttnMoe { model, .. }
            | Self::MimoV2VllmFp8SwaMoe { model, .. }
            | Self::Qwen35VllmNvfp4GdnDense { model, .. }
            | Self::Qwen4ExpVllmNvfp4GdnQsaMoeSpeculative { model, .. } => model,
        }
    }

    /// The longest request this arch serves, in tokens of context (prompt,
    /// output and any tokens a verify reads ahead): its `max_model_len` where
    /// the arch takes one, else the checkpoint's `max_position_embeddings`, as
    /// vLLM defaults it. Every variant is listed so a new one decides which.
    pub fn max_model_len(&self) -> anyhow::Result<u32> {
        match self {
            Self::Glm52VllmNvfp4DsaMoe { max_model_len, .. }
            | Self::Glm52VllmNvfp4PpDsaMoe { max_model_len, .. }
            | Self::Glm52VllmNvfp4DpAttnDsaMoe { max_model_len, .. }
            | Self::Glm52VllmNvfp4DsaMoeSpeculative { max_model_len, .. }
            | Self::Glm53VllmNvfp4DsaMoeDflash2 { max_model_len, .. }
            | Self::Glm53FlashVllmFp8KdaDsaMoe { max_model_len, .. }
            | Self::Glm53FlashVllmFp8PpKdaDsaMoe { max_model_len, .. }
            | Self::Glm53FlashVllmFp8DpAttnEpMoe { max_model_len, .. }
            | Self::Glm53FlashVllmNvfp4KdaDsaMoe { max_model_len, .. }
            | Self::Glm53FlashVllmNvfp4PpKdaDsaMoe { max_model_len, .. }
            | Self::Glm53FlashVllmNvfp4DpAttnEpMoe { max_model_len, .. }
            | Self::Glm53FlashVllmNvfp4DpAttnEpMoeDflash2 { max_model_len, .. }
            | Self::Glm53FlashNvfp4DpAttnKdaTpFusedEpMoeDflash2 { max_model_len, .. }
            | Self::Glm52SglangNvfp4TpDsaMoe { max_model_len, .. }
            | Self::InklingVllmNvfp4RelAttnMoe { max_model_len, .. }
            | Self::MimoV2VllmFp8SwaMoe { max_model_len, .. }
            | Self::Qwen35VllmNvfp4GdnDense { max_model_len, .. }
            | Self::Qwen4ExpVllmNvfp4GdnQsaMoeSpeculative { max_model_len, .. } => {
                Ok(*max_model_len)
            }
            Self::DeepseekV41Vllm {
                model,
                max_model_len,
                ..
            }
            | Self::DeepseekV41VllmSerialStreams {
                model,
                max_model_len,
                ..
            }
            | Self::DeepseekV41VllmDspark {
                model,
                max_model_len,
                ..
            }
            | Self::DeepseekV41VllmDpAttnEpMoeDspark {
                model,
                max_model_len,
                ..
            }
            | Self::DeepseekV41VllmDpAttnEpMoe {
                model,
                max_model_len,
                ..
            }
            | Self::DeepseekV41VllmPp {
                model,
                max_model_len,
                ..
            } => super::deepseek_v41_vllm::resolve_max_model_len(
                *max_model_len,
                model.max_position_embeddings()?,
            ),
            Self::Qwen36Local { model, .. }
            | Self::Llama3Dense { model }
            | Self::Llama3DenseTp { model, .. }
            | Self::Llama3DpAttnTpFfn { model, .. }
            | Self::Qwen3MoeDpAttnEpFfn { model, .. }
            | Self::Qwen3MoeFp8DpAttnEpFfn { model, .. }
            | Self::Qwen3VllmMoeDpAttnEpFfn { model, .. }
            | Self::DeepseekV4Vllm { model, .. }
            | Self::DeepseekV4VllmSerialStreams { model, .. }
            | Self::Glm52VllmDsaMoe { model, .. } => model.max_position_embeddings(),
        }
    }
}

#[cfg(test)]
mod iter_tests {
    use super::*;

    #[test]
    fn qwen36_local_selector_publishes_only_routing_beyond_the_model() {
        let parsed: IterArchSel = serde_json::from_str(
            r#"{"type":"qwen36_local","model_config":"model/config/qwen3_6_35b_a3b_fp8.json","num_layers":40,"sim_num_layers":4,"fp8":true}"#,
        )
        .expect("qwen36_local selector parses");
        let IterArchSel::Qwen36Local {
            model,
            routing,
            routing_seed,
            expert_popularity_file,
        } = &parsed
        else {
            panic!("expected qwen36_local")
        };
        // Routing is optional on the wire and defaults to balanced, so an
        // existing preset keeps its meaning.
        assert_eq!(*routing, RoutingKind::Uniform);
        assert!(routing_seed.is_none() && expert_popularity_file.is_none());
        assert_eq!(model.model_config, "model/config/qwen3_6_35b_a3b_fp8.json");
        assert_eq!(model.num_layers, Some(40));
        assert_eq!(model.sim_num_layers, Some(4));
        assert!(model.fp8);
        assert!(std::ptr::eq(parsed.model(), model));

        let (tag, params) = IterArchSel::SCHEMA
            .iter()
            .find(|(tag, _)| *tag == "qwen36_local")
            .expect("qwen36_local provider schema row");
        assert_eq!(*tag, "qwen36_local");
        // Flattened ModelSpec fields are intentionally published once through
        // schema::dump::arch_common, not duplicated on every provider row. TP1
        // and EP1 are architectural invariants, so routing is the ONLY thing
        // this arch adds: it is a property of the workload's token stream, not
        // of a sharding choice, and the grouped GEMM costs it either way.
        let published = serde_json::to_value(params).unwrap();
        let published: Vec<&str> = published
            .as_array()
            .unwrap()
            .iter()
            .map(|param| param["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            published,
            ["routing", "routing_seed", "expert_popularity_file"]
        );
        let popularity = serde_json::to_value(params).unwrap();
        let popularity = popularity
            .as_array()
            .unwrap()
            .iter()
            .find(|param| param["name"] == "expert_popularity_file")
            .expect("expert popularity schema parameter")
            .clone();
        assert_eq!(popularity["affects_cache"], true);
        let common = serde_json::to_value(ModelSpec::PARAMS).unwrap();
        let names = common
            .as_array()
            .unwrap()
            .iter()
            .map(|param| param["name"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            ["model_config", "num_layers", "sim_num_layers", "fp8"]
        );
    }

    fn parse_qwen(extra: &str) -> Result<IterArchSel, serde_json::Error> {
        serde_json::from_str(&format!(
            r#"{{"type":"qwen3_moe_dp_attn_ep_ffn","model_config":"model/config/qwen3_235b.json","fp8":false{extra}}}"#
        ))
    }

    fn parse_fp8_qwen(extra: &str) -> Result<IterArchSel, serde_json::Error> {
        serde_json::from_str(&format!(
            r#"{{"type":"qwen3_moe_fp8_dp_attn_ep_ffn","model_config":"model/config/qwen3_235b.json","fp8":true{extra}}}"#
        ))
    }

    fn parse_vllm_qwen(extra: &str) -> Result<IterArchSel, serde_json::Error> {
        serde_json::from_str(&format!(
            r#"{{"type":"qwen3_vllm_moe_dp_attn_ep_ffn","model_config":"model/config/qwen3_235b.json","fp8":true{extra}}}"#
        ))
    }

    fn parse_glm52(extra: &str) -> Result<IterArchSel, serde_json::Error> {
        serde_json::from_str(&format!(
            r#"{{"type":"glm52_vllm_dsa_moe","model_config":"model/config/glm52.json","fp8":false{extra}}}"#
        ))
    }

    #[test]
    fn qwen_selector_parses_profile_path_and_marks_it_cache_relevant() {
        let parsed = parse_qwen(
            r#","attn_tp_size":4,"ep_size":4,"hp_size":1,"nvl_num_gpu":4,"routing":"popularity","expert_popularity_file":"profile_expert_popularity/expert_popularity.json""#,
        )
        .expect("qwen selector with popularity profile parses");
        let IterArchSel::Qwen3MoeDpAttnEpFfn {
            attn_tp_size,
            ep_size,
            expert_popularity_file,
            ..
        } = parsed
        else {
            panic!("expected qwen3_moe_dp_attn_ep_ffn")
        };
        assert_eq!(attn_tp_size, 4);
        assert_eq!(ep_size, 4);
        assert_eq!(
            expert_popularity_file.as_deref(),
            Some("profile_expert_popularity/expert_popularity.json")
        );

        let (_, params) = IterArchSel::SCHEMA
            .iter()
            .find(|(tag, _)| *tag == "qwen3_moe_dp_attn_ep_ffn")
            .expect("qwen selector schema row");
        let params = serde_json::to_value(params).unwrap();
        let popularity = params
            .as_array()
            .unwrap()
            .iter()
            .find(|param| param["name"] == "expert_popularity_file")
            .expect("expert popularity schema parameter");
        assert_eq!(popularity["affects_cache"], true);
    }

    #[test]
    fn vllm_qwen_selector_is_a_distinct_public_arch_with_the_same_parallel_contract() {
        assert!(matches!(
            parse_fp8_qwen(r#", "attn_tp_size":4,"ep_size":8,"hp_size":1,"nvl_num_gpu":8"#,)
                .unwrap(),
            IterArchSel::Qwen3MoeFp8DpAttnEpFfn { .. }
        ));
        let parsed = parse_vllm_qwen(
            r#", "attn_tp_size":4,"ep_size":4,"hp_size":1,"nvl_num_gpu":4,"routing":"popularity","expert_popularity_file":"expert_popularity.json""#,
        )
        .expect("vLLM-alignment Qwen selector parses");
        assert!(matches!(
            parsed,
            IterArchSel::Qwen3VllmMoeDpAttnEpFfn {
                attn_tp_size: 4,
                ep_size: 4,
                hp_size: 1,
                nvl_num_gpu: 4,
                expert_popularity_file: Some(ref path),
                ..
            } if path == "expert_popularity.json"
        ));

        let tags = IterArchSel::SCHEMA
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>();
        assert!(tags.contains(&"qwen3_moe_dp_attn_ep_ffn"));
        assert!(tags.contains(&"qwen3_moe_fp8_dp_attn_ep_ffn"));
        assert!(tags.contains(&"qwen3_vllm_moe_dp_attn_ep_ffn"));
    }

    #[test]
    fn glm52_vllm_selector_is_registered_and_shares_the_glm_parameter_surface() {
        // `fp8: true` pairs with the FP8 checkpoint's config, not the BF16 one:
        // `model/work/floors.py` rejects the mismatched combination, and this
        // literal is what gets copied into new presets.
        let parsed: IterArchSel = serde_json::from_str(
            r#"{"type":"glm52_vllm_dsa_moe","model_config":"model/config/glm52_fp8.json","fp8":true}"#,
        )
        .expect("vLLM-granularity GLM selector parses");
        let IterArchSel::Glm52VllmDsaMoe {
            model,
            ep_size,
            nvl_num_gpu,
            routing,
            routing_seed,
            mtp_mode,
            expert_popularity_file,
        } = &parsed
        else {
            panic!("expected glm52_vllm_dsa_moe")
        };
        assert_eq!(model.model_config, "model/config/glm52_fp8.json");
        assert!(model.fp8);
        assert_eq!(*ep_size, 8);
        assert_eq!(*nvl_num_gpu, 8);
        assert_eq!(*routing, RoutingKind::Uniform);
        assert_eq!(*routing_seed, None);
        assert_eq!(*mtp_mode, Glm52MtpMode::Off);
        assert_eq!(*expert_popularity_file, None);
        assert_eq!(parsed.model().model_config, model.model_config);

        let tags = IterArchSel::SCHEMA
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>();
        assert!(tags.contains(&"glm52_vllm_dsa_moe"));
    }

    #[test]
    fn glm52_nvfp4_selector_defaults_to_tp4_ep4() {
        let parsed: IterArchSel = serde_json::from_str(
            r#"{"type":"glm52_vllm_nvfp4_dsa_moe","model_config":"model/config/glm52_nvfp4.json","fp8":false}"#,
        )
        .expect("B200 NVFP4 GLM selector parses");
        let IterArchSel::Glm52VllmNvfp4DsaMoe {
            model,
            ep_size,
            nvl_num_gpu,
            max_model_len,
            routing,
            routing_seed,
            mtp_mode,
            expert_popularity_file,
            token_corpus_file,
        } = &parsed
        else {
            panic!("expected glm52_vllm_nvfp4_dsa_moe")
        };
        assert_eq!(model.model_config, "model/config/glm52_nvfp4.json");
        assert!(!model.fp8);
        assert_eq!((*ep_size, *nvl_num_gpu), (4, 4));
        assert_eq!(*max_model_len, 1_048_576);
        assert_eq!(*routing, RoutingKind::Uniform);
        assert_eq!(*routing_seed, None);
        assert_eq!(*mtp_mode, Glm52MtpMode::Off);
        assert_eq!(*expert_popularity_file, None);
        assert_eq!(*token_corpus_file, None);
        assert!(std::ptr::eq(parsed.model(), model));
        assert!(IterArchSel::SCHEMA
            .iter()
            .any(|(tag, _)| *tag == "glm52_vllm_nvfp4_dsa_moe"));
    }

    #[test]
    fn glm52_sglang_nvfp4_selector_requires_explicit_tp_and_context_without_ep_knobs() {
        let parsed: IterArchSel = serde_json::from_str(
            r#"{"type":"glm52_sglang_nvfp4_tp_dsa_moe","model_config":"model/config/glm52_nvfp4.json","fp8":false,"tp_size":4,"max_model_len":8192}"#,
        )
        .expect("SGLang NVFP4 pure-TP selector parses");
        let IterArchSel::Glm52SglangNvfp4TpDsaMoe {
            model,
            tp_size,
            max_model_len,
            routing,
            routing_seed,
            mtp_mode,
            expert_popularity_file,
            token_corpus_file,
        } = &parsed
        else {
            panic!("expected glm52_sglang_nvfp4_tp_dsa_moe")
        };
        assert_eq!(model.model_config, "model/config/glm52_nvfp4.json");
        assert!(!model.fp8);
        assert_eq!(*tp_size, 4);
        assert_eq!(*max_model_len, 8_192);
        assert_eq!(*routing, RoutingKind::Uniform);
        assert_eq!(*routing_seed, None);
        assert_eq!(*mtp_mode, Glm52MtpMode::Off);
        assert_eq!(*expert_popularity_file, None);
        assert_eq!(*token_corpus_file, None);
        assert!(std::ptr::eq(parsed.model(), model));

        let params = IterArchSel::SCHEMA
            .iter()
            .find(|(tag, _)| *tag == "glm52_sglang_nvfp4_tp_dsa_moe")
            .expect("SGLang pure-TP tag is published")
            .1;
        let names = params.iter().map(|param| param.name).collect::<Vec<_>>();
        assert!(names.contains(&"tp_size"));
        assert!(!names.contains(&"ep_size"));
        assert!(!names.contains(&"nvl_num_gpu"));

        for incomplete in [
            r#"{"type":"glm52_sglang_nvfp4_tp_dsa_moe","model_config":"model/config/glm52_nvfp4.json","fp8":false,"max_model_len":8192}"#,
            r#"{"type":"glm52_sglang_nvfp4_tp_dsa_moe","model_config":"model/config/glm52_nvfp4.json","fp8":false,"tp_size":4}"#,
        ] {
            assert!(serde_json::from_str::<IterArchSel>(incomplete).is_err());
        }
    }

    #[test]
    fn glm52_selector_accepts_a_measured_expert_popularity_profile() {
        let parsed: IterArchSel = serde_json::from_str(
            r#"{"type":"glm52_vllm_dsa_moe","model_config":"model/config/glm52_fp8.json","fp8":true,
                 "routing":"popularity","expert_popularity_file":"profile_expert_popularity/expert_popularity.json"}"#,
        )
        .expect("GLM selector accepts an expert-popularity profile");
        let IterArchSel::Glm52VllmDsaMoe {
            expert_popularity_file,
            ..
        } = parsed
        else {
            panic!("expected glm52_vllm_dsa_moe")
        };
        assert_eq!(
            expert_popularity_file.as_deref(),
            Some("profile_expert_popularity/expert_popularity.json")
        );
    }

    #[test]
    fn glm52_selector_defaults_and_model_are_exact() {
        let parsed = parse_glm52("").expect("default GLM selector parses");
        let IterArchSel::Glm52VllmDsaMoe {
            model,
            ep_size,
            nvl_num_gpu,
            routing,
            routing_seed,
            mtp_mode,
            expert_popularity_file,
        } = &parsed
        else {
            panic!("expected glm52_vllm_dsa_moe")
        };
        assert_eq!(model.model_config, "model/config/glm52.json");
        assert!(!model.fp8);
        assert_eq!(*ep_size, 8);
        assert_eq!(*nvl_num_gpu, 8);
        assert_eq!(*routing, RoutingKind::Uniform);
        assert_eq!(*routing_seed, None);
        assert_eq!(*mtp_mode, Glm52MtpMode::Off);
        assert_eq!(*expert_popularity_file, None);
        assert!(std::ptr::eq(parsed.model(), model));
    }

    #[test]
    fn glm52_selector_parses_parallel_routing_and_every_mtp_mode() {
        for (wire, expected) in [
            ("off", Glm52MtpMode::Off),
            ("full_index", Glm52MtpMode::FullIndex),
            ("index_share", Glm52MtpMode::IndexShare),
        ] {
            let parsed = parse_glm52(&format!(
                r#","ep_size":16,"nvl_num_gpu":8,"routing":"random","routing_seed":73,"mtp_mode":"{wire}""#
            ))
            .unwrap();
            assert!(matches!(
                parsed,
                IterArchSel::Glm52VllmDsaMoe {
                    ep_size: 16,
                    nvl_num_gpu: 8,
                    routing: RoutingKind::Random,
                    routing_seed: Some(73),
                    mtp_mode,
                    ..
                } if mtp_mode == expected
            ));
        }
    }

    #[test]
    fn glm52_selector_rejects_malformed_mtp_mode() {
        let error = parse_glm52(r#","mtp_mode":"shared""#).unwrap_err();
        assert!(error.to_string().contains("unknown variant"));
    }

    #[test]
    fn glm52_schema_exposes_exact_defaults_choices_and_cache_keys() {
        let (_, params) = IterArchSel::SCHEMA
            .iter()
            .find(|(tag, _)| *tag == "glm52_vllm_dsa_moe")
            .expect("glm52 selector schema row");
        let params = serde_json::to_value(params).unwrap();
        let get = |name: &str| {
            params
                .as_array()
                .unwrap()
                .iter()
                .find(|param| param["name"] == name)
                .unwrap()
        };
        for name in ["ep_size", "nvl_num_gpu"] {
            let param = get(name);
            assert_eq!(param["default"], 8);
            assert_eq!(param["affects_cache"], true);
        }
        let routing = get("routing");
        assert_eq!(routing["default"], "uniform");
        assert_eq!(
            routing["choices"],
            serde_json::json!(["uniform", "random", "popularity"])
        );
        let mtp = get("mtp_mode");
        assert_eq!(mtp["default"], "off");
        assert_eq!(
            mtp["choices"],
            serde_json::json!(["off", "full_index", "index_share"])
        );
        assert_eq!(mtp["affects_cache"], true);
    }

    #[test]
    fn retired_glm52_selector_is_not_published_or_accepted() {
        assert!(!IterArchSel::SCHEMA
            .iter()
            .any(|(tag, _)| *tag == "glm52_dsa_moe"));
        let raw =
            r#"{"type":"glm52_dsa_moe","model_config":"model/config/glm52.json","fp8":false}"#;
        assert!(serde_json::from_str::<IterArchSel>(raw).is_err());
    }

    fn checked_in(config: &str) -> String {
        format!("{}/model/config/{config}.json", env!("CARGO_MANIFEST_DIR"))
    }

    /// An arch's own `max_model_len` wins; without one, the checkpoint's
    /// `max_position_embeddings`, read from `text_config` for Qwen3.6.
    #[test]
    fn max_model_len_is_the_arch_param_else_the_checkpoints_positions() {
        let limit = |arch: &str, config: &str, extra: &str| {
            let raw = format!(
                r#"{{"type":"{arch}","model_config":"{}","fp8":false{extra}}}"#,
                checked_in(config)
            );
            serde_json::from_str::<IterArchSel>(&raw)
                .unwrap()
                .max_model_len()
                .unwrap()
        };
        assert_eq!(
            limit("llama3_dense_tp", "llama3_8b", r#","tp_size":2"#),
            131_072
        );
        assert_eq!(limit("qwen36_local", "qwen3_6_35b_a3b_fp8", ""), 262_144);
        assert_eq!(
            limit("glm52_vllm_nvfp4_dsa_moe", "glm52_nvfp4", ""),
            1_048_576
        );
        assert_eq!(
            limit(
                "glm52_vllm_nvfp4_dsa_moe",
                "glm52_nvfp4",
                r#","max_model_len":8192"#
            ),
            8192
        );
        assert_eq!(
            limit("glm53_flash_vllm_fp8_kda_dsa_moe", "glm53_flash", ""),
            8192
        );
        // DeepSeek-V4.1 defaults to the checkpoint's 1M positions; the
        // capture presets pin their 131072.
        for arch in [
            "deepseek_v41_vllm",
            "deepseek_v41_vllm_serial_streams",
            "deepseek_v41_vllm_dp_attn_ep_moe",
        ] {
            assert_eq!(limit(arch, "deepseek_v41_flash", ""), 1_048_576);
            assert_eq!(
                limit(arch, "deepseek_v41_flash", r#","max_model_len":131072"#),
                131_072
            );
        }

        let attn = format!(
            r#"{{"type":"qwen3_attn_tp","model_config":"{}","fp8":false,"attn_tp_size":4}}"#,
            checked_in("qwen3_235b")
        );
        let attn: AttnArchSel = serde_json::from_str(&attn).unwrap();
        assert_eq!(attn.max_model_len().unwrap(), 40_960);
    }

    /// DeepSeek-V4.1 takes no `max_model_len` past the trained positions.
    #[test]
    fn deepseek_v41_refuses_a_max_model_len_past_its_positions() {
        let raw = format!(
            r#"{{"type":"deepseek_v41_vllm","model_config":"{}","fp8":true,"max_model_len":1048577}}"#,
            checked_in("deepseek_v41_flash")
        );
        let error = serde_json::from_str::<IterArchSel>(&raw)
            .unwrap()
            .max_model_len()
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("max_model_len must be 1..=1048576"),
            "{error}"
        );
    }

    #[test]
    fn a_checkpoint_without_positions_needs_an_explicit_max_model_len() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, r#"{"hidden_size": 8}"#).unwrap();
        let raw = format!(
            r#"{{"type":"llama3_dense","model_config":"{}","fp8":false}}"#,
            path.display()
        );
        let error = serde_json::from_str::<IterArchSel>(&raw)
            .unwrap()
            .max_model_len()
            .unwrap_err()
            .to_string();
        assert!(error.contains("has no max_position_embeddings"), "{error}");
    }
}
// ── layer-wise attn / ffn contract (AFD) ────────────────────────────────────

#[derive(Debug, Clone, Deserialize, ProviderSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AttnArchSel {
    /// Qwen3-MoE attention side (layer-wise AFD attn pool). **One worker = one DP
    /// shard**: attention sharded over `attn_tp_size` head-parallel ranks, with its
    /// own KV cache and request stream. Data parallelism is the attn pool's
    /// `replicas` (= the unified arch's `ep_size / attn_tp_size`), not an arch
    /// param. Pairs with the `qwen3_ffn_moe` ffn arch.
    Qwen3AttnTp {
        #[serde(flatten)]
        model: ModelSpec,
        /// Attention tensor-parallelism size (heads sharded across these ranks).
        #[param(default = 4, cache_key)]
        attn_tp_size: u16,
    },
}

impl AttnArchSel {
    /// The model identity/dims this arch operates on (every variant carries it).
    pub fn model(&self) -> &ModelSpec {
        match self {
            Self::Qwen3AttnTp { model, .. } => model,
        }
    }

    /// The longest request this attention side serves; see
    /// [`IterArchSel::max_model_len`]. The FFN side holds no context.
    pub fn max_model_len(&self) -> anyhow::Result<u32> {
        match self {
            Self::Qwen3AttnTp { model, .. } => model.max_position_embeddings(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, ProviderSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FfnArchSel {
    /// BF16/native Qwen3-MoE FFN side (layer-wise AFD ffn pool): qkv / o_proj (on the attn-TP
    /// layout) + post_norm + router + EP MoE, plus the iteration embed / final_norm
    /// / lm_head. Mirrors the FFN-side cost of the unified `qwen3_moe_dp_attn_ep_ffn`;
    /// pairs with a `qwen3_attn` model whose `fp8` field is false.
    Qwen3FfnMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Attention tensor-parallelism size. Drives the qkv / o_proj per-rank
        /// shape AND the MoE combine residing-group width (a token resides on its
        /// qkv/o_proj TP group, so combine fans the reduced output back to all
        /// `attn_tp_size` ranks). Must match the paired attn arch's `attn_tp_size`.
        #[param(default = 4, cache_key)]
        attn_tp_size: u16,
        /// Expert-parallelism size (FFN expert ranks; spans the whole replica).
        /// `num_dp_groups = ep_size / attn_tp_size`.
        #[param(default = 8, cache_key)]
        ep_size: u16,
        /// NVLink-domain size — partitions `ep_size` ranks into NVL domains for the
        /// intra/inter split of MoE dispatch/combine.
        #[param(default = 8, cache_key)]
        nvl_num_gpu: u16,
        /// Expert routing distribution: `uniform` (default) or `random` (seeded by
        /// `routing_seed`). Drives the L2 MoE dispatch/combine simulation.
        #[serde(default)]
        #[param(string, default = "uniform", choices = SYNTHETIC_ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        /// Seed for `routing = random` (ignored for `uniform`).
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
    },
    /// Native FP8 Qwen3-MoE FFN side. This is a separate provider because its
    /// pre/post projection and lm-head slots are quant+GEMM compound ops. It
    /// pairs with a `qwen3_attn` model whose `fp8` field is true.
    Qwen3Fp8FfnMoe {
        #[serde(flatten)]
        model: ModelSpec,
        #[param(default = 4, cache_key)]
        attn_tp_size: u16,
        #[param(default = 8, cache_key)]
        ep_size: u16,
        #[param(default = 8, cache_key)]
        nvl_num_gpu: u16,
        #[serde(default)]
        #[param(string, default = "uniform", choices = SYNTHETIC_ROUTING_KINDS, cache_key)]
        routing: RoutingKind,
        #[serde(default)]
        #[param(cache_key)]
        routing_seed: Option<u64>,
    },
}

impl FfnArchSel {
    /// The model identity/dims this arch operates on (every variant carries it).
    pub fn model(&self) -> &ModelSpec {
        match self {
            Self::Qwen3FfnMoe { model, .. } | Self::Qwen3Fp8FfnMoe { model, .. } => model,
        }
    }
}

#[cfg(test)]
mod layerwise_tests {
    use super::*;

    #[test]
    fn unimplemented_layerwise_placeholders_are_not_published_or_accepted() {
        assert!(!AttnArchSel::SCHEMA
            .iter()
            .any(|(tag, _)| *tag == "llama3_attn_tp"));
        assert!(!FfnArchSel::SCHEMA
            .iter()
            .any(|(tag, _)| *tag == "deepseek_ffn_moe"));

        let llama = r#"{"type":"llama3_attn_tp","model_config":"m.json","fp8":false,"tp_size":2,"head_parallel":1}"#;
        let deepseek = r#"{"type":"deepseek_ffn_moe","model_config":"m.json","fp8":false,"tp_size":2,"ep_size":8}"#;
        assert!(serde_json::from_str::<AttnArchSel>(llama).is_err());
        assert!(serde_json::from_str::<FfnArchSel>(deepseek).is_err());
    }
}
