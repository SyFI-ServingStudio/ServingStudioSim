//! Worker (L5) config surface — the worker *selectors*, co-located with the L5
//! workers they pick (L5 design).
//!
//! Each selector is a serde tagged enum (`#[serde(tag = "type")]`), the symmetric
//! sibling of the arch selector. `barebone` / `hp_unified` are wired for
//! `unified`; `pd_prefill` / `pd_decode` are wired for `pd`; the AFD selectors
//! are wired for `afd`. `chunked_prefill` is the hard-capped whole-iteration
//! lifecycle used when long prompts must be split across iterations, and
//! `speculative` is that same lifecycle driving a target-verify decode engine.
//!
//! `#[derive(ProviderSchema)]` emits each selector's `SCHEMA` of `(tag, params)`
//! rows for the launcher; `schema::dump::list_params` aggregates them.

use anyhow::{ensure, Result};
use serde::Deserialize;

use schema_derive::{ParamStruct, ProviderSchema};

use super::admission::PendingOrderKind;
use super::kv::{
    KvOverhead, PrefixCacheConfig, PrefixCacheMode, PrefixCachePolicy, PrefixResumePoints,
};
use super::types::WorkerConfig;

/// Batch-composition policy for the chunked-prefill worker. Closed set → serde
/// enum (kebab-case matches the historical CLI spelling).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BatchPolicy {
    Mix,
    SeparatePrefillPriority,
}

const BATCH_POLICY_CHOICES: [&str; 2] = ["mix", "separate-prefill-priority"];

/// How a multi-partition worker places a fresh request on an attention DP
/// partition. See [`crate::worker::admission::LoadBalance`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DpPlacement {
    #[default]
    RoundRobin,
    VllmLeastLoaded,
}

const DP_PLACEMENT_CHOICES: [&str; 2] = ["round-robin", "vllm-least-loaded"];

/// Where the chunked-prefill worker may end a non-final prefill chunk of an
/// arch whose KV layout keeps a resumable recurrent state. A layout without
/// one has no checkpoint and always chunks plainly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrefillChunkAlignment {
    /// With prefix caching on, end every non-final chunk on the layout's
    /// engine block, as vLLM's Mamba `align` cache mode does
    /// (`Scheduler._mamba_block_aligned_split`).
    #[default]
    Checkpoint,
    /// `min(remaining prompt, token budget)`: no checkpoint alignment.
    Plain,
}

const PREFILL_CHUNK_ALIGNMENT_CHOICES: [&str; 2] = ["checkpoint", "plain"];

/// KV-capacity rule paired with chunked prefill. `FullFootprint` preserves the
/// historical no-retraction lifecycle. `BoundedFuture` uses an explicit
/// future-token estimate and therefore requires decode retraction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvAdmissionPolicy {
    #[default]
    FullFootprint,
    BoundedFuture,
}

const KV_ADMISSION_POLICY_CHOICES: [&str; 2] = ["full-footprint", "bounded-future"];

/// Ordering used when the next physical decode allocation does not fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DecodeRetractionPolicy {
    /// SGLang: fewest emitted tokens first, requeued behind waiting requests.
    Length,
    /// vLLM FCFS: the most recently admitted request first, requeued ahead of
    /// every waiting request.
    Fcfs,
}

const DECODE_RETRACTION_POLICY_CHOICES: [&str; 2] = ["length", "fcfs"];

/// Fully resolved bounded-future scheduler constants. These are mechanics, not
/// calibrated performance values; the SGLang preset states its captured values
/// explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundedFutureKvAdmissionConfig {
    pub page_size: u32,
    pub max_future_tokens: u32,
    pub initial_new_token_ratio: f64,
    pub minimum_new_token_ratio: f64,
    pub new_token_ratio_decay_steps: u32,
    pub retract_decode_steps: u32,
    pub retraction_policy: DecodeRetractionPolicy,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum KvAdmissionConfig {
    #[default]
    FullFootprint,
    BoundedFuture(BoundedFutureKvAdmissionConfig),
}

/// Selector-level KV admission settings for chunked-prefill and speculative workers.
///
/// This struct is flattened into the worker document to preserve the existing
/// YAML surface while keeping the policy and all of its dependent knobs one
/// typed unit from deserialization through deployment construction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize, ParamStruct)]
pub struct KvAdmissionSpec {
    /// Capacity policy for waiting and running requests. Historical
    /// deployments omit it and retain full-footprint reservation.
    #[serde(default)]
    #[param(string, default = "full-footprint", choices = KV_ADMISSION_POLICY_CHOICES)]
    kv_admission_policy: KvAdmissionPolicy,
    /// Bounded-future only: physical KV allocation page size in tokens.
    #[serde(default)]
    kv_page_size: Option<u32>,
    /// Bounded-future only: cap on future output tokens charged per request.
    #[serde(default)]
    kv_max_future_tokens: Option<u32>,
    /// Bounded-future only: new-token ratio at scheduler start/reset.
    #[serde(default)]
    kv_initial_new_token_ratio: Option<f64>,
    /// Bounded-future only: floor reached after successful decode steps.
    #[serde(default)]
    kv_minimum_new_token_ratio: Option<f64>,
    /// Bounded-future only: successful decode steps from initial to floor.
    #[serde(default)]
    kv_new_token_ratio_decay_steps: Option<u32>,
    /// Bounded-future only: output-token horizon used after retraction.
    #[serde(default)]
    kv_retract_decode_steps: Option<u32>,
    /// Bounded-future only: which resident decode is retracted first.
    #[serde(default)]
    #[param(string, choices = DECODE_RETRACTION_POLICY_CHOICES)]
    decode_retraction_policy: Option<DecodeRetractionPolicy>,
}

impl KvAdmissionSpec {
    pub(crate) fn resolve(self) -> Result<KvAdmissionConfig> {
        let knobs_are_absent = self.kv_page_size.is_none()
            && self.kv_max_future_tokens.is_none()
            && self.kv_initial_new_token_ratio.is_none()
            && self.kv_minimum_new_token_ratio.is_none()
            && self.kv_new_token_ratio_decay_steps.is_none()
            && self.kv_retract_decode_steps.is_none()
            && self.decode_retraction_policy.is_none();
        if self.kv_admission_policy == KvAdmissionPolicy::FullFootprint {
            ensure!(
                knobs_are_absent,
                "full-footprint KV admission does not accept bounded-future knobs"
            );
            return Ok(KvAdmissionConfig::FullFootprint);
        }

        let config = BoundedFutureKvAdmissionConfig {
            page_size: self
                .kv_page_size
                .ok_or_else(|| anyhow::anyhow!("bounded-future requires kv_page_size"))?,
            max_future_tokens: self
                .kv_max_future_tokens
                .ok_or_else(|| anyhow::anyhow!("bounded-future requires kv_max_future_tokens"))?,
            initial_new_token_ratio: self.kv_initial_new_token_ratio.ok_or_else(|| {
                anyhow::anyhow!("bounded-future requires kv_initial_new_token_ratio")
            })?,
            minimum_new_token_ratio: self.kv_minimum_new_token_ratio.ok_or_else(|| {
                anyhow::anyhow!("bounded-future requires kv_minimum_new_token_ratio")
            })?,
            new_token_ratio_decay_steps: self.kv_new_token_ratio_decay_steps.ok_or_else(|| {
                anyhow::anyhow!("bounded-future requires kv_new_token_ratio_decay_steps")
            })?,
            retract_decode_steps: self.kv_retract_decode_steps.ok_or_else(|| {
                anyhow::anyhow!("bounded-future requires kv_retract_decode_steps")
            })?,
            retraction_policy: self.decode_retraction_policy.ok_or_else(|| {
                anyhow::anyhow!("bounded-future requires decode_retraction_policy")
            })?,
        };
        ensure!(
            config.page_size == 1,
            "bounded-future currently requires kv_page_size=1; larger pages need persistent allocated-length accounting"
        );
        ensure!(
            config.max_future_tokens > 0,
            "kv_max_future_tokens must be positive"
        );
        ensure!(
            config.initial_new_token_ratio.is_finite()
                && (0.0..=1.0).contains(&config.initial_new_token_ratio),
            "kv_initial_new_token_ratio must be finite and in [0, 1]"
        );
        ensure!(
            config.minimum_new_token_ratio.is_finite()
                && (0.0..=config.initial_new_token_ratio).contains(&config.minimum_new_token_ratio),
            "kv_minimum_new_token_ratio must be finite and in [0, initial]"
        );
        ensure!(
            config.new_token_ratio_decay_steps > 0,
            "kv_new_token_ratio_decay_steps must be positive"
        );
        ensure!(
            config.retract_decode_steps > 0,
            "kv_retract_decode_steps must be positive"
        );
        Ok(KvAdmissionConfig::BoundedFuture(config))
    }
}

const KV_OVERHEAD_CHOICES: [&str; 2] = ["vllm", "none"];
const PREFIX_RESUME_POINTS_CHOICES: [&str; 4] = [
    "prompt_end",
    "response_end",
    "prompt_and_response_end",
    "every_block",
];

/// How a worker turns the arch's `KvCacheLayout` into charges: whether the
/// serving engine's own allocation counts, and where a finished request keeps
/// the state a later request resumes from. Flattened into every selector that
/// owns KV.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, ParamStruct)]
pub struct KvAccountingSpec {
    /// `vllm` charges what vLLM allocates (padding, state copies, engine
    /// blocks, a sliding window's in-flight tokens) and resumes prefixes where
    /// vLLM does; `none` charges only what the model must keep, an upper bound
    /// on capacity and reuse.
    #[serde(default)]
    #[param(string, default = "vllm", choices = KV_OVERHEAD_CHOICES)]
    kv_overhead: KvOverhead,
    /// Where a finished request keeps sliding-window tails and recurrent state
    /// for a later request to resume from. Absent follows `kv_overhead`:
    /// `prompt_end` under `vllm`, `response_end` under `none`. A layout of
    /// per-token parts only resumes anywhere and ignores it.
    #[serde(default)]
    #[param(string, choices = PREFIX_RESUME_POINTS_CHOICES)]
    prefix_resume_points: Option<PrefixResumePoints>,
    /// Also keep that state at every multiple of this many tokens (a
    /// multiple of the layout's engine block).
    #[serde(default)]
    prefix_resume_interval: Option<u32>,
    /// Tokens one step may have in flight, which caps what admission charges
    /// a sliding window. Absent is two steps of `max_batch_tokens` (vLLM's
    /// async scheduling), or the whole prompt without a token budget.
    #[serde(default)]
    kv_max_in_flight_tokens: Option<u32>,
}

impl KvAccountingSpec {
    /// Validate the knobs and set them on `config`.
    pub(crate) fn apply(self, owner: &str, config: &mut WorkerConfig) -> Result<()> {
        ensure!(
            self.prefix_resume_interval != Some(0),
            "{owner}: prefix_resume_interval must be positive"
        );
        ensure!(
            self.kv_max_in_flight_tokens != Some(0),
            "{owner}: kv_max_in_flight_tokens must be positive"
        );
        config.kv_overhead = self.kv_overhead;
        config.prefix_resume_points = self.prefix_resume_points;
        config.prefix_resume_interval = self.prefix_resume_interval;
        config.kv_max_in_flight_tokens = self.kv_max_in_flight_tokens;
        Ok(())
    }
}

const PENDING_ORDER_CHOICES: [&str; 4] = [
    "session-start",
    "fifo",
    "shortest-job-first",
    "longest-prefix-match",
];

const PREFIX_CACHE_POLICY_CHOICES: [&str; 4] = ["lru", "fifo", "lfu", "largest-first"];
const PREFIX_CACHE_MODE_CHOICES: [&str; 2] = ["disabled", "opportunistic"];

// ── iter-wise contract (unified, pd) ────────────────────────────────────────

/// serde/param fallback for every worker selector's `gpu_time_multiplier`: 1.0
/// = no inter-kernel overhead (kernel-folded time IS the wall time), so presets
/// that omit the field keep their prior behavior. A worker scales the wall time
/// it advances the clock by as `kernel_time * gpu_time_multiplier` (≥ 1.0);
/// cost_log / manifest stay pre-scale (pure kernel), so the overhead surfaces as
/// a gap between iter/section slices in the trace, never inside a kernel slice.
fn default_gpu_time_multiplier() -> f64 {
    1.0
}

/// serde fallback for the speculative selector's `batch_policy`.
const fn default_batch_policy() -> BatchPolicy {
    BatchPolicy::Mix
}

/// Iteration-wise worker provider.
#[derive(Debug, Clone, Deserialize, ProviderSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IterWorkerSel {
    Barebone {
        /// GPU memory for the worker (GB; primarily KV cache budget).
        #[param(default = 80.0)]
        attn_gpu_memory_gb: f64,
        /// How the arch's KV layout is charged; see [`KvAccountingSpec`].
        #[serde(flatten)]
        kv_accounting: KvAccountingSpec,
        /// GPU wall/kernel time multiplier (≥ 1.0); models inter-kernel overhead
        /// (see [`default_gpu_time_multiplier`]). cost_log stays pre-scale.
        #[serde(default = "default_gpu_time_multiplier")]
        #[param(default = 1.0)]
        gpu_time_multiplier: f64,
        /// Optional per-iteration token budget. When set, admission reserves the
        /// budget for the live decodes (1 tok/req) first, then admits whole
        /// prefills (in `pending_order`) until the
        /// remaining budget is exhausted; a single over-long prefill is still
        /// force-admitted when the group holds budget but nothing yet. None =
        /// unbounded (`u32::MAX` internally). (Distinct from
        /// `ChunkedPrefill::max_batch_tokens`, which is a hard cap that chunks
        /// prefills to fit; this budget is soft — one whole prefill may exceed it.)
        #[serde(default)]
        max_batch_tokens: Option<u32>,
        /// Order in which queued requests are offered to the admission gates.
        #[serde(default)]
        #[param(string, default = "session-start", choices = PENDING_ORDER_CHOICES)]
        pending_order: PendingOrderKind,
        /// Whether completed-session KV may reuse currently idle attention KV.
        #[serde(default)]
        #[param(string, default = "opportunistic", choices = PREFIX_CACHE_MODE_CHOICES)]
        prefix_cache_mode: PrefixCacheMode,
        /// Victim policy used only when retained session KV is enabled.
        #[serde(default)]
        #[param(string, default = "lru", choices = PREFIX_CACHE_POLICY_CHOICES)]
        prefix_cache_policy: PrefixCachePolicy,
        /// Optional ceiling (GB) for retained session KV inside the same
        /// attention budget. None uses all dynamically available slack.
        #[serde(default)]
        prefix_cache_max_gpu_memory_gb: Option<f64>,
        /// Multiplier (≥ 1.0) for iterations that schedule any prefill tokens.
        /// See [`IterWorkerSel::ChunkedPrefill`]'s field of the same name.
        #[serde(default)]
        prefill_gpu_time_multiplier: Option<f64>,
    },
    /// Multi-group HP/DP worker: maintains one KV partition state per attention
    /// DP shard (count comes from the arch's `num_attn_dp_groups`). Pairs with a
    /// DP-attention arch such as `llama3_dp_attn_tp_ffn`.
    HpUnified {
        /// GPU memory for the worker (GB; primarily KV cache budget). Sizes each
        /// DP shard's KV pool.
        #[param(default = 80.0)]
        attn_gpu_memory_gb: f64,
        /// How the arch's KV layout is charged; see [`KvAccountingSpec`].
        #[serde(flatten)]
        kv_accounting: KvAccountingSpec,
        /// GPU wall/kernel time multiplier (≥ 1.0); models inter-kernel overhead
        /// (see [`default_gpu_time_multiplier`]). cost_log stays pre-scale.
        #[serde(default = "default_gpu_time_multiplier")]
        #[param(default = 1.0)]
        gpu_time_multiplier: f64,
        /// Optional per-iteration token budget, applied PER DP group (each shard
        /// reserves its own live decodes then fills its own remainder). See
        /// `Barebone::max_batch_tokens`. None = unbounded (`u32::MAX`
        /// internally).
        #[serde(default)]
        max_batch_tokens: Option<u32>,
        /// Order in which queued requests are offered to the admission gates.
        /// `session-start` favours long-lived conversations (and so the largest
        /// retained prefixes); `fifo` is plain request arrival order.
        #[serde(default)]
        #[param(string, default = "session-start", choices = PENDING_ORDER_CHOICES)]
        pending_order: PendingOrderKind,
        /// Whether completed-session KV may reuse each partition's idle KV.
        #[serde(default)]
        #[param(string, default = "opportunistic", choices = PREFIX_CACHE_MODE_CHOICES)]
        prefix_cache_mode: PrefixCacheMode,
        /// Victim policy used only when retained session KV is enabled.
        #[serde(default)]
        #[param(string, default = "lru", choices = PREFIX_CACHE_POLICY_CHOICES)]
        prefix_cache_policy: PrefixCachePolicy,
        /// Optional per-partition retained-prefix ceiling (GB). None uses all
        /// dynamically available attention KV slack.
        #[serde(default)]
        prefix_cache_max_gpu_memory_gb: Option<f64>,
        /// Multiplier (≥ 1.0) for iterations that schedule any prefill tokens.
        /// See [`IterWorkerSel::ChunkedPrefill`]'s field of the same name.
        #[serde(default)]
        prefill_gpu_time_multiplier: Option<f64>,
    },
    ChunkedPrefill {
        /// GPU memory for the worker (GB; primarily KV cache budget).
        #[param(default = 80.0)]
        attn_gpu_memory_gb: f64,
        /// How the arch's KV layout is charged; see [`KvAccountingSpec`].
        #[serde(flatten)]
        kv_accounting: KvAccountingSpec,
        /// Chunked-prefill cap: max tokens per batch.
        max_batch_tokens: u32,
        /// How resident decode shares an iteration with chunked prefill.
        /// `mix` charges both to the same token budget. Under
        /// `separate-prefill-priority`, any runnable prefill makes that
        /// partition's iteration prefill-only; resident decode resumes when no
        /// prefill batch can run.
        #[param(string, default = "mix", choices = BATCH_POLICY_CHOICES)]
        batch_policy: BatchPolicy,
        /// KV capacity and decode-retraction policy, flattened for YAML
        /// compatibility but kept as one typed selector component.
        #[serde(flatten)]
        kv_admission: KvAdmissionSpec,
        /// GPU wall/kernel time multiplier (≥ 1.0); models inter-kernel overhead
        /// (see [`default_gpu_time_multiplier`]). cost_log stays pre-scale.
        #[serde(default = "default_gpu_time_multiplier")]
        #[param(default = 1.0)]
        gpu_time_multiplier: f64,
        /// Multiplier (≥ 1.0) for iterations that schedule any prefill tokens,
        /// replacing `gpu_time_multiplier` there. A calibration knob: an engine
        /// whose decode steps replay CUDA graphs but whose prefill runs eager
        /// pieces leaves the GPU idle only in prefill-bearing iterations, which
        /// one multiplier cannot express. None applies `gpu_time_multiplier` to
        /// every iteration.
        #[serde(default)]
        prefill_gpu_time_multiplier: Option<f64>,
        /// How new requests are spread across the arch's attention DP
        /// partitions when it has more than one (`num_attn_dp_groups`).
        /// `round-robin` rotates blindly; `vllm-least-loaded` is vLLM's DP load
        /// balancer (fewest waiting + running, waiting penalised under KV
        /// pressure). Retained-prefix affinity overrides either.
        #[serde(default)]
        #[param(string, default = "round-robin", choices = DP_PLACEMENT_CHOICES)]
        dp_placement: DpPlacement,
        /// Archs whose KV layout keeps a resumable recurrent state only:
        /// `checkpoint` ends non-final prefill chunks on the layout's engine
        /// block (vLLM's Mamba `align` mode); `plain` chunks as
        /// `min(remaining, budget)`. Absent follows `kv_overhead`: `vllm`
        /// aligns, `none` does not.
        #[serde(default)]
        #[param(string, choices = PREFILL_CHUNK_ALIGNMENT_CHOICES)]
        prefill_chunk_alignment: Option<PrefillChunkAlignment>,
    },
    /// Chunked prefill with a speculating decode engine: one verify pass per
    /// iteration submits `draft_tokens + 1` rows per resident decode, fewer for a
    /// request clipped at `max_model_len`, and retires the target's own token
    /// plus the leading run of accepted drafts.
    ///
    /// A separate selector from `chunked_prefill` rather than a flag on it: the
    /// arch must also be the speculative one (a different model type, not the
    /// ordinary one with a switch), and the pair is validated together.
    Speculative {
        /// GPU memory for the worker (GB; primarily KV cache budget).
        #[param(default = 80.0)]
        attn_gpu_memory_gb: f64,
        /// How the arch's KV layout is charged; see [`KvAccountingSpec`].
        #[serde(flatten)]
        kv_accounting: KvAccountingSpec,
        /// Chunked-prefill cap: max tokens per batch. Decode spends it in query
        /// rows, so a resident decode costs the whole verify width here, and
        /// every scheduled request also costs the arch's drafting slots
        /// (DFlash: `draft_tokens`; MTP: none).
        max_batch_tokens: u32,
        /// Candidate positions drafted per request per iteration. Must equal the
        /// arch selector's `draft_tokens`: it picks the profiled verify shape.
        #[param(default = 5, cache_key)]
        draft_tokens: u32,
        /// Seed for the per-iteration acceptance draws. None uses `0`; this
        /// selects which deterministic stream, not whether there is one.
        #[serde(default)]
        acceptance_seed: Option<u64>,
        /// How resident decode shares an iteration with chunked prefill. See
        /// [`IterWorkerSel::ChunkedPrefill`]. Defaults to mixed batches;
        /// separate-prefill-priority also works and defers resident decode
        /// while a prefill batch runs.
        #[serde(default = "default_batch_policy")]
        #[param(string, default = "mix", choices = BATCH_POLICY_CHOICES)]
        batch_policy: BatchPolicy,
        /// KV capacity and decode-retraction policy. See
        /// [`IterWorkerSel::ChunkedPrefill`]; bounded-future sizes each decode's
        /// next-step headroom at the verify width.
        #[serde(flatten)]
        kv_admission: KvAdmissionSpec,
        /// GPU wall/kernel time multiplier (≥ 1.0); models inter-kernel overhead
        /// (see [`default_gpu_time_multiplier`]). cost_log stays pre-scale.
        #[serde(default = "default_gpu_time_multiplier")]
        #[param(default = 1.0)]
        gpu_time_multiplier: f64,
        /// Multiplier (≥ 1.0) for iterations that schedule any prefill tokens.
        /// See [`IterWorkerSel::ChunkedPrefill`]'s field of the same name.
        #[serde(default)]
        prefill_gpu_time_multiplier: Option<f64>,
        /// Chunk-end alignment for archs whose KV layout keeps a resumable
        /// recurrent state. See [`IterWorkerSel::ChunkedPrefill`]'s field of the
        /// same name; the arch's drafter additionally decides whether the
        /// prompt's last checkpoint sits one block early.
        #[serde(default)]
        #[param(string, choices = PREFILL_CHUNK_ALIGNMENT_CHOICES)]
        prefill_chunk_alignment: Option<PrefillChunkAlignment>,
    },
    /// Pipeline-parallel stage worker for the `pp` deployment: stage 0 admits
    /// requests and owns the pipeline's KV, later stages pull activations and
    /// compute. vLLM V1 chunked prefill under pipeline parallelism: a prompt's
    /// next chunk may be scheduled while its previous chunk is still on a later
    /// stage, and at most one microbatch per stage is in flight. A running
    /// request decodes one step at a time: its next step waits for the previous
    /// step's microbatch to leave the last stage.
    PipelineChunkedPrefill {
        /// GPU memory for each stage (GB; primarily KV cache budget). Every stage
        /// holds the same tokens, so the stage with the most KV bytes per token
        /// sets the pipeline's token capacity.
        #[param(default = 80.0)]
        attn_gpu_memory_gb: f64,
        /// How the arch's KV layout is charged; see [`KvAccountingSpec`].
        #[serde(flatten)]
        kv_accounting: KvAccountingSpec,
        /// Chunked-prefill cap: max tokens per microbatch.
        max_batch_tokens: u32,
        /// GPU wall/kernel time multiplier (≥ 1.0); models inter-kernel overhead
        /// (see [`default_gpu_time_multiplier`]). cost_log stays pre-scale.
        #[serde(default = "default_gpu_time_multiplier")]
        #[param(default = 1.0)]
        gpu_time_multiplier: f64,
        /// Archs whose KV layout keeps a resumable recurrent state only:
        /// `checkpoint` ends non-final prefill chunks on the layout's engine
        /// block (vLLM's Mamba `align` mode); `plain` chunks as
        /// `min(remaining, budget)`. Absent follows `kv_overhead`: `vllm`
        /// aligns, `none` does not.
        #[serde(default)]
        #[param(string, choices = PREFILL_CHUNK_ALIGNMENT_CHOICES)]
        prefill_chunk_alignment: Option<PrefillChunkAlignment>,
        /// Cap each microbatch at `ceil(resident decodes / pp_size)` decodes so
        /// running requests split across the in-flight microbatches. Off is
        /// vLLM: every ready decode joins the next microbatch, so requests that
        /// become ready together stay in one microbatch.
        #[serde(default)]
        #[param(default = false)]
        balance_decode_microbatches: bool,
    },
    /// PD prefill half: prefills then hands off to a decode pool (no local decode).
    PdPrefill {
        /// GPU memory for the worker (GB; primarily KV cache budget).
        #[param(default = 80.0)]
        attn_gpu_memory_gb: f64,
        /// How the arch's KV layout is charged; see [`KvAccountingSpec`].
        #[serde(flatten)]
        kv_accounting: KvAccountingSpec,
        /// GPU wall/kernel time multiplier (≥ 1.0); models inter-kernel overhead
        /// (see [`default_gpu_time_multiplier`]). cost_log stays pre-scale.
        #[serde(default = "default_gpu_time_multiplier")]
        #[param(default = 1.0)]
        gpu_time_multiplier: f64,
        /// Whether completed-session KV may reuse idle prefill-worker KV.
        #[serde(default)]
        #[param(string, default = "opportunistic", choices = PREFIX_CACHE_MODE_CHOICES)]
        prefix_cache_mode: PrefixCacheMode,
        /// Victim policy used only when retained session KV is enabled.
        #[serde(default)]
        #[param(string, default = "lru", choices = PREFIX_CACHE_POLICY_CHOICES)]
        prefix_cache_policy: PrefixCachePolicy,
        /// Optional retained-prefix ceiling (GB). None uses all dynamically
        /// available attention KV slack.
        #[serde(default)]
        prefix_cache_max_gpu_memory_gb: Option<f64>,
    },
    /// PD decode half: admits already-prefilled requests straight into decode.
    PdDecode {
        /// GPU memory for the worker (GB; primarily KV cache budget).
        #[param(default = 80.0)]
        attn_gpu_memory_gb: f64,
        /// How the arch's KV layout is charged; see [`KvAccountingSpec`].
        #[serde(flatten)]
        kv_accounting: KvAccountingSpec,
        /// GPU wall/kernel time multiplier (≥ 1.0); models inter-kernel overhead
        /// (see [`default_gpu_time_multiplier`]). cost_log stays pre-scale.
        #[serde(default = "default_gpu_time_multiplier")]
        #[param(default = 1.0)]
        gpu_time_multiplier: f64,
    },
}

impl IterWorkerSel {
    /// The selector's KV accounting knobs; every iteration-wise worker owns KV.
    pub(crate) fn kv_accounting(&self) -> KvAccountingSpec {
        match self {
            Self::Barebone { kv_accounting, .. }
            | Self::HpUnified { kv_accounting, .. }
            | Self::ChunkedPrefill { kv_accounting, .. }
            | Self::Speculative { kv_accounting, .. }
            | Self::PipelineChunkedPrefill { kv_accounting, .. }
            | Self::PdPrefill { kv_accounting, .. }
            | Self::PdDecode { kv_accounting, .. } => *kv_accounting,
        }
    }

    /// The chunk alignment a chunked selector asks for; `None` follows the
    /// worker's `kv_overhead`.
    pub(crate) fn prefill_chunk_alignment(&self) -> Option<PrefillChunkAlignment> {
        match self {
            Self::ChunkedPrefill {
                prefill_chunk_alignment,
                ..
            }
            | Self::Speculative {
                prefill_chunk_alignment,
                ..
            }
            | Self::PipelineChunkedPrefill {
                prefill_chunk_alignment,
                ..
            } => *prefill_chunk_alignment,
            _ => None,
        }
    }
}

// ── layer-wise attn / ffn contract (afd) ────────────────────────────────────

#[derive(Debug, Clone, Deserialize, ProviderSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AttnWorkerSel {
    DisaggAttn {
        /// GPU memory for the attention worker (GB; KV cache budget).
        #[param(default = 80.0)]
        attn_gpu_memory_gb: f64,
        /// How the arch's KV layout is charged; see [`KvAccountingSpec`].
        #[serde(flatten)]
        kv_accounting: KvAccountingSpec,
        /// GPU wall/kernel time multiplier (≥ 1.0); models inter-kernel overhead
        /// (see [`default_gpu_time_multiplier`]). cost_log stays pre-scale.
        #[serde(default = "default_gpu_time_multiplier")]
        #[param(default = 1.0)]
        gpu_time_multiplier: f64,
        /// Whether completed-session KV may reuse idle attention-worker KV.
        #[serde(default)]
        #[param(string, default = "opportunistic", choices = PREFIX_CACHE_MODE_CHOICES)]
        prefix_cache_mode: PrefixCacheMode,
        /// Victim policy used only when retained session KV is enabled.
        #[serde(default)]
        #[param(string, default = "lru", choices = PREFIX_CACHE_POLICY_CHOICES)]
        prefix_cache_policy: PrefixCachePolicy,
        /// Optional retained-prefix ceiling (GB). None uses all dynamically
        /// available attention KV slack.
        #[serde(default)]
        prefix_cache_max_gpu_memory_gb: Option<f64>,
    },
}

#[derive(Debug, Clone, Deserialize, ProviderSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FfnWorkerSel {
    DisaggFfn {
        /// GPU wall/kernel time multiplier (≥ 1.0); models inter-kernel overhead
        /// (see [`default_gpu_time_multiplier`]). cost_log stays pre-scale.
        #[serde(default = "default_gpu_time_multiplier")]
        #[param(default = 1.0)]
        gpu_time_multiplier: f64,
    },
}

/// Validate the selector-level knobs once and lower them into the runtime KV
/// contract shared by every attention-bearing worker family.
pub(crate) fn resolve_prefix_cache_config(
    owner: &str,
    mode: PrefixCacheMode,
    policy: PrefixCachePolicy,
    max_gpu_memory_gb: Option<f64>,
    attn_gpu_memory_gb: f64,
) -> Result<PrefixCacheConfig> {
    ensure!(
        attn_gpu_memory_gb.is_finite() && attn_gpu_memory_gb > 0.0,
        "{owner}: attn_gpu_memory_gb must be finite and > 0, got {attn_gpu_memory_gb}"
    );

    match mode {
        PrefixCacheMode::Disabled => {
            ensure!(
                max_gpu_memory_gb.is_none(),
                "{owner}: prefix_cache_max_gpu_memory_gb must be absent when prefix_cache_mode is disabled"
            );
            ensure!(
                policy == PrefixCachePolicy::Lru,
                "{owner}: prefix_cache_policy is only meaningful when prefix_cache_mode is opportunistic"
            );
            Ok(PrefixCacheConfig::Disabled)
        }
        PrefixCacheMode::Opportunistic => {
            if let Some(max_gpu_memory_gb) = max_gpu_memory_gb {
                ensure!(
                    max_gpu_memory_gb.is_finite()
                        && max_gpu_memory_gb > 0.0
                        && max_gpu_memory_gb <= attn_gpu_memory_gb,
                    "{owner}: prefix_cache_max_gpu_memory_gb must be finite, > 0, and <= \
                     attn_gpu_memory_gb ({attn_gpu_memory_gb}), got {max_gpu_memory_gb}"
                );
            }
            Ok(PrefixCacheConfig::Opportunistic {
                policy,
                max_retained_bytes: max_gpu_memory_gb.map(|memory_gb| (memory_gb * 1e9) as u64),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_defaults_to_uncapped_opportunistic_prefix_reuse() {
        let worker: IterWorkerSel = serde_yaml::from_str(
            "type: barebone\nattn_gpu_memory_gb: 80.0\ngpu_time_multiplier: 1.0\n",
        )
        .expect("parse barebone worker");

        assert!(matches!(
            worker,
            IterWorkerSel::Barebone {
                prefix_cache_mode: PrefixCacheMode::Opportunistic,
                prefix_cache_policy: PrefixCachePolicy::Lru,
                prefix_cache_max_gpu_memory_gb: None,
                ..
            }
        ));
    }

    #[test]
    fn resolver_keeps_disabled_and_opportunistic_contracts_distinct() {
        assert_eq!(
            resolve_prefix_cache_config(
                "test",
                PrefixCacheMode::Disabled,
                PrefixCachePolicy::Lru,
                None,
                80.0,
            )
            .expect("disabled baseline is valid"),
            PrefixCacheConfig::Disabled
        );
        assert_eq!(
            resolve_prefix_cache_config(
                "test",
                PrefixCacheMode::Opportunistic,
                PrefixCachePolicy::Lfu,
                Some(12.5),
                80.0,
            )
            .expect("bounded opportunistic cache is valid"),
            PrefixCacheConfig::Opportunistic {
                policy: PrefixCachePolicy::Lfu,
                max_retained_bytes: Some(12_500_000_000),
            }
        );
    }

    #[test]
    fn resolver_rejects_dead_or_out_of_budget_knobs() {
        assert!(resolve_prefix_cache_config(
            "test",
            PrefixCacheMode::Disabled,
            PrefixCachePolicy::Lru,
            Some(1.0),
            80.0,
        )
        .is_err());
        assert!(resolve_prefix_cache_config(
            "test",
            PrefixCacheMode::Disabled,
            PrefixCachePolicy::Fifo,
            None,
            80.0,
        )
        .is_err());
        assert!(resolve_prefix_cache_config(
            "test",
            PrefixCacheMode::Opportunistic,
            PrefixCachePolicy::Lru,
            Some(81.0),
            80.0,
        )
        .is_err());
    }

    #[test]
    fn full_footprint_rejects_bounded_future_knobs() {
        assert!(KvAdmissionSpec {
            kv_page_size: Some(1),
            ..KvAdmissionSpec::default()
        }
        .resolve()
        .is_err());
    }

    #[test]
    fn bounded_future_requires_and_preserves_every_source_constant() {
        let spec = KvAdmissionSpec {
            kv_admission_policy: KvAdmissionPolicy::BoundedFuture,
            kv_page_size: Some(1),
            kv_max_future_tokens: Some(4_096),
            kv_initial_new_token_ratio: Some(0.7),
            kv_minimum_new_token_ratio: Some(0.098),
            kv_new_token_ratio_decay_steps: Some(600),
            kv_retract_decode_steps: Some(20),
            decode_retraction_policy: Some(DecodeRetractionPolicy::Length),
        };
        let resolved = spec
            .resolve()
            .expect("complete bounded-future policy should resolve");

        assert_eq!(
            resolved,
            KvAdmissionConfig::BoundedFuture(BoundedFutureKvAdmissionConfig {
                page_size: 1,
                max_future_tokens: 4_096,
                initial_new_token_ratio: 0.7,
                minimum_new_token_ratio: 0.098,
                new_token_ratio_decay_steps: 600,
                retract_decode_steps: 20,
                retraction_policy: DecodeRetractionPolicy::Length,
            })
        );
        assert!(KvAdmissionSpec {
            kv_retract_decode_steps: None,
            ..spec
        }
        .resolve()
        .is_err());
        assert!(KvAdmissionSpec {
            kv_page_size: Some(16),
            ..spec
        }
        .resolve()
        .is_err());
    }

    #[test]
    fn chunked_prefill_keeps_the_flat_kv_admission_yaml_contract() {
        let worker: IterWorkerSel = serde_yaml::from_str(
            "type: chunked_prefill\n\
             attn_gpu_memory_gb: 80.0\n\
             max_batch_tokens: 2048\n\
             batch_policy: separate-prefill-priority\n\
             kv_admission_policy: bounded-future\n\
             kv_page_size: 1\n\
             kv_max_future_tokens: 4096\n\
             kv_initial_new_token_ratio: 0.7\n\
             kv_minimum_new_token_ratio: 0.098\n\
             kv_new_token_ratio_decay_steps: 600\n\
             kv_retract_decode_steps: 20\n\
             decode_retraction_policy: length\n",
        )
        .expect("parse the established flat chunked-prefill YAML");

        let IterWorkerSel::ChunkedPrefill { kv_admission, .. } = worker else {
            panic!("expected chunked-prefill selector");
        };
        assert!(matches!(
            kv_admission.resolve(),
            Ok(KvAdmissionConfig::BoundedFuture(
                BoundedFutureKvAdmissionConfig {
                    page_size: 1,
                    max_future_tokens: 4_096,
                    initial_new_token_ratio: 0.7,
                    minimum_new_token_ratio: 0.098,
                    new_token_ratio_decay_steps: 600,
                    retract_decode_steps: 20,
                    retraction_policy: DecodeRetractionPolicy::Length,
                }
            ))
        ));
    }
}
