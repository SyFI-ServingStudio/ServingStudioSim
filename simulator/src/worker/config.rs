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
use super::kv::{PrefixCacheConfig, PrefixCacheMode, PrefixCachePolicy};

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

/// Where the chunked-prefill worker may end a hybrid (recurrent +
/// full-attention) arch's non-final prefill chunk. A pure full-attention arch
/// has no recurrent checkpoint and always chunks plainly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrefillChunkAlignment {
    /// With prefix caching on, end every non-final chunk on the arch's
    /// recurrent checkpoint interval, as vLLM's Mamba `align` cache mode does
    /// (`Scheduler._mamba_block_aligned_split`).
    #[default]
    Checkpoint,
    /// `min(remaining prompt, token budget)`: no checkpoint alignment.
    Plain,
}

const PREFILL_CHUNK_ALIGNMENT_CHOICES: [&str; 2] = ["checkpoint", "plain"];

/// How a pipeline head sizes each microbatch's prefill.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MicrobatchSplit {
    /// Take as much pending prefill as `max_batch_tokens` allows (vLLM).
    #[default]
    Greedy,
    /// Take one `pp_size`-th of the round: `clamp(ceil((pending + in-flight
    /// prefill tokens) / pp_size), min_microbatch_tokens, max_batch_tokens)`.
    Even,
}

const MICROBATCH_SPLIT_CHOICES: [&str; 2] = ["greedy", "even"];

/// Validated microbatch prefill sizing the pipeline-head recipe applies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MicrobatchSizing {
    #[default]
    Greedy,
    Even {
        min_tokens: u32,
    },
}

/// Lower the `microbatch_split` / `min_microbatch_tokens` selector pair. The
/// minimum only floors the even target: alone it would be a hold gate, which
/// can leave a prompt's last small chunk waiting while other work flows.
pub(crate) fn resolve_microbatch_sizing(
    split: MicrobatchSplit,
    min_microbatch_tokens: u32,
    max_batch_tokens: u32,
) -> Result<MicrobatchSizing> {
    match split {
        MicrobatchSplit::Greedy => {
            ensure!(
                min_microbatch_tokens == 0,
                "min_microbatch_tokens ({min_microbatch_tokens}) only applies with \
                 microbatch_split: even; greedy takes every pending token it can"
            );
            Ok(MicrobatchSizing::Greedy)
        }
        MicrobatchSplit::Even => {
            ensure!(
                min_microbatch_tokens <= max_batch_tokens,
                "min_microbatch_tokens ({min_microbatch_tokens}) must be <= \
                 max_batch_tokens ({max_batch_tokens})"
            );
            Ok(MicrobatchSizing::Even {
                min_tokens: min_microbatch_tokens,
            })
        }
    }
}

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

const PENDING_ORDER_CHOICES: [&str; 5] = [
    "session-start",
    "fifo",
    "shortest-job-first",
    "shortest-prefill-first",
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

/// serde fallback for the pipeline selector's `pending_order`: vLLM's FCFS.
fn default_pipeline_pending_order() -> PendingOrderKind {
    PendingOrderKind::Fifo
}

/// serde fallbacks for the pipeline selector's prefix tiers: a PCIe Gen5 x16
/// link from host memory, and one Gen5 NVMe drive per GPU.
/// The DRAM and SSD tiers a worker selector asks for (`[dram, ssd]`).
pub(crate) fn prefix_tier_specs(
    deployment: &str,
    dram_gb: f64,
    dram_gb_per_s: f64,
    ssd_gb: f64,
    ssd_gb_per_s: f64,
) -> anyhow::Result<[Option<crate::worker::kv::PrefixTierSpec>; 2]> {
    let tier = |name, gb: f64, gb_per_s: f64| {
        anyhow::ensure!(
            gb >= 0.0 && (gb == 0.0 || gb_per_s > 0.0),
            "{deployment}: {name}_tier_gb must be >= 0 and its bandwidth positive"
        );
        Ok((gb > 0.0).then_some(crate::worker::kv::PrefixTierSpec {
            name,
            capacity_gb_per_gpu: gb,
            read_gb_per_s_per_gpu: gb_per_s,
        }))
    };
    Ok([
        tier("dram", dram_gb, dram_gb_per_s)?,
        tier("ssd", ssd_gb, ssd_gb_per_s)?,
    ])
}

fn default_dram_tier_gb_per_s() -> f64 {
    50.0
}
fn default_ssd_tier_gb_per_s() -> f64 {
    10.0
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
        /// Hybrid (recurrent + full-attention) archs only: context-token spacing
        /// of resumable SSM snapshots, i.e. vLLM's aligned hybrid block size.
        /// None uses the value the arch derives from its own layer geometry.
        /// Set it to reproduce a specific vLLM deployment's alignment, or sweep
        /// it to study the reuse-granularity / state-capacity trade-off. Ignored
        /// by a pure full-attention arch, which has no recurrent state.
        #[serde(default)]
        ssm_checkpoint_interval_tokens: Option<u32>,
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
        /// Hybrid archs only: `checkpoint` ends non-final prefill chunks on the
        /// recurrent checkpoint interval (vLLM's Mamba `align` mode); `plain`
        /// chunks as `min(remaining, budget)`, leaving that engine artifact out.
        #[serde(default)]
        #[param(string, default = "checkpoint", choices = PREFILL_CHUNK_ALIGNMENT_CHOICES)]
        prefill_chunk_alignment: PrefillChunkAlignment,
        /// vLLM's `long_prefill_token_threshold`: the most prefill tokens one
        /// request takes per iteration, so a long prompt leaves the rest of the
        /// budget to other requests (`0`: no cap, vLLM's default). Applies per
        /// attention-DP partition.
        #[serde(default)]
        #[param(default = 0)]
        long_prefill_token_threshold: u32,
        /// Order of each rank's waiting queue. `fifo` is vLLM's FCFS;
        /// `shortest-prefill-first` offers the prompt with the fewest tokens to
        /// compute first, counting what HBM or a prefix tier holds as reused
        /// (not vLLM). Started prompts keep priority either way.
        #[serde(default = "default_pipeline_pending_order")]
        #[param(string, default = "fifo", choices = PENDING_ORDER_CHOICES)]
        pending_order: PendingOrderKind,
        /// Host DRAM behind each attention DP rank's HBM prefix cache, per GPU
        /// (GB; `0`: none), as `pipeline_chunked_prefill`'s; hybrid archs only.
        /// Sessions then stick to the rank they last ran on. Not vLLM.
        #[serde(default)]
        #[param(default = 0.0)]
        dram_tier_gb: f64,
        #[serde(default = "default_dram_tier_gb_per_s")]
        #[param(default = 50.0)]
        dram_tier_gb_per_s: f64,
        /// Local SSD behind DRAM, per GPU (GB; `0`: none).
        #[serde(default)]
        #[param(default = 0.0)]
        ssd_tier_gb: f64,
        #[serde(default = "default_ssd_tier_gb_per_s")]
        #[param(default = 10.0)]
        ssd_tier_gb_per_s: f64,
        /// Decode runs elsewhere, as `pipeline_chunked_prefill`'s; hybrid archs
        /// only. Not vLLM.
        /// With a tier: a session request whose declared prefix exceeds every
        /// context the run stored for that session is a conversation that began
        /// before the run, and its context is read from the slowest tier, as a
        /// long-lived session's would be in steady state. For closed-loop traces
        /// that start sessions mid-life (`trace/session_closed_loop.py`).
        #[serde(default)]
        #[param(default = false)]
        prefix_tier_warm_start: bool,
        #[serde(default)]
        #[param(default = false)]
        external_decode: bool,
    },
    /// Chunked prefill with a speculating decode engine: one verify pass per
    /// iteration submits `draft_tokens + 1` rows per resident decode and retires
    /// the target's own token plus the leading run of accepted drafts.
    ///
    /// A separate selector from `chunked_prefill` rather than a flag on it: the
    /// arch must also be the speculative one (a different model type, not the
    /// ordinary one with a switch), and the pair is validated together.
    Speculative {
        /// GPU memory for the worker (GB; primarily KV cache budget).
        #[param(default = 80.0)]
        attn_gpu_memory_gb: f64,
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
        /// Chunked-prefill cap: max tokens per microbatch.
        max_batch_tokens: u32,
        /// GPU wall/kernel time multiplier (≥ 1.0); models inter-kernel overhead
        /// (see [`default_gpu_time_multiplier`]). cost_log stays pre-scale.
        #[serde(default = "default_gpu_time_multiplier")]
        #[param(default = 1.0)]
        gpu_time_multiplier: f64,
        /// Hybrid archs only: `checkpoint` is vLLM's Mamba `align` mode, which
        /// ends non-final prefill chunks on the KV block and floors prefix hits
        /// to it; `plain` chunks as `min(remaining, budget)` and reuses a
        /// retained prefix exactly (one state at its end), leaving both engine
        /// artifacts out.
        #[serde(default)]
        #[param(string, default = "checkpoint", choices = PREFILL_CHUNK_ALIGNMENT_CHOICES)]
        prefill_chunk_alignment: PrefillChunkAlignment,
        /// Cap each microbatch at `ceil(resident decodes / pp_size)` decodes so
        /// running requests split across the in-flight microbatches. Off is
        /// vLLM: every ready decode joins the next microbatch, so requests that
        /// become ready together stay in one microbatch.
        #[serde(default)]
        #[param(default = false)]
        balance_decode_microbatches: bool,
        /// `greedy` (vLLM) fills each microbatch's prefill up to
        /// `max_batch_tokens`; `even` sizes it to one `pp_size`-th of the
        /// pending plus in-flight prefill, so a backlog spreads over the
        /// pipeline instead of one full microbatch followed by small ones.
        #[serde(default)]
        #[param(string, default = "greedy", choices = MICROBATCH_SPLIT_CHOICES)]
        microbatch_split: MicrobatchSplit,
        /// Floor on the `even` target, in prefill tokens (`0`: none). Valid
        /// only with `microbatch_split: even`; never delays a microbatch.
        #[serde(default)]
        #[param(default = 0)]
        min_microbatch_tokens: u32,
        /// vLLM's `long_prefill_token_threshold`: the most prefill tokens one
        /// request takes per microbatch (`0`: no cap, vLLM's default). Composes
        /// with `greedy` and `even`.
        #[serde(default)]
        #[param(default = 0)]
        long_prefill_token_threshold: u32,
        /// Order in which queued fresh prompts are offered to the head. `fifo`
        /// is vLLM; started prompts keep priority over fresh ones unless `srpt`.
        #[serde(default = "default_pipeline_pending_order")]
        #[param(string, default = "fifo", choices = PENDING_ORDER_CHOICES)]
        pending_order: PendingOrderKind,
        /// Load-following budget floor (`0`: off, vLLM's fixed budget). The
        /// microbatch budget is this many tokens while the prefill backlog
        /// (queued, started and in-flight tokens) is at most
        /// `load_budget_backlog_lo_tokens`, rising linearly to
        /// `max_batch_tokens` at `load_budget_backlog_hi_tokens`; both bounds
        /// are required when the floor is set. Not vLLM.
        #[serde(default)]
        #[param(default = 0)]
        load_budget_low_tokens: u32,
        #[serde(default)]
        #[param(default = 0)]
        load_budget_backlog_lo_tokens: u64,
        #[serde(default)]
        #[param(default = 0)]
        load_budget_backlog_hi_tokens: u64,
        /// Shortest remaining prefill first across started and queued prompts
        /// (queue order should be `shortest-prefill-first`), instead of vLLM's
        /// started-first order. Not vLLM.
        #[serde(default)]
        #[param(default = false)]
        srpt: bool,
        /// Host DRAM behind the HBM prefix cache, per GPU (GB; `0`: none).
        /// Every finished context is written through to each tier; a session
        /// whose context is here but not in HBM is read back at
        /// `dram_tier_gb_per_s` per GPU when admission reaches it, as vLLM's
        /// KV connector does. Not vLLM.
        #[serde(default)]
        #[param(default = 0.0)]
        dram_tier_gb: f64,
        #[serde(default = "default_dram_tier_gb_per_s")]
        #[param(default = 50.0)]
        dram_tier_gb_per_s: f64,
        /// Local SSD behind DRAM, per GPU (GB; `0`: none), read at
        /// `ssd_tier_gb_per_s` per GPU.
        #[serde(default)]
        #[param(default = 0.0)]
        ssd_tier_gb: f64,
        #[serde(default = "default_ssd_tier_gb_per_s")]
        #[param(default = 10.0)]
        ssd_tier_gb_per_s: f64,
        /// Decode runs elsewhere: a request completes at its first token and
        /// its context, every target output token but the last (whose KV the
        /// next round computes), is retained as a decode instance would hand
        /// it back. Pair it with a session trace
        /// whose tool waits include the decode time
        /// (`trace/session_decode_wait.py`). Not vLLM.
        /// With a tier: a session request whose declared prefix exceeds every
        /// context the run stored for that session is a conversation that began
        /// before the run, and its context is read from the slowest tier, as a
        /// long-lived session's would be in steady state. For closed-loop traces
        /// that start sessions mid-life (`trace/session_closed_loop.py`).
        #[serde(default)]
        #[param(default = false)]
        prefix_tier_warm_start: bool,
        /// With a tier: start a read only while its tier would begin it within
        /// this many ms; past that, admission stops and the request waits in
        /// its queue holding no HBM. Bounds what in-flight reads hold. 0: no
        /// bound (vLLM). Not vLLM.
        #[serde(default)]
        #[param(default = 0.0)]
        prefix_tier_max_read_wait_ms: f64,
        /// With a tier: every stage reads the pipeline's mean KV bytes per
        /// token, as if the stages held equal shares, instead of the most
        /// loaded stage's. Sizes tier capacity the same way; HBM is unchanged.
        /// A what-if, not vLLM: its layer split leaves the stages unequal.
        #[serde(default)]
        #[param(default = false)]
        prefix_tier_balanced_load: bool,
        #[serde(default)]
        #[param(default = false)]
        external_decode: bool,
    },
    /// PD prefill half: prefills then hands off to a decode pool (no local decode).
    PdPrefill {
        /// GPU memory for the worker (GB; primarily KV cache budget).
        #[param(default = 80.0)]
        attn_gpu_memory_gb: f64,
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
        /// GPU wall/kernel time multiplier (≥ 1.0); models inter-kernel overhead
        /// (see [`default_gpu_time_multiplier`]). cost_log stays pre-scale.
        #[serde(default = "default_gpu_time_multiplier")]
        #[param(default = 1.0)]
        gpu_time_multiplier: f64,
    },
}

// ── layer-wise attn / ffn contract (afd) ────────────────────────────────────

#[derive(Debug, Clone, Deserialize, ProviderSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AttnWorkerSel {
    DisaggAttn {
        /// GPU memory for the attention worker (GB; KV cache budget).
        #[param(default = 80.0)]
        attn_gpu_memory_gb: f64,
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
    fn min_microbatch_tokens_requires_the_even_split() {
        let worker: IterWorkerSel = serde_yaml::from_str(
            "type: pipeline_chunked_prefill\nattn_gpu_memory_gb: 80.0\nmax_batch_tokens: 8192\n",
        )
        .expect("parse pipeline worker");
        let IterWorkerSel::PipelineChunkedPrefill {
            microbatch_split,
            min_microbatch_tokens,
            ..
        } = worker
        else {
            panic!("expected pipeline_chunked_prefill");
        };
        assert_eq!(
            resolve_microbatch_sizing(microbatch_split, min_microbatch_tokens, 8192).unwrap(),
            MicrobatchSizing::Greedy,
            "the default is vLLM's greedy fill"
        );

        let error = resolve_microbatch_sizing(MicrobatchSplit::Greedy, 1024, 8192).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("only applies with microbatch_split: even"),
            "{error}"
        );
        assert!(resolve_microbatch_sizing(MicrobatchSplit::Even, 8193, 8192).is_err());
        assert_eq!(
            resolve_microbatch_sizing(MicrobatchSplit::Even, 1024, 8192).unwrap(),
            MicrobatchSizing::Even { min_tokens: 1024 }
        );
    }

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
