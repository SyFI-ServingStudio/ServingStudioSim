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

use crate::common::Time;

use super::admission::{
    ForceSchedule, HitsFirst, LoadBudget, MicrobatchBudget, PendingOrderKind, PrefillSplit,
    ReadWait, ReadWaitPolicy, StartedOrder,
};
use super::kv::{
    KvOverhead, PartPadding, PrefixCacheConfig, PrefixCacheMode, PrefixCachePolicy,
    PrefixResumePoints, StateRetention, TierSpec, TierWrites, DRAM_TIER,
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
const PREFIX_RESUME_POINTS_CHOICES: [&str; 5] = [
    "prompt_end",
    "response_end",
    "prompt_and_response_end",
    "every_block",
    "context_end",
];

/// How a worker turns the arch's `KvCacheLayout` into charges: whether the
/// serving engine's own allocation counts, and where a finished request keeps
/// the state a later request resumes from. Flattened into every selector that
/// owns KV.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, ParamStruct)]
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
    /// `prompt_end` under `vllm`, `response_end` under `none`. `context_end`
    /// (not vLLM) keeps it at the unfloored context end only. A layout of
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
    /// Entry bytes of named layout parts, overriding `kv_overhead` for those
    /// parts only: `<part>=padded`, `<part>=unpadded`, or `<part>=<bytes per
    /// entry>` (an engine that packs a state differently). Every other
    /// `kv_overhead` rule stays. A name the arch's layout lacks is an error.
    #[serde(default)]
    kv_part_padding: Vec<String>,
    /// The engine block in tokens, overriding the one the arch derives (an
    /// engine with other pages than vLLM's). It drives per-token block
    /// rounding, resume-point flooring, the checkpoint chunk boundary and the
    /// null block; padded parts keep their declared bytes unless
    /// `kv_part_padding` names them. Absent keeps the arch's block.
    #[serde(default)]
    kv_block_tokens: Option<u32>,
}

impl KvAccountingSpec {
    /// Validate the knobs and set them on `config`.
    pub(crate) fn apply(&self, owner: &str, config: &mut WorkerConfig) -> Result<()> {
        ensure!(
            self.prefix_resume_interval != Some(0),
            "{owner}: prefix_resume_interval must be positive"
        );
        ensure!(
            self.kv_max_in_flight_tokens != Some(0),
            "{owner}: kv_max_in_flight_tokens must be positive"
        );
        ensure!(
            self.kv_block_tokens != Some(0),
            "{owner}: kv_block_tokens must be positive"
        );
        config.kv_overhead = self.kv_overhead;
        config.prefix_resume_points = self.prefix_resume_points;
        config.prefix_resume_interval = self.prefix_resume_interval;
        config.kv_max_in_flight_tokens = self.kv_max_in_flight_tokens;
        config.kv_block_tokens = self.kv_block_tokens;
        config.kv_part_padding = Vec::with_capacity(self.kv_part_padding.len());
        for value in &self.kv_part_padding {
            let padding: PartPadding = value
                .parse()
                .map_err(|error| anyhow::anyhow!("{owner}: kv_part_padding: {error:#}"))?;
            ensure!(
                config
                    .kv_part_padding
                    .iter()
                    .all(|named| named.part != padding.part),
                "{owner}: kv_part_padding names part `{}` twice",
                padding.part
            );
            config.kv_part_padding.push(padding);
        }
        Ok(())
    }
}

/// Which round-end recurrent states a retained context keeps. Flattened into
/// both chunked selectors; absent keeps the end state only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, ParamStruct)]
pub struct StateRetentionSpec {
    /// Round-end states a retained context keeps: its end's and the latest
    /// earlier rounds', so a prompt ending inside the context resumes from
    /// the latest kept state at or before it. A context and its states are
    /// charged, kept and evicted as one. Above 1 needs `prefix_resume_points:
    /// context_end`. Not vLLM.
    #[serde(default = "one_state")]
    #[param(default = 1)]
    recurrent_state_checkpoints: u32,
    /// When > 0, a retained context keeps instead the latest handed-back
    /// decode's end, the latest pass's end, and this many anchors: the latest
    /// earlier pass ends of rounds (counted along the session's lineage) that
    /// are a multiple of `recurrent_state_anchor_stride`. Stride 2 records an
    /// anchor every other round, so the kept anchor is one to two rounds back
    /// at a constant state count. Needs `recurrent_state_checkpoints: 1`. Not
    /// vLLM.
    #[serde(default)]
    #[param(default = 0)]
    recurrent_state_anchors: u32,
    /// Round stride of `recurrent_state_anchors` (>= 1). Not vLLM.
    #[serde(default = "one_round")]
    #[param(default = 1)]
    recurrent_state_anchor_stride: u32,
    /// With `recurrent_state_anchors` and a host tier: an anchor, immutable
    /// once written, is written once, to the tier its context first lands in
    /// below HBM, and stays there. A read moves only the state its round
    /// resumes from (an anchor only when an inside round resumes from it,
    /// with the KV HBM's entry for the session still holds counted as
    /// resident); an exclusive DRAM keeps a read entry cut to its latest
    /// surviving anchor; a demotion does not rewrite an anchor (a tier
    /// evicting the context drops its anchors). Needs
    /// `recurrent_state_anchors` > 0. Not vLLM.
    #[serde(default)]
    #[param(default = false)]
    recurrent_state_anchors_in_place: bool,
}

const fn one_state() -> u32 {
    1
}

const fn one_round() -> u32 {
    1
}

impl Default for StateRetentionSpec {
    fn default() -> Self {
        Self {
            recurrent_state_checkpoints: one_state(),
            recurrent_state_anchors: 0,
            recurrent_state_anchor_stride: one_round(),
            recurrent_state_anchors_in_place: false,
        }
    }
}

impl StateRetentionSpec {
    /// Whether anchors stay in the tier they first land in.
    pub(crate) fn anchors_in_place(self, owner: &str) -> Result<bool> {
        ensure!(
            !self.recurrent_state_anchors_in_place || self.recurrent_state_anchors > 0,
            "{owner}: recurrent_state_anchors_in_place needs recurrent_state_anchors > 0"
        );
        Ok(self.recurrent_state_anchors_in_place)
    }

    /// The retention; at least one state, the end's.
    pub(crate) fn resolve(self, owner: &str) -> Result<StateRetention> {
        ensure!(
            self.recurrent_state_checkpoints >= 1,
            "{owner}: recurrent_state_checkpoints must be >= 1 (a context keeps its end state)"
        );
        if self.recurrent_state_anchors == 0 {
            return Ok(StateRetention::Latest {
                k: self.recurrent_state_checkpoints,
            });
        }
        ensure!(
            self.recurrent_state_checkpoints == 1 && self.recurrent_state_anchor_stride >= 1,
            "{owner}: recurrent_state_anchors needs recurrent_state_checkpoints: 1 and \
             recurrent_state_anchor_stride >= 1"
        );
        Ok(StateRetention::Anchors {
            count: self.recurrent_state_anchors,
            stride: self.recurrent_state_anchor_stride,
        })
    }
}

/// Host tiers behind a pipeline head's prefix cache: DRAM, then SSD, each per
/// GPU. Flattened into `pipeline_chunked_prefill`; every key absent is no
/// tier. A session request whose context is in a tier but not in HBM reads
/// it back when admission reaches it, as vLLM's KV connector does; the rest
/// is not vLLM.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, ParamStruct)]
pub struct PrefixTierSpec {
    /// Host DRAM per GPU (GB; `0`: none).
    #[serde(default)]
    #[param(default = 0.0)]
    dram_tier_gb: f64,
    /// DRAM -> HBM read bandwidth per GPU (GB/s): a PCIe Gen5 x16 link.
    #[serde(default = "default_dram_tier_gb_per_s")]
    #[param(default = 50.0)]
    dram_tier_gb_per_s: f64,
    /// What DRAM is written: `every-context`, each finished context
    /// (write-through); `hbm-evictions`, only what the HBM prefix cache evicts
    /// or cannot keep, as what DRAM's entry for the session lacks (a victim
    /// tier); `hbm-evictions-exclusive`, the same, and a read moves the
    /// context back to HBM instead of copying it.
    #[serde(default)]
    #[param(string, default = "every-context", choices = DRAM_TIER_WRITES_CHOICES)]
    dram_tier_writes: DramTierWrites,
    /// Local SSD behind DRAM per GPU (GB; `0`: none).
    #[serde(default)]
    #[param(default = 0.0)]
    ssd_tier_gb: f64,
    /// SSD -> HBM read bandwidth per GPU (GB/s): one Gen5 NVMe drive.
    #[serde(default = "default_ssd_tier_gb_per_s")]
    #[param(default = 10.0)]
    ssd_tier_gb_per_s: f64,
    /// What SSD is written: `every-context` (write-through) or
    /// `dram-evictions`, only what DRAM evicts or cannot hold (a victim tier;
    /// needs both tiers).
    #[serde(default)]
    #[param(string, default = "every-context", choices = SSD_TIER_WRITES_CHOICES)]
    ssd_tier_writes: SsdTierWrites,
    /// Tiers store and read the stages' mean KV bytes, as if the stages held
    /// equal shares, instead of the most loaded stage's. HBM is unchanged. A
    /// what-if: vLLM's layer split leaves the stages unequal.
    #[serde(default)]
    #[param(default = false)]
    prefix_tier_balanced_load: bool,
    /// Start a read only while its tier's channel would begin it within this
    /// many ms; past that, `prefix_tier_read_wait_policy` decides. `0`: no
    /// bound. Needs a tier.
    #[serde(default)]
    #[param(default = 0.0)]
    prefix_tier_max_read_wait_ms: f64,
    /// With `prefix_tier_max_read_wait_ms`: `pass-over` leaves a request whose
    /// read would wait past the bound queued, holding nothing, and admits the
    /// ones behind it; `block` stops admission at it.
    #[serde(default)]
    #[param(string, default = "pass-over", choices = READ_WAIT_POLICY_CHOICES)]
    prefix_tier_read_wait_policy: ReadWaitPolicy,
    /// A session request whose declared prefix exceeds every context the run
    /// stored for its session began before the run: its context is read from
    /// the slowest tier, as a long-lived session's would be in steady state.
    /// For closed-loop traces that start sessions mid-life.
    #[serde(default)]
    #[param(default = false)]
    prefix_tier_warm_start: bool,
    /// Staged loading (not vLLM): reads go through loading areas of this
    /// fraction of HBM and, with an SSD behind DRAM, of DRAM, kept free for
    /// them. Before each admission pass, reads start in queue order while the
    /// area each lands in has room, and admission takes a request only once
    /// its read has landed. `0`: a read starts when admission reaches its
    /// request. In `[0, 0.5)`; needs a tier and a recurrent-state arch, and
    /// takes no read-wait bound.
    #[serde(default)]
    #[param(default = 0.0)]
    prefix_tier_loading_area: f64,
    /// Tiers store each recurrent state compressed by this ratio, rounded up,
    /// encoded and decoded by the host CPU (its time not modeled) while HBM
    /// keeps it whole: a state costs its stored size in tier capacity, writes
    /// and on the SSD channel, its decoded size on DRAM's channel and in
    /// HBM's loading area. Finite and >= 1; `1`: uncompressed. Not vLLM.
    #[serde(default = "default_tier_state_compression_ratio")]
    #[param(default = 1.0)]
    tier_state_compression_ratio: f64,
}

const fn default_dram_tier_gb_per_s() -> f64 {
    50.0
}

const fn default_ssd_tier_gb_per_s() -> f64 {
    10.0
}

const fn default_tier_state_compression_ratio() -> f64 {
    1.0
}

impl Default for PrefixTierSpec {
    fn default() -> Self {
        Self {
            dram_tier_gb: 0.0,
            dram_tier_gb_per_s: default_dram_tier_gb_per_s(),
            dram_tier_writes: DramTierWrites::default(),
            ssd_tier_gb: 0.0,
            ssd_tier_gb_per_s: default_ssd_tier_gb_per_s(),
            ssd_tier_writes: SsdTierWrites::default(),
            prefix_tier_balanced_load: false,
            prefix_tier_max_read_wait_ms: 0.0,
            prefix_tier_read_wait_policy: ReadWaitPolicy::default(),
            prefix_tier_warm_start: false,
            prefix_tier_loading_area: 0.0,
            tier_state_compression_ratio: default_tier_state_compression_ratio(),
        }
    }
}

impl PrefixTierSpec {
    /// Validate the tiers and set them on `config`.
    pub(crate) fn apply(self, owner: &str, config: &mut WorkerConfig) -> Result<()> {
        let (dram_gb, ssd_gb) = (self.dram_tier_gb, self.ssd_tier_gb);
        ensure!(
            self.dram_tier_writes == DramTierWrites::EveryContext || dram_gb > 0.0,
            "{owner}: dram_tier_writes: hbm-evictions(-exclusive) needs dram_tier_gb > 0"
        );
        ensure!(
            self.ssd_tier_writes == SsdTierWrites::EveryContext || (dram_gb > 0.0 && ssd_gb > 0.0),
            "{owner}: ssd_tier_writes: dram-evictions needs dram_tier_gb > 0 and ssd_tier_gb > 0"
        );
        let tier = |name: &'static str, gb: f64, gb_per_s: f64, writes| -> Result<_> {
            ensure!(
                gb >= 0.0 && (gb == 0.0 || gb_per_s > 0.0),
                "{owner}: {name}_tier_gb must be >= 0 and its bandwidth positive"
            );
            Ok((gb > 0.0).then_some(TierSpec {
                name,
                capacity_gb: gb,
                read_gb_per_s: gb_per_s,
                writes,
            }))
        };
        let dram_writes = match self.dram_tier_writes {
            DramTierWrites::EveryContext => TierWrites::EveryContext,
            DramTierWrites::HbmEvictions => TierWrites::Evictions,
            DramTierWrites::HbmEvictionsExclusive => TierWrites::EvictionsExclusive,
        };
        let ssd_writes = match self.ssd_tier_writes {
            SsdTierWrites::EveryContext => TierWrites::EveryContext,
            SsdTierWrites::DramEvictions => TierWrites::Evictions,
        };
        config.prefix_tiers = [
            tier(DRAM_TIER, dram_gb, self.dram_tier_gb_per_s, dram_writes)?,
            tier("ssd", ssd_gb, self.ssd_tier_gb_per_s, ssd_writes)?,
        ]
        .into_iter()
        .flatten()
        .collect();

        let ms = self.prefix_tier_max_read_wait_ms;
        ensure!(
            ms.is_finite() && ms >= 0.0,
            "{owner}: prefix_tier_max_read_wait_ms ({ms}) must be a finite value >= 0"
        );
        ensure!(
            self.prefix_tier_read_wait_policy == ReadWaitPolicy::PassOver || ms > 0.0,
            "{owner}: prefix_tier_read_wait_policy: block needs prefix_tier_max_read_wait_ms > 0"
        );
        ensure!(
            ms == 0.0 || !config.prefix_tiers.is_empty(),
            "{owner}: prefix_tier_max_read_wait_ms needs dram_tier_gb or ssd_tier_gb > 0"
        );
        config.prefix_tier_read_wait = (ms > 0.0).then(|| ReadWait {
            max: Time::from_ms(ms),
            policy: self.prefix_tier_read_wait_policy,
        });
        let area = self.prefix_tier_loading_area;
        ensure!(
            (0.0..0.5).contains(&area),
            "{owner}: prefix_tier_loading_area ({area}) must be in [0, 0.5)"
        );
        ensure!(
            area == 0.0 || !config.prefix_tiers.is_empty(),
            "{owner}: prefix_tier_loading_area needs dram_tier_gb or ssd_tier_gb > 0"
        );
        ensure!(
            area == 0.0 || ms == 0.0,
            "{owner}: prefix_tier_max_read_wait_ms does not apply to staged reads \
             (prefix_tier_loading_area > 0)"
        );
        config.prefix_tier_loading_area = area;
        let ratio = self.tier_state_compression_ratio;
        ensure!(
            ratio.is_finite() && ratio >= 1.0,
            "{owner}: tier_state_compression_ratio ({ratio}) must be a finite ratio >= 1"
        );
        config.tier_state_compression_ratio = ratio;
        config.prefix_tier_balanced_load = self.prefix_tier_balanced_load;
        config.prefix_tier_warm_start = self.prefix_tier_warm_start;
        Ok(())
    }
}

/// What the DRAM tier is written (`dram_tier_writes`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DramTierWrites {
    /// Each finished context (write-through).
    #[default]
    EveryContext,
    /// What HBM evicts or cannot keep (HBM's victim tier).
    HbmEvictions,
    /// The same, and a read moves the context out of DRAM.
    HbmEvictionsExclusive,
}

const DRAM_TIER_WRITES_CHOICES: [&str; 3] =
    ["every-context", "hbm-evictions", "hbm-evictions-exclusive"];

/// What the SSD tier is written (`ssd_tier_writes`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SsdTierWrites {
    /// Each finished context (write-through).
    #[default]
    EveryContext,
    /// What DRAM evicts or cannot hold (DRAM's victim tier).
    DramEvictions,
}

const SSD_TIER_WRITES_CHOICES: [&str; 2] = ["every-context", "dram-evictions"];

const READ_WAIT_POLICY_CHOICES: [&str; 2] = ["pass-over", "block"];

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

/// How much prefill one batch takes: a pipeline microbatch, or one attention
/// DP rank's share of a `chunked_prefill` iteration. Flattened into both
/// chunked selectors; every key absent is vLLM's greedy fill.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, ParamStruct)]
pub struct MicrobatchBudgetSpec {
    /// `greedy` (vLLM) fills each microbatch's prefill up to
    /// `max_batch_tokens`; `even` (`pipeline_chunked_prefill` only) sizes it
    /// to one `pp_size`-th of the pending plus in-flight prefill, so a backlog
    /// spreads over the pipeline instead of one full microbatch followed by
    /// small ones. Not vLLM.
    #[serde(default)]
    #[param(string, default = "greedy", choices = MICROBATCH_SPLIT_CHOICES)]
    microbatch_split: MicrobatchSplit,
    /// Floor on the `even` target, in prefill tokens (`0`: none). Valid only
    /// with `microbatch_split: even`; never delays a microbatch.
    #[serde(default)]
    #[param(default = 0)]
    min_microbatch_tokens: u32,
    /// vLLM's `long_prefill_token_threshold`: the most prefill tokens one
    /// request takes per batch (`0`: no cap, vLLM's default).
    #[serde(default)]
    #[param(default = 0)]
    long_prefill_token_threshold: u32,
    /// Load-following budget floor (`0`: off, vLLM's fixed budget). The batch
    /// budget is this many tokens while the prefill backlog (queued and
    /// started prompts' tokens, and a pipeline's in-flight ones) is at most
    /// `load_budget_backlog_lo_tokens`, rising linearly to `max_batch_tokens`
    /// at `load_budget_backlog_hi_tokens`; both bounds are required when the
    /// floor is set. Each attention DP rank follows its own backlog. Not vLLM.
    #[serde(default)]
    #[param(default = 0)]
    load_budget_low_tokens: u32,
    #[serde(default)]
    #[param(default = 0)]
    load_budget_backlog_lo_tokens: u64,
    #[serde(default)]
    #[param(default = 0)]
    load_budget_backlog_hi_tokens: u64,
}

impl MicrobatchBudgetSpec {
    /// Validate the knobs against `max_batch_tokens`; an even split needs a
    /// pipeline.
    pub(crate) fn resolve(
        self,
        owner: &str,
        max_batch_tokens: u32,
        pipelined: bool,
    ) -> Result<MicrobatchBudget> {
        let split = match self.microbatch_split {
            MicrobatchSplit::Greedy => {
                ensure!(
                    self.min_microbatch_tokens == 0,
                    "{owner}: min_microbatch_tokens ({}) only applies with microbatch_split: \
                     even; greedy takes every pending token it can",
                    self.min_microbatch_tokens
                );
                PrefillSplit::Greedy
            }
            MicrobatchSplit::Even => {
                ensure!(
                    pipelined,
                    "{owner}: microbatch_split: even needs a pipeline (pipeline_chunked_prefill)"
                );
                ensure!(
                    self.min_microbatch_tokens <= max_batch_tokens,
                    "{owner}: min_microbatch_tokens ({}) must be <= max_batch_tokens \
                     ({max_batch_tokens})",
                    self.min_microbatch_tokens
                );
                PrefillSplit::Even {
                    min_tokens: self.min_microbatch_tokens,
                }
            }
        };
        ensure!(
            self.load_budget_low_tokens <= max_batch_tokens,
            "{owner}: load_budget_low_tokens ({}) must be <= max_batch_tokens \
             ({max_batch_tokens})",
            self.load_budget_low_tokens
        );
        let load = if self.load_budget_low_tokens == 0 {
            ensure!(
                self.load_budget_backlog_lo_tokens == 0 && self.load_budget_backlog_hi_tokens == 0,
                "{owner}: load_budget_backlog_*_tokens need load_budget_low_tokens"
            );
            None
        } else {
            ensure!(
                self.load_budget_backlog_lo_tokens < self.load_budget_backlog_hi_tokens,
                "{owner}: load_budget_low_tokens needs load_budget_backlog_lo_tokens < \
                 load_budget_backlog_hi_tokens, got {} and {}",
                self.load_budget_backlog_lo_tokens,
                self.load_budget_backlog_hi_tokens
            );
            Some(LoadBudget {
                low_tokens: self.load_budget_low_tokens,
                backlog_lo_tokens: self.load_budget_backlog_lo_tokens,
                backlog_hi_tokens: self.load_budget_backlog_hi_tokens,
            })
        };
        Ok(MicrobatchBudget {
            split,
            long_prefill_token_threshold: (self.long_prefill_token_threshold > 0)
                .then_some(self.long_prefill_token_threshold),
            load,
        })
    }
}

/// When a queued request is served before the rest of its queue. Flattened
/// into both chunked selectors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Deserialize, ParamStruct)]
pub struct ForceScheduleSpec {
    /// Serve a request that has waited this many ms since arrival before the
    /// rest of its queue, so a shortest-first order cannot starve it (`0`:
    /// off); with `srpt`, overdue started prompts go first too. Per attention
    /// DP rank. Not vLLM.
    #[serde(default)]
    #[param(default = 0.0)]
    force_schedule_after_ms: f64,
    /// With `force_schedule_after_ms` and `srpt`: requests split by cost
    /// class. Those whose session's context is held (HBM or a tier), so they
    /// compute only their new tokens, go first (overdue ones oldest first,
    /// then shortest first); those that compute their context (a new session,
    /// an evicted one, a prompt ending inside its context) follow in the same
    /// order. Waiting behind recomputes cannot then turn a held context into
    /// one. Not vLLM.
    #[serde(default)]
    #[param(default = false)]
    force_schedule_hits_first: bool,
    /// With `force_schedule_hits_first`: split by cost class only while
    /// serving the overdue requests oldest first would evict some queued
    /// request's held context from every prefix tier holding it (its LRU
    /// distance to eviction against the prefill tokens those requests
    /// compute); otherwise oldest first. Without tiers nothing is at risk.
    /// Not vLLM.
    #[serde(default)]
    #[param(default = false)]
    force_schedule_hits_first_when_at_risk: bool,
}

impl ForceScheduleSpec {
    /// The bound and its class order, `None` when off. `srpt`: the started
    /// order is shortest remaining first, the only one that splits by class.
    pub(crate) fn resolve(self, owner: &str, srpt: bool) -> Result<Option<ForceSchedule>> {
        let ms = self.force_schedule_after_ms;
        ensure!(
            ms.is_finite() && ms >= 0.0,
            "{owner}: force_schedule_after_ms ({ms}) must be a finite value >= 0"
        );
        ensure!(
            !self.force_schedule_hits_first || (ms > 0.0 && srpt),
            "{owner}: force_schedule_hits_first needs force_schedule_after_ms > 0 and srpt"
        );
        ensure!(
            !self.force_schedule_hits_first_when_at_risk || self.force_schedule_hits_first,
            "{owner}: force_schedule_hits_first_when_at_risk needs force_schedule_hits_first"
        );
        let hits_first = match (
            self.force_schedule_hits_first,
            self.force_schedule_hits_first_when_at_risk,
        ) {
            (false, _) => HitsFirst::Off,
            (true, false) => HitsFirst::Always,
            (true, true) => HitsFirst::WhenAtRisk,
        };
        Ok((ms > 0.0).then(|| ForceSchedule {
            after: Time::from_ms(ms),
            hits_first,
        }))
    }
}

const PENDING_ORDER_CHOICES: [&str; 5] = [
    "session-start",
    "fifo",
    "shortest-job-first",
    "shortest-prefill-first",
    "longest-prefix-match",
];

/// serde fallback for the chunked selectors' `pending_order`: vLLM's FCFS.
const fn default_fifo_pending_order() -> PendingOrderKind {
    PendingOrderKind::Fifo
}

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
        /// Order of each attention DP rank's waiting queue. `fifo` is vLLM's
        /// FCFS; `shortest-prefill-first` offers the prompt with the fewest
        /// tokens to compute first (not vLLM). Started prompts keep priority
        /// either way unless `srpt`.
        #[serde(default = "default_fifo_pending_order")]
        #[param(string, default = "fifo", choices = PENDING_ORDER_CHOICES)]
        pending_order: PendingOrderKind,
        /// Shortest remaining prefill first across each rank's started and
        /// queued prompts (queue order should be `shortest-prefill-first`),
        /// instead of vLLM's started-first order. Not vLLM.
        #[serde(default)]
        #[param(default = false)]
        srpt: bool,
        /// Per-rank prefill budget; see [`MicrobatchBudgetSpec`].
        #[serde(flatten)]
        microbatch_budget: MicrobatchBudgetSpec,
        /// Per-rank force-schedule bound; see [`ForceScheduleSpec`].
        #[serde(flatten)]
        force_schedule: ForceScheduleSpec,
        /// Round-end states a retained context keeps; see
        /// [`StateRetentionSpec`].
        #[serde(flatten)]
        state_retention: StateRetentionSpec,
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
        /// How new requests spread across the arch's attention DP partitions.
        /// See [`IterWorkerSel::ChunkedPrefill`]'s field of the same name.
        #[serde(default)]
        #[param(string, default = "round-robin", choices = DP_PLACEMENT_CHOICES)]
        dp_placement: DpPlacement,
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
        /// Order in which queued fresh prompts are offered to the head. `fifo`
        /// is vLLM; `shortest-prefill-first` offers the prompt with the fewest
        /// tokens to compute first (not vLLM). Started prompts keep priority
        /// over fresh ones unless `srpt`.
        #[serde(default = "default_fifo_pending_order")]
        #[param(string, default = "fifo", choices = PENDING_ORDER_CHOICES)]
        pending_order: PendingOrderKind,
        /// Shortest remaining prefill first across started and queued prompts
        /// (queue order should be `shortest-prefill-first`), instead of vLLM's
        /// started-first order. Not vLLM.
        #[serde(default)]
        #[param(default = false)]
        srpt: bool,
        /// Microbatch prefill budget; see [`MicrobatchBudgetSpec`].
        #[serde(flatten)]
        microbatch_budget: MicrobatchBudgetSpec,
        /// Force-schedule bound, over queued and started prompts; see
        /// [`ForceScheduleSpec`].
        #[serde(flatten)]
        force_schedule: ForceScheduleSpec,
        /// Decode runs elsewhere: a request completes at its first token, and
        /// its decode hands the context back, every output but the last
        /// extending the prompt's context in HBM and in the tiers (a round
        /// then adds two round-end states). Not vLLM.
        #[serde(default)]
        #[param(default = false)]
        external_decode: bool,
        /// Round-end states a retained context keeps; see
        /// [`StateRetentionSpec`].
        #[serde(flatten)]
        state_retention: StateRetentionSpec,
        /// Host tiers behind the prefix cache; see [`PrefixTierSpec`].
        #[serde(flatten)]
        prefix_tiers: PrefixTierSpec,
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
            | Self::PdDecode { kv_accounting, .. } => kv_accounting.clone(),
        }
    }

    /// Validate a chunked selector's scheduling options and set them on
    /// `config`; every other selector keeps vLLM's scheduler.
    pub(crate) fn apply_chunked_scheduling(
        &self,
        owner: &str,
        config: &mut WorkerConfig,
    ) -> Result<()> {
        let (max_batch_tokens, srpt, microbatch_budget, force_schedule, state_retention, pipelined) =
            match self {
                Self::ChunkedPrefill {
                    max_batch_tokens,
                    srpt,
                    microbatch_budget,
                    force_schedule,
                    state_retention,
                    ..
                } => (
                    max_batch_tokens,
                    srpt,
                    microbatch_budget,
                    force_schedule,
                    state_retention,
                    false,
                ),
                Self::PipelineChunkedPrefill {
                    max_batch_tokens,
                    srpt,
                    microbatch_budget,
                    force_schedule,
                    state_retention,
                    ..
                } => (
                    max_batch_tokens,
                    srpt,
                    microbatch_budget,
                    force_schedule,
                    state_retention,
                    true,
                ),
                _ => return Ok(()),
            };
        config.microbatch_budget =
            microbatch_budget.resolve(owner, *max_batch_tokens, pipelined)?;
        config.started_order = if *srpt {
            StartedOrder::ShortestRemaining
        } else {
            StartedOrder::StartOrder
        };
        config.force_schedule = force_schedule.resolve(owner, *srpt)?;
        config.state_retention = state_retention.resolve(owner)?;
        config.recurrent_state_anchors_in_place = state_retention.anchors_in_place(owner)?;
        Ok(())
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

    /// Parse a chunked selector of `kind` with `max_batch_tokens` 8192 plus
    /// `keys`, and resolve its scheduling options.
    fn chunked_scheduling(kind: &str, keys: &str) -> Result<WorkerConfig> {
        let batch_policy = if kind == "chunked_prefill" {
            "batch_policy: mix\n"
        } else {
            ""
        };
        let worker: IterWorkerSel = serde_yaml::from_str(&format!(
            "type: {kind}\nattn_gpu_memory_gb: 80.0\nmax_batch_tokens: 8192\n{batch_policy}{keys}"
        ))
        .expect("parse chunked worker");
        let mut config = WorkerConfig::default();
        worker.apply_chunked_scheduling("t", &mut config)?;
        Ok(config)
    }

    #[test]
    fn min_microbatch_tokens_requires_the_even_split() {
        let pipeline = |keys: &str| chunked_scheduling("pipeline_chunked_prefill", keys);
        let config = pipeline("").unwrap();
        assert_eq!(
            config.microbatch_budget,
            MicrobatchBudget::default(),
            "the default is vLLM's greedy fill"
        );
        assert_eq!(config.started_order, StartedOrder::StartOrder);
        assert_eq!(config.force_schedule, None);

        let error = pipeline("min_microbatch_tokens: 1024\n").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("only applies with microbatch_split: even"),
            "{error}"
        );
        assert!(pipeline("microbatch_split: even\nmin_microbatch_tokens: 8193\n").is_err());
        assert_eq!(
            pipeline("microbatch_split: even\nmin_microbatch_tokens: 1024\n")
                .unwrap()
                .microbatch_budget
                .split,
            PrefillSplit::Even { min_tokens: 1024 }
        );
        // An even split sizes pipeline microbatches; a whole iteration has none.
        assert!(chunked_scheduling("chunked_prefill", "microbatch_split: even\n").is_err());
    }

    #[test]
    fn chunked_prefill_scheduler_options_default_to_vllm_and_validate() {
        let chunked = |keys: &str| chunked_scheduling("chunked_prefill", keys);
        let config = chunked("").unwrap();
        assert_eq!(config.microbatch_budget, MicrobatchBudget::default());
        assert_eq!(config.started_order, StartedOrder::StartOrder);
        assert_eq!(config.force_schedule, None);

        // The floor needs ordered bounds, the bounds need the floor, and the
        // floor cannot exceed the cap.
        let load = |low: u32, lo: u64, hi: u64| {
            chunked(&format!(
                "load_budget_low_tokens: {low}\nload_budget_backlog_lo_tokens: {lo}\n\
                 load_budget_backlog_hi_tokens: {hi}\n"
            ))
        };
        assert!(load(4096, 0, 0).is_err());
        assert!(load(4096, 3_000_000, 1_000_000).is_err());
        assert!(load(0, 1_000_000, 3_000_000).is_err());
        assert!(load(16384, 1_000_000, 3_000_000).is_err());
        assert_eq!(
            load(4096, 1_000_000, 3_000_000)
                .unwrap()
                .microbatch_budget
                .load
                .map(|budget| budget.target(2_000_000, 8192)),
            Some(6144)
        );

        let config = chunked(
            "pending_order: shortest-prefill-first\nsrpt: true\n\
             long_prefill_token_threshold: 4096\nforce_schedule_after_ms: 2500\n",
        )
        .unwrap();
        assert_eq!(config.started_order, StartedOrder::ShortestRemaining);
        assert_eq!(
            config.microbatch_budget.long_prefill_token_threshold,
            Some(4096)
        );
        assert_eq!(
            config.force_schedule,
            Some(ForceSchedule {
                after: Time::from_ms(2500.0),
                hits_first: HitsFirst::Off,
            })
        );
        assert!(chunked("force_schedule_after_ms: -1.0\n").is_err());

        // Hits first splits the srpt fill and the overdue queue: it needs both.
        let srpt_force = "pending_order: shortest-prefill-first\nsrpt: true\n\
                          force_schedule_after_ms: 2500\n";
        let order = |keys: &str| {
            chunked(&format!("{srpt_force}{keys}"))
                .map(|config| config.force_schedule.map(|schedule| schedule.hits_first))
        };
        assert_eq!(
            order("force_schedule_hits_first: true\n").unwrap(),
            Some(HitsFirst::Always)
        );
        assert_eq!(
            order(
                "force_schedule_hits_first: true\n\
                 force_schedule_hits_first_when_at_risk: true\n"
            )
            .unwrap(),
            Some(HitsFirst::WhenAtRisk)
        );
        assert!(order("force_schedule_hits_first_when_at_risk: true\n").is_err());
        assert!(chunked("srpt: true\nforce_schedule_hits_first: true\n").is_err());
        assert!(
            chunked("force_schedule_after_ms: 2500\nforce_schedule_hits_first: true\n").is_err()
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

    #[test]
    fn kv_part_padding_parses_each_part_once() {
        let apply = |keys: &str| {
            let worker: IterWorkerSel = serde_yaml::from_str(&format!(
                "type: pipeline_chunked_prefill\nattn_gpu_memory_gb: 80.0\n\
                 max_batch_tokens: 8192\n{keys}"
            ))
            .expect("parse pipeline worker");
            let mut config = WorkerConfig::default();
            worker
                .kv_accounting()
                .apply("t", &mut config)
                .map(|()| config)
        };
        assert!(apply("").unwrap().kv_part_padding.is_empty());
        let config = apply("kv_part_padding: [kda=unpadded, kpool_tail=0]\n").unwrap();
        assert_eq!(
            config.kv_part_padding,
            ["kda=unpadded", "kpool_tail=0"].map(|value| value.parse().unwrap())
        );
        let twice = apply("kv_part_padding: [kda=unpadded, kda=padded]\n").unwrap_err();
        assert!(twice.to_string().contains("twice"), "{twice}");
        assert!(apply("kv_part_padding: [kda]\n").is_err());
        assert_eq!(apply("").unwrap().kv_block_tokens, None);
        assert_eq!(
            apply("kv_block_tokens: 128\n").unwrap().kv_block_tokens,
            Some(128)
        );
        assert!(apply("kv_block_tokens: 0\n").is_err());
    }

    #[test]
    fn recurrent_state_checkpoints_keep_at_least_the_end_state() {
        let pipeline = |keys: &str| chunked_scheduling("pipeline_chunked_prefill", keys);
        assert_eq!(
            pipeline("").unwrap().state_retention,
            StateRetention::Latest { k: 1 }
        );
        assert_eq!(
            chunked_scheduling("chunked_prefill", "recurrent_state_checkpoints: 2\n")
                .unwrap()
                .state_retention,
            StateRetention::Latest { k: 2 }
        );
        assert!(pipeline("recurrent_state_checkpoints: 0\n").is_err());
        assert_eq!(
            pipeline("recurrent_state_anchors: 2\nrecurrent_state_anchor_stride: 2\n")
                .unwrap()
                .state_retention,
            StateRetention::Anchors {
                count: 2,
                stride: 2
            }
        );
        assert!(pipeline("recurrent_state_anchors: 1\nrecurrent_state_checkpoints: 2\n").is_err());
        assert!(
            pipeline("recurrent_state_anchors: 1\nrecurrent_state_anchor_stride: 0\n").is_err()
        );
        assert!(
            pipeline("recurrent_state_anchors: 1\nrecurrent_state_anchors_in_place: true\n")
                .unwrap()
                .recurrent_state_anchors_in_place
        );
        assert!(pipeline("recurrent_state_anchors_in_place: true\n").is_err());
    }

    fn prefix_tiers(keys: &str) -> Result<WorkerConfig> {
        let worker: IterWorkerSel = serde_yaml::from_str(&format!(
            "type: pipeline_chunked_prefill\nattn_gpu_memory_gb: 80.0\nmax_batch_tokens: 8192\n{keys}"
        ))
        .expect("parse pipeline worker");
        let IterWorkerSel::PipelineChunkedPrefill { prefix_tiers, .. } = worker else {
            panic!("expected pipeline_chunked_prefill");
        };
        let mut config = WorkerConfig::default();
        prefix_tiers.apply("t", &mut config)?;
        Ok(config)
    }

    #[test]
    fn tier_writes_build_the_victim_chain_and_need_their_tiers() {
        let writes = |keys: &str| -> Vec<(&'static str, TierWrites)> {
            prefix_tiers(keys)
                .unwrap()
                .prefix_tiers
                .iter()
                .map(|tier| (tier.name, tier.writes))
                .collect()
        };
        assert!(prefix_tiers("").unwrap().prefix_tiers.is_empty());
        let both = "dram_tier_gb: 50.0\nssd_tier_gb: 128.0\n";
        assert_eq!(
            writes(both),
            [
                ("dram", TierWrites::EveryContext),
                ("ssd", TierWrites::EveryContext)
            ]
        );
        assert_eq!(
            writes(&format!(
                "{both}dram_tier_writes: hbm-evictions\nssd_tier_writes: dram-evictions\n"
            )),
            [
                ("dram", TierWrites::Evictions),
                ("ssd", TierWrites::Evictions)
            ]
        );
        assert_eq!(
            writes("dram_tier_gb: 50.0\ndram_tier_writes: hbm-evictions-exclusive\n"),
            [("dram", TierWrites::EvictionsExclusive)]
        );
        assert_eq!(
            writes("ssd_tier_gb: 128.0\n"),
            [("ssd", TierWrites::EveryContext)]
        );
        let config = prefix_tiers(both).unwrap();
        assert_eq!(
            (
                config.prefix_tiers[0].read_gb_per_s,
                config.prefix_tiers[1].read_gb_per_s
            ),
            (50.0, 10.0)
        );
        // A victim tier needs the tier above it.
        assert!(prefix_tiers("dram_tier_writes: hbm-evictions\n").is_err());
        assert!(prefix_tiers("ssd_tier_gb: 128.0\nssd_tier_writes: dram-evictions\n").is_err());
        assert!(prefix_tiers("dram_tier_gb: -1.0\n").is_err());
        assert!(prefix_tiers("dram_tier_gb: 50.0\ndram_tier_gb_per_s: 0.0\n").is_err());
    }

    #[test]
    fn a_read_wait_bound_needs_a_tier_and_block_needs_a_bound() {
        let dram = "dram_tier_gb: 50.0\n";
        assert_eq!(prefix_tiers(dram).unwrap().prefix_tier_read_wait, None);
        assert_eq!(
            prefix_tiers(&format!("{dram}prefix_tier_max_read_wait_ms: 1000.0\n"))
                .unwrap()
                .prefix_tier_read_wait,
            Some(ReadWait {
                max: Time::from_ms(1000.0),
                policy: ReadWaitPolicy::PassOver,
            })
        );
        assert!(prefix_tiers("prefix_tier_max_read_wait_ms: 1000.0\n").is_err());
        assert!(prefix_tiers(&format!("{dram}prefix_tier_max_read_wait_ms: -1.0\n")).is_err());
        assert!(prefix_tiers(&format!("{dram}prefix_tier_read_wait_policy: block\n")).is_err());
        assert_eq!(
            prefix_tiers(&format!(
                "{dram}prefix_tier_max_read_wait_ms: 5.0\nprefix_tier_read_wait_policy: block\n"
            ))
            .unwrap()
            .prefix_tier_read_wait
            .map(|wait| wait.policy),
            Some(ReadWaitPolicy::Block)
        );
    }

    #[test]
    fn a_loading_area_is_a_fraction_below_half_that_needs_a_tier_and_no_read_wait() {
        let dram = "dram_tier_gb: 50.0\n";
        let area = |extra: &str| prefix_tiers(extra).map(|config| config.prefix_tier_loading_area);
        assert_eq!(area(dram).unwrap(), 0.0);
        assert_eq!(
            area(&format!("{dram}prefix_tier_loading_area: 0.1\n")).unwrap(),
            0.1
        );
        assert!(area(&format!("{dram}prefix_tier_loading_area: 0.5\n")).is_err());
        assert!(area(&format!("{dram}prefix_tier_loading_area: -0.1\n")).is_err());
        assert!(area("prefix_tier_loading_area: 0.1\n").is_err());
        assert!(area(&format!(
            "{dram}prefix_tier_loading_area: 0.1\nprefix_tier_max_read_wait_ms: 1000.0\n"
        ))
        .is_err());
    }

    #[test]
    fn a_tier_state_compression_ratio_is_finite_and_at_least_one() {
        let ratio =
            |extra: &str| prefix_tiers(extra).map(|config| config.tier_state_compression_ratio);
        assert_eq!(ratio("").unwrap(), 1.0);
        assert_eq!(ratio("tier_state_compression_ratio: 1.86\n").unwrap(), 1.86);
        assert!(ratio("tier_state_compression_ratio: 0.5\n").is_err());
        assert!(ratio("tier_state_compression_ratio: .inf\n").is_err());
    }
}
