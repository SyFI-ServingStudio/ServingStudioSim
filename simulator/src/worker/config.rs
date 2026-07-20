//! Worker (L5) config surface — the worker *selectors*, co-located with the L5
//! workers they pick (new-interface-design §2).
//!
//! Each selector is a serde tagged enum (`#[serde(tag = "type")]`), the symmetric
//! sibling of the arch selector. `barebone` / `hp_unified` are wired for
//! `unified`; `pd_prefill` / `pd_decode` are wired for `pd`; `chunked_prefill`
//! and the AFD selectors parse + are advertised but their deployments bail.
//!
//! `#[derive(ProviderSchema)]` emits each selector's `SCHEMA` of `(tag, params)`
//! rows for the launcher; `schema::dump::list_params` aggregates them.

use serde::Deserialize;

use schema_derive::ProviderSchema;

/// Batch-composition policy for the chunked-prefill worker. Closed set → serde
/// enum (kebab-case matches the historical CLI spelling).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BatchPolicy {
    Mix,
    SeparatePrefillPriority,
    SeparatePrefillPriorityNoInterleave,
}

const BATCH_POLICY_CHOICES: [&str; 3] = [
    "mix",
    "separate-prefill-priority",
    "separate-prefill-priority-no-interleave",
];

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

/// serde/param fallback for `kv_offload_bw_gbps`: ~PCIe gen5 x16 sustained.
fn default_host_bw_gbps() -> f64 {
    55.0
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
        /// prefills (FIFO) until the remaining budget is exhausted; a single
        /// over-long prefill is still force-admitted when the group holds budget
        /// but nothing yet. None = legacy one-prefill/iter. (Distinct from
        /// `ChunkedPrefill::max_batch_tokens`, which is a hard cap that chunks
        /// prefills to fit; this budget is soft — one whole prefill may exceed it.)
        #[serde(default)]
        max_batch_tokens: Option<u32>,
        /// Session-scoped prefix-cache budget (GB). `Some`: a request's
        /// trace-declared `prefix_kv` only hits up to what its session left
        /// resident (LRU under this budget; misses are recomputed). `None` =
        /// always-hit replay.
        #[serde(default)]
        prefix_cache_gb: Option<f64>,
        /// KV-offload host pool (GB). `Some`: KV-blocked heads preempt the
        /// newest decodes, swapping their KV to host and back (vLLM-style swap
        /// space). `None` = no offload.
        #[serde(default)]
        kv_offload_host_gb: Option<f64>,
        /// Host-link bandwidth (GB/s) pricing swap transfers (analytic; default
        /// approximates PCIe gen5 x16). Only read when `kv_offload_host_gb` set.
        #[serde(default = "default_host_bw_gbps")]
        #[param(default = 55.0)]
        kv_offload_bw_gbps: f64,
    },
    /// Multi-group HP/DP worker: maintains one `Batch` per attention DP shard
    /// (count comes from the arch's `num_attn_dp_groups`). Pairs with a DP-attention
    /// arch such as `llama3_dp_attn_tp_ffn`.
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
        /// `Barebone::max_batch_tokens`. None = legacy one-prefill/iter.
        #[serde(default)]
        max_batch_tokens: Option<u32>,
        /// Chunked-prefill mode, applied PER DP group: a HARD per-iteration
        /// token cap that splits long prompts across iterations (decodes
        /// reserve 1 token each first). Mutually exclusive with
        /// `max_batch_tokens`. None = whole-prefill-in-one-iter.
        #[serde(default)]
        chunk_prefill_tokens: Option<u32>,
        /// Session-scoped prefix-cache budget (GB). `Some`: a request's
        /// trace-declared `prefix_kv` only hits up to what its session left
        /// resident (LRU under this budget; misses are recomputed). `None` =
        /// always-hit replay.
        #[serde(default)]
        prefix_cache_gb: Option<f64>,
        /// KV-offload host pool (GB). `Some`: KV-blocked heads preempt the
        /// newest decodes, swapping their KV to host and back (vLLM-style swap
        /// space). `None` = no offload.
        #[serde(default)]
        kv_offload_host_gb: Option<f64>,
        /// Host-link bandwidth (GB/s) pricing swap transfers (analytic; default
        /// approximates PCIe gen5 x16). Only read when `kv_offload_host_gb` set.
        #[serde(default = "default_host_bw_gbps")]
        #[param(default = 55.0)]
        kv_offload_bw_gbps: f64,
    },
    ChunkedPrefill {
        /// GPU memory for the worker (GB; primarily KV cache budget).
        #[param(default = 80.0)]
        attn_gpu_memory_gb: f64,
        /// Chunked-prefill cap: max tokens per batch.
        max_batch_tokens: u32,
        /// How returning decode mixes with pending prefill.
        #[param(string, default = "mix", choices = BATCH_POLICY_CHOICES)]
        batch_policy: BatchPolicy,
        /// Session-scoped prefix-cache budget (GB). `Some`: a request's
        /// trace-declared `prefix_kv` only hits up to what its session left
        /// resident (LRU under this budget; misses are recomputed). `None` =
        /// always-hit replay.
        #[serde(default)]
        prefix_cache_gb: Option<f64>,
        /// GPU wall/kernel time multiplier (≥ 1.0); models inter-kernel overhead
        /// (see [`default_gpu_time_multiplier`]). cost_log stays pre-scale.
        #[serde(default = "default_gpu_time_multiplier")]
        #[param(default = 1.0)]
        gpu_time_multiplier: f64,
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

// ── layer-wise attn / ffn contract (afd) — config types only, build() bails ──

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
