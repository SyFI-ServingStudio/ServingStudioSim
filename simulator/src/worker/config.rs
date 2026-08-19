//! Worker (L5) config surface — the worker *selectors*, co-located with the L5
//! workers they pick (new-interface-design §2).
//!
//! Each selector is a serde tagged enum (`#[serde(tag = "type")]`), the symmetric
//! sibling of the arch selector. `barebone` / `hp_unified` are wired for
//! `unified`; `pd_prefill` / `pd_decode` are wired for `pd`; the AFD selectors
//! are wired for `afd`. `chunked_prefill` parses + is advertised, but its
//! deployment still bails until that lifecycle is implemented.
//!
//! `#[derive(ProviderSchema)]` emits each selector's `SCHEMA` of `(tag, params)`
//! rows for the launcher; `schema::dump::list_params` aggregates them.

use anyhow::{ensure, Result};
use serde::Deserialize;

use schema_derive::ProviderSchema;

use super::admission::PendingOrderKind;
use super::kv::{PrefixCacheConfig, PrefixCacheMode, PrefixCachePolicy};

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
/// `cost_log` / manifest stay pre-scale (pure kernel), so the overhead surfaces as
/// a gap between iter/section slices in the trace, never inside a kernel slice.
fn default_gpu_time_multiplier() -> f64 {
    1.0
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
        /// (see [`default_gpu_time_multiplier`]). `cost_log` stays pre-scale.
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
        /// (see [`default_gpu_time_multiplier`]). `cost_log` stays pre-scale.
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
        /// GPU wall/kernel time multiplier (≥ 1.0); models inter-kernel overhead
        /// (see [`default_gpu_time_multiplier`]). `cost_log` stays pre-scale.
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
        /// (see [`default_gpu_time_multiplier`]). `cost_log` stays pre-scale.
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
        /// (see [`default_gpu_time_multiplier`]). `cost_log` stays pre-scale.
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
        /// (see [`default_gpu_time_multiplier`]). `cost_log` stays pre-scale.
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
        /// (see [`default_gpu_time_multiplier`]). `cost_log` stays pre-scale.
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
}
