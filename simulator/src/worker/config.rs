//! Worker (L5) config surface — the worker *selectors*, co-located with the L5
//! workers they pick (new-interface-design §2).
//!
//! Each selector is a serde tagged enum (`#[serde(tag = "type")]`), the symmetric
//! sibling of the arch selector. Only the iter-wise `barebone` worker is wired to
//! `build()` today; the others parse + are advertised but `build()` bails.
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

/// Iteration-wise worker provider.
#[derive(Debug, Clone, Deserialize, ProviderSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IterWorkerSel {
    Barebone {
        /// GPU memory for the worker (GB; primarily KV cache budget).
        #[param(default = 80.0)]
        attn_gpu_memory_gb: f64,
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
    },
}

#[derive(Debug, Clone, Deserialize, ProviderSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FfnWorkerSel {
    DisaggFfn {},
}
