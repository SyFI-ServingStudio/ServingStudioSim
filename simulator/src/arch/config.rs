//! Arch (L4) config surface — the model-arch *selectors*, co-located with the L4
//! implementations they pick (new-interface-design §2 / §4).
//!
//! Each selector is a serde tagged enum (`#[serde(tag = "type")]`): choosing the
//! tag is the only way that variant's params appear — no global union,
//! provider-first. `model_config` + dims belong to the arch (the layer that
//! consumes them), so every arch variant flattens [`ModelSpec`]. arch and worker
//! are symmetric sibling providers (not nested).
//!
//! Only the iter-wise `llama3_dense` / `llama3_dense_tp` archs are wired to
//! `build()` today; the others parse + are advertised but `build()` bails.
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

// ── iter-wise contract (unified, pd) ────────────────────────────────────────

/// Iteration-wise arch provider. `tp_size` lives ONLY on the TP variant
/// (provider-first: you select the arch, then it exposes its own params).
#[derive(Debug, Clone, Deserialize, ProviderSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IterArchSel {
    Llama3Dense {
        #[serde(flatten)]
        model: ModelSpec,
    },
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
    DeepseekMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Tensor parallelism size.
        #[param(default = 1, cache_key)]
        tp_size: u16,
        /// Expert parallelism size.
        #[param(default = 8, cache_key)]
        ep_size: u16,
    },
}

impl IterArchSel {
    /// The model identity/dims this arch operates on (every variant carries it).
    pub fn model(&self) -> &ModelSpec {
        match self {
            Self::Llama3Dense { model }
            | Self::Llama3DenseTp { model, .. }
            | Self::Llama3DpAttnTpFfn { model, .. }
            | Self::DeepseekMoe { model, .. } => model,
        }
    }
}

// ── layer-wise attn / ffn contract (afd) — config types only, build() bails ──

#[derive(Debug, Clone, Deserialize, ProviderSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AttnArchSel {
    Llama3AttnTp {
        #[serde(flatten)]
        model: ModelSpec,
        /// Tensor parallelism size.
        #[param(default = 2, cache_key)]
        tp_size: u16,
        /// Attention head parallelism.
        #[param(default = 1, cache_key)]
        head_parallel: u16,
    },
}

#[derive(Debug, Clone, Deserialize, ProviderSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FfnArchSel {
    DeepseekFfnMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Tensor parallelism size.
        #[param(default = 2, cache_key)]
        tp_size: u16,
        /// Expert parallelism size.
        #[param(default = 8, cache_key)]
        ep_size: u16,
    },
}
