//! Numeric model dims (`ModelCfg`) consumed by model_arch `build_configs`
//! (L4 design.md §1.1). The parallel/sharding degrees are NOT here: each arch
//! owns its own numeric parallel struct (`DenseParallel`, `DenseTpParallel`, …)
//! co-located with the arch, per new-interface-design §13 (the retired shared
//! `ParallelCfg` union).
//!
//! These are the *resolved numeric* dims, distinct from the CLI parameter layer
//! (`schema::ModelCommon`, where `model_config` is a JSON path). `from_json` loads
//! a HuggingFace `config.json` and is the only runtime source of dims; the
//! `llama3_8b()` preset is a `#[cfg(test)]` fixture, not API.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::timing::bridge::DType;
use crate::timing::Dim;

/// Raw transformer dims (per-model identity, parallelism-agnostic). The shape
/// dims are [`Dim`]s carrying their HF-config provenance (`hidden` → the leaf
/// `Dim::param("hidden", …)`), so every per-op fixed dim derived from them
/// records the formula. `num_layers` is a fold count, not a shape, so it stays a
/// plain `u32`.
#[derive(Clone, Debug)]
pub struct ModelCfg {
    pub hidden: Dim,
    pub intermediate: Dim,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub vocab: Dim,
    pub num_layers: u32,
    pub dtype: DType,
    pub kv_dtype: DType,
}

impl ModelCfg {
    /// Load from a HuggingFace `config.json`. `head_dim` falls back to
    /// `hidden_size / num_attention_heads` when absent; `kv_dtype` mirrors
    /// `torch_dtype`.
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading model config {}", path.display()))?;
        let raw: JsonModelConfig = serde_json::from_str(&text)
            .with_context(|| format!("parsing model config {}", path.display()))?;
        let dtype = parse_dtype(&raw.torch_dtype)?;
        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads);
        Ok(Self {
            hidden: Dim::param("hidden", raw.hidden_size),
            intermediate: Dim::param("intermediate", raw.intermediate_size),
            num_qo_heads: Dim::param("num_qo_heads", raw.num_attention_heads),
            num_kv_heads: Dim::param("num_kv_heads", raw.num_key_value_heads),
            head_dim: Dim::param("head_dim", head_dim),
            vocab: Dim::param("vocab", raw.vocab_size),
            num_layers: raw.num_hidden_layers,
            dtype,
            kv_dtype: dtype,
        })
    }
}

/// HuggingFace `config.json` shape (only the fields model_arch needs).
#[derive(Deserialize)]
struct JsonModelConfig {
    hidden_size: u32,
    intermediate_size: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    #[serde(default)]
    head_dim: Option<u32>,
    vocab_size: u32,
    num_hidden_layers: u32,
    torch_dtype: String,
}

fn parse_dtype(s: &str) -> Result<DType> {
    Ok(match s {
        "float16" | "fp16" => DType::Fp16,
        "bfloat16" | "bf16" => DType::Bf16,
        "float32" | "fp32" => DType::Fp32,
        "float8_e4m3fn" | "fp8_e4m3" => DType::Fp8E4m3,
        "float8_e5m2" | "fp8_e5m2" => DType::Fp8E5m2,
        "int8" => DType::Int8,
        "int4" => DType::Int4,
        other => bail!("unsupported torch_dtype {other:?}"),
    })
}

/// Test-only fixtures, kept out of the production `impl` block. Runtime dims
/// always come from [`ModelCfg::from_json`].
#[cfg(test)]
impl ModelCfg {
    /// Llama3-8B dense preset (bf16).
    pub fn llama3_8b() -> Self {
        Self {
            hidden: Dim::param("hidden", 4096),
            intermediate: Dim::param("intermediate", 14336),
            num_qo_heads: Dim::param("num_qo_heads", 32),
            num_kv_heads: Dim::param("num_kv_heads", 8),
            head_dim: Dim::param("head_dim", 128),
            vocab: Dim::param("vocab", 128256),
            num_layers: 32,
            dtype: DType::Bf16,
            kv_dtype: DType::Bf16,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_json_llama3_shape_with_head_dim_fallback() {
        // Llama3-8B config.json shape, no explicit head_dim.
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "num_hidden_layers": 32,
            "torch_dtype": "bfloat16"
        }"#;
        let dir = std::env::temp_dir();
        let path = dir.join(format!("vibesim_cfg_{}.json", std::process::id()));
        std::fs::write(&path, json).unwrap();
        let cfg = ModelCfg::from_json(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(cfg.hidden, 4096);
        assert_eq!(cfg.intermediate, 14336);
        assert_eq!(cfg.num_qo_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128); // 4096 / 32 fallback
        assert_eq!(cfg.vocab, 128256);
        assert_eq!(cfg.num_layers, 32);
        assert_eq!(cfg.dtype, DType::Bf16);
        assert_eq!(cfg.kv_dtype, DType::Bf16);
    }

    /// The payoff: a per-op fixed dim derived from `ModelCfg` carries its
    /// derivation formula, not just the folded value. The QKV-projection `n`
    /// (as computed in `pre_attn_local::resolve`) renders through `Dim`'s
    /// `Debug` — the same path `KernelConfig::describe_config` (`{:?}`) takes —
    /// as `formula=value` with the originating HF-config param names. This is
    /// what surfaces in `cost_tree.describe()` / the `CostManifest` slot config.
    #[test]
    fn derived_dim_carries_model_config_provenance() {
        let m = ModelCfg::llama3_8b();
        // Fused QKV output width: (num_qo_heads + 2·num_kv_heads)·head_dim.
        let qkv_n = (m.num_qo_heads.clone() + 2 * m.num_kv_heads.clone()) * m.head_dim.clone();

        assert_eq!(qkv_n, 6144); // folds to the same value (cache key unchanged)
        assert_eq!(
            format!("{qkv_n:?}"),
            "(num_qo_heads+2*num_kv_heads)*head_dim=6144"
        );
        // Provenance = exactly the model dims that feed this op.
        let params: Vec<&str> = qkv_n.params().into_iter().collect();
        assert_eq!(params, ["head_dim", "num_kv_heads", "num_qo_heads"]);
    }

    #[test]
    fn parse_dtype_maps_hf_names() {
        assert_eq!(parse_dtype("bfloat16").unwrap(), DType::Bf16);
        assert_eq!(parse_dtype("float16").unwrap(), DType::Fp16);
        assert_eq!(parse_dtype("float8_e4m3fn").unwrap(), DType::Fp8E4m3);
        assert!(parse_dtype("nonsense").is_err());
    }
}
