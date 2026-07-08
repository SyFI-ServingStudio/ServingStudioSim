//! Numeric model dims for MoE archs — the parallelism-agnostic identity of one
//! MoE model. Mirrors [`ModelCfg`](crate::arch::model_cfg::ModelCfg) but folds in
//! the three MoE-specific fields (`num_experts`, `top_k`, `moe_intermediate`)
//! that a dense `ModelCfg` does not carry; the result is a single self-contained
//! struct so a MoE model_arch's `build_configs` takes ONE config object, not a
//! dense `ModelCfg` plus an extras struct.
//!
//! The launcher schema layer (`schema::ModelCommon`) still surfaces a single
//! `model_config` JSON path; [`Self::from_json`] parses a HuggingFace
//! MoE-flavored `config.json` (e.g. Qwen3-MoE / Mixtral) — the dense fields read
//! identically, and the MoE-specific keys (`num_experts`, `num_experts_per_tok`,
//! `moe_intermediate_size`) are read from the same JSON.
//!
//! The `qwen3_235b()` preset is a `#[cfg(test)]` fixture aligned to ref
//! `moesim-rs/src/workload/standard_moe.rs::qwen3_235b`.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::timing::bridge::DType;

/// Raw MoE transformer dims (per-model identity, parallelism-agnostic).
#[derive(Clone, Debug)]
pub struct MoeModelCfg {
    pub hidden: u32,
    pub num_qo_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub vocab: u32,
    pub num_layers: u32,
    /// The model's native dtype (from JSON `torch_dtype`, e.g. bf16). This is the
    /// dtype for the ops that stay 16-bit in an FP8 run — RMSNorm, the attention
    /// output, and the decode query. FP8 GEMM/attention inputs use
    /// [`Self::compute_dtype`] instead.
    pub dtype: DType,
    /// KV cache dtype — FP8 in an FP8 run (KV is bandwidth/capacity bound), else
    /// the model's base `dtype`.
    pub kv_dtype: DType,
    /// Whether this run executes in FP8 (weights + prefill attention q/kv + KV +
    /// all byte transfers at 1 B/elem). Drives GEMM backend (deepgemm) and
    /// [`Self::compute_dtype`]. Set from `ModelSpec.fp8` at build time.
    pub fp8: bool,
    /// Total expert count across all EP ranks (not per-rank).
    pub num_experts: u32,
    /// Experts each token selects (without replacement).
    pub top_k: u32,
    /// Per-expert intermediate dim (MoE FFN gate/up/down width).
    pub moe_intermediate: u32,
}

impl MoeModelCfg {
    /// Load from a HuggingFace MoE `config.json`. `head_dim` falls back to
    /// `hidden_size / num_attention_heads`; `kv_dtype` mirrors `torch_dtype`.
    /// MoE keys (`num_experts`, `num_experts_per_tok`, `moe_intermediate_size`)
    /// are required.
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading model config {}", path.display()))?;
        let raw: JsonMoeModelConfig = serde_json::from_str(&text)
            .with_context(|| format!("parsing MoE model config {}", path.display()))?;
        let dtype = parse_dtype(&raw.torch_dtype)?;
        let head_dim = raw
            .head_dim
            .unwrap_or(raw.hidden_size / raw.num_attention_heads);
        Ok(Self {
            hidden: raw.hidden_size,
            num_qo_heads: raw.num_attention_heads,
            num_kv_heads: raw.num_key_value_heads,
            head_dim,
            vocab: raw.vocab_size,
            num_layers: raw.num_hidden_layers,
            dtype,
            kv_dtype: dtype,
            fp8: false,
            num_experts: raw.num_experts,
            top_k: raw.num_experts_per_tok,
            moe_intermediate: raw.moe_intermediate_size,
        })
    }

    /// The compute dtype for GEMMs and prefill attention q/kv: FP8 in an FP8 run,
    /// else the model's base `dtype`. In a non-FP8 run `compute_dtype() == dtype`.
    pub fn compute_dtype(&self) -> DType {
        if self.fp8 { DType::Fp8E4m3 } else { self.dtype }
    }

    /// Turn FP8 on/off. FP8 moves the KV cache to FP8; the base `dtype` is left
    /// as the model's native (bf16) dtype — it is what the 16-bit-holdout ops
    /// (RMSNorm, attention output, decode query) keep using.
    pub fn with_fp8(mut self, fp8: bool) -> Self {
        self.fp8 = fp8;
        if fp8 {
            self.kv_dtype = DType::Fp8E4m3;
        }
        self
    }

    /// GEMM backend list for this run: the FP8 DeepGEMM backend when fp8, else
    /// torch. The same choice serves `single_gemm` (qkv / o_proj / router /
    /// lm_head) and `grouped_gemm` (MoE experts) — both have exactly these two
    /// backends and fp8 ⇒ deepgemm. Exactly ONE backend by dtype, never both: a
    /// `torch@fp8` or `deepgemm@bf16` lookup has no rows and would abort the
    /// build (`MissingEntry`), not fall back.
    pub fn gemm_backends(&self) -> Vec<&'static str> {
        if self.fp8 {
            vec!["deepgemm"]
        } else {
            vec!["torch"]
        }
    }
}

/// HuggingFace MoE `config.json` shape (only the fields a MoE arch needs).
#[derive(Deserialize)]
struct JsonMoeModelConfig {
    hidden_size: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    #[serde(default)]
    head_dim: Option<u32>,
    vocab_size: u32,
    num_hidden_layers: u32,
    torch_dtype: String,
    num_experts: u32,
    num_experts_per_tok: u32,
    moe_intermediate_size: u32,
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

/// Test-only fixtures. Runtime dims always come from
/// [`MoeModelCfg::from_json`].
#[cfg(test)]
impl MoeModelCfg {
    /// Qwen3-235B-A22B preset (bf16). Aligned to ref
    /// `moesim-rs/src/workload/standard_moe.rs::qwen3_235b`.
    pub fn qwen3_235b() -> Self {
        Self {
            hidden: 4096,
            num_qo_heads: 64,
            num_kv_heads: 4,
            head_dim: 128,
            vocab: 152064,
            num_layers: 94,
            dtype: DType::Bf16,
            kv_dtype: DType::Bf16,
            fp8: false,
            num_experts: 128,
            top_k: 8,
            moe_intermediate: 3072,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_json_qwen3_moe_shape_with_head_dim_fallback() {
        let json = r#"{
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "vocab_size": 152064,
            "num_hidden_layers": 94,
            "torch_dtype": "bfloat16",
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 3072
        }"#;
        let dir = std::env::temp_dir();
        let path = dir.join(format!("vibesim_moe_cfg_{}.json", std::process::id()));
        std::fs::write(&path, json).unwrap();
        let cfg = MoeModelCfg::from_json(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(cfg.hidden, 4096);
        assert_eq!(cfg.num_qo_heads, 64);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.head_dim, 64); // 4096 / 64 fallback
        assert_eq!(cfg.vocab, 152064);
        assert_eq!(cfg.num_layers, 94);
        assert_eq!(cfg.num_experts, 128);
        assert_eq!(cfg.top_k, 8);
        assert_eq!(cfg.moe_intermediate, 3072);
        assert_eq!(cfg.dtype, DType::Bf16);
    }

    #[test]
    fn with_fp8_flips_compute_and_kv_dtype_but_not_base() {
        let bf16 = MoeModelCfg::qwen3_235b();
        assert!(!bf16.fp8);
        assert_eq!(bf16.compute_dtype(), DType::Bf16);
        assert_eq!(bf16.kv_dtype, DType::Bf16);

        let fp8 = MoeModelCfg::qwen3_235b().with_fp8(true);
        assert!(fp8.fp8);
        // Base dtype stays bf16 (norm / attn-o / decode-q keep it); compute + KV
        // move to fp8.
        assert_eq!(fp8.dtype, DType::Bf16);
        assert_eq!(fp8.compute_dtype(), DType::Fp8E4m3);
        assert_eq!(fp8.kv_dtype, DType::Fp8E4m3);
    }

    #[test]
    fn qwen3_235b_preset_matches_ref_constants() {
        let c = MoeModelCfg::qwen3_235b();
        assert_eq!(c.hidden, 4096);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.num_qo_heads, 64);
        assert_eq!(c.num_kv_heads, 4);
        assert_eq!(c.moe_intermediate, 3072);
        assert_eq!(c.num_experts, 128);
        assert_eq!(c.top_k, 8);
        assert_eq!(c.num_layers, 94);
    }
}
