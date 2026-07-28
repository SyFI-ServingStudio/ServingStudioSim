//! Numeric model dims for the Kimi-K3 hybrid KDA+MLA MoE arch — the
//! parallelism-agnostic identity of one Kimi-K3-style model. Mirrors
//! [`MoeModelCfg`](crate::arch::moe_model_cfg::MoeModelCfg) but folds in the
//! Kimi-specific attention split (MLA layers vs KDA linear-attention layers,
//! the MLA LoRA/compressed-KV dims, the KDA short-conv width) and the MoE
//! extras (`routed_expert_hidden_size`, shared experts, the layer-0 dense FFN).
//!
//! [`Self::from_json`] parses `model/config/kimi_k3.json` — real HF keys where
//! they exist (`kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`,
//! `num_experts`, …) plus the custom `full_attn_layer_count` key that fixes how
//! many of `num_hidden_layers` are MLA (full-attention) layers; the remainder
//! are KDA layers.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::timing::bridge::DType;

/// Raw Kimi-K3 transformer dims (per-model identity, parallelism-agnostic).
#[derive(Clone, Debug)]
pub struct KimiModelCfg {
    pub hidden: u32,
    /// Attention heads — shared by the MLA layers (96 qo heads) and the KDA
    /// layers (96 linear-attention heads).
    pub num_heads: u32,
    /// Per-head value / nope dim (128), shared by MLA `v_head_dim` and the KDA
    /// head_dim.
    pub head_dim: u32,
    pub vocab: u32,
    pub num_layers: u32,
    /// MLA (full-attention) layer count; the remaining
    /// `num_layers - mla_layers` layers are KDA.
    pub mla_layers: u32,
    // ── MLA dims ──
    /// Compressed-KV latent rank (512). The KV cache holds ONE
    /// `kv_lora_rank + qk_rope_head_dim` (= 576) vector per token per MLA layer
    /// (K and V share it).
    pub kv_lora_rank: u32,
    /// Query LoRA rank (1536).
    pub q_lora_rank: u32,
    /// Decoupled RoPE head dim (64).
    pub qk_rope_head_dim: u32,
    /// Non-RoPE query/key head dim (128); full qk head dim is nope + rope = 192.
    pub qk_nope_head_dim: u32,
    // ── KDA dims ──
    /// KDA short-convolution kernel width (4).
    pub short_conv_kernel_size: u32,
    // ── MoE dims ──
    /// Total routed expert count across all EP ranks (896).
    pub num_experts: u32,
    /// Experts each token selects (16).
    pub top_k: u32,
    /// Per-expert intermediate dim (3072).
    pub moe_intermediate: u32,
    /// Routed-expert input/output hidden dim (3584): expert GEMMs run at this
    /// REDUCED hidden, not the model `hidden` (7168).
    pub expert_hidden: u32,
    /// Always-on shared experts (2), each a dense FFN at `moe_intermediate`
    /// width on the full `hidden`.
    pub n_shared_experts: u32,
    /// Layers replaced by a plain dense FFN at the front of the stack (1).
    pub first_k_dense_replace: u32,
    /// The layer-0 dense FFN intermediate size (33792).
    pub dense_intermediate: u32,
    /// The model's native dtype (bf16). 16-bit-holdout ops (RMSNorm, attention
    /// output, decode query) keep it in an FP8 run.
    pub dtype: DType,
    /// Compressed-KV cache dtype — FP8 in an FP8 run, else `dtype`.
    pub kv_dtype: DType,
    /// Whether this run executes in FP8. Drives GEMM backend selection and
    /// [`Self::compute_dtype`].
    pub fp8: bool,
}

impl KimiModelCfg {
    /// Load from `model/config/kimi_k3.json` (HF-style keys + the custom
    /// `full_attn_layer_count`). All keys are required except `head_dim`
    /// (falls back to `v_head_dim`).
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading model config {}", path.display()))?;
        let raw: JsonKimiModelConfig = serde_json::from_str(&text)
            .with_context(|| format!("parsing Kimi-K3 model config {}", path.display()))?;
        let dtype = parse_dtype(&raw.torch_dtype)?;
        if raw.full_attn_layer_count > raw.num_hidden_layers {
            bail!(
                "full_attn_layer_count {} exceeds num_hidden_layers {}",
                raw.full_attn_layer_count,
                raw.num_hidden_layers
            );
        }
        Ok(Self {
            hidden: raw.hidden_size,
            num_heads: raw.num_attention_heads,
            head_dim: raw.head_dim.unwrap_or(raw.v_head_dim),
            vocab: raw.vocab_size,
            num_layers: raw.num_hidden_layers,
            mla_layers: raw.full_attn_layer_count,
            kv_lora_rank: raw.kv_lora_rank,
            q_lora_rank: raw.q_lora_rank,
            qk_rope_head_dim: raw.qk_rope_head_dim,
            qk_nope_head_dim: raw.qk_nope_head_dim,
            short_conv_kernel_size: raw.short_conv_kernel_size,
            num_experts: raw.num_experts,
            top_k: raw.num_experts_per_tok,
            moe_intermediate: raw.moe_intermediate_size,
            expert_hidden: raw.routed_expert_hidden_size,
            n_shared_experts: raw.n_shared_experts,
            first_k_dense_replace: raw.first_k_dense_replace,
            dense_intermediate: raw.intermediate_size,
            dtype,
            kv_dtype: dtype,
            fp8: false,
        })
    }

    /// KDA (linear-attention) layer count: every non-MLA layer.
    pub fn kda_layers(&self) -> u32 {
        self.num_layers - self.mla_layers
    }

    /// The per-token compressed-KV vector width of one MLA layer:
    /// `kv_lora_rank + qk_rope_head_dim` (= 576). This is also the decode MQA
    /// head_dim (absorbed-weight decode attends over the compressed cache with
    /// ONE shared KV head of this width).
    pub fn kv_compressed_dim(&self) -> u32 {
        self.kv_lora_rank + self.qk_rope_head_dim
    }

    /// The full per-head query/key dim of the decompressed MLA prefill:
    /// `qk_nope_head_dim + qk_rope_head_dim` (= 192).
    pub fn qk_head_dim(&self) -> u32 {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Truncate the layer COUNT (the `sim_num_layers` / `num_layers` override).
    /// The MLA/KDA split is scaled proportionally (¼ of the real stack is MLA),
    /// keeping at least one layer of each kind when `n >= 2`.
    pub fn with_num_layers(mut self, n: u32) -> Self {
        let mla = if n <= 1 {
            n
        } else {
            (n * self.mla_layers / self.num_layers).clamp(1, n - 1)
        };
        self.num_layers = n;
        self.mla_layers = mla;
        self
    }

    /// The compute dtype for GEMMs and prefill attention q/kv: FP8 in an FP8
    /// run, else the model's base `dtype`.
    pub fn compute_dtype(&self) -> DType {
        if self.fp8 {
            DType::Fp8E4m3
        } else {
            self.dtype
        }
    }

    /// Turn FP8 on/off. FP8 moves the compressed-KV cache to FP8; the base
    /// `dtype` stays bf16 for the 16-bit-holdout ops.
    pub fn with_fp8(mut self, fp8: bool) -> Self {
        self.fp8 = fp8;
        if fp8 {
            self.kv_dtype = DType::Fp8E4m3;
        }
        self
    }

    /// Candidate implementations for dense single GEMMs (mirrors `MoeModelCfg`).
    pub fn single_gemm_backends(&self) -> Vec<&'static str> {
        if self.fp8 {
            vec!["deepgemm"]
        } else {
            vec!["torch", "torch_linear"]
        }
    }

    /// Grouped expert GEMM backends (mirrors `MoeModelCfg`).
    pub fn grouped_gemm_backends(&self) -> Vec<&'static str> {
        if self.fp8 {
            vec!["deepgemm"]
        } else {
            vec!["torch"]
        }
    }
}

/// `model/config/kimi_k3.json` shape (only the fields the arch needs).
#[derive(Deserialize)]
struct JsonKimiModelConfig {
    hidden_size: u32,
    num_attention_heads: u32,
    #[serde(default)]
    head_dim: Option<u32>,
    vocab_size: u32,
    num_hidden_layers: u32,
    torch_dtype: String,
    kv_lora_rank: u32,
    q_lora_rank: u32,
    qk_rope_head_dim: u32,
    qk_nope_head_dim: u32,
    v_head_dim: u32,
    full_attn_layer_count: u32,
    short_conv_kernel_size: u32,
    num_experts: u32,
    num_experts_per_tok: u32,
    moe_intermediate_size: u32,
    routed_expert_hidden_size: u32,
    n_shared_experts: u32,
    first_k_dense_replace: u32,
    intermediate_size: u32,
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

/// Test-only fixture. Runtime dims always come from [`KimiModelCfg::from_json`].
#[cfg(test)]
impl KimiModelCfg {
    /// Kimi-K3 preset (bf16) — mirrors `model/config/kimi_k3.json`.
    pub fn kimi_k3() -> Self {
        Self {
            hidden: 7168,
            num_heads: 96,
            head_dim: 128,
            vocab: 163840,
            num_layers: 93,
            mla_layers: 24,
            kv_lora_rank: 512,
            q_lora_rank: 1536,
            qk_rope_head_dim: 64,
            qk_nope_head_dim: 128,
            short_conv_kernel_size: 4,
            num_experts: 896,
            top_k: 16,
            moe_intermediate: 3072,
            expert_hidden: 3584,
            n_shared_experts: 2,
            first_k_dense_replace: 1,
            dense_intermediate: 33792,
            dtype: DType::Bf16,
            kv_dtype: DType::Bf16,
            fp8: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_json_parses_the_checked_in_kimi_k3_config() {
        // The real checked-in config file is the contract; parse it directly.
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/kimi_k3.json");
        let cfg = KimiModelCfg::from_json(&path).unwrap();
        assert_eq!(cfg.hidden, 7168);
        assert_eq!(cfg.num_heads, 96);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.vocab, 163840);
        assert_eq!(cfg.num_layers, 93);
        assert_eq!(cfg.mla_layers, 24);
        assert_eq!(cfg.kda_layers(), 69);
        assert_eq!(cfg.kv_lora_rank, 512);
        assert_eq!(cfg.q_lora_rank, 1536);
        assert_eq!(cfg.kv_compressed_dim(), 576);
        assert_eq!(cfg.qk_head_dim(), 192);
        assert_eq!(cfg.short_conv_kernel_size, 4);
        assert_eq!(cfg.num_experts, 896);
        assert_eq!(cfg.top_k, 16);
        assert_eq!(cfg.moe_intermediate, 3072);
        assert_eq!(cfg.expert_hidden, 3584);
        assert_eq!(cfg.n_shared_experts, 2);
        assert_eq!(cfg.first_k_dense_replace, 1);
        assert_eq!(cfg.dense_intermediate, 33792);
        assert_eq!(cfg.dtype, DType::Bf16);
    }

    #[test]
    fn with_num_layers_scales_the_mla_kda_split() {
        // 93 → 8 layers: 8·24/93 = 2 MLA + 6 KDA.
        let cfg = KimiModelCfg::kimi_k3().with_num_layers(8);
        assert_eq!(cfg.num_layers, 8);
        assert_eq!(cfg.mla_layers, 2);
        assert_eq!(cfg.kda_layers(), 6);
        // Degenerate truncations keep at least one of each kind when possible.
        let two = KimiModelCfg::kimi_k3().with_num_layers(2);
        assert_eq!(two.mla_layers, 1);
        assert_eq!(two.kda_layers(), 1);
    }

    #[test]
    fn with_fp8_flips_compute_and_kv_dtype_but_not_base() {
        let fp8 = KimiModelCfg::kimi_k3().with_fp8(true);
        assert_eq!(fp8.dtype, DType::Bf16);
        assert_eq!(fp8.compute_dtype(), DType::Fp8E4m3);
        assert_eq!(fp8.kv_dtype, DType::Fp8E4m3);
        assert_eq!(fp8.single_gemm_backends(), vec!["deepgemm"]);
    }
}
