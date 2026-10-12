//! The MiMo-V2 text model (`MiMoV2ForCausalLM`, model_type `mimo_v2`) as its
//! checkpoint describes it (`XiaomiMiMo/MiMo-V2.6-Flash-RL`, config.json kept
//! verbatim as `model/config/mimo_v2_6_flash_rl.json`).
//!
//! Read the way vLLM's `vllm/model_executor/models/mimo_v2.py` reads it:
//!
//! - `hybrid_layer_pattern[layer] == 1` is a sliding-window layer (`swa_*`
//!   heads and dims, `sliding_window_size`, sink logits when
//!   `add_swa_attention_sink_bias`); 0 is a global layer (`num_attention_heads`
//!   etc., sinks when `add_full_attention_sink_bias`).
//! - `moe_layer_freq[layer]` truthy is a routed-MoE layer
//!   (`moe_intermediate_size`), else a dense SwiGLU MLP (`intermediate_size`).
//! - RoPE rotates the first `int(head_dim * partial_rotary_factor)` channels
//!   (`get_rope`), and V is scaled by `attention_value_scale`.
//! - `quantization_config` (`quant_method: fp8`): 128x128 block FP8 linears,
//!   routed experts stored MXFP4 (`store_dtype`, `mxfp4_block_size`), and the
//!   `ignored_layers` (every `o_proj`) left BF16. The router (`GateLinear`)
//!   holds `moe_router_dtype` weights and writes FP32 logits.
//!
//! The vision tower, the audio stack and the MTP layers are not modeled.

use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

/// One attention layer type's shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MimoV2AttnShape {
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub v_head_dim: u32,
    /// Learnable per-head sink logits in the softmax denominator.
    pub sinks: bool,
}

/// The checkpoint's text-model dimensions, read strictly from its config.
#[derive(Clone, Debug, PartialEq)]
pub struct MimoV2ModelCfg {
    pub hidden: u32,
    pub num_layers: u32,
    pub vocab_size: u32,
    pub global_attn: MimoV2AttnShape,
    pub swa_attn: MimoV2AttnShape,
    pub sliding_window_size: u32,
    /// Sliding-window layers; the rest are global.
    pub swa_layer_ids: Vec<u32>,
    /// Channels RoPE rotates per head (the same on both layer types, whose
    /// head dims are equal in every released checkpoint).
    pub rotary_dim: u32,
    /// Whether `v * attention_value_scale` runs (fused with RoPE).
    pub scales_v: bool,
    /// Routed-MoE layers; the rest carry the dense MLP.
    pub moe_layer_ids: Vec<u32>,
    pub dense_intermediate_size: u32,
    pub moe_intermediate_size: u32,
    pub n_routed_experts: u32,
    pub num_experts_per_tok: u32,
    pub n_group: u32,
    pub topk_group: u32,
    /// `routed_scaling_factor` as an exact fraction (`null` = 1).
    pub routed_scaling: (u32, u32),
    /// The FP8 linears' block edge (`weight_block_size`), also the activation
    /// quant group.
    pub fp8_block_size: u32,
    /// Elements per MXFP4 expert-weight scale.
    pub mxfp4_block_size: u32,
}

impl MimoV2ModelCfg {
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading MiMo-V2 config {}", path.display()))?;
        let root: Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing MiMo-V2 config {}", path.display()))?;
        Self::from_value(&root)
    }

    pub fn from_value(cfg: &Value) -> Result<Self> {
        let u = |key: &str| -> Result<u32> {
            let value = cfg
                .get(key)
                .and_then(Value::as_u64)
                .with_context(|| format!("MiMo-V2 config.{key} must be an integer"))?;
            u32::try_from(value).with_context(|| format!("config.{key} exceeds u32"))
        };
        let flag = |key: &str| -> Result<bool> {
            cfg.get(key)
                .and_then(Value::as_bool)
                .with_context(|| format!("MiMo-V2 config.{key} must be a boolean"))
        };
        let text = |key: &str| cfg.get(key).and_then(Value::as_str).unwrap_or("");
        ensure!(
            text("model_type") == "mimo_v2",
            "MiMo-V2 arches require a mimo_v2 config, got model_type {:?}",
            text("model_type")
        );
        ensure!(
            text("scoring_func") == "sigmoid" && text("topk_method") == "noaux_tc",
            "MiMo-V2 arches model sigmoid scoring with noaux_tc (bias-corrected) top-k"
        );
        ensure!(
            !flag("attention_bias")?,
            "MiMo-V2 arches model bias-free QKV"
        );
        let num_layers = u("num_hidden_layers")?;
        let layer_flags = |key: &str| -> Result<Vec<u32>> {
            let entries = cfg
                .get(key)
                .and_then(Value::as_array)
                .with_context(|| format!("MiMo-V2 config.{key} must be an array"))?;
            ensure!(
                entries.len() == num_layers as usize,
                "config.{key} has {} entries for {num_layers} layers",
                entries.len()
            );
            entries
                .iter()
                .enumerate()
                .filter_map(|(layer, entry)| match entry.as_u64() {
                    Some(0) => None,
                    Some(1) => Some(Ok(layer as u32)),
                    _ => Some(Err(anyhow::anyhow!("config.{key} entries must be 0 or 1"))),
                })
                .collect()
        };
        let attn = |prefix: &str, v_key: &str, sink_key: &str| -> Result<MimoV2AttnShape> {
            Ok(MimoV2AttnShape {
                num_attention_heads: u(&format!("{prefix}num_attention_heads"))?,
                num_key_value_heads: u(&format!("{prefix}num_key_value_heads"))?,
                head_dim: u(&format!("{prefix}head_dim"))?,
                v_head_dim: u(v_key)?,
                sinks: flag(sink_key)?,
            })
        };
        let global_attn = attn("", "v_head_dim", "add_full_attention_sink_bias")?;
        let swa_attn = attn("swa_", "swa_v_head_dim", "add_swa_attention_sink_bias")?;
        ensure!(
            global_attn.head_dim == swa_attn.head_dim,
            "MiMo-V2 arches assume one QK head dim for both layer types (one RoPE width)"
        );
        let partial_rotary_factor = cfg
            .get("partial_rotary_factor")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        // vLLM `get_rope`: `int(head_size * partial_rotary_factor)`.
        let rotary_dim = (f64::from(global_attn.head_dim) * partial_rotary_factor) as u32;
        let routed_scaling = match cfg.get("routed_scaling_factor") {
            None | Some(Value::Null) => (1, 1),
            Some(value) => {
                let factor = value
                    .as_f64()
                    .context("routed_scaling_factor must be a number or null")?;
                ensure!(
                    factor > 0.0 && factor.fract() == 0.0,
                    "routed_scaling_factor {factor} must be a positive integer"
                );
                (factor as u32, 1)
            }
        };
        let quant = cfg
            .get("quantization_config")
            .context("MiMo-V2 arches model the FP8 checkpoint's quantization_config")?;
        ensure!(
            quant.get("quant_method").and_then(Value::as_str) == Some("fp8")
                && quant.get("store_dtype").and_then(Value::as_str) == Some("mxfp4"),
            "MiMo-V2 arches model block-FP8 linears with MXFP4-stored experts"
        );
        let block = quant
            .get("weight_block_size")
            .and_then(Value::as_array)
            .context("quantization_config.weight_block_size must be an array")?;
        let fp8_block_size = match block.as_slice() {
            [n, k] if n.as_u64() == k.as_u64() => n
                .as_u64()
                .and_then(|edge| u32::try_from(edge).ok())
                .context("weight_block_size entries must be u32")?,
            _ => bail!("weight_block_size must be a square [n, k] block"),
        };
        let mxfp4_block_size = quant
            .get("mxfp4_block_size")
            .and_then(Value::as_u64)
            .and_then(|size| u32::try_from(size).ok())
            .context("quantization_config.mxfp4_block_size must be an integer")?;
        let ignored: Vec<&str> = quant
            .get("ignored_layers")
            .and_then(Value::as_array)
            .map(|layers| layers.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        ensure!(
            (0..num_layers).all(|layer| ignored
                .contains(&format!("model.layers.{layer}.self_attn.o_proj").as_str())),
            "MiMo-V2 arches model a BF16 o_proj on every layer (quantization_config.ignored_layers)"
        );
        ensure!(
            text("moe_router_dtype") == "bfloat16",
            "MiMo-V2 arches model the BF16 router (moe_router_dtype)"
        );
        Ok(Self {
            hidden: u("hidden_size")?,
            num_layers,
            vocab_size: u("vocab_size")?,
            global_attn,
            swa_attn,
            sliding_window_size: u("sliding_window_size")?,
            swa_layer_ids: layer_flags("hybrid_layer_pattern")?,
            rotary_dim,
            scales_v: cfg
                .get("attention_value_scale")
                .is_some_and(|scale| !scale.is_null()),
            moe_layer_ids: layer_flags("moe_layer_freq")?,
            dense_intermediate_size: u("intermediate_size")?,
            moe_intermediate_size: u("moe_intermediate_size")?,
            n_routed_experts: u("n_routed_experts")?,
            num_experts_per_tok: u("num_experts_per_tok")?,
            n_group: u("n_group")?,
            topk_group: u("topk_group")?,
            routed_scaling,
            fp8_block_size,
            mxfp4_block_size,
        })
    }

    pub fn is_swa(&self, layer: u32) -> bool {
        self.swa_layer_ids.contains(&layer)
    }

    pub fn is_moe(&self, layer: u32) -> bool {
        self.moe_layer_ids.contains(&layer)
    }

    pub fn attn(&self, layer: u32) -> &MimoV2AttnShape {
        if self.is_swa(layer) {
            &self.swa_attn
        } else {
            &self.global_attn
        }
    }

    pub fn num_moe_layers(&self) -> u32 {
        self.moe_layer_ids.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> MimoV2ModelCfg {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/mimo_v2_6_flash_rl.json");
        MimoV2ModelCfg::from_json(&path).unwrap()
    }

    #[test]
    fn reads_the_released_checkpoint() {
        let cfg = cfg();
        assert_eq!(
            (cfg.hidden, cfg.num_layers, cfg.vocab_size),
            (4096, 48, 152576)
        );
        let global: Vec<u32> = (0..48).filter(|&layer| !cfg.is_swa(layer)).collect();
        assert_eq!(global, [0, 5, 11, 17, 23, 29, 35, 41, 47]);
        assert_eq!(
            cfg.global_attn,
            MimoV2AttnShape {
                num_attention_heads: 64,
                num_key_value_heads: 4,
                head_dim: 192,
                v_head_dim: 128,
                sinks: false,
            }
        );
        assert_eq!(
            (cfg.swa_attn.num_key_value_heads, cfg.swa_attn.sinks),
            (8, true)
        );
        assert_eq!((cfg.sliding_window_size, cfg.rotary_dim), (128, 64));
        assert!(cfg.scales_v);
        assert_eq!(cfg.num_moe_layers(), 47);
        assert!(!cfg.is_moe(0) && cfg.is_moe(1) && cfg.is_moe(47));
        assert_eq!(
            (
                cfg.n_routed_experts,
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size
            ),
            (256, 8, 2048)
        );
        assert_eq!(cfg.dense_intermediate_size, 16384);
        assert_eq!(
            (cfg.n_group, cfg.topk_group, cfg.routed_scaling),
            (1, 1, (1, 1))
        );
        assert_eq!((cfg.fp8_block_size, cfg.mxfp4_block_size), (128, 32));
    }

    #[test]
    fn a_quantized_o_proj_is_refused() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/mimo_v2_6_flash_rl.json");
        let mut root: Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        root["quantization_config"]["ignored_layers"] = Value::Array(Vec::new());
        let error = MimoV2ModelCfg::from_value(&root).unwrap_err();
        assert!(error.to_string().contains("o_proj"), "{error}");
    }
}
