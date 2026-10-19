//! The Qwen3.5-family hybrid text decoder (`Qwen3_5ForConditionalGeneration`,
//! model_type `qwen3_5`) as its checkpoint describes it. The config.json is kept
//! verbatim (`model/config/qwen3_8_27b_nvfp4.json` is
//! `nvidia/Qwen3.8-27B-NVFP4`).
//!
//! Read the way vLLM's `qwen3_5.py` / `qwen_gdn_linear_attn.py` read it:
//!
//! - `text_config.layer_types[layer]` is `linear_attention` (a Gated DeltaNet
//!   layer) or `full_attention` (a gated GQA layer: the q projection carries a
//!   per-head sigmoid output gate, `attn_output_gate`). Every layer ends in a
//!   dense SwiGLU MLP (`intermediate_size`).
//! - GDN: `linear_num_key_heads` q/k heads of `linear_key_head_dim`,
//!   `linear_num_value_heads` v heads of `linear_value_head_dim`, a
//!   `linear_conv_kernel_dim`-tap causal conv over the q/k/v channels, and a
//!   `mamba_ssm_dtype` recurrent state.
//! - GQA: RoPE rotates `head_dim * partial_rotary_factor` channels; an
//!   `mrope_section` makes it multimodal RoPE.
//! - `quantization_config` (ModelOpt `MIXED_PRECISION`): `quantized_layers`
//!   names each linear's `quant_algo`. vLLM fuses `in_proj_qkv` + `in_proj_z`
//!   and `q/k/v_proj`, so each fused group must agree; a module absent from the
//!   map is BF16 (`in_proj_ba`, the embedding).
//!
//! The vision tower and the MTP layer are not modeled.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

use crate::timing::bridge::DType;

/// One decoder layer's token mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen35Mixer {
    /// Gated DeltaNet (`linear_attention`).
    Gdn,
    /// Gated GQA (`full_attention`).
    Gqa,
}

/// How a checkpoint stores one linear's weights (and so which GEMM runs it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen35LinearPrecision {
    Bf16,
    /// ModelOpt FP8: per-tensor static weight and activation scales.
    Fp8Tensor,
    /// ModelOpt NVFP4: 16-element FP4 groups with E4M3 scales.
    Nvfp4,
}

impl Qwen35LinearPrecision {
    fn from_algo(algo: &str) -> Result<Self> {
        Ok(match algo {
            "FP8" => Self::Fp8Tensor,
            "NVFP4" => Self::Nvfp4,
            other => bail!("Qwen3.5 config: unmodeled quant_algo {other:?}"),
        })
    }

    pub fn gemm_dtype(self) -> DType {
        match self {
            Self::Bf16 => DType::Bf16,
            Self::Fp8Tensor => DType::Fp8E4m3,
            Self::Nvfp4 => DType::Nvfp4E2m1,
        }
    }
}

/// The precision of each fused linear vLLM runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen35Precisions {
    /// GDN `in_proj_qkvz` (`in_proj_qkv` + `in_proj_z`).
    pub gdn_in_proj_qkvz: Qwen35LinearPrecision,
    /// GDN `in_proj_ba`.
    pub gdn_in_proj_ba: Qwen35LinearPrecision,
    pub gdn_out_proj: Qwen35LinearPrecision,
    /// GQA `qkv_proj` (`q_proj` with its gate half, `k_proj`, `v_proj`).
    pub gqa_qkv_proj: Qwen35LinearPrecision,
    pub gqa_o_proj: Qwen35LinearPrecision,
    /// MLP `gate_up_proj` and `down_proj`.
    pub mlp: Qwen35LinearPrecision,
    pub lm_head: Qwen35LinearPrecision,
}

/// The checkpoint's text-model dimensions, read strictly from its config.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35ModelCfg {
    pub hidden: u32,
    pub intermediate: u32,
    pub vocab_size: u32,
    pub mixers: Vec<Qwen35Mixer>,
    pub gqa_heads: u32,
    pub gqa_kv_heads: u32,
    pub gqa_head_dim: u32,
    pub rotary_dim: u32,
    pub mrope: bool,
    pub gdn_key_heads: u32,
    pub gdn_value_heads: u32,
    pub gdn_key_head_dim: u32,
    pub gdn_value_head_dim: u32,
    pub gdn_conv_kernel: u32,
    pub gdn_state_dtype: DType,
    pub precisions: Qwen35Precisions,
}

impl Qwen35ModelCfg {
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading Qwen3.5 config {}", path.display()))?;
        let root: Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing Qwen3.5 config {}", path.display()))?;
        Self::from_value(&root)
    }

    pub fn from_value(root: &Value) -> Result<Self> {
        let cfg = root
            .get("text_config")
            .context("Qwen3.5 config has no text_config")?;
        let u = |key: &str| -> Result<u32> {
            let value = cfg
                .get(key)
                .and_then(Value::as_u64)
                .with_context(|| format!("Qwen3.5 text_config.{key} must be an integer"))?;
            u32::try_from(value).with_context(|| format!("text_config.{key} exceeds u32"))
        };
        let text = |key: &str| cfg.get(key).and_then(Value::as_str).unwrap_or("");
        ensure!(
            text("model_type") == "qwen3_5_text",
            "Qwen3.5 arches require a qwen3_5_text config, got model_type {:?}",
            text("model_type")
        );
        ensure!(
            cfg.get("attn_output_gate").and_then(Value::as_bool) == Some(true),
            "Qwen3.5 arches model the gated full-attention output"
        );
        ensure!(
            text("hidden_act") == "silu",
            "Qwen3.5 arches model a SiLU MLP"
        );
        let num_layers = u("num_hidden_layers")?;
        let mixers = cfg
            .get("layer_types")
            .and_then(Value::as_array)
            .context("Qwen3.5 text_config.layer_types must be an array")?
            .iter()
            .map(|entry| match entry.as_str() {
                Some("linear_attention") => Ok(Qwen35Mixer::Gdn),
                Some("full_attention") => Ok(Qwen35Mixer::Gqa),
                other => bail!("text_config.layer_types entry {other:?} is not modeled"),
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            mixers.len() == num_layers as usize,
            "text_config.layer_types has {} entries for {num_layers} layers",
            mixers.len()
        );
        let head_dim = u("head_dim")?;
        let rope = cfg.get("rope_parameters");
        let partial = rope
            .and_then(|r| r.get("partial_rotary_factor"))
            .or_else(|| cfg.get("partial_rotary_factor"))
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        let rotary = f64::from(head_dim) * partial;
        ensure!(
            rotary.fract() == 0.0,
            "head_dim {head_dim} x partial_rotary_factor {partial} is not whole"
        );
        let mrope = rope.and_then(|r| r.get("mrope_section")).is_some();
        let gdn_state_dtype = match text("mamba_ssm_dtype") {
            "float32" => DType::Fp32,
            "bfloat16" => DType::Bf16,
            other => bail!("text_config.mamba_ssm_dtype {other:?} is not modeled"),
        };
        Ok(Self {
            hidden: u("hidden_size")?,
            intermediate: u("intermediate_size")?,
            vocab_size: u("vocab_size")?,
            gqa_heads: u("num_attention_heads")?,
            gqa_kv_heads: u("num_key_value_heads")?,
            gqa_head_dim: head_dim,
            rotary_dim: rotary as u32,
            mrope,
            gdn_key_heads: u("linear_num_key_heads")?,
            gdn_value_heads: u("linear_num_value_heads")?,
            gdn_key_head_dim: u("linear_key_head_dim")?,
            gdn_value_head_dim: u("linear_value_head_dim")?,
            gdn_conv_kernel: u("linear_conv_kernel_dim")?,
            gdn_state_dtype,
            precisions: precisions(root.get("quantization_config"), &mixers)?,
            mixers,
        })
    }

    pub fn num_layers(&self) -> u32 {
        self.mixers.len() as u32
    }

    /// The q/k/v channels the GDN causal conv runs over (and caches).
    pub fn gdn_conv_channels(&self) -> u32 {
        2 * self.gdn_key_heads * self.gdn_key_head_dim
            + self.gdn_value_heads * self.gdn_value_head_dim
    }

    /// One GDN layer's value width (`core_attn_out`, `z`, the out_proj input).
    pub fn gdn_value_width(&self) -> u32 {
        self.gdn_value_heads * self.gdn_value_head_dim
    }
}

/// Each fused linear's precision from ModelOpt's `quantized_layers`. Every
/// layer of one module must agree (the arch costs one GEMM per layer group),
/// as must the modules vLLM fuses.
fn precisions(quant: Option<&Value>, mixers: &[Qwen35Mixer]) -> Result<Qwen35Precisions> {
    let layers: BTreeMap<String, Qwen35LinearPrecision> = match quant {
        None => BTreeMap::new(),
        Some(quant) => {
            let method = quant.get("quant_method").and_then(Value::as_str);
            ensure!(
                method == Some("modelopt"),
                "Qwen3.5 config: unmodeled quant_method {method:?}"
            );
            quant
                .get("quantized_layers")
                .and_then(Value::as_object)
                .context("Qwen3.5 config: modelopt quantization_config needs quantized_layers")?
                .iter()
                .map(|(module, entry)| {
                    let algo = entry
                        .get("quant_algo")
                        .and_then(Value::as_str)
                        .with_context(|| format!("quantized_layers.{module} has no quant_algo"))?;
                    Ok((module.clone(), Qwen35LinearPrecision::from_algo(algo)?))
                })
                .collect::<Result<_>>()?
        }
    };
    let of = |module: &str| {
        layers
            .get(module)
            .copied()
            .unwrap_or(Qwen35LinearPrecision::Bf16)
    };
    // One precision for `suffixes` over every layer with `mixer`.
    let layer_module = |mixer: Option<Qwen35Mixer>, suffixes: &[&str]| -> Result<_> {
        let mut found: Option<Qwen35LinearPrecision> = None;
        for (layer, &kind) in mixers.iter().enumerate() {
            if mixer.is_some_and(|m| m != kind) {
                continue;
            }
            for suffix in suffixes {
                let precision = of(&format!("model.language_model.layers.{layer}.{suffix}"));
                match found {
                    None => found = Some(precision),
                    Some(seen) => ensure!(
                        seen == precision,
                        "{suffix} of layer {layer} is {precision:?}, other layers or fused \
                         modules {seen:?}"
                    ),
                }
            }
        }
        Ok(found.unwrap_or(Qwen35LinearPrecision::Bf16))
    };
    let gdn = Some(Qwen35Mixer::Gdn);
    let gqa = Some(Qwen35Mixer::Gqa);
    Ok(Qwen35Precisions {
        gdn_in_proj_qkvz: layer_module(gdn, &["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"])?,
        gdn_in_proj_ba: layer_module(gdn, &["linear_attn.in_proj_b", "linear_attn.in_proj_a"])?,
        gdn_out_proj: layer_module(gdn, &["linear_attn.out_proj"])?,
        gqa_qkv_proj: layer_module(
            gqa,
            &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        )?,
        gqa_o_proj: layer_module(gqa, &["self_attn.o_proj"])?,
        mlp: layer_module(None, &["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"])?,
        lm_head: of("lm_head"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen38() -> Qwen35ModelCfg {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/qwen3_8_27b_nvfp4.json");
        Qwen35ModelCfg::from_json(&path).unwrap()
    }

    #[test]
    fn qwen38_27b_nvfp4_reads_its_shape_and_mixed_precision() {
        let cfg = qwen38();
        assert_eq!(
            (cfg.hidden, cfg.intermediate, cfg.vocab_size),
            (5120, 17408, 248320)
        );
        assert_eq!(cfg.num_layers(), 64);
        let gqa_layers: Vec<usize> = (0..cfg.mixers.len())
            .filter(|&l| cfg.mixers[l] == Qwen35Mixer::Gqa)
            .collect();
        assert_eq!(gqa_layers, (3..64).step_by(4).collect::<Vec<_>>());
        assert_eq!(
            (cfg.gqa_heads, cfg.gqa_kv_heads, cfg.gqa_head_dim),
            (24, 4, 256)
        );
        assert_eq!((cfg.rotary_dim, cfg.mrope), (64, true));
        assert_eq!(cfg.gdn_conv_channels(), 10240);
        assert_eq!(cfg.gdn_value_width(), 6144);
        assert_eq!(cfg.gdn_state_dtype, DType::Fp32);
        use Qwen35LinearPrecision::*;
        assert_eq!(
            cfg.precisions,
            Qwen35Precisions {
                gdn_in_proj_qkvz: Fp8Tensor,
                gdn_in_proj_ba: Bf16,
                gdn_out_proj: Fp8Tensor,
                gqa_qkv_proj: Fp8Tensor,
                gqa_o_proj: Fp8Tensor,
                mlp: Nvfp4,
                lm_head: Nvfp4,
            }
        );
    }

    #[test]
    fn an_unquantized_config_is_bf16_throughout() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/qwen3_6_27b.json");
        let cfg = Qwen35ModelCfg::from_json(&path).unwrap();
        assert_eq!(cfg.precisions.mlp, Qwen35LinearPrecision::Bf16);
        assert_eq!(cfg.precisions.gdn_in_proj_qkvz, Qwen35LinearPrecision::Bf16);
    }

    #[test]
    fn a_layer_quantized_differently_from_its_peers_is_refused() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/qwen3_8_27b_nvfp4.json");
        let mut root: Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        root["quantization_config"]["quantized_layers"]
            .as_object_mut()
            .unwrap()
            .remove("model.language_model.layers.5.mlp.up_proj");
        let error = Qwen35ModelCfg::from_value(&root).unwrap_err().to_string();
        assert!(error.contains("mlp.up_proj"), "{error}");
    }
}
