//! The Qwen4Exp text decoder (`Qwen4ExpForConditionalGeneration`, model_type
//! `qwen4_exp`) as its checkpoint describes it. The config.json is kept
//! verbatim (`model/config/qwen3_8_flash_next_nvfp4.json` is
//! `nvidia/Qwen3.8-Flash-Next-NVFP4` fc694b54).
//!
//! Read the way the vLLM fork's `models/qwen4_exp/nvidia/` reads it:
//!
//! - `text_config.layer_types[layer]` is `linear_attention` (a Gated DeltaNet
//!   layer) or `full_attention` (a QSA layer: sparse gated GQA whose rows attend
//!   the token groups a weight-free indexer selects). Every layer ends in a MoE
//!   block with a sigmoid-gated shared expert, and the residual is `hc_count`
//!   hyper-connection streams.
//! - QSA: RoPE rotates `head_dim * partial_rotary_factor` channels; the indexer
//!   keeps one key per `indexer_compress_ratio` tokens and selects
//!   `indexer_budget` tokens (`indexer_budget / ratio` groups). The check
//!   that this is 512 or 2048 mirrors the fork's own config assert
//!   (`config.py:163-168`), not a modeling limit.
//! - PLE: `ple_layer_ids` are 1-based. Each holds `(ngram_size - 1) *
//!   heads_per_ngram` n-gram heads; head `h` of dense PLE layer `d` has the
//!   `(d * heads + h + 1)`-th prime after `ngram_vocab_size_base - 1` rows,
//!   and the summed table is padded to `make_ngram_vocab_size_divisible_by`
//!   (`ngram_embedding.py:614-694`). The short conv's dilation is `ngram_size`
//!   (`ple_layer.py:92`). The arches price one table, so at most one PLE
//!   layer is accepted (a second would bring its own, larger table).
//! - MTP: `mtp_num_hidden_layers` QSA layers. With
//!   `mtp_use_dedicated_embeddings = false` the drafter reuses the target's
//!   embedding and lm_head.
//! - `quantization_config` (ModelOpt `MIXED_PRECISION`): the modeled mix is
//!   NVFP4 (group 16) body experts, W8A8 FP8 block-128 MTP experts, an FP8
//!   per-tensor PLE table (BF16 when the map leaves it out) and BF16
//!   everywhere else. Any other mix is refused.
//!
//! The vision tower is not modeled (`--language-model-only`, which also strips
//! the multimodal RoPE sections).

use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::timing::bridge::DType;

/// One decoder layer's token mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4ExpMixer {
    /// Gated DeltaNet (`linear_attention`).
    Gdn,
    /// Sparse gated GQA with an indexer (`full_attention`).
    Qsa,
}

/// vLLM `--kv-cache-dtype` / the QSA indexer's `indexer_kv_dtype`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qwen4ExpCacheDtype {
    #[default]
    Fp8,
    Bf16,
}

impl Qwen4ExpCacheDtype {
    pub const CHOICES: [&'static str; 2] = ["fp8", "bf16"];

    pub fn dtype(self) -> DType {
        match self {
            Self::Fp8 => DType::Fp8E4m3,
            Self::Bf16 => DType::Bf16,
        }
    }
}

/// Where the PLE n-gram table lives: pinned host memory read over UVA
/// (`--engram-config '{"cpu_offload": true}'`, the fork's default) or HBM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PleTableResidency {
    #[default]
    HostUva,
    Device,
}

impl PleTableResidency {
    pub const CHOICES: [&'static str; 2] = ["host_uva", "device"];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::HostUva => "host_uva",
            Self::Device => "device",
        }
    }
}

/// Which checkpoint precision a config is read as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Precision {
    /// `nvidia/Qwen3.8-Flash-Next-NVFP4`: the ModelOpt mixed-precision map.
    ModelOptMixed,
    /// `Qwen/Qwen3.8-Flash-Next`: BF16 throughout.
    Bf16,
}

/// The checkpoint's text-model dimensions, read strictly from its config.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen4ExpModelCfg {
    pub hidden: u32,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
    pub mixers: Vec<Qwen4ExpMixer>,
    // QSA.
    pub qsa_heads: u32,
    pub qsa_kv_heads: u32,
    pub qsa_head_dim: u32,
    pub rotary_dim: u32,
    pub indexer_heads: u32,
    pub indexer_kv_heads: u32,
    pub indexer_head_dim: u32,
    pub indexer_compress_ratio: u32,
    /// Tokens each row selects (`indexer_budget`).
    pub indexer_token_topk: u32,
    // GDN.
    pub gdn_key_heads: u32,
    pub gdn_value_heads: u32,
    pub gdn_key_head_dim: u32,
    pub gdn_value_head_dim: u32,
    pub gdn_conv_kernel: u32,
    pub gdn_state_dtype: DType,
    // Hyper-connections.
    pub hc_count: u32,
    pub hc_lowrank: u32,
    // PLE.
    /// 0-based decoder layers that carry a PLE block.
    pub ple_layers: Vec<u32>,
    pub ngram_size: u32,
    pub heads_per_ngram: u32,
    pub ple_embed_dim: u32,
    pub ple_conv_kernel: u32,
    /// Rows of the PLE layer's (padded) n-gram table.
    pub ple_table_rows: u64,
    pub ple_table_dtype: DType,
    // MoE.
    pub num_experts: u32,
    pub experts_per_token: u32,
    pub moe_intermediate: u32,
    pub shared_expert_intermediate: u32,
    // MTP.
    pub mtp_layers: u32,
    /// The MTP experts' FP8 weight/activation block width.
    pub mtp_expert_block: u32,
}

impl Qwen4ExpModelCfg {
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading Qwen4Exp config {}", path.display()))?;
        let root: Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing Qwen4Exp config {}", path.display()))?;
        Self::from_value(&root)
    }

    pub fn from_value(root: &Value) -> Result<Self> {
        Self::parse(root, Precision::ModelOptMixed)
    }

    /// The BF16 checkpoint (`Qwen/Qwen3.8-Flash-Next`): no
    /// `quantization_config`, every weight BF16 (the PLE table and the MTP
    /// experts included, so `mtp_expert_block` is 0). Its config omits
    /// `norm_topk_prob`; absent is read as renormalized top-k, as the
    /// reference implementation routes.
    pub fn from_bf16_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading Qwen4Exp config {}", path.display()))?;
        let root: Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing Qwen4Exp config {}", path.display()))?;
        Self::from_bf16_value(&root)
    }

    pub fn from_bf16_value(root: &Value) -> Result<Self> {
        Self::parse(root, Precision::Bf16)
    }

    fn parse(root: &Value, precision: Precision) -> Result<Self> {
        let cfg = root
            .get("text_config")
            .context("Qwen4Exp config has no text_config")?;
        let u = |key: &str| -> Result<u32> {
            let value = cfg
                .get(key)
                .and_then(Value::as_u64)
                .with_context(|| format!("Qwen4Exp text_config.{key} must be an integer"))?;
            u32::try_from(value).with_context(|| format!("text_config.{key} exceeds u32"))
        };
        let text = |key: &str| cfg.get(key).and_then(Value::as_str).unwrap_or("");
        let flag = |key: &str| cfg.get(key).and_then(Value::as_bool);
        ensure!(
            text("model_type") == "qwen4_exp_text",
            "Qwen4Exp arches require a qwen4_exp_text config, got model_type {:?}",
            text("model_type")
        );
        ensure!(
            text("output_gate_type") == "sigmoid",
            "Qwen4Exp arches model the sigmoid QSA output gate"
        );
        ensure!(
            text("hidden_act") == "silu",
            "Qwen4Exp arches model SiLU experts"
        );
        let renormalized = match precision {
            Precision::ModelOptMixed => flag("norm_topk_prob") == Some(true),
            Precision::Bf16 => flag("norm_topk_prob").unwrap_or(true),
        };
        ensure!(
            renormalized,
            "Qwen4Exp arches model renormalized top-k routing"
        );
        ensure!(
            flag("tie_word_embeddings") == Some(false)
                && flag("mtp_use_dedicated_embeddings") == Some(false),
            "Qwen4Exp arches model untied heads with the MTP layer reusing the target's \
             embedding and lm_head"
        );
        let num_layers = u("num_hidden_layers")?;
        let mixers = cfg
            .get("layer_types")
            .and_then(Value::as_array)
            .context("Qwen4Exp text_config.layer_types must be an array")?
            .iter()
            .map(|entry| match entry.as_str() {
                Some("linear_attention") => Ok(Qwen4ExpMixer::Gdn),
                Some("full_attention") => Ok(Qwen4ExpMixer::Qsa),
                other => bail!("text_config.layer_types entry {other:?} is not modeled"),
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            mixers.len() == num_layers as usize,
            "text_config.layer_types has {} entries for {num_layers} layers",
            mixers.len()
        );
        let mtp_layers = u("mtp_num_hidden_layers")?;
        let mtp_types = cfg
            .get("mtp")
            .and_then(|m| m.get("layer_types"))
            .and_then(Value::as_array)
            .context("Qwen4Exp text_config.mtp.layer_types must be an array")?;
        ensure!(
            mtp_types.len() == mtp_layers as usize
                && mtp_types
                    .iter()
                    .all(|t| t.as_str() == Some("full_attention")),
            "Qwen4Exp arches model QSA MTP layers"
        );

        let head_dim = u("head_dim")?;
        let partial = cfg
            .get("rope_parameters")
            .and_then(|r| r.get("partial_rotary_factor"))
            .or_else(|| cfg.get("partial_rotary_factor"))
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        let rotary = f64::from(head_dim) * partial;
        ensure!(
            rotary.fract() == 0.0,
            "head_dim {head_dim} x partial_rotary_factor {partial} is not whole"
        );
        let ratio = u("indexer_compress_ratio")?;
        let budget = u("indexer_budget")?;
        ensure!(
            ratio > 0 && budget % ratio == 0 && [512, 2048].contains(&(budget / ratio)),
            "indexer_budget {budget} / indexer_compress_ratio {ratio} must be 512 or 2048 groups"
        );
        let gdn_state_dtype = match text("mamba_ssm_dtype") {
            "float32" => DType::Fp32,
            "bfloat16" => DType::Bf16,
            other => bail!("text_config.mamba_ssm_dtype {other:?} is not modeled"),
        };

        let ple_layers = cfg
            .get("ple_layer_ids")
            .and_then(Value::as_array)
            .context("Qwen4Exp text_config.ple_layer_ids must be an array")?
            .iter()
            .map(|id| match id.as_u64() {
                Some(id) if (1..=u64::from(num_layers)).contains(&id) => Ok(id as u32 - 1),
                _ => bail!("ple_layer_ids entry {id} is not a 1-based decoder layer"),
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            ple_layers.len() <= 1,
            "Qwen4Exp arches price one PLE table; ple_layer_ids lists {} layers",
            ple_layers.len()
        );
        let ngram_size = u("ngram_size")?;
        ensure!(
            ngram_size >= 2,
            "ngram_size {ngram_size} must be at least 2"
        );
        let heads_per_ngram = u("heads_per_ngram")?;
        let ple_heads = (ngram_size - 1) * heads_per_ngram;
        let ple_embed_dim = u("ple_embed_dim")?;
        ensure!(
            ple_heads > 0 && ple_embed_dim % ple_heads == 0,
            "ple_embed_dim {ple_embed_dim} must split over {ple_heads} n-gram heads"
        );
        let ple_table_rows = ngram_table_rows(
            u64::from(u("ngram_vocab_size_base")?),
            ple_heads,
            u64::from(u("make_ngram_vocab_size_divisible_by")?),
        );

        let (mtp_expert_block, ple_table_dtype) = match precision {
            Precision::ModelOptMixed => {
                let quant = root
                    .get("quantization_config")
                    .context("Qwen4Exp arches model the ModelOpt mixed-precision checkpoint")?;
                check_quantization(quant, num_layers, &ple_layers)?
            }
            Precision::Bf16 => {
                ensure!(
                    root.get("quantization_config").is_none(),
                    "the BF16 Qwen4Exp reader got a quantized checkpoint config"
                );
                ensure!(
                    text("dtype") == "bfloat16",
                    "the BF16 Qwen4Exp reader needs text_config.dtype bfloat16, got {:?}",
                    text("dtype")
                );
                (0, DType::Bf16)
            }
        };

        Ok(Self {
            hidden: u("hidden_size")?,
            vocab_size: u("vocab_size")?,
            max_position_embeddings: u("max_position_embeddings")?,
            qsa_heads: u("num_attention_heads")?,
            qsa_kv_heads: u("num_key_value_heads")?,
            qsa_head_dim: head_dim,
            rotary_dim: rotary as u32,
            indexer_heads: u("indexer_n_heads")?,
            indexer_kv_heads: u("indexer_kv_heads")?,
            indexer_head_dim: u("indexer_head_dim")?,
            indexer_compress_ratio: ratio,
            indexer_token_topk: budget,
            gdn_key_heads: u("linear_num_key_heads")?,
            gdn_value_heads: u("linear_num_value_heads")?,
            gdn_key_head_dim: u("linear_key_head_dim")?,
            gdn_value_head_dim: u("linear_value_head_dim")?,
            gdn_conv_kernel: u("linear_conv_kernel_dim")?,
            gdn_state_dtype,
            hc_count: u("hc_count")?,
            hc_lowrank: u("hc_lowrank")?,
            ple_layers,
            ngram_size,
            heads_per_ngram,
            ple_embed_dim,
            ple_conv_kernel: u("ple_conv_kernel_size")?,
            ple_table_rows,
            ple_table_dtype,
            num_experts: u("num_experts")?,
            experts_per_token: u("num_experts_per_tok")?,
            moe_intermediate: u("moe_intermediate_size")?,
            shared_expert_intermediate: u("shared_expert_intermediate_size")?,
            mtp_layers,
            mtp_expert_block,
            mixers,
        })
    }

    pub fn num_layers(&self) -> u32 {
        self.mixers.len() as u32
    }

    pub fn layer_count(&self, mixer: Qwen4ExpMixer) -> u32 {
        self.mixers.iter().filter(|&&m| m == mixer).count() as u32
    }

    /// The q/k/v channels the GDN causal conv runs over (and caches).
    pub fn gdn_conv_channels(&self) -> u32 {
        2 * self.gdn_key_heads * self.gdn_key_head_dim
            + self.gdn_value_heads * self.gdn_value_head_dim
    }

    /// The PLE short conv's dilation and the state taps it keeps
    /// (`(kernel - 1) * dilation`, `ple_layer.py:92-93`).
    pub fn ple_conv_dilation(&self) -> u32 {
        self.ngram_size
    }

    pub fn ple_conv_state_len(&self) -> u32 {
        (self.ple_conv_kernel - 1) * self.ple_conv_dilation()
    }
}

/// Rows of the first (dense layer 0) PLE layer's n-gram table: the sum of its
/// heads' prime vocabularies, padded to `divisor` (`_make_vocab_layout`).
fn ngram_table_rows(base: u64, heads: u32, divisor: u64) -> u64 {
    let is_prime = |n: u64| {
        n >= 2
            && (2..)
                .take_while(|d| d * d <= n)
                .all(|d| !n.is_multiple_of(d))
    };
    let mut prime = base.saturating_sub(1);
    let mut total = 0;
    for _ in 0..heads {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
        total += prime;
    }
    total.div_ceil(divisor.max(1)) * divisor.max(1)
}

/// Check the ModelOpt mix this arch models and return the MTP experts' block
/// width and the PLE table dtype: FP8 when every PLE table is in the map,
/// BF16 (unquantized) when none is.
fn check_quantization(quant: &Value, num_layers: u32, ple_layers: &[u32]) -> Result<(u32, DType)> {
    ensure!(
        quant.get("quant_method").and_then(Value::as_str) == Some("modelopt"),
        "Qwen4Exp config: unmodeled quant_method"
    );
    let layers = quant
        .get("quantized_layers")
        .and_then(Value::as_object)
        .context("Qwen4Exp config: modelopt quantization_config needs quantized_layers")?;
    fn algo(entry: &Value) -> (&str, Option<u64>) {
        (
            entry
                .get("quant_algo")
                .and_then(Value::as_str)
                .unwrap_or(""),
            entry.get("group_size").and_then(Value::as_u64),
        )
    }
    let mut mtp_block = None;
    let mut fp8_ple_tables = 0;
    for (module, entry) in layers {
        let body_expert = (0..num_layers)
            .any(|l| module == &format!("model.language_model.layers.{l}.mlp.experts"));
        let ple_table = ple_layers.iter().any(|l| {
            module == &format!("model.language_model.layers.{l}.ple.ple_embedding.ngram_embedding")
        });
        match algo(entry) {
            ("NVFP4", Some(16)) if body_expert => {}
            ("FP8", None) if ple_table => fp8_ple_tables += 1,
            ("FP8_PB_WO", Some(block))
                if module.starts_with("mtp.layers.") && module.ends_with(".mlp.experts") =>
            {
                mtp_block = Some(block as u32)
            }
            other => bail!("Qwen4Exp config: {module} {other:?} is not the modeled precision mix"),
        }
    }
    let body_experts = layers
        .keys()
        .filter(|m| m.starts_with("model.language_model.layers.") && m.ends_with(".mlp.experts"))
        .count();
    ensure!(
        body_experts == num_layers as usize,
        "Qwen4Exp config: {body_experts} of {num_layers} layers have NVFP4 experts"
    );
    let block = mtp_block.context("Qwen4Exp config: MTP experts must be FP8 block-scaled")?;
    // `FP8_PB_WO` names the weights; the experts run W8A8 only when the MTP
    // group also quantizes its activations dynamically per block.
    let w8a8 = quant
        .get("config_groups")
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|groups| groups.values())
        .any(|group| {
            let mtp = group
                .get("targets")
                .and_then(Value::as_array)
                .is_some_and(|t| {
                    t.iter()
                        .any(|t| t.as_str().is_some_and(|t| t.starts_with("mtp.")))
                });
            let act = group.get("input_activations");
            mtp && act.and_then(|a| a.get("dynamic")).and_then(Value::as_bool) == Some(true)
                && act.and_then(|a| a.get("num_bits")).and_then(Value::as_u64) == Some(8)
                && act
                    .and_then(|a| a.get("group_size"))
                    .and_then(Value::as_u64)
                    == Some(u64::from(block))
        });
    ensure!(
        w8a8,
        "Qwen4Exp config: MTP experts must quantize activations per {block}-block"
    );
    let ple_table_dtype = match fp8_ple_tables {
        0 => DType::Bf16,
        n if n == ple_layers.len() => DType::Fp8E4m3,
        n => bail!(
            "Qwen4Exp config: {n} of {} PLE tables are FP8; a mixed set is not modeled",
            ple_layers.len()
        ),
    };
    Ok((block, ple_table_dtype))
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn flash_next() -> Qwen4ExpModelCfg {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("model/config/qwen3_8_flash_next_nvfp4.json");
        Qwen4ExpModelCfg::from_json(&path).unwrap()
    }

    /// The BF16 checkpoint reads the same shape as the NVFP4 one, with every
    /// weight BF16; a quantized config is refused by the BF16 reader.
    #[test]
    fn bf16_checkpoint_reads_the_same_shape_in_bf16() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/qwen3_8_flash_next_bf16.json");
        let bf16 = Qwen4ExpModelCfg::from_bf16_json(&path).unwrap();
        let nvfp4 = flash_next();
        assert_eq!(bf16.ple_table_dtype, DType::Bf16);
        assert_eq!(bf16.mtp_expert_block, 0);
        assert_eq!(
            Qwen4ExpModelCfg {
                ple_table_dtype: nvfp4.ple_table_dtype,
                mtp_expert_block: nvfp4.mtp_expert_block,
                ..bf16.clone()
            },
            nvfp4
        );
        assert!(Qwen4ExpModelCfg::from_bf16_value(&flash_next_json()).is_err());
    }

    #[test]
    fn flash_next_reads_its_shape() {
        let cfg = flash_next();
        assert_eq!((cfg.hidden, cfg.vocab_size), (2560, 248320));
        assert_eq!(cfg.num_layers(), 48);
        assert_eq!(cfg.layer_count(Qwen4ExpMixer::Gdn), 36);
        let qsa: Vec<usize> = (0..48)
            .filter(|&l| cfg.mixers[l] == Qwen4ExpMixer::Qsa)
            .collect();
        assert_eq!(qsa, (3..48).step_by(4).collect::<Vec<_>>());
        assert_eq!(
            (cfg.qsa_heads, cfg.qsa_kv_heads, cfg.qsa_head_dim),
            (24, 2, 256)
        );
        assert_eq!(cfg.rotary_dim, 64);
        assert_eq!(
            (
                cfg.indexer_heads,
                cfg.indexer_kv_heads,
                cfg.indexer_head_dim
            ),
            (4, 1, 128)
        );
        assert_eq!(
            (cfg.indexer_compress_ratio, cfg.indexer_token_topk),
            (4, 2048)
        );
        assert_eq!(cfg.gdn_conv_channels(), 10240);
        assert_eq!(cfg.gdn_state_dtype, DType::Fp32);
        assert_eq!((cfg.hc_count, cfg.hc_lowrank), (4, 320));
        assert_eq!(cfg.ple_layers, [1]);
        assert_eq!((cfg.ple_conv_dilation(), cfg.ple_conv_state_len()), (3, 9));
        // 16 primes after 19,999,999, summed and padded to 128.
        assert_eq!(cfg.ple_table_rows, 320_001_536);
        assert_eq!(cfg.ple_table_dtype, DType::Fp8E4m3);
        assert_eq!((cfg.num_experts, cfg.experts_per_token), (512, 10));
        assert_eq!(
            (cfg.moe_intermediate, cfg.shared_expert_intermediate),
            (640, 640)
        );
        assert_eq!((cfg.mtp_layers, cfg.mtp_expert_block), (1, 128));
    }

    fn flash_next_json() -> Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("model/config/qwen3_8_flash_next_nvfp4.json");
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn another_precision_mix_is_refused() {
        let mut root = flash_next_json();
        root["quantization_config"]["quantized_layers"]
            ["model.language_model.layers.3.mlp.experts"]["quant_algo"] = "FP8".into();
        assert!(Qwen4ExpModelCfg::from_value(&root).is_err());
        root["quantization_config"]["quantized_layers"]
            .as_object_mut()
            .unwrap()
            .remove("model.language_model.layers.3.mlp.experts");
        assert!(Qwen4ExpModelCfg::from_value(&root).is_err());
    }

    /// The PLE table dtype follows the quantization map: FP8 when the table
    /// is listed, BF16 when it is not.
    #[test]
    fn an_unlisted_ple_table_is_bf16() {
        let mut root = flash_next_json();
        root["quantization_config"]["quantized_layers"]
            .as_object_mut()
            .unwrap()
            .remove("model.language_model.layers.1.ple.ple_embedding.ngram_embedding")
            .unwrap();
        let cfg = Qwen4ExpModelCfg::from_value(&root).unwrap();
        assert_eq!(cfg.ple_table_dtype, DType::Bf16);
    }

    #[test]
    fn a_second_ple_layer_is_refused() {
        let mut root = flash_next_json();
        root["text_config"]["ple_layer_ids"] = serde_json::json!([2, 5]);
        assert!(Qwen4ExpModelCfg::from_value(&root).is_err());
    }
}
