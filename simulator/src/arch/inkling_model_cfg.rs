//! The Inkling text model (`InklingForConditionalGeneration`, model_type
//! `inkling_mm_model`) as its checkpoint describes it, shared by every Inkling
//! arch and by both sizes (`thinkingmachines/Inkling-NVFP4`, 66 layers, and
//! `Inkling-Small-NVFP4`, 42 layers).
//!
//! The dimensions come from `config.json`'s `text_config`; the vision tower,
//! the audio stack and the MTP head are not modeled. ModelOpt ships the NVFP4
//! scheme in a separate `hf_quant_config.json`; the model config file keeps it
//! verbatim under `hf_quant_config` (or as vLLM's merged
//! `quantization_config`). Its `exclude_modules` list fixes each GEMM's
//! precision: attention, the router (`mlp.gate`) and the sink experts
//! (`mlp.shared_experts`) stay BF16 in every layer, the dense FFN of layers
//! `0..dense_mlp_idx` is BF16, and the routed experts are NVFP4 except where a
//! layer's `mlp.experts` is excluded (layer 2 in both released checkpoints).

use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

/// One routed-MoE layer's expert weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InklingExpertPrecision {
    /// `mlp.experts` excluded from quantization.
    Bf16,
    /// ModelOpt NVFP4 W4A4, 16-element groups.
    Nvfp4,
}

/// The checkpoint's text-model dimensions, read strictly from its config.
#[derive(Clone, Debug, PartialEq)]
pub struct InklingModelCfg {
    pub hidden: u32,
    pub num_layers: u32,
    /// Global (full-causal) attention heads.
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    /// Sliding-window attention heads.
    pub swa_num_attention_heads: u32,
    pub swa_num_key_value_heads: u32,
    pub swa_head_dim: u32,
    /// Width of one head's relative-position branch.
    pub d_rel: u32,
    /// Relative distances a global layer covers.
    pub rel_extent: u32,
    pub sliding_window_size: u32,
    /// Layers with sliding-window attention; the rest are global.
    pub local_layer_ids: Vec<u32>,
    pub sconv_kernel_size: u32,
    pub n_routed_experts: u32,
    pub num_experts_per_tok: u32,
    /// Sink experts every token activates (`shared_expert_sink`).
    pub n_shared_experts: u32,
    /// Per-expert intermediate width (routed and sink experts).
    pub intermediate_size: u32,
    pub dense_intermediate_size: u32,
    /// Layers `0..dense_mlp_idx` carry a dense FFN, the rest an MoE.
    pub dense_mlp_idx: u32,
    pub vocab_size: u32,
    /// `route_scale` as an exact fraction.
    pub route_scale: (u32, u32),
    pub use_gate_bias: bool,
    pub use_global_scale: bool,
    /// One entry per MoE layer, `dense_mlp_idx..num_layers` in order.
    pub expert_precision: Vec<InklingExpertPrecision>,
}

impl InklingModelCfg {
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading Inkling config {}", path.display()))?;
        let root: Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing Inkling config {}", path.display()))?;
        Self::from_value(&root)
    }

    pub fn from_value(root: &Value) -> Result<Self> {
        let cfg = root
            .get("text_config")
            .context("Inkling config needs a text_config")?;
        let u = |key: &str| -> Result<u32> {
            let value = cfg
                .get(key)
                .and_then(Value::as_u64)
                .with_context(|| format!("Inkling text_config.{key} must be an integer"))?;
            u32::try_from(value).with_context(|| format!("text_config.{key} exceeds u32"))
        };
        let flag = |key: &str| -> Result<bool> {
            cfg.get(key)
                .and_then(Value::as_bool)
                .with_context(|| format!("Inkling text_config.{key} must be a boolean"))
        };
        let model_type = root.get("model_type").and_then(Value::as_str).unwrap_or("");
        ensure!(
            model_type == "inkling_mm_model",
            "Inkling arches require an inkling_mm_model config, got model_type {model_type:?}"
        );
        let local_layer_ids: Vec<u32> = cfg
            .get("local_layer_ids")
            .and_then(Value::as_array)
            .context("text_config.local_layer_ids must be an array")?
            .iter()
            .map(|layer| {
                layer
                    .as_u64()
                    .and_then(|layer| u32::try_from(layer).ok())
                    .context("local_layer_ids entries must be u32")
            })
            .collect::<Result<_>>()?;
        let route_scale = cfg
            .get("route_scale")
            .and_then(Value::as_f64)
            .context("Inkling text_config.route_scale must be a number")?;
        ensure!(
            route_scale > 0.0 && route_scale.fract() == 0.0,
            "route_scale {route_scale} must be a positive integer"
        );
        ensure!(
            flag("use_sconv")? && flag("shared_expert_sink")? && flag("norm_after_topk")?,
            "Inkling arches model the sconv residual, sink shared experts and \
             post-top-k renormalization"
        );
        ensure!(
            cfg.get("gate_activation").and_then(Value::as_str) == Some("sigmoid"),
            "Inkling arches model the sigmoid gate"
        );
        let num_layers = u("num_hidden_layers")?;
        let dense_mlp_idx = u("dense_mlp_idx")?;
        ensure!(
            dense_mlp_idx <= num_layers,
            "dense_mlp_idx exceeds num_hidden_layers"
        );
        ensure!(
            local_layer_ids.iter().all(|&layer| layer < num_layers),
            "local_layer_ids must index model layers"
        );
        let expert_precision = expert_precision(root, num_layers, dense_mlp_idx)?;
        Ok(Self {
            hidden: u("hidden_size")?,
            num_layers,
            num_attention_heads: u("num_attention_heads")?,
            num_key_value_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            swa_num_attention_heads: u("swa_num_attention_heads")?,
            swa_num_key_value_heads: u("swa_num_key_value_heads")?,
            swa_head_dim: u("swa_head_dim")?,
            d_rel: u("d_rel")?,
            rel_extent: u("rel_extent")?,
            sliding_window_size: u("sliding_window_size")?,
            local_layer_ids,
            sconv_kernel_size: u("sconv_kernel_size")?,
            n_routed_experts: u("n_routed_experts")?,
            num_experts_per_tok: u("num_experts_per_tok")?,
            n_shared_experts: u("n_shared_experts")?,
            intermediate_size: u("intermediate_size")?,
            dense_intermediate_size: u("dense_intermediate_size")?,
            dense_mlp_idx,
            vocab_size: u("vocab_size")?,
            route_scale: (route_scale as u32, 1),
            use_gate_bias: flag("use_gate_bias")?,
            use_global_scale: flag("use_global_scale")?,
            expert_precision,
        })
    }

    pub fn is_local(&self, layer: u32) -> bool {
        self.local_layer_ids.contains(&layer)
    }

    pub fn num_moe_layers(&self) -> u32 {
        self.num_layers - self.dense_mlp_idx
    }

    /// The MoE layer's expert precision; `None` on a dense layer.
    pub fn expert_precision(&self, layer: u32) -> Option<InklingExpertPrecision> {
        layer
            .checked_sub(self.dense_mlp_idx)
            .and_then(|moe| self.expert_precision.get(moe as usize).copied())
    }

    /// The MoE-layer index range (a token corpus's layer axis) holding every
    /// layer of `precision`. Refused unless those layers are contiguous, since
    /// one routed fold stands for all of them.
    pub fn moe_layer_range(
        &self,
        precision: InklingExpertPrecision,
    ) -> Result<Option<std::ops::Range<usize>>> {
        let indices: Vec<usize> = self
            .expert_precision
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p == precision)
            .map(|(index, _)| index)
            .collect();
        let (Some(&first), Some(&last)) = (indices.first(), indices.last()) else {
            return Ok(None);
        };
        ensure!(
            last - first + 1 == indices.len(),
            "Inkling {precision:?} MoE layers must be contiguous to share one routed fold"
        );
        Ok(Some(first..last + 1))
    }

    /// `round_up(routed + sink, 8)`: the router GEMM's padded output width
    /// (258 -> 264), which is also the gate select's logits row stride.
    pub fn router_logits_width(&self) -> u32 {
        (self.n_routed_experts + self.n_shared_experts).next_multiple_of(8)
    }
}

/// Per MoE layer, the routed experts' precision from the ModelOpt
/// `exclude_modules`. Only the modeled scheme is accepted: NVFP4 with
/// 16-element groups (a config that names no group size is taken as 16, the
/// ModelOpt NVFP4 default), a BF16 KV cache, and exclusions that keep exactly
/// attention, the router, the sink experts and the dense FFN in BF16.
fn expert_precision(
    root: &Value,
    num_layers: u32,
    dense_mlp_idx: u32,
) -> Result<Vec<InklingExpertPrecision>> {
    let quant = root
        .pointer("/hf_quant_config/quantization")
        .or_else(|| root.get("quantization_config"))
        .context(
            "Inkling arches are modeled for the ModelOpt NVFP4 checkpoint; the config \
             has neither hf_quant_config.quantization nor quantization_config",
        )?;
    ensure!(
        quant.get("quant_algo").and_then(Value::as_str) == Some("NVFP4"),
        "Inkling is modeled for quant_algo NVFP4"
    );
    let group = quant
        .get("group_size")
        .or_else(|| quant.pointer("/config_groups/group_0/weights/group_size"));
    ensure!(
        group.is_none() || group == Some(&serde_json::json!(16)),
        "Inkling NVFP4 is modeled with 16-element groups, got {group:?}"
    );
    let kv = quant.get("kv_cache_quant_algo");
    ensure!(
        kv.is_none_or(|kv| kv.is_null() || kv.as_str() == Some("none")),
        "Inkling is modeled with a BF16 KV cache, got kv_cache_quant_algo {kv:?}"
    );
    let entries = quant
        .get("exclude_modules")
        .or_else(|| quant.get("ignore"))
        .and_then(Value::as_array)
        .context("ModelOpt NVFP4 config needs an exclude_modules list")?;
    // Per layer: attn, mlp.gate, mlp.shared_experts, mlp.w13_dn, mlp.w2_md,
    // mlp.experts excluded.
    let mut seen = vec![[false; 6]; num_layers as usize];
    for entry in entries {
        let entry = entry
            .as_str()
            .context("ModelOpt exclude_modules entries must be strings")?;
        let Some(rest) = entry.strip_prefix("model.llm.layers.") else {
            ensure!(
                entry.starts_with("model.audio.")
                    || entry.starts_with("model.visual.")
                    || entry.starts_with("model.mtp")
                    || matches!(
                        entry,
                        "model.llm.embed" | "model.llm.embed_norm" | "model.llm.norm"
                            | "model.llm.unembed"
                    ),
                "Inkling NVFP4 does not model exclude_modules entry {entry:?}"
            );
            continue;
        };
        let (layer, module) = rest
            .split_once('.')
            .with_context(|| format!("exclude_modules entry {entry:?} names no module"))?;
        let layer: u32 = layer
            .parse()
            .with_context(|| format!("exclude_modules entry {entry:?} has no layer index"))?;
        ensure!(layer < num_layers, "exclude_modules entry {entry:?} is past the body");
        let dense = layer < dense_mlp_idx;
        let slot = match module {
            // Norms and the sconv taps carry no GEMM.
            "attn_norm" | "attn_sconv" | "mlp_norm" | "mlp_sconv" => continue,
            "mlp.global_scale" if dense => continue,
            "attn" => 0,
            "mlp.gate" if !dense => 1,
            "mlp.shared_experts" if !dense => 2,
            "mlp.w13_dn" if dense => 3,
            "mlp.w2_md" if dense => 4,
            "mlp.experts" if !dense => 5,
            _ => bail!("Inkling NVFP4 does not model exclude_modules entry {entry:?}"),
        };
        seen[layer as usize][slot] = true;
    }
    let mut precision = Vec::new();
    for (layer, [attn, gate, sinks, w13, w2, experts]) in seen.into_iter().enumerate() {
        ensure!(attn, "Inkling NVFP4 is modeled with layer {layer} attn left BF16");
        if (layer as u32) < dense_mlp_idx {
            ensure!(
                w13 && w2,
                "Inkling NVFP4 is modeled with layer {layer}'s dense FFN left BF16"
            );
        } else {
            ensure!(
                gate && sinks,
                "Inkling NVFP4 is modeled with layer {layer}'s router and sink experts left BF16"
            );
            precision.push(if experts {
                InklingExpertPrecision::Bf16
            } else {
                InklingExpertPrecision::Nvfp4
            });
        }
    }
    Ok(precision)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(stem: &str) -> InklingModelCfg {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("model/config")
            .join(format!("{stem}.json"));
        InklingModelCfg::from_json(&path).unwrap()
    }

    #[test]
    fn small_checkpoint_reads_its_layer_schedule_and_precisions() {
        let model = load("inkling_small_nvfp4");
        assert_eq!((model.hidden, model.num_layers), (4096, 42));
        let globals: Vec<u32> = (0..42).filter(|&l| !model.is_local(l)).collect();
        assert_eq!(globals, [5, 11, 17, 23, 29, 35, 41]);
        assert_eq!(model.expert_precision(1), None);
        assert_eq!(
            model.expert_precision(2),
            Some(InklingExpertPrecision::Bf16)
        );
        assert_eq!(
            model.expert_precision(41),
            Some(InklingExpertPrecision::Nvfp4)
        );
        assert_eq!(
            model.moe_layer_range(InklingExpertPrecision::Bf16).unwrap(),
            Some(0..1)
        );
        assert_eq!(
            model.moe_layer_range(InklingExpertPrecision::Nvfp4).unwrap(),
            Some(1..40)
        );
        assert_eq!(model.router_logits_width(), 264);
        assert_eq!(model.route_scale, (8, 1));
    }

    #[test]
    fn large_checkpoint_parses_with_wider_sliding_window_kv() {
        let model = load("inkling_nvfp4");
        assert_eq!((model.hidden, model.num_layers), (6144, 66));
        assert_eq!(
            (model.num_key_value_heads, model.swa_num_key_value_heads),
            (8, 16)
        );
        assert_eq!(model.num_moe_layers(), 64);
    }

    #[test]
    fn an_unmodeled_exclusion_is_refused() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("model/config/inkling_small_nvfp4.json");
        let mut root: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        root.pointer_mut("/hf_quant_config/quantization/exclude_modules")
            .and_then(Value::as_array_mut)
            .unwrap()
            .push("model.llm.layers.7.mlp.w13_dn".into());
        let error = InklingModelCfg::from_value(&root).unwrap_err();
        assert!(error.to_string().contains("layers.7.mlp.w13_dn"), "{error}");
    }
}
