//! Arch (L4) config surface — the model-arch *selectors*, co-located with the L4
//! implementations they pick (new-interface-design §2 / §4).
//!
//! Each selector is a serde tagged enum (`#[serde(tag = "type")]`): choosing the
//! tag is the only way that variant's params appear — no global union,
//! provider-first. `model_config` + dims belong to the arch (the layer that
//! consumes them), so every arch variant flattens [`ModelSpec`]. arch and worker
//! are symmetric sibling providers (not nested).
//!
//! Iter-wise Llama selectors and three Qwen3-MoE execution graphs build today:
//! native BF16, native FP8, and vLLM-aligned FP8. Each Qwen graph has a distinct
//! selector and L4 implementation; precision/backend selection is therefore not
//! hidden behind an in-arch recipe branch. They pair with the same `hp_unified`
//! worker as the DP-attn dense variant (one KV partition state per attn-DP shard).
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

/// MoE expert routing distribution kind (the `routing` selector on
/// [`IterArchSel::Qwen3MoeDpAttnEpFfn`]). v1 exposes `uniform` and a seeded
/// `random`; the model layer also carries `power_law` / explicit `from_profile`
/// (see `timing::routing::RoutingDistribution`), while a measured profile is
/// supplied separately through `expert_popularity_file`. Kept a small closed
/// set so the launcher validates it as a `string` param with `choices`
/// (mirrored by [`ROUTING_KINDS`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingKind {
    /// Even load across all experts (the v1 default).
    #[default]
    Uniform,
    /// Deterministic pseudo-random skew seeded by `routing_seed`.
    Random,
}

/// The `routing` choices the launcher schema advertises (mirror of [`RoutingKind`]).
const ROUTING_KINDS: [&str; 2] = ["uniform", "random"];

/// Iteration-wise arch provider. Sharding parameters live only on the variants
/// that consume them (provider-first: select the arch, then it exposes its own
/// params).
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
    Qwen3MoeDpAttnEpFfn {
        #[serde(flatten)]
        model: ModelSpec,
        /// Attention tensor-parallelism size (heads sharded across these ranks).
        /// `num_dp_groups = ep_size / attn_tp_size`.
        #[param(default = 4, cache_key)]
        attn_tp_size: u16,
        /// Expert-parallelism size (FFN expert ranks; spans the whole replica).
        #[param(default = 8, cache_key)]
        ep_size: u16,
        /// `ReplicatedHeadParallel` residing-group size on the FFN side
        /// (drives whether the L2 MoE combine bcast/fanout legs are non-zero).
        #[param(default = 1, cache_key)]
        hp_size: u16,
        /// NVLink-domain size — partitions `ep_size` ranks into NVL domains
        /// for the intra/inter split of MoE dispatch/combine.
        #[param(default = 8, cache_key)]
        nvl_num_gpu: u16,
        /// Expert routing distribution: `uniform` (default) spreads load evenly;
        /// `random` draws a deterministic pseudo-random skew seeded by
        /// `routing_seed`. Drives the L2 MoE dispatch/combine `BottleneckCurve`
        /// and the L3 grouped-GEMM `local_ppm` shards. Omitted → `uniform`.
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS)]
        routing: RoutingKind,
        /// Seed for `routing = random` (ignored for `uniform`). Fixed so a run is
        /// reproducible (the throughput golden is bit-identical); vary it to
        /// sample a different random skew.
        #[serde(default)]
        routing_seed: Option<u64>,
        /// Optional measured expert-popularity JSON. Its all-layer expert
        /// totals replace the synthetic routing distribution and are split
        /// into one raw `local_ppm` shard per EP rank. The homogeneous iter-wise
        /// arch intentionally uses one aggregate snapshot for every layer.
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
    },
    /// Native FP8 Qwen recipe: per-token/group quantized dense projections,
    /// DeepGEMM expert kernels, and VibeSim's P2P EP dispatch/combine graph.
    Qwen3MoeFp8DpAttnEpFfn {
        #[serde(flatten)]
        model: ModelSpec,
        #[param(default = 4, cache_key)]
        attn_tp_size: u16,
        #[param(default = 8, cache_key)]
        ep_size: u16,
        #[param(default = 1, cache_key)]
        hp_size: u16,
        #[param(default = 8, cache_key)]
        nvl_num_gpu: u16,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS)]
        routing: RoutingKind,
        #[serde(default)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
    },
    /// Alignment-only Qwen recipe matching the target vLLM execution path:
    /// FlashInfer/TensorRT-LLM block-scale expert GEMMs plus local finalize and
    /// an EP all-reduce. The generic Qwen tag above intentionally retains
    /// VibeSim's original DeepGEMM + P2P communication recipe.
    Qwen3VllmMoeDpAttnEpFfn {
        #[serde(flatten)]
        model: ModelSpec,
        #[param(default = 4, cache_key)]
        attn_tp_size: u16,
        #[param(default = 8, cache_key)]
        ep_size: u16,
        #[param(default = 1, cache_key)]
        hp_size: u16,
        #[param(default = 8, cache_key)]
        nvl_num_gpu: u16,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS)]
        routing: RoutingKind,
        #[serde(default)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
    },
}

impl IterArchSel {
    /// The model identity/dims this arch operates on (every variant carries it).
    pub fn model(&self) -> &ModelSpec {
        match self {
            Self::Llama3Dense { model }
            | Self::Llama3DenseTp { model, .. }
            | Self::Llama3DpAttnTpFfn { model, .. }
            | Self::Qwen3MoeDpAttnEpFfn { model, .. }
            | Self::Qwen3MoeFp8DpAttnEpFfn { model, .. }
            | Self::Qwen3VllmMoeDpAttnEpFfn { model, .. } => model,
        }
    }
}

#[cfg(test)]
mod iter_tests {
    use super::*;

    fn parse_qwen(extra: &str) -> Result<IterArchSel, serde_json::Error> {
        serde_json::from_str(&format!(
            r#"{{"type":"qwen3_moe_dp_attn_ep_ffn","model_config":"model/config/qwen3_235b.json","fp8":false{extra}}}"#
        ))
    }

    fn parse_fp8_qwen(extra: &str) -> Result<IterArchSel, serde_json::Error> {
        serde_json::from_str(&format!(
            r#"{{"type":"qwen3_moe_fp8_dp_attn_ep_ffn","model_config":"model/config/qwen3_235b.json","fp8":true{extra}}}"#
        ))
    }

    fn parse_vllm_qwen(extra: &str) -> Result<IterArchSel, serde_json::Error> {
        serde_json::from_str(&format!(
            r#"{{"type":"qwen3_vllm_moe_dp_attn_ep_ffn","model_config":"model/config/qwen3_235b.json","fp8":true{extra}}}"#
        ))
    }

    #[test]
    fn qwen_selector_parses_profile_path_and_marks_it_cache_relevant() {
        let parsed = parse_qwen(
            r#","attn_tp_size":4,"ep_size":4,"hp_size":1,"nvl_num_gpu":4,"expert_popularity_file":"profile_expert_popularity/expert_popularity.json""#,
        )
        .expect("qwen selector with popularity profile parses");
        let IterArchSel::Qwen3MoeDpAttnEpFfn {
            attn_tp_size,
            ep_size,
            expert_popularity_file,
            ..
        } = parsed
        else {
            panic!("expected qwen3_moe_dp_attn_ep_ffn")
        };
        assert_eq!(attn_tp_size, 4);
        assert_eq!(ep_size, 4);
        assert_eq!(
            expert_popularity_file.as_deref(),
            Some("profile_expert_popularity/expert_popularity.json")
        );

        let (_, params) = IterArchSel::SCHEMA
            .iter()
            .find(|(tag, _)| *tag == "qwen3_moe_dp_attn_ep_ffn")
            .expect("qwen selector schema row");
        let params = serde_json::to_value(params).unwrap();
        let popularity = params
            .as_array()
            .unwrap()
            .iter()
            .find(|param| param["name"] == "expert_popularity_file")
            .expect("expert popularity schema parameter");
        assert_eq!(popularity["affects_cache"], true);
    }

    #[test]
    fn vllm_qwen_selector_is_a_distinct_public_arch_with_the_same_parallel_contract() {
        assert!(matches!(
            parse_fp8_qwen(r#", "attn_tp_size":4,"ep_size":8,"hp_size":1,"nvl_num_gpu":8"#,)
                .unwrap(),
            IterArchSel::Qwen3MoeFp8DpAttnEpFfn { .. }
        ));
        let parsed = parse_vllm_qwen(
            r#", "attn_tp_size":4,"ep_size":4,"hp_size":1,"nvl_num_gpu":4,"expert_popularity_file":"expert_popularity.json""#,
        )
        .expect("vLLM-alignment Qwen selector parses");
        assert!(matches!(
            parsed,
            IterArchSel::Qwen3VllmMoeDpAttnEpFfn {
                attn_tp_size: 4,
                ep_size: 4,
                hp_size: 1,
                nvl_num_gpu: 4,
                expert_popularity_file: Some(ref path),
                ..
            } if path == "expert_popularity.json"
        ));

        let tags = IterArchSel::SCHEMA
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>();
        assert!(tags.contains(&"qwen3_moe_dp_attn_ep_ffn"));
        assert!(tags.contains(&"qwen3_moe_fp8_dp_attn_ep_ffn"));
        assert!(tags.contains(&"qwen3_vllm_moe_dp_attn_ep_ffn"));
    }
}
// ── layer-wise attn / ffn contract (AFD) ────────────────────────────────────

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
    /// Qwen3-MoE attention side (layer-wise AFD attn pool). **One worker = one DP
    /// shard**: attention sharded over `attn_tp_size` head-parallel ranks, with its
    /// own KV cache and request stream. Data parallelism is the attn pool's
    /// `replicas` (= the unified arch's `ep_size / attn_tp_size`), not an arch
    /// param. Pairs with the `qwen3_ffn_moe` ffn arch.
    Qwen3AttnTp {
        #[serde(flatten)]
        model: ModelSpec,
        /// Attention tensor-parallelism size (heads sharded across these ranks).
        #[param(default = 4, cache_key)]
        attn_tp_size: u16,
    },
}

impl AttnArchSel {
    /// The model identity/dims this arch operates on (every variant carries it).
    pub fn model(&self) -> &ModelSpec {
        match self {
            Self::Llama3AttnTp { model, .. } | Self::Qwen3AttnTp { model, .. } => model,
        }
    }
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
    /// BF16/native Qwen3-MoE FFN side (layer-wise AFD ffn pool): qkv / o_proj (on the attn-TP
    /// layout) + post_norm + router + EP MoE, plus the iteration embed / final_norm
    /// / lm_head. Mirrors the FFN-side cost of the unified `qwen3_moe_dp_attn_ep_ffn`;
    /// pairs with a `qwen3_attn` model whose `fp8` field is false.
    Qwen3FfnMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Attention tensor-parallelism size. Drives the qkv / o_proj per-rank
        /// shape AND the MoE combine residing-group width (a token resides on its
        /// qkv/o_proj TP group, so combine fans the reduced output back to all
        /// `attn_tp_size` ranks). Must match the paired attn arch's `attn_tp_size`.
        #[param(default = 4, cache_key)]
        attn_tp_size: u16,
        /// Expert-parallelism size (FFN expert ranks; spans the whole replica).
        /// `num_dp_groups = ep_size / attn_tp_size`.
        #[param(default = 8, cache_key)]
        ep_size: u16,
        /// NVLink-domain size — partitions `ep_size` ranks into NVL domains for the
        /// intra/inter split of MoE dispatch/combine.
        #[param(default = 8, cache_key)]
        nvl_num_gpu: u16,
        /// Expert routing distribution: `uniform` (default) or `random` (seeded by
        /// `routing_seed`). Drives the L2 MoE dispatch/combine simulation.
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS)]
        routing: RoutingKind,
        /// Seed for `routing = random` (ignored for `uniform`).
        #[serde(default)]
        routing_seed: Option<u64>,
    },
    /// Native FP8 Qwen3-MoE FFN side. This is a separate provider because its
    /// pre/post projection and lm-head slots are quant+GEMM compound ops. It
    /// pairs with a `qwen3_attn` model whose `fp8` field is true.
    Qwen3Fp8FfnMoe {
        #[serde(flatten)]
        model: ModelSpec,
        #[param(default = 4, cache_key)]
        attn_tp_size: u16,
        #[param(default = 8, cache_key)]
        ep_size: u16,
        #[param(default = 8, cache_key)]
        nvl_num_gpu: u16,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS)]
        routing: RoutingKind,
        #[serde(default)]
        routing_seed: Option<u64>,
    },
}

impl FfnArchSel {
    /// The model identity/dims this arch operates on (every variant carries it).
    pub fn model(&self) -> &ModelSpec {
        match self {
            Self::DeepseekFfnMoe { model, .. }
            | Self::Qwen3FfnMoe { model, .. }
            | Self::Qwen3Fp8FfnMoe { model, .. } => model,
        }
    }
}
