//! Arch (L4) config surface — the model-arch *selectors*, co-located with the L4
//! implementations they pick (new-interface-design §2 / §4).
//!
//! Each selector is a serde tagged enum (`#[serde(tag = "type")]`): choosing the
//! tag is the only way that variant's params appear — no global union,
//! provider-first. `model_config` + dims belong to the arch (the layer that
//! consumes them), so every arch variant flattens [`ModelSpec`]. arch and worker
//! are symmetric sibling providers (not nested).
//!
//! Iter-wise Llama selectors, three Qwen3-MoE execution graphs (native BF16,
//! native FP8, vLLM-aligned FP8), and two GLM-5.2 DSA/MoE graphs (native and
//! vLLM-aligned) build today. Each has a distinct selector and L4
//! implementation; precision/backend selection is therefore not hidden behind an
//! in-arch recipe branch, and a `_vllm_` variant differs from its sibling only in
//! how finely it cuts the same work into leaves, so a measured nsys timeline can
//! be reconciled leaf by leaf. They pair with the same `hp_unified` worker as the
//! DP-attn dense variant (one KV partition state per attn-DP shard).
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

use super::glm52_dsa_moe::Glm52MtpMode;

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

/// GLM-5.2 proposer modes exposed by the iter selector. This mirrors
/// [`Glm52MtpMode`]'s serde representation.
const GLM52_MTP_MODES: [&str; 3] = ["off", "full_index", "index_share"];

const fn default_glm52_parallel_size() -> u16 {
    8
}

/// Iteration-wise arch provider. Sharding parameters live only on the variants
/// that consume them (provider-first: select the arch, then it exposes its own
/// params).
#[derive(Debug, Clone, Deserialize, ProviderSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IterArchSel {
    /// Exact text-only Qwen3.6-35B-A3B FP8 execution graph on one local H200.
    /// TP1/EP1 are architectural invariants, so this selector exposes no
    /// sharding parameters — but routing is not a sharding parameter. Every
    /// expert lives on the one GPU, which removes dispatch/combine traffic, not
    /// the grouped GEMM's dependence on how many tokens each expert draws: skew
    /// leaves the total token-expert selections unchanged while redistributing
    /// them into fuller and emptier groups.
    Qwen36Local {
        #[serde(flatten)]
        model: ModelSpec,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS)]
        routing: RoutingKind,
        #[serde(default)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
    },
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
    /// DeepSeek-V4-Flash-0731 at vLLM's physical kernel boundaries. EP4 and
    /// local attention DP4 are checkpoint/deployment invariants.
    DeepseekV4Vllm {
        #[serde(flatten)]
        model: ModelSpec,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS)]
        routing: RoutingKind,
        #[serde(default)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
    },
    /// The same physical DeepSeek kernels with each source-level parallel
    /// region serialized. This is an explicit alignment counterfactual, not a
    /// hidden runtime knob on the production selector.
    DeepseekV4VllmSerialStreams {
        #[serde(flatten)]
        model: ModelSpec,
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS)]
        routing: RoutingKind,
        #[serde(default)]
        routing_seed: Option<u64>,
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
    },
    /// The same GLM-5.2 schedule as `glm52_dsa_moe`, expressed in vLLM's kernel
    /// granularity for framework alignment. Same parameters, same topology; the
    /// graphs differ only in how leaves are cut. See
    /// `arch/glm52_vllm_dsa_moe.rs` for the itemised divergences.
    Glm52VllmDsaMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Expert-parallel ranks and independent local-attention DP groups.
        #[serde(default = "default_glm52_parallel_size")]
        #[param(default = 8, cache_key)]
        ep_size: u16,
        /// NVLink-domain size for the MoE dispatch/combine split.
        #[serde(default = "default_glm52_parallel_size")]
        #[param(default = 8, cache_key)]
        nvl_num_gpu: u16,
        /// Expert routing distribution used by dispatch/combine.
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS)]
        routing: RoutingKind,
        /// Seed for `routing = random`; ignored for uniform routing.
        #[serde(default)]
        routing_seed: Option<u64>,
        /// Optional MTP proposer work: off, full-index step 0, or a later
        /// IndexShare step.
        #[serde(default)]
        #[param(string, default = "off", choices = GLM52_MTP_MODES, cache_key)]
        mtp_mode: Glm52MtpMode,
        /// Measured per-expert popularity from a profile pass. Requires
        /// `routing = uniform`; the profile replaces the synthetic distribution.
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
    },
    /// GLM-5.2's exact heterogeneous 78-layer DSA/MoE schedule. Attention is
    /// local (TP1) on every EP rank and pairs with `hp_unified`, whose KV/input
    /// partitions correspond one-for-one with the EP ranks.
    Glm52DsaMoe {
        #[serde(flatten)]
        model: ModelSpec,
        /// Expert-parallel ranks and independent local-attention DP groups.
        #[serde(default = "default_glm52_parallel_size")]
        #[param(default = 8, cache_key)]
        ep_size: u16,
        /// NVLink-domain size for the MoE dispatch/combine split.
        #[serde(default = "default_glm52_parallel_size")]
        #[param(default = 8, cache_key)]
        nvl_num_gpu: u16,
        /// Expert routing distribution used by dispatch/combine.
        #[serde(default)]
        #[param(string, default = "uniform", choices = ROUTING_KINDS)]
        routing: RoutingKind,
        /// Seed for `routing = random`; ignored for uniform routing.
        #[serde(default)]
        routing_seed: Option<u64>,
        /// Optional MTP proposer work: off, full-index step 0, or a later
        /// IndexShare step.
        #[serde(default)]
        #[param(string, default = "off", choices = GLM52_MTP_MODES, cache_key)]
        mtp_mode: Glm52MtpMode,
        /// Measured per-expert popularity from a profile pass. Requires
        /// `routing = uniform`; the profile replaces the synthetic distribution.
        #[serde(default)]
        #[param(cache_key)]
        expert_popularity_file: Option<String>,
    },
}

impl IterArchSel {
    /// The model identity/dims this arch operates on (every variant carries it).
    pub fn model(&self) -> &ModelSpec {
        match self {
            Self::Qwen36Local { model, .. }
            | Self::Llama3Dense { model }
            | Self::Llama3DenseTp { model, .. }
            | Self::Llama3DpAttnTpFfn { model, .. }
            | Self::Qwen3MoeDpAttnEpFfn { model, .. }
            | Self::Qwen3MoeFp8DpAttnEpFfn { model, .. }
            | Self::Qwen3VllmMoeDpAttnEpFfn { model, .. }
            | Self::DeepseekV4Vllm { model, .. }
            | Self::DeepseekV4VllmSerialStreams { model, .. }
            | Self::Glm52DsaMoe { model, .. }
            | Self::Glm52VllmDsaMoe { model, .. } => model,
        }
    }
}

#[cfg(test)]
mod iter_tests {
    use super::*;

    #[test]
    fn qwen36_local_selector_publishes_only_routing_beyond_the_model() {
        let parsed: IterArchSel = serde_json::from_str(
            r#"{"type":"qwen36_local","model_config":"model/config/qwen3_6_35b_a3b_fp8.json","num_layers":40,"sim_num_layers":4,"fp8":true}"#,
        )
        .expect("qwen36_local selector parses");
        let IterArchSel::Qwen36Local {
            model,
            routing,
            routing_seed,
            expert_popularity_file,
        } = &parsed
        else {
            panic!("expected qwen36_local")
        };
        // Routing is optional on the wire and defaults to balanced, so an
        // existing preset keeps its meaning.
        assert_eq!(*routing, RoutingKind::Uniform);
        assert!(routing_seed.is_none() && expert_popularity_file.is_none());
        assert_eq!(model.model_config, "model/config/qwen3_6_35b_a3b_fp8.json");
        assert_eq!(model.num_layers, Some(40));
        assert_eq!(model.sim_num_layers, Some(4));
        assert!(model.fp8);
        assert!(std::ptr::eq(parsed.model(), model));

        let (tag, params) = IterArchSel::SCHEMA
            .iter()
            .find(|(tag, _)| *tag == "qwen36_local")
            .expect("qwen36_local provider schema row");
        assert_eq!(*tag, "qwen36_local");
        // Flattened ModelSpec fields are intentionally published once through
        // schema::dump::arch_common, not duplicated on every provider row. TP1
        // and EP1 are architectural invariants, so routing is the ONLY thing
        // this arch adds: it is a property of the workload's token stream, not
        // of a sharding choice, and the grouped GEMM costs it either way.
        let published = serde_json::to_value(params).unwrap();
        let published: Vec<&str> = published
            .as_array()
            .unwrap()
            .iter()
            .map(|param| param["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            published,
            ["routing", "routing_seed", "expert_popularity_file"]
        );
        let popularity = serde_json::to_value(params).unwrap();
        let popularity = popularity
            .as_array()
            .unwrap()
            .iter()
            .find(|param| param["name"] == "expert_popularity_file")
            .expect("expert popularity schema parameter")
            .clone();
        assert_eq!(popularity["affects_cache"], true);
        let common = serde_json::to_value(ModelSpec::PARAMS).unwrap();
        let names = common
            .as_array()
            .unwrap()
            .iter()
            .map(|param| param["name"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            ["model_config", "num_layers", "sim_num_layers", "fp8"]
        );
    }

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

    fn parse(extra: &str) -> Result<IterArchSel, serde_json::Error> {
        serde_json::from_str(&format!(
            r#"{{"type":"glm52_dsa_moe","model_config":"model/config/glm52.json","fp8":false{extra}}}"#
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

    #[test]
    fn glm52_vllm_selector_is_registered_and_shares_the_glm_parameter_surface() {
        // `fp8: true` pairs with the FP8 checkpoint's config, not the BF16 one:
        // `model/work/floors.py` rejects the mismatched combination, and this
        // literal is what gets copied into new presets.
        let parsed: IterArchSel = serde_json::from_str(
            r#"{"type":"glm52_vllm_dsa_moe","model_config":"model/config/glm52_fp8.json","fp8":true}"#,
        )
        .expect("vLLM-granularity GLM selector parses");
        let IterArchSel::Glm52VllmDsaMoe {
            model,
            ep_size,
            nvl_num_gpu,
            routing,
            routing_seed,
            mtp_mode,
            expert_popularity_file,
        } = &parsed
        else {
            panic!("expected glm52_vllm_dsa_moe")
        };
        assert_eq!(model.model_config, "model/config/glm52_fp8.json");
        assert!(model.fp8);
        assert_eq!(*ep_size, 8);
        assert_eq!(*nvl_num_gpu, 8);
        assert_eq!(*routing, RoutingKind::Uniform);
        assert_eq!(*routing_seed, None);
        assert_eq!(*mtp_mode, Glm52MtpMode::Off);
        assert_eq!(*expert_popularity_file, None);
        assert_eq!(parsed.model().model_config, model.model_config);

        let tags = IterArchSel::SCHEMA
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>();
        assert!(tags.contains(&"glm52_vllm_dsa_moe"));
    }

    #[test]
    fn glm52_selector_accepts_a_measured_expert_popularity_profile() {
        // The GLM graphs previously had no way to consume a measured expert
        // distribution, so a DP+EP alignment run could only assume uniform
        // routing -- a silent confound on exactly the grouped-GEMM operations
        // that deviate most. Same field and same semantics as the Qwen
        // variants: the profile replaces the synthetic distribution and is
        // rejected outright if a non-uniform `routing` is also requested.
        for tag in ["glm52_dsa_moe", "glm52_vllm_dsa_moe"] {
            let parsed: IterArchSel = serde_json::from_str(&format!(
                r#"{{"type":"{tag}","model_config":"model/config/glm52_fp8.json","fp8":true,
                     "expert_popularity_file":"profile_expert_popularity/expert_popularity.json"}}"#
            ))
            .expect("GLM selector accepts an expert-popularity profile");
            let file = match &parsed {
                IterArchSel::Glm52DsaMoe {
                    expert_popularity_file,
                    ..
                }
                | IterArchSel::Glm52VllmDsaMoe {
                    expert_popularity_file,
                    ..
                } => expert_popularity_file.clone(),
                _ => panic!("expected a GLM arch"),
            };
            assert_eq!(
                file.as_deref(),
                Some("profile_expert_popularity/expert_popularity.json")
            );
        }
    }

    #[test]
    fn glm52_selector_defaults_and_model_are_exact() {
        let parsed = parse("").expect("default GLM selector parses");
        let IterArchSel::Glm52DsaMoe {
            model,
            ep_size,
            nvl_num_gpu,
            routing,
            routing_seed,
            mtp_mode,
            expert_popularity_file,
        } = &parsed
        else {
            panic!("expected glm52_dsa_moe")
        };
        assert_eq!(model.model_config, "model/config/glm52.json");
        assert!(!model.fp8);
        assert_eq!(*ep_size, 8);
        assert_eq!(*nvl_num_gpu, 8);
        assert_eq!(*routing, RoutingKind::Uniform);
        assert_eq!(*routing_seed, None);
        assert_eq!(*mtp_mode, Glm52MtpMode::Off);
        assert_eq!(*expert_popularity_file, None);
        assert!(std::ptr::eq(parsed.model(), model));
    }

    #[test]
    fn glm52_selector_parses_parallel_routing_and_every_mtp_mode() {
        for (wire, expected) in [
            ("off", Glm52MtpMode::Off),
            ("full_index", Glm52MtpMode::FullIndex),
            ("index_share", Glm52MtpMode::IndexShare),
        ] {
            let parsed = parse(&format!(
                r#","ep_size":16,"nvl_num_gpu":8,"routing":"random","routing_seed":73,"mtp_mode":"{wire}""#
            ))
            .unwrap();
            assert!(matches!(
                parsed,
                IterArchSel::Glm52DsaMoe {
                    ep_size: 16,
                    nvl_num_gpu: 8,
                    routing: RoutingKind::Random,
                    routing_seed: Some(73),
                    mtp_mode,
                    ..
                } if mtp_mode == expected
            ));
        }
    }

    #[test]
    fn glm52_selector_rejects_malformed_mtp_mode() {
        let error = parse(r#","mtp_mode":"shared""#).unwrap_err();
        assert!(error.to_string().contains("unknown variant"));
    }

    #[test]
    fn glm52_schema_exposes_exact_defaults_choices_and_cache_keys() {
        let (_, params) = IterArchSel::SCHEMA
            .iter()
            .find(|(tag, _)| *tag == "glm52_dsa_moe")
            .expect("glm52 selector schema row");
        let params = serde_json::to_value(params).unwrap();
        let get = |name: &str| {
            params
                .as_array()
                .unwrap()
                .iter()
                .find(|param| param["name"] == name)
                .unwrap()
        };
        for name in ["ep_size", "nvl_num_gpu"] {
            let param = get(name);
            assert_eq!(param["default"], 8);
            assert_eq!(param["affects_cache"], true);
        }
        let routing = get("routing");
        assert_eq!(routing["default"], "uniform");
        assert_eq!(routing["choices"], serde_json::json!(["uniform", "random"]));
        let mtp = get("mtp_mode");
        assert_eq!(mtp["default"], "off");
        assert_eq!(
            mtp["choices"],
            serde_json::json!(["off", "full_index", "index_share"])
        );
        assert_eq!(mtp["affects_cache"], true);
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
