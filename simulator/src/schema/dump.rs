//! `list-params` schema dump — the launcher's param-schema registry.
//!
//! `simulator list-params` prints the JSON produced here; the launcher writes it
//! to `target/<profile>/deployment_schema.json` and walks a concrete config
//! against it (L7 design). It is NOT a cartesian product of
//! (deployment × arch × worker) — it publishes the structure:
//!   - `deployments`: each deployment's pool roles → contract class;
//!   - `providers`: per contract class, each arch/worker tag's params;
//!   - `arch_common`: the model fields every arch tag carries;
//!   - `group_common`: the flat fields every group carries (gpu / replicas);
//!   - `pool_common`: the flat fields every pool carries (placement);
//!   - `common`: run-global workload / io params;
//!   - `predict_cases`: per `timing-predict` arch selector, the fields of one
//!     case. `groups`: a case is `{groups: [...]}`, each group these fields;
//!     `case`: a case is an object of these fields. How many groups a model
//!     takes is the model's (`cost-trees` gives it as `predict.groups`).
//!
//! This module only *arranges* — every param's defaults / choices / cache-key
//! flag is *derived from the config types themselves*: `#[derive(ParamStruct)]`
//! emits a `PARAMS` const (ModelSpec / GroupSpec / PoolSpec / WorkloadSpec /
//! IoSpec) and `#[derive(ProviderSchema)]` a per-variant `SCHEMA` const on each
//! arch/worker selector. `GroupSpec` / `PoolSpec` are generic but their params
//! don't touch the type params, so we read them off a `<(), ()>` instantiation.
//! Side-effect-free.

use serde_json::{json, Map, Value};

use crate::arch::config::{AttnArchSel, FfnArchSel, IterArchSel, ModelSpec};
use crate::arch::contract::FfnArchInput;
use crate::deployment::config::{IoSpec, WorkloadSpec};
use crate::orchestrator::config::{GroupSpec, PoolSpec};
use crate::schema::ParamDef;
use crate::timing_predict::{PredictGroup, SpeculativePredictGroup};
use crate::worker::config::{AttnWorkerSel, FfnWorkerSel, IterWorkerSel, KvAdmissionSpec};

/// Serialize a `const PARAMS` slice to a JSON array of ParamDef objects.
fn params(p: &[ParamDef]) -> Value {
    serde_json::to_value(p).expect("ParamDef is infallibly Serialize")
}

/// Turn a layer's `(tag, params)` schema slice into `{ tag: { "params": [...] } }`.
fn providers(schema: &[(&str, &[ParamDef])]) -> Value {
    providers_with_flattened(schema, &[])
}

/// Add params from selector components flattened into only one provider tag.
fn providers_with_flattened(
    schema: &[(&str, &[ParamDef])],
    flattened: &[(&str, &[ParamDef])],
) -> Value {
    let mut m = Map::new();
    for (tag, p) in schema {
        let mut provider_params = p.to_vec();
        if let Some((_, extra)) = flattened.iter().find(|(owner, _)| owner == tag) {
            provider_params.extend_from_slice(extra);
        }
        m.insert(
            tag.to_string(),
            json!({ "params": params(&provider_params) }),
        );
    }
    Value::Object(m)
}

/// Build the full `list-params` registry JSON.
pub fn list_params() -> Value {
    json!({
        "deployments": {
            "unified": { "pools": { "main": "iter_wise" } },
            "pd":      { "pools": { "prefill": "iter_wise", "decode": "iter_wise" } },
            "afd":     { "pools": { "attn": "layer_wise_attn", "ffn": "layer_wise_ffn" } },
            "pp":      { "pools": { "stage": "iter_wise" } },
        },
        "providers": {
            "arch": {
                "iter_wise":       providers(IterArchSel::SCHEMA),
                "layer_wise_attn": providers(AttnArchSel::SCHEMA),
                "layer_wise_ffn":  providers(FfnArchSel::SCHEMA),
            },
            "worker": {
                "iter_wise":       providers_with_flattened(
                    IterWorkerSel::SCHEMA,
                    &[
                        ("chunked_prefill", KvAdmissionSpec::PARAMS),
                        ("speculative", KvAdmissionSpec::PARAMS),
                    ],
                ),
                "layer_wise_attn": providers(AttnWorkerSel::SCHEMA),
                "layer_wise_ffn":  providers(FfnWorkerSel::SCHEMA),
            },
        },
        "arch_common": params(ModelSpec::PARAMS),
        "group_common": params(GroupSpec::<(), ()>::PARAMS),
        "pool_common": params(PoolSpec::<(), ()>::PARAMS),
        "common": {
            "workload": params(WorkloadSpec::PARAMS),
            "io":       params(IoSpec::PARAMS),
        },
        "predict_cases": {
            "iter":             {"groups": params(PredictGroup::PARAMS)},
            "speculative_iter": {"groups": params(SpeculativePredictGroup::PARAMS)},
            "attn":             {"groups": params(PredictGroup::PARAMS)},
            "ffn":              {"case": params(FfnArchInput::PARAMS)},
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each `timing-predict` selector publishes its case fields, typed as the
    /// case structs deserialize them: `[a, b]` pairs as `int_pair_list`.
    #[test]
    fn predict_cases_list_each_selectors_case_fields() {
        let cases = &list_params()["predict_cases"];
        let fields = |selector: &str, key: &str| -> Vec<(String, String)> {
            cases[selector][key]
                .as_array()
                .unwrap_or_else(|| panic!("{selector}.{key}"))
                .iter()
                .map(|p| {
                    (
                        p["name"].as_str().unwrap().into(),
                        p["type"].as_str().unwrap().into(),
                    )
                })
                .collect()
        };
        let pair = |n: &str, t: &str| (n.to_string(), t.to_string());
        let group = vec![
            pair("prefill_chunk_pairs", "int_pair_list"),
            pair("decode_kv_lens", "int_list"),
            pair("decode_count", "int"),
            pair("average_decode_length", "int"),
        ];
        assert_eq!(fields("iter", "groups"), group);
        assert_eq!(fields("attn", "groups"), group);
        assert_eq!(
            fields("speculative_iter", "groups"),
            [
                pair("prefill_chunk_pairs", "int_pair_list"),
                pair("decode_requests", "int_pair_list"),
            ]
        );
        assert_eq!(
            fields("ffn", "case"),
            [pair("tokens_per_group", "int_list")]
        );
    }

    /// The group's GPU and an arch's routing pick kernels, so the launcher's
    /// cache-key dedup must tell configs apart by them.
    #[test]
    fn gpu_and_routing_affect_the_kernel_cache() {
        let schema = list_params();
        let gpu = schema["group_common"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["name"] == "gpu")
            .expect("gpu in group_common");
        assert_eq!(gpu["affects_cache"], true);
        for providers in schema["providers"]["arch"].as_object().unwrap().values() {
            for (tag, provider) in providers.as_object().unwrap() {
                for param in provider["params"].as_array().unwrap() {
                    if matches!(param["name"].as_str(), Some("routing" | "routing_seed")) {
                        assert_eq!(param["affects_cache"], true, "{tag}.{}", param["name"]);
                    }
                }
            }
        }
    }

    #[test]
    fn registry_advertises_tp_size_on_dense_tp() {
        let schema = list_params();
        let p = &schema["providers"]["arch"]["iter_wise"]["llama3_dense_tp"]["params"];
        let arr = p.as_array().expect("dense_tp params is an array");
        let tp = arr
            .iter()
            .find(|param| param["name"] == "tp_size")
            .expect("tp_size present on llama3_dense_tp");
        assert_eq!(tp["type"], "int");
        assert_eq!(tp["affects_cache"], true);
        // llama3_dense (no sharding) has no params.
        assert!(
            schema["providers"]["arch"]["iter_wise"]["llama3_dense"]["params"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn pool_common_placement_has_choices() {
        let schema = list_params();
        let pc = schema["pool_common"].as_array().expect("pool_common array");
        let placement = pc
            .iter()
            .find(|p| p["name"] == "placement")
            .expect("placement present");
        let choices = placement["choices"].as_array().expect("placement choices");
        assert!(choices.iter().any(|c| c.as_str() == Some("least-queued")));
        assert!(choices.iter().any(|c| c.as_str() == Some("round-robin")));
    }

    #[test]
    fn arch_common_model_config_required_and_cache_key() {
        let schema = list_params();
        let ac = schema["arch_common"].as_array().expect("arch_common array");
        let mc = ac
            .iter()
            .find(|p| p["name"] == "model_config")
            .expect("model_config present");
        assert_eq!(mc["required"], true);
        assert_eq!(mc["affects_cache"], true);
    }

    #[test]
    fn deployments_grammar_maps_roles_to_contracts() {
        let schema = list_params();
        assert_eq!(
            schema["deployments"]["unified"]["pools"]["main"],
            "iter_wise"
        );
        assert_eq!(
            schema["deployments"]["afd"]["pools"]["attn"],
            "layer_wise_attn"
        );
    }

    #[test]
    fn workload_schema_exposes_orthogonal_replay_axes() {
        let schema = list_params();
        let workload = schema["common"]["workload"]
            .as_array()
            .expect("workload params are an array");
        let input_file_format = workload
            .iter()
            .find(|param| param["name"] == "input_file_format")
            .expect("the complete input file format is exposed");
        assert!(input_file_format["choices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|choice| choice == "text-generation-session-execution-v2"));
        assert!(workload.iter().all(|param| !matches!(
            param["name"].as_str(),
            Some("trace_kind" | "trace_source_schema")
        )));
        let arrival_mode = workload
            .iter()
            .find(|param| param["name"] == "arrival_mode")
            .expect("arrival_mode is exposed");
        assert_eq!(arrival_mode["choices"], json!(["trace_timed", "saturated"]));
        // Capacity is its own field, not a payload of the arrival mode: every
        // combination of the two must be expressible.
        assert!(
            workload
                .iter()
                .any(|param| param["name"] == "max_concurrency"),
            "max_concurrency is exposed independently of arrival_mode"
        );
        // The trace decides whether rounds chain.
        for retired in ["replay_mode", "session_dependency"] {
            assert!(workload.iter().all(|param| param["name"] != retired));
        }
    }

    #[test]
    fn attention_workers_expose_prefix_cache_mode_policy_and_optional_ceiling() {
        let schema = list_params();
        for params in [
            &schema["providers"]["worker"]["iter_wise"]["barebone"]["params"],
            &schema["providers"]["worker"]["iter_wise"]["hp_unified"]["params"],
            &schema["providers"]["worker"]["iter_wise"]["pd_prefill"]["params"],
            &schema["providers"]["worker"]["layer_wise_attn"]["disagg_attn"]["params"],
        ] {
            let params = params.as_array().expect("worker params are an array");
            let mode = params
                .iter()
                .find(|param| param["name"] == "prefix_cache_mode")
                .expect("prefix_cache_mode is exposed");
            assert_eq!(mode["default"], "opportunistic");
            assert_eq!(mode["choices"], json!(["disabled", "opportunistic"]));

            let policy = params
                .iter()
                .find(|param| param["name"] == "prefix_cache_policy")
                .expect("prefix_cache_policy is exposed");
            assert_eq!(policy["default"], "lru");

            let ceiling = params
                .iter()
                .find(|param| param["name"] == "prefix_cache_max_gpu_memory_gb")
                .expect("optional prefix cache ceiling is exposed");
            assert_eq!(ceiling["type"], "float");
            assert_eq!(ceiling["required"], false);
        }

        let pd_decode = schema["providers"]["worker"]["iter_wise"]["pd_decode"]["params"]
            .as_array()
            .expect("pd_decode params are an array");
        assert!(pd_decode.iter().all(|param| !param["name"]
            .as_str()
            .unwrap_or_default()
            .starts_with("prefix_cache_")));
    }

    #[test]
    fn iter_worker_schemas_include_flattened_kv_admission_spec() {
        let schema = list_params();
        for worker in ["chunked_prefill", "speculative"] {
            check_flattened_kv_admission_spec(&schema, worker);
        }
    }

    fn check_flattened_kv_admission_spec(schema: &serde_json::Value, worker: &str) {
        let params = schema["providers"]["worker"]["iter_wise"][worker]["params"]
            .as_array()
            .expect("worker params are an array");
        let policy = params
            .iter()
            .find(|param| param["name"] == "kv_admission_policy")
            .expect("flattened KV admission policy is exposed");
        assert_eq!(policy["default"], "full-footprint");
        assert_eq!(
            policy["choices"],
            json!(["full-footprint", "bounded-future"])
        );
        for name in [
            "kv_page_size",
            "kv_max_future_tokens",
            "kv_initial_new_token_ratio",
            "kv_minimum_new_token_ratio",
            "kv_new_token_ratio_decay_steps",
            "kv_retract_decode_steps",
            "decode_retraction_policy",
        ] {
            assert_eq!(
                params.iter().filter(|param| param["name"] == name).count(),
                1,
                "flattened parameter {name} must appear exactly once"
            );
        }
    }
}
