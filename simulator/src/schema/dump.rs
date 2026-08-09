//! `list-params` schema dump — the launcher's param-schema registry.
//!
//! `simulator list-params` prints the JSON produced here; the launcher writes it
//! to `target/<profile>/deployment_schema.json` and walks a concrete config
//! against it (new-interface-design §11). It is NOT a cartesian product of
//! (deployment × arch × worker) — it publishes the structure:
//!   - `deployments`: each deployment's pool roles → contract class;
//!   - `providers`: per contract class, each arch/worker tag's params;
//!   - `arch_common`: the model fields every arch tag carries;
//!   - `group_common`: the flat fields every group carries (gpu / replicas);
//!   - `pool_common`: the flat fields every pool carries (placement);
//!   - `common`: run-global workload / io params.
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
use crate::deployment::config::{IoSpec, WorkloadSpec};
use crate::orchestrator::config::{GroupSpec, PoolSpec};
use crate::schema::ParamDef;
use crate::worker::config::{AttnWorkerSel, FfnWorkerSel, IterWorkerSel};

/// Serialize a `const PARAMS` slice to a JSON array of ParamDef objects.
fn params(p: &[ParamDef]) -> Value {
    serde_json::to_value(p).expect("ParamDef is infallibly Serialize")
}

/// Turn a layer's `(tag, params)` schema slice into `{ tag: { "params": [...] } }`.
fn providers(schema: &[(&str, &[ParamDef])]) -> Value {
    let mut m = Map::new();
    for (tag, p) in schema {
        m.insert(tag.to_string(), json!({ "params": params(p) }));
    }
    Value::Object(m)
}

/// Build the full `list-params` registry JSON (new-interface-design §11.1).
pub fn list_params() -> Value {
    json!({
        "deployments": {
            "unified": { "pools": { "main": "iter_wise" } },
            "pd":      { "pools": { "prefill": "iter_wise", "decode": "iter_wise" } },
            "afd":     { "pools": { "attn": "layer_wise_attn", "ffn": "layer_wise_ffn" } },
        },
        "providers": {
            "arch": {
                "iter_wise":       providers(IterArchSel::SCHEMA),
                "layer_wise_attn": providers(AttnArchSel::SCHEMA),
                "layer_wise_ffn":  providers(FfnArchSel::SCHEMA),
            },
            "worker": {
                "iter_wise":       providers(IterWorkerSel::SCHEMA),
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
