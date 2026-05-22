//! `list-params` schema dump — the launcher's single source of per-param data.
//!
//! `simulator list-params` prints the JSON produced here; the launcher writes
//! it to `target/<profile>/deployment_schema.json` (L7 design.md §1.2.7). The
//! `deployment_schemas` block is authoritative (each entry is a complete,
//! flattened schema). The `pool_fragments` block is emitted **solely** so
//! `list-params --human` can group a deployment's params under their source
//! fragment for readability — no validation / expansion logic consumes it, so
//! it does not reintroduce a second source of param truth (INV-7 / §1.8.3).
//! Side-effect-free: it walks `const` data and serializes, no GPU / DB / PyO3.

use serde_json::{Map, Value};

use crate::deployment::{flatten_params, unified::UnifiedDeployment, Deployment};
use crate::schema::common_pool::{IoCommon, ModelCommon, ParallelismCommon, WorkloadCommon};
use crate::schema::ParamDef;

/// Serialize one deployment's flattened `PARAM_GROUPS` to a JSON array.
fn deployment_schema(groups: &[&[ParamDef]]) -> Value {
    serde_json::to_value(flatten_params(groups))
        .expect("ParamDef is infallibly Serialize (no maps with non-string keys)")
}

/// Param names of a pool fragment, for the display-only `pool_fragments` block.
fn names(params: &[ParamDef]) -> Value {
    Value::Array(params.iter().map(|p| Value::from(p.name)).collect())
}

/// Build the full `list-params` JSON document. Add a deployment by inserting
/// one line into `deployment_schemas` (mirrors the dispatch arm in `main.rs`).
pub fn list_params() -> Value {
    let mut deployment_schemas = Map::new();
    deployment_schemas.insert(
        UnifiedDeployment::NAME.to_string(),
        deployment_schema(UnifiedDeployment::PARAM_GROUPS),
    );

    // Display-only: lets `--human` group a deployment's params by source
    // fragment. Declaration order here is the order `--human` prints the groups.
    let mut pool_fragments = Map::new();
    pool_fragments.insert("ModelCommon".to_string(), names(ModelCommon::OWN_PARAMS));
    pool_fragments.insert(
        "ParallelismCommon".to_string(),
        names(ParallelismCommon::OWN_PARAMS),
    );
    pool_fragments.insert(
        "WorkloadCommon".to_string(),
        names(WorkloadCommon::OWN_PARAMS),
    );
    pool_fragments.insert("IoCommon".to_string(), names(IoCommon::OWN_PARAMS));

    let mut root = Map::new();
    root.insert(
        "deployment_schemas".to_string(),
        Value::Object(deployment_schemas),
    );
    root.insert("pool_fragments".to_string(), Value::Object(pool_fragments));
    Value::Object(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_params_emits_rust_owned_choices() {
        let schema = list_params();
        let params = schema["deployment_schemas"]["unified"]
            .as_array()
            .expect("unified schema is an array");
        let cp_plan = params
            .iter()
            .find(|param| param["name"] == "cp_plan")
            .expect("cp_plan ParamDef is present");
        let choices = cp_plan["choices"]
            .as_array()
            .expect("cp_plan choices are serialized");

        assert!(choices
            .iter()
            .any(|choice| choice.as_str() == Some("no-cp")));
        assert!(choices
            .iter()
            .any(|choice| choice.as_str() == Some("replicated-q")));
    }
}
