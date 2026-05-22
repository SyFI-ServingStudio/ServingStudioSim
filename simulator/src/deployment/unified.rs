//! `unified` deployment — single worker type runs the whole model per
//! iteration (co-located attention + FFN). First milestone target
//! (Llama3-8B dense, local single server).
//!
//! Shape per L7 design.md §1.8.3: `UnifiedParams` flattens the four pool
//! fragments + unified-own fields; `PARAM_GROUPS` lists the matching `ParamDef`
//! slices in the same order. The `clap` struct and the slices are kept in sync
//! by `unified_clap_matches_paramdef` (design §1.8.4). `build()` is deferred
//! until the L6 `Flow` / L7-β tick driver exist (see `deployment/mod.rs`).

use crate::schema::common_pool::{IoCommon, ModelCommon, ParallelismCommon, WorkloadCommon};

use super::Deployment;

// `DeploymentParams` (in-tree `schema-derive`) reads the clap fields below and
// generates `UnifiedParams::OWN_PARAMS`, the `ParamDef` schema slice for the
// unified-own params — so the CLI surface and `list-params` schema can't drift
// (design §1.8.3/§1.8.4). `#[command(flatten)]` fragments are skipped; their
// params come from the fragment slices in `PARAM_GROUPS`. Name / type / default
// / required are inferred from the field; only the clap-invisible bits ride on
// `#[param(...)]`.
#[derive(clap::Args, Debug, Clone, schema_derive::DeploymentParams)]
pub struct UnifiedParams {
    #[command(flatten)]
    pub model: ModelCommon,
    #[command(flatten)]
    pub parallelism: ParallelismCommon,
    #[command(flatten)]
    pub workload: WorkloadCommon,
    #[command(flatten)]
    pub io: IoCommon,

    // ── unified-own fields (design §1.8.0) ──────────────────────────────────
    /// GPU memory for the co-located worker (GB; primarily KV cache budget).
    #[arg(long, default_value_t = 80.0)]
    pub attn_gpu_memory_gb: f64,

    /// Chunked-prefill cap: max tokens per batch (omit = unlimited).
    #[arg(long)]
    pub max_batch_tokens: Option<u32>,

    /// Delay prefill admission until all HP groups have pending work, up to
    /// this many batch-formation passes (omit = disabled).
    #[arg(long)]
    pub prefill_delayer_max_delay_passes: Option<u32>,

    /// Batch composition policy: how returning decode mixes with pending
    /// prefill (e.g. mix / separate-prefill-priority).
    #[arg(
        long,
        default_value = "mix",
        value_parser = clap::builder::PossibleValuesParser::new(UNIFIED_BATCH_POLICY_CHOICES)
    )]
    #[param(choices = UNIFIED_BATCH_POLICY_CHOICES)]
    pub unified_batch_policy: String,

    /// NVLink domain size for MoE communication modeling (0 = disabled).
    // cache_key: NVLink domain size feeds fabric resolution for MoE all-to-all,
    // so it changes which comm rows profile.db needs (safe-direction: tag it).
    #[arg(long, default_value_t = 0)]
    #[param(cache_key)]
    pub nvl_num_gpu: u16,
}

const UNIFIED_BATCH_POLICY_CHOICES: [&str; 3] = [
    "mix",
    "separate-prefill-priority",
    "separate-prefill-priority-no-interleave",
];

pub struct UnifiedDeployment;

impl Deployment for UnifiedDeployment {
    const NAME: &'static str = "unified";
    type Args = UnifiedParams;
    // PARAM_GROUPS defaults to <UnifiedParams as ParamSchema>::PARAM_GROUPS,
    // composed by #[derive(DeploymentParams)] from the flattened fragments + own.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::flatten_params;
    use clap::{error::ErrorKind, Args as _, Command};
    use std::collections::HashSet;

    /// design §1.8.4: the clap arg surface must exactly match the `ParamDef`
    /// schema slices, so `list-params` JSON never drifts from what `run` parses.
    /// `UnifiedParams` is `#[derive(Args)]` (flattened, not a top-level
    /// `Parser`), so build its `Command` via `augment_args` rather than
    /// `CommandFactory::command()`.
    #[test]
    fn unified_clap_matches_paramdef() {
        let clap_ids: HashSet<String> = UnifiedParams::augment_args(Command::new("unified"))
            .get_arguments()
            .map(|a| a.get_id().to_string())
            .filter(|id| id != "help" && id != "version")
            .collect();

        let pd_names: HashSet<String> = flatten_params(UnifiedDeployment::PARAM_GROUPS)
            .iter()
            .map(|p| p.name.to_string())
            .collect();

        assert_eq!(
            clap_ids, pd_names,
            "UnifiedParams clap args ≠ flattened PARAM_GROUPS names"
        );
    }

    #[test]
    fn unified_clap_rejects_invalid_choices() {
        for (flag, value) in [
            ("--cp-plan", "bad-plan"),
            ("--log-level", "verbose"),
            ("--unified-batch-policy", "bad-policy"),
        ] {
            let err = UnifiedParams::augment_args(Command::new("unified"))
                .try_get_matches_from(["unified", "--model-config", "model.json", flag, value])
                .expect_err("invalid enum-like choice should fail clap parsing");
            assert_eq!(err.kind(), ErrorKind::InvalidValue);
        }
    }
}
