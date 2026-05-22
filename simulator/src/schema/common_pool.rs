//! Pool fragments — reusable groups of CLI params shared across deployments.
//!
//! Per L7 design.md §1.8.0 / §1.8.3 the cross-deployment sharing is a
//! **code-reuse** mechanism, not a schema concept: each fragment is a clap
//! `Args` struct (parsed via `#[command(flatten)]` inside a deployment's
//! `Params`). A deployment *selects* the fragments it accepts; the launcher
//! only ever sees the flattened result and has no notion of "common" params.
//!
//! **Single source.** Each fragment also `#[derive(DeploymentParams)]`
//! (in-tree `schema-derive`), which reads the clap fields and generates the
//! fragment's `OWN_PARAMS: &[ParamDef]` slice — the schema half that
//! `schema::dump` dumps into each deployment's `list-params` JSON. Name / type
//! / default / required are inferred from the field, so adding a field updates
//! the CLI surface and the schema at once; the per-deployment
//! `*_clap_matches_paramdef` test (design §1.8.4) stays as a drift sentinel.
//!
//! Defaults / descriptions are lifted from the reference launcher
//! (`ref/.../moesim-rs/launcher_ui/schema.py`) so behavior matches today's
//! simulator. Closed string value sets are declared on the clap
//! `value_parser` and surfaced to the schema via `#[param(choices = CONST)]`;
//! the launcher validates them generically.
//!
//! **`#[param(cache_key)]` tagging.** Params tagged `#[param(cache_key)]`
//! change which kernel/comm configs `profile.db` must contain (model
//! identity/dims, sharding, dtype, fabric). The launcher groups sweep runs by
//! the tuple of cache-key params and runs one `build-cache-only` per unique
//! group (design §1.2.2; rationale + safe-direction rule in `param_def.rs`).
//! Tagged here: `model_config` (model dims / identity), `fp8` (dtype),
//! `tp_size` / `ep_size` / `head_parallel` (per-rank shapes), and `cp_plan`
//! (attention composition — ring vs no-cp run different attention kernels).
//! Architecture or timing variants that need different param surfaces become
//! separate deployments or deployment-owned fragments, not a string-valued
//! common param. NOT tagged: `num_layers` / `sim_num_layers` (change layer
//! COUNT, not per-layer kernel shape), workload / IO params (request count /
//! logging only).

use std::path::PathBuf;

use schema_derive::DeploymentParams;

pub const CP_PLAN_CHOICES: [&str; 4] = ["no-cp", "ring", "replicated-q", "optimal-prefill"];
pub const LOG_LEVEL_CHOICES: [&str; 5] = ["trace", "debug", "info", "warn", "error"];

/// Model identity + common layer controls. Flattened by every deployment.
#[derive(clap::Args, Debug, Clone, DeploymentParams)]
pub struct ModelCommon {
    /// Path to the model config JSON (or a known model name).
    #[arg(long)]
    #[param(cache_key)]
    pub model_config: String,

    /// Number of transformer layers (omit to use the model config's value).
    #[arg(long)]
    pub num_layers: Option<u32>,

    /// Simulate only this many layers with scaled timing (omit = all layers;
    /// KV cache still uses the real layer count).
    #[arg(long)]
    pub sim_num_layers: Option<u32>,

    /// Use FP8 precision (DeepGEMM / fp8 prefill, halved transfers).
    #[arg(long)]
    #[param(cache_key)]
    pub fp8: bool,
}

/// Parallelism degrees. Flattened by every deployment.
#[derive(clap::Args, Debug, Clone, DeploymentParams)]
pub struct ParallelismCommon {
    /// Tensor parallelism size.
    #[arg(long, default_value_t = 4)]
    #[param(cache_key)]
    pub tp_size: u16,

    /// Expert parallelism size.
    #[arg(long, default_value_t = 8)]
    #[param(cache_key)]
    pub ep_size: u16,

    /// Attention head parallelism.
    #[arg(long, default_value_t = 1)]
    #[param(cache_key)]
    pub head_parallel: u16,

    /// Context-parallel attention plan.
    #[arg(
        long,
        default_value = "no-cp",
        value_parser = clap::builder::PossibleValuesParser::new(CP_PLAN_CHOICES)
    )]
    #[param(cache_key, choices = CP_PLAN_CHOICES)]
    pub cp_plan: String,
}

/// Workload / trace inputs. Flattened by every deployment.
#[derive(clap::Args, Debug, Clone, DeploymentParams)]
pub struct WorkloadCommon {
    /// Trace CSV files to simulate (repeat the flag; runs sequentially).
    #[arg(long = "trace-files", value_name = "PATH")]
    pub trace_files: Vec<PathBuf>,

    /// Simulation duration (ms); minimum window when --run-to-end is set.
    #[arg(long, default_value_t = 5000.0)]
    pub duration_ms: f64,

    /// Keep ticking past duration_ms until every request completes.
    #[arg(long)]
    pub run_to_end: bool,

    /// Request arrival rate (requests/s).
    #[arg(long, default_value_t = 10.0)]
    pub request_rate: f64,
}

/// Output location + logging controls. Flattened by every deployment.
#[derive(clap::Args, Debug, Clone, DeploymentParams)]
pub struct IoCommon {
    /// Directory for this run's logs / parquet outputs.
    #[arg(long, default_value = "logs")]
    pub log_dir: PathBuf,

    /// Log verbosity (trace / debug / info / warn / error).
    #[arg(
        long,
        default_value = "info",
        value_parser = clap::builder::PossibleValuesParser::new(LOG_LEVEL_CHOICES)
    )]
    #[param(choices = LOG_LEVEL_CHOICES)]
    pub log_level: String,

    /// Suppress per-tick progress chatter on stdout.
    #[arg(long)]
    pub quiet: bool,

    /// Force-refresh perf_api rows while building startup caches.
    #[arg(long)]
    pub force_cache_build: bool,
}
