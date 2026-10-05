//! Deployment registry — each deployment deserializes a structured run config
//! and assembles the L6b `Flow` it represents.
//!
//! Per the L6 design, the `deployment` tag (top-level serde tag on
//! `config::RunConfig`) fixes the orchestrator (L6, 1:1), the topology, and each
//! pool's contract class. A `Deployment` impl turns its concrete config into a
//! `Box<dyn Flow>`; the per-arch/worker providers inside the config carry their
//! own params (no global param union). The launcher-facing param schema lives in
//! `schema::dump` (a structural registry), not on this trait.

pub mod afd;
pub mod config;
pub mod pd;
pub mod unified;

pub use config::{
    AfdConfig, BackendOverrides, IoSpec, LogLevel, PdConfig, RunConfig, UnifiedConfig, WorkloadSpec,
};

use anyhow::ensure;

use crate::common::SharedRequests;
use crate::orchestrator::Flow;
use crate::timing::PerfApiBridge;
use crate::worker::IterWorkerSel;

/// Dispatch a deserialized `RunConfig` to its deployment's `build`. The single
/// place the `deployment` tag routes to a concrete topology. `unified`, `pd`, and
/// `afd` are all wired. What every deployment requires of its run config is
/// checked here first, once.
pub fn build_flow(
    cfg: &RunConfig,
    bridge: &PerfApiBridge,
    store: SharedRequests,
) -> anyhow::Result<Box<dyn Flow>> {
    ensure_speculative_trace(cfg)?;
    match cfg {
        RunConfig::Unified(c) => unified::UnifiedDeployment::build(c, bridge, store),
        RunConfig::Pd(c) => pd::PdDeployment::build(c, bridge, store),
        RunConfig::Afd(c) => afd::AfdDeployment::build(c, bridge, store),
    }
}

/// A speculative worker draws each request's accepted draft length from the
/// request's own `accept_rate`, which only a `speculative`-tagged trace carries
/// (the tag makes the column required on every row). Without the tag every
/// request is a standard one, which the worker cannot serve and would panic on
/// at its first decode; refuse the pairing before anything is built, for a
/// speculative worker in any pool.
fn ensure_speculative_trace(cfg: &RunConfig) -> anyhow::Result<()> {
    if cfg
        .workload()
        .input_file_tags
        .iter()
        .any(|tag| tag == "speculative")
    {
        return Ok(());
    }
    for (role, worker) in cfg.iter_workers() {
        ensure!(
            !matches!(worker, IterWorkerSel::Speculative { .. }),
            "pool {role}: the speculative worker needs each request's accept_rate; tag the \
             trace `speculative` (workload.input_file_tags) and give every row an accept_rate"
        );
    }
    Ok(())
}

/// One simulation topology. `Config` is the deserialized per-deployment config
/// (a variant of `config::RunConfig`); `build` runs the L4 cascade (via the
/// `bridge`) and assembles the L6b `Flow` the tick driver runs. `store` is the
/// shared `RequestStore` injected into every worker.
pub trait Deployment {
    /// serde `deployment` tag value + `list-params` grammar key (e.g. `"unified"`).
    const NAME: &'static str;

    /// The deserialized config this deployment consumes.
    type Config;

    fn build(
        cfg: &Self::Config,
        bridge: &PerfApiBridge,
        store: SharedRequests,
    ) -> anyhow::Result<Box<dyn Flow>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORKLOAD: &str = "workload: { trace_files: [t.csv], input_file_format: \
        text-generation-independent, arrival_mode: trace_timed, duration_ms: 1000.0, run_to_end: true, request_rate: 1.0, \
        input_file_tags: [TAGS] }\n";
    const IO: &str = "io: { log_dir: logs, log_level: info, quiet: true, \
        force_cache_build: false, log_output_token_times: false }\n";
    const ARCH: &str = "{type: llama3_dense_tp, model_config: m.json, fp8: false, tp_size: 1}";
    const SPECULATIVE: &str = "{type: speculative, attn_gpu_memory_gb: 80.0, \
        max_batch_tokens: 8192, draft_tokens: 3}";

    fn config(tags: &str, pools: &str) -> RunConfig {
        let yaml = format!("{}{IO}{pools}", WORKLOAD.replace("TAGS", tags));
        serde_yaml::from_str(&yaml).expect("the config parses")
    }

    fn pool(role: &str, worker: &str) -> String {
        format!(
            "  {role}:\n    placement: least-queued\n    groups:\n      - {{ gpu: H200, replicas: 1, arch: {ARCH}, \
             worker: {worker} }}\n"
        )
    }

    #[test]
    fn a_speculative_worker_in_any_pool_refuses_a_trace_without_acceptance() {
        let unified = format!("deployment: unified\npools:\n{}", pool("main", SPECULATIVE));
        let pd = format!(
            "deployment: pd\npools:\n{}{}",
            pool("prefill", "{type: pd_prefill, attn_gpu_memory_gb: 80.0}"),
            pool("decode", SPECULATIVE),
        );
        for (pools, role) in [(&unified, "main"), (&pd, "decode")] {
            let error = ensure_speculative_trace(&config("", pools))
                .unwrap_err()
                .to_string();
            assert!(error.contains(&format!("pool {role}")), "{error}");
            assert!(error.contains("accept_rate"), "{error}");
            ensure_speculative_trace(&config("speculative", pools))
                .expect("a speculative trace carries acceptance");
        }
        // Other workers read no acceptance, tagged or not.
        let barebone = format!(
            "deployment: unified\npools:\n{}",
            pool("main", "{type: barebone, attn_gpu_memory_gb: 80.0}")
        );
        ensure_speculative_trace(&config("", &barebone)).expect("not speculative");
    }
}
