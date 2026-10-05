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

use anyhow::{bail, ensure, Context};

use crate::arch::IterArchSel;
use crate::common::{AcceptanceProfile, DecodingStrategy, SharedRequests, TooLong};
use crate::orchestrator::{Flow, PoolSpec};
use crate::sim::TraceFrontend;
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

/// One pool group's bound on the requests it serves: the pool's role, its
/// arch's longest request, and a speculative worker's draft width (`None` for
/// any other worker). `simulator dry-run --report-json` writes these under
/// `pools`, so a service reads the bounds a run enforces instead of
/// re-deriving them from the preset.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PoolBound {
    pub role: &'static str,
    pub max_model_len: u32,
    pub draft_tokens: Option<u32>,
}

/// Every request-serving pool group's [`PoolBound`], in pool order.
pub fn pool_bounds(cfg: &RunConfig) -> anyhow::Result<Vec<PoolBound>> {
    fn iter_pool(
        role: &'static str,
        pool: &PoolSpec<IterArchSel, IterWorkerSel>,
        out: &mut Vec<PoolBound>,
    ) -> anyhow::Result<()> {
        for group in &pool.groups {
            out.push(PoolBound {
                role,
                max_model_len: group
                    .arch
                    .max_model_len()
                    .with_context(|| format!("pool {role}"))?,
                draft_tokens: match group.worker {
                    IterWorkerSel::Speculative { draft_tokens, .. } => Some(draft_tokens),
                    _ => None,
                },
            });
        }
        Ok(())
    }
    let mut out = Vec::new();
    match cfg {
        RunConfig::Unified(c) => iter_pool("main", &c.pools.main, &mut out)?,
        RunConfig::Pd(c) => {
            iter_pool("prefill", &c.pools.prefill, &mut out)?;
            iter_pool("decode", &c.pools.decode, &mut out)?;
        }
        // The ffn pool holds no request context.
        RunConfig::Afd(c) => {
            for group in &c.pools.attn.groups {
                out.push(PoolBound {
                    role: "attn",
                    max_model_len: group.arch.max_model_len().context("pool attn")?,
                    draft_tokens: None,
                });
            }
        }
    }
    Ok(out)
}

/// Refuse a trace a pool cannot serve, before anything runs, rather than fail
/// mid-run on its first such request:
///
/// - every request's whole context (its declared prefix, its input and its
///   output; a speculative worker's last verify reads `draft_tokens` beyond
///   that) must fit each pool's [`IterArchSel::max_model_len`];
/// - a per-position `accept_rate` must have one probability per draft
///   position of each speculative worker (a single probability fits any).
///
/// The message names the pool, its bound, how many requests break it and the
/// first of their ids. `simulator run` checks before its tick loop, and
/// `simulator workload-plan --config` with the same function, so a service can
/// refuse the trace before queueing a run.
pub fn check_trace(cfg: &RunConfig, trace: &TraceFrontend) -> anyhow::Result<()> {
    const SHOWN: usize = 5;
    let total = trace.expected_count();
    for pool in pool_bounds(cfg)? {
        let draft = pool.draft_tokens.unwrap_or(0);
        let (mut long, mut long_ids) = (0usize, Vec::new());
        let (mut misfit, mut misfit_ids) = (0usize, Vec::new());
        for (id, request) in trace.source_requests() {
            let context = u64::from(request.session.declared_prefix_tokens())
                + u64::from(request.prompt_tokens)
                + u64::from(request.target_output_tokens)
                + u64::from(draft);
            if context > u64::from(pool.max_model_len) {
                long += 1;
                if long_ids.len() < SHOWN {
                    long_ids.push(id.clone());
                }
            }
            if let (
                Some(draft_tokens),
                DecodingStrategy::Speculative {
                    accept_rate: AcceptanceProfile::ByPosition(rates),
                },
            ) = (pool.draft_tokens, &request.decoding)
            {
                if rates.len() != draft_tokens as usize {
                    misfit += 1;
                    if misfit_ids.len() < SHOWN {
                        misfit_ids.push(id);
                    }
                }
            }
        }
        let role = pool.role;
        if long > 0 {
            let fit = if draft > 0 {
                format!("prefix_len + input_len + output_len + {draft} draft tokens")
            } else {
                "prefix_len + input_len + output_len".to_string()
            };
            return Err(TooLong {
                max_model_len: pool.max_model_len,
                requests: Some(long),
                total: Some(total),
                message: format!(
                    "{long} of {total} requests exceed pool {role}'s max_model_len {} ({fit}), \
                     first ids {long_ids:?}",
                    pool.max_model_len
                ),
            }
            .into());
        }
        if misfit > 0 {
            bail!(
                "{misfit} of {total} requests give pool {role}'s speculative worker, which \
                 drafts {draft} tokens, an accept_rate vector that is not {draft} \
                 probabilities long (one per draft position), first ids {misfit_ids:?}"
            );
        }
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

    /// A Llama 3 8B pool (`max_position_embeddings` 131072) with `worker`.
    fn llama_config(tags: &str, worker: &str) -> RunConfig {
        let model = concat!(env!("CARGO_MANIFEST_DIR"), "/model/config/llama3_8b.json");
        let pools = format!("deployment: unified\npools:\n{}", pool("main", worker))
            .replace("model_config: m.json", &format!("model_config: {model}"));
        config(tags, &pools)
    }

    fn trace(dir: &std::path::Path, tags: &[&str], rows: &str) -> TraceFrontend {
        use crate::sim::{
            ArrivalSchedule, CapacityLimit, InputFileFormat, InputFileSchema, TraceTag,
        };
        let accept = if tags.contains(&"speculative") {
            ",accept_rate"
        } else {
            ""
        };
        let path = dir.join("trace.csv");
        std::fs::write(
            &path,
            format!("id,input_len,output_len,arrival_time{accept}\n{rows}"),
        )
        .unwrap();
        let schema = InputFileSchema::new(
            InputFileFormat::parse("text-generation-independent").unwrap(),
            tags.iter()
                .map(|tag| TraceTag::parse(tag).unwrap())
                .collect(),
        )
        .unwrap();
        TraceFrontend::load(
            &[path],
            &schema,
            ArrivalSchedule::parse("trace_timed", 1.0).unwrap(),
            CapacityLimit::parse(None).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn each_pool_group_reports_its_bound_as_the_dry_run_writes_it() {
        let bounds = |worker| serde_json::to_value(pool_bounds(&llama_config("", worker)).unwrap());
        assert_eq!(
            bounds(SPECULATIVE).unwrap(),
            serde_json::json!([{"role": "main", "max_model_len": 131072, "draft_tokens": 3}])
        );
        assert_eq!(
            bounds("{type: barebone, attn_gpu_memory_gb: 80.0}").unwrap(),
            serde_json::json!([{"role": "main", "max_model_len": 131072, "draft_tokens": null}])
        );
    }

    #[test]
    fn a_request_longer_than_the_pools_max_model_len_is_refused_before_the_run() {
        let dir = tempfile::tempdir().unwrap();
        let barebone = llama_config("", "{type: barebone, attn_gpu_memory_gb: 80.0}");
        let fits = trace(dir.path(), &[], "a,130872,200,0.0\n");
        check_trace(&barebone, &fits).expect("131072 tokens is the limit itself");

        let long = trace(
            dir.path(),
            &[],
            "a,100,10,0.0\nb,131000,200,0.0\nc,131000,73,0.0\nd,131000,72,0.0\n",
        );
        let error = check_trace(&barebone, &long).unwrap_err();
        assert_eq!(
            error.to_string(),
            "2 of 4 requests exceed pool main's max_model_len 131072 \
             (prefix_len + input_len + output_len), first ids [\"b\", \"c\"]"
        );
        let too_long = error.downcast_ref::<TooLong>().expect("a TooLong refusal");
        assert_eq!(
            (too_long.max_model_len, too_long.requests, too_long.total),
            (131072, Some(2), Some(4))
        );
    }

    #[test]
    fn a_speculative_worker_counts_its_draft_tokens_and_its_acceptance_width() {
        let dir = tempfile::tempdir().unwrap();
        let speculative = llama_config("speculative", SPECULATIVE);
        // 131069 + 3 draft tokens fits; one more does not.
        let fits = trace(dir.path(), &["speculative"], "a,131000,69,0.0,0.7\n");
        check_trace(&speculative, &fits).expect("fits with its draft tokens");
        let long = trace(dir.path(), &["speculative"], "a,131000,70,0.0,0.7\n");
        let error = check_trace(&speculative, &long).unwrap_err().to_string();
        assert!(
            error.contains(
                "max_model_len 131072 (prefix_len + input_len + output_len + 3 draft tokens)"
            ),
            "{error}"
        );

        // A per-position vector needs one probability per draft position; a
        // single probability fits any width.
        let rows = "a,8,4,0.0,0.7\nb,8,4,0.0,\"[0.9,0.8,0.7]\"\nc,8,4,0.0,\"[0.9,0.8]\"\n";
        let error = check_trace(&speculative, &trace(dir.path(), &["speculative"], rows))
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            "1 of 3 requests give pool main's speculative worker, which drafts 3 tokens, \
             an accept_rate vector that is not 3 probabilities long (one per draft \
             position), first ids [\"c\"]"
        );
        let rows = "a,8,4,0.0,0.7\nb,8,4,0.0,\"[0.9,0.8,0.7]\"\n";
        check_trace(&speculative, &trace(dir.path(), &["speculative"], rows))
            .expect("every vector is three long");
    }
}
