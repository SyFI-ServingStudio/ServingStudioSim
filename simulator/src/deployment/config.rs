//! Structured run config (L7 assembly) — new-interface-design §6.
//!
//! These are the top-level types a concrete config file deserializes into, and
//! the run-global specs they embed. They live with the deployment layer because
//! their whole job is to compose the lower layers: `RunConfig` dispatches on the
//! `deployment` tag, each per-deployment config fixes which pool *roles* exist
//! and instantiates [`PoolSpec`](crate::orchestrator::PoolSpec) with that role's
//! contract-class arch (L4) + worker (L5) selectors.
//!
//! The launcher expands sweeps + fills defaults, then writes ONE fully-concrete
//! config file per run; the Rust binary only reads that concrete config. So
//! these types carry NO `#[serde(default)]` for valued params (defaults live in
//! each layer's launcher schema, filled by the launcher) — every such field is
//! required. Genuinely-omittable fields (`num_layers`, `sim_num_layers`) are
//! `Option` with `#[serde(default)]` (absence = None).
//!
//! Shape: `deployment` tag → per-deployment config → `pools: { <role>: pool }` →
//! `groups: [...]` → each group has `gpu`/`replicas` + an arch + a worker tagged
//! enum. `model_config` + dims live inside the arch tag (§4); only `workload` /
//! `io` are run-global.

use serde::Deserialize;
use std::path::PathBuf;

use schema_derive::ParamStruct;

use crate::arch::{AttnArchSel, FfnArchSel, IterArchSel};
use crate::orchestrator::PoolSpec;
use crate::worker::{AttnWorkerSel, FfnWorkerSel, IterWorkerSel};

/// Log verbosity. Closed set → serde enum (lowercase matches the wire spelling).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Run-global workload / trace inputs.
#[derive(Debug, Clone, Deserialize, ParamStruct)]
#[serde(deny_unknown_fields)]
pub struct WorkloadSpec {
    /// Trace CSV files to simulate (runs sequentially).
    pub trace_files: Vec<PathBuf>,
    /// Simulation duration (ms); minimum window when run_to_end is set.
    #[param(default = 5000.0)]
    pub duration_ms: f64,
    /// Keep ticking past duration_ms until every request completes.
    pub run_to_end: bool,
    /// Request arrival rate (requests/s).
    #[param(default = 10.0)]
    pub request_rate: f64,
    /// Fixed simulation tick step (µs) — the time quantum the loop advances by
    /// each iteration. Finer ticks mean less TTFT/TPOT quantization (and smaller
    /// inter-slice gaps in the trace) at ~no throughput cost, since per-tick work
    /// is O(events), not O(ticks). Omit to keep the 100 µs default; `TickCfg`
    /// clamps 0 → 1 µs so this can never stall the loop.
    #[serde(default = "default_tick_dt_us")]
    #[param(default = 100)]
    pub tick_dt_us: u64,
}

/// serde fallback for `tick_dt_us`: 100 µs matches the historical hardcoded tick,
/// so presets/fixtures that omit the field keep their prior behavior. The
/// launcher still emits the schema default explicitly (see `#[param]` above).
fn default_tick_dt_us() -> u64 {
    100
}

/// Run-global output location + logging controls.
#[derive(Debug, Clone, Deserialize, ParamStruct)]
#[serde(deny_unknown_fields)]
pub struct IoSpec {
    /// Directory for this run's logs / parquet outputs.
    #[param(default = "logs")]
    pub log_dir: PathBuf,
    /// Log verbosity.
    #[param(string, default = "info", choices = LOG_LEVEL_CHOICES)]
    pub log_level: LogLevel,
    /// Suppress per-tick progress chatter on stdout.
    pub quiet: bool,
    /// Force-refresh perf_api rows while building startup caches.
    pub force_cache_build: bool,
    /// Record + persist the full per-token `output_token_times` array on each
    /// `request_slo` row (enables the analyzer's `slo-detailed` ITL metric). OFF
    /// by default: building that per-token array on the sim hot path is the
    /// single largest cost on high-throughput runs (a scattered write per token
    /// per request), and the per-request scalars feeding `slo-general` (ttft,
    /// tpot mean, last_token_time → E2E) are derived without it. When off, the
    /// worker neither allocates nor appends the array. Turn on only when
    /// per-token granularity (ITL) is actually needed.
    pub log_output_token_times: bool,
}

/// Top-level config, dispatched on the `deployment` tag.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "deployment", rename_all = "snake_case")]
pub enum RunConfig {
    Unified(UnifiedConfig),
    Pd(PdConfig),
    Afd(AfdConfig),
}

impl RunConfig {
    pub fn workload(&self) -> &WorkloadSpec {
        match self {
            RunConfig::Unified(c) => &c.workload,
            RunConfig::Pd(c) => &c.workload,
            RunConfig::Afd(c) => &c.workload,
        }
    }

    pub fn io(&self) -> &IoSpec {
        match self {
            RunConfig::Unified(c) => &c.io,
            RunConfig::Pd(c) => &c.io,
            RunConfig::Afd(c) => &c.io,
        }
    }
}

// ── per-deployment configs + their fixed-role pool maps ─────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct UnifiedConfig {
    pub workload: WorkloadSpec,
    pub io: IoSpec,
    pub pools: UnifiedPools,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnifiedPools {
    pub main: PoolSpec<IterArchSel, IterWorkerSel>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PdConfig {
    pub workload: WorkloadSpec,
    pub io: IoSpec,
    pub pools: PdPools,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PdPools {
    pub prefill: PoolSpec<IterArchSel, IterWorkerSel>,
    pub decode: PoolSpec<IterArchSel, IterWorkerSel>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AfdConfig {
    pub workload: WorkloadSpec,
    pub io: IoSpec,
    pub pools: AfdPools,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AfdPools {
    pub attn: PoolSpec<AttnArchSel, AttnWorkerSel>,
    pub ffn: PoolSpec<FfnArchSel, FfnWorkerSel>,
}

// ── launcher schema choices (closed-set enums; the rest is field-derived) ────

const LOG_LEVEL_CHOICES: [&str; 5] = ["trace", "debug", "info", "warn", "error"];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::PlacementPolicy;

    // A complete unified config (launcher writes complete configs — no defaults
    // applied by serde). Exercises: deployment tag dispatch, pools→groups, the
    // arch tagged enum, and the flatten(ModelSpec)+internally-tagged interaction
    // (G8) — model_config sits flat alongside tp_size under `arch`.
    const UNIFIED_YAML: &str = r#"
deployment: unified
workload: { trace_files: ["trace/smoke.csv"], duration_ms: 5000.0, run_to_end: true, request_rate: 10.0 }
io: { log_dir: "logs/smoke", log_level: info, quiet: false, force_cache_build: false, log_output_token_times: false }
pools:
  main:
    placement: least-queued
    groups:
      - gpu: "NVIDIA H200"
        replicas: 1
        arch: { type: llama3_dense_tp, model_config: "model/config/llama3_8b.json", fp8: false, tp_size: 4 }
        worker: { type: barebone, attn_gpu_memory_gb: 80.0 }
"#;

    #[test]
    fn unified_yaml_roundtrip() {
        let cfg: RunConfig = serde_yaml::from_str(UNIFIED_YAML).expect("parse unified yaml");
        let RunConfig::Unified(u) = &cfg else {
            panic!("expected unified");
        };
        assert_eq!(u.pools.main.placement, PlacementPolicy::LeastQueued);
        assert_eq!(u.pools.main.groups.len(), 1);
        let g = &u.pools.main.groups[0];
        assert_eq!(g.gpu, "NVIDIA H200");
        assert_eq!(g.replicas, 1);
        // G8: flatten(ModelSpec) + internally-tagged enum — both model_config and
        // tp_size resolved under the same `arch` map.
        match &g.arch {
            IterArchSel::Llama3DenseTp { model, tp_size } => {
                assert_eq!(model.model_config, "model/config/llama3_8b.json");
                assert!(!model.fp8);
                assert_eq!(*tp_size, 4);
            }
            other => panic!("expected llama3_dense_tp, got {other:?}"),
        }
        assert!(matches!(
            g.worker,
            IterWorkerSel::Barebone { attn_gpu_memory_gb } if attn_gpu_memory_gb == 80.0
        ));
        assert_eq!(cfg.io().log_level, LogLevel::Info);
        assert_eq!(cfg.workload().request_rate, 10.0);
    }

    #[test]
    fn json_parses_same_as_yaml() {
        // YAML is a JSON superset; the equivalent JSON must parse identically.
        let json = serde_json::json!({
            "deployment": "unified",
            "workload": {"trace_files": ["t.csv"], "duration_ms": 5000.0, "run_to_end": true, "request_rate": 10.0},
            "io": {"log_dir": "logs", "log_level": "info", "quiet": false, "force_cache_build": false, "log_output_token_times": false},
            "pools": {"main": {"placement": "least-queued", "groups": [
                {"gpu": "H200", "replicas": 1,
                 "arch": {"type": "llama3_dense", "model_config": "m.json", "fp8": false},
                 "worker": {"type": "barebone", "attn_gpu_memory_gb": 80.0}}
            ]}}
        });
        let cfg: RunConfig = serde_json::from_value(json).expect("parse unified json");
        let RunConfig::Unified(u) = &cfg else {
            panic!("expected unified");
        };
        assert!(matches!(
            u.pools.main.groups[0].arch,
            IterArchSel::Llama3Dense { .. }
        ));
    }

    #[test]
    fn num_layers_omittable() {
        // num_layers / sim_num_layers absent → None (genuine optionality).
        let cfg: RunConfig = serde_yaml::from_str(UNIFIED_YAML).unwrap();
        let RunConfig::Unified(u) = &cfg else { unreachable!() };
        assert!(u.pools.main.groups[0].arch.model().num_layers.is_none());
    }

    #[test]
    fn bad_arch_tag_rejected() {
        let bad = UNIFIED_YAML.replace("llama3_dense_tp", "no_such_arch");
        assert!(serde_yaml::from_str::<RunConfig>(&bad).is_err());
    }

    #[test]
    fn unknown_group_field_rejected() {
        // deny_unknown_fields on GroupSpec (a plain struct) catches stray keys.
        let bad = UNIFIED_YAML.replace("replicas: 1", "replicas: 1\n        bogus: 3");
        assert!(serde_yaml::from_str::<RunConfig>(&bad).is_err());
    }

    #[test]
    fn pd_two_pools_parse() {
        let yaml = r#"
deployment: pd
workload: { trace_files: ["t.csv"], duration_ms: 5000.0, run_to_end: false, request_rate: 10.0 }
io: { log_dir: "logs", log_level: info, quiet: false, force_cache_build: false, log_output_token_times: false }
pools:
  prefill:
    placement: least-queued
    groups:
      - { gpu: "H200", replicas: 2, arch: {type: llama3_dense_tp, model_config: "m.json", fp8: false, tp_size: 8}, worker: {type: barebone, attn_gpu_memory_gb: 80.0} }
  decode:
    placement: least-queued
    groups:
      - { gpu: "H200", replicas: 4, arch: {type: llama3_dense_tp, model_config: "m.json", fp8: false, tp_size: 2}, worker: {type: barebone, attn_gpu_memory_gb: 80.0} }
"#;
        let cfg: RunConfig = serde_yaml::from_str(yaml).expect("parse pd");
        let RunConfig::Pd(p) = &cfg else { panic!("expected pd") };
        assert_eq!(p.pools.prefill.groups[0].replicas, 2);
        assert_eq!(p.pools.decode.groups[0].replicas, 4);
    }

    #[test]
    fn afd_two_pools_parse() {
        // AFD config: an attn pool (DP shards, qwen3_attn) + an ffn pool (one
        // aggregated replica, qwen3_ffn_moe). Mirrors `pd_two_pools_parse`.
        let yaml = r#"
deployment: afd
workload: { trace_files: ["t.csv"], duration_ms: 5000.0, run_to_end: false, request_rate: 10.0 }
io: { log_dir: "logs", log_level: info, quiet: false, force_cache_build: false, log_output_token_times: false }
pools:
  attn:
    placement: least-queued
    groups:
      - { gpu: "H200", replicas: 8, arch: {type: qwen3_attn_tp, model_config: "m.json", fp8: false, attn_tp_size: 4}, worker: {type: disagg_attn, attn_gpu_memory_gb: 80.0} }
  ffn:
    placement: least-queued
    groups:
      - { gpu: "H200", replicas: 1, arch: {type: qwen3_ffn_moe, model_config: "m.json", fp8: false, attn_tp_size: 4, ep_size: 8, nvl_num_gpu: 8}, worker: {type: disagg_ffn} }
"#;
        let cfg: RunConfig = serde_yaml::from_str(yaml).expect("parse afd");
        let RunConfig::Afd(a) = &cfg else { panic!("expected afd") };
        assert_eq!(a.pools.attn.groups[0].replicas, 8);
        assert_eq!(a.pools.ffn.groups[0].replicas, 1);
        match &a.pools.attn.groups[0].arch {
            AttnArchSel::Qwen3AttnTp { attn_tp_size, .. } => assert_eq!(*attn_tp_size, 4),
            other => panic!("expected qwen3_attn_tp, got {other:?}"),
        }
        match &a.pools.ffn.groups[0].arch {
            FfnArchSel::Qwen3FfnMoe {
                attn_tp_size,
                ep_size,
                ..
            } => {
                assert_eq!(*attn_tp_size, 4);
                assert_eq!(*ep_size, 8);
            }
            other => panic!("expected qwen3_ffn_moe, got {other:?}"),
        }
        assert!(matches!(
            a.pools.ffn.groups[0].worker,
            FfnWorkerSel::DisaggFfn {}
        ));
    }

    #[test]
    fn dp_attn_tp_ffn_with_hp_unified_parses() {
        // DP-attention arch carries two TP degrees; pairs with the hp_unified worker.
        let yaml = r#"
deployment: unified
workload: { trace_files: ["t.csv"], duration_ms: 5000.0, run_to_end: true, request_rate: 10.0 }
io: { log_dir: "logs", log_level: info, quiet: false, force_cache_build: false, log_output_token_times: false }
pools:
  main:
    placement: least-queued
    groups:
      - gpu: "NVIDIA H200"
        replicas: 1
        arch: { type: llama3_dp_attn_tp_ffn, model_config: "m.json", fp8: false, attn_tp_size: 4, ffn_tp_size: 8 }
        worker: { type: hp_unified, attn_gpu_memory_gb: 80.0 }
"#;
        let cfg: RunConfig = serde_yaml::from_str(yaml).expect("parse dp-attn unified");
        let RunConfig::Unified(u) = &cfg else {
            panic!("expected unified");
        };
        let g = &u.pools.main.groups[0];
        match &g.arch {
            IterArchSel::Llama3DpAttnTpFfn {
                model,
                attn_tp_size,
                ffn_tp_size,
            } => {
                assert_eq!(model.model_config, "m.json");
                assert_eq!(*attn_tp_size, 4);
                assert_eq!(*ffn_tp_size, 8);
            }
            other => panic!("expected llama3_dp_attn_tp_ffn, got {other:?}"),
        }
        assert!(matches!(
            g.worker,
            IterWorkerSel::HpUnified { attn_gpu_memory_gb } if attn_gpu_memory_gb == 80.0
        ));
    }
}
