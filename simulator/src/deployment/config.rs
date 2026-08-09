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
use std::collections::HashMap;
use std::path::PathBuf;

use schema_derive::ParamStruct;

use crate::arch::{AttnArchSel, FfnArchSel, IterArchSel};
use crate::orchestrator::PoolSpec;
use crate::worker::{AttnWorkerSel, FfnWorkerSel, IterWorkerSel};

/// User-configurable per-kernel backend overrides, `pool → role → backends`.
///
/// Outer key = pool name (`main` / `attn` / `ffn` / `prefill` / `decode`); inner
/// key = a kernel's dotted role `name` (pool prefix stripped, e.g.
/// `afd.moe_expert_compute.gate_up`); value = the candidate backend list that
/// replaces that kernel's arch const-default (best-of-N still picks the fastest).
/// The launcher fills this from the preset's `backends_file` (a normal sweep
/// variable), and each deployment's `build` scopes one pool's submap onto the
/// bridge via [`PerfApiBridge::with_backend_overrides`] while that pool's model
/// builds. Absent (the default) = every kernel keeps its arch-declared backends.
///
/// [`PerfApiBridge::with_backend_overrides`]: crate::timing::PerfApiBridge::with_backend_overrides
pub type BackendOverrides = HashMap<String, HashMap<String, Vec<String>>>;

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
    /// What every file in `trace_files` is. Declared, never inferred from the
    /// header: the frontend computes the exact column set this implies and
    /// rejects any header that differs. One family per run; a native omni model
    /// uses `omni_generation`, whose rows may themselves contain mixed ordered
    /// modality segments.
    #[param(choices = simulator::sim::TraceKind::CHOICES)]
    pub trace_kind: String,
    /// Cross-cutting disciplines those files also carry (`session`, `slo`,
    /// `speculative`). Each adds its own required columns to the expected set.
    #[serde(default)]
    #[param(choices = simulator::sim::TraceTag::CHOICES)]
    pub trace_tags: Vec<String>,
    /// Simulation duration (ms); minimum window when run_to_end is set.
    #[param(default = 5000.0)]
    pub duration_ms: f64,
    /// Keep ticking past duration_ms until every request completes.
    pub run_to_end: bool,
    /// Request arrival rate (requests/s). Read by `open_loop`; ignored by
    /// `closed_loop`.
    #[param(default = 10.0)]
    pub request_rate: f64,
    /// Closed-loop concurrency cap. When set, the frontend IGNORES CSV
    /// arrival_time / request_rate and instead keeps at most this many requests
    /// in flight, admitting a new one the instant a slot frees — mirroring the
    /// alignment load-generator's --max-concurrency (a tokio Semaphore of N
    /// permits acquired *after* arrival, held until completion). Required by
    /// `closed_loop`; must be absent for `open_loop`.
    #[serde(default)]
    pub max_concurrency: Option<u32>,
    /// How workload pressure releases eligible requests. Session causality is
    /// selected independently by `session_dependency`.
    #[param(choices = simulator::sim::ReplayPacing::CHOICES)]
    pub replay_pacing: String,
    /// Whether every trace row is independently eligible or later rounds wait
    /// for predecessor completion plus `tool_wait_after_ms`. `chained` requires
    /// the `session` trace tag and composes with either replay pacing.
    #[param(choices = simulator::sim::SessionDependency::CHOICES)]
    pub session_dependency: String,
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

fn default_kv_log_stride() -> u32 {
    8
}

/// Stage timelines are part of the normal request-level diagnostics. Keep the
/// serde fallback aligned with the launcher's emitted default.
fn default_log_stage_transitions() -> bool {
    true
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
    /// Record + persist the per-request stage/location transition timeline on
    /// each `request_slo` row (the stage list columns): every time a request
    /// moves between queues/workers, one `(time, code, pool_id, worker_id)`
    /// event is appended. ON by default; disable it explicitly when the timeline
    /// allocation/append cost is not wanted. Codes decode
    /// to `"category:detail"` names via the per-deployment table in `run_meta.json`.
    /// Genuinely omittable (absent = true), so it carries `serde`/`param`
    /// defaults rather than the required-field treatment.
    #[serde(default = "default_log_stage_transitions")]
    #[param(default = true)]
    pub log_stage_transitions: bool,
    /// The per-worker `KvSampler` emits one `kv_snapshot` occupancy row every
    /// `kv_log_stride` per-iteration submits (running-max throttle). Genuinely
    /// omittable — absent = 8 — so, like `tick_dt_us`, it carries a `serde` default
    /// (Rust-side) plus a `param` default (launcher schema), rather than the
    /// required-field treatment the other IO params get.
    #[serde(default = "default_kv_log_stride")]
    #[param(default = 8)]
    pub kv_log_stride: u32,
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

    /// The per-deployment stage vocabulary (`code → "category:detail"` names) written
    /// into `run_meta.json` so the analyzer can decode the `request_slo`
    /// stage-transition codes. Barebone and HP unified share `UnifiedStage`.
    pub fn stage_vocab(&self) -> crate::common::StageVocab {
        match self {
            RunConfig::Unified(_) => crate::common::UnifiedStage::VOCAB,
            RunConfig::Pd(_) => crate::common::PdStage::VOCAB,
            RunConfig::Afd(_) => crate::common::AfdStage::VOCAB,
        }
    }
}

// ── per-deployment configs + their fixed-role pool maps ─────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct UnifiedConfig {
    pub workload: WorkloadSpec,
    pub io: IoSpec,
    pub pools: UnifiedPools,
    /// Per-kernel backend overrides (see [`BackendOverrides`]). Genuinely
    /// optional — absent = no overrides — so it carries a serde default rather
    /// than the required-field treatment the valued params get.
    #[serde(default)]
    pub backends: BackendOverrides,
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
    /// Per-kernel backend overrides (see [`BackendOverrides`]); absent = none.
    #[serde(default)]
    pub backends: BackendOverrides,
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
    /// Per-kernel backend overrides (see [`BackendOverrides`]); absent = none.
    #[serde(default)]
    pub backends: BackendOverrides,
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
workload: { trace_files: ["trace/smoke.csv"], trace_kind: text_generation, replay_pacing: open_loop, session_dependency: independent, duration_ms: 5000.0, run_to_end: true, request_rate: 10.0 }
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
            IterWorkerSel::Barebone { attn_gpu_memory_gb, .. } if attn_gpu_memory_gb == 80.0
        ));
        assert_eq!(cfg.io().log_level, LogLevel::Info);
        assert!(cfg.io().log_stage_transitions);
        assert_eq!(cfg.workload().request_rate, 10.0);
        assert_eq!(cfg.workload().replay_pacing, "open_loop");
        assert_eq!(cfg.workload().session_dependency, "independent");
    }

    #[test]
    fn legacy_replay_mode_is_not_accepted() {
        let legacy = UNIFIED_YAML
            .replace("replay_pacing: open_loop", "replay_mode: open_loop")
            .replace(", session_dependency: independent", "");
        let error = serde_yaml::from_str::<RunConfig>(&legacy).unwrap_err();
        assert!(error.to_string().contains("replay_mode"), "{error}");
    }

    #[test]
    fn json_parses_same_as_yaml() {
        // YAML is a JSON superset; the equivalent JSON must parse identically.
        let json = serde_json::json!({
            "deployment": "unified",
            "workload": {"trace_files": ["t.csv"], "trace_kind": "text_generation", "replay_pacing": "open_loop", "session_dependency": "independent", "duration_ms": 5000.0, "run_to_end": true, "request_rate": 10.0},
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
    fn backends_overrides_parse_and_default_empty() {
        // Absent `backends:` → empty map (serde default), i.e. no overrides.
        let cfg: RunConfig = serde_yaml::from_str(UNIFIED_YAML).unwrap();
        let RunConfig::Unified(u) = &cfg else {
            unreachable!()
        };
        assert!(u.backends.is_empty());

        // Present → `pool → role → candidate backends`, the shape the launcher
        // writes from a preset's `backends_file`. Keys are pool-prefix-stripped
        // dotted role names; values are the best-of-N candidate lists.
        let with = format!(
            "{UNIFIED_YAML}\nbackends:\n  \
             main:\n    \
             \"unified.attn.qkv\": [fa2, fa3]\n    \
             \"unified.mlp.down\": [torch]\n"
        );
        let cfg: RunConfig = serde_yaml::from_str(&with).expect("parse backends block");
        let RunConfig::Unified(u) = &cfg else {
            unreachable!()
        };
        assert_eq!(u.backends["main"]["unified.attn.qkv"], vec!["fa2", "fa3"]);
        assert_eq!(u.backends["main"]["unified.mlp.down"], vec!["torch"]);
    }

    #[test]
    fn num_layers_omittable() {
        // num_layers / sim_num_layers absent → None (genuine optionality).
        let cfg: RunConfig = serde_yaml::from_str(UNIFIED_YAML).unwrap();
        let RunConfig::Unified(u) = &cfg else {
            unreachable!()
        };
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
workload: { trace_files: ["t.csv"], trace_kind: text_generation, replay_pacing: open_loop, session_dependency: independent, duration_ms: 5000.0, run_to_end: false, request_rate: 10.0 }
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
        let RunConfig::Pd(p) = &cfg else {
            panic!("expected pd")
        };
        assert_eq!(p.pools.prefill.groups[0].replicas, 2);
        assert_eq!(p.pools.decode.groups[0].replicas, 4);
    }

    #[test]
    fn afd_two_pools_parse() {
        // AFD config: an attn pool (DP shards, qwen3_attn) + an ffn pool (one
        // aggregated replica, qwen3_ffn_moe). Mirrors `pd_two_pools_parse`.
        let yaml = r#"
deployment: afd
workload: { trace_files: ["t.csv"], trace_kind: text_generation, replay_pacing: open_loop, session_dependency: independent, duration_ms: 5000.0, run_to_end: false, request_rate: 10.0 }
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
        let RunConfig::Afd(a) = &cfg else {
            panic!("expected afd")
        };
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
            FfnWorkerSel::DisaggFfn { .. }
        ));
    }

    #[test]
    fn dp_attn_tp_ffn_with_hp_unified_parses() {
        // DP-attention arch carries two TP degrees; pairs with the hp_unified worker.
        let yaml = r#"
deployment: unified
workload: { trace_files: ["t.csv"], trace_kind: text_generation, replay_pacing: open_loop, session_dependency: independent, duration_ms: 5000.0, run_to_end: true, request_rate: 10.0 }
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
            IterWorkerSel::HpUnified { attn_gpu_memory_gb, .. } if attn_gpu_memory_gb == 80.0
        ));
    }
}
