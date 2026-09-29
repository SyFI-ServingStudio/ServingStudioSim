//! `timing-predict` — offline per-building-block timing prediction.
//!
//! This is the same family as `dry-run` / `build-cache-only` / `kernel-query`:
//! it does NOT run the discrete-event sim (no scheduler, no trace, no clock). It
//! takes a batch of explicit batch shapes and, for each, runs the cost model over
//! the compiled `CostTree(s)`, predicting timings WITHOUT a GPU (a warm
//! `profile.db`) or JIT-profiling on demand (a cold one).
//!
//! Three arch kinds, one tool (the `arch` selector picks):
//!   - `iter` — ONE [`IterwiseUnifiedModel::eval_iter`] over the whole forward
//!     pass (embedding → all layers via the `Scale{n}` fold → lm_head). One row
//!     per case, `section = "iter"`, `layer = -1`.
//!   - `attn` — the AFD attn side ([`qwen3_attn`]): one `attn_cost` per case
//!     (`section = "attn"`). One DP shard = one group.
//!   - `ffn` — the AFD ffn side ([`qwen3_ffn_moe`]): the per-section building
//!     blocks of one iteration — `prologue` (embed), `pre_attn` (layer-0 qkv),
//!     a representative mid-layer `post_attn` (o_proj + router + MoE + fused
//!     next-layer qkv), the terminal `post_attn_last`, and `epilogue`
//!     (final_norm + lm_head). One row per section.
//!
//! **One arch, not a bundle.** AFD is predicted by running this tool TWICE — once
//! with the attn arch, once with the ffn arch — each independent. The cross-pool
//! attn↔ffn handoff is a `GpuCluster` transfer (not a cost-tree leaf) and is out
//! of scope here; the MoE EP dispatch/combine comm, by contrast, IS an in-tree
//! compute leaf inside the ffn `post_attn` section and is logged automatically.
//!
//! The output is not a bespoke format: every case/section is emitted as one row of
//! the standard `raw/cost_log/worker_predict_0.parquet` + the matching
//! `raw/cost_manifest/worker_predict_0.json` — byte-for-byte the artifacts a real
//! `run` writes (one worker, `iter_id` = case index). All three paths reuse the
//! worker's [`CostBuffers`]: the iter path via [`CostBuffers::run_iter`], the
//! layer-wise attn/ffn paths via [`CostBuffers::run_section`], whose rows carry
//! the `section` + `layer` fields.
//!
//! Config (a minimal file, NOT a `RunConfig`): ONE arch selector + its GPU + an
//! optional per-role backend policy + a log dir + a batched cases file. No
//! workload / pools / io — those belong to `run`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, ensure, Context, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};

use crate::arch::build::{
    build_attn_model, build_ffn_model, build_iter_model, build_speculative_iter_model,
};
use crate::arch::contract::{
    ArchGroupInput, AttnArchInput, AttnLayerwiseModel, FfnArchInput, FfnLayerwiseModel,
    IterwiseUnifiedModel, SpeculativeArchGroupInput, SpeculativeArchInput, SpeculativeDecodeInput,
    SpeculativeUnifiedModel, UnifiedArchInput,
};
use crate::arch::{AttnArchSel, FfnArchSel, IterArchSel};
use crate::common::{Time, WorkerId};
use crate::deployment::BackendOverrides;
use crate::timing::bridge::{write_config_records, KernelData};
use crate::timing::SlotInput;
use crate::timing::{CostManifestDoc, CostTree, FlatCostNode, LeafMetrics, PerfApiBridge};
use crate::worker::cost_buffers::GroupLogSource;
use crate::worker::CostBuffers;

/// The AFD deployment's dotted-leaf prefix (see `deployment/afd.rs`). Predicting
/// the attn / ffn arches under the same name makes a predicted manifest's leaf
/// names match a real AFD run's, so the analyzer renders them identically.
const AFD_MODEL_NAME: &str = "afd";
/// The co-located unified run's leaf prefix, reused by the `iter` arch.
const UNIFIED_MODEL_NAME: &str = "unified";
/// All predict streams write under one writer/file tag; the building block is the
/// per-row `section` field, NOT the pool tag (which only names pool/worker identity).
const PREDICT_POOL_TAG: &str = "predict";
/// Narrow execution provenance consumed by the launcher when publishing the
/// first-class prediction resource. Timing-predict has no L5 worker allocation,
/// so it must not manufacture `run_meta.json`; L4 remains authoritative for the
/// physical GPU extent of the model replica being costed.
const PREDICTION_PROVENANCE_FILE: &str = "prediction_provenance.json";

#[derive(Serialize)]
struct PredictionProvenance<'a> {
    schema_version: u32,
    gpu_name: &'a str,
    gpu_count: u16,
}

/// Which arch to predict. The wire representation is the same single-key map in
/// JSON and YAML: `{ iter: {...} } | { attn: {...} } | { ffn: {...} }`, or the
/// arch block alone (`{type: ..., ...}`, as a run config's group carries it),
/// whose type decides the selector.
/// `serde_yaml` otherwise encodes an externally tagged enum as `!iter`, which
/// Python's safe YAML loader deliberately rejects. The explicit map adapter
/// keeps the launcher and simulator on one portable, tag-free document shape.
#[derive(Debug)]
enum PredictArchSel {
    /// A whole-iteration arch — costed as one fused `eval_iter`.
    Iter(IterArchSel),
    /// A draft/verify iteration with explicit per-request query widths.
    SpeculativeIter(IterArchSel),
    /// The AFD attn side — costed as one `attn_cost` per case.
    Attn(AttnArchSel),
    /// The AFD ffn side — costed as the per-section building blocks of one iteration.
    Ffn(FfnArchSel),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IterPredictArch {
    iter: IterArchSel,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeculativePredictArch {
    speculative_iter: IterArchSel,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AttnPredictArch {
    attn: AttnArchSel,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FfnPredictArch {
    ffn: FfnArchSel,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PredictArchWire {
    Iter(IterPredictArch),
    SpeculativeIter(SpeculativePredictArch),
    Attn(AttnPredictArch),
    Ffn(FfnPredictArch),
    Block(serde_json::Value),
}

impl PredictArchSel {
    /// The selector an arch block's type implies: its contract, and for an
    /// iter-wise arch whether it drafts. A type every contract rejects as
    /// unknown is reported as such; a known type with bad fields reports why.
    fn from_block(block: serde_json::Value) -> std::result::Result<Self, String> {
        let unknown = |e: &serde_json::Error| e.to_string().starts_with("unknown variant");
        let iter = match IterArchSel::deserialize(&block) {
            Ok(sel) if sel.is_speculative() => return Ok(Self::SpeculativeIter(sel)),
            Ok(sel) => return Ok(Self::Iter(sel)),
            Err(e) => e,
        };
        let attn = match AttnArchSel::deserialize(&block) {
            Ok(sel) => return Ok(Self::Attn(sel)),
            Err(e) => e,
        };
        let ffn = match FfnArchSel::deserialize(&block) {
            Ok(sel) => return Ok(Self::Ffn(sel)),
            Err(e) => e,
        };
        let known = [iter, attn, ffn].into_iter().find(|e| !unknown(e));
        Err(match known {
            Some(error) => format!("arch block: {error}"),
            None => format!("arch block: no contract knows type {}", block["type"]),
        })
    }
}

impl<'de> Deserialize<'de> for PredictArchSel {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match PredictArchWire::deserialize(deserializer)? {
            PredictArchWire::Iter(value) => Self::Iter(value.iter),
            PredictArchWire::SpeculativeIter(value) => {
                Self::SpeculativeIter(value.speculative_iter)
            }
            PredictArchWire::Attn(value) => Self::Attn(value.attn),
            PredictArchWire::Ffn(value) => Self::Ffn(value.ffn),
            PredictArchWire::Block(block) => {
                Self::from_block(block).map_err(serde::de::Error::custom)?
            }
        })
    }
}

/// Minimal offline config for `timing-predict`. `arch` is the generalized
/// [`PredictArchSel`]; the rest mirror the legacy iter config.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PredictConfig {
    arch: PredictArchSel,
    gpu: String,
    /// Same pool→role backend override contract as `RunConfig`. Iter prediction
    /// uses pool `main`; AFD attn/ffn prediction uses its matching pool name.
    /// Absent keeps the arch-declared defaults for standalone predict configs.
    #[serde(default)]
    backends: BackendOverrides,
    log_dir: PathBuf,
    /// Path to the batched cases JSON/YAML (a top-level array of [`PredictCase`]).
    /// Resolved relative to this config file's directory when not absolute.
    cases_file: PathBuf,
}

/// One predicted case: a list of attention-DP shards. The expected count depends
/// on the arch — `iter`/`ffn` want `num_attn_dp_groups`/`num_dp_groups` shards,
/// `attn` wants exactly one (one model instance = one DP shard).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PredictCase {
    groups: Vec<PredictGroup>,
}

/// One attention-DP shard's batch shape. Prefill is the exact per-request
/// `[prefix_len, append_len]` list (a fresh prefill has `prefix_len = 0`); decode
/// is EITHER the exact per-request KV-length list (`decode_kv_lens`) OR the
/// uniform shorthand (`decode_count` + `average_decode_length`) — never both.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PredictGroup {
    #[serde(default)]
    prefill_chunk_pairs: Vec<[u32; 2]>,
    #[serde(default)]
    decode_kv_lens: Vec<u32>,
    #[serde(default)]
    decode_count: Option<u32>,
    #[serde(default)]
    average_decode_length: Option<u32>,
}

impl PredictGroup {
    /// Lower to an [`ArchGroupInput`], deriving `batch_tokens` / `prefill_tokens`
    /// / `total_kv_len` exactly as the worker does in
    /// `worker/execution/unified_iter_execution.rs::build_input`, so a predicted iteration costs
    /// identically to the same shape inside a real run.
    fn into_arch_group(self) -> Result<ArchGroupInput> {
        // decode source: exact list xor uniform shorthand xor neither.
        let decode_kv_lens = match (self.decode_kv_lens.is_empty(), self.decode_count) {
            (false, None) => self.decode_kv_lens,
            (true, Some(count)) => {
                let avg = self
                    .average_decode_length
                    .context("decode_count requires average_decode_length")?;
                ensure!(avg > 0, "average_decode_length must be > 0");
                vec![avg; count as usize]
            }
            (false, Some(_)) => {
                bail!("group sets both decode_kv_lens and decode_count; use exactly one")
            }
            (true, None) => Vec::new(), // prefill-only group
        };

        let prefill_chunk_pairs: Vec<(u32, u32)> = self
            .prefill_chunk_pairs
            .iter()
            .map(|p| (p[0], p[1]))
            .collect();
        let prefill_tokens: u32 = prefill_chunk_pairs.iter().map(|(_, append)| *append).sum();
        let decode_tokens = decode_kv_lens.len() as u32;
        let total_kv: u64 = decode_kv_lens.iter().map(|&k| u64::from(k)).sum();

        Ok(ArchGroupInput {
            batch_tokens: prefill_tokens + decode_tokens,
            prefill_tokens,
            decode_tokens,
            prefill_chunk_pairs,
            decode_kv_lens,
            total_kv_len: total_kv as u32,
        })
    }
}

impl PredictCase {
    /// Validate the group count against the model's expected DP degree and lower
    /// every group to its [`ArchGroupInput`]. This is the attention-shaped lowering
    /// shared by the iter and attn drivers — both of whose input types wrap
    /// `Vec<ArchGroupInput>`. It is NOT a universal case→input step: each driver
    /// owns the construction of its own input type from these groups (iter wraps
    /// them in a [`UnifiedArchInput`], attn in an [`AttnArchInput`]), and the ffn
    /// side does not pass through here at all — its case is the lean [`FfnArchInput`]
    /// itself, whose token counts the ffn cost reads directly. We do not assume a
    /// future arch's input aligns with this `Vec<ArchGroupInput>` shape.
    fn into_groups(self, expected_groups: usize) -> Result<Vec<ArchGroupInput>> {
        ensure!(
            self.groups.len() == expected_groups,
            "case has {} group(s) but the model expects {}",
            self.groups.len(),
            expected_groups,
        );
        self.groups
            .into_iter()
            .map(PredictGroup::into_arch_group)
            .collect()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeculativePredictCase {
    groups: Vec<SpeculativePredictGroup>,
}

/// Decode pairs are [final verify-row KV length, query width]. Ordinary decode
/// shorthand is deliberately absent: it loses request/query cardinality.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeculativePredictGroup {
    #[serde(default)]
    prefill_chunk_pairs: Vec<[u32; 2]>,
    #[serde(default)]
    decode_requests: Vec<[u32; 2]>,
}

impl SpeculativePredictGroup {
    fn into_arch_group(
        self,
        query_width: u32,
        max_model_len: u32,
    ) -> Result<SpeculativeArchGroupInput> {
        let mut group = SpeculativeArchGroupInput::default();
        for [prefix, append] in self.prefill_chunk_pairs {
            ensure!(append > 0, "prefill append must be positive");
            ensure!(
                prefix
                    .checked_add(append)
                    .is_some_and(|length| length <= max_model_len),
                "prefill context exceeds max_model_len"
            );
            group.prefill_tokens = group
                .prefill_tokens
                .checked_add(append)
                .context("prefill token sum overflows u32")?;
            group.prefill_chunk_pairs.push((prefix, append));
        }
        for [kv_len, query_len] in self.decode_requests {
            ensure!(
                query_len == query_width,
                "query_len must equal draft_tokens + 1 ({query_width})"
            );
            ensure!(
                (query_width..=max_model_len).contains(&kv_len),
                "final verify KV length must be within query width..=max_model_len"
            );
            group.decode_tokens = group
                .decode_tokens
                .checked_add(query_len)
                .context("decode query sum overflows u32")?;
            group.total_kv_len = group
                .total_kv_len
                .checked_add(kv_len - query_len)
                .context("decode KV sum overflows u32")?;
            group
                .decode_requests
                .push(SpeculativeDecodeInput { kv_len, query_len });
        }
        group.batch_tokens = group
            .prefill_tokens
            .checked_add(group.decode_tokens)
            .context("batch token sum overflows u32")?;
        Ok(group)
    }
}

/// How `run_timing_predict` treats the cases once they are validated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictMode {
    /// Cost every case, JIT-profiling missing rows, and write the artifacts.
    Run,
    /// Build the model on the dry-run bridge and validate every case, then report
    /// the `profile.db` specs a real run would JIT. Evaluates nothing and writes
    /// nothing, so it needs no GPU.
    DryRun,
}

/// Entry point for `simulator timing-predict <config>` — the generalized tool.
/// Dispatches on the arch kind: `iter` drives [`CostBuffers::run_iter`];
/// `attn` / `ffn` build the AFD layer-wise model and drive the per-section evals
/// through [`CostBuffers::run_section`].
///
/// Every case is lowered and checked against the model before the first one is
/// costed, so a bad case fails the run before any row is written.
///
/// Files `run_timing_predict` writes besides the prediction, on success.
#[derive(Clone, Copy, Debug, Default)]
pub struct PredictOutputs<'a> {
    /// Every kernel config the model asks profile.db for (see `simulator
    /// build-cache-only --kernel-configs-out`).
    pub kernel_configs: Option<&'a Path>,
}

/// Kernel API config documents from `path` (see [`KernelData::from_json`]).
pub fn read_kernel_data(path: &Path) -> Result<KernelData> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading kernel data {}", path.display()))?;
    KernelData::from_json(&text).with_context(|| format!("parsing kernel data {}", path.display()))
}

/// `kernel_data` names the kernel API config documents to fit every cache from
/// instead of profile.db (see [`PerfApiBridge::kernel_data`]).
pub fn run_timing_predict(
    config_path: &Path,
    mode: PredictMode,
    outputs: &PredictOutputs,
    kernel_data: Option<&Path>,
) -> Result<()> {
    let cfg: PredictConfig = parse_config_file(config_path)?;
    let bridge = match (kernel_data, mode) {
        (Some(path), _) => PerfApiBridge::kernel_data(std::sync::Arc::new(read_kernel_data(path)?)),
        (None, PredictMode::Run) => predict_bridge()?,
        (None, PredictMode::DryRun) => {
            PerfApiBridge::new().context("starting the PyO3 perf_api bridge")?
        }
    };
    if mode == PredictMode::DryRun {
        bridge.enable_dry_run();
    }
    let kernel_configs_out = outputs.kernel_configs;
    if kernel_configs_out.is_some() {
        bridge.enable_config_records();
    }
    let run = mode == PredictMode::Run;

    // Cases are loaded per arch family — each arch owns its own case type, so we do
    // not assume a shared shape: iter/attn take the attention-shaped [`PredictCase`];
    // ffn takes [`FfnArchInput`] itself (token counts only), which rejects the
    // attention vocabulary the ffn cost never reads.
    let (num_cases, gpu_count) = match &cfg.arch {
        PredictArchSel::Iter(sel) => {
            let _scope = bridge.with_backend_overrides("main", cfg.backends.get("main"));
            let model = build_iter_model(sel, &cfg.gpu, UNIFIED_MODEL_NAME, &bridge)
                .context("building the iter-wise arch model")?;
            let cases: Vec<PredictCase> = load_cases(&cfg.cases_file, config_path)?;
            let inputs = iter_inputs(&*model, cases)?;
            let n = inputs.len();
            if run {
                run_iter_cases(&*model, inputs, &cfg.log_dir);
            }
            (n, model.gpus_per_replica())
        }
        PredictArchSel::SpeculativeIter(sel) => {
            let _scope = bridge.with_backend_overrides("main", cfg.backends.get("main"));
            let (model, draft_tokens) =
                build_speculative_iter_model(sel, &cfg.gpu, UNIFIED_MODEL_NAME, &bridge)
                    .context("building the speculative iter-wise model")?;
            let cases: Vec<SpeculativePredictCase> = load_cases(&cfg.cases_file, config_path)?;
            let inputs = speculative_iter_inputs(&*model, draft_tokens, cases)?;
            let n = inputs.len();
            if run {
                run_speculative_iter_cases(&*model, inputs, &cfg.log_dir);
            }
            (n, model.gpus_per_replica())
        }
        PredictArchSel::Attn(sel) => {
            let _scope = bridge.with_backend_overrides("attn", cfg.backends.get("attn"));
            let model = build_attn_model(sel, &cfg.gpu, AFD_MODEL_NAME, &bridge)?;
            let cases: Vec<PredictCase> = load_cases(&cfg.cases_file, config_path)?;
            let inputs = attn_inputs(&*model, cases)?;
            let n = inputs.len();
            if run {
                run_attn_cases(&*model, inputs, &cfg.log_dir);
            }
            (n, model.gpus_per_replica())
        }
        PredictArchSel::Ffn(sel) => {
            let _scope = bridge.with_backend_overrides("ffn", cfg.backends.get("ffn"));
            let model = build_ffn_model(sel, &cfg.gpu, AFD_MODEL_NAME, &bridge)?;
            let cases: Vec<FfnArchInput> = load_cases(&cfg.cases_file, config_path)?;
            check_ffn_cases(&*model, &cases)?;
            let n = cases.len();
            if run {
                run_ffn_cases(&*model, cases, &cfg.log_dir);
            }
            (n, model.gpus_per_replica())
        }
    };
    if let Some(path) = kernel_configs_out {
        write_config_records(path, &bridge.take_config_records())?;
    }
    if !run {
        print_dry_run(&bridge, num_cases, gpu_count);
        return Ok(());
    }
    write_prediction_provenance(&cfg.log_dir, &cfg.gpu, gpu_count)?;

    tracing::info!(
        log_dir = %cfg.log_dir.display(),
        "timing-predict wrote {num_cases} case(s)"
    );
    Ok(())
}

/// One case's predicted times: the `cost_log` row's `total_time_ms` and
/// `slot_time_ms`, plus `node_time_ms[i]` for flat node `i` of
/// [`Predictor::manifest`] (inside a `Scale{n}` subtree a node holds one
/// repeat; the `Scale` node holds all `n`).
/// One costed section of a case: a row of the native cost_log.
#[derive(Debug, Serialize)]
pub struct SectionTimes {
    /// The [`PredictorManifest`] section whose slots and nodes the times index.
    pub section: &'static str,
    /// The layer the section was costed at; `-1` for once-per-iteration ones.
    pub layer: i16,
    pub total_time_ms: f64,
    pub slot_time_ms: Vec<f32>,
    pub node_time_ms: Vec<f32>,
}

/// A case's sections, in the order `timing-predict` writes their rows: one
/// `iter` section for the iter and speculative selectors, `attn` for attn, and
/// `prologue` .. `epilogue` for ffn (see [`ffn_sections`]).
#[derive(Debug, Serialize)]
pub struct CaseTimes {
    pub sections: Vec<SectionTimes>,
}

/// What a caller needs to shape cases for a [`Predictor`].
#[derive(Debug, Serialize)]
pub struct PredictorInfo {
    /// `iter`, `speculative_iter`, `attn` or `ffn`: which case shape
    /// [`Predictor::predict`] takes.
    pub selector: &'static str,
    /// The groups a case carries (`groups`, or ffn's `tokens_per_group`).
    pub num_groups: u16,
    pub gpus_per_replica: u16,
    /// The longest request context a case may carry, when the model bounds it.
    pub max_model_len: Option<u32>,
    /// `speculative_iter` only: draft tokens per step, so a decode request's
    /// query is `draft_tokens + 1` rows.
    pub draft_tokens: Option<u32>,
}

/// The cost trees a [`Predictor`] costs, one per section, without the
/// per-slot `kernel_config` (what a caller needs to lay [`CaseTimes`] onto them).
#[derive(Debug, Serialize)]
pub struct PredictorManifest {
    pub sections: Vec<SectionManifest>,
}

#[derive(Debug, Serialize)]
pub struct SectionManifest {
    pub section: String,
    pub slots: Vec<String>,
    pub nodes: Vec<FlatCostNode>,
    pub node_labels: Vec<Option<String>>,
}

enum PredictModel {
    Iter(Box<dyn IterwiseUnifiedModel>),
    Speculative {
        model: Box<dyn SpeculativeUnifiedModel>,
        draft_tokens: u32,
    },
    Attn(Box<dyn AttnLayerwiseModel>),
    Ffn(Box<dyn FfnLayerwiseModel>),
}

/// A predict config's arch built once, costing cases in memory: the
/// `timing-predict` paths without files, parquet or artifacts. Built on any
/// bridge, a kernel-data one included (the wasm32 entry).
pub struct Predictor {
    model: PredictModel,
    manifest: CostManifestDoc,
    buffers: SectionBuffers,
}

/// The scratch a section is costed into, reused across cases.
#[derive(Default)]
struct SectionBuffers {
    slots: Vec<LeafMetrics>,
    scratch: Vec<LeafMetrics>,
    nodes: Vec<LeafMetrics>,
}

impl SectionBuffers {
    fn times(
        &mut self,
        manifest: &CostManifestDoc,
        section: &'static str,
        layer: i16,
        total: LeafMetrics,
    ) -> SectionTimes {
        let tree = &manifest
            .sections
            .iter()
            .find(|s| s.section == section)
            .unwrap_or_else(|| panic!("the model's cost manifest has no {section} section"))
            .manifest;
        CostTree::aggregate(&tree.nodes, &self.slots, &mut self.nodes);
        SectionTimes {
            section,
            layer,
            total_time_ms: total.m.time_ms as f64,
            slot_time_ms: self.slots.iter().map(|leaf| leaf.m.time_ms).collect(),
            node_time_ms: self.nodes.iter().map(|node| node.m.time_ms).collect(),
        }
    }
}

/// Sections as in-memory times: a [`Predictor`]'s [`SectionSink`].
struct TimesSink<'a> {
    manifest: &'a CostManifestDoc,
    buffers: &'a mut SectionBuffers,
    sections: Vec<SectionTimes>,
}

impl SectionSink for TimesSink<'_> {
    fn section<G, F>(&mut self, section: &'static str, layer: i16, _groups: &G, eval: F)
    where
        G: GroupLogSource,
        F: FnOnce(
            &mut Vec<LeafMetrics>,
            &mut Vec<LeafMetrics>,
            Option<&mut Vec<SlotInput>>,
        ) -> LeafMetrics,
    {
        let buffers = &mut *self.buffers;
        let total = eval(&mut buffers.slots, &mut buffers.scratch, None);
        let times = buffers.times(self.manifest, section, layer, total);
        self.sections.push(times);
    }
}

impl Predictor {
    /// Build from a predict config's `arch` value (`{"iter": {...}}`,
    /// `{"speculative_iter": {...}}`, `{"attn": {...}}` or `{"ffn": {...}}`),
    /// its GPU and optional `backends`.
    pub fn build(
        arch: serde_json::Value,
        gpu: &str,
        backends: Option<serde_json::Value>,
        bridge: &PerfApiBridge,
    ) -> Result<Self> {
        let sel: PredictArchSel =
            serde_json::from_value(arch).context("parsing the predict arch selector")?;
        let backends: BackendOverrides = match backends {
            Some(value) => serde_json::from_value(value).context("parsing backends")?,
            None => BackendOverrides::default(),
        };
        let model = match &sel {
            PredictArchSel::Iter(sel) => {
                let _scope = bridge.with_backend_overrides("main", backends.get("main"));
                PredictModel::Iter(
                    build_iter_model(sel, gpu, UNIFIED_MODEL_NAME, bridge)
                        .context("building the iter-wise arch model")?,
                )
            }
            PredictArchSel::SpeculativeIter(sel) => {
                let _scope = bridge.with_backend_overrides("main", backends.get("main"));
                let (model, draft_tokens) =
                    build_speculative_iter_model(sel, gpu, UNIFIED_MODEL_NAME, bridge)
                        .context("building the speculative iter-wise model")?;
                PredictModel::Speculative {
                    model,
                    draft_tokens,
                }
            }
            PredictArchSel::Attn(sel) => {
                let _scope = bridge.with_backend_overrides("attn", backends.get("attn"));
                PredictModel::Attn(build_attn_model(sel, gpu, AFD_MODEL_NAME, bridge)?)
            }
            PredictArchSel::Ffn(sel) => {
                let _scope = bridge.with_backend_overrides("ffn", backends.get("ffn"));
                PredictModel::Ffn(build_ffn_model(sel, gpu, AFD_MODEL_NAME, bridge)?)
            }
        };
        Ok(Self::from_model(model))
    }

    fn from_model(model: PredictModel) -> Self {
        let manifest = match &model {
            PredictModel::Iter(model) => CostManifestDoc::single("iter", model.cost_log_manifest()),
            PredictModel::Speculative { model, .. } => {
                CostManifestDoc::single("iter", model.cost_log_manifest())
            }
            PredictModel::Attn(model) => model.cost_log_manifest(),
            PredictModel::Ffn(model) => model.cost_log_manifest(),
        };
        Self {
            model,
            manifest,
            buffers: SectionBuffers::default(),
        }
    }

    pub fn info(&self) -> PredictorInfo {
        match &self.model {
            PredictModel::Iter(model) => PredictorInfo {
                selector: "iter",
                num_groups: model.num_attn_dp_groups(),
                gpus_per_replica: model.gpus_per_replica(),
                max_model_len: model.max_model_len(),
                draft_tokens: None,
            },
            PredictModel::Speculative {
                model,
                draft_tokens,
            } => PredictorInfo {
                selector: "speculative_iter",
                num_groups: model.num_attn_dp_groups(),
                gpus_per_replica: model.gpus_per_replica(),
                max_model_len: Some(model.max_model_len()),
                draft_tokens: Some(*draft_tokens),
            },
            PredictModel::Attn(model) => PredictorInfo {
                selector: "attn",
                num_groups: model.num_attn_dp_groups(),
                gpus_per_replica: model.gpus_per_replica(),
                max_model_len: None,
                draft_tokens: None,
            },
            PredictModel::Ffn(model) => PredictorInfo {
                selector: "ffn",
                num_groups: model.num_dp_groups(),
                gpus_per_replica: model.gpus_per_replica(),
                max_model_len: None,
                draft_tokens: None,
            },
        }
    }

    pub fn manifest(&self) -> PredictorManifest {
        PredictorManifest {
            sections: self
                .manifest
                .sections
                .iter()
                .map(|s| SectionManifest {
                    section: s.section.clone(),
                    slots: s
                        .manifest
                        .slots
                        .iter()
                        .map(|leaf| leaf.name.clone())
                        .collect(),
                    nodes: s.manifest.nodes.clone(),
                    node_labels: s.manifest.node_labels.clone(),
                })
                .collect(),
        }
    }

    /// Cost `cases` (a `timing-predict` cases array of the selector's shape).
    /// Every case is lowered and checked before the first is costed, as the
    /// file path does, so an invalid one fails as `case N: <reason>`.
    pub fn predict(&mut self, cases: serde_json::Value) -> Result<Vec<CaseTimes>> {
        let (manifest, buffers) = (&self.manifest, &mut self.buffers);
        // An iteration is one `iter` row, at no particular layer (`-1`).
        let one = |section: &'static str, buffers: &mut SectionBuffers, total| CaseTimes {
            sections: vec![buffers.times(manifest, section, -1, total)],
        };
        match &self.model {
            PredictModel::Iter(model) => {
                let cases: Vec<PredictCase> =
                    serde_json::from_value(cases).context("parsing cases")?;
                let inputs = iter_inputs(&**model, cases)?;
                Ok(inputs
                    .iter()
                    .map(|input| {
                        let total =
                            model.eval_iter(input, &mut buffers.slots, &mut buffers.scratch);
                        one("iter", buffers, total)
                    })
                    .collect())
            }
            PredictModel::Speculative {
                model,
                draft_tokens,
            } => {
                let cases: Vec<SpeculativePredictCase> =
                    serde_json::from_value(cases).context("parsing cases")?;
                let inputs = speculative_iter_inputs(&**model, *draft_tokens, cases)?;
                Ok(inputs
                    .iter()
                    .map(|input| {
                        let total = model.eval_speculative_iter(
                            input,
                            &mut buffers.slots,
                            &mut buffers.scratch,
                        );
                        one("iter", buffers, total)
                    })
                    .collect())
            }
            PredictModel::Attn(model) => {
                let cases: Vec<PredictCase> =
                    serde_json::from_value(cases).context("parsing cases")?;
                let inputs = attn_inputs(&**model, cases)?;
                Ok(inputs
                    .iter()
                    .map(|input| {
                        let mut sink = TimesSink {
                            manifest,
                            buffers: &mut *buffers,
                            sections: Vec::new(),
                        };
                        attn_sections(&**model, input, &mut sink);
                        CaseTimes {
                            sections: sink.sections,
                        }
                    })
                    .collect())
            }
            PredictModel::Ffn(model) => {
                let cases: Vec<FfnArchInput> =
                    serde_json::from_value(cases).context("parsing cases")?;
                check_ffn_cases(&**model, &cases)?;
                Ok(cases
                    .iter()
                    .map(|input| {
                        let mut sink = TimesSink {
                            manifest,
                            buffers: &mut *buffers,
                            sections: Vec::new(),
                        };
                        ffn_sections(&**model, input, &mut sink);
                        CaseTimes {
                            sections: sink.sections,
                        }
                    })
                    .collect())
            }
        }
    }
}

/// The dry-run report: the cases that validated, then one line per kernel with
/// the specs `profile.db` lacks (what a real run would JIT-profile).
fn print_dry_run(bridge: &PerfApiBridge, num_cases: usize, gpu_count: u16) {
    let report = bridge.take_dry_run_report();
    let total_missing: usize = report.iter().map(|k| k.missing).sum();
    let total_specs: usize = report.iter().map(|k| k.total).sum();
    println!("timing-predict dry run: {num_cases} case(s) valid, {gpu_count} GPU(s) per replica");
    for k in &report {
        println!(
            "  {:<40} ({:<16}) {:>8} / {:<8} missing",
            k.name, k.kind, k.missing, k.total
        );
    }
    println!(
        "total: {total_missing} / {total_specs} specs missing across {} kernels to JIT",
        report.len()
    );
}

fn write_prediction_provenance(log_dir: &Path, gpu_name: &str, gpu_count: u16) -> Result<()> {
    ensure!(gpu_count > 0, "timing-predict model has no physical GPUs");
    let raw_dir = log_dir.join("raw");
    fs::create_dir_all(&raw_dir)
        .with_context(|| format!("creating prediction raw directory {}", raw_dir.display()))?;
    let output_path = raw_dir.join(PREDICTION_PROVENANCE_FILE);
    let temporary_path = raw_dir.join(format!(".{PREDICTION_PROVENANCE_FILE}.tmp"));
    let bytes = serde_json::to_vec_pretty(&PredictionProvenance {
        schema_version: 1,
        gpu_name,
        gpu_count,
    })?;
    fs::write(&temporary_path, bytes)
        .with_context(|| format!("writing prediction provenance {}", temporary_path.display()))?;
    fs::rename(&temporary_path, &output_path)
        .with_context(|| format!("publishing prediction provenance {}", output_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod provenance_tests {
    use serde_json::Value;
    use tempfile::tempdir;

    use super::{write_prediction_provenance, PREDICTION_PROVENANCE_FILE};

    #[test]
    fn prediction_provenance_preserves_l4_gpu_extent() {
        let directory = tempdir().expect("temporary prediction directory");

        write_prediction_provenance(directory.path(), "NVIDIA H200", 4)
            .expect("write prediction provenance");

        let provenance: Value = serde_json::from_slice(
            &std::fs::read(
                directory
                    .path()
                    .join("raw")
                    .join(PREDICTION_PROVENANCE_FILE),
            )
            .expect("read prediction provenance"),
        )
        .expect("parse prediction provenance");
        assert_eq!(provenance["schema_version"], 1);
        assert_eq!(provenance["gpu_name"], "NVIDIA H200");
        assert_eq!(provenance["gpu_count"], 4);
        assert!(!directory.path().join("raw/run_meta.json").exists());
    }
}

/// Offline what-if bridge: JIT-fill missing `profile.db` rows on build (like
/// `kernel-query`), rather than the strict fail-fast a real `run` uses. A warm
/// cache then needs no GPU; a cold one profiles on demand.
fn predict_bridge() -> Result<PerfApiBridge> {
    let bridge = PerfApiBridge::new().context("starting the PyO3 perf_api bridge")?;
    bridge
        .enable_jit_profiling()
        .context("enabling JIT profiling for offline prediction")?;
    Ok(bridge)
}

/// Lower the iter-wise cases to the model's input, checking each against it.
fn iter_inputs(
    model: &dyn IterwiseUnifiedModel,
    cases: Vec<PredictCase>,
) -> Result<Vec<UnifiedArchInput>> {
    let expected_groups = model.num_attn_dp_groups() as usize;
    cases
        .into_iter()
        .enumerate()
        .map(|(idx, case)| {
            // Mirror `UnifiedIterExecution`: exact ragged EP source sizes are a
            // model-declared input capability, derived from the same DP groups.
            let groups = case
                .into_groups(expected_groups)
                .with_context(|| format!("case {idx}"))?;
            let tokens_per_source_rank = if groups.len() > 1 {
                groups.iter().map(|group| group.batch_tokens).collect()
            } else {
                Vec::new()
            };
            let input = UnifiedArchInput {
                groups,
                tokens_per_source_rank,
            };
            model
                .check_input(&input)
                .map_err(|reason| anyhow!("case {idx}: {reason}"))?;
            Ok(input)
        })
        .collect()
}

/// Run the iter-wise cases: one [`CostBuffers::run_iter`] per case (one parquet row
/// each, `section = "iter"`). Lays cases back-to-back on the wall-clock axis.
fn run_iter_cases(model: &dyn IterwiseUnifiedModel, inputs: Vec<UnifiedArchInput>, log_dir: &Path) {
    // The shared `CostBuffers` bundle every iter-wise worker carries: it owns the
    // eval scratch + the cost_log writer (pool_tag "predict", worker 0) and per
    // case runs `eval_iter_with_inputs` and writes one byte-identical parquet row.
    let mut cost = CostBuffers::new_iter(
        Some(log_dir.to_path_buf()),
        PREDICT_POOL_TAG,
        WorkerId(0),
        model,
        1.0, // predict = pure building-block cost; no inter-kernel overhead
    );
    let mut now = Time::from_ms(0.0);
    for (idx, arch_input) in inputs.iter().enumerate() {
        now += cost.run_iter(model, arch_input, idx as u64, now);
    }
    // Flush the writer's tail + join on drop (`CostLogger`'s `Drop`), same as a
    // worker at sim end; force it before reporting success.
    drop(cost);
}

/// Lower the draft/verify cases to the model's input, checking each against it.
fn speculative_iter_inputs(
    model: &dyn SpeculativeUnifiedModel,
    draft_tokens: u32,
    cases: Vec<SpeculativePredictCase>,
) -> Result<Vec<SpeculativeArchInput>> {
    let query_width = draft_tokens
        .checked_add(1)
        .context("verify width overflows u32")?;
    cases
        .into_iter()
        .enumerate()
        .map(|(index, case)| {
            ensure!(
                case.groups.len() == usize::from(model.num_attn_dp_groups()),
                "case {index}: group count does not match speculative model"
            );
            let groups: Vec<_> = case
                .groups
                .into_iter()
                .map(|group| group.into_arch_group(query_width, model.max_model_len()))
                .collect::<Result<_>>()
                .with_context(|| format!("case {index}"))?;
            let tokens_per_source_rank = if groups.len() > 1 {
                groups.iter().map(|group| group.batch_tokens).collect()
            } else {
                Vec::new()
            };
            let input = SpeculativeArchInput {
                draft_tokens,
                groups,
                tokens_per_source_rank,
            };
            model
                .check_input(&input)
                .map_err(|reason| anyhow!("case {index}: {reason}"))?;
            Ok(input)
        })
        .collect()
}

fn run_speculative_iter_cases(
    model: &dyn SpeculativeUnifiedModel,
    inputs: Vec<SpeculativeArchInput>,
    log_dir: &Path,
) {
    let mut cost = CostBuffers::new(
        Some(log_dir.to_path_buf()),
        PREDICT_POOL_TAG,
        WorkerId(0),
        &CostManifestDoc::single("iter", model.cost_log_manifest()),
        1.0,
    );
    let mut now = Time::from_ms(0.0);
    for (index, input) in inputs.iter().enumerate() {
        now += cost.run_speculative_iter(model, input, index as u64, now);
    }
    drop(cost);
}

/// Lower the AFD attn-side cases to the model's input, checking each against it.
fn attn_inputs(
    model: &dyn AttnLayerwiseModel,
    cases: Vec<PredictCase>,
) -> Result<Vec<AttnArchInput>> {
    let expected_groups = model.num_attn_dp_groups() as usize;
    cases
        .into_iter()
        .enumerate()
        .map(|(idx, case)| {
            let groups = case
                .into_groups(expected_groups)
                .with_context(|| format!("case {idx}"))?;
            Ok(AttnArchInput { groups })
        })
        .collect()
}

/// Where the sections of predict cases go: `timing-predict` writes each as a
/// cost_log row ([`CostLogSink`]); a [`Predictor`] keeps their times. One
/// description of a case's sections ([`attn_sections`], [`ffn_sections`])
/// serves both, so the rows cannot drift apart.
trait SectionSink {
    fn section<G, F>(&mut self, section: &'static str, layer: i16, groups: &G, eval: F)
    where
        G: GroupLogSource,
        F: FnOnce(
            &mut Vec<LeafMetrics>,
            &mut Vec<LeafMetrics>,
            Option<&mut Vec<SlotInput>>,
        ) -> LeafMetrics;
}

/// Sections as cost_log rows: case `iter_id`'s, one after another from `now`.
struct CostLogSink {
    cost: CostBuffers,
    iter_id: u64,
    now: Time,
}

impl CostLogSink {
    fn new(log_dir: &Path, manifest: &CostManifestDoc) -> Self {
        Self {
            cost: CostBuffers::new(
                Some(log_dir.to_path_buf()),
                PREDICT_POOL_TAG,
                WorkerId(0),
                manifest,
                1.0, // predict = pure building-block cost; no inter-kernel overhead
            ),
            iter_id: 0,
            now: Time::from_ms(0.0),
        }
    }
}

impl SectionSink for CostLogSink {
    fn section<G, F>(&mut self, section: &'static str, layer: i16, groups: &G, eval: F)
    where
        G: GroupLogSource,
        F: FnOnce(
            &mut Vec<LeafMetrics>,
            &mut Vec<LeafMetrics>,
            Option<&mut Vec<SlotInput>>,
        ) -> LeafMetrics,
    {
        self.now += self.cost.run_section(
            section,
            layer,
            self.iter_id,
            0,
            groups,
            None,
            self.now,
            eval,
        );
    }
}

/// An AFD attn-side case's one section: `attn_cost` (`section = "attn"`,
/// `layer = 0` -- every layer sees the same batch within an iteration, so the
/// per-layer cost is homogeneous). One model instance = one DP shard, so each
/// case carries exactly one group.
fn attn_sections<S: SectionSink>(
    model: &dyn AttnLayerwiseModel,
    input: &AttnArchInput,
    sink: &mut S,
) {
    sink.section(
        "attn",
        0,
        &input.groups,
        |slots, scratch, inputs| match inputs {
            Some(i) => model.attn_cost_with_inputs(0, input, slots, scratch, i),
            None => model.attn_cost(0, input, slots, scratch),
        },
    );
}

/// An AFD ffn-side case's sections: the per-section building blocks of one
/// iteration. The repeating per-mid-layer `post_attn` cost is homogeneous, so a
/// single representative mid layer stands for all of them; the terminal layer
/// (`post_attn_last`, post-only) is costed once. `prologue` and `epilogue` are
/// the once-per-iteration embed / lm_head, `layer = -1`.
///
/// The ffn case IS its input: [`FfnArchInput`] deserializes straight from the cases
/// file (a list of `{ "tokens_per_group": [...] }`), so there is no separate case
/// type and no lowering — the ffn cost reads the token counts directly. This is the
/// deliberate counterpoint to the iter/attn drivers' shared attention-shaped case:
/// each arch's case→input matches the shape its cost actually depends on.
fn ffn_sections<S: SectionSink>(model: &dyn FfnLayerwiseModel, input: &FfnArchInput, sink: &mut S) {
    let num_layers = model.num_layers();
    let last = num_layers.saturating_sub(1) as usize;
    let groups = &input.tokens_per_group;

    // prologue (embedding), once per iteration.
    sink.section(
        "prologue",
        -1,
        groups,
        |slots, scratch, inputs| match inputs {
            Some(i) => model.prologue_cost_with_inputs(input, slots, scratch, i),
            None => model.prologue_cost(input, slots, scratch),
        },
    );

    // pre_attn bootstrap: layer-0 qkv (layers > 0 are fused into the prior
    // layer's post_attn, so only layer 0 has a standalone pre cost).
    sink.section(
        "pre_attn",
        0,
        groups,
        |slots, scratch, inputs| match inputs {
            Some(i) => model.pre_attn_cost_with_inputs(0, input, slots, scratch, i),
            None => model.pre_attn_cost(0, input, slots, scratch),
        },
    );

    // post_attn for a representative mid layer (Bridge: post(L) + fused pre(L+1)),
    // standing for every layer in [0, last). Only when there IS a mid layer.
    if num_layers >= 2 {
        let mid = (num_layers as usize - 1) / 2; // clearly < last for num_layers >= 2
        sink.section(
            "post_attn",
            mid as i16,
            groups,
            |slots, scratch, inputs| match inputs {
                Some(i) => model.post_attn_cost_with_inputs(mid, input, slots, scratch, i),
                None => model.post_attn_cost(mid, input, slots, scratch),
            },
        );
    }

    // post_attn terminal (last layer, post-only).
    sink.section(
        "post_attn_last",
        last as i16,
        groups,
        |slots, scratch, inputs| match inputs {
            Some(i) => model.post_attn_cost_with_inputs(last, input, slots, scratch, i),
            None => model.post_attn_cost(last, input, slots, scratch),
        },
    );

    // epilogue (final_norm + lm_head), once per iteration.
    sink.section(
        "epilogue",
        -1,
        groups,
        |slots, scratch, inputs| match inputs {
            Some(i) => model.epilogue_cost_with_inputs(input, slots, scratch, i),
            None => model.epilogue_cost(input, slots, scratch),
        },
    );
}

/// Run the AFD attn-side cases, one cost_log row each (see [`attn_sections`]).
fn run_attn_cases(model: &dyn AttnLayerwiseModel, inputs: Vec<AttnArchInput>, log_dir: &Path) {
    let mut sink = CostLogSink::new(log_dir, &model.cost_log_manifest());
    for (idx, input) in inputs.iter().enumerate() {
        sink.iter_id = idx as u64;
        attn_sections(model, input, &mut sink);
    }
    // Flush the writer's tail + join (`CostLogger`'s `Drop`).
    drop(sink);
}

/// Run the AFD ffn-side cases, one cost_log row per section (see [`ffn_sections`]).
fn run_ffn_cases(model: &dyn FfnLayerwiseModel, cases: Vec<FfnArchInput>, log_dir: &Path) {
    let mut sink = CostLogSink::new(log_dir, &model.cost_log_manifest());
    for (idx, input) in cases.iter().enumerate() {
        sink.iter_id = idx as u64;
        ffn_sections(model, input, &mut sink);
    }
    drop(sink);
}

/// Check each ffn case's group count against the model.
fn check_ffn_cases(model: &dyn FfnLayerwiseModel, cases: &[FfnArchInput]) -> Result<()> {
    let expected_groups = model.num_dp_groups() as usize;
    for (idx, input) in cases.iter().enumerate() {
        ensure!(
            input.tokens_per_group.len() == expected_groups,
            "ffn case {idx} has {} group(s) but the model expects {expected_groups} (num_dp_groups)",
            input.tokens_per_group.len(),
        );
    }
    Ok(())
}

/// Parse a predict config (JSON via the JSON parser for clearer errors, else
/// YAML — YAML is a JSON superset). Generic over the config type so the legacy
/// iter config and the generalized one share it. Mirrors `main::load_config`.
fn parse_config_file<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading predict config {}", path.display()))?;
    if path.extension().and_then(|e| e.to_str()) == Some("json") {
        serde_json::from_str(&text)
            .with_context(|| format!("parsing JSON predict config {}", path.display()))
    } else {
        serde_yaml::from_str(&text)
            .with_context(|| format!("parsing YAML predict config {}", path.display()))
    }
}

/// Load the batched cases (a top-level array). `cases_file` resolves relative to
/// the config file's directory so a hand-written config + sibling cases file work
/// regardless of the launch CWD.
fn load_cases<T: DeserializeOwned>(cases_file: &Path, config_path: &Path) -> Result<Vec<T>> {
    let resolved = if cases_file.is_absolute() {
        cases_file.to_path_buf()
    } else {
        config_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(cases_file)
    };
    let text = fs::read_to_string(&resolved)
        .with_context(|| format!("reading cases file {}", resolved.display()))?;
    let cases: Vec<T> = if resolved.extension().and_then(|e| e.to_str()) == Some("json") {
        serde_json::from_str(&text)
            .with_context(|| format!("parsing JSON cases file {}", resolved.display()))?
    } else {
        serde_yaml::from_str(&text)
            .with_context(|| format!("parsing YAML cases file {}", resolved.display()))?
    };
    ensure!(
        !cases.is_empty(),
        "cases file {} is empty",
        resolved.display()
    );
    Ok(cases)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(json: &str) -> PredictGroup {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn speculative_prediction_keeps_requests_separate_from_query_rows() {
        let group: SpeculativePredictGroup = serde_json::from_str(
            r#"{"prefill_chunk_pairs": [[32, 8]], "decode_requests": [[106, 6], [206, 6]]}"#,
        )
        .unwrap();
        let lowered = group.into_arch_group(6, 8192).unwrap();
        assert_eq!(lowered.request_count(), 3);
        assert_eq!(lowered.decode_tokens, 12);
        assert_eq!(lowered.batch_tokens, 20);
        assert_eq!(lowered.total_kv_len, 300);
        assert_eq!(
            lowered.decode_requests[1],
            SpeculativeDecodeInput {
                kv_len: 206,
                query_len: 6
            }
        );
        assert!(
            serde_json::from_str::<PredictGroup>(r#"{"decode_requests": [[105, 6]]}"#).is_err()
        );
        assert!(serde_json::from_str::<SpeculativePredictGroup>(r#"{"decode_count": 2}"#).is_err());
    }

    #[test]
    fn speculative_prediction_rejects_partial_or_out_of_range_verification() {
        for pair in [[105, 2], [5, 6], [8193, 6]] {
            let group = SpeculativePredictGroup {
                prefill_chunk_pairs: vec![],
                decode_requests: vec![pair],
            };
            assert!(group.into_arch_group(6, 8192).is_err());
        }
        let group = SpeculativePredictGroup {
            prefill_chunk_pairs: vec![[u32::MAX, 1]],
            decode_requests: vec![],
        };
        assert!(group.into_arch_group(6, 8192).is_err());
    }

    #[test]
    fn exact_decode_list_is_used_verbatim() {
        let g = group(
            r#"{"prefill_chunk_pairs": [[0, 1025], [1024, 8]], "decode_kv_lens": [100, 200, 300]}"#,
        )
        .into_arch_group()
        .unwrap();
        // prefill_tokens = Σ append_len; decode from the explicit list.
        assert_eq!(g.prefill_tokens, 1025 + 8);
        assert_eq!(g.decode_tokens, 3);
        assert_eq!(g.total_kv_len, 600);
        assert_eq!(g.batch_tokens, 1025 + 8 + 3);
        assert_eq!(g.prefill_chunk_pairs, vec![(0, 1025), (1024, 8)]);
    }

    #[test]
    fn uniform_shorthand_expands_to_a_flat_decode_list() {
        let g = group(r#"{"decode_count": 4, "average_decode_length": 50}"#)
            .into_arch_group()
            .unwrap();
        assert_eq!(g.decode_tokens, 4);
        assert_eq!(g.decode_kv_lens, vec![50, 50, 50, 50]);
        assert_eq!(g.total_kv_len, 200);
        assert_eq!(g.prefill_tokens, 0);
    }

    #[test]
    fn both_decode_forms_is_an_error() {
        let err =
            group(r#"{"decode_kv_lens": [10], "decode_count": 1, "average_decode_length": 10}"#)
                .into_arch_group()
                .unwrap_err()
                .to_string();
        assert!(err.contains("exactly one"), "got: {err}");
    }

    #[test]
    fn decode_count_without_average_is_an_error() {
        let err = group(r#"{"decode_count": 4}"#)
            .into_arch_group()
            .unwrap_err()
            .to_string();
        assert!(err.contains("average_decode_length"), "got: {err}");
    }

    #[test]
    fn group_count_must_match_dp_degree() {
        let case: PredictCase = serde_json::from_str(
            r#"{"groups": [{"decode_count": 1, "average_decode_length": 8}]}"#,
        )
        .unwrap();
        // Model expects 2 DP shards but the case gave 1.
        let err = case.into_groups(2).unwrap_err().to_string();
        assert!(err.contains("expects 2"), "got: {err}");
    }

    #[test]
    fn model_rejection_names_the_case_before_any_eval() {
        use crate::timing::LeafMetrics;

        struct ContextCapped(crate::test_helpers::FakeModel);
        impl IterwiseUnifiedModel for ContextCapped {
            fn check_input(&self, batch: &UnifiedArchInput) -> Result<(), String> {
                match batch.groups[0].decode_kv_lens.iter().max() {
                    Some(&context) if context > 8192 => {
                        Err(format!("context {context} exceeds max_model_len 8192"))
                    }
                    _ => Ok(()),
                }
            }
            fn eval_iter(
                &self,
                _batch: &UnifiedArchInput,
                _slots: &mut Vec<LeafMetrics>,
                _scratch: &mut Vec<LeafMetrics>,
            ) -> LeafMetrics {
                unreachable!("a rejected case must not be costed")
            }
            fn total_kv_bytes_per_token(&self) -> u64 {
                self.0.total_kv_bytes_per_token()
            }
            fn gpus_per_replica(&self) -> u16 {
                self.0.gpus_per_replica()
            }
            fn num_attn_dp_groups(&self) -> u16 {
                self.0.num_attn_dp_groups()
            }
        }
        let cases: Vec<PredictCase> = serde_json::from_str(
            r#"[{"groups": [{"decode_kv_lens": [8192]}]},
                {"groups": [{"decode_kv_lens": [16384]}]}]"#,
        )
        .unwrap();
        let model = ContextCapped(crate::test_helpers::FakeModel::for_ms(1.0));
        let err = iter_inputs(&model, cases).unwrap_err().to_string();
        assert_eq!(err, "case 1: context 16384 exceeds max_model_len 8192");

        // The in-memory predictor lowers through the same check.
        let mut predictor = Predictor::from_model(PredictModel::Iter(Box::new(ContextCapped(
            crate::test_helpers::FakeModel::for_ms(1.0),
        ))));
        let cases = serde_json::json!([
            {"groups": [{"decode_kv_lens": [8192]}]},
            {"groups": [{"decode_kv_lens": [16384]}]},
        ]);
        let err = predictor.predict(cases).unwrap_err().to_string();
        assert_eq!(err, "case 1: context 16384 exceeds max_model_len 8192");
    }

    #[test]
    fn a_kernel_data_predictor_fails_on_a_config_without_a_document() {
        let bridge = PerfApiBridge::kernel_data(std::sync::Arc::new(KernelData::default()));
        let arch = serde_json::json!({"iter": {
            "type": "llama3_dense_tp",
            "model_config": "model/config/llama3_8b.json",
            "fp8": false,
            "tp_size": 1,
        }});
        let err = Predictor::build(arch, "NVIDIA H200", None, &bridge)
            .err()
            .expect("no documents build no kernel");
        assert!(format!("{err:#}").contains("no config document"), "{err:#}");
    }

    #[test]
    fn ffn_case_is_the_input_struct_directly() {
        // The ffn case deserializes straight into `FfnArchInput` — no separate case
        // type, no lowering. Token counts ARE the input.
        let input: FfnArchInput =
            serde_json::from_str(r#"{"tokens_per_group": [256, 256]}"#).unwrap();
        assert_eq!(input.tokens_per_group, vec![256, 256]);
        // deny_unknown_fields rejects stray attention vocabulary the ffn never reads.
        let bad: Result<FfnArchInput, _> =
            serde_json::from_str(r#"{"tokens_per_group": [1], "decode_count": 1}"#);
        assert!(bad.is_err());
    }

    #[test]
    fn unknown_field_in_group_is_rejected() {
        // deny_unknown_fields guards a typo like `decode_kv_len` (missing the `s`).
        let parsed: Result<PredictGroup, _> = serde_json::from_str(r#"{"decode_kv_len": [10]}"#);
        assert!(parsed.is_err());
    }

    #[test]
    fn predict_arch_sel_is_externally_tagged_by_kind() {
        // The generalized config selects the arch family by key; the inner selector
        // keeps its own `type` tag.
        let attn: PredictArchSel = serde_json::from_str(
            r#"{"attn": {"type": "qwen3_attn_tp", "model_config": "qwen3_235b", "fp8": false, "attn_tp_size": 4}}"#,
        )
        .expect("attn variant parses");
        assert!(matches!(attn, PredictArchSel::Attn(_)));

        let spec: PredictArchSel = serde_json::from_str(
            r#"{"speculative_iter": {"type": "glm52_vllm_nvfp4_dsa_moe_speculative", "model_config": "model/config/glm52_nvfp4.json", "ep_size": 4, "nvl_num_gpu": 4, "max_model_len": 8192, "fp8": false, "draft_tokens": 5}}"#,
        ).unwrap();
        assert!(matches!(spec, PredictArchSel::SpeculativeIter(_)));

        // An unknown kind is rejected (lists iter/attn/ffn).
        let bad: Result<PredictArchSel, _> = serde_json::from_str(r#"{"bogus": {"type": "x"}}"#);
        assert!(bad.is_err());
    }

    #[test]
    fn predict_arch_sel_uses_the_same_plain_map_in_yaml() {
        let iter: PredictArchSel = serde_yaml::from_str(
            r#"
iter:
  type: llama3_dense
  model_config: model/config/llama3_8b.json
  fp8: false
"#,
        )
        .expect("tag-free YAML iter selector parses");
        assert!(matches!(iter, PredictArchSel::Iter(_)));
    }

    #[test]
    fn an_arch_block_alone_selects_its_contract_by_type() {
        let parse = |block: &str| serde_json::from_str::<PredictArchSel>(block);
        let iter = parse(r#"{"type": "llama3_dense", "model_config": "m.json", "fp8": false}"#);
        assert!(matches!(iter, Ok(PredictArchSel::Iter(_))));
        let spec = parse(
            r#"{"type": "glm52_vllm_nvfp4_dsa_moe_speculative", "model_config": "m.json",
                "ep_size": 4, "nvl_num_gpu": 4, "max_model_len": 8192, "fp8": false,
                "draft_tokens": 5}"#,
        );
        assert!(matches!(spec, Ok(PredictArchSel::SpeculativeIter(_))));
        let attn = parse(
            r#"{"type": "qwen3_attn_tp", "model_config": "m.json", "attn_tp_size": 4, "fp8": false}"#,
        );
        assert!(matches!(attn, Ok(PredictArchSel::Attn(_))));
        let ffn = parse(
            r#"{"type": "qwen3_ffn_moe", "model_config": "m.json", "attn_tp_size": 4,
                "ep_size": 8, "nvl_num_gpu": 8, "routing": "uniform", "fp8": false}"#,
        );
        assert!(matches!(ffn, Ok(PredictArchSel::Ffn(_))));

        let unknown = parse(r#"{"type": "no_such_arch"}"#)
            .unwrap_err()
            .to_string();
        assert!(unknown.contains("no contract knows type"), "{unknown}");
        // A known type with a bad field says what is wrong with it.
        let bad = parse(r#"{"type": "llama3_dense", "model_config": "m.json"}"#)
            .unwrap_err()
            .to_string();
        assert!(bad.contains("fp8"), "{bad}");
    }

    #[test]
    fn predict_config_preserves_backend_overrides() {
        let cfg: PredictConfig = serde_json::from_str(
            r#"{
                "arch": {"iter": {
                    "type": "llama3_dense",
                    "model_config": "model/config/llama3_8b.json",
                    "fp8": false
                }},
                "gpu": "NVIDIA H200",
                "backends": {"main": {
                    "unified.pre_attn.qkv_proj": ["torch_linear"]
                }},
                "log_dir": "logs/predict",
                "cases_file": "cases.json"
            }"#,
        )
        .expect("predict config with backend overrides parses");
        assert_eq!(
            cfg.backends["main"]["unified.pre_attn.qkv_proj"],
            vec!["torch_linear"]
        );
    }
}
