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

use anyhow::{bail, ensure, Context, Result};
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
use crate::timing::{CostManifestDoc, PerfApiBridge};
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
/// JSON and YAML: `{ iter: {...} } | { attn: {...} } | { ffn: {...} }`.
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
pub fn run_timing_predict(config_path: &Path, mode: PredictMode) -> Result<()> {
    let cfg: PredictConfig = parse_config_file(config_path)?;
    let bridge = match mode {
        PredictMode::Run => predict_bridge()?,
        PredictMode::DryRun => {
            let bridge = PerfApiBridge::new().context("starting the PyO3 perf_api bridge")?;
            bridge.enable_dry_run();
            bridge
        }
    };
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
            Ok(UnifiedArchInput {
                groups,
                tokens_per_source_rank,
            })
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
            Ok(SpeculativeArchInput {
                draft_tokens,
                groups,
                tokens_per_source_rank,
            })
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

/// Run the AFD attn-side cases: one `attn_cost` per case (`section = "attn"`,
/// `layer = 0` — every layer sees the same batch within an iteration, so the
/// per-layer cost is homogeneous). One model instance = one DP shard, so each
/// case carries exactly one group.
fn run_attn_cases(model: &dyn AttnLayerwiseModel, inputs: Vec<AttnArchInput>, log_dir: &Path) {
    let manifest = model.cost_log_manifest();
    let mut cost = CostBuffers::new(
        Some(log_dir.to_path_buf()),
        PREDICT_POOL_TAG,
        WorkerId(0),
        &manifest,
        1.0, // predict = pure building-block cost; no inter-kernel overhead
    );
    let mut now = Time::from_ms(0.0);
    for (idx, input) in inputs.iter().enumerate() {
        let seg = cost.run_section(
            "attn",
            0,
            idx as u64,
            0,
            &input.groups,
            None,
            now,
            |slots, scratch, inputs| match inputs {
                Some(i) => model.attn_cost_with_inputs(0, input, slots, scratch, i),
                None => model.attn_cost(0, input, slots, scratch),
            },
        );
        now += seg;
    }
    drop(cost);
}

/// Run the AFD ffn-side cases: emit the per-section building blocks of one
/// iteration, each as one row. The repeating per-mid-layer `post_attn` cost is
/// homogeneous, so a single representative mid layer stands for all of them; the
/// terminal layer (`post_attn_last`, post-only) is emitted once. `prologue` and
/// `epilogue` are the once-per-iteration embed / lm_head, `layer = -1`.
///
/// The ffn case IS its input: [`FfnArchInput`] deserializes straight from the cases
/// file (a list of `{ "tokens_per_group": [...] }`), so there is no separate case
/// type and no lowering — the ffn cost reads the token counts directly. This is the
/// deliberate counterpoint to the iter/attn drivers' shared attention-shaped case:
/// each arch's case→input matches the shape its cost actually depends on.
fn run_ffn_cases(model: &dyn FfnLayerwiseModel, cases: Vec<FfnArchInput>, log_dir: &Path) {
    let num_layers = model.num_layers();
    let last = num_layers.saturating_sub(1) as usize;
    let manifest = model.cost_log_manifest();
    let mut cost = CostBuffers::new(
        Some(log_dir.to_path_buf()),
        PREDICT_POOL_TAG,
        WorkerId(0),
        &manifest,
        1.0, // predict = pure building-block cost; no inter-kernel overhead
    );
    let mut now = Time::from_ms(0.0);
    for (idx, input) in cases.into_iter().enumerate() {
        let iid = idx as u64;

        // prologue (embedding), once per iteration.
        let seg = cost.run_section(
            "prologue",
            -1,
            iid,
            0,
            &input.tokens_per_group,
            None,
            now,
            |slots, scratch, inputs| match inputs {
                Some(i) => model.prologue_cost_with_inputs(&input, slots, scratch, i),
                None => model.prologue_cost(&input, slots, scratch),
            },
        );
        now += seg;

        // pre_attn bootstrap: layer-0 qkv (layers > 0 are fused into the prior
        // layer's post_attn, so only layer 0 has a standalone pre cost).
        let seg = cost.run_section(
            "pre_attn",
            0,
            iid,
            0,
            &input.tokens_per_group,
            None,
            now,
            |slots, scratch, inputs| match inputs {
                Some(i) => model.pre_attn_cost_with_inputs(0, &input, slots, scratch, i),
                None => model.pre_attn_cost(0, &input, slots, scratch),
            },
        );
        now += seg;

        // post_attn for a representative mid layer (Bridge: post(L) + fused pre(L+1)),
        // standing for every layer in [0, last). Only when there IS a mid layer.
        if num_layers >= 2 {
            let mid = (num_layers as usize - 1) / 2; // clearly < last for num_layers >= 2
            let seg = cost.run_section(
                "post_attn",
                mid as i16,
                iid,
                0,
                &input.tokens_per_group,
                None,
                now,
                |slots, scratch, inputs| match inputs {
                    Some(i) => model.post_attn_cost_with_inputs(mid, &input, slots, scratch, i),
                    None => model.post_attn_cost(mid, &input, slots, scratch),
                },
            );
            now += seg;
        }

        // post_attn terminal (last layer, post-only).
        let seg = cost.run_section(
            "post_attn_last",
            last as i16,
            iid,
            0,
            &input.tokens_per_group,
            None,
            now,
            |slots, scratch, inputs| match inputs {
                Some(i) => model.post_attn_cost_with_inputs(last, &input, slots, scratch, i),
                None => model.post_attn_cost(last, &input, slots, scratch),
            },
        );
        now += seg;

        // epilogue (final_norm + lm_head), once per iteration.
        let seg = cost.run_section(
            "epilogue",
            -1,
            iid,
            0,
            &input.tokens_per_group,
            None,
            now,
            |slots, scratch, inputs| match inputs {
                Some(i) => model.epilogue_cost_with_inputs(&input, slots, scratch, i),
                None => model.epilogue_cost(&input, slots, scratch),
            },
        );
        now += seg;
    }
    drop(cost);
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
