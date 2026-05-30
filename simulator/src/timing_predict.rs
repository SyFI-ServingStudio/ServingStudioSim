//! `iter-timing-predict` — offline whole-iteration timing prediction.
//!
//! This is the same family as `dry-run` / `build-cache-only` / `kernel-query`:
//! it does NOT run the discrete-event sim (no scheduler, no trace, no clock). It
//! takes a batch of explicit iteration shapes and, for each, runs ONE
//! `IterwiseUnifiedModel::eval_iter` over the compiled `CostTree`, predicting the
//! whole forward pass (embedding → all layers via the `Scale{n}` fold → lm_head).
//! The unit is always a full **iteration**, even for an arch whose model is
//! layer-wise internally — hence `iter`, not `layer` (a future per-layer-section
//! `eval_layer` is an *in-sim* AFD method, not this tool).
//!
//! The output is not a bespoke format: each case is emitted as ONE row of the
//! standard `raw/cost_log/worker_predict_0.parquet` + the matching
//! `raw/cost_manifest/worker_predict_0.json` — byte-for-byte the artifacts a real
//! `run` writes (one worker, `iter_id` = case index). So `analyze trace` renders
//! the `iter → layers → kernels` Perfetto tree and `analyze run` computes its
//! reports with **zero** changes (`docs/analyzer.md`).
//!
//! Config (a minimal file, NOT a `RunConfig`): ONE iter-wise arch selector + its
//! GPU + a log dir + a batched cases file. No workload / pools / io — those
//! belong to `run`. Built via the shared `arch::build::build_iter_model` seam
//! (the offline path is `dyn`; the sim's cost path stays monomorphized), and the
//! per-case eval + cost_log write reuse the worker's `CostBuffers` bundle.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;

use crate::arch::build::build_iter_model;
use crate::arch::contract::{ArchGroupInput, UnifiedArchInput};
use crate::arch::IterArchSel;
use crate::common::{Time, WorkerId};
use crate::timing::PerfApiBridge;
use crate::worker::CostBuffers;

/// Minimal offline config. `arch` reuses the run-side [`IterArchSel`] (it already
/// flattens `ModelSpec` — model path, layer overrides, parallel dims), so a
/// predict config is just that selector plus where to run it and what to predict.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TimingPredictConfig {
    arch: IterArchSel,
    gpu: String,
    log_dir: PathBuf,
    /// Path to the batched cases JSON/YAML (a top-level array of [`PredictCase`]).
    /// Resolved relative to this config file's directory when not absolute.
    cases_file: PathBuf,
}

/// One predicted iteration: exactly `num_attn_dp_groups()` attention-DP shards
/// (validated against the built model), each a per-rank batch view.
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
    /// `worker/unified.rs::build_arch_input`, so a predicted iteration costs
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
    /// Validate the group count against the model's attention-DP degree and lower
    /// every group. `tokens_per_source_rank` stays empty (same as the workers; its
    /// L4/L5 ownership is unsettled — see memory `tokens_per_source_rank_layer_conflict`).
    fn into_arch_input(self, expected_groups: usize) -> Result<UnifiedArchInput> {
        ensure!(
            self.groups.len() == expected_groups,
            "case has {} group(s) but the model expects {} (num_attn_dp_groups)",
            self.groups.len(),
            expected_groups,
        );
        let groups = self
            .groups
            .into_iter()
            .map(PredictGroup::into_arch_group)
            .collect::<Result<Vec<_>>>()?;
        Ok(UnifiedArchInput {
            groups,
            tokens_per_source_rank: Vec::new(),
        })
    }
}

/// Entry point for `simulator iter-timing-predict <config>`.
pub fn run_iter_timing_predict(config_path: &Path) -> Result<()> {
    let cfg = load_config(config_path)?;
    let cases = load_cases(&cfg.cases_file, config_path)?;
    let num_cases = cases.len();

    // Offline what-if: JIT-fill missing `profile.db` rows on build (like
    // `kernel-query`), rather than the strict fail-fast a real `run` uses. A warm
    // cache then needs no GPU; a cold one profiles on demand.
    let bridge = PerfApiBridge::new().context("starting the PyO3 perf_api bridge")?;
    bridge
        .enable_jit_profiling()
        .context("enabling JIT profiling for offline prediction")?;

    // `name` is the model's dotted-leaf prefix; reuse the co-located unified
    // run's so a predicted manifest matches a real run's leaf names.
    let model = build_iter_model(&cfg.arch, &cfg.gpu, "unified", &bridge)
        .context("building the arch model for iter-timing-predict")?;
    let expected_groups = model.num_attn_dp_groups() as usize;

    // Standard per-"worker" artifacts via the shared `CostBuffers` bundle — the
    // same one every iter-wise worker carries (`worker/cost_buffers.rs`): it owns
    // the eval scratch (slots / scratch / slot_inputs + the per-group input log)
    // plus the cost_log writer, and per case runs `eval_iter_with_inputs` and
    // writes one parquet row (pool_tag "predict", worker 0). The manifest sidecar +
    // parquet are byte-identical to a real run, so the analyzer consumes them
    // unchanged.
    let mut cost = CostBuffers::new(Some(cfg.log_dir.clone()), "predict", WorkerId(0), &*model);

    // Lay cases out back-to-back on the trace's wall-clock axis: `run_iter` stamps
    // `now` as this row's `wall_start_ms` and returns the iter's predicted time, so
    // accumulating it starts the next case where this one ends.
    let mut now = Time::from_ms(0.0);
    for (idx, case) in cases.into_iter().enumerate() {
        let arch_input = case
            .into_arch_input(expected_groups)
            .with_context(|| format!("case {idx}"))?;
        now += cost.run_iter(&*model, &arch_input, WorkerId(0), idx as u64, now);
    }
    // The writer flushes its tail and joins on drop (`CostLogger`'s `Drop`), the
    // same as a worker at sim end; force it here so the parquet is complete before
    // we report success.
    drop(cost);

    tracing::info!(
        log_dir = %cfg.log_dir.display(),
        "iter-timing-predict wrote {num_cases} case(s)"
    );
    Ok(())
}

/// Parse the minimal config (JSON via the JSON parser for clearer errors, else
/// YAML — YAML is a JSON superset). Mirrors `main::load_config`.
fn load_config(path: &Path) -> Result<TimingPredictConfig> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading predict config {}", path.display()))?;
    let cfg = if path.extension().and_then(|e| e.to_str()) == Some("json") {
        serde_json::from_str(&text)
            .with_context(|| format!("parsing JSON predict config {}", path.display()))?
    } else {
        serde_yaml::from_str(&text)
            .with_context(|| format!("parsing YAML predict config {}", path.display()))?
    };
    Ok(cfg)
}

/// Load the batched cases (a top-level array). `cases_file` resolves relative to
/// the config file's directory so a hand-written config + sibling cases file work
/// regardless of the launch CWD.
fn load_cases(cases_file: &Path, config_path: &Path) -> Result<Vec<PredictCase>> {
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
    let cases: Vec<PredictCase> = if resolved.extension().and_then(|e| e.to_str()) == Some("json") {
        serde_json::from_str(&text)
            .with_context(|| format!("parsing JSON cases file {}", resolved.display()))?
    } else {
        serde_yaml::from_str(&text)
            .with_context(|| format!("parsing YAML cases file {}", resolved.display()))?
    };
    ensure!(!cases.is_empty(), "cases file {} is empty", resolved.display());
    Ok(cases)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn group(json: &str) -> PredictGroup {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn exact_decode_list_is_used_verbatim() {
        let g = group(r#"{"prefill_chunk_pairs": [[0, 1025], [1024, 8]], "decode_kv_lens": [100, 200, 300]}"#)
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
        let err = group(r#"{"decode_kv_lens": [10], "decode_count": 1, "average_decode_length": 10}"#)
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
        let case: PredictCase =
            serde_json::from_str(r#"{"groups": [{"decode_count": 1, "average_decode_length": 8}]}"#)
                .unwrap();
        // Model expects 2 DP shards but the case gave 1.
        let err = case.into_arch_input(2).unwrap_err().to_string();
        assert!(err.contains("num_attn_dp_groups"), "got: {err}");
    }

    #[test]
    fn unknown_field_in_group_is_rejected() {
        // deny_unknown_fields guards a typo like `decode_kv_len` (missing the `s`).
        let parsed: Result<PredictGroup, _> =
            serde_json::from_str(r#"{"decode_kv_len": [10]}"#);
        assert!(parsed.is_err());
    }
}
