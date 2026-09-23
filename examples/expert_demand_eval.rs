//! Score every MoE expert-demand source against recorded per-iteration truth.
//!
//! The `expert_popularity` profile pass keeps only a full-run marginal, but the
//! same pass also streams one `logical_expert_counts` record per forward step.
//! That stream is the exact demand each iteration placed on the grouped GEMM,
//! so it can referee the sources the simulator actually offers: the full-run
//! popularity resample and the recorded token corpus. Both are reduced through
//! the production fold, so the comparison is between the vectors the kernel
//! cache really receives — not between abstractions of them.
//!
//! ```text
//! uv run cargo run --release --example expert_demand_eval -- \
//!   --expert-load <expert_load.jsonl> --popularity <expert_popularity.json> \
//!   --corpus <manifest.json> --out <rows.csv>
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use anyhow::{Context, Result};
use serde::Deserialize;
use simulator::arch::build::load_expert_popularity;
use simulator::timing::routing::{
    fold_layerwise_expert_counts, sample_and_fold_layerwise_topk_expert_counts,
};
use simulator::timing::token_corpus::{median_candidate, TokenCorpusConfig};

/// The seed the fused-MoE kernel configs sample with; reused so the popularity
/// column is the production draw and not a fresh one.
const FOLD_SEED: u64 = 0xF01D_5EED;
/// Candidate stride from `TokenCorpus::sample_and_fold`, so the popularity
/// median column differs from the corpus median column only in its source.
const CANDIDATE_STRIDE: u64 = 0xd1b5_4a32_d192_ed03;

#[derive(Deserialize)]
struct ExpertLoadRecord {
    logical_expert_counts: Vec<Vec<u32>>,
}

/// The scheduler record the profiler writes beside every expert-load record, so
/// each iteration can be scored under the composition that produced it.
#[derive(Deserialize)]
struct IterationMetrics {
    prefill_tokens: u32,
    decode_requests: u32,
    decode_tokens_scheduled: u32,
}

/// One folded 256-slot histogram reduced to the features that set kernel cost:
/// the heaviest rank's rows and how many of its expert groups are non-empty.
struct Summary {
    load: u64,
    active: usize,
    top: u32,
}

fn summarize(folded: &[u32], experts_per_rank: usize) -> Summary {
    let rank = &folded[..experts_per_rank];
    Summary {
        load: rank.iter().map(|&count| u64::from(count)).sum(),
        active: rank.iter().filter(|&&count| count > 0).count(),
        top: rank.iter().copied().max().unwrap_or(0),
    }
}

/// Total variation between two folded histograms, as a fraction of one of them.
/// Both carry the same assignment count up to fold rounding, so this is the
/// share of token rows the source places in the wrong expert group.
fn total_variation(left: &[u32], right: &[u32]) -> f64 {
    let diff: u64 = left
        .iter()
        .zip(right)
        .map(|(&a, &b)| u64::from(a.abs_diff(b)))
        .sum();
    let total: u64 = left.iter().map(|&count| u64::from(count)).sum();
    if total == 0 {
        0.0
    } else {
        diff as f64 / (2.0 * total as f64)
    }
}

fn arg(name: &str) -> Option<String> {
    let mut args = std::env::args();
    while let Some(item) = args.next() {
        if item == name {
            return args.next();
        }
    }
    None
}

fn main() -> Result<()> {
    let expert_load = arg("--expert-load").context("--expert-load <jsonl> is required")?;
    let metrics_path = arg("--metrics").context("--metrics <jsonl> is required")?;
    let popularity = arg("--popularity").context("--popularity <json> is required")?;
    // Optional: a model with a recorded capture but no corpus yet still gets the
    // popularity columns and the floor, which is what says how much a corpus
    // would be worth on that model.
    let corpus_manifest = arg("--corpus");
    let out = arg("--out").context("--out <csv> is required")?;
    let experts_per_rank: usize = arg("--experts-per-rank").map_or(Ok(64), |v| v.parse())?;
    let ep_size: u16 = arg("--ep-size").map_or(Ok(4), |v| v.parse())?;
    let top_k: u32 = arg("--top-k").map_or(Ok(8), |v| v.parse())?;
    let group_size: u32 = arg("--group-size").map_or(Ok(8), |v| v.parse())?;
    let candidates: u32 = arg("--candidates").map_or(Ok(16), |v| v.parse())?;
    let limit: usize = arg("--limit").map_or(Ok(usize::MAX), |v| v.parse())?;
    // `expert_count_reduction_group_size` ranks each log the same global logical
    // histogram, and the profiler sums them. Undo that before comparing against
    // a sampler, which produces one global histogram for the step's tokens.
    let reduction: u32 = arg("--reduction").map_or(Ok(4), |v| v.parse())?;
    // Skip the popularity columns; a `--group-size` sweep would otherwise
    // recompute the same draws once per group size.
    let skip_popularity = std::env::args().any(|a| a == "--no-popularity");

    let num_experts = (experts_per_rank * usize::from(ep_size)) as u32;
    let num_layers: u32 = arg("--num-layers").map_or(Ok(0), |v| v.parse())?;

    let corpora = corpus_manifest
        .map(|path| -> Result<_> {
            let mut single = TokenCorpusConfig::from_manifest(&path, group_size, FOLD_SEED)?;
            single.sampling_candidates = 1;
            let mut median = single.clone();
            median.sampling_candidates = candidates;
            Ok((single.num_layers as u32, single.load()?, median.load()?))
        })
        .transpose()?;
    let num_layers = match (&corpora, num_layers) {
        (Some((layers, ..)), 0) => *layers,
        (Some((layers, ..)), given) => {
            anyhow::ensure!(*layers == given, "corpus has {layers} layers, --num-layers says {given}");
            given
        }
        (None, 0) => anyhow::bail!("--num-layers is required when no --corpus is given"),
        (None, given) => given,
    };

    // Schema v4 profiles carry a model role; the v2/v3 captures predate it.
    let role = arg("--role").unwrap_or_else(|| "target".to_owned());
    let routing = load_expert_popularity(
        &popularity,
        num_experts,
        ep_size,
        num_layers,
        top_k,
        (role != "none").then_some(role.as_str()),
    )?;
    let layer_ppm = routing.layerwise_ppm(num_layers);

    let mut writer = BufWriter::new(File::create(&out)?);
    writeln!(
        writer,
        "index,tokens,prefill_tokens,decode_requests,decode_tokens,\
         truth_load,truth_active,truth_top,\
         pop1_load,pop1_active,pop1_top,pop1_tv,\
         pop{candidates}_load,pop{candidates}_active,pop{candidates}_top,pop{candidates}_tv,\
         corpus1_load,corpus1_active,corpus1_top,corpus1_tv,\
         corpus{candidates}_load,corpus{candidates}_active,corpus{candidates}_top,corpus{candidates}_tv,\
         prev_load,prev_active,prev_top,prev_tv"
    )?;

    // The irreducible floor: the most recent *other* recorded step with the same
    // token count. No source that knows only the token count can beat the
    // spread between two real steps of that size, so this column calibrates how
    // much of a source's error is model error rather than sampling noise.
    let mut previous: HashMap<u32, Vec<u32>> = HashMap::new();

    let metrics: Vec<IterationMetrics> = BufReader::new(File::open(&metrics_path)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect::<Result<_>>()?;

    let reader = BufReader::new(File::open(&expert_load)?);
    for (index, line) in reader.lines().enumerate().take(limit) {
        let record: ExpertLoadRecord = serde_json::from_str(&line?)
            .with_context(|| format!("parsing expert load record {index}"))?;
        let step = metrics
            .get(index)
            .with_context(|| format!("no scheduler record beside expert load record {index}"))?;
        let counts: Vec<Vec<u32>> = record
            .logical_expert_counts
            .iter()
            .map(|layer| layer.iter().map(|&count| count / reduction).collect())
            .collect();
        let assignments: u64 = counts[0].iter().map(|&count| u64::from(count)).sum();
        if assignments == 0 {
            continue;
        }
        // The first recorded step can carry warmup work that reached only part
        // of the stack -- DeepSeek-V4-Flash records 364 tokens in layers 0..2
        // and 495 in the rest. The production fold rejects that outright, and
        // it is not a step any source is meant to reproduce, so drop it.
        if counts
            .iter()
            .any(|layer| layer.iter().map(|&c| u64::from(c)).sum::<u64>() != assignments)
        {
            eprintln!("skipping record {index}: layers disagree on assignment count");
            continue;
        }
        let tokens = (assignments / u64::from(top_k)) as u32;

        let truth = fold_layerwise_expert_counts(&counts, experts_per_rank);
        // The popularity columns do not depend on `--group-size`, so a sweep over
        // it can skip them instead of recomputing 16 draws per record per run.
        let pop_candidates: Vec<Vec<u32>> = (0..if skip_popularity { 0 } else { candidates })
            .map(|draw| {
                sample_and_fold_layerwise_topk_expert_counts(
                    &layer_ppm,
                    top_k,
                    tokens,
                    experts_per_rank,
                    FOLD_SEED.wrapping_add(u64::from(draw).wrapping_mul(CANDIDATE_STRIDE)),
                )
            })
            .collect();
        let popularity_folds = (!pop_candidates.is_empty()).then(|| {
            (
                &pop_candidates[0],
                &pop_candidates[median_candidate(&pop_candidates, experts_per_rank)],
            )
        });
        let corpus_folds = corpora.as_ref().map(|(_, single, median)| {
            (
                single.sample_and_fold(tokens, experts_per_rank),
                median.sample_and_fold(tokens, experts_per_rank),
            )
        });

        let mut row = format!(
            "{index},{tokens},{},{},{}",
            step.prefill_tokens, step.decode_requests, step.decode_tokens_scheduled
        );
        let t = summarize(&truth, experts_per_rank);
        row.push_str(&format!(",{},{},{}", t.load, t.active, t.top));
        // Keep the column layout fixed so one summarizer reads every run: a
        // source that was not computed leaves its four columns blank.
        for pair in [
            popularity_folds.map(|(one, median)| (one.as_slice(), median.as_slice())),
            corpus_folds
                .as_ref()
                .map(|(one, median)| (one.as_slice(), median.as_slice())),
        ] {
            match pair {
                Some((one, median)) => {
                    for folded in [one, median] {
                        let s = summarize(folded, experts_per_rank);
                        row.push_str(&format!(
                            ",{},{},{},{:.6}",
                            s.load,
                            s.active,
                            s.top,
                            total_variation(&truth, folded)
                        ));
                    }
                }
                None => row.push_str(",,,,,,,,"),
            }
        }
        match previous.insert(tokens, truth.clone()) {
            Some(prior) => {
                let s = summarize(&prior, experts_per_rank);
                row.push_str(&format!(
                    ",{},{},{},{:.6}",
                    s.load,
                    s.active,
                    s.top,
                    total_variation(&truth, &prior)
                ));
            }
            // First step at this token count: there is nothing to pair it with.
            None => row.push_str(",,,,"),
        }
        writeln!(writer, "{row}")?;
    }
    writer.flush()?;
    eprintln!("wrote {out}");
    Ok(())
}
