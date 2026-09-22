//! Where a fused-MoE kernel's routed demand comes from.
//!
//! A grouped GEMM is billed by two things, and neither is the token count: how
//! many rows land on the busiest EP rank, and how many of that rank's expert
//! groups are non-empty. The second is a statement about *co-occurrence* —
//! which experts a step's tokens jointly select — so it is not answerable from
//! a per-expert marginal. Resampling a marginal is, by construction, drawing
//! tokens independently, and a serving batch is not independent.
//!
//! Both sources therefore reduce to the same thing here: one complete folded
//! histogram for a given token count. Kernels see only that, so the backing can
//! change (a pre-computed table over the sweep grid, say) without touching a
//! call site.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use super::routing::{sample_and_fold_layerwise_topk_expert_counts, RoutingDistribution};
use super::token_corpus::{TokenCorpus, TokenCorpusConfig};

/// The seed every demand source folds at, so a profiled shape is reproducible.
const FOLD_SEED: u64 = 0xF01D_5EED;

/// Payloads already read this process, keyed by the file and the checksum that
/// identifies its contents.
///
/// One build resolves the same corpus once per MoE callable and again per
/// kernel config per backend, and each resolution is a hundreds-of-megabytes
/// read plus a full top-k distinctness scan. The manifest's checksum is the
/// corpus's identity, so two configs naming the same one are the same bytes and
/// a second read can only confirm what the first proved.
type CorpusCache = Mutex<HashMap<(String, u64), Arc<Vec<u16>>>>;
static LOADED_CORPORA: OnceLock<CorpusCache> = OnceLock::new();

fn load_cached(config: &TokenCorpusConfig) -> anyhow::Result<TokenCorpus> {
    let key = (config.data_file.clone(), config.checksum_fnv1a64);
    let cache = LOADED_CORPORA.get_or_init(Default::default);
    let cached = cache.lock().expect("corpus cache").get(&key).cloned();
    if let Some(ids) = cached {
        // Sampling parameters live in the config, not the payload, so another
        // callable binds its own group size and layer slice to these bytes.
        return TokenCorpus::rebind(config, ids);
    }
    let corpus = config.load()?;
    cache
        .lock()
        .expect("corpus cache")
        .insert(key, corpus.payload());
    Ok(corpus)
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertDemand {
    /// Resample a per-layer expert marginal. Synthetic uniform and random
    /// routing land here too: they differ from a measured profile only in where
    /// the distribution came from, not in how it is drawn.
    Popularity { layerwise_global_ppm: Vec<Vec<u32>> },
    /// Sample recorded per-token routes in contiguous groups the width of the
    /// deployment's verify block.
    Corpus(TokenCorpusConfig),
}

impl ExpertDemand {
    /// Materialize a routing distribution over the layers a model actually
    /// routes. A distribution carries no layer count of its own — a synthetic
    /// one is a single vector — so the architecture supplies it.
    pub fn popularity(routing: &RoutingDistribution, num_layers: u32) -> Self {
        Self::Popularity {
            layerwise_global_ppm: routing.layerwise_ppm(num_layers),
        }
    }

    /// A marginal read at one layer's resolution.
    ///
    /// The MTP layer sits outside the layers an expert-popularity profile was
    /// captured over, and a marginal has no layer axis left to extend, so it
    /// folds the profile's layer-summed distribution — the same evidence, at
    /// the only resolution a marginal has. Measuring that layer's own routing
    /// is what `routing: corpus` is for. Synthetic uniform and random routing
    /// carry no layer axis to begin with, so for them this is [`Self::popularity`]
    /// at one layer.
    pub fn popularity_summed(routing: &RoutingDistribution) -> Self {
        Self::Popularity {
            layerwise_global_ppm: vec![routing.ppm().to_vec()],
        }
    }

    /// Bind a recorded corpus to one MoE callable: its verify width and the
    /// layers it covers. The fold seed is shared with the popularity path so
    /// both sources are reproducible the same way.
    ///
    /// Validated here rather than at first use: a kernel's `enumerate` runs
    /// deep inside a build cascade, where an unreadable corpus would surface as
    /// a missing profile row instead of a bad config.
    pub fn corpus(
        manifest: &str,
        group_size: u32,
        layers: std::ops::Range<usize>,
    ) -> anyhow::Result<Self> {
        let config = TokenCorpusConfig::from_manifest(manifest, group_size, FOLD_SEED, layers)?;
        config.load().context("reading the token corpus payload")?;
        Ok(Self::Corpus(config))
    }

    /// Expert count this source produces histograms over, so a consumer can
    /// check it against the model without knowing which source it holds.
    pub fn num_experts(&self) -> usize {
        match self {
            Self::Popularity {
                layerwise_global_ppm,
            } => layerwise_global_ppm.first().map_or(0, Vec::len),
            Self::Corpus(config) => config.num_experts,
        }
    }

    /// Resolve the payload once, before a sweep folds every grid point.
    ///
    /// The split exists because a corpus is hundreds of megabytes on disk and a
    /// grid has tens of points per backend; reading it per point would dominate
    /// cache construction. It is also where a future pre-computed table would
    /// be opened, leaving [`PreparedDemand::sample_and_fold`] unchanged.
    pub fn prepare(&self) -> anyhow::Result<PreparedDemand<'_>> {
        Ok(match self {
            Self::Popularity {
                layerwise_global_ppm,
            } => PreparedDemand::Popularity(layerwise_global_ppm),
            // Loading here rather than in the config keeps the payload out of
            // the cache key; the checksum in the manifest is the identity.
            Self::Corpus(config) => PreparedDemand::Corpus(load_cached(config)?),
        })
    }

    /// Extend a kernel's token axis with the shapes this source makes reachable.
    ///
    /// A block-structured batch only ever presents multiples of the verify
    /// width, and the small end of the axis is where the sources disagree most,
    /// so profiling those points is what keeps the cache from interpolating
    /// across the interesting region.
    pub fn token_axis(&self, base: Vec<f64>) -> Vec<f64> {
        match self {
            Self::Popularity { .. } => base,
            Self::Corpus(config) => super::sweep::Axis::chain([
                base,
                (1..=8)
                    .map(|multiple| f64::from(config.group_size) * f64::from(multiple))
                    .collect(),
            ]),
        }
    }
}

/// A demand source with its payload resolved, ready to fold grid points.
pub enum PreparedDemand<'a> {
    Popularity(&'a [Vec<u32>]),
    Corpus(TokenCorpus),
}

impl PreparedDemand<'_> {
    /// One complete global histogram for `num_tokens`, canonicalized by the
    /// production fold into the vector the kernel cache receives.
    ///
    /// Deterministic: same source, same token count, same vector.
    pub fn sample_and_fold(
        &self,
        top_k: u32,
        num_tokens: u32,
        experts_per_rank: usize,
    ) -> Vec<u32> {
        match self {
            Self::Popularity(layerwise_global_ppm) => sample_and_fold_layerwise_topk_expert_counts(
                layerwise_global_ppm,
                top_k,
                num_tokens,
                experts_per_rank,
                FOLD_SEED,
            ),
            Self::Corpus(corpus) => corpus.sample_and_fold(num_tokens, experts_per_rank),
        }
    }

    /// The per-expert row counts a fused-MoE profiler receives: the folded
    /// histogram rotated so the requested workload rank leads.
    ///
    /// The fold ranks EP shards by active experts then rows, so
    /// `folded_rank_position` selects a representative local histogram —
    /// position 0 is the critical rank — without putting physical rank identity
    /// into the profiler key.
    pub fn per_expert_batches(
        &self,
        top_k: u32,
        num_tokens: u32,
        num_experts: usize,
        num_local_experts: usize,
        folded_rank_position: u32,
    ) -> Vec<u32> {
        assert!(num_local_experts > 0, "num_local_experts must be non-zero");
        assert_eq!(
            num_experts % num_local_experts,
            0,
            "local experts must evenly partition global experts"
        );
        let rank_offset = folded_rank_position as usize * num_local_experts;
        assert!(
            rank_offset + num_local_experts <= num_experts,
            "folded_rank_position must select an EP rank"
        );
        let mut folded = self.sample_and_fold(top_k, num_tokens, num_local_experts);
        assert_eq!(
            folded.len(),
            num_experts,
            "demand source width must match num_experts"
        );
        folded.rotate_left(rank_offset);
        folded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn popularity(num_experts: usize, num_layers: usize) -> ExpertDemand {
        let share = 1_000_000 / num_experts as u32;
        let mut layer = vec![share; num_experts];
        layer[0] += 1_000_000 - share * num_experts as u32;
        ExpertDemand::Popularity {
            layerwise_global_ppm: vec![layer; num_layers],
        }
    }

    #[test]
    fn popularity_folds_every_assignment_and_is_reproducible() {
        let demand = popularity(64, 4);
        let prepared = demand.prepare().unwrap();
        let folded = prepared.sample_and_fold(8, 24, 16);
        assert_eq!(folded.len(), 64);
        assert_eq!(folded.iter().map(|&c| u64::from(c)).sum::<u64>(), 24 * 8);
        assert_eq!(folded, prepared.sample_and_fold(8, 24, 16));
    }

    #[test]
    fn popularity_leaves_the_token_axis_alone() {
        let base = vec![1.0, 4.0, 8.0];
        assert_eq!(popularity(64, 4).token_axis(base.clone()), base);
    }

    #[test]
    fn corpus_widens_the_token_axis_to_verify_width_multiples() {
        let corpus = crate::timing::token_corpus::tests::synthetic(
            &crate::timing::token_corpus::tests::temp_dir("axis"),
            256,
            8,
            4,
            64,
        );
        let axis = ExpertDemand::Corpus(corpus).token_axis(vec![1.0, 32.0]);
        assert_eq!(
            axis[..2],
            [1.0, 32.0],
            "the base axis comes first, in order"
        );
        for multiple in 1..=8 {
            assert!(
                axis.contains(&(8.0 * f64::from(multiple))),
                "{multiple} verify blocks must be profiled"
            );
        }
    }

    /// The recorded GLM-5.3 corpus this work was measured on. Opt-in: it is a
    /// 155 MB artifact that exists only where the capture was taken, so it
    /// would silently pass everywhere else while costing the default gate
    /// tens of seconds. Run with `cargo test -p simulator --lib -- --ignored`.
    ///
    /// The expected values were produced by the implementation this port
    /// replaces, so they pin the sampler's behaviour rather than restating its
    /// formula.
    #[test]
    #[ignore = "needs the recorded 155 MB GLM-5.3 corpus"]
    fn corpus_reproduces_the_folds_the_prototype_measured() {
        const RECORDED: &str = concat!(
            "/raid/kanzhu/ServingStudio/wt-glm53-dflash2/logs/",
            "20260921_0_token_sample/corpus/manifest.json"
        );
        if !std::path::Path::new(RECORDED).is_file() {
            return;
        }
        // Grouped by corpus config so the payload is read once per group.
        // Each case is tokens -> (assignments, active on the critical rank,
        // first eight folded slots).
        let groups = [
            (
                (8, 0xF01D_5EED, 16),
                vec![
                    (8u32, 64u64, 12usize, [5u32, 4, 3, 2, 1, 1, 1, 1]),
                    (24, 192, 28, [7, 5, 4, 4, 3, 3, 3, 2]),
                    (48, 384, 41, [11, 8, 7, 6, 5, 5, 4, 4]),
                ],
            ),
            // Group 1 draws independently -- the popularity path's model -- and
            // visibly hits more experts at the same token count.
            (
                (1, 7, 1),
                vec![
                    (8, 64, 17, [2, 2, 2, 1, 1, 1, 1, 1]),
                    (48, 384, 48, [8, 6, 5, 5, 4, 4, 4, 3]),
                ],
            ),
            (
                (6, 0xF01D_5EED, 4),
                vec![(24, 192, 29, [6, 5, 4, 4, 3, 3, 3, 2])],
            ),
        ];
        for ((group_size, seed, candidates), cases) in groups {
            let mut config = TokenCorpusConfig::from_manifest(RECORDED, group_size, seed, 0..75)
                .expect("recorded corpus manifest loads");
            config.sampling_candidates = candidates;
            let demand = ExpertDemand::Corpus(config);
            let prepared = demand.prepare().expect("recorded corpus payload loads");
            for (tokens, assignments, active, head) in cases {
                let folded = prepared.sample_and_fold(8, tokens, 64);
                let label = format!("group {group_size} seed {seed:#x} at {tokens} tokens");
                assert_eq!(
                    folded.iter().map(|&c| u64::from(c)).sum::<u64>(),
                    assignments,
                    "{label}"
                );
                assert_eq!(
                    folded[..64].iter().filter(|&&c| c > 0).count(),
                    active,
                    "{label}"
                );
                assert_eq!(folded[..8], head, "{label}");
            }
        }
    }
}
