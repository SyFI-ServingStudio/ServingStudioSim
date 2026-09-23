//! Recorded per-token expert routes, sampled in contiguous groups.
//!
//! A full-run popularity marginal can only be resampled independently, and a
//! serving batch is not independent: with speculative decoding one sequence
//! submits `draft_tokens + 1` consecutive positions that route almost alike.
//! This corpus keeps whole tokens, so a draw can reproduce that correlation by
//! taking contiguous windows the width of the verify block.
use std::path::Path;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use super::routing::fold_layerwise_expert_counts;

/// Corpus identity plus the sampling parameters a deployment fixes. Small and
/// hashable on purpose: this is what reaches a kernel config, while the payload
/// stays on disk until a cache is built.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenCorpusConfig {
    pub schema_version: u32,
    pub data_file: String,
    pub num_tokens: usize,
    pub num_layers: usize,
    pub num_experts: usize,
    pub top_k: usize,
    /// FNV-1a of the little-endian u16 payload; detects changed corpus bytes.
    pub checksum_fnv1a64: u64,
    /// Consecutive recorded positions per draw. This is the deployment's verify
    /// width, not a tuning knob: sampling wider or narrower than the batch's
    /// block structure is what the corpus exists to avoid.
    #[serde(default = "one")]
    pub group_size: u32,
    /// Layers this binding folds over, as a half-open range into the
    /// artifact's layer axis. A capture covers every routed layer of the
    /// deployment at once, so a body MoE and an MTP MoE are two slices of one
    /// corpus rather than two artifacts — which is what a per-expert marginal
    /// can never be, having already summed the layer axis away.
    #[serde(default)]
    pub layer_start: usize,
    #[serde(default)]
    pub layer_end: usize,
    #[serde(default)]
    pub seed: u64,
    /// Candidate draws to choose between. One *real* draw is selected; an
    /// average of histograms is a histogram no step ever had, and it inflates
    /// the active-group count the same way assuming balanced routing does.
    #[serde(default = "default_candidates")]
    pub sampling_candidates: u32,
}

fn default_candidates() -> u32 {
    16
}

fn one() -> u32 {
    1
}

/// A validated corpus with its payload resident.
pub struct TokenCorpus {
    config: TokenCorpusConfig,
    /// Shared so a second callable of the same corpus -- the MTP layer, or the
    /// same arch under another backend -- binds its own sampling parameters to
    /// bytes that are already read and already validated.
    ids: Arc<Vec<u16>>,
}

/// Summarised rather than derived: `ids` holds one entry per recorded
/// (token, layer, slot) and runs to tens of millions.
impl std::fmt::Debug for TokenCorpus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenCorpus")
            .field("config", &self.config)
            .field("expert_ids", &self.ids.len())
            .finish()
    }
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

impl TokenCorpusConfig {
    /// Read a manifest and bind it to one deployment's sampling parameters.
    /// `data_file` is rewritten to an absolute path so the config stays valid
    /// however the process later moves.
    ///
    /// `layers` is required rather than defaulted to the whole artifact: which
    /// layers a MoE callable covers is the caller's fact, and a silent whole-
    /// corpus default would price an MTP layer with the body's routing.
    pub fn from_manifest(
        path: &str,
        group_size: u32,
        seed: u64,
        layers: std::ops::Range<usize>,
    ) -> Result<Self> {
        let path = Path::new(path);
        // The manifest's own directory, canonicalized *without* resolving the
        // manifest itself. A hub snapshot links `snapshots/<sha>/manifest.json`
        // at a content-addressed blob, so canonicalizing the file would put the
        // payload lookup in the blob store, where nothing named `routes.u16`
        // exists. `data_file` is relative to where the manifest is published,
        // not to where its bytes happen to live.
        let base = match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            _ => Path::new("."),
        }
        .canonicalize()
        .with_context(|| format!("locating token corpus manifest directory for {path:?}"))?;
        let mut config: Self = serde_json::from_slice(
            &std::fs::read(path)
                .with_context(|| format!("reading token corpus manifest {path:?}"))?,
        )
        .context("parsing token corpus manifest")?;
        let data = base
            .join(&config.data_file)
            .canonicalize()
            .with_context(|| format!("locating token corpus data {:?}", config.data_file))?;
        config.data_file = data.to_string_lossy().into_owned();
        config.group_size = group_size;
        config.seed = seed;
        config.layer_start = layers.start;
        config.layer_end = layers.end;
        config.validate()?;
        Ok(config)
    }

    /// Layers the folded histogram averages over.
    fn layers(&self) -> std::ops::Range<usize> {
        self.layer_start..self.layer_end
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.schema_version == 1, "unsupported token corpus version");
        ensure!(
            self.sampling_candidates > 0,
            "sampling_candidates must be positive"
        );
        ensure!(
            self.group_size > 0,
            "token corpus group size must be positive"
        );
        ensure!(
            self.num_tokens >= self.group_size as usize,
            "token corpus is shorter than one sampling group"
        );
        ensure!(
            self.layer_start < self.layer_end && self.layer_end <= self.num_layers,
            "token corpus layer slice {}..{} is not inside its {} layers",
            self.layer_start,
            self.layer_end,
            self.num_layers
        );
        ensure!(
            self.num_layers > 0
                && self.top_k > 0
                && self.top_k <= self.num_experts
                && self.num_experts <= 65536,
            "invalid token corpus dimensions"
        );
        Ok(())
    }

    /// Read and verify the payload. Both the byte length and the checksum are
    /// checked, so a corpus that changed under a cached config fails loudly
    /// rather than silently shifting every profiled shape.
    pub fn load(&self) -> Result<TokenCorpus> {
        self.validate()?;
        let bytes = std::fs::read(&self.data_file).context("reading token corpus data")?;
        let expected = self
            .num_tokens
            .checked_mul(self.num_layers)
            .and_then(|n| n.checked_mul(self.top_k))
            .and_then(|n| n.checked_mul(2))
            .context("token corpus size overflow")?;
        ensure!(
            bytes.len() == expected,
            "token corpus byte length does not match metadata"
        );
        ensure!(
            checksum(&bytes) == self.checksum_fnv1a64,
            "token corpus checksum mismatch"
        );
        let ids: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        for selected in ids.chunks_exact(self.top_k) {
            for (position, &expert) in selected.iter().enumerate() {
                ensure!(
                    (expert as usize) < self.num_experts,
                    "expert ID outside corpus expert range"
                );
                ensure!(
                    !selected[..position].contains(&expert),
                    "duplicate expert in a token's top-k"
                );
            }
        }
        Ok(TokenCorpus {
            config: self.clone(),
            ids: Arc::new(ids),
        })
    }
}

impl TokenCorpus {
    /// The validated payload, for a caller that keeps it across configs.
    pub(super) fn payload(&self) -> Arc<Vec<u16>> {
        Arc::clone(&self.ids)
    }

    /// The same payload under another config's sampling parameters.
    ///
    /// Only the group size, seed and layer slice differ between callables of
    /// one corpus; the dimensions and checksum are the corpus's identity, and
    /// `config.validate()` re-checks the slice against them.
    pub(super) fn rebind(config: &TokenCorpusConfig, ids: Arc<Vec<u16>>) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config: config.clone(),
            ids,
        })
    }

    /// One complete global histogram per layer for `num_tokens` sampled tokens,
    /// over the config's layer slice.
    ///
    /// Tokens arrive in runs of `group_size`: a decode step of N tokens is
    /// `⌈N/w⌉` sequences contributing w consecutive positions each, so the draw
    /// takes that many independent windows. All layers share the chosen
    /// positions, because a token's routing is correlated across layers too —
    /// which is also why slicing happens here and not in the packer: the body
    /// and MTP slices of one draw must see the same tokens.
    fn sample_layer_counts(&self, num_tokens: u32, seed: u64) -> Vec<Vec<u32>> {
        let config = &self.config;
        let layers = config.layers();
        let mut counts = vec![vec![0u32; config.num_experts]; layers.len()];
        let mut state = seed;
        let mut remaining = num_tokens as usize;
        while remaining > 0 {
            let take = remaining.min(config.group_size as usize);
            let bound = (config.num_tokens - take + 1) as u64;
            // Unbiased bounded SplitMix64. Starts are drawn with replacement,
            // and a window may cross the seam between two concatenated
            // requests; both are properties of the recorded corpus, not of the
            // sampler, and the packer is what decides how many seams exist.
            let threshold = bound.wrapping_neg() % bound;
            let start = loop {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut value = state;
                value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                value ^= value >> 31;
                if value >= threshold {
                    break (value % bound) as usize;
                }
            };
            for token in start..start + take {
                for (layer, row) in layers.clone().zip(counts.iter_mut()) {
                    let offset = (token * config.num_layers + layer) * config.top_k;
                    for &expert in &self.ids[offset..offset + config.top_k] {
                        row[expert as usize] += 1;
                    }
                }
            }
            remaining -= take;
        }
        counts
    }

    /// Draw `sampling_candidates` complete folded histograms and return the
    /// most central *real* one. Deterministic in the config's seed.
    pub fn sample_and_fold(&self, num_tokens: u32, experts_per_rank: usize) -> Vec<u32> {
        let candidates: Vec<_> = (0..self.config.sampling_candidates)
            .map(|draw| {
                let seed = self
                    .config
                    .seed
                    .wrapping_add(u64::from(draw).wrapping_mul(CANDIDATE_STRIDE));
                fold_layerwise_expert_counts(
                    &self.sample_layer_counts(num_tokens, seed),
                    experts_per_rank,
                )
            })
            .collect();
        let selected = median_candidate(&candidates, experts_per_rank);
        candidates
            .into_iter()
            .nth(selected)
            .expect("median_candidate returns an in-range index")
    }
}

/// Decorrelates the candidate seeds without needing a second RNG.
const CANDIDATE_STRIDE: u64 = 0xd1b5_4a32_d192_ed03;

/// Choose a real candidate, never an average of histograms. Both features come
/// from the same heaviest rank (active count first, assignments second).
/// Midrank percentile distances have a common denominator, so integer scores
/// give exact, deterministic comparisons even when many candidates tie.
fn median_candidate(candidates: &[Vec<u32>], experts_per_rank: usize) -> usize {
    assert!(!candidates.is_empty() && experts_per_rank > 0);
    let features: Vec<_> = candidates
        .iter()
        .map(|counts| {
            assert_eq!(counts.len() % experts_per_rank, 0);
            counts
                .chunks_exact(experts_per_rank)
                .map(|rank| {
                    (
                        rank.iter().filter(|&&count| count > 0).count() as u64,
                        rank.iter().map(|&count| u64::from(count)).sum::<u64>(),
                    )
                })
                .max()
                .expect("at least one rank")
        })
        .collect();
    let count = features.len() as i128;
    features
        .iter()
        .enumerate()
        .map(|(index, &(active, load))| {
            let distance = |value: u64, axis: usize| {
                let values = features.iter().map(|&(a, l)| if axis == 0 { a } else { l });
                let (mut less, mut equal) = (0i128, 0i128);
                for other in values {
                    less += i128::from(other < value);
                    equal += i128::from(other == value);
                }
                let centered = 2 * less + equal - count;
                centered * centered
            };
            (distance(active, 0) + distance(load, 1), index)
        })
        .min()
        .expect("candidates is non-empty")
        .1
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "token-corpus-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// A corpus whose token `t` selects experts `t, t+1, ... t+top_k-1` mod
    /// `num_experts`, offset per layer. Consecutive tokens therefore overlap
    /// heavily and distant ones do not, which is the property a contiguous-group
    /// sampler exists to exploit.
    pub(crate) fn synthetic(
        dir: &std::path::Path,
        num_experts: usize,
        top_k: usize,
        num_layers: usize,
        num_tokens: usize,
    ) -> TokenCorpusConfig {
        let mut ids: Vec<u16> = Vec::with_capacity(num_tokens * num_layers * top_k);
        for token in 0..num_tokens {
            for layer in 0..num_layers {
                for slot in 0..top_k {
                    ids.push(((token + layer * 17 + slot) % num_experts) as u16);
                }
            }
        }
        let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        std::fs::write(dir.join("routes.u16"), &bytes).expect("corpus payload");
        let manifest = serde_json::json!({
            "schema_version": 1,
            "data_file": "routes.u16",
            "num_tokens": num_tokens,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "checksum_fnv1a64": checksum(&bytes),
        });
        let path = dir.join("manifest.json");
        std::fs::write(&path, manifest.to_string()).expect("corpus manifest");
        TokenCorpusConfig::from_manifest(path.to_str().unwrap(), 8, 0, 0..num_layers)
            .expect("manifest loads")
    }

    #[test]
    fn a_manifest_reached_through_a_symlink_still_finds_its_payload() {
        // The hub's cache layout: a revision directory of symlinks into a
        // content-addressed blob store. Resolving the manifest before looking
        // for `data_file` would search the blob store, where the payload has no
        // name of its own.
        let dir = temp_dir("hub-layout");
        let blobs = dir.join("blobs");
        let snapshot = dir.join("snapshots").join("0123456789abcdef");
        std::fs::create_dir_all(&blobs).expect("blob store");
        std::fs::create_dir_all(&snapshot).expect("snapshot");
        let config = synthetic(&blobs, 64, 8, 4, 128);
        for (blob, name) in [
            ("routes.u16", "routes.u16"),
            ("manifest.json", "manifest.json"),
        ] {
            let link = snapshot.join(name);
            let _ = std::fs::remove_file(&link);
            std::os::unix::fs::symlink(blobs.join(blob), &link).expect("hub symlink");
        }

        let resolved = TokenCorpusConfig::from_manifest(
            snapshot.join("manifest.json").to_str().unwrap(),
            8,
            7,
            0..4,
        )
        .expect("a snapshot manifest resolves its payload");
        assert_eq!(resolved.num_tokens, config.num_tokens);
        resolved.load().expect("the payload behind the link loads");
    }

    #[test]
    fn a_changed_payload_fails_the_checksum() {
        let dir = temp_dir("checksum");
        let config = synthetic(&dir, 64, 8, 4, 128);
        config.load().expect("the untouched corpus loads");

        let mut bytes = std::fs::read(&config.data_file).unwrap();
        bytes[0] ^= 1;
        std::fs::write(&config.data_file, &bytes).unwrap();
        let error = config.load().expect_err("a changed payload must be caught");
        assert!(format!("{error:#}").contains("checksum"));
    }

    #[test]
    fn a_truncated_payload_fails_before_the_checksum() {
        let dir = temp_dir("truncated");
        let config = synthetic(&dir, 64, 8, 4, 128);
        let bytes = std::fs::read(&config.data_file).unwrap();
        std::fs::write(&config.data_file, &bytes[..bytes.len() - 2]).unwrap();
        let error = config.load().expect_err("a short payload must be caught");
        assert!(format!("{error:#}").contains("byte length"));
    }

    #[test]
    fn a_group_longer_than_the_corpus_is_rejected() {
        let dir = temp_dir("short");
        let mut config = synthetic(&dir, 64, 8, 4, 128);
        config.group_size = 129;
        let error = config.load().expect_err("no window of that width exists");
        assert!(format!("{error:#}").contains("shorter than one sampling group"));
    }

    #[test]
    fn every_assignment_is_folded_and_the_draw_is_reproducible() {
        let dir = temp_dir("fold");
        let corpus = synthetic(&dir, 64, 8, 4, 256).load().unwrap();
        let folded = corpus.sample_and_fold(24, 16);
        assert_eq!(folded.len(), 64);
        assert_eq!(folded.iter().map(|&c| u64::from(c)).sum::<u64>(), 24 * 8);
        assert_eq!(folded, corpus.sample_and_fold(24, 16));
    }

    /// The unification this replaces two popularity files with: one artifact,
    /// two callables, and the last layer really does route differently from the
    /// body — so a slice is not a relabelling of the same histogram.
    #[test]
    fn a_layer_slice_folds_only_its_own_layers() {
        let dir = temp_dir("slice");
        let whole = synthetic(&dir, 64, 8, 4, 256);
        let slice = |layers: std::ops::Range<usize>| {
            let mut config = whole.clone();
            config.layer_start = layers.start;
            config.layer_end = layers.end;
            config.load().unwrap().sample_and_fold(24, 16)
        };
        let last = slice(3..4);
        assert_eq!(last.iter().map(|&c| u64::from(c)).sum::<u64>(), 24 * 8);
        assert_ne!(last, slice(0..3), "the MTP slice is not the body's fold");
        assert_ne!(last, slice(0..4));
    }

    #[test]
    fn a_layer_slice_outside_the_corpus_is_rejected() {
        let dir = temp_dir("slice-range");
        let mut config = synthetic(&dir, 64, 8, 4, 128);
        config.layer_end = 5;
        let error = config.load().expect_err("layer 4 was never recorded");
        assert!(format!("{error:#}").contains("layer slice"));
    }

    #[test]
    fn contiguous_groups_touch_fewer_experts_than_independent_draws() {
        let dir = temp_dir("groups");
        let mut config = synthetic(&dir, 256, 8, 4, 4096);
        config.sampling_candidates = 8;

        let active = |group_size: u32| {
            let mut config = config.clone();
            config.group_size = group_size;
            let folded = config.load().unwrap().sample_and_fold(32, 64);
            folded[..64].iter().filter(|&&count| count > 0).count()
        };
        // The whole thesis in one assertion: a marginal cannot express this,
        // because resampling one is exactly the group-of-one case.
        assert!(
            active(8) < active(1),
            "groups of 8 hit {} experts, independent draws {}",
            active(8),
            active(1)
        );
    }
}
