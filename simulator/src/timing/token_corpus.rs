//! Recorded token routes sampled in contiguous groups during cache construction.
use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

use super::routing::fold_layerwise_expert_counts;

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
    #[serde(default = "one")]
    pub group_size: u32,
    #[serde(default)]
    pub seed: u64,
    /// CPU candidates; select one complete folded histogram before profiling.
    #[serde(default = "default_candidates")]
    pub sampling_candidates: u32,
}

fn default_candidates() -> u32 {
    16
}

fn one() -> u32 {
    1
}

pub struct TokenCorpus {
    config: TokenCorpusConfig,
    ids: Vec<u16>,
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

impl TokenCorpusConfig {
    pub fn from_manifest(path: &str, group_size: u32, seed: u64) -> Result<Self> {
        let path = Path::new(path)
            .canonicalize()
            .context("locating token corpus manifest")?;
        let mut config: Self = serde_json::from_slice(&std::fs::read(&path)?)
            .context("reading token corpus manifest")?;
        let data = path
            .parent()
            .unwrap()
            .join(&config.data_file)
            .canonicalize()
            .context("locating token corpus data")?;
        config.data_file = data.to_string_lossy().into_owned();
        config.group_size = group_size;
        config.seed = seed;
        ensure!(
            config.schema_version == 1,
            "unsupported token corpus version"
        );
        ensure!(
            config.sampling_candidates > 0,
            "sampling_candidates must be positive"
        );
        ensure!(group_size > 0, "token corpus group size must be positive");
        ensure!(
            config.num_tokens >= group_size as usize,
            "token corpus is shorter than verify width"
        );
        ensure!(
            config.num_layers > 0
                && config.top_k > 0
                && config.top_k <= config.num_experts
                && config.num_experts <= 65536,
            "invalid token corpus dimensions"
        );
        Ok(config)
    }

    pub fn load(&self) -> Result<TokenCorpus> {
        ensure!(
            self.schema_version == 1
                && self.sampling_candidates > 0
                && self.group_size > 0
                && self.num_tokens >= self.group_size as usize
                && self.num_layers > 0
                && self.top_k > 0
                && self.top_k <= self.num_experts
                && self.num_experts <= 65536,
            "invalid token corpus metadata"
        );
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
            .map(|v| u16::from_le_bytes([v[0], v[1]]))
            .collect();
        for selected in ids.chunks_exact(self.top_k) {
            for (i, &expert) in selected.iter().enumerate() {
                ensure!(
                    (expert as usize) < self.num_experts,
                    "expert ID outside corpus expert range"
                );
                ensure!(
                    !selected[..i].contains(&expert),
                    "duplicate expert in a token's top-k"
                );
            }
        }
        Ok(TokenCorpus {
            config: self.clone(),
            ids,
        })
    }
}

impl TokenCorpus {
    pub fn sample_layer_counts(&self, num_tokens: u32) -> Vec<Vec<u32>> {
        self.sample_layer_counts_with_seed(num_tokens, self.config.seed)
    }

    fn sample_layer_counts_with_seed(&self, num_tokens: u32, seed: u64) -> Vec<Vec<u32>> {
        let c = &self.config;
        let mut counts = vec![vec![0u32; c.num_experts]; c.num_layers];
        let mut state = seed;
        let mut remaining = num_tokens as usize;
        while remaining > 0 {
            let take = remaining.min(c.group_size as usize);
            let bound = (c.num_tokens - take + 1) as u64;
            // Unbiased bounded SplitMix64. Starts are sampled with replacement;
            // windows may cross concatenated request boundaries.
            let threshold = bound.wrapping_neg() % bound;
            let start = loop {
                state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut value = state;
                value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
                value ^= value >> 31;
                if value >= threshold {
                    break (value % bound) as usize;
                }
            };
            for token in start..start + take {
                for (layer, row) in counts.iter_mut().enumerate() {
                    let offset = (token * c.num_layers + layer) * c.top_k;
                    for &expert in &self.ids[offset..offset + c.top_k] {
                        row[expert as usize] += 1;
                    }
                }
            }
            remaining -= take;
        }
        counts
    }

    pub fn sample_and_fold(&self, num_tokens: u32, experts_per_rank: usize) -> Vec<u32> {
        let candidates: Vec<_> = (0..self.config.sampling_candidates)
            .map(|draw| {
                let seed = self
                    .config
                    .seed
                    .wrapping_add(u64::from(draw).wrapping_mul(0xd1b54a32d192ed03));
                fold_layerwise_expert_counts(
                    &self.sample_layer_counts_with_seed(num_tokens, seed),
                    experts_per_rank,
                )
            })
            .collect();
        let selected = median_candidate(&candidates, experts_per_rank);
        candidates.into_iter().nth(selected).unwrap()
    }
}

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
                        rank.iter().filter(|&&n| n > 0).count() as u64,
                        rank.iter().map(|&n| u64::from(n)).sum::<u64>(),
                    )
                })
                .max()
                .expect("at least one rank")
        })
        .collect();
    let n = features.len() as i128;
    features
        .iter()
        .enumerate()
        .map(|(index, &(active, load))| {
            let distance = |value: u64, axis: usize| {
                let values = features.iter().map(|&(a, l)| if axis == 0 { a } else { l });
                let (mut less, mut equal) = (0i128, 0i128);
                for x in values {
                    less += i128::from(x < value);
                    equal += i128::from(x == value);
                }
                let centered = 2 * less + equal - n;
                centered * centered
            };
            (distance(active, 0) + distance(load, 1), index)
        })
        .min()
        .unwrap()
        .1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> TokenCorpus {
        TokenCorpus {
            config: TokenCorpusConfig {
                schema_version: 1,
                data_file: String::new(),
                num_tokens: 8,
                num_layers: 2,
                num_experts: 8,
                top_k: 1,
                checksum_fnv1a64: 0,
                group_size: 8,
                seed: 42,
                sampling_candidates: 16,
            },
            ids: (0..8).flat_map(|i| [i, 7 - i]).collect(),
        }
    }

    #[test]
    fn selection_uses_both_features_and_keeps_a_real_candidate() {
        // Equal active counts: load decides, then stable input order breaks ties.
        let candidates = vec![vec![1, 1], vec![5, 5], vec![2, 2]];
        assert_eq!(median_candidate(&candidates, 2), 2);
        // Equal loads: active count decides.
        let candidates = vec![vec![6, 0, 0], vec![2, 2, 2], vec![3, 3, 0]];
        assert_eq!(median_candidate(&candidates, 3), 2);
        assert_eq!(median_candidate(&[vec![1, 1], vec![1, 1]], 2), 0);
    }

    #[test]
    fn one_candidate_preserves_original_sampling() {
        let mut c = corpus();
        c.config.sampling_candidates = 1;
        assert_eq!(
            c.sample_and_fold(3, 4),
            fold_layerwise_expert_counts(&c.sample_layer_counts(3), 4)
        );
        c.config.sampling_candidates = 16;
        let actual = c.sample_and_fold(3, 4);
        assert_eq!(actual, c.sample_and_fold(3, 4));
        assert!((0..16u64).any(|draw| {
            let seed = c
                .config
                .seed
                .wrapping_add(draw.wrapping_mul(0xd1b54a32d192ed03));
            actual == fold_layerwise_expert_counts(&c.sample_layer_counts_with_seed(3, seed), 4)
        }));
    }

    #[test]
    fn whole_windows_preserve_layers_and_contiguity() {
        let c = corpus();
        assert_eq!(c.sample_layer_counts(16), vec![vec![2; 8]; 2]);
        assert_eq!(c.sample_and_fold(16, 4).iter().sum::<u32>(), 16);
        let counts = c.sample_layer_counts(3);
        assert_eq!(counts[0].iter().sum::<u32>(), 3);
        assert_eq!(
            counts[0].iter().rev().copied().collect::<Vec<_>>(),
            counts[1]
        );
        let occupied: Vec<_> = counts[0]
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(occupied[2] - occupied[0], 2);
        assert_eq!(counts, c.sample_layer_counts(3));
    }

    #[test]
    fn manifest_roundtrip_rejects_changed_data_and_duplicate_routes() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("routes.u16");
        let bytes: Vec<_> = corpus().ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&data, &bytes).unwrap();
        let mut c = corpus().config;
        c.data_file = "routes.u16".into();
        c.checksum_fnv1a64 = checksum(&bytes);
        let manifest = dir.path().join("manifest.json");
        std::fs::write(&manifest, serde_json::to_vec(&c).unwrap()).unwrap();
        let c = TokenCorpusConfig::from_manifest(manifest.to_str().unwrap(), 8, 42).unwrap();
        assert!(c.load().is_ok());
        std::fs::write(&data, vec![0; bytes.len()]).unwrap();
        assert!(c.load().is_err());
        let mut duplicate = c.clone();
        duplicate.top_k = 2;
        duplicate.num_tokens = 4;
        duplicate.checksum_fnv1a64 = checksum(&vec![0; bytes.len()]);
        assert!(duplicate.load().is_err());
        assert!(TokenCorpusConfig::from_manifest(manifest.to_str().unwrap(), 9, 42).is_err());
    }
}
