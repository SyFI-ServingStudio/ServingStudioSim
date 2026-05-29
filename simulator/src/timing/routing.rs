#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct RoutingDistribution {
    ppm: Vec<u32>,
}

/// Split `ep_size` ranks into contiguous NVL domains of at most `nvl_num_gpu`
/// ranks each (the last domain may be smaller). `nvl_num_gpu == 0` or
/// `>= ep_size` collapses to a single domain. Mirrors ref `ranks_per_nvl_domain`
/// so the L2 NVL layout and the per-token router agree on domain boundaries.
pub fn ranks_per_nvl_domain(ep_size: u32, nvl_num_gpu: u32) -> Vec<u32> {
    if ep_size == 0 {
        return Vec::new();
    }
    if nvl_num_gpu == 0 || nvl_num_gpu >= ep_size {
        return vec![ep_size];
    }
    let mut remaining = ep_size;
    let mut counts = Vec::new();
    while remaining > 0 {
        let count = remaining.min(nvl_num_gpu);
        counts.push(count);
        remaining -= count;
    }
    counts
}

impl RoutingDistribution {
    pub const TOTAL_PPM: u32 = 1_000_000;

    pub fn from_profile(per_expert_ratios: &[f32]) -> Self {
        if per_expert_ratios.is_empty() {
            return Self { ppm: Vec::new() };
        }
        let weights: Vec<f64> = per_expert_ratios
            .iter()
            .map(|ratio| f64::from((*ratio).max(0.0)))
            .collect();
        Self::from_weights(&weights)
    }

    pub fn uniform(num_experts: u32) -> Self {
        if num_experts == 0 {
            return Self { ppm: Vec::new() };
        }
        let base = Self::TOTAL_PPM / num_experts;
        let remainder = Self::TOTAL_PPM % num_experts;
        let mut ppm = vec![base; num_experts as usize];
        for ppm_slot in ppm.iter_mut().take(remainder as usize) {
            *ppm_slot += 1;
        }
        Self { ppm }
    }

    pub fn power_law(num_experts: u32, alpha: f32) -> Self {
        if num_experts == 0 {
            return Self { ppm: Vec::new() };
        }
        let weights: Vec<f64> = (0..num_experts)
            .map(|rank| 1.0 / f64::from(rank + 1).powf(f64::from(alpha.max(0.0))))
            .collect();
        Self::from_weights(&weights)
    }

    pub fn num_experts(&self) -> u32 {
        self.ppm.len() as u32
    }

    pub fn ppm(&self) -> &[u32] {
        &self.ppm
    }

    /// Largest-remainder (Hamilton) apportionment of `global_expert_selections` across
    /// the experts described by `ppm`, one integer count per `ppm` entry.
    ///
    /// `ppm` is a slice of per-expert ppm values that **need not sum to**
    /// [`Self::TOTAL_PPM`] — this is the EP-sharding case. Pass the full global
    /// distribution (`Σ == TOTAL_PPM`, from [`Self::ppm`]) to split across all
    /// experts, or pass just the slice of values for one GPU's local experts
    /// (`Σ < TOTAL_PPM`) to get that GPU's per-local-expert counts. Returned
    /// counts sum to `round(global_expert_selections × Σ ppm / TOTAL_PPM)` — the
    /// shard's proportional share, **not** `global_expert_selections`.
    ///
    /// `global_expert_selections` is the post-top-k count (token-expert assignments):
    /// the `× top_k` expansion and the global→GPU expert mapping both happen
    /// upstream (routing layer), not here. Per-shard rounding is independent, so
    /// summed across all shards the counts match `global_expert_selections` only within
    /// a ±0.5-per-shard drift — acceptable and intended for sharded EP.
    ///
    /// Associated (not `&self`) so the caller passes whatever ppm slice it
    /// wants: `RoutingDistribution::to_per_expert_counts(total, &dist.ppm()[lo..hi])`.
    pub fn to_per_expert_counts(global_expert_selections: u32, ppm: &[u32]) -> Vec<u32> {
        let mut counts = Vec::with_capacity(ppm.len());
        let mut residuals = Vec::with_capacity(ppm.len());
        let mut assigned: u64 = 0;
        let mut numerator_sum: u128 = 0;
        for (idx, slot) in ppm.iter().copied().enumerate() {
            debug_assert!(
                slot <= Self::TOTAL_PPM,
                "ppm entry {slot} exceeds TOTAL_PPM"
            );
            let numerator = u64::from(global_expert_selections) * u64::from(slot);
            let base = (numerator / u64::from(Self::TOTAL_PPM)) as u32;
            counts.push(base);
            residuals.push((idx, numerator % u64::from(Self::TOTAL_PPM)));
            assigned += u64::from(base);
            numerator_sum += u128::from(numerator);
        }
        // Target = round(Σ exact shares) = round(total × Σppm / TOTAL_PPM). For
        // a shard (Σppm < TOTAL_PPM) this is the shard's proportional count, not
        // the global total; for a full distribution it equals global_expert_selections.
        let total_ppm = u128::from(Self::TOTAL_PPM);
        let target = ((numerator_sum + total_ppm / 2) / total_ppm) as u64;
        let deficit = target.saturating_sub(assigned);
        residuals.sort_by(|lhs, rhs| rhs.1.cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
        for (idx, _) in residuals.into_iter().take(deficit as usize) {
            counts[idx] += 1;
        }
        counts
    }

    fn from_weights(weights: &[f64]) -> Self {
        let total: f64 = weights.iter().sum();
        if total <= f64::EPSILON {
            return Self::uniform(weights.len() as u32);
        }

        let mut ppm = Vec::with_capacity(weights.len());
        let mut residuals = Vec::with_capacity(weights.len());
        let mut assigned = 0u32;
        for (idx, weight) in weights.iter().copied().enumerate() {
            let raw = weight / total * f64::from(Self::TOTAL_PPM);
            let base = raw.floor() as u32;
            ppm.push(base);
            residuals.push((idx, raw - f64::from(base)));
            assigned += base;
        }
        residuals.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
        for (idx, _) in residuals
            .into_iter()
            .take((Self::TOTAL_PPM - assigned) as usize)
        {
            ppm[idx] += 1;
        }
        Self { ppm }
    }
}

/// Balanced contiguous split of `num_experts` across `buckets` ranks: the first
/// `num_experts % buckets` ranks get `⌈E/buckets⌉`, the rest `⌊E/buckets⌋`.
/// Mirrors ref `balanced_counts`.
pub fn balanced_expert_counts(num_experts: u32, buckets: u32) -> Vec<u32> {
    if buckets == 0 {
        return Vec::new();
    }
    let base = num_experts / buckets;
    let rem = num_experts % buckets;
    (0..buckets).map(|idx| base + u32::from(idx < rem)).collect()
}

/// Deterministic splitmix64 — the RNG for build-time Monte-Carlo routing. No
/// transcendental ops, fixed iteration order ⇒ bit-identical across
/// runs/machines, so the simulator stays reproducible.
pub(crate) struct RoutingRng(u64);

impl RoutingRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// Drive weighted-WITHOUT-replacement top_k routing for `n_tokens` tokens and
/// invoke `on_token` once per token with that token's realized hit state:
/// `hit_rank[r]` / `hit_dom[d]` are the per-rank / per-domain DISTINCT-hit masks
/// (deduped — a token hitting a rank's experts twice still flags it once).
/// Experts map to ranks via [`balanced_expert_counts`], ranks to domains via
/// [`ranks_per_nvl_domain`].
///
/// This is the single source of the per-token routing law: the MoE comm
/// simulator (`op::moe::sim`) consumes each token's hit set to price all six
/// dispatch/combine stages' per-GPU bytes, so the sampling stays bit-identical
/// (splitmix64 + fixed iteration order, only +/*/< on f64).
pub(crate) fn for_each_routed_token(
    ppm: &[u32],
    top_k: u32,
    ep_size: u32,
    nvl_num_gpu: u32,
    n_tokens: u32,
    rng: &mut RoutingRng,
    mut on_token: impl FnMut(&[bool], &[bool]),
) {
    let e = ppm.len();
    let counts = balanced_expert_counts(e as u32, ep_size);
    let mut expert_rank = vec![0usize; e];
    let mut idx = 0usize;
    for (rank, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            expert_rank[idx] = rank;
            idx += 1;
        }
    }
    let ranks_per_domain = ranks_per_nvl_domain(ep_size, nvl_num_gpu);
    let mut rank_domain = vec![0usize; ep_size as usize];
    let mut rr = 0usize;
    for (d, &dr) in ranks_per_domain.iter().enumerate() {
        for _ in 0..dr {
            rank_domain[rr] = d;
            rr += 1;
        }
    }
    let base_w: Vec<f64> = ppm.iter().map(|&p| f64::from(p)).collect();
    let k = top_k.min(e as u32);
    let mut w = base_w.clone();
    let mut hit_rank = vec![false; ep_size as usize];
    let mut hit_dom = vec![false; ranks_per_domain.len()];
    for _ in 0..n_tokens {
        w.clone_from(&base_w); // reuse allocation across tokens
        hit_rank.iter_mut().for_each(|h| *h = false);
        hit_dom.iter_mut().for_each(|h| *h = false);
        let mut total: f64 = w.iter().sum();
        for _ in 0..k {
            if total <= 0.0 {
                break;
            }
            let u = rng.next_f64() * total;
            let mut acc = 0.0;
            let mut chosen = e - 1;
            for (i, &wi) in w.iter().enumerate() {
                acc += wi;
                if u < acc {
                    chosen = i;
                    break;
                }
            }
            let rank = expert_rank[chosen];
            hit_rank[rank] = true;
            hit_dom[rank_domain[rank]] = true;
            total -= w[chosen];
            w[chosen] = 0.0;
        }
        on_token(&hit_rank, &hit_dom);
    }
}

#[cfg(test)]
mod tests {
    use crate::timing::routing::{
        balanced_expert_counts, for_each_routed_token, ranks_per_nvl_domain, RoutingDistribution,
        RoutingRng,
    };

    #[test]
    fn routing_distribution_normalizes_and_allocates_counts() {
        let dist = RoutingDistribution::from_profile(&[0.1, 0.2, 0.7]);
        assert_eq!(dist.num_experts(), 3);
        assert_eq!(
            dist.ppm().iter().sum::<u32>(),
            RoutingDistribution::TOTAL_PPM
        );

        // Full distribution (Σ ppm == TOTAL_PPM): counts sum to total exactly.
        let counts = RoutingDistribution::to_per_expert_counts(10, dist.ppm());
        assert_eq!(counts.iter().sum::<u32>(), 10);
        assert!(counts[2] >= counts[1]);
        assert!(counts[1] >= counts[0]);
    }

    #[test]
    fn uniform_handles_remainder_deterministically() {
        let dist = RoutingDistribution::uniform(3);
        assert_eq!(dist.ppm(), &[333_334, 333_333, 333_333]);
    }

    #[test]
    fn to_per_expert_counts_shards_subset_proportionally() {
        // Global 4-expert dist: ppm ≈ [400k, 300k, 200k, 100k] (Σ == 1e6).
        let dist = RoutingDistribution::from_profile(&[0.4, 0.3, 0.2, 0.1]);
        let full = dist.ppm();

        // Shard = experts {1, 2}: ppm 300k + 200k = 500k (50% of global mass).
        let shard = &full[1..3];
        let counts = RoutingDistribution::to_per_expert_counts(100, shard);

        // 100 selections × 50% shard mass → 50, NOT 100.
        assert_eq!(counts.iter().sum::<u32>(), 50);
        assert_eq!(counts.len(), 2);
        assert!(counts[0] >= counts[1]); // expert1 (300k) ≥ expert2 (200k)
    }

    #[test]
    fn to_per_expert_counts_shard_rounds_proportional_total() {
        // 50% shard of 7 selections → round(3.5) == 4, distributed by remainder.
        let dist = RoutingDistribution::from_profile(&[0.4, 0.3, 0.2, 0.1]);
        let shard = &dist.ppm()[1..3]; // 300k + 200k = 500k
        let counts = RoutingDistribution::to_per_expert_counts(7, shard);
        assert_eq!(counts.iter().sum::<u32>(), 4);
    }

    #[test]
    fn ranks_per_nvl_domain_chunks_and_collapses() {
        assert_eq!(ranks_per_nvl_domain(8, 4), vec![4, 4]);
        assert_eq!(ranks_per_nvl_domain(10, 4), vec![4, 4, 2]); // ragged last
        assert_eq!(ranks_per_nvl_domain(8, 0), vec![8]); // 0 → single domain
        assert_eq!(ranks_per_nvl_domain(8, 16), vec![8]); // nvl ≥ ep → single
        assert_eq!(ranks_per_nvl_domain(0, 4), Vec::<u32>::new());
    }

    #[test]
    fn balanced_expert_counts_splits_remainder_to_front() {
        assert_eq!(balanced_expert_counts(8, 4), vec![2, 2, 2, 2]);
        assert_eq!(balanced_expert_counts(10, 4), vec![3, 3, 2, 2]); // rem=2 → front
        assert_eq!(balanced_expert_counts(3, 4), vec![1, 1, 1, 0]);
        assert!(balanced_expert_counts(8, 0).is_empty());
    }

    /// `for_each_routed_token` is the bit-identical routing core every comm
    /// stage prices against. Two runs with the same seed must produce the same
    /// per-token hit sets, and each token must flag exactly its distinct hit
    /// ranks/domains (≤ top_k ranks, each in its domain).
    #[test]
    fn for_each_routed_token_is_deterministic_and_consistent() {
        let dist = RoutingDistribution::power_law(64, 1.0);
        let (top_k, ep, nvl, n) = (4u32, 16u32, 8u32, 200u32);
        let n_domains = ranks_per_nvl_domain(ep, nvl).len();

        let collect = || {
            let mut rng = RoutingRng::new(0xD1CE_5EED);
            let mut log: Vec<(Vec<bool>, Vec<bool>)> = Vec::new();
            for_each_routed_token(dist.ppm(), top_k, ep, nvl, n, &mut rng, |hr, hd| {
                log.push((hr.to_vec(), hd.to_vec()));
            });
            log
        };
        let run_a = collect();
        let run_b = collect();
        assert_eq!(run_a, run_b, "same seed must replay identical hit sets");

        for (hit_rank, hit_dom) in &run_a {
            assert_eq!(hit_rank.len(), ep as usize);
            assert_eq!(hit_dom.len(), n_domains);
            let n_hit_ranks = hit_rank.iter().filter(|&&h| h).count();
            assert!(n_hit_ranks >= 1 && n_hit_ranks <= top_k as usize);
            // Every flagged rank's domain must also be flagged; no spurious domains.
            let ranks_per_dom = ranks_per_nvl_domain(ep, nvl);
            let mut dom_of = vec![0usize; ep as usize];
            let mut r = 0usize;
            for (d, &dr) in ranks_per_dom.iter().enumerate() {
                for _ in 0..dr {
                    dom_of[r] = d;
                    r += 1;
                }
            }
            for (rank, &hit) in hit_rank.iter().enumerate() {
                if hit {
                    assert!(hit_dom[dom_of[rank]], "hit rank's domain must be flagged");
                }
            }
        }
    }
}
