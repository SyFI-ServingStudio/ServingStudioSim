#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct RoutingDistribution {
    ppm: Vec<u32>,
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

#[cfg(test)]
mod tests {
    use crate::timing::routing::RoutingDistribution;

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
}
