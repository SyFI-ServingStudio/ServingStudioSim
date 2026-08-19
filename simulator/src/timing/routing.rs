#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct RoutingDistribution {
    ppm: Vec<u32>,
}

/// Split `ep_size` ranks into contiguous NVL domains of at most `nvl_num_gpu`
/// ranks each (the last domain may be smaller). `nvl_num_gpu == 0` or
/// `>= ep_size` collapses to a single domain. Mirrors ref `ranks_per_nvl_domain`
/// so the L2 NVL layout and the per-token router agree on domain boundaries.
/// `base^exp` by binary exponentiation over `f64` multiplies only.
///
/// IEEE-754 fully specifies multiplication, so this is bit-identical on every
/// host; `f64::powf` is not, and its result feeds a cache key here.
fn pow_exact(base: f64, exp: u32) -> f64 {
    let mut result = 1.0f64;
    let mut factor = base;
    let mut remaining = exp;
    while remaining > 0 {
        if remaining & 1 == 1 {
            result *= factor;
        }
        factor *= factor;
        remaining >>= 1;
    }
    result
}

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

    /// A seeded pseudo-random expert skew: draw one weight per expert from the
    /// deterministic splitmix64 stream and normalize (the same `from_weights`
    /// path as `power_law` / `from_profile`). Uses only `+`/`*`/`>>` and one
    /// division (no transcendental ops), so a fixed `seed` yields a bit-identical
    /// ppm across runs and machines — a `routing = random` run stays reproducible
    /// (the throughput golden is bit-identical). Vary `seed` to sample a
    /// different skew.
    pub fn random(num_experts: u32, seed: u64) -> Self {
        if num_experts == 0 {
            return Self { ppm: Vec::new() };
        }
        let mut rng = RoutingRng::new(seed);
        let weights: Vec<f64> = (0..num_experts).map(|_| rng.next_f64()).collect();
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
    /// The apportionment is restricted to an **active set** whose size is the
    /// expected number of experts that receive at least one assignment; see
    /// [`Self::active_set_len`]. Spreading the mass over every expert
    /// instead — giving each its expected count — is what a Hamilton
    /// apportionment does by construction, and it is wrong here: assigning every
    /// expert its mean maximizes the number of non-empty cells, while a real
    /// step's routing leaves a long tail empty. The grouped-GEMM kernels pad per
    /// expert, so non-empty cells *are* the cost, and the difference is
    /// measurable rather than cosmetic. Against a vLLM Qwen3.6-35B-A3B-FP8
    /// decode step (64 tokens, top-8, 256 experts, EP1) the full-spread form put
    /// 219 experts on the padded-block grid where vLLM used 173, and the same
    /// `vllm_fused_moe` kernel measured +18.5% (gate/up) and +21.6% (down) on
    /// that vector versus −3.0% / +2.8% on this one.
    ///
    /// The correction shrinks as the shard does, because it only exists when a
    /// shard holds many more experts than it receives assignments: at EP≥8 on
    /// that model every local expert is hit with near-certainty and the active
    /// set is the whole shard, leaving the result within 0.6% of the old
    /// behavior.
    ///
    /// v1 low-end floor: a non-empty shard that receives any global mass returns
    /// at least 1 (placed on its top-residual expert) even when its proportional
    /// share rounds to 0. This keeps `per_group_batches` non-empty for the
    /// grouped-GEMM cache; it widens the cross-shard drift at small `global` (Σ
    /// shards > global), tolerated because the MoE cost path maxes — not sums —
    /// across EP ranks. `global == 0` still yields all-zero.
    ///
    /// Associated (not `&self`) so the caller passes whatever ppm slice it
    /// wants: `RoutingDistribution::to_per_expert_counts(total, &dist.ppm()[lo..hi])`.
    pub fn to_per_expert_counts(global_expert_selections: u32, ppm: &[u32]) -> Vec<u32> {
        let mut counts = vec![0u32; ppm.len()];
        let mut numerator_sum: u128 = 0;
        for slot in ppm.iter().copied() {
            debug_assert!(
                slot <= Self::TOTAL_PPM,
                "ppm entry {slot} exceeds TOTAL_PPM"
            );
            numerator_sum += u128::from(u64::from(global_expert_selections) * u64::from(slot));
        }

        // Rank by ppm so the active set is the shard's heaviest experts; index
        // breaks ties so a uniform shard still yields a stable, reproducible
        // vector (it is part of the grouped-GEMM cache key).
        let mut by_weight: Vec<usize> = (0..ppm.len()).collect();
        by_weight.sort_by(|&lhs, &rhs| ppm[rhs].cmp(&ppm[lhs]).then_with(|| lhs.cmp(&rhs)));
        let active_len = Self::active_set_len(global_expert_selections, ppm);
        let active = &by_weight[..active_len];

        let mut residuals = Vec::with_capacity(active_len);
        let mut assigned: u64 = 0;
        for &idx in active {
            let numerator = u64::from(global_expert_selections) * u64::from(ppm[idx]);
            let base = (numerator / u64::from(Self::TOTAL_PPM)) as u32;
            counts[idx] = base;
            residuals.push((idx, numerator % u64::from(Self::TOTAL_PPM)));
            assigned += u64::from(base);
        }
        // Target = round(Σ exact shares) = round(total × Σppm / TOTAL_PPM). For
        // a shard (Σppm < TOTAL_PPM) this is the shard's proportional count, not
        // the global total; for a full distribution it equals global_expert_selections.
        let total_ppm = u128::from(Self::TOTAL_PPM);
        let mut target = ((numerator_sum + total_ppm / 2) / total_ppm) as u64;
        // v1 floor: a shard whose proportional share rounds below 0.5 (small
        // `global_expert_selections` split across many EP ranks) would otherwise
        // get 0 on every local expert, yielding an all-zero `per_group_batches`
        // the grouped-GEMM profiler/cache cannot represent (DeepGEMM has no
        // zero-batch row). Any shard that receives *some* global mass loads at
        // least its top-residual expert once — and only that one (the deficit is
        // 1, so ep=32's four local experts do NOT all jump to 1). This
        // over-counts the global total when summed across ranks (Σ shards >
        // global), but the MoE cost path takes Max across EP ranks rather than
        // Sum, so the broken conservation never reaches the model output. The
        // exact fix (re-axis the grouped-GEMM cache to per-rank token counts and
        // quantile each rank's load at L4) is deferred — see
        // agent-trace/moe_arch_qwen3_dp_attn_ep_ffn.md.
        if numerator_sum > 0 {
            target = target.max(1);
        }
        // The target still comes from the WHOLE shard, so the per-shard total is
        // exactly what it was before this active-set restriction: only the shape
        // moves, never the sum. That matters because `local_quant_rows` sums
        // these counts to size the quantize leaf, which must stay
        // distribution-invariant.
        //
        // Cycling (rather than one pass) is therefore required: the mass the
        // inactive tail would have held has to land somewhere, and it belongs on
        // the heaviest residuals. With a one-pass take() a shard whose deficit
        // exceeds its active set would silently lose tokens.
        residuals.sort_by(|lhs, rhs| rhs.1.cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
        if !residuals.is_empty() {
            let mut cursor = 0usize;
            while assigned < target {
                counts[residuals[cursor % residuals.len()].0] += 1;
                assigned += 1;
                cursor += 1;
            }
        }
        counts
    }

    /// Size of the active set: `round(Σ_e P(expert e receives ≥ 1 assignment))`
    /// under the marginal `Binomial(global_expert_selections, ppm_e / TOTAL_PPM)`
    /// each expert's count follows.
    ///
    /// Deliberately a per-expert quantity — `1 − (1 − p_e)^G` depends only on
    /// that expert's own global ppm and the global assignment count, never on
    /// which shard it sits in or who its neighbours are. So slicing `ppm` slices
    /// this result exactly, and an EP fan-out computed shard by shard sums to
    /// the same value as the unsharded model at every `ep_size`. Renormalizing
    /// within a shard would *not* have that property (it drifts upward as the
    /// shard narrows, ~2% by ep=32).
    ///
    /// The power uses binary exponentiation over plain `f64` multiplies, whose
    /// results IEEE-754 pins exactly, rather than `powf`, whose last bit is
    /// libm-dependent. This value decides a grouped-GEMM cache key, so it has to
    /// be reproducible across hosts, not merely close.
    fn active_set_len(global_expert_selections: u32, ppm: &[u32]) -> usize {
        if ppm.is_empty() || global_expert_selections == 0 {
            return 0;
        }
        let expected: f64 = ppm
            .iter()
            .copied()
            .map(|slot| {
                let miss = 1.0 - f64::from(slot) / f64::from(Self::TOTAL_PPM);
                1.0 - pow_exact(miss, global_expert_selections)
            })
            .sum();
        // At least one: a shard holding any mass must present a non-empty
        // `per_group_batches`, matching the v1 floor below.
        (expected.round() as usize).clamp(1, ppm.len())
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
    (0..buckets)
        .map(|idx| base + u32::from(idx < rem))
        .collect()
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
    fn random_is_seeded_and_normalized() {
        // Same seed ⇒ identical ppm (reproducible, golden-safe); ppm normalizes
        // to TOTAL_PPM; a different seed gives a different skew.
        let a = RoutingDistribution::random(32, 0xD1CE_5EED);
        let b = RoutingDistribution::random(32, 0xD1CE_5EED);
        assert_eq!(a, b, "same seed must replay an identical distribution");
        assert_eq!(a.num_experts(), 32);
        assert_eq!(a.ppm().iter().sum::<u32>(), RoutingDistribution::TOTAL_PPM);
        let c = RoutingDistribution::random(32, 0x0BAD_F00D);
        assert_ne!(a.ppm(), c.ppm(), "a different seed should skew differently");
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

    /// The defect this whole active-set construction exists to fix: with 512
    /// assignments spread over 256 experts, giving every expert its mean leaves
    /// almost none empty, but a real step's routing empties a long tail. The
    /// grouped-GEMM kernels pad per expert, so those empty cells are free and
    /// the non-empty ones are the cost.
    #[test]
    fn a_short_batch_over_many_experts_leaves_a_tail_empty() {
        let uniform = RoutingDistribution::uniform(256);
        let counts = RoutingDistribution::to_per_expert_counts(512, uniform.ppm());

        // 256 x (1 - (1 - 1/256)^512) = 256 x (1 - e^-2) ~ 221.4.
        let active = counts.iter().filter(|count| **count > 0).count();
        assert_eq!(active, 221);
        assert!(active < 256, "a mean-per-expert spread would fill all 256");

        // Only the SHAPE moves: the total is still the full proportional share,
        // because `local_quant_rows` sums these to size the quantize leaf.
        assert_eq!(counts.iter().sum::<u32>(), 512);
    }

    /// A long batch saturates every expert, so the construction must collapse
    /// back to the plain proportional spread — this is why prefill iterations
    /// and high-`ep_size` shards are left alone.
    #[test]
    fn a_long_batch_activates_every_expert() {
        let uniform = RoutingDistribution::uniform(256);
        let counts = RoutingDistribution::to_per_expert_counts(16_384, uniform.ppm());
        assert_eq!(counts.iter().filter(|count| **count > 0).count(), 256);
        assert_eq!(counts.iter().sum::<u32>(), 16_384);
    }

    /// The active-set size is a sum of per-expert terms in the *global* ppm and
    /// the *global* assignment count, so slicing the ppm slices the result. An
    /// EP fan-out computed shard by shard must therefore agree exactly with the
    /// unsharded model at every `ep_size` — renormalizing inside a shard would
    /// instead drift upward as the shard narrows.
    #[test]
    fn the_active_set_is_identical_however_the_ppm_is_sharded() {
        let dist = RoutingDistribution::power_law(256, 0.7);
        let unsharded = RoutingDistribution::active_set_len(512, dist.ppm());
        for ep_size in [2usize, 4, 8, 16, 32] {
            let experts_per_rank = 256 / ep_size;
            let sharded: usize = (0..ep_size)
                .map(|rank| {
                    let start = rank * experts_per_rank;
                    RoutingDistribution::active_set_len(
                        512,
                        &dist.ppm()[start..start + experts_per_rank],
                    )
                })
                .sum();
            // Only per-shard rounding to whole experts separates them.
            let drift = sharded.abs_diff(unsharded);
            assert!(
                drift <= ep_size / 2,
                "ep={ep_size}: sharded {sharded} vs unsharded {unsharded}"
            );
        }
    }

    #[test]
    fn pow_exact_is_a_fixed_multiply_tree_not_a_libm_call() {
        use super::pow_exact;
        assert_eq!(pow_exact(0.5, 0), 1.0);
        assert_eq!(pow_exact(0.5, 1), 0.5);

        // The contract is a FIXED squaring tree, so each power is bit-equal to
        // the tree spelled out by hand. It is deliberately not equal to a naive
        // n-step loop -- that has a different rounding order and would differ in
        // the last bit, which is exactly why `powf` cannot be trusted either.
        let base = 0.996_f64;
        let squared = base * base;
        assert_eq!(pow_exact(base, 2), squared);
        assert_eq!(pow_exact(base, 4), squared * squared);
        assert_eq!(pow_exact(base, 5), base * (squared * squared));
    }

    fn to_per_expert_counts_floors_tiny_shard_to_one() {
        // ep=32 of 128 experts → 4 local experts, uniform ppm ≈ 7812 each
        // (Σ ≈ 31248 ≈ TOTAL_PPM/32). At low `global` the proportional share
        // rounds below 0.5, but a shard that got *some* mass must still load one
        // expert (an all-zero per_group_batches has no grouped-GEMM cache row).
        let local = vec![7_812u32; 4];
        // global=1 → share 0.031 → floored onto the top-residual expert only.
        assert_eq!(
            RoutingDistribution::to_per_expert_counts(1, &local),
            vec![1, 0, 0, 0]
        );
        // Floor stays at ONE active expert across the low range (NOT [1,1,1,1]):
        // global=16 → share ≈ 0.5, global=48 → share ≈ 1.5, both deficit 1.
        assert_eq!(
            RoutingDistribution::to_per_expert_counts(16, &local),
            vec![1, 0, 0, 0]
        );
        assert_eq!(
            RoutingDistribution::to_per_expert_counts(48, &local),
            vec![1, 0, 0, 0]
        );
        // Two experts only once the rounded share reaches 2 (global=64 → 2.0).
        assert_eq!(
            RoutingDistribution::to_per_expert_counts(64, &local),
            vec![1, 1, 0, 0]
        );
        // No work → no floor.
        assert_eq!(
            RoutingDistribution::to_per_expert_counts(0, &local),
            vec![0, 0, 0, 0]
        );
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
