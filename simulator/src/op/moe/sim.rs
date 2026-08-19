//! `op/moe/sim` — the unified "simulate-as-model" core: one per-token routing +
//! placement realization that prices ALL SIX `MoE` network stages' per-GPU bytes,
//! sampling each stage's realized bottleneck on a token-count grid into a
//! [`BottleneckCurve`].
//!
//! ## Why one simulator for all six stages
//!
//! The pre-unify design split `MoE` comm two ways: dispatch used an analytic
//! per-rank hit probability `q` (exact, since dispatch bytes are per-rank
//! independent and linear in `T`), while combine's reduce stages needed a
//! token-level sim (the reduce volume is non-linear in a token's hit *set*:
//! intra-domain reduce costs `(|H∩d|−1)·B`, inter-domain `(g−1)·B`). Keeping two
//! mechanisms — analytic `q` for dispatch, sim for combine, plus a separate dense
//! formula for the delivery stages — was both redundant and inconsistent.
//!
//! Realizing the routing per token already produces, for free, every quantity
//! the analytic path computed: the per-token hit set determines dispatch fan-out
//! AND combine reduce volume AND the rail-aligned NIC hops. So a single
//! [`simulate_once`] prices the whole 6-stage critical path off one realization,
//! and the multi-point curve naturally captures the busiest-of-K-symmetric-GPU
//! `√(ep/T)` margin that a single `q×T` coefficient cannot express.
//!
//! ## The home-rank model (deviates from ref — surfaced for sign-off)
//!
//! Real EP gives each token a *home* rank: where it resides before dispatch and
//! where its combined result returns. [`Placement`] fixes the home at config time
//! (balanced-EP default: home = `token_idx % ep_size`). For a token homed on rank
//! `home` (in domain `home_dom`) hitting rank set `H`:
//!
//! * **`dispatch_inter`** (NIC): for each remote hit domain `d`, `home` sends one
//!   copy to the rail-aligned ingress `gateway(d) = rail_peer(home, d)`.
//! * **`dispatch_intra`** (NVLink): inside each hit domain, the gateway fans the
//!   token to every hit rank `≠ gateway` (in `home_dom` the gateway is `home`).
//! * **`combine_intra_reduce`** (NVLink): the mirror — each hit rank `≠ gateway`
//!   reduces its partial to the domain gateway.
//! * **`combine_inter_reduce`** (NIC): each remote domain's gateway reduces back to
//!   `home`.
//! * **`combine_inter_bcast` / `combine_intra_fanout`**: identically zero under a
//!   single-home placement (they deliver a *replicated* output to many target
//!   ranks; round-robin homes have exactly one sink). Kept as stages so the op
//!   taxonomy and leaf count stay stable for a future replicated placement.
//!
//! This makes dispatch and combine exact mirrors (home↔gateway over NIC, fan↔
//! reduce within a domain) and drops ref's distinct dense reduce-scatter formula
//! — all six stages now ride one consistent whole-`B`/token transfer tree.
//!
//! Build-time only: per-token per-GPU byte loads are realized once per grid point
//! over `trials` seeded draws (bit-identical), averaged into the stage's `E[max]`
//! bottleneck. Runtime just interpolates the curve at the live token count.

use super::{BottleneckCurve, MoeNetParams, MoeStep, NvlLayout, P2pTier, Placement};
use crate::timing::routing::{for_each_routed_token, RoutingRng};

/// The six `MoE` network stages in critical-path order, paired with their fabric
/// tier. Index layout: `0,1` dispatch (read by [`MoeDispatchOp`]); `2..6` combine
/// (read by [`MoeCombineOp`]).
const STAGES: [(&str, P2pTier); 6] = [
    ("moe_dispatch_inter", P2pTier::InterDomain),
    ("moe_dispatch_intra", P2pTier::IntraDomain),
    ("moe_combine_intra_reduce", P2pTier::IntraDomain),
    ("moe_combine_inter_reduce", P2pTier::InterDomain),
    ("moe_combine_inter_bcast", P2pTier::InterDomain),
    ("moe_combine_intra_fanout", P2pTier::IntraDomain),
];

/// Realize `n_tokens` routings under `placement` and accumulate the per-rank
/// `send`/`recv` bytes of all six stages. Deterministic for a fixed `seed`
/// (splitmix64 + fixed iteration order). `ppm` is the global per-expert
/// popularity (same layout as [`crate::timing::routing::RoutingDistribution`]).
#[must_use]
pub fn simulate_once(
    ppm: &[u32],
    top_k: u32,
    params: &MoeNetParams,
    placement: Placement,
    n_tokens: u32,
    seed: u64,
) -> [MoeStep; 6] {
    let ep = params.ep_size as usize;
    let hidden_bytes = u64::from(params.hidden_bytes);
    let mut steps = std::array::from_fn(|i| MoeStep::empty(STAGES[i].0, STAGES[i].1, ep));
    if ep == 0 || hidden_bytes == 0 || ppm.is_empty() || n_tokens == 0 {
        return steps;
    }

    let layout = NvlLayout::new(params.ep_size, params.nvl_num_gpu);
    let mut rng = RoutingRng::new(seed);
    let mut token_index: u64 = 0;
    for_each_routed_token(
        ppm,
        top_k,
        params.ep_size,
        params.nvl_num_gpu,
        n_tokens,
        &mut rng,
        |hit_rank, hit_dom| {
            price_token(
                token_index,
                hit_rank,
                hit_dom,
                placement,
                params.ep_size,
                hidden_bytes,
                &layout,
                &mut steps,
            );
            token_index += 1;
        },
    );
    steps
}

/// One token's contribution to the six per-stage `MoeStep` byte counters under
/// the residing-set / rail-aligned routing model described in the module-level
/// doc.
///
/// Logic in two phases:
/// 1. **dispatch + producer-side reduce** (over `hit_dom`): for each hit domain,
///    pick a `gateway` (a resident `S` rank if any, else `rail_peer(primary, d)`),
///    cost the home↔gateway NIC hop in `dispatch_inter` / `combine_inter_reduce`,
///    and the intra-domain fan/reduce hops in `dispatch_intra` /
///    `combine_intra_reduce`.
/// 2. **combine delivery** (over target domains, i.e. those holding `S` ranks):
///    bcast the result from `primary` to a `result_holder` in each non-root
///    target domain, then fan within that domain to the other resident ranks.
///    These two stages stay zero whenever `|S| == 1` (single combine sink).
#[allow(
    clippy::too_many_arguments,
    reason = "each arg is a distinct per-token routing/cost input; bundling would just move the same fan-out into a struct"
)]
fn price_token(
    token_index: u64,
    hit_rank: &[bool],
    hit_dom: &[bool],
    placement: Placement,
    ep_size: u32,
    hidden_bytes: u64,
    layout: &NvlLayout,
    steps: &mut [MoeStep; 6],
) {
    // The token's residing rank set S = [residing_start, residing_end) and its
    // rotating leader `primary` (dispatch forwarder / combine global-reduce
    // root). For RoundRobin |S| == 1 and primary IS the single home rank.
    let (residing_start, residing_count) = placement.residing_group(token_index, ep_size);
    let residing_end = residing_start + residing_count;
    let is_resident = |rank: u16| rank >= residing_start && rank < residing_end;
    let primary = residing_start + (token_index % u64::from(residing_count)) as u16;
    let primary_domain = layout.domain_of(primary);

    // Slice of S that resides in `domain` — a contiguous window of resident
    // ranks. Returned as (start, end); end == start means "no residents".
    let residents_in_domain = |domain: usize| -> (u16, u16) {
        let domain_start = layout.first_rank(domain);
        let domain_end = domain_start + layout.domain_size(domain) as u16;
        (
            residing_start.max(domain_start),
            residing_end.min(domain_end),
        )
    };
    // Rotate among `window_size` residents by the token index — spreads gateway
    // / bcast-landing duty so the bottleneck doesn't pile on one rank.
    let rotate = |window_start: u16, window_size: u16| -> u16 {
        window_start + (token_index % u64::from(window_size)) as u16
    };

    // ── dispatch + producer-side reduce (over hit domains) ──────────────────
    for (domain, &domain_was_hit) in hit_dom.iter().enumerate() {
        if !domain_was_hit {
            continue;
        }
        let domain_start = layout.first_rank(domain);
        let domain_end = domain_start + layout.domain_size(domain) as u16;
        // Gateway: a rotated resident S rank in `domain` if any, else the
        // rail-aligned ingress peer of `primary`. Same rank serves as the
        // dispatch fan-out source AND the combine reduce egress.
        let (resident_lo, resident_hi) = residents_in_domain(domain);
        let has_resident = resident_lo < resident_hi;
        let gateway = if has_resident {
            rotate(resident_lo, resident_hi - resident_lo)
        } else {
            layout.rail_peer(primary, domain)
        };

        if !has_resident {
            // Stage 0 dispatch_inter: primary → remote gateway (1 NIC copy).
            steps[0].send_bytes[usize::from(primary)] += hidden_bytes;
            steps[0].recv_bytes[usize::from(gateway)] += hidden_bytes;
        }
        if domain != primary_domain {
            // Stage 3 combine_inter_reduce: this domain's reduced partial (held
            // by its gateway) flows back to the global root.
            steps[3].send_bytes[usize::from(gateway)] += hidden_bytes;
            steps[3].recv_bytes[usize::from(primary)] += hidden_bytes;
        }

        for hit in domain_start..domain_end {
            if !hit_rank[usize::from(hit)] {
                continue;
            }
            // Stage 1 dispatch_intra: deliver the token to hit `hit`. In a
            // domain holding S ranks the token is already on every resident, so
            // only NON-resident hits need a copy; elsewhere the inter copy
            // landed on the gateway, which fans to the rest.
            let needs_dispatch = if has_resident {
                !is_resident(hit)
            } else {
                hit != gateway
            };
            if needs_dispatch {
                steps[1].send_bytes[usize::from(gateway)] += hidden_bytes;
                steps[1].recv_bytes[usize::from(hit)] += hidden_bytes;
            }
            // Stage 2 combine_intra_reduce: every producer except the gateway
            // reduces its partial into the gateway.
            if hit != gateway {
                steps[2].send_bytes[usize::from(hit)] += hidden_bytes;
                steps[2].recv_bytes[usize::from(gateway)] += hidden_bytes;
            }
        }
    }

    // ── combine delivery: replicate the result to every S rank ──────────────
    // Runs over TARGET domains (those holding S ranks), independent of routing
    // — the output must reach all residence ranks regardless of where its
    // experts were. Zero whenever |S| == 1 (single combine sink).
    for domain in 0..layout.num_domains() {
        let (resident_lo, resident_hi) = residents_in_domain(domain);
        if resident_lo >= resident_hi {
            continue; // not a target domain
        }
        // The rank holding the consolidated result in this domain: `primary` in
        // the root domain, else the rail-aligned bcast landing (a rotated
        // resident — same pick as the reduce gateway).
        let result_holder = if domain == primary_domain {
            primary
        } else {
            let landing = rotate(resident_lo, resident_hi - resident_lo);
            // Stage 4 combine_inter_bcast: root → this target domain.
            steps[4].send_bytes[usize::from(primary)] += hidden_bytes;
            steps[4].recv_bytes[usize::from(landing)] += hidden_bytes;
            landing
        };
        // Stage 5 combine_intra_fanout: holder → the other resident ranks.
        for resident in resident_lo..resident_hi {
            if resident != result_holder {
                steps[5].send_bytes[usize::from(result_holder)] += hidden_bytes;
                steps[5].recv_bytes[usize::from(resident)] += hidden_bytes;
            }
        }
    }
}

/// Sample each of the six stages' realized bottleneck bytes (`E[max]` over GPUs,
/// estimated by averaging `trials` seeded draws) at every token count in
/// `t_grid`, returning one [`BottleneckCurve`] per stage. This is the op's
/// build-time entry point: heavy (a grid × trials of full routing sims), run once
/// per config; the runtime path is a single curve interpolation per stage.
///
/// Sampling the bottleneck at MULTIPLE `T` (rather than one point scaled linearly)
/// captures its `T`-dependence: under near-symmetric routing the realized busiest
/// GPU sits `~√(ep/T)` above the mean-field per-GPU load, a margin a single
/// coefficient × `T` cannot express.
#[must_use]
pub fn simulate_moe_comm(
    ppm: &[u32],
    top_k: u32,
    params: &MoeNetParams,
    placement: Placement,
    t_grid: &[u32],
    trials: u32,
    seed: u64,
) -> [BottleneckCurve; 6] {
    let n = f64::from(trials.max(1));
    let mut points: [Vec<(u32, f64)>; 6] =
        std::array::from_fn(|_| Vec::with_capacity(t_grid.len()));
    for &t in t_grid {
        let mut acc = [0.0f64; 6];
        for trial in 0..trials.max(1) {
            // Per-(t, trial) seed: independent draws, still deterministic.
            let s = seed
                ^ u64::from(t).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ u64::from(trial).wrapping_mul(0xD1B5_4A32_D192_ED03);
            let steps = simulate_once(ppm, top_k, params, placement, t, s);
            for (i, a) in acc.iter_mut().enumerate() {
                *a += steps[i].bottleneck_bytes() as f64;
            }
        }
        for (i, p) in points.iter_mut().enumerate() {
            p.push((t, acc[i] / n));
        }
    }
    let mut points = points.into_iter();
    std::array::from_fn(|i| BottleneckCurve {
        tier: STAGES[i].1,
        points: points.next().unwrap(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::routing::RoutingDistribution;

    fn params() -> MoeNetParams {
        MoeNetParams {
            ep_size: 8,
            nvl_num_gpu: 4, // domains [0..4], [4..8]
            hidden_bytes: 4096 * 2,
        }
    }

    fn cv(xs: &[u64]) -> f64 {
        let n = xs.len() as f64;
        let mean = xs.iter().map(|&x| x as f64).sum::<f64>() / n;
        if mean <= 0.0 {
            return 0.0;
        }
        let var = xs.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
        var.sqrt() / mean
    }

    fn load(step: &MoeStep, r: usize) -> u64 {
        step.send_bytes[r] + step.recv_bytes[r]
    }

    #[test]
    fn produces_six_named_tiered_stages() {
        let dist = RoutingDistribution::uniform(64);
        let steps = simulate_once(
            dist.ppm(),
            8,
            &params(),
            Placement::RoundRobin,
            4096,
            0xABCD,
        );
        let expect = [
            ("moe_dispatch_inter", P2pTier::InterDomain),
            ("moe_dispatch_intra", P2pTier::IntraDomain),
            ("moe_combine_intra_reduce", P2pTier::IntraDomain),
            ("moe_combine_inter_reduce", P2pTier::InterDomain),
            ("moe_combine_inter_bcast", P2pTier::InterDomain),
            ("moe_combine_intra_fanout", P2pTier::IntraDomain),
        ];
        for (s, (name, tier)) in steps.iter().zip(expect) {
            assert_eq!(s.name, name);
            assert_eq!(s.tier, tier);
            assert_eq!(s.send_bytes.len(), 8);
            assert_eq!(s.recv_bytes.len(), 8);
        }
    }

    #[test]
    fn dispatch_and_combine_reduce_mirror_each_other() {
        // The home↔gateway NIC hop and the within-domain fan/reduce are exact
        // mirrors, so dispatch_inter total bytes == combine_inter_reduce total,
        // and dispatch_intra total == combine_intra_reduce total.
        let dist = RoutingDistribution::power_law(64, 1.0);
        let steps = simulate_once(
            dist.ppm(),
            4,
            &params(),
            Placement::RoundRobin,
            8192,
            0x1234,
        );
        let total = |s: &MoeStep| -> u64 { s.send_bytes.iter().sum::<u64>() };
        assert_eq!(
            total(&steps[0]),
            total(&steps[3]),
            "inter dispatch vs reduce"
        );
        assert_eq!(total(&steps[1]), total(&steps[2]), "intra fan vs reduce");
    }

    #[test]
    fn bcast_and_fanout_are_zero_for_single_home() {
        let dist = RoutingDistribution::uniform(64);
        let steps = simulate_once(dist.ppm(), 8, &params(), Placement::RoundRobin, 4096, 0x55);
        assert_eq!(steps[4].bottleneck_bytes(), 0, "inter_bcast");
        assert_eq!(steps[5].bottleneck_bytes(), 0, "intra_fanout");
    }

    #[test]
    fn hp_single_domain_fires_fanout_not_bcast() {
        // HP group == one NVL domain (hp_size == nvl_num_gpu): the combine output
        // replicates within one domain, so intra_fanout fires but inter_bcast
        // stays zero (only one target domain). Each token's fanout delivers to
        // hp_size−1 other resident ranks.
        let dist = RoutingDistribution::uniform(64);
        let hp = Placement::ReplicatedHeadParallel { hp_size: 4 };
        let steps = simulate_once(dist.ppm(), 8, &params(), hp, 4096, 0x71);
        assert_eq!(
            steps[4].bottleneck_bytes(),
            0,
            "inter_bcast (single target domain)"
        );
        assert!(steps[5].bottleneck_bytes() > 0, "intra_fanout should fire");
    }

    #[test]
    fn hp_multi_domain_fires_bcast() {
        // HP group spans two NVL domains (hp_size == 2 × nvl_num_gpu): the output
        // replicates across domains, so inter_bcast fires too.
        let dist = RoutingDistribution::uniform(64);
        let hp = Placement::ReplicatedHeadParallel { hp_size: 8 }; // ep8/nvl4 → 2 domains
        let steps = simulate_once(dist.ppm(), 8, &params(), hp, 4096, 0x72);
        assert!(
            steps[4].bottleneck_bytes() > 0,
            "inter_bcast should fire across domains"
        );
        assert!(steps[5].bottleneck_bytes() > 0, "intra_fanout should fire");
    }

    /// Fast regression: on a few pinned configs, sim's per-stage system
    /// copies/token must match ref's analytic baseline on the EXACT-class
    /// stages (dispatch_inter, inter_bcast, intra_fanout, and inter_reduce when
    /// the HP group sits in one domain), and exceed it on the rail-gap stages
    /// (dispatch_intra, combine_intra_reduce) by the expected DeepEP rail-
    /// alignment surplus. Guards every refactor that touches `price_token`,
    /// `Placement`, or the ref baseline math.
    #[test]
    fn bytes_align_with_ref_on_pinned_configs() {
        // (E, ep, nvl, k, hp). hp=1 ≡ RoundRobin; hp=nvl single-domain HP;
        // hp=2·nvl multi-domain HP (inter_bcast fires); hp=ep ≡ full replication.
        let configs = [
            (64u32, 8u32, 4u32, 8u32, 1u32), // RoundRobin
            (128, 16, 8, 8, 8),              // HP one domain
            (64, 8, 4, 8, 8),                // HP == ep (zero dispatch)
            (128, 16, 8, 8, 16),             // HP spans 2 domains (inter_bcast fires)
        ];
        let hidden_bytes = 4096 * 2;
        let n_tokens = 30_000u32;

        for &(e, ep, nvl, k, hp) in &configs {
            let dist = RoutingDistribution::uniform(e);
            let params = MoeNetParams {
                ep_size: ep,
                nvl_num_gpu: nvl,
                hidden_bytes,
            };
            let placement = if hp == 1 {
                Placement::RoundRobin
            } else {
                Placement::ReplicatedHeadParallel { hp_size: hp }
            };
            let steps = simulate_once(dist.ppm(), k, &params, placement, n_tokens, 0xA110_C0DE);
            let hidden_f = f64::from(hidden_bytes);
            let sim_copies: [f64; 6] = std::array::from_fn(|i| {
                steps[i].send_bytes.iter().sum::<u64>() as f64 / hidden_f / f64::from(n_tokens)
            });
            let ref_copies = ref_hp_copies(e, ep, nvl, k, hp);
            let single_target_domain = hp <= nvl;
            let label = format!("E{e} ep{ep} nvl{nvl} k{k} hp{hp}");

            // EXACT-class stages: must align with ref (small MC tolerance).
            let exact_stages = [
                ("dispatch_inter", 0),
                ("combine_inter_bcast", 4),
                ("combine_intra_fanout", 5),
            ]
            .into_iter()
            .chain(single_target_domain.then_some(("combine_inter_reduce", 3)));

            for (stage_name, stage) in exact_stages {
                let (sim, rf) = (sim_copies[stage], ref_copies[stage]);
                if rf < 1e-9 {
                    assert!(
                        sim.abs() < 1e-9,
                        "{label} {stage_name}: ref=0 but sim={sim}"
                    );
                    continue;
                }
                let rel = (sim - rf) / rf * 100.0;
                assert!(
                    rel.abs() < 3.0,
                    "{label} {stage_name}: sim={sim:.4} vs ref={rf:.4} (rel {rel:+.2}%) — ref alignment broken",
                );
            }

            // Rail-gap stages: ours ≥ ref (sanity-check the direction of the gap).
            for (stage_name, stage) in [("dispatch_intra", 1), ("combine_intra_reduce", 2)] {
                let (sim, rf) = (sim_copies[stage], ref_copies[stage]);
                if rf < 1e-9 {
                    continue; // hp == ep degenerate: nothing to dispatch / reduce.
                }
                assert!(
                    sim + 1e-6 >= rf,
                    "{label} {stage_name}: sim={sim:.4} < ref={rf:.4} (rail surplus should be non-negative)",
                );
            }
        }
    }

    /// Hand-calculated golden, **cross-domain extreme skew**. Setup:
    /// `E=4, ep=2` (rank 0 owns {0,1}, rank 1 owns {2,3}); `nvl=1` ⇒ each rank
    /// gets its own NVL domain; `top_k=1`; PPM = `[1.0, 0, 0, 0]` so EVERY token
    /// picks expert 0, owned by rank 0. `RoundRobin` homes token `i` on rank
    /// `i % 2`. Therefore:
    /// * half the tokens (`home==rank 0`) already host the only hit rank →
    ///   nothing crosses the wire.
    /// * the other half (`home==rank 1`) need one NIC hop home→rank 0 for
    ///   dispatch and the mirror hop rank 0→home for combine. dispatch_inter
    ///   lands on the rail-aligned gateway (= rank 0, which is itself the hit
    ///   rank in its singleton domain), so dispatch_intra is zero.
    ///
    /// Expected per-rank bytes for `n=1000`, `hidden=100`:
    /// `dispatch_inter.send = [0, 50_000]`, `recv = [50_000, 0]`;
    /// `combine_inter_reduce` is the mirror; every other stage is identically zero.
    #[test]
    fn hand_calc_golden_cross_domain_skew() {
        let dist = RoutingDistribution::from_profile(&[1.0, 0.0, 0.0, 0.0]);
        let params = MoeNetParams {
            ep_size: 2,
            nvl_num_gpu: 1,
            hidden_bytes: 100,
        };
        let n_tokens = 1_000u32;
        let steps = simulate_once(
            dist.ppm(),
            1,
            &params,
            Placement::RoundRobin,
            n_tokens,
            0xDEAD,
        );

        let cross_hop = u64::from(n_tokens / 2) * u64::from(params.hidden_bytes);
        assert_eq!(
            steps[0].send_bytes,
            vec![0, cross_hop],
            "dispatch_inter send"
        );
        assert_eq!(
            steps[0].recv_bytes,
            vec![cross_hop, 0],
            "dispatch_inter recv"
        );
        assert_eq!(
            steps[3].send_bytes,
            vec![cross_hop, 0],
            "combine_inter_reduce send"
        );
        assert_eq!(
            steps[3].recv_bytes,
            vec![0, cross_hop],
            "combine_inter_reduce recv"
        );
        for stage in [1usize, 2, 4, 5] {
            assert_eq!(steps[stage].send_bytes, vec![0, 0], "stage {stage} send");
            assert_eq!(steps[stage].recv_bytes, vec![0, 0], "stage {stage} recv");
        }
    }

    /// Hand-calculated golden, **intra-domain extreme skew**. Setup:
    /// `E=4, ep=2` (rank 0 owns {0,1}, rank 1 owns {2,3}); `nvl=2` ⇒ one NVL
    /// domain holding both ranks; `top_k=1`; PPM = `[0, 0, 1.0, 0]` so EVERY
    /// token picks expert 2, owned by rank 1. `RoundRobin` again:
    /// * `home==rank 0`: hit (rank 1) is in the same NVL domain but not the
    ///   home; one intra (NVLink) hop home→hit for dispatch and the mirror for
    ///   combine reduce.
    /// * `home==rank 1`: home is the only hit; nothing moves.
    ///
    /// Expected per-rank bytes for `n=1000`, `hidden=100`:
    /// `dispatch_intra.send = [50_000, 0]`, `recv = [0, 50_000]`;
    /// `combine_intra_reduce` is the mirror; all inter / bcast / fanout stages
    /// are identically zero.
    #[test]
    fn hand_calc_golden_intra_domain_skew() {
        let dist = RoutingDistribution::from_profile(&[0.0, 0.0, 1.0, 0.0]);
        let params = MoeNetParams {
            ep_size: 2,
            nvl_num_gpu: 2,
            hidden_bytes: 100,
        };
        let n_tokens = 1_000u32;
        let steps = simulate_once(
            dist.ppm(),
            1,
            &params,
            Placement::RoundRobin,
            n_tokens,
            0xBEEF,
        );

        let intra_hop = u64::from(n_tokens / 2) * u64::from(params.hidden_bytes);
        assert_eq!(
            steps[1].send_bytes,
            vec![intra_hop, 0],
            "dispatch_intra send"
        );
        assert_eq!(
            steps[1].recv_bytes,
            vec![0, intra_hop],
            "dispatch_intra recv"
        );
        assert_eq!(
            steps[2].send_bytes,
            vec![0, intra_hop],
            "combine_intra_reduce send"
        );
        assert_eq!(
            steps[2].recv_bytes,
            vec![intra_hop, 0],
            "combine_intra_reduce recv"
        );
        for stage in [0usize, 3, 4, 5] {
            assert_eq!(steps[stage].send_bytes, vec![0, 0], "stage {stage} send");
            assert_eq!(steps[stage].recv_bytes, vec![0, 0], "stage {stage} recv");
        }
    }

    #[test]
    fn hp_dispatch_intra_skips_resident_ranks() {
        // With hp_size == nvl_num_gpu the home domain's ranks are all sources, so
        // no intra dispatch is needed there — total dispatch_intra is strictly
        // less than the single-home (RoundRobin) case, which fans within the home
        // domain too.
        let dist = RoutingDistribution::uniform(64);
        let rr = simulate_once(
            dist.ppm(),
            8,
            &params(),
            Placement::RoundRobin,
            40_000,
            0x73,
        );
        let hp = simulate_once(
            dist.ppm(),
            8,
            &params(),
            Placement::ReplicatedHeadParallel { hp_size: 4 },
            40_000,
            0x73,
        );
        let total = |s: &MoeStep| s.send_bytes.iter().sum::<u64>();
        assert!(
            total(&hp[1]) < total(&rr[1]),
            "HP dispatch_intra {} should be < RoundRobin {}",
            total(&hp[1]),
            total(&rr[1])
        );
    }

    #[test]
    fn dispatch_stages_carry_traffic() {
        let dist = RoutingDistribution::uniform(64);
        let steps = simulate_once(dist.ppm(), 8, &params(), Placement::RoundRobin, 4096, 0x77);
        assert!(steps[0].bottleneck_bytes() > 0, "dispatch_inter");
        assert!(steps[1].bottleneck_bytes() > 0, "dispatch_intra");
        assert!(steps[2].bottleneck_bytes() > 0, "combine_intra_reduce");
        assert!(steps[3].bottleneck_bytes() > 0, "combine_inter_reduce");
    }

    #[test]
    fn single_domain_has_no_inter_traffic() {
        // nvl_num_gpu >= ep_size → one NVL domain → no NIC hops at all.
        let p = MoeNetParams {
            ep_size: 8,
            nvl_num_gpu: 8,
            hidden_bytes: 4096 * 2,
        };
        let dist = RoutingDistribution::uniform(64);
        let steps = simulate_once(dist.ppm(), 8, &p, Placement::RoundRobin, 4096, 0x99);
        assert_eq!(steps[0].bottleneck_bytes(), 0, "dispatch_inter");
        assert_eq!(steps[3].bottleneck_bytes(), 0, "combine_inter_reduce");
        assert!(steps[1].bottleneck_bytes() > 0, "dispatch_intra");
        assert!(steps[2].bottleneck_bytes() > 0, "combine_intra_reduce");
    }

    #[test]
    fn balanced_routing_spreads_load_evenly() {
        // Uniform popularity + round-robin homes → per-rank dispatch_intra load is
        // nearly flat (small cv): the regime where skew-blind models would agree.
        let dist = RoutingDistribution::uniform(64);
        let steps = simulate_once(
            dist.ppm(),
            8,
            &params(),
            Placement::RoundRobin,
            40_000,
            0x1234,
        );
        let loads: Vec<u64> = (0..8).map(|r| load(&steps[1], r)).collect();
        assert!(
            cv(&loads) < 0.05,
            "uniform load not flat: cv={}",
            cv(&loads)
        );
    }

    #[test]
    fn skew_concentrates_load_on_hot_domain() {
        // 8 hot experts all in domain 0 (experts 0..31 → ranks 0..3). Those ranks
        // are hit far more often, so their combine reduce load must dominate the
        // cold domain — the per-domain skew a distribution-blind model can't show.
        let mut ratios = vec![0.001f32; 64];
        for r in ratios.iter_mut().take(8) {
            *r = 1.0;
        }
        let dist = RoutingDistribution::from_profile(&ratios);
        let steps = simulate_once(
            dist.ppm(),
            8,
            &params(),
            Placement::RoundRobin,
            40_000,
            0x55,
        );
        let dom0: u64 = (0..4).map(|r| load(&steps[2], r)).sum();
        let dom1: u64 = (4..8).map(|r| load(&steps[2], r)).sum();
        assert!(
            dom0 > dom1,
            "hot domain reduce {dom0} should exceed cold {dom1}"
        );
        let loads: Vec<u64> = (0..8).map(|r| load(&steps[2], r)).collect();
        assert!(
            cv(&loads) > 0.1,
            "skew should make load uneven: cv={}",
            cv(&loads)
        );
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let dist = RoutingDistribution::power_law(64, 1.0);
        let a = simulate_once(
            dist.ppm(),
            4,
            &params(),
            Placement::RoundRobin,
            8192,
            0x9E37,
        );
        let b = simulate_once(
            dist.ppm(),
            4,
            &params(),
            Placement::RoundRobin,
            8192,
            0x9E37,
        );
        for i in 0..6 {
            assert_eq!(a[i].send_bytes, b[i].send_bytes);
            assert_eq!(a[i].recv_bytes, b[i].recv_bytes);
        }
    }

    #[test]
    fn curve_has_one_point_per_grid_and_is_increasing() {
        let dist = RoutingDistribution::uniform(64);
        let grid = [128u32, 512, 2_048, 8_192];
        let curves = simulate_moe_comm(
            dist.ppm(),
            8,
            &params(),
            Placement::RoundRobin,
            &grid,
            16,
            0xC0FFEE,
        );
        for stage in [0usize, 1, 2, 3] {
            let pts = &curves[stage].points;
            assert_eq!(pts.len(), grid.len());
            for w in pts.windows(2) {
                assert!(w[1].1 >= w[0].1, "stage {stage} non-increasing: {w:?}");
            }
        }
        // bcast/fanout curves are all-zero under single-home.
        assert!(curves[4].points.iter().all(|&(_, b)| b == 0.0));
        assert!(curves[5].points.iter().all(|&(_, b)| b == 0.0));
    }

    // ── Eval: balanced case vs ref's analytic expected-copies baseline ───────
    //   uv run cargo test --release --lib \
    //     op::moe::sim::tests::eval_balanced_vs_ref -- --ignored --nocapture
    //
    // Validates our per-token sim against ref `moesim-rs/workload/moe_net.rs`'s
    // closed-form `expected_*_plan` (the distribution-blind dense baseline) in the
    // BALANCED (uniform routing) case. The model-independent quantity is
    // **system copies per token** = Σ_r send_bytes[r] / B / T — exactly ref's
    // `*_copies_per_token_system`. Our `Placement::RoundRobin` (each token homed
    // on one rank, homes uniform over ranks) corresponds to ref's
    // `SequenceParallelSharded` placement, whose expected copies average
    // `expected_*_for_source` over a singleton source/target on each rank.
    //
    // Expected outcome (a real, surfaced modeling difference): the INTER stages
    // match exactly (both = Σ_{d≠home_dom} q_domain[d]); the INTRA stages run
    // slightly HIGHER than ref by ~(n_dom−1)(q_domain − q_rank) per token, because
    // our rail-aligned ingress lands on a FIXED `rail_peer(home,d)` rank (DeepEP
    // behavior — saves an intra hop only when that specific gateway is itself a
    // hit rank, prob q_rank), whereas ref optimistically assumes the inter copy
    // lands ON a hit rank whenever the domain is hit (prob q_domain ≥ q_rank).

    /// Hypergeometric hit prob: P(a top-`k` WITHOUT-replacement draw from `total`
    /// experts lands ≥1 pick in a bucket of `bucket` experts) — ref's `hit_prob`.
    fn hit_prob(total: u32, bucket: u32, k: u32) -> f64 {
        let (t, b, k) = (f64::from(total), f64::from(bucket), k);
        let mut miss = 1.0;
        for i in 0..k {
            let fi = f64::from(i);
            miss *= (t - b - fi).max(0.0) / (t - fi);
        }
        1.0 - miss
    }

    #[test]
    #[ignore = "eval: run explicitly with --ignored --nocapture"]
    fn eval_balanced_vs_ref_analytic() {
        // (num_experts, ep_size, nvl_num_gpu, top_k)
        let shapes = [
            (64u32, 8u32, 4u32, 8u32),
            (128, 16, 8, 8),
            (256, 32, 8, 8),
            (128, 16, 8, 2),
        ];
        let hidden_bytes = 4096 * 2;
        let n_tokens = 200_000u32;

        println!(
            "\n{:<22} {:>10} {:>4} {:>4} {:>4}   {:>12} {:>12} {:>8}",
            "shape", "stage", "ep", "nvl", "k", "sim_cpt", "ref_cpt", "rel%"
        );
        let mut inter_max = 0.0f64;
        let mut intra_max = 0.0f64;
        for &(e, ep, nvl, k) in &shapes {
            let dist = RoutingDistribution::uniform(e);
            let p = MoeNetParams {
                ep_size: ep,
                nvl_num_gpu: nvl,
                hidden_bytes,
            };
            // Our realized system copies/token per stage.
            let steps = simulate_once(dist.ppm(), k, &p, Placement::RoundRobin, n_tokens, 0xBA1A);
            let b = f64::from(hidden_bytes);
            let sim_cpt = |s: &MoeStep| -> f64 {
                s.send_bytes.iter().sum::<u64>() as f64 / b / f64::from(n_tokens)
            };

            // ref's analytic expected copies (balanced uniform, singleton
            // placement averaged over which rank is the single home).
            let experts_per_rank = crate::timing::routing::balanced_expert_counts(e, ep);
            let ranks_per_domain = crate::timing::routing::ranks_per_nvl_domain(ep, nvl);
            let rank_hit_prob: Vec<f64> = experts_per_rank
                .iter()
                .map(|&c| hit_prob(e, c, k))
                .collect();
            let mut domain_starts = Vec::with_capacity(ranks_per_domain.len());
            let mut domain_start_acc = 0usize;
            for &domain_size in &ranks_per_domain {
                domain_starts.push(domain_start_acc);
                domain_start_acc += domain_size as usize;
            }
            let domain_of_rank = |rank: usize| -> usize {
                ranks_per_domain
                    .iter()
                    .scan(0usize, |running, &domain_size| {
                        let lo = *running;
                        *running += domain_size as usize;
                        Some((lo, *running))
                    })
                    .position(|(lo, hi)| rank >= lo && rank < hi)
                    .unwrap()
            };
            let expected_hits_per_domain: Vec<f64> = ranks_per_domain
                .iter()
                .enumerate()
                .map(|(domain, &domain_size)| {
                    let lo = domain_starts[domain];
                    rank_hit_prob[lo..lo + domain_size as usize]
                        .iter()
                        .sum::<f64>()
                })
                .collect();
            let domain_hit_prob: Vec<f64> = ranks_per_domain
                .iter()
                .enumerate()
                .map(|(domain, &domain_size)| {
                    let lo = domain_starts[domain];
                    let domain_experts: u32 =
                        experts_per_rank[lo..lo + domain_size as usize].iter().sum();
                    hit_prob(e, domain_experts, k)
                })
                .collect();

            // Average ref copies over every possible single home rank, and also
            // accumulate OUR predicted intra. ref's remote-domain intra subtracts
            // the whole-domain hit prob `domain_hit_prob[d]` (optimal landing);
            // ours subtracts only the fixed rail-aligned gateway's own
            // `rank_hit_prob[gateway]` (DeepEP rail alignment). So
            //     predicted_intra = ref_intra + Σ(q_domain − q_gateway).
            let (mut ref_inter, mut ref_intra, mut predicted_intra) = (0.0f64, 0.0f64, 0.0f64);
            for owner in 0..ep as usize {
                let owner_domain = domain_of_rank(owner);
                let remote_domains = || (0..ranks_per_domain.len()).filter(|&d| d != owner_domain);
                let owner_inter: f64 = remote_domains().map(|d| domain_hit_prob[d]).sum();
                let home_domain_intra =
                    expected_hits_per_domain[owner_domain] - rank_hit_prob[owner];
                let remote_intra_ref: f64 = remote_domains()
                    .map(|d| expected_hits_per_domain[d] - domain_hit_prob[d])
                    .sum();
                // Our remote intra subtracts the rail-aligned gateway's own hit prob.
                let owner_local_index = owner - domain_starts[owner_domain];
                let remote_intra_predicted: f64 = remote_domains()
                    .map(|d| {
                        let domain_size = ranks_per_domain[d] as usize;
                        let gateway = domain_starts[d] + (owner_local_index % domain_size);
                        expected_hits_per_domain[d] - rank_hit_prob[gateway]
                    })
                    .sum();
                ref_inter += owner_inter;
                ref_intra += home_domain_intra + remote_intra_ref;
                predicted_intra += home_domain_intra + remote_intra_predicted;
            }
            ref_inter /= f64::from(ep);
            ref_intra /= f64::from(ep);
            predicted_intra /= f64::from(ep);

            // dispatch_inter == combine_inter_reduce (compare to ref_inter);
            // dispatch_intra == combine_intra_reduce (compare to OUR predicted
            // intra, which is ref_intra + the rail-alignment correction). Print
            // ref_intra too so the modeling gap is visible.
            let inter_rows = [
                ("dispatch_inter", sim_cpt(&steps[0])),
                ("combine_inter_reduce", sim_cpt(&steps[3])),
            ];
            for (stage_name, sim) in inter_rows {
                let rel = (sim - ref_inter) / ref_inter * 100.0;
                inter_max = inter_max.max(rel.abs());
                println!(
                    "{:<8} {:>20} {:>4} {:>4} {:>4}   {:>12.4} {:>12.4} {:>+8.2}",
                    format!("E{e}"),
                    stage_name,
                    ep,
                    nvl,
                    k,
                    sim,
                    ref_inter,
                    rel
                );
            }
            let intra_rows = [
                ("dispatch_intra", sim_cpt(&steps[1])),
                ("combine_intra_reduce", sim_cpt(&steps[2])),
            ];
            for (stage_name, sim) in intra_rows {
                // vs OUR closed-form prediction (the strong check)…
                let rel_predicted = (sim - predicted_intra) / predicted_intra * 100.0;
                intra_max = intra_max.max(rel_predicted.abs());
                // …and vs ref's optimal-landing baseline (the surfaced gap).
                let rel_ref = (sim - ref_intra) / ref_intra * 100.0;
                println!(
                    "{:<8} {:>20} {:>4} {:>4} {:>4}   {:>12.4} pred={:>9.4} vs_pred={:>+6.2}%  ref={:>8.4} vs_ref={:>+6.2}%",
                    format!("E{e}"), stage_name, ep, nvl, k, sim, predicted_intra, rel_predicted, ref_intra, rel_ref
                );
            }
        }
        println!(
            "\nINTER max|rel vs ref| = {inter_max:.2}%  (same formula in both models → must match to MC noise)\n\
             INTRA max|rel vs our closed form| = {intra_max:.2}%  (sim == ref_intra + rail-alignment\n\
             correction Σ(q_domain − q_gateway); ref under-counts intra by assuming the inter copy\n\
             lands on a hit rank, which rail-aligned DeepEP routing cannot do)"
        );
        // Inter stages: identical formula → match to MC noise.
        assert!(inter_max < 1.0, "inter vs ref diverged by {inter_max:.2}%");
        // Intra stages: must match OUR closed-form prediction to MC noise — this
        // proves the gap vs ref is fully explained by the rail-alignment term.
        assert!(
            intra_max < 1.0,
            "intra vs our closed form diverged by {intra_max:.2}%"
        );
    }

    // ── Eval: HP-replicated case vs ref's ReplicatedAcrossHeadParallel ───────
    //   uv run cargo test --release --lib \
    //     op::moe::sim::tests::eval_hp_vs_ref -- --ignored --nocapture
    //
    // Validates `Placement::ReplicatedHeadParallel` against ref `moe_net.rs`'s
    // `expected_source_aware_dispatch_plan` + `expected_sparse_tree_combine_plan`
    // for the `ReplicatedAcrossHeadParallel` placement (system copies/token).
    //
    // Expected outcome:
    // * dispatch_inter, combine_inter_bcast, combine_intra_fanout: EXACT (the
    //   inter NIC hops and the placement-only delivery counts are identical in
    //   both models). The bcast/fanout being NON-ZERO is the whole point of HP.
    // * combine_inter_reduce: exact when the HP group is one domain (hp==nvl);
    //   a root-selection gap appears when the group spans domains (hp>nvl).
    // * dispatch_intra, combine_intra_reduce: ours ≥ ref by the same rail-
    //   alignment term as the single-home case (fixed gateway vs optimal landing).

    /// ref's `expected_*_plan` copy counts for `ReplicatedAcrossHeadParallel`,
    /// averaged over every owner rank. Returns per-stage system copies/token in
    /// the [`STAGES`] order: `[dispatch_inter, dispatch_intra,
    /// combine_intra_reduce, combine_inter_reduce, combine_inter_bcast,
    /// combine_intra_fanout]`.
    fn ref_hp_copies(e: u32, ep: u32, nvl: u32, k: u32, hp: u32) -> [f64; 6] {
        let experts_per_rank = crate::timing::routing::balanced_expert_counts(e, ep);
        let ranks_per_domain = crate::timing::routing::ranks_per_nvl_domain(ep, nvl);
        let mut domain_starts = Vec::with_capacity(ranks_per_domain.len());
        let mut domain_start_acc = 0usize;
        for &domain_size in &ranks_per_domain {
            domain_starts.push(domain_start_acc);
            domain_start_acc += domain_size as usize;
        }
        let rank_hit_prob: Vec<f64> = experts_per_rank
            .iter()
            .map(|&c| hit_prob(e, c, k))
            .collect();
        let expected_hits_per_domain: Vec<f64> = ranks_per_domain
            .iter()
            .enumerate()
            .map(|(domain, &domain_size)| {
                let lo = domain_starts[domain];
                rank_hit_prob[lo..lo + domain_size as usize].iter().sum()
            })
            .collect();
        let experts_per_domain: Vec<u32> = ranks_per_domain
            .iter()
            .enumerate()
            .map(|(domain, &domain_size)| {
                let lo = domain_starts[domain];
                experts_per_rank[lo..lo + domain_size as usize].iter().sum()
            })
            .collect();
        let domain_hit_prob: Vec<f64> = experts_per_domain
            .iter()
            .map(|&c| hit_prob(e, c, k))
            .collect();
        let expected_hit_domains: f64 = domain_hit_prob.iter().sum();
        let is_in_residing_group = |rank: usize, group_start: usize, group_end: usize| {
            rank >= group_start && rank < group_end
        };

        let mut copies_per_stage = [0.0f64; 6];
        for owner in 0..ep {
            let group_start = (owner / hp * hp) as usize;
            let group_end = ((group_start as u32 + hp).min(ep)) as usize;
            let in_group = |rank: usize| is_in_residing_group(rank, group_start, group_end);

            // dispatch (source = HP group of `owner`)
            for (domain, &domain_size) in ranks_per_domain.iter().enumerate() {
                let lo = domain_starts[domain];
                let hi = lo + domain_size as usize;
                let domain_has_source = (lo..hi).any(in_group);
                if domain_has_source {
                    copies_per_stage[1] += (lo..hi)
                        .filter(|&r| !in_group(r))
                        .map(|r| rank_hit_prob[r])
                        .sum::<f64>();
                } else {
                    copies_per_stage[0] += domain_hit_prob[domain];
                    copies_per_stage[1] +=
                        expected_hits_per_domain[domain] - domain_hit_prob[domain];
                }
            }

            // combine (target = same HP group)
            let mut target_domain_count = 0u32;
            let mut target_domain_experts = 0u32;
            for (domain, &domain_size) in ranks_per_domain.iter().enumerate() {
                let lo = domain_starts[domain];
                let hi = lo + domain_size as usize;
                let target_ranks_here: Vec<usize> = (lo..hi).filter(|&r| in_group(r)).collect();
                if target_ranks_here.is_empty() {
                    copies_per_stage[2] +=
                        expected_hits_per_domain[domain] - domain_hit_prob[domain];
                } else {
                    target_domain_count += 1;
                    target_domain_experts += experts_per_domain[domain];
                    let target_rank_experts: u32 =
                        target_ranks_here.iter().map(|&r| experts_per_rank[r]).sum();
                    let q_target_rank_overlap = hit_prob(e, target_rank_experts, k);
                    copies_per_stage[2] += expected_hits_per_domain[domain] - q_target_rank_overlap;
                    copies_per_stage[5] += (target_ranks_here.len() - 1) as f64;
                }
            }
            let q_target_domain_overlap = if target_domain_count > 0 {
                hit_prob(e, target_domain_experts, k)
            } else {
                0.0
            };
            copies_per_stage[3] += expected_hit_domains - q_target_domain_overlap;
            copies_per_stage[4] += target_domain_count.saturating_sub(1) as f64;
        }
        for stage in &mut copies_per_stage {
            *stage /= f64::from(ep);
        }
        copies_per_stage
    }

    #[test]
    #[ignore = "eval: run explicitly with --ignored --nocapture"]
    fn eval_hp_vs_ref_analytic() {
        // (E, ep, nvl, k, hp). hp==nvl → group is one domain; hp>nvl → spans domains.
        let shapes = [
            (64u32, 8u32, 4u32, 8u32, 4u32),
            (128, 16, 8, 8, 8),
            (256, 32, 8, 8, 8),
            (64, 8, 4, 8, 8),    // hp==ep → every rank resident → dispatch == 0
            (256, 32, 8, 8, 16), // hp == 2·nvl → group spans 2 domains, bcast fires
        ];
        let hidden_bytes = 4096 * 2;
        let n_tokens = 200_000u32;
        let stage_names = STAGES.map(|(name, _)| name);

        let mut exact_max = 0.0f64; // dispatch_inter, inter_bcast, intra_fanout
        for &(e, ep, nvl, k, hp) in &shapes {
            let dist = RoutingDistribution::uniform(e);
            let params = MoeNetParams {
                ep_size: ep,
                nvl_num_gpu: nvl,
                hidden_bytes,
            };
            let placement = Placement::ReplicatedHeadParallel { hp_size: hp };
            let steps = simulate_once(dist.ppm(), k, &params, placement, n_tokens, 0x4850_5F56);
            let hidden_f = f64::from(hidden_bytes);
            let sim_copies: [f64; 6] = std::array::from_fn(|i| {
                steps[i].send_bytes.iter().sum::<u64>() as f64 / hidden_f / f64::from(n_tokens)
            });
            let ref_copies = ref_hp_copies(e, ep, nvl, k, hp);
            let single_target_domain = hp <= nvl;

            println!(
                "\n[E{e} ep{ep} nvl{nvl} k{k} hp{hp}]  ({} target domain(s))",
                if single_target_domain { "1" } else { ">1" }
            );
            for stage in 0..6 {
                let rel = if ref_copies[stage] > 1e-9 {
                    (sim_copies[stage] - ref_copies[stage]) / ref_copies[stage] * 100.0
                } else if sim_copies[stage].abs() < 1e-9 {
                    0.0
                } else {
                    f64::INFINITY
                };
                let agreement = match stage {
                    0 | 4 | 5 => "EXACT", // inter dispatch + placement-only delivery
                    3 if single_target_domain => "EXACT", // inter_reduce exact when 1 target domain
                    1 | 2 => "rail-gap",
                    _ => "root-gap",
                };
                if agreement == "EXACT" {
                    exact_max = exact_max.max(rel.abs());
                }
                println!(
                    "  {:<26} sim={:>9.4} ref={:>9.4} rel={:>+7.2}%  [{agreement}]",
                    stage_names[stage], sim_copies[stage], ref_copies[stage], rel
                );
            }
        }
        println!(
            "\nEXACT-class stages max|rel| = {exact_max:.2}%  (dispatch_inter, inter_bcast,\n\
                  intra_fanout, and inter_reduce@1-domain — same formula in both models)"
        );
        assert!(
            exact_max < 1.0,
            "an EXACT-class stage diverged from ref by {exact_max:.2}%"
        );
    }

    // ── Eval: prebuilt curve interp vs per-T routing truth ───────────────────
    //   uv run cargo test --release --lib \
    //     op::moe::sim::tests::eval_ -- --ignored --nocapture
    //
    // Validates the simulate-as-model design for ALL SIX stages: build per-stage
    // E[max] bottleneck curves ONCE (grid × trials), interpolate at runtime T.
    // Truth = fresh independent T-token routing sims. Hard gate: every stage's
    // bottleneck-byte error ≤ 5% across distributions, shapes, and off-grid T.

    #[test]
    #[ignore = "eval: run explicitly with --ignored --nocapture"]
    fn eval_moe_comm_sim_vs_truth() {
        let dists: Vec<(&str, Box<dyn Fn(u32) -> RoutingDistribution>)> = vec![
            ("uniform", Box::new(RoutingDistribution::uniform)),
            (
                "powerlaw_a0.5",
                Box::new(|e| RoutingDistribution::power_law(e, 0.5)),
            ),
            (
                "powerlaw_a1.0",
                Box::new(|e| RoutingDistribution::power_law(e, 1.0)),
            ),
            (
                "powerlaw_a1.5",
                Box::new(|e| RoutingDistribution::power_law(e, 1.5)),
            ),
            (
                "hot8",
                Box::new(|e| {
                    let mut r = vec![0.01f32; e as usize];
                    for x in r.iter_mut().take(8) {
                        *x = 1.0;
                    }
                    RoutingDistribution::from_profile(&r)
                }),
            ),
        ];
        // (num_experts, ep_size, nvl_num_gpu, top_k)
        let shapes = [(64u32, 8u32, 4u32, 8u32), (128, 16, 8, 8), (256, 32, 8, 8)];
        let grid = [128u32, 512, 2_048, 8_192, 32_768];
        let build_trials = 64u32;
        // Off-grid token counts (interpolation only) across the production range.
        let t_values = [1_000u32, 3_000, 6_000, 12_000, 24_000];
        let trials = 100u32;
        let hidden_bytes = 4096 * 2;
        let stage_names = STAGES.map(|(n, _)| n);

        let mut gate_max = 0.0f64;
        let mut worst = (String::new(), 0.0f64);

        for (dname, make) in &dists {
            for &(e, ep, nvl, k) in &shapes {
                let dist = make(e);
                let p = MoeNetParams {
                    ep_size: ep,
                    nvl_num_gpu: nvl,
                    hidden_bytes,
                };
                let curves = simulate_moe_comm(
                    dist.ppm(),
                    k,
                    &p,
                    Placement::RoundRobin,
                    &grid,
                    build_trials,
                    0xC0FFEE,
                );

                println!("\n[{dname}] E={e} ep={ep} nvl={nvl} k={k}  (model = interp(curve, T))");
                for &t in &t_values {
                    // Truth: average bottleneck over independent T-token sims.
                    let mut truth = [0.0f64; 6];
                    for trial in 0..trials {
                        let seed = 0x5151_2727u64
                            .wrapping_mul(t as u64 + 1)
                            .wrapping_add(u64::from(trial));
                        let steps =
                            simulate_once(dist.ppm(), k, &p, Placement::RoundRobin, t, seed);
                        for (i, tr) in truth.iter_mut().enumerate() {
                            *tr += steps[i].bottleneck_bytes() as f64;
                        }
                    }
                    for tr in &mut truth {
                        *tr /= f64::from(trials);
                    }
                    // Only the 4 active stages (bcast/fanout are 0 under single-home).
                    for stage in [0usize, 1, 2, 3] {
                        let model = curves[stage].bottleneck(u64::from(t)) as f64;
                        let truth_v = truth[stage];
                        if truth_v <= 0.0 {
                            continue;
                        }
                        let rel = (truth_v - model) / truth_v * 100.0;
                        println!(
                            "  T={t:>6} {:<26} model={:>9.4}MB truth={:>9.4}MB rel={:>+7.2}%",
                            stage_names[stage],
                            model / 1e6,
                            truth_v / 1e6,
                            rel
                        );
                        if rel.abs() > worst.1 {
                            worst = (
                                format!("{dname} E{e} ep{ep} k{k} T{t} {}", stage_names[stage]),
                                rel.abs(),
                            );
                        }
                        gate_max = gate_max.max(rel.abs());
                    }
                }
            }
        }
        println!(
            "\nGATE max|rel| over all stages/shapes/T = {gate_max:.2}%   worst: {} -> {:.2}%",
            worst.0, worst.1
        );
        assert!(
            gate_max < 5.0,
            "sim-as-model bottleneck (curve interp) diverged from routing truth by {gate_max:.2}%"
        );
    }
}
