//! Analytic expert-parallel dispatch/combine for the GLM-5.3 NVFP4
//! data-parallel-attention arches beyond one NVLink node (GB200 NVL72).
//!
//! The measured `moe_ep_quantized_all_gather` / `moe_ep_reduce_scatter` rows
//! span at most 8 GPUs, so an EP group of 16..64 ranks in one NVL72 domain
//! needs a model. This one is a link-bandwidth plus fixed-latency model per
//! collective, priced on the busiest rank (the collective ends when it does).
//!
//! ## Exchanges
//!
//! * **All-gather / reduce-scatter** (vLLM's naive DP/EP prepare-finalize,
//!   the measured path at EP <= 8). Rank `r` with `t_r` tokens receives
//!   `sum(t) - t_r` gathered rows of `H/2 + H/16 + 8 top_k` bytes (NVFP4
//!   activations, fp8 block scales, int32 ids and fp32 weights) and sends its
//!   own rows to `N - 1` peers; the reduce-scatter moves bf16 partial sums the
//!   other way. Latency is linear in the peer count, fitted through the
//!   measured B200 EP4 and EP8 small-message floors (NCCL's grouped
//!   all-gatherv/reduce-scatterv step once per peer).
//! * **All-to-all** (`--all2all-backend flashinfer_nvlink_two_sided`,
//!   `prepare_finalize/flashinfer_nvlink_two_sided.py`). A metadata prepare
//!   (`mnnvl_moe_alltoallv_prepare_without_allgather`), then the NVFP4
//!   activations and their block scales in two `moe_comm` calls (one row per
//!   (token, distinct destination rank)), the scale swizzle on the receiver,
//!   the experts, and the combine: a zeroed `tokens x top_k` bf16 staging
//!   buffer, one `moe_comm` of bf16 rows back, and a `torch.sum` over top-k.
//!   The swizzle, fill and sum are measured elementwise leaves in the arch;
//!   this module prices the prepare and the two transfers. One `moe_comm`
//!   costs `latency + max(send, recv) bytes / link` on the busiest rank: one
//!   CTA per peer, so the latency is flat in the peer count.
//!
//! ## Rows
//!
//! A token's top-k experts reach `D = N (1 - C(E - E/N, k) / C(E, k))`
//! distinct ranks on average (uniform, without replacement), so the busiest
//! sender sends `max(t_r) D` rows. The busiest receiver gets `T p_max`, `T`
//! the tokens in the exchange and `p_max` the chance a token reaches the
//! busiest rank: the routing demand's folded histogram at `T` gives the
//! busiest rank's share `s` of the `T k` (token, expert) pairs, mapped to a
//! hit probability `1 - (1 - s)^k`, normalized so a uniform share reproduces
//! `D / N`. The corpus's co-occurrence inside one rank is ignored on both
//! sides (an upper bound on rows).
//!
//! ## Calibration caveat
//!
//! The link default is NVLink5's 900 GB/s per direction times the 0.69
//! efficiency the flashinfer MNNVL all-to-all reaches on H200 NVLink4
//! (~310 of 450 GB/s); the all-to-all latency is the H200 graph-replay floor
//! (a vLLM trace reads 25-50% higher at decode); the prepare is the H200 EP8
//! fit. The peer scaling of the prepare is an assumption.

use serde::{Deserialize, Serialize};

use crate::timing::expert_demand::ExpertDemand;
use crate::timing::slot_input::EpCommAnalyticLog;
use crate::timing::{CostNode, CostTreeBuilder, CoverageFlags, Evaluator, LeafMetrics, Metrics4};

/// The slot kind of every analytic exchange leaf.
pub const EP_COMM_ANALYTIC_KIND: &str = "ep_comm_analytic";

/// Default effective per-direction NVLink bandwidth per GPU: NVLink5 900 GB/s
/// times the MNNVL all-to-all efficiency measured on H200 (0.69).
pub const DEFAULT_LINK_GB_PER_S: f64 = 620.0;
/// One `moe_comm` call's floor: the H200 EP8 graph-replay fit at <= 1024 rows.
pub const DEFAULT_A2A_LATENCY_US: f64 = 6.0;
/// The all-to-all prepare at EP8: intercept and per-token slope of the H200
/// EP8 `moe_alltoall_prepare` rows up to 2,048 tokens per rank.
pub const DEFAULT_PREPARE_LATENCY_US: f64 = 16.8;
const PREPARE_US_PER_TOKEN: f64 = 0.0147;
/// Growth of the prepare's fixed part per doubling of the EP group past 8
/// (more peers to signal and wait on; assumption).
const PREPARE_GROWTH_PER_DOUBLING: f64 = 0.25;
/// B200 `vllm_pynccl` small-message floors: `a + b (N - 4)` through the EP4
/// and EP8 balanced rows (all-gather ~21 / ~36 us, reduce-scatter ~18 /
/// ~26.5 us).
const AG_LATENCY_EP4_US: f64 = 21.0;
const AG_LATENCY_PER_PEER_US: f64 = 3.75;
const RS_LATENCY_EP4_US: f64 = 18.0;
const RS_LATENCY_PER_PEER_US: f64 = 2.125;
/// Largest exchange size the receive-skew table samples; beyond it the skew
/// is held (it tends to 1).
const SKEW_TABLE_MAX_LOG2: u32 = 17;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpCommKind {
    /// vLLM's naive DP/EP prepare-finalize: quantized all-gather, then a bf16
    /// reduce-scatter.
    #[default]
    AllgatherReducescatter,
    /// flashinfer's two-sided MNNVL all-to-all.
    Alltoall,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpCommModel {
    /// Profiled collectives (all-gather / reduce-scatter on 2, 4 or 8 GPUs).
    #[default]
    Measured,
    /// This module's link model.
    Analytic,
}

/// The arch's `ep_*` params.
#[derive(Clone, Debug, PartialEq)]
pub struct EpCommConfig {
    pub kind: EpCommKind,
    pub model: EpCommModel,
    /// Effective per-direction NVLink bandwidth per GPU, GB/s (1e9).
    pub link_gb_per_s: f64,
    /// One all-to-all `moe_comm` call's fixed cost, us.
    pub a2a_latency_us: f64,
    /// Multiplies every collective's fixed cost (all-gather, reduce-scatter
    /// and all-to-all), for sensitivity.
    pub latency_scale: f64,
    /// The all-to-all prepare's fixed cost at EP8, us.
    pub prepare_latency_us: f64,
    /// Multiplies the whole prepare, for sensitivity.
    pub prepare_scale: f64,
}

impl Default for EpCommConfig {
    fn default() -> Self {
        Self {
            kind: EpCommKind::default(),
            model: EpCommModel::default(),
            link_gb_per_s: DEFAULT_LINK_GB_PER_S,
            a2a_latency_us: DEFAULT_A2A_LATENCY_US,
            latency_scale: 1.0,
            prepare_latency_us: DEFAULT_PREPARE_LATENCY_US,
            prepare_scale: 1.0,
        }
    }
}

impl EpCommConfig {
    pub fn is_default_measured(&self) -> bool {
        self.model == EpCommModel::Measured && self.kind == EpCommKind::AllgatherReducescatter
    }

    pub fn validate(&self) -> Result<(), String> {
        let positive = |value: f64| value.is_finite() && value > 0.0;
        let non_negative = |value: f64| value.is_finite() && value >= 0.0;
        if !positive(self.link_gb_per_s)
            || !non_negative(self.a2a_latency_us)
            || !non_negative(self.latency_scale)
            || !non_negative(self.prepare_latency_us)
            || !non_negative(self.prepare_scale)
        {
            return Err(format!("invalid ep_comm params {self:?}"));
        }
        Ok(())
    }
}

/// `D / N`: the chance one token's `top_k` distinct experts (uniform over
/// `num_experts`) include one of a given rank's `num_experts / ep_size`.
pub fn uniform_hit_probability(num_experts: u32, ep_size: u32, top_k: u32) -> f64 {
    let local = num_experts / ep_size;
    if local + top_k > num_experts {
        // Too few experts elsewhere to fill the top-k: always hit.
        return 1.0;
    }
    let mut miss = 1.0;
    for i in 0..top_k {
        miss *= f64::from(num_experts - local - i) / f64::from(num_experts - i);
    }
    1.0 - miss
}

/// One exchange leg on the busiest rank.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EpCommLeg {
    pub max_send_rows: f64,
    pub max_recv_rows: f64,
    pub bytes: f64,
    pub latency_us: f64,
    pub time_us: f64,
}

/// One sparse layer's analytic exchange, in eval order.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EpCommStep {
    /// All-to-all only.
    pub prepare: EpCommLeg,
    pub dispatch: EpCommLeg,
    pub combine: EpCommLeg,
}

/// The analytic exchange of one EP group.
#[derive(Clone, Debug)]
pub struct AnalyticEpComm {
    pub config: EpCommConfig,
    pub ep_size: u16,
    pub hidden: u32,
    pub top_k: u32,
    /// `D / N` under uniform routing.
    pub uniform_hit: f64,
    /// `(log2 tokens, busiest-rank hit probability)` ascending.
    recv_hit: Vec<(f64, f64)>,
}

impl AnalyticEpComm {
    /// `demand` is the body's routed demand at this EP size; its folded
    /// histograms give the busiest rank's share of the pairs.
    pub fn new(
        config: EpCommConfig,
        ep_size: u16,
        hidden: u32,
        num_experts: u32,
        top_k: u32,
        demand: &ExpertDemand,
    ) -> anyhow::Result<Self> {
        let ranks = u32::from(ep_size);
        anyhow::ensure!(
            ranks >= 2 && num_experts % ranks == 0 && top_k <= num_experts - num_experts / ranks,
            "ep_size {ep_size} must be >= 2 and divide {num_experts} experts"
        );
        let local = (num_experts / ranks) as usize;
        let uniform_hit = uniform_hit_probability(num_experts, ranks, top_k);
        let uniform_share = 1.0 / f64::from(ranks);
        let hit = |share: f64| 1.0 - (1.0 - share.min(1.0)).powi(top_k as i32);
        let norm = uniform_hit / hit(uniform_share);
        let tokens: Vec<u32> = (0..=SKEW_TABLE_MAX_LOG2).map(|power| 1_u32 << power).collect();
        let folded =
            demand.per_expert_batches(top_k, &tokens, num_experts as usize, local, 0)?;
        let recv_hit = tokens
            .iter()
            .zip(folded)
            .map(|(&count, histogram)| {
                let pairs = f64::from(count) * f64::from(top_k);
                let busiest = histogram
                    .chunks(local)
                    .map(|rank| rank.iter().map(|&rows| f64::from(rows)).sum::<f64>())
                    .fold(0.0, f64::max);
                let share = (busiest / pairs).max(uniform_share);
                (f64::from(count).log2(), (hit(share) * norm).min(1.0))
            })
            .collect();
        Ok(Self {
            config,
            ep_size,
            hidden,
            top_k,
            uniform_hit,
            recv_hit,
        })
    }

    /// The busiest receiver's chance of being reached by one token of a
    /// `tokens`-token exchange (log-linear in the table, held at its ends).
    pub fn busiest_hit(&self, tokens: u32) -> f64 {
        let x = f64::from(tokens.max(1)).log2();
        let table = &self.recv_hit;
        if x <= table[0].0 {
            return table[0].1;
        }
        for pair in table.windows(2) {
            let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
            if x <= x1 {
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
            }
        }
        table[table.len() - 1].1
    }

    fn bytes_per_us(&self) -> f64 {
        self.config.link_gb_per_s * 1e3
    }

    fn leg(&self, send: f64, recv: f64, row_bytes: f64, latency_us: f64) -> EpCommLeg {
        let bytes = send.max(recv) * row_bytes;
        EpCommLeg {
            max_send_rows: send,
            max_recv_rows: recv,
            bytes,
            latency_us,
            time_us: latency_us + bytes / self.bytes_per_us(),
        }
    }

    fn peers(&self) -> f64 {
        f64::from(self.ep_size) - 1.0
    }

    /// Gathered row: NVFP4 activations, fp8 block scales, int32 top-k ids and
    /// fp32 top-k weights.
    pub fn gather_row_bytes(&self) -> f64 {
        f64::from(self.hidden / 2 + self.hidden / 16 + 8 * self.top_k)
    }

    /// One bf16 hidden row.
    pub fn bf16_row_bytes(&self) -> f64 {
        f64::from(self.hidden * 2)
    }

    /// NVFP4 activations plus block scales of one dispatched row.
    pub fn dispatch_row_bytes(&self) -> f64 {
        f64::from(self.hidden / 2 + self.hidden / 16)
    }

    pub fn ag_latency_us(&self) -> f64 {
        self.config.latency_scale
            * (AG_LATENCY_EP4_US + AG_LATENCY_PER_PEER_US * (f64::from(self.ep_size) - 4.0))
                .max(0.0)
    }

    pub fn rs_latency_us(&self) -> f64 {
        self.config.latency_scale
            * (RS_LATENCY_EP4_US + RS_LATENCY_PER_PEER_US * (f64::from(self.ep_size) - 4.0))
                .max(0.0)
    }

    pub fn a2a_latency_us(&self) -> f64 {
        self.config.latency_scale * self.config.a2a_latency_us
    }

    pub fn prepare_fixed_us(&self) -> f64 {
        let doublings = (f64::from(self.ep_size) / 8.0).log2();
        let growth = (1.0 + PREPARE_GROWTH_PER_DOUBLING * doublings).max(0.5);
        self.config.prepare_latency_us * growth
    }

    /// One sparse layer's exchange; `tokens` are each rank's rows in the
    /// exchange (padded, with idle ranks' dummy token).
    pub fn step(&self, tokens: &[u32]) -> EpCommStep {
        let total: f64 = tokens.iter().map(|&t| f64::from(t)).sum();
        if total == 0.0 {
            return EpCommStep::default();
        }
        match self.config.kind {
            EpCommKind::AllgatherReducescatter => {
                let peers = self.peers();
                // The busiest rank on each side: max of own rows times peers
                // (what it sends / is reduced into) and everyone else's rows.
                let (gather_send, gather_recv) = tokens.iter().fold((0.0_f64, 0.0_f64), |acc, &t| {
                    let own = f64::from(t);
                    (acc.0.max(own * peers), acc.1.max(total - own))
                });
                EpCommStep {
                    prepare: EpCommLeg::default(),
                    dispatch: self.leg(
                        gather_send,
                        gather_recv,
                        self.gather_row_bytes(),
                        self.ag_latency_us(),
                    ),
                    combine: self.leg(
                        gather_recv,
                        gather_send,
                        self.bf16_row_bytes(),
                        self.rs_latency_us(),
                    ),
                }
            }
            EpCommKind::Alltoall => {
                let busiest = tokens.iter().copied().max().unwrap_or(0);
                let fanout = self.uniform_hit * f64::from(self.ep_size);
                let send = f64::from(busiest) * fanout;
                // Never below the mean, `total D / N`: busiest_hit >= D / N.
                let recv = total * self.busiest_hit(total.round() as u32);
                // Prepare: the per-token scan, then the ids and weights of
                // every received row over the link.
                let metadata = recv * f64::from(8 * self.top_k);
                let prepare_us = self.config.prepare_scale
                    * (self.prepare_fixed_us() + PREPARE_US_PER_TOKEN * f64::from(busiest))
                    + metadata / self.bytes_per_us();
                let latency = self.a2a_latency_us();
                // Activations and scales are two `moe_comm` calls.
                let dispatch = self.leg(send, recv, self.dispatch_row_bytes(), 2.0 * latency);
                EpCommStep {
                    prepare: EpCommLeg {
                        max_send_rows: f64::from(busiest),
                        max_recv_rows: recv,
                        bytes: metadata,
                        latency_us: self.config.prepare_scale * self.prepare_fixed_us(),
                        time_us: prepare_us,
                    },
                    dispatch,
                    // Each expert rank returns one reduced row per row it got.
                    combine: self.leg(recv, send, self.bf16_row_bytes(), latency),
                }
            }
        }
    }

    pub fn label(&self) -> String {
        format!(
            "analytic {:?} over {} GB/s per direction; latency x{}",
            self.config.kind, self.config.link_gb_per_s, self.config.latency_scale
        )
    }

    /// One leaf: the slowest rank's leg (the collective ends with it).
    pub fn compile_leg(&self, name: String, op: &str, builder: &mut CostTreeBuilder) -> CostNode {
        builder.leaf(
            name,
            EP_COMM_ANALYTIC_KIND,
            serde_json::json!({
                "backends": [],
                "op": op,
                "kind": self.config.kind,
                "ep_size": self.ep_size,
                "link_gb_per_s": self.config.link_gb_per_s,
                "latency_scale": self.config.latency_scale,
                "prepare_scale": self.config.prepare_scale,
            }),
        )
    }
}

/// Push one leg's leaf.
pub fn push_leg(op: &'static str, leg: EpCommLeg, ev: &mut Evaluator) {
    let metrics = LeafMetrics {
        m: Metrics4 {
            time_ms: (leg.time_us / 1e3) as f32,
            flops: 0.0,
            bytes: leg.bytes as f32,
            energy_j: 0.0,
        },
        coverage: CoverageFlags::EMPTY,
        backend_index: LeafMetrics::NO_BACKEND,
    };
    ev.push(metrics, || {
        EpCommAnalyticLog {
            op,
            max_send_rows: leg.max_send_rows,
            max_recv_rows: leg.max_recv_rows,
            bytes: leg.bytes,
            latency_us: leg.latency_us,
        }
        .into()
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(num_experts: usize) -> ExpertDemand {
        let share = 1_000_000 / num_experts as u32;
        let mut layer = vec![share; num_experts];
        layer[0] += 1_000_000 - share * num_experts as u32;
        ExpertDemand::Popularity {
            layerwise_global_ppm: vec![layer; 4],
        }
    }

    fn comm(kind: EpCommKind, ep_size: u16) -> AnalyticEpComm {
        AnalyticEpComm::new(
            EpCommConfig {
                kind,
                model: EpCommModel::Analytic,
                ..EpCommConfig::default()
            },
            ep_size,
            6144,
            256,
            8,
            &uniform(256),
        )
        .unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * b.abs().max(1.0)
    }

    #[test]
    fn the_uniform_fanout_is_exact() {
        // EP8: 1 - C(224, 8) / C(256, 8).
        let mut miss = 1.0;
        for i in 0..8 {
            miss *= f64::from(224 - i) / f64::from(256 - i);
        }
        assert!(close(uniform_hit_probability(256, 8, 8), 1.0 - miss));
        // One rank owns everything: always hit.
        assert!(close(uniform_hit_probability(256, 1, 8), 1.0));
        // EP64: 4 experts per rank, ~0.12 of tokens reach a given rank.
        let p = uniform_hit_probability(256, 64, 8);
        assert!(p > 0.11 && p < 0.125, "{p}");
    }

    #[test]
    fn allgather_reducescatter_moves_every_peer_row() {
        let ag = comm(EpCommKind::AllgatherReducescatter, 8);
        let step = ag.step(&[128; 8]);
        assert!(close(step.dispatch.max_recv_rows, 7.0 * 128.0));
        assert!(close(step.dispatch.bytes, 7.0 * 128.0 * 3520.0));
        assert!(close(step.dispatch.latency_us, 36.0));
        assert!(close(
            step.dispatch.time_us,
            36.0 + 7.0 * 128.0 * 3520.0 / 620e3
        ));
        assert!(close(step.combine.bytes, 7.0 * 128.0 * 12288.0));
        assert!(close(step.combine.latency_us, 26.5));
        assert_eq!(step.prepare, EpCommLeg::default());
        assert_eq!(ag.step(&[0; 8]), EpCommStep::default());
        // Latency grows with peers.
        assert!(close(comm(EpCommKind::AllgatherReducescatter, 64).ag_latency_us(), 246.0));
    }

    #[test]
    fn alltoall_sends_one_row_per_destination_rank() {
        let a2a = comm(EpCommKind::Alltoall, 64);
        let step = a2a.step(&[32; 64]);
        let fanout = 64.0 * a2a.uniform_hit;
        assert!(close(step.dispatch.max_send_rows, 32.0 * fanout));
        // The busiest receiver is at least the mean and the sampled skew is
        // bounded.
        let mean = 64.0 * 32.0 * a2a.uniform_hit;
        assert!(step.dispatch.max_recv_rows >= mean * 0.999);
        assert!(step.dispatch.max_recv_rows <= mean * 2.5);
        assert!(close(step.dispatch.latency_us, 12.0));
        assert!(close(step.combine.max_send_rows, step.dispatch.max_recv_rows));
        assert!(close(
            step.combine.bytes,
            step.dispatch.max_recv_rows.max(step.dispatch.max_send_rows) * 12288.0
        ));
        // Prepare: fixed part grows 1.75x from EP8 to EP64.
        assert!(close(a2a.prepare_fixed_us(), 16.8 * 1.75));
        assert!(step.prepare.time_us > 16.8 * 1.75);
        // Skew decays to uniform at large exchanges.
        assert!(a2a.busiest_hit(64) >= a2a.busiest_hit(131_072));
        assert!((a2a.busiest_hit(131_072) / a2a.uniform_hit - 1.0).abs() < 0.05);
    }
}
