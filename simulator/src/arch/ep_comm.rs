//! Analytic expert-parallel dispatch/combine for the GLM-5.3 NVFP4
//! data-parallel-attention arches beyond one NVLink node (GB200 NVL72).
//!
//! The measured `moe_ep_quantized_all_gather` / `moe_ep_reduce_scatter` and
//! `moe_alltoall` rows span at most 8 GPUs, so an EP group of 16..64 ranks in
//! one NVL72 domain needs a model. This one prices each collective on the
//! busiest rank (the collective ends when it does) from its rows, their width
//! and a fixed cost.
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
//!   this module prices the prepare and the three `moe_comm` calls. One call
//!   of `B`-byte rows costs, on the busiest rank,
//!   `a + max(B / c, rows hypot(p, B / L))` with `rows = max(send, w recv)`:
//!   a fixed cost `a`, flat in the peer count (one CTA per peer); a floor of
//!   one row moved at `c`, which sets the small-message time and grows with
//!   the row width; then a per-row cost that is the issue cost `p` for narrow
//!   rows (the 384-byte scales run at ~140 GB/s) and the link `L` for wide
//!   ones, joined in quadrature (a sum overshoots the 3,072-byte rows, a max
//!   undershoots them). A received row weighs `w < 1` of a sent one: fan-in
//!   arrives from every sender in parallel, egress leaves one GPU.
//!
//! * **Fast exchanges** (`fast_allgather`, `fused_allgather`, `fast_sparse`;
//!   not in vLLM or SGLang, design candidates). No prepare. The dispatch is
//!   one direct-to-peer-memory write of each rank's rows (NVFP4 activations,
//!   block scales, ids and weights in one row) and the combine moves bf16
//!   partial sums (each rank sums its local experts first) by a reduce-scatter
//!   or an all-reduce on the same backend. Each leg is `a + bytes / L` on the
//!   busiest rank, `(a, L)` per backend fitted on B200 8-GPU rows
//!   ([`FastBackend`]); the legs are flagged `extrapolated` in the cost log,
//!   since no B200 row measures them. `fast_allgather` with
//!   `ep_fast_combine: all_reduce` and `ep_comm_model: measured` prices the
//!   combine from the backend's measured all-reduce instead (in the arch).
//!   `fused_allgather` assumes the dispatch write is fused into the
//!   activation-quant kernel and the combine read into the next op: the same
//!   bytes with the fixed cost `a` replaced by `ep_fast_fused_residual_us`
//!   (default 0). `fast_sparse` writes one row per (token, distinct remote
//!   destination rank) both ways, `ep_fanout_ranks` distinct ranks per token
//!   (default the uniform `D`; an expert placement by co-occurrence lowers
//!   it), at the fast model's price. `ep_combine_row_bytes` narrows the
//!   combine row (bf16 `2 H` by default) for every fast kind.
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
//! The all-to-all constants are a fit to the B200 EP8 rows of both backends
//! of the same `moe_comm` kernel: `flashinfer_mnnvl` (FlashInfer 0.6,
//! 2026-10-11; 705 rows over 384-, 3,072- and 12,288-byte rows at 256 slots)
//! and `flashinfer_nvlink_two_sided` (FlashInfer 0.7 `moe_ep`, 2026-11-12;
//! 256-, 2,048-, 2,368- and 8,192-byte rows at 288 slots, plus 54 repeats of
//! the old shapes, which read 0.7-1.0% slower at the median); send x recv
//! 1..65,536; `moe_alltoall_prepare`: 147 rows, 1..65,536 tokens per rank;
//! graph replay. The transfer fit is relative least squares over the 546
//! shapes with both sides >= 8 rows and within 2x of each other, the shapes
//! this model produces. Per row width (median / worst): 256 B +1.7 / +14%,
//! 384 B +1.4 / +16%, 2,048 B -17.5 / -25%, 2,368 B -13.8 / -16%, 3,072 B
//! +4.8 / +21%, 8,192 B -2.0 / +18%, 12,288 B +3.4 / +24%. The 2,048- and
//! 2,368-byte rows read low because the measured per-row cost is not smooth
//! in the row width: 2,048-, 2,368- and 3,072-byte rows all cost ~5.5 ns a
//! row at large counts, which `hypot(p, B / L)` cannot follow; the
//! fast exchange's 2,368-byte dispatch reads 8-16% low and its 8,192-byte
//! combine 3.5-10% low on the send = recv diagonal. Off the band it
//! overprices receive-heavy shapes (4-8x more received than sent: up to
//! +25..+55% at a few hundred to a few thousand rows), and at 1-4 rows it
//! reads 4.2-7.3 us against 3.0-4.3 us measured (not every peer is reached).
//! At 2 and 4 GPUs (8,192-byte rows, 32..1,024 rows) the EP8 constants read
//! +8 / +2% at the median. The prepare is the B200 EP8 intercept and slope
//! over all 147 rows (rms 3.9%, worst +11% at 1 token).
//! Applying them past EP8 assumes the per-call cost stays flat in the peer
//! count and the link and per-row costs do not change on NVL72's
//! inter-tray NVLink; the prepare's peer scaling is an assumption. The
//! all-gather / reduce-scatter link is NVLink5's 900 GB/s per direction
//! times the 0.69 efficiency the MNNVL all-to-all reached on H200 NVLink4.
//! `tools/ep-comm-fit/fit_alltoall.py` refits them and prints the error
//! tables.

use serde::{Deserialize, Serialize};

use crate::timing::expert_demand::ExpertDemand;
use crate::timing::slot_input::EpCommAnalyticLog;
use crate::timing::{CostNode, CostTreeBuilder, CoverageFlags, Evaluator, LeafMetrics, Metrics4};

/// The slot kind of every analytic exchange leaf.
pub const EP_COMM_ANALYTIC_KIND: &str = "ep_comm_analytic";

/// Default effective per-direction NVLink bandwidth per GPU of the
/// all-gather / reduce-scatter: NVLink5 900 GB/s times the MNNVL all-to-all
/// efficiency measured on H200 (0.69).
pub const DEFAULT_LINK_GB_PER_S: f64 = 620.0;
/// The all-to-all's link `L`: B200 EP8 `moe_alltoall` fit (2026-11-12; the
/// 12,288-byte combine tops out at ~625 GB/s, the 8,192-byte one at ~580).
pub const DEFAULT_A2A_LINK_GB_PER_S: f64 = 605.0;
/// One `moe_comm` call's fixed cost `a`: B200 EP8 `moe_alltoall` fit
/// (2026-11-12; floors of 4.5 / 5.0 / 7.4 us at 8..128 rows of 384 / 3,072 /
/// 12,288 bytes).
pub const DEFAULT_A2A_LATENCY_US: f64 = 4.11;
/// The all-to-all's one-row floor `c`, GB/s (B200 EP8 fit, 2026-11-12).
const A2A_ROW_FLOOR_GB_PER_S: f64 = 3.86;
/// The all-to-all's per-row issue cost `p`, us (B200 EP8 fit, 2026-11-12:
/// 2.63 ns; the 256- and 384-byte scales run at ~2.7 ns per row at 64k rows).
const A2A_PER_ROW_US: f64 = 0.002_63;
/// Weight `w` of a received row against a sent one (B200 EP8 fit,
/// 2026-11-12).
const A2A_RECV_ROW_WEIGHT: f64 = 0.9;
/// The all-to-all prepare at EP8: intercept and per-token slope of the B200
/// EP8 `moe_alltoall_prepare` rows, 1..65,536 tokens per rank, with the
/// metadata transfer over [`DEFAULT_A2A_LINK_GB_PER_S`] (2026-11-12; rms 3.9%).
pub const DEFAULT_PREPARE_LATENCY_US: f64 = 21.6;
const PREPARE_US_PER_TOKEN: f64 = 0.0132;
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
    /// Direct-write all-gather dispatch, then a reduce-scatter or all-reduce
    /// of the partial sums on the same backend, no prepare.
    FastAllgather,
    /// [`Self::FastAllgather`] with the dispatch fused into the quant kernel
    /// and the combine into the next op: no fixed cost per leg.
    FusedAllgather,
    /// Direct writes of one row per (token, distinct remote rank) each way.
    FastSparse,
}

impl EpCommKind {
    pub fn is_fast(self) -> bool {
        matches!(
            self,
            Self::FastAllgather | Self::FusedAllgather | Self::FastSparse
        )
    }
}

/// The transport a fast exchange runs on, with its B200 8-GPU fit `(a, L)`
/// (`tools/fast-moe-exchange/fit_fast_collectives.py`, 2026-11-11): one
/// collective's fixed cost and link over the busiest rank's one-direction
/// bytes of a two-shot all-reduce, `2 (N - 1) / N` x message.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastBackend {
    /// FlashInfer MNNVL: the `moe_alltoall` transport kernel (`moe_comm`)
    /// called at the fast exchange's rows, priced by this module's B200 EP8
    /// all-to-all fit (a = 4.11 us, L = 605 GB/s, per-row issue cost, recv
    /// weight 0.9). Cross-check: `all_reduce_fusion` flashinfer_mnnvl TP8
    /// hidden 4096 two-shot rows (17..128 tokens) fit a = 5.37 us at 638 GB/s
    /// (rms 5.0%).
    #[default]
    Mnnvl,
    /// NVSHMEM (`all_reduce` nvshmem 8 GPUs, 4 KiB..64 MiB, 15 rows):
    /// 15.0 us, 729 GB/s; rms 14%, worst +36% (4 KiB) / -26% (256 KiB).
    Nvshmem,
    /// vLLM Lamport Triton (`rs_sconv_ag_residual_rms_norm` TP8 hidden 6144,
    /// 45 token counts 1..16,384, fused short conv + RMSNorm included):
    /// 14.9 us, 495 GB/s; rms 7.3%, worst +22%.
    Lamport,
}

impl FastBackend {
    pub fn latency_us(self) -> f64 {
        match self {
            Self::Mnnvl => DEFAULT_A2A_LATENCY_US,
            Self::Nvshmem => 15.0,
            Self::Lamport => 14.9,
        }
    }

    pub fn link_gb_per_s(self) -> f64 {
        match self {
            Self::Mnnvl => DEFAULT_A2A_LINK_GB_PER_S,
            Self::Nvshmem => 729.0,
            Self::Lamport => 495.0,
        }
    }
}

/// How a fast exchange returns the partial sums.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastCombine {
    /// Each rank receives only its own tokens' partials and sums them.
    #[default]
    ReduceScatter,
    /// Every rank gets every token's sum (two-shot: twice the bytes) and keeps
    /// its own.
    AllReduce,
}

/// The fast exchanges' params (`ep_fast_*`, `ep_fanout_ranks`,
/// `ep_combine_row_bytes`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EpFastConfig {
    pub backend: FastBackend,
    pub combine: FastCombine,
    /// Overrides the backend's fixed cost `a`, us.
    pub latency_us: Option<f64>,
    /// `fused_allgather`: what is left of `a` per leg once fused (default 0).
    pub fused_residual_us: f64,
    /// `fast_sparse`: distinct ranks a token's top-k reach (default: the
    /// routing corpus's mean, else uniform).
    pub fanout_ranks: Option<f64>,
    /// The combine's row bytes (default bf16, `2 H`).
    pub combine_row_bytes: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpCommModel {
    /// Profiled collectives: all-gather / reduce-scatter on 2, 4 or 8 GPUs,
    /// the MNNVL all-to-all on 8 (at this module's rows).
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
    /// Effective per-direction NVLink bandwidth per GPU, GB/s (1e9); `None`
    /// is the kind's default ([`DEFAULT_LINK_GB_PER_S`] or
    /// [`DEFAULT_A2A_LINK_GB_PER_S`]).
    pub link_gb_per_s: Option<f64>,
    /// One all-to-all `moe_comm` call's fixed cost, us.
    pub a2a_latency_us: f64,
    /// Multiplies every collective's fixed cost (all-gather, reduce-scatter
    /// and all-to-all), for sensitivity.
    pub latency_scale: f64,
    /// The all-to-all prepare's fixed cost at EP8, us.
    pub prepare_latency_us: f64,
    /// Multiplies the whole prepare, for sensitivity.
    pub prepare_scale: f64,
    /// The fast exchanges' params; unused by the other kinds.
    pub fast: EpFastConfig,
}

impl Default for EpCommConfig {
    fn default() -> Self {
        Self {
            kind: EpCommKind::default(),
            model: EpCommModel::default(),
            link_gb_per_s: None,
            a2a_latency_us: DEFAULT_A2A_LATENCY_US,
            latency_scale: 1.0,
            prepare_latency_us: DEFAULT_PREPARE_LATENCY_US,
            prepare_scale: 1.0,
            fast: EpFastConfig::default(),
        }
    }
}

impl EpCommConfig {
    pub fn is_default_measured(&self) -> bool {
        self.model == EpCommModel::Measured && self.kind == EpCommKind::AllgatherReducescatter
    }

    /// The link the analytic model prices this kind's rows over, GB/s.
    pub fn link_gb_per_s(&self) -> f64 {
        self.link_gb_per_s.unwrap_or(match self.kind {
            EpCommKind::AllgatherReducescatter => DEFAULT_LINK_GB_PER_S,
            EpCommKind::Alltoall => DEFAULT_A2A_LINK_GB_PER_S,
            EpCommKind::FastAllgather | EpCommKind::FusedAllgather | EpCommKind::FastSparse => {
                self.fast.backend.link_gb_per_s()
            }
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        let positive = |value: f64| value.is_finite() && value > 0.0;
        let non_negative = |value: f64| value.is_finite() && value >= 0.0;
        if !positive(self.link_gb_per_s())
            || !non_negative(self.a2a_latency_us)
            || !non_negative(self.latency_scale)
            || !non_negative(self.prepare_latency_us)
            || !non_negative(self.prepare_scale)
            || !non_negative(self.fast.fused_residual_us)
            || !self.fast.latency_us.is_none_or(non_negative)
            || !self.fast.fanout_ranks.is_none_or(|d| d.is_finite() && d >= 1.0)
            || self.fast.combine_row_bytes == Some(0)
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

/// The all-to-all's dispatch rows on the busiest rank (combine is the
/// transpose).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AlltoallRows {
    /// The busiest rank's tokens: vLLM sizes the prepare and the buffers by it.
    pub busiest_tokens: u32,
    /// Rows leaving the busiest sender, `max(t_r) D`.
    pub send: f64,
    /// Rows reaching the busiest receiver, `T p_max`.
    pub recv: f64,
}

/// One sparse layer's analytic exchange, in eval order.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EpCommStep {
    /// All-to-all only.
    pub prepare: EpCommLeg,
    /// The all-gather, or the all-to-all's NVFP4 activations.
    pub dispatch: EpCommLeg,
    /// All-to-all only: the block scales' own `moe_comm` call.
    pub dispatch_scales: EpCommLeg,
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
    /// A routing corpus's mean distinct ranks per token, `fast_sparse`'s
    /// fan-out when `ep_fanout_ranks` is unset; `None` for popularity.
    pub corpus_fanout: Option<f64>,
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
        let corpus_fanout = if config.kind == EpCommKind::FastSparse {
            demand.corpus_mean_distinct_ranks(local)?
        } else {
            None
        };
        Ok(Self {
            config,
            ep_size,
            hidden,
            top_k,
            uniform_hit,
            corpus_fanout,
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
        self.config.link_gb_per_s() * 1e3
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

    /// One all-to-all `moe_comm` call of `row_bytes`-wide rows on the busiest
    /// rank (the module doc's form).
    pub fn alltoall_call(&self, send: f64, recv: f64, row_bytes: f64) -> EpCommLeg {
        self.alltoall_call_with(send, recv, row_bytes, self.a2a_latency_us())
    }

    /// [`Self::alltoall_call`] with its fixed cost `latency_us`.
    fn alltoall_call_with(
        &self,
        send: f64,
        recv: f64,
        row_bytes: f64,
        latency_us: f64,
    ) -> EpCommLeg {
        let rows = send.max(A2A_RECV_ROW_WEIGHT * recv);
        let per_row_us = A2A_PER_ROW_US.hypot(row_bytes / self.bytes_per_us());
        let floor_us = row_bytes / (A2A_ROW_FLOOR_GB_PER_S * 1e3);
        EpCommLeg {
            max_send_rows: send,
            max_recv_rows: recv,
            bytes: send.max(recv) * row_bytes,
            latency_us,
            time_us: latency_us + floor_us.max(rows * per_row_us),
        }
    }

    /// The all-to-all prepare: the per-token scan of the busiest rank's
    /// `busiest_tokens`, then the ids and weights of the busiest receiver's
    /// `recv` rows over the link.
    pub fn alltoall_prepare(&self, busiest_tokens: u32, recv: f64) -> EpCommLeg {
        let metadata = recv * f64::from(8 * self.top_k);
        let fixed_us = self.config.prepare_scale * self.prepare_fixed_us();
        EpCommLeg {
            max_send_rows: f64::from(busiest_tokens),
            max_recv_rows: recv,
            bytes: metadata,
            latency_us: fixed_us,
            time_us: fixed_us
                + self.config.prepare_scale * PREPARE_US_PER_TOKEN * f64::from(busiest_tokens)
                + metadata / self.bytes_per_us(),
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

    /// The NVFP4 activations of one dispatched row.
    pub fn nvfp4_row_bytes(&self) -> f64 {
        f64::from(self.hidden / 2)
    }

    /// The fp8 block scales of one dispatched row (one per 16 elements).
    pub fn scale_row_bytes(&self) -> f64 {
        f64::from(self.hidden / 16)
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

    /// The all-to-all's busiest-rank rows for one exchange of `tokens` (each
    /// rank's rows, as [`Self::step`] takes them). The measured EP8 path looks
    /// its `moe_alltoall` rows up at exactly these counts.
    pub fn alltoall_rows(&self, tokens: &[u32]) -> AlltoallRows {
        let total: f64 = tokens.iter().map(|&t| f64::from(t)).sum();
        let busiest_tokens = tokens.iter().copied().max().unwrap_or(0);
        if total == 0.0 {
            return AlltoallRows::default();
        }
        let fanout = self.uniform_hit * f64::from(self.ep_size);
        AlltoallRows {
            busiest_tokens,
            send: f64::from(busiest_tokens) * fanout,
            // Never below the mean, `total D / N`: busiest_hit >= D / N.
            recv: total * self.busiest_hit(total.round() as u32),
        }
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
                    dispatch_scales: EpCommLeg::default(),
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
            EpCommKind::FastAllgather | EpCommKind::FusedAllgather | EpCommKind::FastSparse => {
                self.fast_step(tokens)
            }
            EpCommKind::Alltoall => {
                let AlltoallRows {
                    busiest_tokens: busiest,
                    send,
                    recv,
                } = self.alltoall_rows(tokens);
                EpCommStep {
                    prepare: self.alltoall_prepare(busiest, recv),
                    // Activations and scales are two `moe_comm` calls.
                    dispatch: self.alltoall_call(send, recv, self.nvfp4_row_bytes()),
                    dispatch_scales: self.alltoall_call(send, recv, self.scale_row_bytes()),
                    // Each expert rank returns one reduced row per row it got.
                    combine: self.alltoall_call(recv, send, self.bf16_row_bytes()),
                }
            }
        }
    }

    /// A fast leg's fixed cost: the backend's `a` (or the override) x the
    /// latency scale, or the fused residual.
    pub fn fast_latency_us(&self) -> f64 {
        let fast = &self.config.fast;
        if self.config.kind == EpCommKind::FusedAllgather {
            return fast.fused_residual_us;
        }
        self.config.latency_scale * fast.latency_us.unwrap_or(fast.backend.latency_us())
    }

    /// The fast combine's row: bf16 unless `ep_combine_row_bytes` narrows it.
    pub fn fast_combine_row_bytes(&self) -> f64 {
        self.config
            .fast
            .combine_row_bytes
            .map_or(self.bf16_row_bytes(), f64::from)
    }

    /// `fast_sparse`'s distinct ranks per token.
    pub fn fanout_ranks(&self) -> f64 {
        self.config
            .fast
            .fanout_ranks
            .or(self.corpus_fanout)
            .unwrap_or(self.uniform_hit * f64::from(self.ep_size))
            .min(f64::from(self.ep_size))
    }

    /// One fast leg on the busiest rank. MNNVL is the same `moe_alltoall`
    /// transport kernel at the fast exchange's rows (the B200 EP8 fit above,
    /// fixed cost `a` = `fast_latency_us`); the other backends are
    /// `a + max(send, recv) row / L`.
    fn fast_leg(&self, send: f64, recv: f64, row_bytes: f64) -> EpCommLeg {
        match self.config.fast.backend {
            FastBackend::Mnnvl => {
                self.alltoall_call_with(send, recv, row_bytes, self.fast_latency_us())
            }
            FastBackend::Nvshmem | FastBackend::Lamport => {
                self.leg(send, recv, row_bytes, self.fast_latency_us())
            }
        }
    }

    /// The fast exchanges on the busiest rank (module doc). The all-reduce
    /// combine is two-shot: a reduce-scatter, then an all-gather of the sums.
    fn fast_step(&self, tokens: &[u32]) -> EpCommStep {
        let total: f64 = tokens.iter().map(|&t| f64::from(t)).sum();
        let ranks = f64::from(self.ep_size);
        let peers = self.peers();
        let combine_row = self.fast_combine_row_bytes();
        let (mut send, mut recv) = (0.0_f64, 0.0_f64);
        let (remote_out, remote_in) = if self.config.kind == EpCommKind::FastSparse {
            // Of a token's D distinct ranks, its own is one with chance D / N.
            let fanout = self.fanout_ranks();
            (fanout * peers / ranks, fanout / ranks)
        } else {
            (peers, 1.0)
        };
        for &t in tokens {
            let own = f64::from(t);
            send = send.max(own * remote_out);
            recv = recv.max((total - own) * remote_in);
        }
        let dispatch = self.fast_leg(send, recv, self.gather_row_bytes());
        // The combine is the transpose: partials of others' tokens out, own in.
        let mut combine = self.fast_leg(recv, send, combine_row);
        if self.config.fast.combine == FastCombine::AllReduce
            && self.config.kind != EpCommKind::FastSparse
        {
            // Two-shot: the sums go back out to every peer (all-gather leg).
            let gather = self.leg(send, recv, combine_row, 0.0);
            combine = EpCommLeg {
                max_send_rows: combine.max_send_rows + gather.max_send_rows,
                max_recv_rows: combine.max_recv_rows + gather.max_recv_rows,
                bytes: combine.bytes + gather.bytes,
                latency_us: combine.latency_us,
                time_us: combine.time_us + gather.time_us,
            };
        }
        EpCommStep {
            prepare: EpCommLeg::default(),
            dispatch,
            dispatch_scales: EpCommLeg::default(),
            combine,
        }
    }

    pub fn label(&self) -> String {
        if self.config.kind.is_fast() {
            return format!(
                "analytic {:?} on {:?} (a {:.2} us, {} GB/s per direction), combine {:?}",
                self.config.kind,
                self.config.fast.backend,
                self.fast_latency_us(),
                self.config.link_gb_per_s(),
                self.config.fast.combine
            );
        }
        format!(
            "analytic {:?} over {} GB/s per direction; latency x{}",
            self.config.kind,
            self.config.link_gb_per_s(),
            self.config.latency_scale
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
                "link_gb_per_s": self.config.link_gb_per_s(),
                "latency_scale": self.config.latency_scale,
                "prepare_scale": self.config.prepare_scale,
                "fast": self.config.kind.is_fast().then(|| serde_json::json!({
                    "backend": self.config.fast.backend,
                    "combine": self.config.fast.combine,
                    "latency_us": self.fast_latency_us(),
                    "fanout_ranks": (self.config.kind == EpCommKind::FastSparse)
                        .then(|| self.fanout_ranks()),
                    "combine_row_bytes": self.fast_combine_row_bytes(),
                })),
            }),
        )
    }
}

/// Push one leg's leaf.
pub fn push_leg(op: &'static str, leg: EpCommLeg, ev: &mut Evaluator) {
    push_leg_with(op, leg, CoverageFlags::EMPTY, ev);
}

/// Push one leg's leaf, flagged `extrapolated` (a fast exchange's model leg).
pub fn push_extrapolated_leg(op: &'static str, leg: EpCommLeg, ev: &mut Evaluator) {
    push_leg_with(op, leg, CoverageFlags::EXTRAPOLATED, ev);
}

fn push_leg_with(op: &'static str, leg: EpCommLeg, coverage: CoverageFlags, ev: &mut Evaluator) {
    let metrics = LeafMetrics {
        m: Metrics4 {
            time_ms: (leg.time_us / 1e3) as f32,
            flops: 0.0,
            bytes: leg.bytes as f32,
            energy_j: 0.0,
        },
        coverage,
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
        // Activations and scales: two calls of the same rows.
        assert!(close(step.dispatch.latency_us, 4.11));
        assert!(close(step.dispatch_scales.latency_us, 4.11));
        assert_eq!(step.dispatch_scales.max_send_rows, step.dispatch.max_send_rows);
        assert!(close(step.dispatch.bytes, 8.0 * step.dispatch_scales.bytes));
        assert!(close(step.combine.max_send_rows, step.dispatch.max_recv_rows));
        assert!(close(
            step.combine.bytes,
            step.dispatch.max_recv_rows.max(step.dispatch.max_send_rows) * 12288.0
        ));
        // Prepare: fixed part grows 1.75x from EP8 to EP64.
        assert!(close(a2a.prepare_fixed_us(), 21.6 * 1.75));
        assert!(step.prepare.time_us > 21.6 * 1.75);
        // Skew decays to uniform at large exchanges.
        assert!(a2a.busiest_hit(64) >= a2a.busiest_hit(131_072));
        assert!((a2a.busiest_hit(131_072) / a2a.uniform_hit - 1.0).abs() < 0.05);
    }

    fn fast(kind: EpCommKind, fast: EpFastConfig) -> AnalyticEpComm {
        AnalyticEpComm::new(
            EpCommConfig {
                kind,
                model: EpCommModel::Analytic,
                fast,
                ..EpCommConfig::default()
            },
            8,
            4096,
            288,
            8,
            &uniform(288),
        )
        .unwrap()
    }

    /// Catches a fast leg priced off its `a + bytes / L` form or the wrong
    /// rows: GLM-5.3-Flash k7 c128, 128 rows on each of 8 ranks.
    #[test]
    fn fast_exchanges_price_the_busiest_rank_bytes() {
        let tokens = [128_u32; 8];
        let m2 = fast(EpCommKind::FastAllgather, EpFastConfig::default());
        let step = m2.step(&tokens);
        // Dispatch: 896 gathered rows of 2,048 + 256 + 64 B in, 7 x 128 out.
        assert!(close(step.dispatch.max_recv_rows, 896.0));
        assert!(close(step.dispatch.bytes, 896.0 * 2368.0));
        // MNNVL: the moe_alltoall fit at 896 x 896 rows.
        assert!(close(step.dispatch.time_us, m2.alltoall_call(896.0, 896.0, 2368.0).time_us));
        // 8.34 us against 9.61 us measured (flashinfer_nvlink_two_sided, 288
        // slots): the fit reads the 2,368-byte rows low (module doc).
        assert!((step.dispatch.time_us - 8.34).abs() < 0.01, "{}", step.dispatch.time_us);
        // Combine: the bf16 partials of everyone else's tokens.
        assert!(close(step.combine.bytes, 896.0 * 8192.0));
        assert!(close(step.combine.time_us, m2.alltoall_call(896.0, 896.0, 8192.0).time_us));
        assert_eq!(step.prepare, EpCommLeg::default());
        // All-reduce: twice the combine bytes, one fixed cost.
        let ar = fast(
            EpCommKind::FastAllgather,
            EpFastConfig {
                combine: FastCombine::AllReduce,
                ..EpFastConfig::default()
            },
        )
        .step(&tokens);
        assert!(close(
            ar.combine.time_us,
            step.combine.time_us + 896.0 * 8192.0 / 605e3
        ));
        // Fused: the same bytes without the fixed cost.
        let m3 = fast(EpCommKind::FusedAllgather, EpFastConfig::default()).step(&tokens);
        assert!(close(m3.dispatch.time_us, step.dispatch.time_us - 4.11));
        assert!(close(m3.combine.time_us, step.combine.time_us - 4.11));
        // Sparse: D distinct ranks per token, D (N-1)/N of them remote.
        let sparse = fast(EpCommKind::FastSparse, EpFastConfig::default());
        let d = sparse.fanout_ranks();
        assert!(close(d, 8.0 * uniform_hit_probability(288, 8, 8)));
        let s = sparse.step(&tokens);
        assert!(close(s.dispatch.max_send_rows, 128.0 * d * 7.0 / 8.0));
        assert!(close(s.dispatch.max_recv_rows, 896.0 * d / 8.0));
        let placed = fast(
            EpCommKind::FastSparse,
            EpFastConfig {
                fanout_ranks: Some(3.0),
                combine_row_bytes: Some(4224),
                backend: FastBackend::Nvshmem,
                ..EpFastConfig::default()
            },
        );
        let p = placed.step(&tokens);
        assert!(close(p.combine.bytes, 896.0 * 3.0 / 8.0 * 4224.0));
        assert!(close(p.combine.latency_us, 15.0));
        assert_eq!(placed.config.link_gb_per_s(), 729.0);
    }

    /// The all-to-all constants are the 2026-11-12 fit to the B200 EP8
    /// `flashinfer_mnnvl` and `flashinfer_nvlink_two_sided` rows; the targets
    /// below are `flashinfer_mnnvl` rows (us, graph replay, send = recv).
    /// Re-fitting moves this test on purpose.
    #[test]
    fn the_alltoall_fit_reproduces_the_b200_ep8_rows() {
        assert_eq!(DEFAULT_A2A_LINK_GB_PER_S, 605.0);
        assert_eq!(DEFAULT_A2A_LATENCY_US, 4.11);
        assert_eq!(A2A_ROW_FLOOR_GB_PER_S, 3.86);
        assert_eq!(A2A_PER_ROW_US, 0.002_63);
        assert_eq!(A2A_RECV_ROW_WEIGHT, 0.9);
        assert_eq!(DEFAULT_PREPARE_LATENCY_US, 21.6);
        assert_eq!(PREPARE_US_PER_TOKEN, 0.0132);
        // The all-gather / reduce-scatter keep their link.
        let ag = comm(EpCommKind::AllgatherReducescatter, 8);
        assert_eq!(ag.config.link_gb_per_s(), 620.0);
        let a2a = comm(EpCommKind::Alltoall, 8);
        assert_eq!(a2a.config.link_gb_per_s(), 605.0);
        assert_eq!(
            EpCommConfig {
                link_gb_per_s: Some(900.0),
                ..a2a.config.clone()
            }
            .link_gb_per_s(),
            900.0
        );
        let within = |model: f64, measured: f64, tolerance: f64, what: &str| {
            let error = model / measured - 1.0;
            assert!(error.abs() <= tolerance, "{what}: {model:.2} vs {measured:.2} us");
        };
        // (rows, NVFP4 activations, block scales, bf16 combine) on the diagonal.
        for (rows, x, scales, combine) in [
            (64.0, 4.97, 4.52, 7.38),
            (1_024.0, 10.05, 7.18, 24.46),
            (8_192.0, 49.25, 27.02, 168.58),
            (65_536.0, 372.23, 180.16, 1_290.41),
        ] {
            for (bytes, measured) in [(3_072.0, x), (384.0, scales), (12_288.0, combine)] {
                let model = a2a.alltoall_call(rows, rows, bytes).time_us;
                within(model, measured, 0.06, &format!("{bytes} B x {rows}"));
            }
        }
        // Off the diagonal a received row weighs 0.9 of a sent one (the
        // 12,288-byte rows read ~+5% since the 2026-11-12 refit).
        within(a2a.alltoall_call(8_192.0, 4_096.0, 12_288.0).time_us, 162.67, 0.07, "send-heavy");
        within(a2a.alltoall_call(4_096.0, 8_192.0, 12_288.0).time_us, 145.57, 0.07, "recv-heavy");
        // Prepare at the busiest rank's tokens, uniform EP8 receive.
        for (tokens, measured) in [(64, 22.56), (1_024, 36.83), (8_192, 131.55), (24_576, 342.57)] {
            let recv = f64::from(tokens) * 8.0 * a2a.uniform_hit;
            let model = a2a.alltoall_prepare(tokens, recv).time_us;
            within(model, measured, 0.06, &format!("prepare x {tokens}"));
        }
    }
}
