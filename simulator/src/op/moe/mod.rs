//! `op/moe` — MoE dispatch/combine compound ops (L2), unified on the
//! **simulate-as-model** design.
//!
//! One [`sim::simulate_moe_comm`] realizes per-token routing + placement once at
//! op build time and prices ALL six network stages' per-GPU bytes, sampling each
//! stage's realized bottleneck on a token-count grid into a [`BottleneckCurve`]
//! (the L2 analogue of an L1 1D kernel cache: the op's config — routing, top_k,
//! placement, shape — is fixed at init, and the single sweep axis `T` is
//! interpolated at runtime). [`dispatch::MoeDispatchOp`] reads the two dispatch
//! curves; [`combine::MoeCombineOp`] reads the four combine curves. Both price
//! against the `p2p_intra` / `p2p_inter` L1 kernels.

pub mod combine;
pub mod dispatch;
pub mod sim;

use crate::common::Fabric;
use crate::timing::bridge::DType;
use crate::timing::kernels::{
    P2pInterKernel, P2pInterKernelConfig, P2pIntraKernel, P2pIntraKernelConfig,
};
use crate::timing::routing::RoutingDistribution;

pub use combine::MoeCombineOp;
pub use dispatch::MoeDispatchOp;
pub use sim::simulate_moe_comm;

/// Op-level identity shared by [`MoeDispatchOp`] and [`MoeCombineOp`]. The two
/// ops are leaf-views over the SAME unified 6-stage simulation, so they MUST be
/// built from the same parameters — sharing one config type makes that an
/// invariant of the type system rather than a convention.
#[derive(Clone, Debug)]
pub struct MoeNetConfig {
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub dtype: DType,
    /// Fabric for the intra-NVL p2p leg (e.g. `Nvlink`).
    pub intra_fabric: Fabric,
    /// Fabric for the inter-NVL (NIC) p2p leg (e.g. `Infiniband`).
    pub inter_fabric: Fabric,
    pub ep_size: u32,
    pub nvl_num_gpu: u32,
    /// Hidden dimension; one token's transferred vector is `h × dtype_bytes`.
    pub h: u32,
    pub top_k: u32,
    pub routing: RoutingDistribution,
    pub placement: Placement,
}

impl MoeNetConfig {
    /// The intra-domain (NVLink) p2p sub-kernel config. Comm is size-keyed, so
    /// `self.dtype` does not enter here — it drives only the `hidden_bytes`
    /// message width in `net_params` (fp8 → half the bytes on the same curve).
    pub(crate) fn p2p_intra_config(&self) -> P2pIntraKernelConfig {
        P2pIntraKernelConfig {
            backends: self.backends.clone(),
            gpu_name: self.gpu_name.clone(),
            fabric: self.intra_fabric,
        }
    }

    /// The inter-domain (NIC) p2p sub-kernel config. Size-keyed (see above).
    pub(crate) fn p2p_inter_config(&self) -> P2pInterKernelConfig {
        P2pInterKernelConfig {
            backends: self.backends.clone(),
            gpu_name: self.gpu_name.clone(),
            fabric: self.inter_fabric,
        }
    }

    /// The static network shape fed to the per-token simulator.
    pub(crate) fn net_params(&self) -> MoeNetParams {
        MoeNetParams {
            ep_size: self.ep_size,
            nvl_num_gpu: self.nvl_num_gpu,
            hidden_bytes: self.h * self.dtype.size_bytes(),
        }
    }
}

/// One MoE comm op invocation: just the token count `T`. Same shape for both
/// dispatch and combine (placement is config-time, so only `T` varies at
/// runtime — the single curve sweep axis).
#[derive(Clone, Debug, Default)]
pub struct MoeNetInput {
    pub tokens: u64,
}

/// Token-count grid the per-stage bottleneck is sampled on (geometric, covering
/// the production decode/prefill range and beyond). Interpolated at runtime — the
/// L2 analogue of an L1 1D cache's `message_size` grid.
pub(crate) const SIM_T_GRID: [u32; 5] = [128, 512, 2_048, 8_192, 32_768];
/// Draws averaged per grid point to estimate `E[max]` (the realized busiest GPU).
/// One-time per config, so a generous count keeps the curve smooth.
pub(crate) const SIM_TRIALS: u32 = 64;
/// Fixed seed → deterministic curve → bit-identical sim.
pub(crate) const SIM_SEED: u64 = 0x_4D6F_4543_6F6D_6200; // "MoEComb\0"

/// Token placement — the config-time identity that fixes which rank(s) each token
/// resides on (dispatch source / combine sink), the L2 analogue of an L1 kernel's
/// static `Config` dims. Fixed at op build → one curve set per config; only the
/// token count `T` varies at runtime.
///
/// A token resides on a CONTIGUOUS group of ranks `[start, start+count)`:
/// * [`Placement::RoundRobin`] → `count == 1` (pure EP, one home rank).
/// * [`Placement::ReplicatedHeadParallel`] → `count == hp_size` (the token is
///   replicated across its head-parallel group, so combine must deliver the
///   reduced output to all `hp_size` ranks — this is what makes the
///   `combine_inter_bcast` / `combine_intra_fanout` stages non-zero).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Balanced EP: token `i` homes on rank `i % ep_size`, spreading home duty
    /// evenly so no rank is a structural dispatch/combine hotspot.
    RoundRobin,
    /// HP-replicated: token `i`'s owner is `i % ep_size`, and it resides on the
    /// whole contiguous head-parallel group of `hp_size` ranks containing that
    /// owner. `hp_size == nvl_num_gpu` makes the group exactly one NVL domain.
    ReplicatedHeadParallel { hp_size: u32 },
}

impl Placement {
    /// The contiguous residing group `(start_rank, count)` of `token_idx`. The
    /// owner rotates as `token_idx % ep_size` so the group coverage is uniform
    /// over many tokens (matching ref's average over every owner rank).
    pub fn residing_group(self, token_idx: u64, ep_size: u32) -> (u16, u16) {
        let ep = ep_size.max(1);
        let owner = (token_idx % u64::from(ep)) as u32;
        match self {
            Placement::RoundRobin => (owner as u16, 1),
            Placement::ReplicatedHeadParallel { hp_size } => {
                let hp = hp_size.clamp(1, ep);
                let start = owner / hp * hp;
                let count = hp.min(ep - start);
                (start as u16, count as u16)
            }
        }
    }
}

/// Which p2p fabric tier a stage rides — selects the intra (NVLink) vs inter
/// (NIC) L1 kernel at lookup time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum P2pTier {
    IntraDomain,
    InterDomain,
}

/// Static network shape shared by the stage pricer.
#[derive(Debug, Clone, Copy)]
pub struct MoeNetParams {
    pub ep_size: u32,
    pub nvl_num_gpu: u32,
    /// Bytes of one token's hidden vector (`h × dtype_bytes`).
    pub hidden_bytes: u32,
}

/// One MoE network stage's per-rank byte loads (indexed by global rank;
/// `len() == ep_size`).
#[derive(Debug, Clone)]
pub struct MoeStep {
    pub name: &'static str,
    pub tier: P2pTier,
    pub send_bytes: Vec<u64>,
    pub recv_bytes: Vec<u64>,
}

impl MoeStep {
    pub fn empty(name: &'static str, tier: P2pTier, ep_size: usize) -> Self {
        Self {
            name,
            tier,
            send_bytes: vec![0u64; ep_size],
            recv_bytes: vec![0u64; ep_size],
        }
    }

    /// Bottleneck rank's byte load — `max_r max(send[r], recv[r])` — the size fed
    /// to the p2p curve for this stage (ref's `max_gpu`).
    pub fn bottleneck_bytes(&self) -> u64 {
        self.send_bytes
            .iter()
            .zip(&self.recv_bytes)
            .map(|(&s, &r)| s.max(r))
            .max()
            .unwrap_or(0)
    }
}

/// Contiguous NVL-domain layout: ranks `[start, start+size)` per domain.
pub struct NvlLayout {
    domain_starts: Vec<u32>,
    domain_sizes: Vec<u32>,
}

impl NvlLayout {
    pub fn new(ep_size: u32, nvl_num_gpu: u32) -> Self {
        let sizes = crate::timing::routing::ranks_per_nvl_domain(ep_size, nvl_num_gpu);
        let mut starts = Vec::with_capacity(sizes.len());
        let mut acc = 0u32;
        for &n in &sizes {
            starts.push(acc);
            acc += n;
        }
        Self {
            domain_starts: starts,
            domain_sizes: sizes,
        }
    }

    pub fn num_domains(&self) -> usize {
        self.domain_sizes.len()
    }

    pub fn domain_size(&self, domain: usize) -> u32 {
        self.domain_sizes[domain]
    }

    pub fn first_rank(&self, domain: usize) -> u16 {
        self.domain_starts[domain] as u16
    }

    /// The contiguous global-rank range of `domain`.
    pub fn ranks_in(&self, domain: usize) -> std::ops::Range<u16> {
        let start = self.domain_starts[domain] as u16;
        start..(start + self.domain_sizes[domain] as u16)
    }

    pub fn domain_of(&self, rank: u16) -> usize {
        let r = u32::from(rank);
        for (i, &start) in self.domain_starts.iter().enumerate() {
            if r >= start && r < start + self.domain_sizes[i] {
                return i;
            }
        }
        self.domain_starts.len().saturating_sub(1)
    }

    /// Rail-aligned peer of `rank` in `domain`: the rank at the same local index
    /// (wrapped to the domain size). Models a NIC link landing on the matching
    /// position in the remote NVL domain (DeepEP-style rail alignment).
    pub fn rail_peer(&self, rank: u16, domain: usize) -> u16 {
        let src_dom = self.domain_of(rank);
        let local = u32::from(rank) - self.domain_starts[src_dom];
        let size = self.domain_sizes[domain].max(1);
        self.first_rank(domain) + (local % size) as u16
    }
}

/// A network stage's realized-bottleneck-bytes-vs-token-count curve: `E[max]`
/// over GPUs sampled at each build grid point, linearly interpolated by the
/// runtime token count. The L2 analogue of L1's `Cache1DLinear` over
/// `message_size` — here the axis is the token count `T`.
#[derive(Clone, Debug)]
pub struct BottleneckCurve {
    pub tier: P2pTier,
    /// `(t, bottleneck_bytes)` sorted ascending by `t`.
    pub points: Vec<(u32, f64)>,
}

impl BottleneckCurve {
    /// Interpolate this stage's bottleneck bytes at `tokens` and forward the
    /// lookup to the matching p2p L1 kernel (intra=NVLink, inter=NIC), recording
    /// the leaf into `ev`. Shared by both `MoeDispatchOp` and `MoeCombineOp` —
    /// the curve knows its own tier, so each op just hands over both p2p kernels
    /// and lets the curve route itself.
    pub(crate) fn push_to(
        &self,
        tokens: u64,
        p2p_intra: &P2pIntraKernel,
        p2p_inter: &P2pInterKernel,
        ev: &mut crate::timing::Evaluator,
    ) {
        use crate::timing::kernels::{P2pInterKernelInput, P2pIntraKernelInput};
        use crate::timing::LeafMetrics;
        let message_size_bytes = self.bottleneck(tokens);
        // A zero-byte stage is one that does not occur under this config — e.g.
        // the inter (NIC) legs when `nvl_num_gpu >= ep_size` (single NVL domain,
        // no cross-domain hop), or `combine_inter_bcast` / `combine_intra_fanout`
        // under a single-home placement. The per-token sim assigned it zero
        // send/recv bytes: NO collective is launched, so the cost is zero. Push a
        // zero leaf (keeping the slot/taxonomy stable) WITHOUT a p2p lookup — the
        // p2p curve returns a nonzero launch-overhead floor at size 0, which would
        // otherwise bill a phantom ~30µs (inter) / ~109µs (intra) per absent stage
        // across every layer.
        if message_size_bytes == 0 {
            match self.tier {
                P2pTier::IntraDomain => {
                    let input = P2pIntraKernelInput { message_size_bytes };
                    ev.push(LeafMetrics::ZERO, || input.clone().into());
                }
                P2pTier::InterDomain => {
                    let input = P2pInterKernelInput { message_size_bytes };
                    ev.push(LeafMetrics::ZERO, || input.clone().into());
                }
            }
            return;
        }
        match self.tier {
            P2pTier::IntraDomain => {
                let input = P2pIntraKernelInput { message_size_bytes };
                ev.push(p2p_intra.eval(&input), || input.clone().into());
            }
            P2pTier::InterDomain => {
                let input = P2pInterKernelInput { message_size_bytes };
                ev.push(p2p_inter.eval(&input), || input.clone().into());
            }
        }
    }

    /// Bytes at `tokens`, linearly interpolated between bracketing grid points
    /// (extrapolated by the end segments' slope outside the grid). The curve
    /// `≈ a·T + b·√T` is smooth, so linear interp on a geometric grid is tight.
    pub fn bottleneck(&self, tokens: u64) -> u64 {
        let p = &self.points;
        if p.is_empty() {
            return 0;
        }
        let tf = tokens as f64;
        if p.len() == 1 {
            let (t0, b0) = p[0];
            return (b0 * tf / f64::from(t0.max(1))).max(0.0).round() as u64;
        }
        let (lo, hi) = if tf <= f64::from(p[0].0) {
            (p[0], p[1])
        } else if tf >= f64::from(p[p.len() - 1].0) {
            (p[p.len() - 2], p[p.len() - 1])
        } else {
            let mut idx = 0;
            for i in 0..p.len() - 1 {
                if tf <= f64::from(p[i + 1].0) {
                    idx = i;
                    break;
                }
            }
            (p[idx], p[idx + 1])
        };
        let (t0, b0) = (f64::from(lo.0), lo.1);
        let (t1, b1) = (f64::from(hi.0), hi.1);
        let slope = (b1 - b0) / (t1 - t0);
        (b0 + slope * (tf - t0)).max(0.0).round() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cfg() -> MoeNetConfig {
        MoeNetConfig {
            backends: vec!["nccl"],
            gpu_name: "H100".to_string(),
            dtype: DType::Bf16,
            intra_fabric: Fabric::Nvlink,
            inter_fabric: Fabric::Infiniband,
            ep_size: 8,
            nvl_num_gpu: 4,
            h: 4096,
            top_k: 8,
            routing: RoutingDistribution::uniform(64),
            placement: Placement::RoundRobin,
        }
    }

    #[test]
    fn net_params_compute_hidden_bytes_from_h_and_dtype() {
        let p = sample_cfg().net_params();
        assert_eq!(p.ep_size, 8);
        assert_eq!(p.nvl_num_gpu, 4);
        assert_eq!(p.hidden_bytes, 4096 * 2);
    }

    #[test]
    fn p2p_sub_configs_carry_their_fabrics() {
        let cfg = sample_cfg();
        assert_eq!(cfg.p2p_intra_config().fabric, Fabric::Nvlink);
        assert_eq!(cfg.p2p_inter_config().fabric, Fabric::Infiniband);
    }

    /// An absent stage's curve is identically zero, so `bottleneck` returns 0 and
    /// `push_to` takes its zero-leaf short-circuit (asserted here at the curve
    /// level; the short-circuit itself needs no p2p lookup, so it cannot pick up
    /// the kernel's nonzero size-0 launch-overhead floor — the phantom ~30µs/
    /// ~109µs that absent inter/bcast/fanout stages used to bill every layer).
    #[test]
    fn absent_stage_curve_is_zero_so_push_short_circuits() {
        let curve = BottleneckCurve {
            tier: P2pTier::InterDomain,
            points: vec![(128, 0.0), (32_768, 0.0)],
        };
        // Zero at every probed T (on-grid, off-grid, and extrapolated).
        for t in [0u64, 1, 8_192, 50_000] {
            assert_eq!(curve.bottleneck(t), 0, "absent-stage curve must be 0 at T={t}");
        }
    }
}
