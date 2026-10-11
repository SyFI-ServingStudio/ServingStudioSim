//! A rough analytic model of SGLang HiSparse: hierarchical sparse-attention KV
//! offload for a DSA (DeepSeek-V3.2-style) model.
//!
//! Sources: the SGLang HiSparse docs and arXiv 2608.07009.
//!
//! ## What HiSparse does
//!
//! The decode side keeps each request's whole MLA latent cache in host pinned
//! DRAM. The GPU keeps:
//!
//! * the DSA indexer keys for the whole context, for the request's life; and
//! * a fixed per-request LRU device buffer of `buffer_tokens` latent slots per
//!   layer.
//!
//! Each step the indexer picks `top_k` positions per layer; those not in the
//! buffer are copied host -> device over PCIe, then sparse attention reads the
//! buffer. A context that fits in the buffer takes a no-miss fast path. GLM-5.2
//! IndexShare layers let selections be prefetched on a side stream, which hides
//! part of the copy.
//!
//! ## What this module models
//!
//! * **Capacity** ([`HisparseConfig::kv_host_offload`]): per-request GPU bytes
//!   are `ctx * indexer + buffer_tokens * latent` (plus whatever windowed
//!   cache the caller adds); host bytes are `ctx * latent`, capped per KV
//!   partition at its share of the node's host DRAM. The buffer is charged in
//!   full even when `ctx < buffer_tokens`: an upper bound for short requests.
//! * **Misses** ([`HisparseConfig::misses_per_layer`]): a step of verify width
//!   `w` queries `w` positions per request, each with its own top-k. Their
//!   union per layer is assumed to be `U = k (1 + (w - 1)(1 - o))`, `o` the
//!   intra-step overlap of consecutive positions' selections. The miss fraction
//!   of `U` follows the paper's LRU measurements at `r = buffer / U` (see
//!   [`miss_fraction`]). Every attention layer misses: an IndexShare layer
//!   reuses its leader's selection but holds its own latent.
//! * **IO cost** ([`HisparseConfig::rank_step`]): bytes are misses times the
//!   latent bytes per layer-token times the layers; the exposed time is
//!   `exposed_io_fraction * bytes / h2d bandwidth`. Kernels are unchanged. The
//!   device -> host backup of the step's new tokens is asynchronous and not
//!   billed.
//!
//! Prefill chunks are not charged any IO: HiSparse is a decode-side mechanism.
//!
//! ## Calibration caveat
//!
//! The miss curve is three points from one GLM-5.1 trace (k = 2048, 100K
//! context, width-1 decode). The width-`w` union overlap `o` is an assumption,
//! not a measurement.

use crate::arch::contract::KvHostOffload;
use crate::timing::slot_input::HisparseH2dIoLog;
use crate::timing::{CostNode, CostTreeBuilder, CoverageFlags, Evaluator, LeafMetrics, Metrics4};

/// The paper's LRU miss fraction of `top_k` per layer per step at buffer /
/// selection ratio 1, 2 and 4 (GLM-5.1, k = 2048, 100K context, width 1).
const MISS_CURVE: [(f64, f64); 3] = [(1.0, 0.30), (2.0, 0.134), (4.0, 0.067)];
/// Below ratio 1 the buffer cannot hold one step's selection and thrashes:
/// `f = 1 - slope * r`, continuous with the curve at `r = 1`.
const THRASH_SLOPE: f64 = 1.0 - MISS_CURVE[0].1;

/// The slot kind of the analytic host -> device IO leaf.
pub const HISPARSE_H2D_IO_KIND: &str = "hisparse_h2d_io";

const fn default_buffer_tokens() -> u32 {
    4096
}
const fn default_top_k() -> u32 {
    2048
}
const fn default_host_dram_gb_per_node() -> f64 {
    2000.0
}
const fn default_h2d_gb_per_s() -> f64 {
    50.0
}
const fn default_half() -> f64 {
    0.5
}

/// HiSparse parameters, from the arch's `hisparse_*` params
/// (`IterArchSel::hisparse`). Absent: the cache is GPU resident.
#[derive(Clone, Debug, PartialEq)]
pub struct HisparseConfig {
    /// Per-request LRU device buffer, in latent token slots per layer.
    pub buffer_tokens: u32,
    /// DSA indexer selection per layer per query position.
    pub top_k: u32,
    /// Host DRAM per node available to the offloaded latent cache, in GB (1e9).
    pub host_dram_gb_per_node: f64,
    /// GPUs sharing one node's host DRAM. Default: the arch's NVLink domain
    /// (`nvl_num_gpu`), which is one node on an HGX B200.
    pub gpus_per_node: Option<u16>,
    /// Effective host -> device bandwidth per GPU, GB/s (B200 PCIe Gen5 x16).
    pub h2d_gb_per_s: f64,
    /// Share of the copy not hidden by the IndexShare side-stream prefetch.
    pub exposed_io_fraction: f64,
    /// Overlap of consecutive query positions' top-k inside one verify step.
    pub intra_step_overlap: f64,
}

impl Default for HisparseConfig {
    fn default() -> Self {
        Self {
            buffer_tokens: default_buffer_tokens(),
            top_k: default_top_k(),
            host_dram_gb_per_node: default_host_dram_gb_per_node(),
            gpus_per_node: None,
            h2d_gb_per_s: default_h2d_gb_per_s(),
            exposed_io_fraction: default_half(),
            intra_step_overlap: default_half(),
        }
    }
}

/// LRU miss fraction of the per-layer selection union at `ratio = buffer /
/// union`.
///
/// `ratio < 1`: `1 - 0.70 ratio` (the buffer thrashes). `1 <= ratio <= 4`:
/// piecewise log-log interpolation through the paper's three points, so each
/// segment is a power law (the 2 -> 4 segment is exactly `1 / ratio`).
/// `ratio > 4`: clamped at the last point, 0.067 -- no extrapolation.
pub fn miss_fraction(ratio: f64) -> f64 {
    assert!(ratio.is_finite() && ratio >= 0.0, "buffer ratio {ratio}");
    if ratio < MISS_CURVE[0].0 {
        return 1.0 - THRASH_SLOPE * ratio;
    }
    for pair in MISS_CURVE.windows(2) {
        let ((r0, f0), (r1, f1)) = (pair[0], pair[1]);
        if ratio <= r1 {
            let t = (ratio / r0).ln() / (r1 / r0).ln();
            return (f0.ln() + t * (f1 / f0).ln()).exp();
        }
    }
    MISS_CURVE[MISS_CURVE.len() - 1].1
}

/// One rank's HiSparse work in one step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HisparseRankStep {
    /// Decode requests on the rank.
    pub decode_requests: u32,
    /// Those whose context exceeds the buffer (the rest take the fast path).
    pub offloaded_requests: u32,
    /// Sum of the decode requests' KV lengths (the host tier's occupancy).
    pub context_tokens: u64,
    /// Misses per layer, summed over the rank's requests.
    pub misses_per_layer: f64,
    /// Host -> device bytes over all layers.
    pub io_bytes: f64,
    /// Exposed copy time added to the step.
    pub exposed_ms: f64,
}

impl HisparseConfig {
    /// Distinct positions one request selects per layer in a step of `width`
    /// query rows: `k (1 + (width - 1)(1 - overlap))`.
    pub fn selection_union(&self, width: u32) -> f64 {
        assert!(width >= 1, "a step queries at least one position");
        let extra = f64::from(width - 1) * (1.0 - self.intra_step_overlap);
        f64::from(self.top_k) * (1.0 + extra)
    }

    /// Host -> device misses per layer for one request of `context` tokens in
    /// a step of `width` query rows; zero when the context fits the buffer.
    pub fn misses_per_layer(&self, context: u32, width: u32) -> f64 {
        if context <= self.buffer_tokens {
            return 0.0;
        }
        let union = self.selection_union(width);
        miss_fraction(f64::from(self.buffer_tokens) / union) * union
    }

    /// GPUs one node's host DRAM is split across.
    fn gpus_per_node(&self, nvl_num_gpu: u16) -> u16 {
        self.gpus_per_node.unwrap_or(nvl_num_gpu).max(1)
    }

    /// Capacity facts for one KV partition of `gpus_per_partition` GPUs.
    /// `latent_bytes_per_token` is the latent over every layer on that
    /// partition; `extra_device_bytes_per_request` any further fixed GPU cache
    /// (a windowed draft).
    pub fn kv_host_offload(
        &self,
        latent_bytes_per_token: u64,
        extra_device_bytes_per_request: u64,
        nvl_num_gpu: u16,
        gpus_per_partition: u16,
    ) -> KvHostOffload {
        let host_bytes_per_gpu =
            self.host_dram_gb_per_node * 1e9 / f64::from(self.gpus_per_node(nvl_num_gpu));
        KvHostOffload {
            device_bytes_per_request: u64::from(self.buffer_tokens) * latent_bytes_per_token
                + extra_device_bytes_per_request,
            host_bytes_per_token: latent_bytes_per_token,
            host_bytes_per_partition: (host_bytes_per_gpu * f64::from(gpus_per_partition)) as u64,
        }
    }

    /// One rank's step: `decode_kv_lens` are its decode requests' KV lengths,
    /// `width` the query rows each presents, `latent_bytes_per_layer_token` and
    /// `layers` the latent cache shape.
    pub fn rank_step(
        &self,
        decode_kv_lens: &[u32],
        width: u32,
        latent_bytes_per_layer_token: u64,
        layers: u32,
    ) -> HisparseRankStep {
        let mut step = HisparseRankStep {
            decode_requests: decode_kv_lens.len() as u32,
            ..HisparseRankStep::default()
        };
        for &context in decode_kv_lens {
            step.context_tokens += u64::from(context);
            let misses = self.misses_per_layer(context, width);
            if misses > 0.0 {
                step.offloaded_requests += 1;
                step.misses_per_layer += misses;
            }
        }
        step.io_bytes =
            step.misses_per_layer * latent_bytes_per_layer_token as f64 * f64::from(layers);
        step.exposed_ms =
            self.exposed_io_fraction * step.io_bytes / (self.h2d_gb_per_s * 1e9) * 1e3;
        step
    }

    /// One leaf per rank under a `Max` over ranks: the ranks meet at the next
    /// expert-parallel collective, so the slowest rank's copy is exposed.
    pub fn compile(&self, name: &str, ranks: u16, builder: &mut CostTreeBuilder) -> CostNode {
        let config = serde_json::json!({
            "backends": [],
            "buffer_tokens": self.buffer_tokens,
            "top_k": self.top_k,
            "h2d_gb_per_s": self.h2d_gb_per_s,
            "exposed_io_fraction": self.exposed_io_fraction,
            "intra_step_overlap": self.intra_step_overlap,
        });
        CostNode::Labeled {
            label: format!(
                "hisparse host->device miss copy [analytic; B={} k={}; {}x exposed at {} GB/s; \
                 Max over DP ranks]",
                self.buffer_tokens, self.top_k, self.exposed_io_fraction, self.h2d_gb_per_s
            ),
            child: Box::new(CostNode::Max {
                overlap: 1.0,
                children: (0..ranks)
                    .map(|_| {
                        builder.leaf(
                            format!("{name}.hisparse.h2d_miss_io"),
                            HISPARSE_H2D_IO_KIND,
                            config.clone(),
                        )
                    })
                    .collect(),
            }),
        }
    }
}

/// Push one rank's leaf, in [`HisparseConfig::compile`] order.
pub fn push_rank_step(step: HisparseRankStep, ev: &mut Evaluator) {
    let metrics = LeafMetrics {
        m: Metrics4 {
            time_ms: step.exposed_ms as f32,
            flops: 0.0,
            bytes: step.io_bytes as f32,
            energy_j: 0.0,
        },
        coverage: CoverageFlags::EMPTY,
        backend_index: LeafMetrics::NO_BACKEND,
    };
    ev.push(metrics, || {
        HisparseH2dIoLog {
            decode_requests: step.decode_requests,
            offloaded_requests: step.offloaded_requests,
            context_tokens: step.context_tokens,
            misses_per_layer: step.misses_per_layer,
            io_bytes: step.io_bytes,
        }
        .into()
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9 * b.abs().max(1.0)
    }

    #[test]
    fn the_miss_curve_passes_through_the_paper_points_and_clamps() {
        assert!(close(miss_fraction(1.0), 0.30));
        assert!(close(miss_fraction(2.0), 0.134));
        assert!(close(miss_fraction(4.0), 0.067));
        assert!(close(miss_fraction(8.0), 0.067));
        // Thrash branch, continuous at 1.
        assert!(close(miss_fraction(0.0), 1.0));
        assert!(close(miss_fraction(0.5), 0.65));
        // The 2 -> 4 segment is 1 / r.
        assert!(close(miss_fraction(3.0), 0.134 * 2.0 / 3.0));
        // Monotone non-increasing.
        let mut last = f64::INFINITY;
        for i in 0..=100 {
            let f = miss_fraction(f64::from(i) * 0.06);
            assert!(f <= last + 1e-12);
            last = f;
        }
    }

    #[test]
    fn width_one_reproduces_the_paper() {
        let at = |buffer_tokens| HisparseConfig {
            buffer_tokens,
            ..HisparseConfig::default()
        };
        // r = B / k at width 1.
        assert!(close(at(2048).misses_per_layer(100_000, 1), 0.30 * 2048.0));
        assert!(close(at(4096).misses_per_layer(100_000, 1), 0.134 * 2048.0));
        assert_eq!(at(4096).misses_per_layer(100_000, 1).round(), 274.0);
        assert!(close(at(8192).misses_per_layer(100_000, 1), 0.067 * 2048.0));
        // A context that fits the buffer never misses.
        assert_eq!(at(4096).misses_per_layer(4096, 1), 0.0);
        assert_eq!(at(4096).misses_per_layer(4096, 8), 0.0);
    }

    #[test]
    fn verify_widths_grow_the_union_and_the_misses() {
        let at = |buffer_tokens| HisparseConfig {
            buffer_tokens,
            ..HisparseConfig::default()
        };
        // w = 4: U = 2048 * 2.5 = 5120.
        let u4 = 5120.0;
        assert!(close(at(4096).selection_union(4), u4));
        // B = 4096: r = 0.8 thrashes, f = 1 - 0.7 * 0.8 = 0.44.
        assert!(close(at(4096).misses_per_layer(100_000, 4), 0.44 * u4));
        // B = 8192: r = 1.6, log-log between (1, .30) and (2, .134).
        let f = (0.30_f64.ln() + (1.6_f64).log2() * (0.134_f64 / 0.30).ln()).exp();
        assert!(close(at(8192).misses_per_layer(100_000, 4), f * u4));
        // w = 8: U = 2048 * 4.5 = 9216.
        let u8 = 9216.0;
        assert!(close(at(8192).selection_union(8), u8));
        let r = 4096.0 / u8;
        assert!(close(
            at(4096).misses_per_layer(100_000, 8),
            (1.0 - 0.7 * r) * u8
        ));
        let r = 8192.0 / u8;
        assert!(close(
            at(8192).misses_per_layer(100_000, 8),
            (1.0 - 0.7 * r) * u8
        ));
        // More query rows never miss less.
        for buffer in [4096, 8192] {
            let mut last = 0.0;
            for width in 1..=8 {
                let misses = at(buffer).misses_per_layer(100_000, width);
                assert!(misses >= last);
                last = misses;
            }
        }
    }

    #[test]
    fn capacity_splits_the_node_host_dram_and_charges_the_buffer() {
        let config = HisparseConfig::default();
        let offload = config.kv_host_offload(44_928, 25_165_824, 8, 1);
        assert_eq!(offload.device_bytes_per_request, 4096 * 44_928 + 25_165_824);
        assert_eq!(offload.host_bytes_per_token, 44_928);
        assert_eq!(offload.host_bytes_per_partition, 250_000_000_000);
        let explicit = HisparseConfig {
            gpus_per_node: Some(4),
            ..HisparseConfig::default()
        };
        assert_eq!(
            explicit
                .kv_host_offload(44_928, 0, 8, 2)
                .host_bytes_per_partition,
            1_000_000_000_000
        );
    }

    #[test]
    fn a_rank_step_sums_misses_and_bills_the_exposed_share() {
        let config = HisparseConfig::default();
        let step = config.rank_step(&[100_000, 1_000, 200_000], 1, 576, 78);
        assert_eq!(step.decode_requests, 3);
        assert_eq!(step.offloaded_requests, 2);
        assert_eq!(step.context_tokens, 301_000);
        assert!(close(step.misses_per_layer, 2.0 * 0.134 * 2048.0));
        let bytes = step.misses_per_layer * 576.0 * 78.0;
        assert!(close(step.io_bytes, bytes));
        assert!(close(step.exposed_ms, 0.5 * bytes / 50e9 * 1e3));
        assert_eq!(
            config.rank_step(&[], 8, 576, 78),
            HisparseRankStep::default()
        );
    }
}
