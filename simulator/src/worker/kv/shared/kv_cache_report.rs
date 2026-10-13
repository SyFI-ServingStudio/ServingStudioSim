//! One KV pool's capacity, restated as numbers: the layout the arch declared,
//! the worker options that shaped it, what [`KvAccounting`] charges each part,
//! and what that leaves for requests of `max_model_len` tokens.
//!
//! Built once per pool at construction ([`KvAccounting::report`]) and carried
//! with the pool's facts to L7, which writes `raw/kv_cache.json` and prints
//! `simulator kv-cache`. Nothing here is read on the hot path.

use serde::Serialize;

use crate::arch::kv_layout::{KvCacheLayout, KvGrowth};
use crate::worker::kv::shared::kv_accounting::{
    charged_bytes_per_entry, state_copies, KvAccounting, KvAccountingOptions, KvOverhead,
    PrefixResumePoints,
};

/// One KV pool's report. Bytes are over one KV partition, as in the layout;
/// divide by `attention_shards` for one GPU's.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KvCacheReport {
    /// GPUs one partition spans; each holds an equal share of its bytes.
    pub attention_shards: u16,
    /// The arch's declared layout.
    pub layout: KvCacheLayout,
    pub options: KvOptionsReport,
    pub accounting: KvAccountingReport,
    pub result: KvCapacityReport,
}

/// The worker options, as resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct KvOptionsReport {
    pub kv_overhead: KvOverhead,
    pub prefix_caching: bool,
    pub prefix_resume_points: PrefixResumePoints,
    pub prefix_resume_interval_tokens: Option<u32>,
    pub max_in_flight_tokens: Option<u32>,
}

/// What the accounting charges.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct KvAccountingReport {
    /// The partition's budget: the per-GPU KV budget times its ranks.
    pub budget_bytes: u64,
    /// Engine bytes taken out of the budget before sizing (0 under `none`).
    pub charged_fixed_device_bytes: u64,
    pub bytes_per_unit: u64,
    pub capacity_units: u64,
    /// Units the worker set aside out of the resolved capacity (PD decode's
    /// pull backlog); `capacity_units` is what remains.
    pub reserved_units: u64,
    /// Tokens per-token parts are allocated in; 1 when blocks are not charged.
    pub charged_block_tokens: u64,
    pub parts: Vec<KvPartChargeReport>,
}

/// What one part is charged, in the layout's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct KvPartChargeReport {
    pub name: &'static str,
    /// The padded entry under `vllm`, the model's own under `none`.
    pub charged_bytes_per_entry: u64,
    pub units_per_entry: u64,
    /// Copies a running request holds; 1 for a part that grows with context.
    pub held_copies: u32,
    /// For a per-request part, whether vLLM's concurrency estimate adds align
    /// mode's previous-step copy to `held_copies`; null for any other part.
    pub align_copy: Option<bool>,
}

/// What the pool holds, at the worker's `max_model_len`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KvCapacityReport {
    /// Null when the arch cannot give one.
    pub max_model_len: Option<u32>,
    /// Units vLLM's concurrency estimate charges one `max_model_len`-token
    /// request: what admitting it needs plus any align copies.
    pub units_per_request_at_max_model_len: Option<u64>,
    /// Units admitting and running that request holds (no align copy).
    pub admission_units_at_max_model_len: Option<u64>,
    /// The estimate's units split over the layout's parts, in its order;
    /// empty when `max_model_len` is unknown.
    pub request_parts: Vec<KvRequestPartReport>,
    /// vLLM's "Maximum concurrency for N tokens per request".
    pub max_concurrency_at_max_model_len: Option<f64>,
    pub other_kv_overhead: KvOverhead,
    /// The same concurrency resolved under `other_kv_overhead`, with the same
    /// reserved bytes set aside; null when that mode cannot resolve the
    /// options.
    pub other_max_concurrency_at_max_model_len: Option<f64>,
    /// The longest context the whole pool holds, when every part grows per
    /// token; null otherwise.
    pub capacity_tokens: Option<u64>,
}

/// What one part holds for a `max_model_len`-token request in vLLM's
/// concurrency estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct KvRequestPartReport {
    pub name: &'static str,
    /// Context tokens held: the block-rounded request for a per-token part,
    /// the kept window for a sliding window; null for per-request state.
    pub tokens_held: Option<u64>,
    /// Entries charged: tokens over `tokens_per_entry`, window tokens, or
    /// state copies (the align copy included).
    pub entries: u64,
    /// Bytes over one KV partition.
    pub bytes: u64,
}

impl KvAccounting {
    /// This pool's report at `max_model_len`. Resolves the layout a second
    /// time under the other `kv_overhead` mode for the comparison, so build it
    /// once per pool.
    pub(crate) fn report(&self, max_model_len: Option<u32>) -> KvCacheReport {
        let layout = self.layout();
        let options = self.options();
        let overhead = options.overhead;
        let parts = layout
            .parts
            .iter()
            .map(|part| {
                let charged_bytes_per_entry = charged_bytes_per_entry(part, overhead);
                let (held_copies, align_copy) = match part.growth {
                    KvGrowth::PerRequest => {
                        let (copies, aligns) = state_copies(part, layout, options);
                        (copies, Some(aligns))
                    }
                    KvGrowth::PerToken { .. } | KvGrowth::SlidingWindow { .. } => (1, None),
                };
                KvPartChargeReport {
                    name: part.name,
                    charged_bytes_per_entry,
                    units_per_entry: charged_bytes_per_entry / self.bytes_per_unit(),
                    held_copies,
                    align_copy,
                }
            })
            .collect();
        let charged_fixed_device_bytes = match overhead {
            KvOverhead::Vllm => layout.engine.fixed_device_bytes,
            KvOverhead::None => 0,
        };

        let other_kv_overhead = overhead.other();
        let other = KvAccounting::resolve(
            layout,
            KvAccountingOptions {
                overhead: other_kv_overhead,
                ..options
            },
            self.budget_bytes(),
        )
        .ok()
        .map(|other| {
            let reserved_bytes = self.reserved_units() * self.bytes_per_unit();
            other.reserving(reserved_bytes / other.bytes_per_unit())
        });
        let model_len = max_model_len.map(u64::from);
        let grows_per_token = layout
            .parts
            .iter()
            .all(|part| matches!(part.growth, KvGrowth::PerToken { .. }));

        let request_parts: Vec<KvRequestPartReport> = model_len
            .map(|len| {
                let split = self.estimated_request_by_part(len);
                debug_assert_eq!(
                    split.iter().map(|part| part.units).sum::<u64>(),
                    self.estimated_request_units(len),
                    "the per-part split must sum to the estimate's charge"
                );
                layout
                    .parts
                    .iter()
                    .zip(split)
                    .map(|(part, held)| KvRequestPartReport {
                        name: part.name,
                        tokens_held: held.tokens_held,
                        entries: held.entries,
                        bytes: held.units * self.bytes_per_unit(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        KvCacheReport {
            attention_shards: self.attention_shards(),
            layout: layout.clone(),
            options: KvOptionsReport {
                kv_overhead: overhead,
                prefix_caching: options.prefix_caching,
                prefix_resume_points: self.resume_points(),
                prefix_resume_interval_tokens: options.resume_interval,
                max_in_flight_tokens: options.max_in_flight_tokens,
            },
            accounting: KvAccountingReport {
                budget_bytes: self.budget_bytes(),
                charged_fixed_device_bytes,
                bytes_per_unit: self.bytes_per_unit(),
                capacity_units: self.capacity_units(),
                reserved_units: self.reserved_units(),
                charged_block_tokens: self.block_tokens(),
                parts,
            },
            result: KvCapacityReport {
                max_model_len,
                units_per_request_at_max_model_len: model_len
                    .map(|len| self.estimated_request_units(len)),
                admission_units_at_max_model_len: model_len.map(|len| self.needed_to_admit(len)),
                request_parts,
                max_concurrency_at_max_model_len: model_len.map(|len| self.concurrency_at(len)),
                other_kv_overhead,
                other_max_concurrency_at_max_model_len: other
                    .zip(model_len)
                    .map(|(other, len)| other.concurrency_at(len)),
                capacity_tokens: grows_per_token
                    .then(|| self.tokens_fitting(self.capacity_units())),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::kv_layout::{EngineOverhead, KvCachePart};

    /// Per-token global attention, a 64-token window and a padded recurrent
    /// state, with a 16-token engine block and a 1000-byte fixed buffer.
    fn hybrid_layout() -> KvCacheLayout {
        KvCacheLayout::new()
            .with_part(KvCachePart::per_token("global", 100))
            .with_part(KvCachePart::sliding_window("swa", 64, 100))
            .with_part(KvCachePart::per_request("state", 1_000).padded_to(1_200))
            .with_engine(EngineOverhead {
                block_tokens: Some(16),
                fixed_device_bytes: 1_000,
            })
    }

    #[test]
    fn the_report_restates_the_hand_computed_charges() {
        let options = KvAccountingOptions {
            max_in_flight_tokens: Some(32),
            prefix_caching: true,
            ..KvAccountingOptions::default()
        };
        let accounting = KvAccounting::resolve(&hybrid_layout(), options, 1_000_000).unwrap();
        let report = accounting.report(Some(100));

        assert_eq!(report.layout, hybrid_layout());
        assert_eq!(
            report.options.prefix_resume_points,
            PrefixResumePoints::PromptEnd
        );
        // Unit gcd(100, 100, 1200) = 100 B; (1_000_000 - 1000) / 100 units.
        let charged = &report.accounting;
        assert_eq!(charged.bytes_per_unit, 100);
        assert_eq!(charged.charged_fixed_device_bytes, 1_000);
        assert_eq!(charged.capacity_units, 9_990);
        assert_eq!(charged.charged_block_tokens, 16);
        let charges: Vec<_> = charged
            .parts
            .iter()
            .map(|part| {
                (
                    part.name,
                    part.units_per_entry,
                    part.held_copies,
                    part.align_copy,
                )
            })
            .collect();
        assert_eq!(
            charges,
            [
                ("global", 1, 1, None),
                ("swa", 1, 1, None),
                // The padded state, held once; vLLM's estimate adds align
                // mode's previous-step copy.
                ("state", 12, 1, Some(true)),
            ]
        );

        // At 100 tokens: global in seven 16-token blocks (112), the window
        // capped at W - 1 + in-flight = 95, and two 12-unit states in the
        // estimate, one held.
        let result = report.result;
        assert_eq!(
            result.units_per_request_at_max_model_len,
            Some(112 + 95 + 24)
        );
        assert_eq!(result.admission_units_at_max_model_len, Some(112 + 95 + 12));
        assert_eq!(
            result.max_concurrency_at_max_model_len,
            Some(accounting.concurrency_at(100))
        );
        assert_eq!(
            result.max_concurrency_at_max_model_len,
            Some(9_990.0 / 231.0)
        );
        // Under `none`: 10_000 units, 100 + a 64-token window + one 10-unit state.
        assert_eq!(result.other_kv_overhead, KvOverhead::None);
        assert_eq!(
            result.other_max_concurrency_at_max_model_len,
            Some(10_000.0 / 174.0)
        );
        assert_eq!(result.capacity_tokens, None);

        let unknown = accounting.report(None).result;
        assert_eq!(unknown.max_concurrency_at_max_model_len, None);
        assert_eq!(unknown.other_max_concurrency_at_max_model_len, None);
    }

    #[test]
    fn a_layout_that_grows_per_token_reports_its_capacity_in_tokens() {
        let layout = KvCacheLayout::new()
            .with_part(KvCachePart::per_token("global", 100))
            .with_part(KvCachePart::compressed("indexer", 4, 200));
        let accounting =
            KvAccounting::resolve(&layout, KvAccountingOptions::default(), 1_000_000).unwrap();
        // L + 2 * ceil(L / 4) <= 10_000 units holds up to L = 6666.
        assert_eq!(accounting.report(None).result.capacity_tokens, Some(6_666));

        let reserved = accounting.reserving(1_000).report(Some(1_000));
        assert_eq!(reserved.accounting.capacity_units, 9_000);
        assert_eq!(reserved.accounting.reserved_units, 1_000);
        // Both modes charge 1500 units per 1000-token request from 9000.
        assert_eq!(reserved.result.max_concurrency_at_max_model_len, Some(6.0));
        assert_eq!(
            reserved.result.other_max_concurrency_at_max_model_len,
            Some(6.0)
        );
    }
}
