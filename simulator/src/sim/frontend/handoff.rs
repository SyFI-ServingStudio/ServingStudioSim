//! WHEN a released request becomes visible to the serving engine.
//!
//! Release ([`super::release`]) decides when a request reaches the API server.
//! The opt-in hand-off model adds the server's own serialization before the
//! engine sees it: the server turns requests into engine arrivals one after
//! another. It is a single FIFO stage:
//!
//! - a request that reaches an idle stage (every earlier request already
//!   handed off) becomes visible `ingest_delay` after its arrival;
//! - a request that arrives while earlier ones are still being handed off
//!   becomes visible one `handoff_gap` after its predecessor.
//!
//! Both delays are drawn from a measured empirical distribution, so a burst of
//! simultaneous arrivals reaches the engine spread out the way the measured
//! server spread it. Draws are keyed by `(seed, request id, quantity)` rather
//! than taken from one stream: each request's delays are a pure function of the
//! seed and its id, independent of how releases interleave, and a different
//! seed is a different deterministic realization.
//!
//! The request keeps its release instant as its arrival time, so latency
//! measured from arrival includes the hand-off, as a client-side measurement
//! does.

use std::collections::VecDeque;

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;

use crate::common::{RequestId, Time};

/// The measured stock vLLM Neuron 0.24 distribution. Extracted from 13 burst
/// captures by `alignment/neuron/vllm_handoff_distribution.py`; the file
/// records its provenance.
const VLLM_NEURON_HANDOFF_JSON: &str =
    include_str!("../../../../alignment/neuron/data/vllm_neuron_handoff.json");

/// Opt-in hand-off configuration: which measured distribution and which seed.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontendHandoffConfig {
    distribution: HandoffDistribution,
    seed: u64,
}

impl FrontendHandoffConfig {
    pub const VLLM_NEURON: &'static str = "vllm-neuron-0.24";
    pub const CONFIG_CHOICES: &'static [&'static str] = &[Self::VLLM_NEURON];

    /// `None` disables the model. A seed without a distribution is rejected
    /// rather than silently ignored; an absent seed with a distribution is 0.
    pub fn parse(name: Option<&str>, seed: Option<u64>) -> Result<Option<Self>> {
        let Some(name) = name else {
            ensure!(
                seed.is_none(),
                "workload.frontend_handoff_seed needs workload.frontend_handoff"
            );
            return Ok(None);
        };
        let distribution = match name {
            Self::VLLM_NEURON => HandoffDistribution::from_json(VLLM_NEURON_HANDOFF_JSON)
                .context("embedded vLLM Neuron hand-off distribution")?,
            other => bail!(
                "workload.frontend_handoff must be one of {:?} (got {other:?})",
                Self::CONFIG_CHOICES
            ),
        };
        Ok(Some(Self {
            distribution,
            seed: seed.unwrap_or(0),
        }))
    }
}

/// Empirical samples, resampled uniformly (each measured value equally likely).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct HandoffDistribution {
    ingest_delay: Vec<Time>,
    handoff_gap: Vec<Time>,
}

impl HandoffDistribution {
    fn from_json(text: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct Samples {
            ingest_delay_ms: Vec<f64>,
            handoff_gap_ms: Vec<f64>,
        }
        let samples: Samples = serde_json::from_str(text)?;
        Self::from_ms(&samples.ingest_delay_ms, &samples.handoff_gap_ms)
    }

    pub(super) fn from_ms(ingest_delay_ms: &[f64], handoff_gap_ms: &[f64]) -> Result<Self> {
        let to_times = |name: &str, values: &[f64]| -> Result<Vec<Time>> {
            ensure!(!values.is_empty(), "{name} has no samples");
            values
                .iter()
                .map(|&ms| {
                    ensure!(
                        ms.is_finite() && ms >= 0.0,
                        "{name} sample {ms} is not >= 0"
                    );
                    Ok(Time::from_ms(ms))
                })
                .collect()
        };
        Ok(Self {
            ingest_delay: to_times("ingest_delay_ms", ingest_delay_ms)?,
            handoff_gap: to_times("handoff_gap_ms", handoff_gap_ms)?,
        })
    }
}

/// Which delay a draw is for. Part of the draw key, so the two never share a
/// value.
#[derive(Clone, Copy)]
enum Quantity {
    IngestDelay = 1,
    HandoffGap = 2,
}

/// Runtime state of the serial hand-off stage.
#[derive(Debug)]
pub(super) struct FrontendHandoff {
    config: FrontendHandoffConfig,
    /// When the stage finishes handing off its latest request. A request that
    /// arrives at or after this instant finds the stage idle.
    busy_until: Option<Time>,
    /// Requests in hand-off, in visibility order: (visible at, index into the
    /// trace, arrival time to stamp).
    in_handoff: VecDeque<(Time, usize, Time)>,
}

impl FrontendHandoff {
    pub(super) fn new(config: FrontendHandoffConfig) -> Self {
        Self {
            config,
            busy_until: None,
            in_handoff: VecDeque::new(),
        }
    }

    /// Accept a released request and schedule its visibility.
    pub(super) fn accept(&mut self, index: usize, request: RequestId, arrival: Time) {
        let visible_at = match self.busy_until {
            Some(busy_until) if arrival < busy_until => {
                busy_until + self.draw(request, Quantity::HandoffGap)
            }
            _ => arrival + self.draw(request, Quantity::IngestDelay),
        };
        self.busy_until = Some(visible_at);
        self.in_handoff.push_back((visible_at, index, arrival));
    }

    /// The next request visible by `now`, as (trace index, arrival time).
    pub(super) fn pop_visible(&mut self, now: Time) -> Option<(usize, Time)> {
        let &(visible_at, index, arrival) = self.in_handoff.front()?;
        (visible_at <= now).then(|| {
            self.in_handoff.pop_front();
            (index, arrival)
        })
    }

    /// Requests released by the replay scheduler but not yet visible.
    pub(super) fn in_handoff(&self) -> u64 {
        self.in_handoff.len() as u64
    }

    fn draw(&self, request: RequestId, quantity: Quantity) -> Time {
        let samples = match quantity {
            Quantity::IngestDelay => &self.config.distribution.ingest_delay,
            Quantity::HandoffGap => &self.config.distribution.handoff_gap,
        };
        samples[sample_index(self.config.seed, request, quantity as u64, samples.len())]
    }
}

/// Uniform index in `0..len` keyed by `(seed, request, quantity)`.
fn sample_index(seed: u64, request: RequestId, quantity: u64, len: usize) -> usize {
    let key = seed
        ^ u64::from(request.0).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ quantity.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    // Multiply-shift maps the 64-bit draw onto `0..len` without a modulo.
    ((u128::from(splitmix64(key)) * len as u128) >> 64) as usize
}

/// splitmix64 finalizer: no carried state, bit-identical across machines.
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(seed: u64) -> FrontendHandoffConfig {
        FrontendHandoffConfig::parse(Some(FrontendHandoffConfig::VLLM_NEURON), Some(seed))
            .unwrap()
            .unwrap()
    }

    fn burst_visibility(seed: u64, requests: u32) -> Vec<Time> {
        let mut handoff = FrontendHandoff::new(config(seed));
        for id in 0..requests {
            handoff.accept(id as usize, RequestId(id), Time::ZERO);
        }
        handoff
            .in_handoff
            .iter()
            .map(|&(visible_at, _, _)| visible_at)
            .collect()
    }

    #[test]
    fn embedded_distribution_matches_its_extraction() {
        let distribution = config(0).distribution;
        assert_eq!(distribution.ingest_delay.len(), 13);
        assert_eq!(distribution.handoff_gap.len(), 13 * 15);
    }

    #[test]
    fn absent_model_is_off_and_a_lone_seed_is_rejected() {
        assert_eq!(FrontendHandoffConfig::parse(None, None).unwrap(), None);
        assert!(FrontendHandoffConfig::parse(None, Some(3)).is_err());
        assert!(FrontendHandoffConfig::parse(Some("other"), None).is_err());
        assert_eq!(
            FrontendHandoffConfig::parse(Some(FrontendHandoffConfig::VLLM_NEURON), None)
                .unwrap()
                .unwrap()
                .seed,
            0
        );
    }

    #[test]
    fn a_seed_is_one_deterministic_realization() {
        assert_eq!(burst_visibility(7, 16), burst_visibility(7, 16));
        assert_ne!(burst_visibility(7, 16), burst_visibility(8, 16));
    }

    #[test]
    fn simultaneous_arrivals_are_serialized_by_measured_gaps() {
        let distribution = config(0).distribution;
        for seed in 0..20 {
            let visible = burst_visibility(seed, 16);
            assert!(distribution.ingest_delay.contains(&visible[0]));
            for pair in visible.windows(2) {
                assert!(distribution.handoff_gap.contains(&(pair[1] - pair[0])));
            }
        }
    }

    #[test]
    fn an_arrival_after_the_stage_drains_pays_the_ingest_delay_again() {
        let mut handoff = FrontendHandoff::new(config(1));
        handoff.accept(0, RequestId(0), Time::ZERO);
        let first = handoff.busy_until.unwrap();
        let late = first + Time::from_ms(1.0);
        handoff.accept(1, RequestId(1), late);
        let second = handoff.busy_until.unwrap() - late;
        assert!(config(1).distribution.ingest_delay.contains(&second));
        assert_eq!(handoff.pop_visible(first), Some((0, Time::ZERO)));
        assert_eq!(handoff.pop_visible(first), None);
        assert_eq!(handoff.in_handoff(), 1);
    }

    #[test]
    fn draws_are_keyed_by_request_not_by_release_order() {
        let seed = 11;
        let gap = |request| {
            FrontendHandoff::new(config(seed)).draw(RequestId(request), Quantity::HandoffGap)
        };
        let mut handoff = FrontendHandoff::new(config(seed));
        handoff.accept(0, RequestId(5), Time::ZERO);
        handoff.accept(1, RequestId(2), Time::ZERO);
        let visible: Vec<_> = handoff.in_handoff.iter().map(|entry| entry.0).collect();
        assert_eq!(visible[1] - visible[0], gap(2));
    }

    #[test]
    fn resampling_covers_the_empirical_support_uniformly() {
        let len = 195;
        let draws = 195 * 400;
        let mut counts = vec![0_u32; len];
        for request in 0..draws {
            counts[sample_index(3, RequestId(request), 2, len)] += 1;
        }
        // Expected 400 per value; a uniform resampler stays well inside 2x.
        assert!(counts.iter().all(|&count| (200..800).contains(&count)));
    }
}
