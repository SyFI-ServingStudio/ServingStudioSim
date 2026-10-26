//! WHEN a released request becomes visible to the serving engine.
//!
//! Release ([`super::release`]) decides when a request reaches the API server.
//! The opt-in hand-off model adds the server's own serialization and the
//! transport to the engine. Measured on stock vLLM Neuron with a send stamp
//! per request, it is two stages:
//!
//! - **API-server send (serial FIFO).** A request that reaches an idle server
//!   (every earlier request already sent) is sent `ingest` after its arrival.
//!   A request that arrives while earlier ones are still unsent is sent one
//!   hand-off gap after its predecessor. The gap is position-aware: the second
//!   request of a busy period draws from the measured first-gap distribution,
//!   every later one from the later-gap distribution.
//! - **Send to visible.** A sent request becomes visible to the engine one
//!   `send_to_visible` delay after its send (inter-process transport and the
//!   engine's input thread). One socket and one input thread keep visibility
//!   in send order: a request is never visible before its predecessor.
//!
//! So for a burst into an idle server, request `k` (in send order) is visible
//! at `max(visible_{k-1}, arrival + ingest + gap_1 + ... + gap_k + send_to_visible_k)`.
//!
//! Every delay is drawn from a measured empirical distribution, selected by
//! the serving context (the deployment's `max_model_len`, see
//! [`FrontendHandoffConfig::parse`]). Draws are keyed by `(seed, request id,
//! quantity)` rather than taken from one stream: each request's delays are a
//! pure function of the seed and its id, independent of how releases
//! interleave, and a different seed is a different deterministic realization.
//! The same seed also draws one run-level quantity that belongs to the engine,
//! not the frontend: the plan offset of the step after one started from an
//! idle engine ([`FrontendHandoffConfig::first_plan_offset`]), which the
//! deployment hands to an async-lookahead worker.
//!
//! The request keeps its release instant as its arrival time, so latency
//! measured from arrival includes the hand-off, as a client-side measurement
//! does.

use std::collections::{BTreeMap, VecDeque};

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;

use crate::common::{RequestId, Time};

/// The measured stock vLLM Neuron 0.24 distributions, one entry per measured
/// context. Extracted from hand-off-stamped burst captures by
/// `alignment/neuron/vllm_handoff_distribution.py`; the file records the
/// samples' provenance and which quantities are keyed by context.
const VLLM_NEURON_HANDOFF_JSON: &str =
    include_str!("../../../../alignment/neuron/data/vllm_neuron_handoff.json");

/// Opt-in hand-off configuration: which measured distribution, for which
/// context, and which seed.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontendHandoffConfig {
    distribution: HandoffDistribution,
    /// The measured context whose samples are used (see [`Self::parse`]).
    measured_context: u32,
    seed: u64,
}

impl FrontendHandoffConfig {
    pub const VLLM_NEURON: &'static str = "vllm-neuron-0.24";
    pub const CONFIG_CHOICES: &'static [&'static str] = &[Self::VLLM_NEURON];

    /// `None` disables the model. A seed without a distribution is rejected
    /// rather than silently ignored; an absent seed with a distribution is 0.
    ///
    /// `max_model_len` is the serving context, read from the deployment's
    /// arch (the run has exactly one; see `main.rs`). The measured context used
    /// is the largest one at or below it, or the smallest measured one when
    /// every measured context is larger: a context between two measured ones
    /// borrows the lower one's samples.
    pub fn parse(
        name: Option<&str>,
        seed: Option<u64>,
        max_model_len: Option<u32>,
    ) -> Result<Option<Self>> {
        let Some(name) = name else {
            ensure!(
                seed.is_none(),
                "workload.frontend_handoff_seed needs workload.frontend_handoff"
            );
            return Ok(None);
        };
        let text = match name {
            Self::VLLM_NEURON => VLLM_NEURON_HANDOFF_JSON,
            other => bail!(
                "workload.frontend_handoff must be one of {:?} (got {other:?})",
                Self::CONFIG_CHOICES
            ),
        };
        let max_model_len = max_model_len.context(
            "workload.frontend_handoff is keyed by the serving context and needs \
             exactly one max_model_len across the deployment's request-serving pools",
        )?;
        let (measured_context, distribution) =
            HandoffDistribution::for_context(text, max_model_len)
                .context("embedded vLLM Neuron hand-off distribution")?;
        Ok(Some(Self {
            distribution,
            measured_context,
            seed: seed.unwrap_or(0),
        }))
    }

    /// The measured context whose samples this run uses.
    pub fn measured_context(&self) -> u32 {
        self.measured_context
    }

    /// The run's plan offset of the step after one that started from an idle
    /// engine: one draw per seed from the measured first-step offsets (step 1
    /// planned minus request 0 visible). An engine property calibrated from the
    /// same captures; the deployment passes it to an async-lookahead worker.
    pub fn first_plan_offset(&self) -> Time {
        let samples = &self.distribution.first_plan_offset;
        samples[sample_index(
            self.seed,
            RequestId(0),
            Quantity::FirstPlanOffset as u64,
            samples.len(),
        )]
    }
}

/// Empirical samples of one context, resampled uniformly (each measured value
/// equally likely).
#[derive(Clone, Debug, PartialEq)]
pub(super) struct HandoffDistribution {
    ingest: Vec<Time>,
    first_gap: Vec<Time>,
    later_gap: Vec<Time>,
    send_to_visible: Vec<Time>,
    first_plan_offset: Vec<Time>,
}

/// One context entry of the data file, in milliseconds.
#[derive(Deserialize)]
struct ContextSamples {
    ingest_ms: Vec<f64>,
    first_gap_ms: Vec<f64>,
    later_gap_ms: Vec<f64>,
    send_to_visible_ms: Vec<f64>,
    first_plan_offset_ms: Vec<f64>,
}

impl HandoffDistribution {
    /// The entry for `max_model_len` (see [`FrontendHandoffConfig::parse`]) and
    /// the measured context it belongs to.
    fn for_context(text: &str, max_model_len: u32) -> Result<(u32, Self)> {
        #[derive(Deserialize)]
        struct File {
            schema_version: u32,
            contexts: BTreeMap<String, ContextSamples>,
        }
        let file: File = serde_json::from_str(text)?;
        ensure!(
            file.schema_version == 2,
            "hand-off distribution schema_version {} (expected 2)",
            file.schema_version
        );
        let mut contexts = file
            .contexts
            .into_iter()
            .map(|(context, samples)| {
                let context: u32 = context
                    .parse()
                    .with_context(|| format!("context key {context:?}"))?;
                Ok((context, samples))
            })
            .collect::<Result<BTreeMap<u32, ContextSamples>>>()?;
        let context = contexts
            .range(..=max_model_len)
            .next_back()
            .or_else(|| contexts.iter().next())
            .map(|(&context, _)| context)
            .context("no measured context")?;
        let samples = contexts.remove(&context).expect("selected key exists");
        Ok((context, Self::from_samples(&samples)?))
    }

    fn from_samples(samples: &ContextSamples) -> Result<Self> {
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
            ingest: to_times("ingest_ms", &samples.ingest_ms)?,
            first_gap: to_times("first_gap_ms", &samples.first_gap_ms)?,
            later_gap: to_times("later_gap_ms", &samples.later_gap_ms)?,
            send_to_visible: to_times("send_to_visible_ms", &samples.send_to_visible_ms)?,
            first_plan_offset: to_times("first_plan_offset_ms", &samples.first_plan_offset_ms)?,
        })
    }
}

/// Which delay a draw is for. Part of the draw key, so no two quantities share
/// a value.
#[derive(Clone, Copy)]
enum Quantity {
    Ingest = 1,
    LaterGap = 2,
    FirstGap = 3,
    SendToVisible = 4,
    FirstPlanOffset = 5,
}

/// Runtime state of the hand-off stages.
#[derive(Debug)]
pub(super) struct FrontendHandoff {
    config: FrontendHandoffConfig,
    /// When the server sends its latest request. A request that arrives at or
    /// after this instant finds the server idle.
    last_send: Option<Time>,
    /// Requests sent in the current busy period, which picks the gap
    /// distribution of the next one.
    sent_in_busy_period: u32,
    /// When the latest request becomes visible: the floor of the next one.
    last_visible: Time,
    /// Requests in hand-off, in visibility order: (visible at, index into the
    /// trace, arrival time to stamp).
    in_handoff: VecDeque<(Time, usize, Time)>,
}

impl FrontendHandoff {
    pub(super) fn new(config: FrontendHandoffConfig) -> Self {
        Self {
            config,
            last_send: None,
            sent_in_busy_period: 0,
            last_visible: Time::ZERO,
            in_handoff: VecDeque::new(),
        }
    }

    /// Accept a released request and schedule its send and visibility.
    pub(super) fn accept(&mut self, index: usize, request: RequestId, arrival: Time) {
        let send = match self.last_send {
            Some(last_send) if arrival < last_send => {
                let gap = if self.sent_in_busy_period == 1 {
                    Quantity::FirstGap
                } else {
                    Quantity::LaterGap
                };
                self.sent_in_busy_period += 1;
                last_send + self.draw(request, gap)
            }
            _ => {
                self.sent_in_busy_period = 1;
                arrival + self.draw(request, Quantity::Ingest)
            }
        };
        self.last_send = Some(send);
        let visible_at =
            (send + self.draw(request, Quantity::SendToVisible)).max(self.last_visible);
        self.last_visible = visible_at;
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
        let distribution = &self.config.distribution;
        let samples = match quantity {
            Quantity::Ingest => &distribution.ingest,
            Quantity::FirstGap => &distribution.first_gap,
            Quantity::LaterGap => &distribution.later_gap,
            Quantity::SendToVisible => &distribution.send_to_visible,
            Quantity::FirstPlanOffset => &distribution.first_plan_offset,
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

    fn config_at(seed: u64, max_model_len: u32) -> FrontendHandoffConfig {
        FrontendHandoffConfig::parse(
            Some(FrontendHandoffConfig::VLLM_NEURON),
            Some(seed),
            Some(max_model_len),
        )
        .unwrap()
        .unwrap()
    }

    fn config(seed: u64) -> FrontendHandoffConfig {
        config_at(seed, 512)
    }

    /// (send, visible) of each request of a simultaneous burst, in send order.
    fn burst(config: FrontendHandoffConfig, requests: u32) -> Vec<(Time, Time)> {
        let mut handoff = FrontendHandoff::new(config);
        let mut sends = Vec::new();
        for id in 0..requests {
            handoff.accept(id as usize, RequestId(id), Time::ZERO);
            sends.push(handoff.last_send.unwrap());
        }
        sends
            .into_iter()
            .zip(
                handoff
                    .in_handoff
                    .iter()
                    .map(|&(visible_at, _, _)| visible_at),
            )
            .collect()
    }

    fn data_file() -> serde_json::Value {
        serde_json::from_str(VLLM_NEURON_HANDOFF_JSON).unwrap()
    }

    #[test]
    fn embedded_distribution_matches_its_extraction() {
        let file = data_file();
        let contexts = file["contexts"].as_object().unwrap();
        assert_eq!(
            contexts.keys().map(String::as_str).collect::<Vec<_>>(),
            ["512", "2048", "8192"]
        );
        for (context, entry) in contexts {
            let captures = entry["captures"].as_array().unwrap().len();
            let distribution = config_at(0, context.parse().unwrap()).distribution;
            // One sample per capture for the context-keyed per-capture inputs.
            assert_eq!(distribution.ingest.len(), captures);
            assert_eq!(distribution.first_gap.len(), captures);
            assert_eq!(distribution.first_plan_offset.len(), captures);
            // Pooled over the 11 stamped captures.
            assert_eq!(distribution.send_to_visible.len(), 11);
            assert_eq!(distribution.later_gap.len(), 11 * 14);
        }
    }

    #[test]
    fn the_serving_context_selects_the_measured_context_at_or_below_it() {
        let measured = |len| config_at(0, len).measured_context();
        assert_eq!(measured(512), 512);
        assert_eq!(measured(1024), 512);
        assert_eq!(measured(2048), 2048);
        assert_eq!(measured(4096), 2048);
        assert_eq!(measured(8192), 8192);
        assert_eq!(measured(32768), 8192);
        assert_eq!(measured(128), 512, "below every measured context");
        assert_ne!(
            config_at(0, 512).distribution,
            config_at(0, 8192).distribution
        );
    }

    #[test]
    fn absent_model_is_off_and_a_lone_seed_or_missing_context_is_rejected() {
        assert_eq!(
            FrontendHandoffConfig::parse(None, None, None).unwrap(),
            None
        );
        assert_eq!(
            FrontendHandoffConfig::parse(None, None, Some(512)).unwrap(),
            None
        );
        assert!(FrontendHandoffConfig::parse(None, Some(3), Some(512)).is_err());
        assert!(FrontendHandoffConfig::parse(Some("other"), None, Some(512)).is_err());
        assert!(
            FrontendHandoffConfig::parse(Some(FrontendHandoffConfig::VLLM_NEURON), None, None)
                .is_err()
        );
        assert_eq!(
            FrontendHandoffConfig::parse(Some(FrontendHandoffConfig::VLLM_NEURON), None, Some(512))
                .unwrap()
                .unwrap()
                .seed,
            0
        );
    }

    #[test]
    fn a_seed_is_one_deterministic_realization() {
        assert_eq!(burst(config(7), 16), burst(config(7), 16));
        assert_ne!(burst(config(7), 16), burst(config(8), 16));
    }

    #[test]
    fn a_burst_is_sent_by_position_aware_gaps_and_visible_in_send_order() {
        for len in [512, 2048, 8192] {
            let distribution = config_at(0, len).distribution;
            for seed in 0..20 {
                let burst = burst(config_at(seed, len), 16);
                assert!(distribution.ingest.contains(&burst[0].0));
                assert!(distribution
                    .send_to_visible
                    .contains(&(burst[0].1 - burst[0].0)));
                assert!(distribution.first_gap.contains(&(burst[1].0 - burst[0].0)));
                for pair in burst[1..].windows(2) {
                    assert!(distribution.later_gap.contains(&(pair[1].0 - pair[0].0)));
                }
                for (k, pair) in burst.windows(2).enumerate() {
                    let (send, visible) = pair[1];
                    let transport = distribution.send_to_visible.contains(&(visible - send));
                    // Visible one transport delay after its send, unless held
                    // behind its predecessor on the same channel.
                    assert!(transport || visible == pair[0].1, "request {}", k + 1);
                    assert!(visible >= pair[0].1);
                }
            }
        }
    }

    #[test]
    fn an_arrival_after_the_last_send_restarts_the_busy_period() {
        let mut handoff = FrontendHandoff::new(config(1));
        handoff.accept(0, RequestId(0), Time::ZERO);
        let first_send = handoff.last_send.unwrap();
        let first_visible = handoff.last_visible;
        let late = first_send + Time::from_ms(1.0);
        handoff.accept(1, RequestId(1), late);
        let ingest = handoff.last_send.unwrap() - late;
        assert!(config(1).distribution.ingest.contains(&ingest));
        assert_eq!(handoff.sent_in_busy_period, 1);
        // A third simultaneous request is the busy period's second: first gap.
        handoff.accept(2, RequestId(2), late);
        let gap = handoff.last_send.unwrap() - (late + ingest);
        assert_eq!(gap, handoff.draw(RequestId(2), Quantity::FirstGap));
        assert_eq!(handoff.pop_visible(first_visible), Some((0, Time::ZERO)));
        assert_eq!(handoff.pop_visible(first_visible), None);
        assert_eq!(handoff.in_handoff(), 2);
    }

    #[test]
    fn draws_are_keyed_by_request_not_by_release_order() {
        let seed = 11;
        let mut handoff = FrontendHandoff::new(config(seed));
        handoff.accept(0, RequestId(5), Time::ZERO);
        let first = handoff.last_send.unwrap();
        handoff.accept(1, RequestId(2), Time::ZERO);
        assert_eq!(
            handoff.last_send.unwrap() - first,
            handoff.draw(RequestId(2), Quantity::FirstGap)
        );
    }

    #[test]
    fn the_first_plan_offset_is_one_measured_draw_per_seed() {
        for len in [512, 8192] {
            let samples = config_at(0, len).distribution.first_plan_offset;
            let draws: Vec<_> = (0..64)
                .map(|seed| config_at(seed, len).first_plan_offset())
                .collect();
            assert!(draws.iter().all(|draw| samples.contains(draw)));
            assert_eq!(draws[3], config_at(3, len).first_plan_offset());
            let distinct: std::collections::BTreeSet<_> = draws.iter().collect();
            assert_eq!(distinct.len(), samples.len(), "64 seeds reach every sample");
        }
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
