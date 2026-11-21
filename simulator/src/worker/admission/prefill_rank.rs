//! Ranked prefill order (not vLLM): one key per request, smallest first,
//! across started and queued prompts.
//!
//! [`SquareCost`] targets the per-request objective
//! `TTFT + lambda * max(0, TTFT - tau)^2` with `tau = max(D0, k * P / mu)`
//! (`P` the request's prefill tokens at arrival, `mu` the server's prefill
//! throughput). Without lateness the order is shortest remaining first (srpt
//! minimizes summed TTFT on one server). A request whose projected finish
//! `waited + remaining / mu` is past `tau` by `L` has marginal cost
//! `1 + 2 * lambda * L` per second of delay instead of `1`, so it ranks as if it
//! were that many times shorter: key `remaining / (1 + 2 * lambda * L)`
//! (generalized c-mu rule: serve the highest cost-rate per unit of work).
//!
//! The pad rule follows from TTFT being realized only when a request's last
//! chunk runs: a chunk that does not finish its request lengthens the step for
//! every request that does finish in it. With `n` finishing requests already in
//! the step, a non-finishing chunk of a request with weight `w` is worth adding
//! only when `w > n` (it advances that request by as much as it delays each of
//! the `n`).
//!
//! [`SquareCost`] keeps the queue in keys that do not change with time, so a
//! formation costs O(log n) in the queue length, never a pass over the queue.
//! A queued request has not started, so its remaining tokens are its size and
//! its lateness starts at a fixed instant, `arrival + tau - size / mu`
//! ([`SquareCost::late_at`]). Until then its key is its size (shortest first);
//! from then on it is in the late tier, earliest deadline (`arrival + tau`)
//! first. The head is whichever tier head has the smaller index key, the late
//! head's computed at the formation. Once late, a request's cost is in its
//! convex part; when many are late at once (a backlog), spreading the
//! lateness by deadline costs less than the index, which keeps passing the
//! small late requests ahead of the larger late ones: each such swap is right
//! for the pair, but under a lasting backlog the larger request's work is only
//! deferred, not avoided (TraceLab v3.1, fold, warm-up backlog: 16-100k
//! requests waited 49 s and >= 100k 213 s under the index, 15 s for every class
//! under fcfs). Started prompts (at most the worker's sequence limit) are keyed
//! by the index at each formation.

use crate::common::Time;

/// Squared-lateness cost parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SquareCost {
    /// Penalty per squared second past `tau`.
    pub lambda_per_s: f64,
    /// Floor of `tau` (s).
    pub d0_s: f64,
    /// `tau` is at least `k` times the request's isolated prefill time.
    pub k: f64,
    /// Prefill throughput of one server (tokens/s) that turns tokens into time.
    pub tokens_per_s: f64,
    /// Apply the pad rule.
    pub pad: bool,
}

impl SquareCost {
    /// Lateness bound of a request with `size` prefill tokens (s).
    pub fn tau_s(&self, size: u32) -> f64 {
        self.d0_s.max(self.k * f64::from(size) / self.tokens_per_s)
    }

    /// When a queued request (nothing computed yet) starts to project past
    /// its bound: `arrival + tau - size / mu`.
    pub fn late_at(&self, size: u32, arrival: Time) -> Time {
        Time::from_s(
            (arrival.as_s() + self.tau_s(size) - f64::from(size) / self.tokens_per_s).max(0.0),
        )
    }

    /// The request's deadline, `arrival + tau`.
    pub fn due(&self, size: u32, arrival: Time) -> Time {
        Time::from_s(arrival.as_s() + self.tau_s(size))
    }

    /// Marginal cost of delay, `1 + 2 * lambda * projected lateness`.
    pub fn weight(&self, size: u32, arrival: Time, remaining: u32, now: Time) -> f64 {
        let waited = now.as_s() - arrival.as_s();
        let late = waited + f64::from(remaining) / self.tokens_per_s - self.tau_s(size);
        1.0 + 2.0 * self.lambda_per_s * late.max(0.0)
    }
}

/// How a ranked lifecycle keys a request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrefillRank {
    Square(SquareCost),
}

/// A request's rank: `key` (smaller first) and its delay weight for the pad rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rank {
    pub key: u64,
    pub weight: f64,
}

impl PrefillRank {
    /// Rank of a request of `size` prefill tokens that arrived at `arrival`
    /// and has `remaining` tokens left at `now`.
    pub fn rank(&self, size: u32, arrival: Time, remaining: u32, now: Time) -> Rank {
        match self {
            Self::Square(cost) => {
                let weight = cost.weight(size, arrival, remaining, now);
                Rank {
                    key: (f64::from(remaining) / weight).ceil() as u64,
                    weight,
                }
            }
        }
    }

    /// Whether non-finishing chunks are gated by the pad rule.
    pub fn pads(&self) -> bool {
        match self {
            Self::Square(cost) => cost.pad,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COST: SquareCost = SquareCost {
        lambda_per_s: 1.0,
        d0_s: 10.0,
        k: 3.0,
        tokens_per_s: 100_000.0,
        pad: true,
    };

    #[test]
    fn an_on_time_request_ranks_by_remaining_tokens() {
        let rank = PrefillRank::Square(COST);
        let r = rank.rank(50_000, Time::ZERO, 20_000, Time::from_s(2.0));
        assert_eq!(
            r,
            Rank {
                key: 20_000,
                weight: 1.0
            }
        );
    }

    #[test]
    fn a_queued_request_turns_late_when_its_projected_finish_passes_tau() {
        // 600k at 100k tok/s: tau = max(10, 18) = 18 s, 6 s of work: late from
        // 12 s, due at 18 s; 1k: tau 10 s, late from 9.99 s.
        assert_eq!(COST.late_at(600_000, Time::ZERO), Time::from_s(12.0));
        assert_eq!(COST.due(600_000, Time::ZERO), Time::from_s(18.0));
        assert_eq!(COST.late_at(1_000, Time::from_s(5.0)), Time::from_s(14.99));
    }

    #[test]
    fn a_late_request_ranks_as_if_shorter() {
        // tau = max(10, 3 * 6 s) = 18 s; projected finish 17 + 6 = 23 s: 5 s late.
        let rank = PrefillRank::Square(COST);
        let r = rank.rank(600_000, Time::ZERO, 600_000, Time::from_s(17.0));
        assert!((r.weight - 11.0).abs() < 1e-9);
        assert_eq!(r.key, 54_546);
    }
}
