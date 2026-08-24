//! WHEN a loaded arrival is allowed to enter the system.
//!
//! Release has three orthogonal axes over the same immutable arrival list:
//! [`ArrivalMode`] decides when a new top-level unit becomes *eligible*,
//! [`CapacityLimit`] decides how many units may be *active* at once, and
//! [`SessionDependency`] decides whether a row is independent or waits for the
//! preceding round of its session. Keeping them separate permits every
//! combination without growing a cross-product enum.
//!
//! Arrival and capacity used to be one axis, which made two of the four
//! combinations unrepresentable: a timeline replay could not be capped, and a
//! capped run had to discard the timeline. `TraceLab` has always composed them —
//! it waits for a session's arrival and *then* acquires a permit — so the weld
//! was also the reason a measured run and a simulated run could not be said to
//! release work the same way.

use anyhow::{bail, Result};

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use super::ReleaseMetadata;
use crate::common::{RequestId, Time};

pub use req_frontend::release::ArrivalMode;

/// `VibeSim`'s resolved pacing input.
///
/// [`ArrivalMode`] is the shared cross-consumer choice. The rate stays beside
/// it here because `VibeSim` stores rate-1-normalized arrivals, while the measured
/// client rescales a trace from its observed absolute rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArrivalSchedule {
    mode: ArrivalMode,
    request_rate: f64,
}

impl ArrivalSchedule {
    pub const CONFIG_CHOICES: &'static [&'static str] = ArrivalMode::CONFIG_CHOICES;

    pub fn parse(name: &str, request_rate: f64) -> Result<Self> {
        let mode = ArrivalMode::parse_config(name)?;
        if mode == ArrivalMode::TraceTimed && !(request_rate.is_finite() && request_rate > 0.0) {
            bail!("request_rate must be finite and > 0 (got {request_rate})");
        }
        Ok(Self { mode, request_rate })
    }

    pub fn trace_timed(request_rate: f64) -> Result<Self> {
        Self::parse("trace_timed", request_rate)
    }

    #[must_use]
    pub fn saturated() -> Self {
        Self {
            mode: ArrivalMode::Saturated,
            // Unused in saturated mode; finite so debugging never shows a
            // sentinel value that resembles a real arithmetic failure.
            request_rate: 1.0,
        }
    }
}

/// How many top-level units may be active at once.
///
/// A *unit* is a session when the trace declares sessions and rounds are
/// chained, and a request otherwise. That distinction is the whole point: a
/// session owns its slot from the moment its first round is released until its
/// last round completes, including across every tool wait in between. A cap of
/// two therefore means two conversations, not two requests — matching the
/// permit `TraceLab` holds for the lifetime of a session task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct CapacityLimit {
    max_active_units: Option<usize>,
}

impl CapacityLimit {
    #[must_use]
    pub fn unlimited() -> Self {
        Self {
            max_active_units: None,
        }
    }

    pub fn parse(max_active_units: Option<usize>) -> Result<Self> {
        if max_active_units == Some(0) {
            bail!("workload.max_concurrency must be greater than 0");
        }
        Ok(Self { max_active_units })
    }

    fn admits(self, active_units: u64) -> bool {
        match self.max_active_units {
            None => true,
            Some(limit) => active_units < limit as u64,
        }
    }
}

/// Whether requests are causally independent or chained within each session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionDependency {
    Independent,
    Chained,
}

impl SessionDependency {
    pub const CHOICES: &'static [&'static str] = &["independent", "chained"];

    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "independent" => Ok(Self::Independent),
            "chained" => Ok(Self::Chained),
            other => bail!(
                "unknown session_dependency {other:?} (expected one of {:?})",
                Self::CHOICES
            ),
        }
    }
}

#[derive(Debug)]
pub(super) struct ReplayScheduler {
    arrival: ArrivalSchedule,
    capacity: CapacityLimit,
    dependency: SessionDependencyState,
    active: ActiveUnits,
    /// Whether the previous call turned away a unit that had already arrived,
    /// purely for lack of a slot.
    ///
    /// It decides which instant a head is stamped with. Normally that is the
    /// unit's own trace arrival, deliberately, so the tick granularity of the
    /// drain loop never leaks into arrival times. But a unit the cap held back
    /// did not arrive when the trace says: the measured runner acquires its
    /// permit *after* waiting for the arrival and only then sends, so its clock
    /// starts at the permit. Stamping the trace time instead would charge the
    /// simulated request for a wait the measured one never reports.
    capacity_deferred: bool,
}

/// Which top-level units currently hold a capacity slot.
///
/// Under [`SessionDependency::Independent`] a unit is a request, and the count
/// the frontend already maintains — emitted minus completed — is exactly right.
///
/// Under [`SessionDependency::Chained`] it is not. A session between rounds has
/// no request in flight but is still very much active: it is sitting in a tool
/// wait, and its next round is already scheduled. Counting requests would hand
/// its slot to a different conversation and then let both run, so a cap of two
/// would admit three sessions. This ledger counts the conversation instead.
#[derive(Debug)]
enum ActiveUnits {
    /// Delegated to the caller's in-flight count.
    Requests,
    Sessions {
        /// Sessions whose head has been released and whose final round has not
        /// completed. Held across tool waits.
        open_sessions: HashSet<u32>,
        /// Rows declaring no session at all; each is its own unit while in
        /// flight, exactly as an independent request would be.
        standalone_in_flight: u64,
    },
}

impl ActiveUnits {
    fn count(&self, in_flight: u64) -> u64 {
        match self {
            Self::Requests => in_flight,
            Self::Sessions {
                open_sessions,
                standalone_in_flight,
            } => open_sessions.len() as u64 + standalone_in_flight,
        }
    }

    /// Record that `release` just entered the system.
    ///
    /// A later round is deliberately not counted again: its session was already
    /// admitted when its head was released and has held the slot ever since.
    fn on_release(&mut self, release: &ReleaseMetadata, has_predecessor: bool) {
        let Self::Sessions {
            open_sessions,
            standalone_in_flight,
        } = self
        else {
            return;
        };
        match release.session {
            Some(session) => {
                if !has_predecessor {
                    open_sessions.insert(session.session_id);
                }
            }
            None => *standalone_in_flight += 1,
        }
    }

    /// Record that `release` completed. A session frees its slot only when its
    /// *last* round finishes.
    fn on_completion(&mut self, release: &ReleaseMetadata, has_successor: bool) {
        let Self::Sessions {
            open_sessions,
            standalone_in_flight,
        } = self
        else {
            return;
        };
        match release.session {
            Some(session) => {
                if !has_successor {
                    open_sessions.remove(&session.session_id);
                }
            }
            None => *standalone_in_flight = standalone_in_flight.saturating_sub(1),
        }
    }
}

#[derive(Debug)]
enum SessionDependencyState {
    Independent {
        cursor: usize,
    },
    /// A conversation's rounds enter one at a time: a session **head** replays
    /// its pacing discipline, and every later round waits for its predecessor
    /// to complete plus that predecessor's `tool_wait`.
    ///
    /// A successor deliberately ignores its own trace `arrival_time`. That
    /// timestamp records how fast the machine the trace was captured on served
    /// the previous round — which is the very thing this simulation exists to
    /// predict. Replaying it would smuggle the recording system's throughput in
    /// as a floor.
    ///
    Chained {
        /// `successor[i]` = the next round of the same session, if any.
        successor: Vec<Option<u32>>,
        /// `true` where a row is a later round: it enters through `ready`, not
        /// through the head cursor, so the cursor scan skips it.
        has_predecessor: Vec<bool>,
        /// Next head candidate, in trace order.
        cursor: usize,
        /// Successors unlocked by a completion, keyed by the instant their
        /// predecessor's tool wait elapses. A heap rather than a queue because
        /// waits differ per round: a long wait at the front must not hold back a
        /// short one unlocked later.
        ready: BinaryHeap<Reverse<(Time, u32)>>,
    },
}

impl ReplayScheduler {
    pub(super) fn new(
        arrival: ArrivalSchedule,
        capacity: CapacityLimit,
        dependency: SessionDependency,
        releases: &[ReleaseMetadata],
    ) -> Self {
        let (dependency, active) = match dependency {
            SessionDependency::Independent => (
                SessionDependencyState::Independent { cursor: 0 },
                ActiveUnits::Requests,
            ),
            SessionDependency::Chained => {
                let (successor, has_predecessor) = build_chains(releases);
                (
                    SessionDependencyState::Chained {
                        successor,
                        has_predecessor,
                        cursor: 0,
                        ready: BinaryHeap::new(),
                    },
                    ActiveUnits::Sessions {
                        open_sessions: HashSet::new(),
                        standalone_in_flight: 0,
                    },
                )
            }
        };
        Self {
            arrival,
            capacity,
            dependency,
            active,
            capacity_deferred: false,
        }
    }

    /// The next arrival that may enter right now: its index into `arrivals` plus
    /// the arrival time to stamp it with. `None` = nothing releasable this tick.
    ///
    /// Advances the dependency state's cursor on success, so the caller's drain
    /// loop is just "keep asking until it says no".
    ///
    /// The scheduler accepts only [`ReleaseMetadata`], so it cannot inspect a
    /// request definition even accidentally.
    pub(super) fn next_ready(
        &mut self,
        releases: &[ReleaseMetadata],
        in_flight: u64,
        now: Time,
    ) -> Option<(usize, Time)> {
        let admits_new_unit = self.capacity.admits(self.active.count(in_flight));

        let (index, mut release_time) = match &mut self.dependency {
            SessionDependencyState::Independent { cursor } => {
                let release = releases.get(*cursor)?;
                let release_time = release_time(self.arrival, release, now)?;
                if !admits_new_unit {
                    self.capacity_deferred = true;
                    return None;
                }
                (take(cursor), release_time)
            }
            SessionDependencyState::Chained {
                has_predecessor,
                cursor,
                ready,
                ..
            } => {
                // Live conversations first: a successor whose tool wait has
                // elapsed is already mid-session, so it outranks starting a new
                // one. Stamped with `now`, not its trace time — the release
                // instant is its real arrival.
                //
                // Capacity is not consulted here, and that is the contract, not
                // an oversight: its session was admitted when its head was
                // released and has held the slot through the tool wait. Gating
                // it again would deadlock a full run, since the only thing that
                // frees a slot is the completion of a session this very branch
                // has to release.
                if let Some(&Reverse((due, index))) = ready.peek() {
                    if due <= now {
                        ready.pop();
                        return self.admit(releases, index as usize, now);
                    }
                }
                // Otherwise the next session head due by its own arrival. Rows
                // already spoken for by a chain are stepped over; they enter
                // through `ready` and must not be released twice.
                let mut head = None;
                while let Some(release) = releases.get(*cursor) {
                    if has_predecessor[*cursor] {
                        *cursor += 1;
                        continue;
                    }
                    // Arrival is checked before capacity so that a unit that has
                    // not arrived yet is never recorded as capacity-deferred.
                    let release_time = release_time(self.arrival, release, now)?;
                    if !admits_new_unit {
                        self.capacity_deferred = true;
                        return None;
                    }
                    head = Some((take(cursor), release_time));
                    break;
                }
                head?
            }
        };
        if std::mem::take(&mut self.capacity_deferred) {
            release_time = release_time.max(now);
        }
        self.admit(releases, index, release_time)
    }

    /// Book a release into the ledger and hand it back to the drain loop.
    fn admit(
        &mut self,
        releases: &[ReleaseMetadata],
        index: usize,
        release_time: Time,
    ) -> Option<(usize, Time)> {
        let has_predecessor = match &self.dependency {
            SessionDependencyState::Chained {
                has_predecessor, ..
            } => has_predecessor[index],
            SessionDependencyState::Independent { .. } => false,
        };
        self.active.on_release(&releases[index], has_predecessor);
        Some((index, release_time))
    }

    /// A request finished at `now`. Only chaining reacts: it unlocks that round's
    /// successor, due once the completing round's own `tool_wait` elapses.
    pub(super) fn on_completion(
        &mut self,
        releases: &[ReleaseMetadata],
        request: RequestId,
        now: Time,
    ) {
        let SessionDependencyState::Chained {
            successor, ready, ..
        } = &mut self.dependency
        else {
            return;
        };
        let finished = request.0 as usize;
        let next = successor.get(finished).copied().flatten();
        self.active
            .on_completion(&releases[finished], next.is_some());
        let Some(next) = next else {
            return;
        };
        let tool_wait_after = releases[finished]
            .session
            .map_or(Time::ZERO, |session| session.tool_wait_after);
        ready.push(Reverse((now + tool_wait_after, next)));
    }
}

fn release_time(arrival: ArrivalSchedule, release: &ReleaseMetadata, now: Time) -> Option<Time> {
    match arrival.mode {
        ArrivalMode::TraceTimed => {
            let due = effective_arrival(release.trace_arrival_time_ms, arrival.request_rate);
            (due <= now).then_some(due)
        }
        ArrivalMode::Saturated => Some(now),
    }
}

/// Link each session's rows into a chain, in trace order.
///
/// One pass, keyed by `session_id`: whatever row of a session was seen last
/// becomes the predecessor of the next one. Rows declaring no session are left
/// unlinked, which makes them heads. They therefore follow whichever pacing
/// discipline was selected.
///
/// Note there is no predecessor *pointer* on the arrival: a chain is derived
/// here from a stable `session_id`, not carried per-row. A pointer would have to
/// be re-keyed every time a round completes; a session id never moves.
fn build_chains(releases: &[ReleaseMetadata]) -> (Vec<Option<u32>>, Vec<bool>) {
    let mut successor = vec![None; releases.len()];
    let mut has_predecessor = vec![false; releases.len()];
    let mut latest_round: HashMap<u32, usize> = HashMap::new();
    for (index, release) in releases.iter().enumerate() {
        let Some(session_id) = release.session.map(|session| session.session_id) else {
            continue;
        };
        if let Some(previous) = latest_round.insert(session_id, index) {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "index is a position within this run's release trace, far below u32::MAX"
            )]
            let index_u32 = index as u32;
            successor[previous] = Some(index_u32);
            has_predecessor[index] = true;
        }
    }
    (successor, has_predecessor)
}

/// Read a cursor and step it past the entry it just yielded.
#[inline]
fn take(cursor: &mut usize) -> usize {
    let index = *cursor;
    *cursor += 1;
    index
}

/// Effective arrival time: the rate-1-normalized `arrival_time` remapped to the
/// target `request_rate` (req/s). Higher rate ⇒ arrivals sooner.
fn effective_arrival(arrival_time: f64, request_rate: f64) -> Time {
    Time::from_ms(arrival_time / request_rate)
}
