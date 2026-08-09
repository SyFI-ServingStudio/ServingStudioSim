//! WHEN a loaded arrival is allowed to enter the system.
//!
//! Release has two orthogonal axes over the same immutable arrival list:
//! [`ReplayPacing`] decides when workload pressure permits another release, and
//! [`SessionDependency`] decides whether a row is independent or waits for the
//! preceding round of its session. Keeping them separate permits all four
//! combinations without growing a cross-product enum.

use anyhow::{bail, Result};

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use super::ReleaseMetadata;
use crate::common::{RequestId, Time};

/// Workload-pressure discipline, independent of session causality.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReplayPacing {
    /// Replay the trace's own arrival timeline.
    OpenLoop { request_rate: f64 },
    /// Ignore the timeline; keep at most `max_concurrency` requests in flight.
    ClosedLoop { max_concurrency: usize },
}

impl ReplayPacing {
    pub const CHOICES: &'static [&'static str] = &["open_loop", "closed_loop"];

    /// Build from the declared name plus the payload fields the flat config
    /// carries alongside it.
    ///
    /// A flat config cannot nest a payload under its variant, so `request_rate`
    /// and optional `max_concurrency` sit beside the name. This boundary erases
    /// the unrelated payload from the resulting enum and rejects a supplied
    /// `max_concurrency` under `open_loop` instead of leaving it silently dead.
    pub fn parse(name: &str, request_rate: f64, max_concurrency: Option<usize>) -> Result<Self> {
        let unused_max_concurrency = |pacing: &str| -> anyhow::Error {
            anyhow::anyhow!(
                "workload.max_concurrency is set but replay_pacing is {pacing:?}, which \
                 replays the trace's arrival timeline and would ignore it — either \
                 drop max_concurrency or declare replay_pacing: closed_loop"
            )
        };
        Ok(match name {
            "open_loop" => match max_concurrency {
                Some(_) => return Err(unused_max_concurrency(name)),
                None => Self::OpenLoop { request_rate },
            },
            "closed_loop" => match max_concurrency {
                Some(max_concurrency) => Self::ClosedLoop { max_concurrency },
                None => bail!(
                    "replay_pacing: closed_loop needs workload.max_concurrency — the \
                     cap is the whole discipline"
                ),
            },
            other => bail!(
                "unknown replay_pacing {other:?} (expected one of {:?})",
                Self::CHOICES
            ),
        })
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
    pacing: ReplayPacing,
    dependency: SessionDependencyState,
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
        pacing: ReplayPacing,
        dependency: SessionDependency,
        releases: &[ReleaseMetadata],
    ) -> Self {
        let dependency = match dependency {
            SessionDependency::Independent => SessionDependencyState::Independent { cursor: 0 },
            SessionDependency::Chained => {
                let (successor, has_predecessor) = build_chains(releases);
                SessionDependencyState::Chained {
                    successor,
                    has_predecessor,
                    cursor: 0,
                    ready: BinaryHeap::new(),
                }
            }
        };
        Self { pacing, dependency }
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
        if !self.pacing.has_capacity(in_flight) {
            return None;
        }

        match &mut self.dependency {
            SessionDependencyState::Independent { cursor } => {
                let release = releases.get(*cursor)?;
                let release_time = self.pacing.release_time(release, now)?;
                Some((take(cursor), release_time))
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
                if let Some(&Reverse((due, index))) = ready.peek() {
                    if due <= now {
                        ready.pop();
                        return Some((index as usize, now));
                    }
                }
                // Otherwise the next session head due by its own trace arrival.
                // Rows already spoken for by a chain are stepped over; they enter
                // through `ready` and must not be released twice.
                while let Some(release) = releases.get(*cursor) {
                    if has_predecessor[*cursor] {
                        *cursor += 1;
                        continue;
                    }
                    let release_time = self.pacing.release_time(release, now)?;
                    return Some((take(cursor), release_time));
                }
                None
            }
        }
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
        let Some(next) = successor.get(finished).copied().flatten() else {
            return;
        };
        let tool_wait_after = releases[finished]
            .session
            .map_or(Time::ZERO, |session| session.tool_wait_after);
        ready.push(Reverse((now + tool_wait_after, next)));
    }
}

impl ReplayPacing {
    fn has_capacity(self, in_flight: u64) -> bool {
        match self {
            Self::OpenLoop { .. } => true,
            Self::ClosedLoop { max_concurrency } => in_flight < max_concurrency as u64,
        }
    }

    fn release_time(self, release: &ReleaseMetadata, now: Time) -> Option<Time> {
        match self {
            Self::OpenLoop { request_rate } => {
                let due = effective_arrival(release.trace_arrival_time_ms, request_rate);
                (due <= now).then_some(due)
            }
            Self::ClosedLoop { .. } => Some(now),
        }
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
            successor[previous] = Some(index as u32);
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
