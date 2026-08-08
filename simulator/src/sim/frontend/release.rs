//! WHEN a loaded arrival is allowed to enter the system.
//!
//! This is the axis of the frontend that actually varies. Replaying a trace's
//! own timeline, capping in-flight concurrency, and chaining a conversation's
//! rounds behind each other are three different disciplines over the same
//! arrival list — they are not three different kinds of request.
//!
//! Each mode carries its own state inside its own variant, so a field can never
//! be live in a mode that does not read it: `request_rate` exists only where
//! trace arrival times are replayed, `cap` only where a cap is meaningful, the
//! chain bookkeeping only under the scheduler's `SessionChain` state. The pre-refactor
//! shape kept all of them side by side on one struct and discriminated on
//! "which `Option` is set", which made `request_rate` a dead field in
//! closed-loop and would have needed a second `Option` to express a third mode.
//!
//! Shaped after `worker::admission::LoadBalance`: a state-carrying enum with one
//! `&mut self` method per event, not a trait object. The set of modes is closed
//! and lives here, so an enum keeps dispatch in a single exhaustiveness-checked
//! `match` — adding a mode makes the compiler point at every place that must
//! handle it.

use anyhow::{bail, Result};

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use super::ReleaseMetadata;
use crate::common::{RequestId, Time};

/// The pacing discipline a caller asks for, resolved from config once.
///
/// The *selection* is a closed three-way choice, so it is a three-way type —
/// even though the config layer spells it as two independent fields. Collapsing
/// that pair happens exactly once, where the config is read; nothing downstream
/// re-derives "are we closed-loop?" from a set `Option`.
///
/// Distinct from [`ReplayScheduler`] the way configuration is distinct from
/// running state: this is the pure request, and the scheduler serves it.
#[derive(Clone, Copy, Debug)]
pub enum ReplayMode {
    /// Replay the trace's own arrival timeline.
    OpenLoop { request_rate: f64 },
    /// Ignore the timeline; keep at most `cap` requests in flight.
    ClosedLoop { cap: usize },
    /// Replay session-head arrivals from the timeline, but hold each later round
    /// of a conversation until its predecessor completes.
    SessionChain { request_rate: f64 },
}

impl ReplayMode {
    /// The closed vocabulary, handed to the launcher as the `replay_mode`
    /// param's `choices` so a misspelling fails config validation.
    pub const CHOICES: &'static [&'static str] = &["open_loop", "closed_loop", "session_chain"];

    /// Build from the declared name plus the payload fields the flat config
    /// carries alongside it.
    ///
    /// A flat config cannot nest a payload under its variant, so `request_rate`
    /// and `max_concurrency` sit beside the name. This is where that flatness is
    /// made safe: each arm states which payloads its mode reads and rejects the
    /// ones it would silently ignore. Setting `max_concurrency` under
    /// `open_loop` is a mistake worth a hard error, not a dead field.
    pub fn parse(name: &str, request_rate: f64, max_concurrency: Option<usize>) -> Result<Self> {
        let unused_cap = |mode: &str| -> anyhow::Error {
            anyhow::anyhow!(
                "workload.max_concurrency is set but replay_mode is {mode:?}, which \
                 replays the trace's arrival timeline and would ignore it — either \
                 drop max_concurrency or declare replay_mode: closed_loop"
            )
        };
        Ok(match name {
            "open_loop" => match max_concurrency {
                Some(_) => return Err(unused_cap(name)),
                None => Self::OpenLoop { request_rate },
            },
            "session_chain" => match max_concurrency {
                Some(_) => return Err(unused_cap(name)),
                None => Self::SessionChain { request_rate },
            },
            "closed_loop" => match max_concurrency {
                Some(cap) => Self::ClosedLoop { cap },
                None => bail!(
                    "replay_mode: closed_loop needs workload.max_concurrency — the \
                     cap is the whole discipline"
                ),
            },
            other => bail!(
                "unknown replay_mode {other:?} (expected one of {:?})",
                Self::CHOICES
            ),
        })
    }
}

#[derive(Debug)]
pub(super) struct ReplayScheduler {
    state: ReplayState,
}

#[derive(Debug)]
enum ReplayState {
    /// Replay the trace's own arrival timeline. CSV `arrival_time` is normalized
    /// to rate=1, so the effective arrival is `arrival_time / request_rate`
    /// (higher rate ⇒ sooner arrivals).
    OpenLoop { cursor: usize, request_rate: f64 },
    /// Ignore the arrival timeline; keep at most `cap` requests in flight and
    /// release the next one the instant a slot frees. Mirrors the alignment
    /// load-generator's `--max-concurrency` (a tokio `Semaphore(N)` acquired
    /// *after* arrival, held until completion), so a released request is stamped
    /// with the admission clock rather than its trace arrival.
    ClosedLoop { cursor: usize, cap: usize },
    /// A conversation's rounds enter one at a time: a session **head** replays
    /// its own trace arrival, and every later round waits for its predecessor to
    /// complete plus that predecessor's `tool_wait`.
    ///
    /// A successor deliberately ignores its own trace `arrival_time`. That
    /// timestamp records how fast the machine the trace was captured on served
    /// the previous round — which is the very thing this simulation exists to
    /// predict. Replaying it would smuggle the recording system's throughput in
    /// as a floor.
    ///
    /// With no session columns in the trace every row is a head, so this mode
    /// degenerates exactly to `OpenLoop`.
    SessionChain {
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
        request_rate: f64,
    },
}

impl ReplayScheduler {
    /// Resolve a [`ReplayMode`] into running state. Only `SessionChain` needs to
    /// look at the arrivals — it derives the per-session chains up front, in one
    /// pass, because the links never change after load.
    pub(super) fn new(mode: ReplayMode, releases: &[ReleaseMetadata]) -> Self {
        let state = match mode {
            ReplayMode::OpenLoop { request_rate } => ReplayState::OpenLoop {
                cursor: 0,
                request_rate,
            },
            ReplayMode::ClosedLoop { cap } => ReplayState::ClosedLoop { cursor: 0, cap },
            ReplayMode::SessionChain { request_rate } => {
                let (successor, has_predecessor) = build_chains(releases);
                ReplayState::SessionChain {
                    successor,
                    has_predecessor,
                    cursor: 0,
                    ready: BinaryHeap::new(),
                    request_rate,
                }
            }
        };
        Self { state }
    }

    /// The next arrival that may enter right now: its index into `arrivals` plus
    /// the arrival time to stamp it with. `None` = nothing releasable this tick.
    ///
    /// Advances this mode's own cursor on success, so the caller's drain loop is
    /// just "keep asking until it says no".
    ///
    /// The scheduler accepts only [`ReleaseMetadata`], so it cannot inspect a
    /// request definition even accidentally.
    pub(super) fn next_ready(
        &mut self,
        releases: &[ReleaseMetadata],
        in_flight: u64,
        now: Time,
    ) -> Option<(usize, Time)> {
        match &mut self.state {
            ReplayState::OpenLoop {
                cursor,
                request_rate,
            } => {
                let arrival =
                    effective_arrival(releases.get(*cursor)?.trace_arrival_time_ms, *request_rate);
                if arrival > now {
                    return None;
                }
                Some((take(cursor), arrival))
            }
            ReplayState::ClosedLoop { cursor, cap } => {
                if *cursor >= releases.len() || in_flight >= *cap as u64 {
                    return None;
                }
                Some((take(cursor), now))
            }
            ReplayState::SessionChain {
                has_predecessor,
                cursor,
                ready,
                request_rate,
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
                    let due = effective_arrival(release.trace_arrival_time_ms, *request_rate);
                    if due > now {
                        return None;
                    }
                    return Some((take(cursor), due));
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
        let ReplayState::SessionChain {
            successor, ready, ..
        } = &mut self.state
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

/// Link each session's rows into a chain, in trace order.
///
/// One pass, keyed by `session_id`: whatever row of a session was seen last
/// becomes the predecessor of the next one. Rows declaring no session are left
/// unlinked, which makes them heads — that is what degenerates this mode to
/// open-loop replay on a trace with no session columns.
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
