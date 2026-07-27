//! L7-γ trace frontend — loads single-round workload CSV(s) once at startup
//! into an ordered arrival queue, then emits `Request`s during the tick loop.
//!
//! Two replay modes:
//! - **Open-loop** (default): emit each request at its scheduled arrival time.
//! - **Closed-loop** (`max_concurrency` set): IGNORE the CSV arrival timeline
//!   and instead keep at most N requests in flight, admitting the next one the
//!   instant a slot frees. This mirrors the alignment load-generator's
//!   `--max-concurrency` (a tokio `Semaphore(N)` acquired *after* arrival, held
//!   until completion).
//!
//! Closed-loop additionally supports **session-dependent** admission
//! (`session_dependent`): round N+1 of a session becomes eligible only after
//! round N completes, mirroring a real agent that cannot issue its next call
//! before receiving the previous response. Eligible requests are emitted from a
//! ready queue in trace order, so admission is non-FIFO across sessions (a
//! blocked session never head-of-line-blocks an independent one). Think time
//! between rounds is not modeled (successor eligible the tick after its
//! predecessor completes) — measured load is an upper bound.
//!
//! To gate closed-loop, the frontend must know how many requests are in flight,
//! so it OWNS the in-flight ledger: `cursor` (emitted count) minus `completed`
//! (fed back by the run loop via [`TraceFrontend::record_completion`]). The run
//! loop is a pure consumer of these counts — it keeps no arrival/completion
//! tally of its own.
//!
//! Single-round shape (mirrors `ref/moesim-rs/src/trace/mod.rs`):
//! `id,input_len,output_len,arrival_time`, plus an optional `prefix_kv` column
//! declaring KV tokens already cached at arrival (a prefix-cache hit measured
//! by the trace producer): `input_len` then counts only the uncached suffix
//! that is actually prefilled, while `prefix_kv` still occupies KV and is
//! attended over. Absent column = 0 (no cached prefix). Multi-round (a
//! `round_idx` column) is rejected — per-round `Request` allocation +
//! continuation linkage is deferred (see L7 design §3.2/§3.3).

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::common::{Request, RequestId, Time};

/// One single-round trace row.
#[derive(Clone, Debug, Deserialize)]
pub struct TraceEntry {
    pub id: u32,
    pub input_len: u32,
    pub output_len: u32,
    pub arrival_time: f64,
    /// KV tokens already cached at arrival (optional column; default 0).
    #[serde(default)]
    pub prefix_kv: u32,
    /// Session this request belongs to (optional column). Enables the
    /// prefix-cache model and `session-sticky` placement; absent = sessionless.
    #[serde(default)]
    pub session_id: Option<u32>,
}

/// Immutable arrival queue + a cursor into it, plus the in-flight ledger.
/// `drain_due` advances the cursor in arrival order; the trace is never mutated
/// after `load`. In open-loop, CSV `arrival_time` is normalized to rate=1, so
/// the effective arrival is `arrival_time / request_rate` (higher rate ⇒ sooner
/// arrivals). In closed-loop (`max_concurrency` set), the arrival timeline is
/// ignored and admission is gated on `in_flight() < max_concurrency`.
#[derive(Debug)]
pub struct TraceFrontend {
    entries: Vec<TraceEntry>,
    cursor: usize,
    request_rate: f64,
    /// Closed-loop concurrency cap. `None` = open-loop arrival replay.
    max_concurrency: Option<usize>,
    /// Requests the run loop has reported complete. With `emitted` this makes
    /// the frontend the sole owner of the in-flight ledger:
    /// `in_flight = emitted - completed`.
    completed: u64,
    /// Count of requests emitted so far. Equals `cursor` in the in-order modes;
    /// in session-dependent mode emission is ready-queue order, so this is the
    /// authoritative emitted tally (`cursor` stays 0).
    emitted: usize,
    /// Session-dependent admission state, `Some` only when enabled:
    /// per-request successor id (the next round of the same session, if any)
    /// and the ready queue of eligible-but-unemitted requests.
    session_dep: Option<SessionDep>,
}

/// Session-dependency bookkeeping: `succ_of[id]` is the id of the next round of
/// the same session (`None` for a session's last round / sessionless rows);
/// `ready` holds eligible-but-unemitted ids, initialized to every request with
/// no predecessor (session heads + sessionless), in trace order.
#[derive(Debug)]
struct SessionDep {
    succ_of: Vec<Option<u32>>,
    ready: std::collections::VecDeque<u32>,
}

/// Effective arrival time: the rate-1-normalized `arrival_time` remapped to the
/// target `request_rate` (req/s). Higher rate ⇒ arrivals sooner.
fn effective_arrival(arrival_time: f64, request_rate: f64) -> Time {
    Time::from_ms(arrival_time / request_rate)
}

impl TraceFrontend {
    /// Load + validate one or more single-round CSV files (concatenated in the
    /// given order). Each file must carry the 4-column single-round header.
    /// `request_rate` (req/s, > 0) remaps the rate-1-normalized arrival times in
    /// open-loop. `max_concurrency` (> 0 when set) switches to closed-loop
    /// replay: the arrival timeline / `request_rate` are ignored and admission
    /// is gated on the in-flight count instead.
    pub fn load(
        files: &[PathBuf],
        request_rate: f64,
        max_concurrency: Option<usize>,
        session_dependent: bool,
    ) -> Result<Self> {
        if files.is_empty() {
            bail!("no trace files given (--trace-files)");
        }
        if !(request_rate.is_finite() && request_rate > 0.0) {
            bail!("request_rate must be finite and > 0 (got {request_rate})");
        }
        if max_concurrency == Some(0) {
            bail!("max_concurrency must be greater than 0 (got 0)");
        }
        if session_dependent && max_concurrency.is_none() {
            bail!("session_dependent requires closed-loop replay (set max_concurrency)");
        }
        let mut entries = Vec::new();
        for file in files {
            load_one(file, &mut entries)?;
        }
        if entries.is_empty() {
            bail!("trace files contained no rows");
        }
        validate_single_round(&entries)?;
        let session_dep = session_dependent.then(|| build_session_dep(&entries));
        Ok(Self {
            entries,
            cursor: 0,
            request_rate,
            max_concurrency,
            completed: 0,
            emitted: 0,
            session_dep,
        })
    }

    /// Total number of requests in the trace (capacity hint for `RequestStore`).
    pub fn expected_count(&self) -> usize {
        self.entries.len()
    }

    /// All arrivals have been emitted.
    pub fn exhausted(&self) -> bool {
        self.emitted >= self.entries.len()
    }

    /// Whether session-dependent admission is enabled.
    pub fn session_dependent(&self) -> bool {
        self.session_dep.is_some()
    }

    /// Insert every trace row into `store` up front (dense id order, trace
    /// arrival facts, zero progress). Session-dependent mode emits requests
    /// out of trace order, which the store's dense in-order `insert` cannot
    /// accept at emission time — so the driver pre-fills the store here and
    /// flows `upsert` each record at admission (re-stamping the arrival clock).
    /// Call once, before the tick loop.
    pub fn preinsert_all(&self, store: &crate::common::SharedRequests) {
        let mut s = store.borrow_mut();
        for e in &self.entries {
            let req = Request::new(
                RequestId(e.id),
                e.input_len,
                e.output_len,
                Time::from_ms(e.arrival_time),
            )
            .with_prefix_kv(e.prefix_kv)
            .with_session(e.session_id);
            s.insert(&req);
        }
    }

    /// Emit due requests, in trace order, to `emit`. The drain loop lives here so
    /// callers can't under-drain by polling once per tick — a single call empties
    /// the tick. What "due" means depends on the mode:
    /// - **open-loop**: effective `arrival_time <= now`; the request keeps its
    ///   rate-remapped arrival time.
    /// - **closed-loop**: `in_flight() < max_concurrency`, ignoring the arrival
    ///   timeline; the request's arrival is stamped `now` (the admission clock),
    ///   matching the load-generator's post-`acquire` submit. Each emit bumps
    ///   `cursor`, so `in_flight()` rises within the loop and admission stops
    ///   exactly at the cap.
    pub fn drain_due(&mut self, now: Time, mut emit: impl FnMut(Request)) {
        // Session-dependent closed-loop: emit from the ready queue (eligible =
        // predecessor round completed), capped by in-flight. Arrival is the
        // admission clock, matching plain closed-loop.
        if self.session_dep.is_some() {
            let cap = self
                .max_concurrency
                .expect("session_dep implies closed-loop");
            while self.in_flight() < cap as u64 {
                let Some(id) = self.session_dep.as_mut().and_then(|d| d.ready.pop_front()) else {
                    break;
                };
                let e = &self.entries[id as usize];
                let req = Request::new(RequestId(e.id), e.input_len, e.output_len, now)
                    .with_prefix_kv(e.prefix_kv)
                    .with_session(e.session_id);
                self.emitted += 1;
                emit(req);
            }
            return;
        }
        while self.cursor < self.entries.len() {
            let arrival = match self.max_concurrency {
                Some(cap) => {
                    if self.in_flight() >= cap as u64 {
                        break;
                    }
                    now
                }
                None => {
                    let t = effective_arrival(
                        self.entries[self.cursor].arrival_time,
                        self.request_rate,
                    );
                    if t > now {
                        break;
                    }
                    t
                }
            };
            let e = &self.entries[self.cursor];
            let req = Request::new(RequestId(e.id), e.input_len, e.output_len, arrival)
                .with_prefix_kv(e.prefix_kv)
                .with_session(e.session_id);
            self.cursor += 1;
            self.emitted += 1;
            emit(req);
        }
    }

    /// Record one request the run loop reported complete. Paired with `emitted`,
    /// this keeps the in-flight ledger owned entirely here. In session-dependent
    /// mode, completion of round N unblocks round N+1 of the same session.
    pub fn record_completion(&mut self, id: RequestId) {
        self.completed += 1;
        if let Some(dep) = self.session_dep.as_mut() {
            if let Some(next) = dep.succ_of[id.0 as usize] {
                dep.ready.push_back(next);
            }
        }
    }

    /// Requests emitted but not yet completed (`submitted - completed`).
    /// Closed-loop gating and the run loop's termination/watchdog checks read it.
    pub fn in_flight(&self) -> u64 {
        self.emitted as u64 - self.completed
    }

    /// Total requests emitted into the system so far.
    pub fn submitted(&self) -> u64 {
        self.emitted as u64
    }

    /// Total requests the run loop has reported complete.
    pub fn num_completed(&self) -> u64 {
        self.completed
    }
}

/// Walk the trace once, linking each row to the next row of the same session
/// (`succ_of`), and seed `ready` with every row that has no predecessor
/// (session heads + sessionless rows), in trace order.
fn build_session_dep(entries: &[TraceEntry]) -> SessionDep {
    use std::collections::HashMap;
    let mut succ_of: Vec<Option<u32>> = vec![None; entries.len()];
    let mut ready = std::collections::VecDeque::new();
    // session → id of its most recent (so far, last) round.
    let mut last_of: HashMap<u32, u32> = HashMap::new();
    for e in entries {
        match e.session_id {
            Some(s) => {
                if let Some(prev) = last_of.insert(s, e.id) {
                    succ_of[prev as usize] = Some(e.id);
                } else {
                    ready.push_back(e.id);
                }
            }
            None => ready.push_back(e.id),
        }
    }
    SessionDep { succ_of, ready }
}

fn load_one(path: &Path, out: &mut Vec<TraceEntry>) -> Result<()> {
    let mut rdr = csv::Reader::from_path(path)
        .with_context(|| format!("opening trace file {}", path.display()))?;

    // Reject multi-round by header presence (matches ref's `round_idx` discriminator).
    let headers = rdr
        .headers()
        .with_context(|| format!("reading header of {}", path.display()))?;
    if headers.iter().any(|h| h == "round_idx") {
        bail!(
            "{}: multi-round traces (round_idx column) are not supported yet",
            path.display()
        );
    }

    for (i, result) in rdr.deserialize().enumerate() {
        let entry: TraceEntry =
            result.with_context(|| format!("{}: parsing row {i}", path.display()))?;
        if entry.input_len == 0 {
            bail!("{}: row {i} has input_len=0", path.display());
        }
        if entry.output_len == 0 {
            bail!("{}: row {i} has output_len=0", path.display());
        }
        if !entry.arrival_time.is_finite() || entry.arrival_time < 0.0 {
            bail!(
                "{}: row {i} has invalid arrival_time={} (must be finite, non-negative)",
                path.display(),
                entry.arrival_time
            );
        }
        out.push(entry);
    }
    Ok(())
}

/// Ids must be sequential `0..N`; `arrival_time` non-decreasing across rows.
fn validate_single_round(entries: &[TraceEntry]) -> Result<()> {
    for (i, e) in entries.iter().enumerate() {
        if e.id != i as u32 {
            bail!("trace row {i} has id={}, expected sequential id={i}", e.id);
        }
        if i > 0 && e.arrival_time < entries[i - 1].arrival_time {
            bail!(
                "trace row {i} arrival_time={} < previous {}",
                e.arrival_time,
                entries[i - 1].arrival_time
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_csv(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        path
    }

    fn drain_at(fe: &mut TraceFrontend, now_ms: f64) -> Vec<RequestId> {
        let mut ids = Vec::new();
        fe.drain_due(Time::from_ms(now_ms), |r| ids.push(r.id));
        ids
    }

    /// Drain returning `(id, arrival_ms)` so closed-loop tests can assert the
    /// admission-clock stamping, not just ordering.
    fn drain_pairs(fe: &mut TraceFrontend, now_ms: f64) -> Vec<(RequestId, f64)> {
        let mut out = Vec::new();
        fe.drain_due(Time::from_ms(now_ms), |r| {
            out.push((r.id, r.arrival_time.as_ms()))
        });
        out
    }

    #[test]
    fn drains_due_in_arrival_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "t.csv",
            "id,input_len,output_len,arrival_time\n\
             0,8,2,0.0\n\
             1,16,4,5.0\n\
             2,4,1,5.0\n",
        );
        // rate=1 → effective arrival == CSV arrival (ms).
        let mut fe = TraceFrontend::load(&[path], 1.0, None, false).unwrap();
        assert_eq!(fe.expected_count(), 3);

        // At t=0 only req 0 is due; a second drain at t=0 yields nothing.
        assert_eq!(drain_at(&mut fe, 0.0), vec![RequestId(0)]);
        assert_eq!(drain_at(&mut fe, 0.0), vec![]);

        // At t=5 both remaining drain in one call, in row order.
        assert_eq!(drain_at(&mut fe, 5.0), vec![RequestId(1), RequestId(2)]);
        assert!(fe.exhausted());
    }

    #[test]
    fn request_rate_compresses_arrivals() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "rate.csv",
            "id,input_len,output_len,arrival_time\n\
             0,8,2,0.0\n\
             1,8,2,10.0\n",
        );
        // rate=2 halves the rate-1 timeline: req 1 at 10.0/2 = 5.0ms.
        let mut fe = TraceFrontend::load(&[path], 2.0, None, false).unwrap();
        assert_eq!(drain_at(&mut fe, 4.9), vec![RequestId(0)]); // req 1 not due yet
        assert_eq!(drain_at(&mut fe, 5.0), vec![RequestId(1)]);
    }

    #[test]
    fn rejects_nonpositive_rate() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "r.csv",
            "id,input_len,output_len,arrival_time\n0,8,2,0.0\n",
        );
        assert!(TraceFrontend::load(&[path], 0.0, None, false).is_err());
    }

    #[test]
    fn prefix_kv_column_is_optional_and_passed_through() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "pfx.csv",
            "id,input_len,output_len,arrival_time,prefix_kv\n\
             0,8,2,0.0,0\n\
             1,16,4,1.0,4096\n",
        );
        let mut fe = TraceFrontend::load(&[path], 1.0, None, false).unwrap();
        let mut reqs = Vec::new();
        fe.drain_due(Time::from_ms(10.0), |r| reqs.push(r));
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].prefix_kv, 0);
        assert_eq!(reqs[1].prefix_kv, 4096);
        // 4-column traces (no prefix_kv header) default the field to 0.
        let p4 = write_csv(
            dir.path(),
            "p4.csv",
            "id,input_len,output_len,arrival_time\n0,8,2,0.0\n",
        );
        let mut fe4 = TraceFrontend::load(&[p4], 1.0, None, false).unwrap();
        let mut r4 = Vec::new();
        fe4.drain_due(Time::from_ms(0.0), |r| r4.push(r));
        assert_eq!(r4[0].prefix_kv, 0);
    }

    #[test]
    fn rejects_multi_round() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "mr.csv",
            "id,input_len,output_len,arrival_time,round_idx,tool_wait_after_ms,prefix_len\n\
             0,100,20,0.0,0,0.0,0\n",
        );
        let err = TraceFrontend::load(&[path], 1.0, None, false).unwrap_err();
        assert!(err.to_string().contains("multi-round"));
    }

    #[test]
    fn rejects_non_sequential_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "bad.csv",
            "id,input_len,output_len,arrival_time\n\
             0,8,2,0.0\n\
             5,8,2,1.0\n",
        );
        assert!(TraceFrontend::load(&[path], 1.0, None, false).is_err());
    }

    #[test]
    fn rejects_decreasing_arrival() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "bad2.csv",
            "id,input_len,output_len,arrival_time\n\
             0,8,2,5.0\n\
             1,8,2,1.0\n",
        );
        assert!(TraceFrontend::load(&[path], 1.0, None, false).is_err());
    }

    #[test]
    fn closed_loop_caps_in_flight_and_stamps_admission_clock() {
        let dir = tempfile::tempdir().unwrap();
        // Spread-out CSV arrival times — all IGNORED in closed-loop.
        let path = write_csv(
            dir.path(),
            "cl.csv",
            "id,input_len,output_len,arrival_time\n\
             0,8,2,0.0\n\
             1,8,2,100.0\n\
             2,8,2,200.0\n\
             3,8,2,300.0\n",
        );
        // cap=2: at t=10 only two admit despite all four being past their CSV
        // arrival, and both are stamped with the admission clock (10ms).
        let mut fe = TraceFrontend::load(&[path], 1.0, Some(2), false).unwrap();
        assert_eq!(
            drain_pairs(&mut fe, 10.0),
            vec![(RequestId(0), 10.0), (RequestId(1), 10.0)]
        );
        assert_eq!(fe.in_flight(), 2);
        assert_eq!(fe.submitted(), 2);

        // Full: a second drain admits nothing until a slot frees.
        assert_eq!(drain_pairs(&mut fe, 20.0), vec![]);

        // One completes → exactly one slot opens; the next admits at 30ms.
        fe.record_completion(RequestId(0));
        assert_eq!(fe.in_flight(), 1);
        assert_eq!(drain_pairs(&mut fe, 30.0), vec![(RequestId(2), 30.0)]);
        assert_eq!(fe.in_flight(), 2);

        // Complete the rest; the tail request drains and the ledger zeroes out.
        fe.record_completion(RequestId(1));
        fe.record_completion(RequestId(2));
        assert_eq!(drain_pairs(&mut fe, 40.0), vec![(RequestId(3), 40.0)]);
        assert!(fe.exhausted());
        fe.record_completion(RequestId(3));
        assert_eq!(fe.in_flight(), 0);
        assert_eq!(fe.num_completed(), 4);
    }

    #[test]
    fn session_dependent_gates_rounds_and_skips_blocked_sessions() {
        let dir = tempfile::tempdir().unwrap();
        // Session 7: rounds 0→2→3; session 9: round 1; row 4 sessionless.
        let path = write_csv(
            dir.path(),
            "sd.csv",
            "id,input_len,output_len,arrival_time,prefix_kv,session_id\n\
             0,8,2,0.0,0,7\n\
             1,8,2,1.0,0,9\n\
             2,8,2,2.0,4,7\n\
             3,8,2,3.0,8,7\n\
             4,8,2,4.0,0,\n",
        );
        let mut fe = TraceFrontend::load(&[path], 1.0, Some(3), true).unwrap();
        assert!(fe.session_dependent());

        // Initially ready: session heads (0, 1) + sessionless (4). Rounds 2/3
        // are blocked behind 0 even though the cap (3) has room for them.
        assert_eq!(
            drain_at(&mut fe, 0.0),
            vec![RequestId(0), RequestId(1), RequestId(4)]
        );
        assert_eq!(drain_at(&mut fe, 1.0), vec![]);

        // Completing session 9's round frees a slot but unblocks nothing new.
        fe.record_completion(RequestId(1));
        assert_eq!(drain_at(&mut fe, 2.0), vec![]);

        // Completing round 0 unblocks round 2 (only), which admits with its
        // prefix_kv intact; round 3 stays blocked behind round 2.
        fe.record_completion(RequestId(0));
        let mut reqs = Vec::new();
        fe.drain_due(Time::from_ms(3.0), |r| reqs.push(r));
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].id, RequestId(2));
        assert_eq!(reqs[0].prefix_kv, 4);
        assert_eq!(reqs[0].arrival_time.as_ms(), 3.0);

        fe.record_completion(RequestId(2));
        assert_eq!(drain_at(&mut fe, 4.0), vec![RequestId(3)]);
        assert!(fe.exhausted());
        fe.record_completion(RequestId(4));
        fe.record_completion(RequestId(3));
        assert_eq!(fe.in_flight(), 0);
    }

    #[test]
    fn session_dependent_requires_closed_loop() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "sdo.csv",
            "id,input_len,output_len,arrival_time\n0,8,2,0.0\n",
        );
        let err = TraceFrontend::load(&[path], 1.0, None, true).unwrap_err();
        assert!(err.to_string().contains("session_dependent"));
    }

    #[test]
    fn rejects_zero_max_concurrency() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(
            dir.path(),
            "z.csv",
            "id,input_len,output_len,arrival_time\n0,8,2,0.0\n",
        );
        let err = TraceFrontend::load(&[path], 1.0, Some(0), false).unwrap_err();
        assert!(err.to_string().contains("max_concurrency"));
    }
}
