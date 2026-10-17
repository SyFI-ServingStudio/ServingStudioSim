//! Reads of session contexts from DRAM/SSD tiers behind HBM, issued when
//! admission reaches the request, as vLLM's KV connector does.
//!
//! vLLM asks its connector (LMCache, the offloading connector) for external
//! hits when the scheduler takes a request from the waiting queue, not when the
//! request arrives. A request whose context a slower tier holds, beyond what
//! HBM still has, leaves the queue while its read runs
//! (`WAITING_FOR_REMOTE_KVS`), and other requests go ahead. Once the read lands
//! it returns to the front, and its context goes back into HBM just before
//! admission resolves its prefix. A lookup at arrival instead misses every
//! context HBM evicts while the request queues, which under a deep queue is
//! all of them.
//!
//! A cache-aware order (shortest prefill first) ranks a request by what HBM or
//! a tier holds, since either is reusable.
//!
//! With `warm_start`, a session request whose declared prefix exceeds every
//! context the run stored for that session by more than one token is a
//! conversation that began before the run: its context is assumed in the
//! slowest tier, as a long-lived session's would be in steady state. (A stored
//! context lacks its last output token, whose KV the next round computes, so
//! the next round may declare one token more.)
//!
//! With a victim DRAM tier (`dram_tier_writes: hbm-evictions`), the KV store
//! logs every context its HBM prefix cache lets go of, and the tiers take
//! them before any lookup, read, or write, in the order they left HBM. A
//! victim SSD tier (`ssd_tier_writes: dram-evictions`) takes what DRAM lets go
//! of. A session request's admission cuts each victim entry to what it shares
//! with it.
//!
//! One tier set per KV partition (a pipeline head's one, or each attention DP
//! rank's). `prefix_tiers_<tag>.csv` records, per admitted session request,
//! where its declared prefix was found; `prefix_tiers_<tag>.json` the totals.
//! The tag is `w<worker>`, or `w<worker>_p<rank>` for a DP rank.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::common::{RequestId, SessionInput, Time};
use crate::worker::config::PrefixTierReadWaitPolicy;
use crate::worker::kv::{
    HbmEvictionLog, PrefixKv, PrefixTierHit, PrefixTierSpec, PrefixTiers, ResumePoint, RoundContext,
    WRITE_WINDOW,
};

use super::AdmissionCandidate;

/// One partition's tiers and log.
pub(crate) struct SessionPrefixTiers {
    tiers: PrefixTiers,
    /// Where a context can be resumed from, the same in HBM and the tiers.
    resume: ResumePoint,
    /// The longest context the run stored for each session, so `warm_start`
    /// can tell a pre-run context from an evicted one.
    stored: HashMap<u32, u64>,
    rows: Option<BufWriter<File>>,
    /// Every write, eviction, cut and read of each tier.
    events: Option<BufWriter<File>>,
    summary_path: Option<PathBuf>,
    /// Session requests admitted, their declared prefix tokens, and where
    /// those were found: HBM, each tier, or nowhere.
    requests: u64,
    declared_tokens: u64,
    hbm_tokens: u64,
    resident_tokens: u64,
    tier_tokens: Vec<u64>,
    tier_requests: Vec<u64>,
    warm_start_reads: u64,
    warm_start_tokens: u64,
    /// Reads of a request whose context an earlier read put back into HBM but
    /// which was evicted before admission.
    rereads: u64,
}

impl SessionPrefixTiers {
    pub(crate) fn new(
        specs: &[PrefixTierSpec],
        kv_bytes_per_token: u64,
        state_tokens: u64,
        resume: ResumePoint,
        round_end_states: u32,
        log_dir: Option<&Path>,
        tag: &str,
    ) -> Self {
        let rows = log_dir.map(|dir| {
            let mut file = BufWriter::new(
                File::create(dir.join(format!("prefix_tiers_{tag}.csv")))
                    .expect("create the prefix tier log"),
            );
            writeln!(
                file,
                "request_id,session_id,read_ms,admit_ms,declared_tokens,hbm_tokens,tier,tier_tokens,ready_ms,resident_tokens"
            )
            .expect("write the prefix tier log");
            file
        });
        let events = log_dir.map(|dir| {
            let mut file = BufWriter::new(
                File::create(dir.join(format!("prefix_tier_events_{tag}.csv")))
                    .expect("create the prefix tier event log"),
            );
            writeln!(
                file,
                "time_ms,tier,event,session_id,tokens,prev_tokens,declared"
            )
            .expect("write the prefix tier event log");
            file
        });
        let mut tiers = PrefixTiers::new(specs, kv_bytes_per_token, state_tokens, resume)
            .with_round_end_states(round_end_states);
        if events.is_some() {
            tiers.record_events();
        }
        Self {
            tiers,
            resume,
            stored: HashMap::new(),
            rows,
            events,
            summary_path: log_dir.map(|dir| dir.join(format!("prefix_tiers_{tag}.json"))),
            requests: 0,
            declared_tokens: 0,
            hbm_tokens: 0,
            resident_tokens: 0,
            tier_tokens: vec![0; specs.len()],
            tier_requests: vec![0; specs.len()],
            warm_start_reads: 0,
            warm_start_tokens: 0,
            rereads: 0,
        }
    }

    /// Where a read of the session's declared prefix would come from, and how
    /// much of it resumes; `seeded` when it is a pre-run context.
    fn lookup(
        &self,
        session_id: u32,
        declared_tokens: u32,
        warm_start: bool,
    ) -> Option<(PrefixTierHit, bool)> {
        let stored = self.stored.get(&session_id).copied().unwrap_or(0);
        if warm_start && u64::from(declared_tokens) > stored + 1 && self.tiers.len() > 0 {
            let declared = u64::from(declared_tokens);
            let tokens = self.resume.reusable_tokens(declared, declared) as u32;
            return (tokens > 0).then_some((
                PrefixTierHit {
                    tier: self.tiers.len() - 1,
                    tokens,
                },
                true,
            ));
        }
        self.tiers
            .lookup(session_id, declared_tokens)
            .map(|hit| (hit, false))
    }

    pub(crate) fn store(&mut self, session_id: u32, round: RoundContext) {
        let stored = self.stored.entry(session_id).or_insert(0);
        *stored = (*stored).max(round.tokens);
        self.tiers.store_round(session_id, round);
    }

    /// Start a read of `hit`, of what HBM's `hbm_tokens` lack; returns when it
    /// lands.
    fn read(
        &mut self,
        session_id: u32,
        hit: PrefixTierHit,
        seeded: bool,
        hbm_tokens: u32,
        now: Time,
    ) -> Time {
        if seeded {
            self.tiers.seed(session_id, u64::from(hit.tokens));
            let stored = self.stored.entry(session_id).or_insert(0);
            *stored = (*stored).max(u64::from(hit.tokens));
            self.warm_start_reads += 1;
            self.warm_start_tokens += u64::from(hit.tokens);
        }
        self.tiers.load(session_id, hit, hbm_tokens, now)
    }

    fn queue_wait(&self, hit: PrefixTierHit, now: Time) -> Time {
        self.tiers.queue_wait(hit.tier, now)
    }

    /// Append the tiers' events since the last call to the event log.
    fn write_events(&mut self) {
        let Some(file) = &mut self.events else {
            return;
        };
        for tier in 0..self.tiers.len() {
            let name = self.tiers.tier_name(tier);
            for e in self.tiers.take_events(tier) {
                writeln!(
                    file,
                    "{:.3},{},{},{},{},{},{}",
                    e.at.as_ms(),
                    name,
                    format!("{:?}", e.kind).to_lowercase(),
                    e.session_id,
                    e.tokens,
                    e.prev_tokens,
                    e.declared,
                )
                .expect("write the prefix tier event log");
            }
        }
    }

    fn record_admission(&mut self, request: RequestId, session_id: u32, admission: Admitted) {
        self.write_events();
        self.requests += 1;
        self.declared_tokens += u64::from(admission.declared_tokens);
        self.resident_tokens += u64::from(admission.resident_tokens);
        match admission.read {
            Some(read) => {
                self.tier_tokens[read.hit.tier] += u64::from(read.hit.tokens);
                self.tier_requests[read.hit.tier] += 1;
            }
            None => self.hbm_tokens += u64::from(admission.resident_tokens),
        }
        if let Some(rows) = &mut self.rows {
            let (tier, tokens, read_ms, ready_ms) = match admission.read {
                Some(read) => (
                    self.tiers.tier_name(read.hit.tier),
                    read.hit.tokens,
                    read.started.as_ms(),
                    read.ready.as_ms(),
                ),
                None if admission.resident_tokens > 0 => (
                    "hbm",
                    admission.resident_tokens,
                    admission.at.as_ms(),
                    admission.at.as_ms(),
                ),
                None => ("none", 0, admission.at.as_ms(), admission.at.as_ms()),
            };
            let hbm_tokens = admission
                .read
                .map_or(admission.resident_tokens, |read| read.hbm_tokens);
            writeln!(
                rows,
                "{},{},{:.3},{:.3},{},{},{},{},{:.3},{}",
                request.0,
                session_id,
                read_ms,
                admission.at.as_ms(),
                admission.declared_tokens,
                hbm_tokens,
                tier,
                tokens,
                ready_ms,
                admission.resident_tokens,
            )
            .expect("write the prefix tier log");
        }
    }
}

impl Drop for SessionPrefixTiers {
    fn drop(&mut self) {
        if let Some(rows) = &mut self.rows {
            let _ = rows.flush();
        }
        self.write_events();
        if let Some(events) = &mut self.events {
            let _ = events.flush();
        }
        let Some(path) = &self.summary_path else {
            return;
        };
        let tiers: Vec<serde_json::Value> = (0..self.tiers.len())
            .map(|tier| {
                let counters = self.tiers.counters(tier);
                serde_json::json!({
                    "tier": self.tiers.tier_name(tier),
                    "capacity_tokens": self.tiers.capacity_tokens(tier),
                    "used_tokens": self.tiers.used_tokens(tier),
                    "hit_requests": self.tier_requests[tier],
                    "hit_tokens": self.tier_tokens[tier],
                    "stored_entries": counters.stored_entries,
                    "written_tokens": counters.written_tokens,
                    "evicted_entries": counters.evicted_entries,
                    "evicted_tokens": counters.evicted_tokens,
                    "loads": counters.loads,
                    "loaded_tokens": counters.loaded_tokens,
                    "written_states": counters.written_states,
                    "earlier_state_drops": counters.earlier_state_drops,
                    "earlier_states": self.tiers.earlier_states(tier),
                    "write_window_ms": WRITE_WINDOW.as_ms(),
                    "window_writes": self.tiers.window_writes(tier),
                })
            })
            .collect();
        let summary = serde_json::json!({
            "session_requests": self.requests,
            "declared_tokens": self.declared_tokens,
            "hbm_hit_tokens": self.hbm_tokens,
            "resident_tokens_at_admission": self.resident_tokens,
            "warm_start_reads": self.warm_start_reads,
            "warm_start_tokens": self.warm_start_tokens,
            "rereads": self.rereads,
            "tiers": tiers,
        });
        if let Ok(file) = File::create(path) {
            let _ = serde_json::to_writer_pretty(file, &summary);
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Read {
    hit: PrefixTierHit,
    /// What HBM held when the read started.
    hbm_tokens: u32,
    started: Time,
    ready: Time,
}

#[derive(Clone, Copy, Debug)]
struct Admitted {
    declared_tokens: u32,
    resident_tokens: u32,
    read: Option<Read>,
    at: Time,
}

#[derive(Clone, Copy, Debug)]
struct Fetch {
    candidate: AdmissionCandidate,
    partition: u16,
    read: Read,
}

/// What admission does with the request at its queue's head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AtHead {
    /// Resolve its prefix and admit it if it fits.
    Admit,
    /// A read started: take it out of the queue until the read lands.
    Read,
    /// A read is needed but the request does not fit in HBM beside what is
    /// running and the other reads' holds (vLLM's waiting loop breaks when an
    /// async load cannot allocate its blocks), or, under the `block` policy,
    /// its tier's queue is past the read-wait bound: stop admitting.
    Blocked,
    /// A read is needed but its tier's queue is past the read-wait bound
    /// (`pass-over` policy): pass it over, holding nothing, and admit the
    /// requests behind it.
    PassedOver,
}

/// The tiers of every partition and the reads in flight.
pub(crate) struct PrefixFetch {
    partitions: Vec<SessionPrefixTiers>,
    warm_start: bool,
    /// Start a read only while its tier would begin it within this long.
    max_read_wait: Option<Time>,
    /// What a request past `max_read_wait` gets.
    read_wait_policy: PrefixTierReadWaitPolicy,
    /// Reads in flight, by landing time.
    reads: BinaryHeap<Reverse<(Time, RequestId)>>,
    /// Requests out of their queue while their read runs.
    reading: HashMap<RequestId, Fetch>,
    /// Requests whose read has landed, still holding their HBM blocks. Each
    /// partition admits them before its own queue, in landing order, as
    /// vLLM's FCFS scheduler takes its skipped-waiting queue (loaded async
    /// requests) first; the context goes into HBM when admission reaches them.
    landed: HashMap<RequestId, Fetch>,
    landed_order: Vec<VecDeque<RequestId>>,
    /// Landed requests whose context went back into HBM when admission
    /// reached them, until admitted. One left unadmitted then re-checks HBM
    /// at each later visit and is read again if its context was evicted.
    restored: HashMap<RequestId, Fetch>,
    /// Cancelled requests whose holds the next [`Self::land`] gives back.
    cancelled: Vec<RequestId>,
    /// What the KV store's HBM prefix cache let go of, for a victim tier.
    hbm_evictions: Option<HbmEvictionLog>,
}

impl PrefixFetch {
    pub(crate) fn new(partitions: Vec<SessionPrefixTiers>, warm_start: bool) -> Self {
        assert!(
            !partitions.is_empty(),
            "prefix fetch needs one tier set per partition"
        );
        let landed_order = vec![VecDeque::new(); partitions.len()];
        Self {
            partitions,
            warm_start,
            max_read_wait: None,
            read_wait_policy: PrefixTierReadWaitPolicy::PassOver,
            reads: BinaryHeap::new(),
            reading: HashMap::new(),
            landed: HashMap::new(),
            landed_order,
            restored: HashMap::new(),
            cancelled: Vec::new(),
            hbm_evictions: None,
        }
    }

    /// The log the KV store reports HBM evictions to, when the fastest tier
    /// is HBM's victim tier; `None` otherwise.
    pub(crate) fn hbm_eviction_log(&mut self) -> Option<HbmEvictionLog> {
        if !self.partitions[0].tiers.takes_hbm_evictions() {
            return None;
        }
        Some(
            self.hbm_evictions
                .get_or_insert_with(Default::default)
                .clone(),
        )
    }

    /// Count the writes from here on at `now`.
    fn set_clock(&mut self, now: Time) {
        for partition in &mut self.partitions {
            partition.tiers.set_clock(now);
        }
    }

    /// Write what HBM let go of since the last call to the victim tiers.
    fn take_hbm_evictions(&mut self) {
        let Some(log) = &self.hbm_evictions else {
            return;
        };
        for eviction in log.borrow_mut().drain(..) {
            let tiers = &mut self.partitions[usize::from(eviction.partition)].tiers;
            if eviction.state_only {
                tiers.store_evicted_state(eviction.session_id, eviction.tokens);
            } else {
                tiers.store_evicted(eviction.session_id, eviction.tokens);
            }
        }
    }

    /// Start a read only while its tier would begin it within `ms`; past
    /// that, `policy` decides. `PassOver` leaves the request queued, holding
    /// nothing, and admission goes on to the requests behind it; `Block` stops
    /// admission at it, as a read that cannot fit does. A long queue of reads
    /// otherwise holds their HBM blocks for its whole wait; a blocked head
    /// also keeps the HBM and DRAM hits behind it out. 0 leaves reads
    /// unbounded, as vLLM does. Not vLLM.
    pub(crate) fn with_max_read_wait_ms(
        mut self,
        ms: f64,
        policy: PrefixTierReadWaitPolicy,
    ) -> Self {
        assert!(ms >= 0.0, "prefix_tier_max_read_wait_ms must be >= 0");
        self.max_read_wait = (ms > 0.0).then(|| Time::from_ms(ms));
        self.read_wait_policy = policy;
        self
    }

    /// What a cache-aware order ranks the request by: the longer of HBM's
    /// `hbm_tokens` and what a tier would read back.
    pub(crate) fn rank_tokens(
        &mut self,
        partition: u16,
        session_input: SessionInput,
        hbm_tokens: u32,
    ) -> u32 {
        self.take_hbm_evictions();
        let SessionInput::Session {
            session_id,
            declared_prefix_tokens,
            ..
        } = session_input
        else {
            return hbm_tokens;
        };
        self.partitions[usize::from(partition)]
            .lookup(session_id, declared_prefix_tokens, self.warm_start)
            .map_or(hbm_tokens, |(hit, _)| hit.tokens.max(hbm_tokens))
    }

    /// Decide what happens to the request at the head of `partition`'s queue:
    /// put a landed context back into HBM, or start a read when a tier holds
    /// more than HBM. A read starts only if the whole request fits, and holds
    /// its footprint in HBM until it lands.
    pub(crate) fn at_head<K: PrefixKv>(
        &mut self,
        kv_store: &mut K,
        partition: u16,
        candidate: AdmissionCandidate,
        now: Time,
    ) -> AtHead {
        self.set_clock(now);
        self.take_hbm_evictions();
        let SessionInput::Session {
            session_id,
            declared_prefix_tokens,
            ..
        } = candidate.session_input
        else {
            return AtHead::Admit;
        };
        if let Some(fetch) = self.landed.remove(&candidate.request_id) {
            // Its blocks were held since the read started: the context fills
            // them and the admission that follows takes them over.
            kv_store.release_read_hold(candidate.request_id);
            // The read is a prefix of the session's latest context, as HBM's
            // entry is.
            let tokens = u64::from(fetch.read.hit.tokens);
            kv_store.restore_prefix(
                candidate.request_id,
                partition,
                session_id,
                tokens,
                tokens,
                now,
            );
            self.restored.insert(candidate.request_id, fetch);
            return AtHead::Admit;
        }
        let hbm_tokens = kv_store
            .preview_prefill_context(
                partition,
                candidate.fresh_prompt_tokens,
                candidate.session_input,
            )
            .resident_prefix_tokens();
        let reread = match self.restored.get(&candidate.request_id) {
            Some(fetch) if hbm_tokens >= fetch.read.hit.tokens => return AtHead::Admit,
            Some(_) => true,
            None => false,
        };
        let tiers = &mut self.partitions[usize::from(partition)];
        let Some((hit, seeded)) = tiers
            .lookup(session_id, declared_prefix_tokens, self.warm_start)
            .filter(|(hit, _)| hit.tokens > hbm_tokens)
        else {
            return AtHead::Admit;
        };
        // The whole request must fit before its load starts; its blocks stay
        // held until the read lands (the pinned fork reserves every block the
        // request still needs, `_inflight_prefill_reserved_blocks`).
        let post_prefill_context_tokens = kv_store
            .preview_prefill_context(
                partition,
                candidate.fresh_prompt_tokens,
                candidate.session_input,
            )
            .post_prefill_context_tokens();
        let footprint = kv_store.footprint(
            candidate.request_id,
            post_prefill_context_tokens,
            candidate.remaining_output_tokens,
        );
        if self
            .max_read_wait
            .is_some_and(|max| tiers.queue_wait(hit, now) > max)
        {
            return match self.read_wait_policy {
                PrefixTierReadWaitPolicy::PassOver => AtHead::PassedOver,
                PrefixTierReadWaitPolicy::Block => AtHead::Blocked,
            };
        }
        if !kv_store.fits(partition, &footprint) {
            return AtHead::Blocked;
        }
        kv_store.hold_for_read(candidate.request_id, partition, &footprint, now);
        if reread {
            // Evicted after admission first reached it unadmitted.
            tiers.rereads += 1;
            self.restored.remove(&candidate.request_id);
        }
        let ready = tiers.read(session_id, hit, seeded, hbm_tokens, now);
        self.reads.push(Reverse((ready, candidate.request_id)));
        self.reading.insert(
            candidate.request_id,
            Fetch {
                candidate,
                partition,
                read: Read {
                    hit,
                    hbm_tokens,
                    started: now,
                    ready,
                },
            },
        );
        AtHead::Read
    }

    /// Land every read done by `now`: its request, still holding its blocks,
    /// waits for admission in its partition's landed queue. Returns the latest
    /// landing time, if any landed.
    pub(crate) fn land<K: PrefixKv>(&mut self, kv_store: &mut K, now: Time) -> Option<Time> {
        self.set_clock(now);
        for request in self.cancelled.drain(..) {
            kv_store.release_read_hold(request);
        }
        let mut latest = None;
        while let Some(&Reverse((ready, request))) = self.reads.peek() {
            if ready > now {
                break;
            }
            self.reads.pop();
            let Some(fetch) = self.reading.remove(&request) else {
                continue;
            };
            self.landed_order[usize::from(fetch.partition)].push_back(request);
            self.landed.insert(request, fetch);
            latest = Some(ready);
        }
        latest
    }

    /// Log an admitted session request with `resident_tokens` of its prefix
    /// found in HBM.
    pub(crate) fn admitted(
        &mut self,
        partition: u16,
        candidate: AdmissionCandidate,
        resident_tokens: u32,
        now: Time,
    ) {
        self.set_clock(now);
        let SessionInput::Session {
            session_id,
            declared_prefix_tokens,
            ..
        } = candidate.session_input
        else {
            return;
        };
        let read = self
            .restored
            .remove(&candidate.request_id)
            .map(|fetch| fetch.read);
        self.take_hbm_evictions();
        self.partitions[usize::from(partition)]
            .tiers
            .drop_stale(session_id, u64::from(declared_prefix_tokens));
        self.partitions[usize::from(partition)].record_admission(
            candidate.request_id,
            session_id,
            Admitted {
                declared_tokens: declared_prefix_tokens,
                resident_tokens,
                read,
                at: now,
            },
        );
    }

    /// Write a finished session context through to its partition's tiers.
    pub(crate) fn store(&mut self, partition: u16, session_id: u32, round: RoundContext) {
        self.take_hbm_evictions();
        self.partitions[usize::from(partition)].store(session_id, round);
    }

    pub(crate) fn next_landing(&self) -> Option<Time> {
        self.reads.peek().map(|Reverse((ready, _))| *ready)
    }

    /// The landed request `partition` admits next, ahead of its queue.
    pub(crate) fn peek_landed(&self, partition: u16) -> Option<AdmissionCandidate> {
        let request = self.landed_order[usize::from(partition)].front()?;
        self.landed
            .get(request)
            .or_else(|| self.restored.get(request))
            .map(|fetch| fetch.candidate)
    }

    /// Take the request [`Self::peek_landed`] returned out of the landed queue.
    pub(crate) fn pop_landed(&mut self, partition: u16) {
        self.landed_order[usize::from(partition)].pop_front();
    }

    /// Requests out of their queue for a read: reading, or landed and not yet
    /// admitted.
    pub(crate) fn reading(&self) -> u32 {
        (self.reading.len() + self.landed_order.iter().map(VecDeque::len).sum::<usize>()) as u32
    }

    /// Forget a cancelled request; true if it was out of its queue for a read.
    pub(crate) fn cancel(&mut self, request: RequestId) -> bool {
        let mut out_of_queue = false;
        for order in &mut self.landed_order {
            if let Some(at) = order.iter().position(|&landed| landed == request) {
                order.remove(at);
                out_of_queue = true;
            }
        }
        self.restored.remove(&request);
        if self.landed.remove(&request).is_some() || self.reading.remove(&request).is_some() {
            self.cancelled.push(request);
            out_of_queue = true;
        }
        out_of_queue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiers() -> SessionPrefixTiers {
        SessionPrefixTiers::new(
            &[
                PrefixTierSpec {
                    name: "dram",
                    capacity_gb_per_gpu: 1e-6,
                    read_gb_per_s_per_gpu: 1e-4,
                    writes: crate::worker::kv::PrefixTierWrites::EveryContext,
                },
                PrefixTierSpec {
                    name: "ssd",
                    capacity_gb_per_gpu: 1e-5,
                    read_gb_per_s_per_gpu: 1e-5,
                    writes: crate::worker::kv::PrefixTierWrites::EveryContext,
                },
            ],
            1,
            0,
            ResumePoint::AnyToken,
            1,
            None,
            "test",
        )
    }

    #[test]
    fn warm_start_seeds_only_a_context_the_run_never_stored() {
        let mut t = tiers();
        // A pre-run context: nothing stored, so it reads from the slowest tier.
        let (hit, seeded) = t.lookup(7, 500, true).unwrap();
        assert_eq!((hit.tier, hit.tokens, seeded), (1, 500, true));
        assert_eq!(t.lookup(7, 500, false), None);
        // The next round declares one token more than the stored context (the
        // last output's KV): an ordinary hit from the fastest tier, not a seed.
        t.store(
            7,
            RoundContext {
                tokens: 554,
                prompt_end: 554,
                shared: 0,
            },
        );
        let (hit, seeded) = t.lookup(7, 555, true).unwrap();
        assert_eq!((hit.tier, hit.tokens, seeded), (0, 554, false));
        // A placeholder round's one-token context does not cover a joining
        // round's context.
        t.store(
            8,
            RoundContext {
                tokens: 1,
                prompt_end: 1,
                shared: 0,
            },
        );
        assert!(t.lookup(8, 300, true).unwrap().1);
    }
}
