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
//! A read moves the KV HBM lacks and, of a context keeping `k` round-end
//! states, only those that survive the round it serves: the state it resumes
//! from and the latest earlier ones still among the session's latest `k` once
//! the round adds its own (`kv/shared/prefix_tiers.rs`). Its time, its holds
//! and `loaded_states` count only what moves.
//!
//! With staged loading (not vLLM), loads go pending -> loading -> ready ->
//! batch instead. Before each admission pass, the overdue and pending queues
//! are walked in admission order and every request whose context a tier
//! holds beyond HBM starts its read while its loading area has room; requests
//! needing no read stay where they are. A DRAM read lands in HBM's loading
//! area (kept free of admission and the retained-prefix tier); an SSD read
//! lands in DRAM's loading area (carved out of the DRAM tier) and moves on to
//! HBM's, on DRAM's channel, when that has room. Each loading queue stops at
//! the first read that does not fit its area (head of line: a later read
//! never jumps it; a read larger than the whole area starts when the area is
//! empty), so loads keep the admission order and never wait on compute
//! admission. Admission takes only ready requests: landed ones, then those
//! HBM already holds; a request still needing a read is passed over, and one
//! whose landed context HBM evicted before admission goes back to pending and
//! is read again (`rereads`). Two rules size and release the areas:
//!
//! - held ([`PrefixFetch::with_loading_area`]): each area is a fixed fraction
//!   of its tier, and a read holds it from its start until its request is
//!   admitted. Large reads that land but wait for admission fill it.
//! - bounded ([`PrefixFetch::with_bounded_loading`]): each area bounds the
//!   reads in flight into it: what the channel feeding it moves in one batch
//!   period (HBM's: the fastest tier's channel, DRAM's: SSD's), the period
//!   being one `max_batch_tokens` prefill microbatch's stage time, so the
//!   channel never idles for want of area while a queue of at most one
//!   period builds behind it (Little's law). DRAM's area is carved out of the
//!   DRAM tier, where SSD data lands. HBM's reserves nothing: a read holds its
//!   request's whole footprint from its start (as an unstaged read does),
//!   counted against HBM like a running request's (retained prefixes make
//!   room), and starts only if it fits; so whatever lands can be admitted
//!   once its hold gives way to its reservation, and landed data can never
//!   deadlock admission. A read gives its area back when it lands (an SSD
//!   read's DRAM area when its hop lands); its data stays held in HBM until
//!   admission. Two bounds keep landed data from flushing HBM's cache: a new
//!   read starts only while the landed, unadmitted requests' prefill work
//!   (new tokens plus any declared prefix past the hit) is below
//!   `pp_size x max_batch_tokens`, what fills the pipeline, and while their
//!   data plus its own fits HBM's in-flight area (as for the area itself, a
//!   read larger than that starts when nothing has landed). Landed data so
//!   stays below two areas: one landed, one still in flight. A landing starts
//!   a loading pass at once.
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
    EndState, HbmEvictionLog, PrefixKv, PrefixTierHit, PrefixTierSpec, PrefixTiers, ResumePoint,
    RoundContext, RoundState, StateRetention, StateTag, WRITE_WINDOW,
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
    /// The staged-loading rule and the sizes it derived, for the summary.
    loading: Option<serde_json::Value>,
}

impl SessionPrefixTiers {
    pub(crate) fn new(
        specs: &[PrefixTierSpec],
        kv_bytes_per_token: u64,
        state_tokens: u64,
        resume: ResumePoint,
        retention: StateRetention,
        states_added_per_round: u32,
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
            .with_round_end_states(retention, states_added_per_round);
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
            loading: None,
        }
    }

    /// The tiers store each state as `stored` tokens, compressed
    /// ([`PrefixTiers::set_stored_state_tokens`]).
    pub(crate) fn with_stored_state_tokens(mut self, stored: u64) -> Self {
        self.tiers.set_stored_state_tokens(stored);
        self
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

    /// What a read of `hit` moves on its tier's channel, in tier tokens: the
    /// KV HBM's `hbm_tokens` lack and the states that survive the round
    /// ([`PrefixTiers::read_states`]; a pre-run context has one).
    fn read_charge(
        &self,
        session_id: u32,
        hit: PrefixTierHit,
        seeded: bool,
        hbm_tokens: u32,
    ) -> u64 {
        if seeded {
            u64::from(hit.tokens - hbm_tokens) + self.tiers.read_state_tokens(hit.tier)
        } else {
            self.tiers.read_charge(session_id, hit, hbm_tokens)
        }
    }

    /// What that read lands in HBM, its states decoded.
    fn landed_charge(
        &self,
        session_id: u32,
        hit: PrefixTierHit,
        seeded: bool,
        hbm_tokens: u32,
    ) -> u64 {
        if seeded {
            u64::from(hit.tokens - hbm_tokens) + self.tiers.landed_state_tokens()
        } else {
            self.tiers.landed_charge(session_id, hit, hbm_tokens)
        }
    }

    /// How many round-end states a read of `hit` moves besides the one at
    /// the hit (a pre-run context has none).
    fn earlier_read_states(&self, session_id: u32, hit: PrefixTierHit, seeded: bool) -> usize {
        if seeded {
            0
        } else {
            self.tiers.read_states(session_id, hit).len()
        }
    }

    /// Start a read of `hit`, of what HBM's `hbm_tokens` lack; returns when it
    /// lands and the states it moves below the hit.
    fn read(
        &mut self,
        session_id: u32,
        hit: PrefixTierHit,
        seeded: bool,
        hbm_tokens: u32,
        now: Time,
    ) -> (Time, Vec<RoundState>, StateTag) {
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
                    "loaded_states": counters.loaded_states,
                    "released_entries": counters.released_entries,
                    "earlier_states": self.tiers.earlier_states(tier),
                    "write_window_ms": WRITE_WINDOW.as_ms(),
                    "window_writes": self.tiers.window_writes(tier),
                    "window_state_writes": self.tiers.window_states(tier),
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
            "loading": self.loading,
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

/// Where a staged read is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Into HBM's loading area (unstaged reads too).
    Hbm,
    /// From SSD into DRAM's loading area.
    Ssd,
    /// From DRAM's loading area on to HBM's.
    Hop,
}

#[derive(Clone, Debug)]
struct Fetch {
    candidate: AdmissionCandidate,
    partition: u16,
    read: Read,
    phase: Phase,
    /// Tokens (KV and its decoded states) the read holds in HBM's loading
    /// area, and its hop moves on DRAM's channel.
    charge: u64,
    /// Tokens (KV and its stored states) an SSD read holds in DRAM's.
    dram_charge: u64,
    /// The round-end states below the hit the read moves into HBM.
    states: Vec<RoundState>,
    /// The state at the hit.
    hit_tag: StateTag,
}

impl Fetch {
    /// Tokens its prefill computes: the prompt's new tokens and any declared
    /// prefix past the hit.
    fn prefill_work(&self) -> u64 {
        let declared = match self.candidate.session_input {
            SessionInput::Session {
                declared_prefix_tokens,
                ..
            } => declared_prefix_tokens,
            _ => 0,
        };
        u64::from(self.candidate.fresh_prompt_tokens)
            + u64::from(declared.saturating_sub(self.read.hit.tokens))
    }
}

/// When a staged read gives its HBM loading area back (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StagedRule {
    /// At its request's admission.
    Held,
    /// When it lands; new reads start only while the landed requests'
    /// prefill work is below `ready_bound` tokens.
    Bounded { ready_bound: u64 },
}

/// Loading areas of the staged mode, per partition, in tier tokens.
struct Staged {
    rule: StagedRule,
    hbm_area: Vec<u64>,
    hbm_used: Vec<u64>,
    dram_area: Vec<u64>,
    dram_used: Vec<u64>,
    /// SSD reads landed in DRAM's area, waiting for room in HBM's.
    waiting_hop: Vec<VecDeque<RequestId>>,
}

/// Context tokens a read's HBM hold covers: the `kv_tokens` it lands and one
/// retained state's charge for each of the `earlier` round-end states it
/// moves beside the resume state (the footprint carries that one), so the
/// context goes back into HBM with all of them.
fn read_hold_tokens<K: PrefixKv>(kv_store: &K, kv_tokens: u32, earlier: usize) -> u32 {
    kv_tokens + (earlier as u64 * kv_store.retained_state_tokens()) as u32
}

/// What a staged read holds in HBM from its start. Held: the context it lands
/// ([`read_hold_tokens`]), inside the reserved area. Bounded: its request's
/// whole footprint (context after prefill, output, every state it moves), as
/// an unstaged read does, so any landed request can be admitted once its
/// hold gives way to its reservation: the pipeline cannot deadlock on landed
/// data.
fn hold_footprint<K: PrefixKv>(
    kv_store: &K,
    rule: StagedRule,
    partition: u16,
    candidate: AdmissionCandidate,
    landed_kv: u32,
    earlier: usize,
) -> K::Footprint {
    match rule {
        StagedRule::Held => kv_store.footprint(
            candidate.request_id,
            read_hold_tokens(kv_store, landed_kv, earlier),
            0,
        ),
        StagedRule::Bounded { .. } => {
            let context = kv_store
                .preview_prefill_context(
                    partition,
                    candidate.fresh_prompt_tokens,
                    candidate.session_input,
                )
                .post_prefill_context_tokens();
            kv_store.footprint(
                candidate.request_id,
                read_hold_tokens(kv_store, context, earlier),
                candidate.remaining_output_tokens,
            )
        }
    }
}

/// Whether a read of `charge` may start in an area holding `used` of `area`:
/// it fits, or the area is empty (a read larger than the whole area).
fn area_fits(used: u64, area: u64, charge: u64) -> bool {
    used == 0 || used + charge <= area
}

/// Which loading queues stopped this pass (staged mode).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LoadGate {
    hbm_full: bool,
    dram_full: bool,
    /// Bounded rule: tier tokens landed and waiting for admission.
    landed: u64,
}

impl LoadGate {
    /// Both queues stopped: nothing more can start this pass.
    pub(crate) fn closed(&self) -> bool {
        self.hbm_full && self.dram_full
    }
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
    /// Staged loading ([`Self::with_loading_area`]).
    staged: Option<Staged>,
    /// The bounded rule's batch period, until its first loading pass
    /// ([`Self::with_bounded_loading`]).
    batch_period: Option<Box<dyn FnOnce() -> Time>>,
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
            staged: None,
            batch_period: None,
        }
    }

    /// Stage loads through loading areas held until admission (module docs):
    /// `hbm_area` tokens of each partition's HBM, and `fraction` of a DRAM
    /// tier with an SSD tier behind it. The KV store must keep `hbm_area`
    /// free ([`PrefixKv::set_loading_area`]).
    pub(crate) fn with_loading_area(mut self, hbm_area: u64, fraction: f64) -> Self {
        let n = self.partitions.len();
        let dram_area: Vec<u64> = self
            .partitions
            .iter_mut()
            .map(|p| {
                if p.tiers.len() == 2 {
                    p.tiers.carve_loading_area(0, fraction)
                } else {
                    0
                }
            })
            .collect();
        for (p, dram) in self.partitions.iter_mut().zip(&dram_area) {
            p.loading = Some(serde_json::json!({
                "rule": "held",
                "fraction": fraction,
                "hbm_area_tokens": hbm_area,
                "dram_area_tokens": dram,
            }));
        }
        self.staged = Some(Staged {
            rule: StagedRule::Held,
            hbm_area: vec![hbm_area; n],
            hbm_used: vec![0; n],
            dram_area,
            dram_used: vec![0; n],
            waiting_hop: vec![VecDeque::new(); n],
        });
        self
    }

    /// Stage loads under the bounded rule (module docs): each partition's HBM
    /// in-flight area, a bound on the reads in progress into HBM, is what its
    /// fastest tier's channel reads in one batch period; a DRAM tier with
    /// SSD behind it gives up what the SSD channel reads in that time as
    /// DRAM's landing area. New reads start only while the landed requests'
    /// prefill work is below `ready_bound` tokens. `batch_period` evaluates
    /// the cost model, which is ready only once the run starts, so the areas
    /// are sized at the first loading pass.
    pub(crate) fn with_bounded_loading(
        mut self,
        batch_period: impl FnOnce() -> Time + 'static,
        ready_bound: u64,
    ) -> Self {
        let n = self.partitions.len();
        self.staged = Some(Staged {
            rule: StagedRule::Bounded { ready_bound },
            hbm_area: vec![0; n],
            hbm_used: vec![0; n],
            dram_area: vec![0; n],
            dram_used: vec![0; n],
            waiting_hop: vec![VecDeque::new(); n],
        });
        self.batch_period = Some(Box::new(batch_period));
        self
    }

    /// Size the bounded rule's areas from the batch period, once.
    fn size_bounded_areas(&mut self) {
        let Some(batch_period) = self.batch_period.take() else {
            return;
        };
        let batch_period = batch_period();
        let staged = self.staged.as_mut().expect("bounded staging");
        let StagedRule::Bounded { ready_bound } = staged.rule else {
            unreachable!("a batch period belongs to the bounded rule");
        };
        for (i, p) in self.partitions.iter_mut().enumerate() {
            let hbm = p.tiers.read_tokens_in(0, batch_period);
            let dram = if p.tiers.len() == 2 {
                let tokens = p.tiers.read_tokens_in(1, batch_period);
                p.tiers.carve_tokens(0, tokens)
            } else {
                0
            };
            tracing::info!(
                partition = i,
                batch_period_ms = batch_period.as_ms(),
                hbm_inflight_tier_tokens = hbm,
                dram_inflight_tier_tokens = dram,
                ready_bound,
                "bounded staged loading: in-flight areas and ready bound"
            );
            p.loading = Some(serde_json::json!({
                "rule": "bounded",
                "batch_period_ms": batch_period.as_ms(),
                "hbm_area_tokens": hbm,
                "dram_area_tokens": dram,
                "ready_bound_tokens": ready_bound,
            }));
            staged.hbm_area[i] = hbm;
            staged.dram_area[i] = dram;
        }
    }

    /// Bounded staging: a read gives its HBM in-flight area back when it
    /// lands, and a loading pass may start the reads behind it then.
    pub(crate) fn releases_at_landing(&self) -> bool {
        self.staged
            .as_ref()
            .is_some_and(|staged| matches!(staged.rule, StagedRule::Bounded { .. }))
    }

    /// The gate a loading pass of `partition` starts with: closed when the
    /// bounded rule's landed work already fills the pipeline.
    pub(crate) fn load_gate(&mut self, partition: u16) -> LoadGate {
        self.size_bounded_areas();
        let full = match self.staged.as_ref().map(|staged| staged.rule) {
            Some(StagedRule::Bounded { ready_bound }) => {
                self.ready_work(partition) >= ready_bound
            }
            _ => false,
        };
        LoadGate {
            hbm_full: full,
            dram_full: full,
            landed: self.landed_charge(partition),
        }
    }

    /// Tier tokens of `partition`'s landed requests not yet admitted.
    fn landed_charge(&self, partition: u16) -> u64 {
        self.landed_order[usize::from(partition)]
            .iter()
            .filter_map(|request| self.landed.get(request).or_else(|| self.restored.get(request)))
            .map(|fetch| fetch.charge)
            .sum()
    }

    /// Prefill work of `partition`'s landed requests not yet admitted.
    fn ready_work(&self, partition: u16) -> u64 {
        self.landed_order[usize::from(partition)]
            .iter()
            .filter_map(|request| self.landed.get(request).or_else(|| self.restored.get(request)))
            .map(Fetch::prefill_work)
            .sum()
    }

    pub(crate) fn is_staged(&self) -> bool {
        self.staged.is_some()
    }

    /// Staged mode: start the read of a pending request whose context a tier
    /// holds beyond HBM, if its loading queue has not stopped and its area has
    /// room; true when it started (the request leaves its queue). A request
    /// needing no read, or whose queue stopped, stays.
    pub(crate) fn stage<K: PrefixKv>(
        &mut self,
        kv_store: &mut K,
        partition: u16,
        candidate: AdmissionCandidate,
        gate: &mut LoadGate,
        now: Time,
    ) -> bool {
        self.set_clock(now);
        self.take_hbm_evictions();
        let SessionInput::Session {
            session_id,
            declared_prefix_tokens,
            ..
        } = candidate.session_input
        else {
            return false;
        };
        let p = usize::from(partition);
        let hbm_tokens = kv_store
            .preview_prefill_context(
                partition,
                candidate.fresh_prompt_tokens,
                candidate.session_input,
            )
            .resident_prefix_tokens();
        let tiers = &mut self.partitions[p];
        let Some((hit, seeded)) = tiers
            .lookup(session_id, declared_prefix_tokens, self.warm_start)
            .filter(|(hit, _)| hit.tokens > hbm_tokens)
        else {
            return false;
        };
        let charge = tiers.landed_charge(session_id, hit, seeded, hbm_tokens);
        let dram_charge = tiers.read_charge(session_id, hit, seeded, hbm_tokens);
        let earlier = tiers.earlier_read_states(session_id, hit, seeded);
        let staged = self.staged.as_mut().expect("staged mode");
        if staged.rule != StagedRule::Held && !area_fits(gate.landed, staged.hbm_area[p], charge) {
            // Landed data waiting for admission already fills an in-flight
            // area's worth of HBM.
            gate.hbm_full = true;
            gate.dram_full = true;
            return false;
        }
        let through_dram = staged.dram_area[p] > 0 && hit.tier == 1;
        let phase = if through_dram {
            if gate.dram_full || !area_fits(staged.dram_used[p], staged.dram_area[p], dram_charge) {
                gate.dram_full = true;
                return false;
            }
            staged.dram_used[p] += dram_charge;
            Phase::Ssd
        } else {
            let footprint = hold_footprint(
                kv_store,
                staged.rule,
                partition,
                candidate,
                hit.tokens - hbm_tokens,
                earlier,
            );
            if gate.hbm_full
                || !area_fits(staged.hbm_used[p], staged.hbm_area[p], charge)
                || (staged.rule != StagedRule::Held && !kv_store.fits(partition, &footprint))
            {
                gate.hbm_full = true;
                return false;
            }
            staged.hbm_used[p] += charge;
            kv_store.hold_for_read(candidate.request_id, partition, &footprint, now);
            Phase::Hbm
        };
        self.restored.remove(&candidate.request_id);
        let (ready, states, hit_tag) = tiers.read(session_id, hit, seeded, hbm_tokens, now);
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
                phase,
                charge,
                dram_charge,
                states,
                hit_tag,
            },
        );
        true
    }

    /// Staged mode: move SSD reads waiting in DRAM's loading area on to HBM's,
    /// in landing order, while it has room.
    fn start_hops<K: PrefixKv>(&mut self, kv_store: &mut K, now: Time) {
        let Some(staged) = self.staged.as_mut() else {
            return;
        };
        for p in 0..self.partitions.len() {
            while let Some(&request) = staged.waiting_hop[p].front() {
                let fetch = self.reading.get_mut(&request).expect("a waiting hop is reading");
                let footprint = hold_footprint(
                    kv_store,
                    staged.rule,
                    p as u16,
                    fetch.candidate,
                    fetch.read.hit.tokens - fetch.read.hbm_tokens,
                    fetch.states.len(),
                );
                if !area_fits(staged.hbm_used[p], staged.hbm_area[p], fetch.charge)
                    || (staged.rule != StagedRule::Held && !kv_store.fits(p as u16, &footprint))
                {
                    break;
                }
                staged.waiting_hop[p].pop_front();
                staged.hbm_used[p] += fetch.charge;
                kv_store.hold_for_read(request, p as u16, &footprint, now);
                let ready = self.partitions[p].tiers.transfer(0, fetch.charge, now);
                fetch.phase = Phase::Hop;
                fetch.read.ready = ready;
                self.reads.push(Reverse((ready, request)));
            }
        }
    }

    /// Give back what `fetch` holds in the loading areas.
    fn release_area(&mut self, fetch: &Fetch, dram: bool, hbm: bool) {
        let Some(staged) = self.staged.as_mut() else {
            return;
        };
        let p = usize::from(fetch.partition);
        if dram {
            staged.dram_used[p] -= fetch.dram_charge;
        }
        if hbm {
            staged.hbm_used[p] -= fetch.charge;
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
            self.partitions[usize::from(eviction.partition)].tiers.store_evicted(
                eviction.session_id,
                eviction.tokens,
                eviction.end,
                &eviction.earlier_states,
            );
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
            let held_area = !self.releases_at_landing();
            self.release_area(&fetch, false, held_area);
            // The read is a prefix of the session's latest context, as HBM's
            // entry is; its states come with it.
            let tokens = u64::from(fetch.read.hit.tokens);
            kv_store.restore_prefix(
                candidate.request_id,
                partition,
                session_id,
                tokens,
                tokens,
                &fetch.states,
                EndState::Read(fetch.hit_tag),
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
        if self.staged.is_some() {
            // Not ready: the loading pass reads it when its area has room.
            if reread {
                tiers.rereads += 1;
                self.restored.remove(&candidate.request_id);
            }
            return AtHead::PassedOver;
        }
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
        let earlier = tiers.earlier_read_states(session_id, hit, seeded);
        let footprint = kv_store.footprint(
            candidate.request_id,
            read_hold_tokens(kv_store, post_prefill_context_tokens, earlier),
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
        let (ready, states, hit_tag) = tiers.read(session_id, hit, seeded, hbm_tokens, now);
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
                phase: Phase::Hbm,
                charge: 0,
                dram_charge: 0,
                states,
                hit_tag,
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
            let Some(fetch) = self.reading.get(&request).cloned() else {
                continue;
            };
            if fetch.read.ready != ready {
                continue; // a hop re-queued it at a later time
            }
            match fetch.phase {
                Phase::Ssd => {
                    let staged = self.staged.as_mut().expect("staged mode");
                    staged.waiting_hop[usize::from(fetch.partition)].push_back(request);
                    continue;
                }
                Phase::Hop => self.release_area(&fetch, true, false),
                Phase::Hbm => {}
            }
            if self.releases_at_landing() {
                // Its data stays held in HBM until admission.
                self.release_area(&fetch, false, true);
            }
            self.reading.remove(&request);
            self.landed_order[usize::from(fetch.partition)].push_back(request);
            self.landed.insert(request, fetch);
            latest = Some(ready);
        }
        self.start_hops(kv_store, now);
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
        if let Some(staged) = self.staged.as_mut() {
            for waiting in &mut staged.waiting_hop {
                waiting.retain(|&r| r != request);
            }
        }
        if let Some(fetch) = self.landed.remove(&request) {
            let held_area = !self.releases_at_landing();
            self.release_area(&fetch, false, held_area);
            self.cancelled.push(request);
            out_of_queue = true;
        } else if let Some(fetch) = self.reading.remove(&request) {
            match fetch.phase {
                Phase::Hbm => self.release_area(&fetch, false, true),
                Phase::Ssd => self.release_area(&fetch, true, false),
                Phase::Hop => self.release_area(&fetch, true, true),
            }
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
            StateRetention::Latest(1),
            1,
            None,
            "test",
        )
    }

    use crate::worker::kv::{FullAttnKv, PrefixCachePolicy, PrefixCacheTokenConfig};

    /// One DRAM tier of 1M tokens read at 1000 tokens/ms (1000 B/token, 1
    /// GB/s), and an empty 1M-token HBM; sessions 1..=4 each hold a context
    /// of `tokens[i]` in DRAM.
    fn fetch_with(tokens: &[u64]) -> (PrefixFetch, FullAttnKv) {
        let tiers = SessionPrefixTiers::new(
            &[PrefixTierSpec {
                name: "dram",
                capacity_gb_per_gpu: 1.0,
                read_gb_per_s_per_gpu: 1.0,
                writes: crate::worker::kv::PrefixTierWrites::EveryContext,
            }],
            1000,
            0,
            ResumePoint::AnyToken,
            StateRetention::Latest(1),
            1,
            None,
            "test",
        );
        let mut fetch = PrefixFetch::new(vec![tiers], false);
        for (i, &context) in tokens.iter().enumerate() {
            fetch.store(
                0,
                i as u32 + 1,
                RoundContext {
                    tokens: context,
                    prompt_end: context,
                    shared: 0,
                },
            );
        }
        let kv = FullAttnKv::with_prefix_cache(
            1,
            1_000_000,
            PrefixCacheTokenConfig {
                max_retained_tokens: 1_000_000,
                policy: PrefixCachePolicy::Lru,
            },
            None,
            None,
        );
        (fetch, kv)
    }

    /// Request `id` of session `id`, declaring its whole stored context.
    fn candidate(id: u64, declared: u32, fresh: u32) -> AdmissionCandidate {
        AdmissionCandidate {
            request_id: RequestId(id as u32),
            enqueue_sequence: id,
            fresh_prompt_tokens: fresh,
            remaining_output_tokens: 1,
            session_input: SessionInput::Session {
                session_id: id as u32,
                session_start_time: Time::ZERO,
                declared_prefix_tokens: declared,
            },
            conversation_start_time: Time::ZERO,
            resident_prefix_tokens: 0,
            retracted: false,
        }
    }

    fn hbm_used(fetch: &PrefixFetch) -> u64 {
        fetch.staged.as_ref().unwrap().hbm_used[0]
    }

    /// Admit the landed request at the head of the landed queue.
    fn admit_landed(fetch: &mut PrefixFetch, kv: &mut FullAttnKv, now: Time) {
        let c = fetch.peek_landed(0).expect("a landed request");
        assert_eq!(fetch.at_head(kv, 0, c, now), AtHead::Admit);
        fetch.pop_landed(0);
        fetch.admitted(0, c, 0, now);
    }

    #[test]
    fn bounded_areas_are_what_each_channel_reads_in_a_batch_period() {
        let (fetch, _) = fetch_with(&[]);
        let mut fetch = fetch.with_bounded_loading(|| Time::from_ms(10.0), 5_000);
        fetch.load_gate(0);
        assert_eq!(fetch.staged.as_ref().unwrap().hbm_area[0], 10_000);
        assert!(fetch.releases_at_landing());
    }

    #[test]
    fn a_bounded_read_gives_its_area_back_when_it_lands() {
        for bounded in [false, true] {
            let (fetch, mut kv) = fetch_with(&[6_000, 4_000, 4_000]);
            let mut fetch = if bounded {
                fetch.with_bounded_loading(|| Time::from_ms(10.0), 1_000_000)
            } else {
                fetch.with_loading_area(10_000, 0.0)
            };
            let mut gate = fetch.load_gate(0);
            assert!(fetch.stage(&mut kv, 0, candidate(1, 6_000, 10), &mut gate, Time::ZERO));
            assert!(fetch.stage(&mut kv, 0, candidate(2, 4_000, 10), &mut gate, Time::ZERO));
            assert!(!fetch.stage(&mut kv, 0, candidate(3, 4_000, 10), &mut gate, Time::ZERO));
            // 6000 tokens at 1000 tokens/ms land at 6 ms (2's at 10 ms).
            let at = Time::from_ms(6.0);
            assert_eq!(fetch.land(&mut kv, at), Some(at));
            assert_eq!(hbm_used(&fetch), if bounded { 4_000 } else { 10_000 });
            let mut gate = fetch.load_gate(0);
            assert_eq!(
                fetch.stage(&mut kv, 0, candidate(3, 4_000, 10), &mut gate, at),
                bounded
            );
            // The held rule gives it back at admission.
            admit_landed(&mut fetch, &mut kv, at);
            assert_eq!(hbm_used(&fetch), if bounded { 8_000 } else { 4_000 });
        }
    }

    #[test]
    fn landed_data_past_an_area_stops_new_reads() {
        let (fetch, mut kv) = fetch_with(&[6_000, 5_000]);
        let mut fetch = fetch.with_bounded_loading(|| Time::from_ms(10.0), 1_000_000);
        let mut gate = fetch.load_gate(0);
        assert!(fetch.stage(&mut kv, 0, candidate(1, 6_000, 10), &mut gate, Time::ZERO));
        let at = Time::from_ms(6.0);
        fetch.land(&mut kv, at);
        // Nothing is in flight, but 6000 landed + 5000 exceed the 10000 area.
        let mut gate = fetch.load_gate(0);
        assert!(!fetch.stage(&mut kv, 0, candidate(2, 5_000, 10), &mut gate, at));
        admit_landed(&mut fetch, &mut kv, at);
        let mut gate = fetch.load_gate(0);
        assert!(fetch.stage(&mut kv, 0, candidate(2, 5_000, 10), &mut gate, at));
    }

    #[test]
    fn landed_work_past_the_ready_bound_stops_new_reads() {
        let (fetch, mut kv) = fetch_with(&[1_000, 1_000]);
        let mut fetch = fetch.with_bounded_loading(|| Time::from_ms(10.0), 5_000);
        let mut gate = fetch.load_gate(0);
        // Session 1's round computes 5000 new tokens.
        assert!(fetch.stage(&mut kv, 0, candidate(1, 1_000, 5_000), &mut gate, Time::ZERO));
        let at = Time::from_ms(1.0);
        fetch.land(&mut kv, at);
        let mut gate = fetch.load_gate(0);
        assert!(gate.closed());
        assert!(!fetch.stage(&mut kv, 0, candidate(2, 1_000, 10), &mut gate, at));
        // Once admitted, the pipeline needs more.
        admit_landed(&mut fetch, &mut kv, at);
        let mut gate = fetch.load_gate(0);
        assert!(fetch.stage(&mut kv, 0, candidate(2, 1_000, 10), &mut gate, at));
    }

    #[test]
    fn reads_start_in_order_and_a_later_one_never_jumps_the_head() {
        let (fetch, mut kv) = fetch_with(&[6_000, 6_000, 1_000]);
        let mut fetch = fetch.with_bounded_loading(|| Time::from_ms(10.0), 1_000_000);
        let mut gate = fetch.load_gate(0);
        assert!(fetch.stage(&mut kv, 0, candidate(1, 6_000, 10), &mut gate, Time::ZERO));
        assert!(!fetch.stage(&mut kv, 0, candidate(2, 6_000, 10), &mut gate, Time::ZERO));
        // Session 3's read would fit, but the head (2) did not.
        assert!(!fetch.stage(&mut kv, 0, candidate(3, 1_000, 10), &mut gate, Time::ZERO));
        assert_eq!(hbm_used(&fetch), 6_000);
    }

    #[test]
    fn a_read_larger_than_the_area_starts_when_it_is_empty() {
        let (fetch, mut kv) = fetch_with(&[25_000, 100]);
        let mut fetch = fetch.with_bounded_loading(|| Time::from_ms(10.0), 1_000_000);
        let mut gate = fetch.load_gate(0);
        assert!(fetch.stage(&mut kv, 0, candidate(1, 25_000, 10), &mut gate, Time::ZERO));
        assert!(!fetch.stage(&mut kv, 0, candidate(2, 100, 10), &mut gate, Time::ZERO));
        let at = Time::from_ms(25.0);
        fetch.land(&mut kv, at);
        // Landed, it still fills more than the area until admitted.
        let mut gate = fetch.load_gate(0);
        assert!(!fetch.stage(&mut kv, 0, candidate(2, 100, 10), &mut gate, at));
        admit_landed(&mut fetch, &mut kv, at);
        let mut gate = fetch.load_gate(0);
        assert!(fetch.stage(&mut kv, 0, candidate(2, 100, 10), &mut gate, at));
    }

    #[test]
    fn a_bounded_read_starts_only_if_its_blocks_fit_in_hbm() {
        let (fetch, _) = fetch_with(&[6_000, 100]);
        let mut fetch = fetch.with_bounded_loading(|| Time::from_ms(10.0), 1_000_000);
        let mut kv = FullAttnKv::with_prefix_cache(
            1,
            5_000,
            PrefixCacheTokenConfig {
                max_retained_tokens: 5_000,
                policy: PrefixCachePolicy::Lru,
            },
            None,
            None,
        );
        let mut gate = fetch.load_gate(0);
        // 6000 tokens fit the in-flight area but not HBM: it waits, and the
        // read behind it does not jump it.
        assert!(!fetch.stage(&mut kv, 0, candidate(1, 6_000, 10), &mut gate, Time::ZERO));
        assert!(!fetch.stage(&mut kv, 0, candidate(2, 100, 10), &mut gate, Time::ZERO));
        assert_eq!(hbm_used(&fetch), 0);
    }

    #[test]
    fn a_bounded_read_holds_its_whole_request_so_landed_data_can_always_be_admitted() {
        let (fetch, _) = fetch_with(&[6_000, 100]);
        let mut fetch = fetch.with_bounded_loading(|| Time::from_ms(10.0), 1_000_000);
        // Room for one request's whole footprint (6000 + 50 new + 1 output).
        let mut kv = FullAttnKv::with_prefix_cache(
            1,
            6_100,
            PrefixCacheTokenConfig {
                max_retained_tokens: 6_100,
                policy: PrefixCachePolicy::Lru,
            },
            None,
            None,
        );
        let a = candidate(1, 6_000, 50);
        let mut gate = fetch.load_gate(0);
        assert!(fetch.stage(&mut kv, 0, a, &mut gate, Time::ZERO));
        // A second read would leave no room to admit the first: it waits.
        let mut gate = fetch.load_gate(0);
        assert!(!fetch.stage(&mut kv, 0, candidate(2, 100, 10), &mut gate, Time::ZERO));
        let at = Time::from_ms(6.0);
        fetch.land(&mut kv, at);
        assert_eq!(fetch.at_head(&mut kv, 0, a, at), AtHead::Admit);
        let resolved = kv.preview_prefill_context(0, a.fresh_prompt_tokens, a.session_input);
        assert_eq!(resolved.resident_prefix_tokens(), 6_000);
        let footprint = kv.footprint(a.request_id, resolved.post_prefill_context_tokens(), 1);
        assert!(kv.fits(0, &footprint));
    }

    #[test]
    fn a_landed_context_evicted_before_admission_is_read_again() {
        let (fetch, mut kv) = fetch_with(&[6_000]);
        let mut fetch = fetch.with_bounded_loading(|| Time::from_ms(10.0), 1_000_000);
        let a = candidate(1, 6_000, 10);
        let mut gate = fetch.load_gate(0);
        assert!(fetch.stage(&mut kv, 0, a, &mut gate, Time::ZERO));
        let at = Time::from_ms(6.0);
        fetch.land(&mut kv, at);
        // Admission reaches it and puts the context back in HBM, but it does
        // not fit beside the running requests and stays.
        assert_eq!(fetch.at_head(&mut kv, 0, a, at), AtHead::Admit);
        // Another session's context fills HBM's retained tier, evicting it.
        kv.restore_prefix(RequestId(9), 0, 9, 1_000_000, 0, &[], EndState::Pass, at);
        assert_eq!(fetch.at_head(&mut kv, 0, a, at), AtHead::PassedOver);
        assert_eq!(fetch.partitions[0].rereads, 1);
        fetch.pop_landed(0);
        let mut gate = fetch.load_gate(0);
        assert!(fetch.stage(&mut kv, 0, a, &mut gate, at));
    }

    #[test]
    fn a_read_starts_if_it_fits_its_area_or_the_area_is_empty() {
        assert!(area_fits(0, 100, 500));
        assert!(area_fits(40, 100, 60));
        assert!(!area_fits(41, 100, 60));
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
