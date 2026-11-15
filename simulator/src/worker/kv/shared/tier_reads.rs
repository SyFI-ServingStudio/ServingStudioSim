//! Reads from host tiers in flight, and what each partition's admissions
//! found where.
//!
//! A read starts when admission reaches its request (`TieredPrefixKv`), lands
//! when its tier's channel has moved it, and ends when admission puts its
//! context back into HBM. A context put back but evicted again before its
//! request is admitted is read again.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::common::{RequestId, Time};
use crate::worker::shared::advance_scope::PartitionId;

use super::tier_chain::{TierHit, TierRead};

/// Where a read is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadPhase {
    /// On its tier's channel.
    Reading,
    /// Moved, its request still holding its HBM footprint.
    Landed,
    /// Its context back in HBM, its request not yet admitted.
    Restored,
}

#[derive(Clone, Debug)]
pub(crate) struct ReadRecord {
    pub(crate) partition: PartitionId,
    pub(crate) session_id: u32,
    pub(crate) hit: TierHit,
    /// What HBM held when the read started.
    pub(crate) resident: u64,
    pub(crate) read: TierRead,
    pub(crate) phase: ReadPhase,
}

/// What one partition's session admissions found where.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct AdmissionStats {
    pub(crate) session_requests: u64,
    pub(crate) declared_tokens: u64,
    /// Resident tokens of the requests admitted without a read.
    pub(crate) hbm_hit_tokens: u64,
    /// Resident tokens at admission, read or not.
    pub(crate) resident_tokens: u64,
    /// Per tier: requests admitted after reading from it, and their hits.
    pub(crate) tier_hit_requests: Vec<u64>,
    pub(crate) tier_hit_tokens: Vec<u64>,
    pub(crate) warm_start_reads: u64,
    pub(crate) warm_start_tokens: u64,
    /// Reads of a context an earlier read put back but HBM evicted before
    /// admission.
    pub(crate) rereads: u64,
}

pub(crate) struct TierReads {
    records: HashMap<RequestId, ReadRecord>,
    /// Reads on a channel, by landing time.
    landing: BinaryHeap<Reverse<(Time, RequestId)>>,
    /// Per partition: the longest context the run stored for each session.
    stored: Vec<HashMap<u32, u64>>,
    stats: Vec<AdmissionStats>,
}

impl TierReads {
    pub(crate) fn new(num_partitions: usize, num_tiers: usize) -> Self {
        Self {
            records: HashMap::new(),
            landing: BinaryHeap::new(),
            stored: vec![HashMap::new(); num_partitions],
            stats: vec![
                AdmissionStats {
                    tier_hit_requests: vec![0; num_tiers],
                    tier_hit_tokens: vec![0; num_tiers],
                    ..AdmissionStats::default()
                };
                num_partitions
            ],
        }
    }

    pub(crate) fn record(&self, request: RequestId) -> Option<&ReadRecord> {
        self.records.get(&request)
    }

    /// Track a read that just started.
    pub(crate) fn start(&mut self, request: RequestId, record: ReadRecord) {
        if self
            .records
            .get(&request)
            .is_some_and(|previous| previous.phase == ReadPhase::Restored)
        {
            self.stats[record.partition as usize].rereads += 1;
        }
        self.landing.push(Reverse((record.read.ready, request)));
        self.records.insert(request, record);
    }

    pub(crate) fn next_landing(&self) -> Option<Time> {
        self.landing.peek().map(|Reverse((ready, _))| *ready)
    }

    /// Land every read done by `now`, in landing order.
    pub(crate) fn land(&mut self, now: Time) -> Vec<(RequestId, Time)> {
        let mut landed = Vec::new();
        while let Some(&Reverse((ready, request))) = self.landing.peek() {
            if ready > now {
                break;
            }
            self.landing.pop();
            let record = self
                .records
                .get_mut(&request)
                .expect("a landing belongs to a read");
            record.phase = ReadPhase::Landed;
            landed.push((request, ready));
        }
        landed
    }

    /// The landed read of `request`, now restored into HBM.
    pub(crate) fn restore(&mut self, request: RequestId) -> &ReadRecord {
        let record = self
            .records
            .get_mut(&request)
            .expect("a restored request has a landed read");
        debug_assert_eq!(record.phase, ReadPhase::Landed);
        record.phase = ReadPhase::Restored;
        record
    }

    /// Count a session request admitted on `partition` with `declared`
    /// tokens, `resident` of them in HBM; returns the read it took, if any.
    pub(crate) fn admit(
        &mut self,
        partition: PartitionId,
        request: RequestId,
        declared: u64,
        resident: u64,
    ) -> Option<ReadRecord> {
        let record = self.records.remove(&request);
        let stats = &mut self.stats[partition as usize];
        stats.session_requests += 1;
        stats.declared_tokens += declared;
        stats.resident_tokens += resident;
        match &record {
            Some(record) => {
                stats.tier_hit_requests[record.hit.tier] += 1;
                stats.tier_hit_tokens[record.hit.tier] += record.hit.tokens;
            }
            None => stats.hbm_hit_tokens += resident,
        }
        record
    }

    /// The longest context the run stored for the session on `partition`.
    pub(crate) fn stored(&self, partition: PartitionId, session_id: u32) -> u64 {
        self.stored[partition as usize]
            .get(&session_id)
            .copied()
            .unwrap_or(0)
    }

    pub(crate) fn note_stored(&mut self, partition: PartitionId, session_id: u32, tokens: u64) {
        let stored = self.stored[partition as usize]
            .entry(session_id)
            .or_insert(0);
        *stored = (*stored).max(tokens);
    }

    pub(crate) fn note_warm_start(&mut self, partition: PartitionId, tokens: u64) {
        let stats = &mut self.stats[partition as usize];
        stats.warm_start_reads += 1;
        stats.warm_start_tokens += tokens;
    }

    pub(crate) fn stats(&self, partition: PartitionId) -> &AdmissionStats {
        &self.stats[partition as usize]
    }
}
