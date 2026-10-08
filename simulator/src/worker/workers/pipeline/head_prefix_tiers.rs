//! A pipeline head's DRAM/SSD prefix tiers and their per-request log.
//!
//! The head looks a session up here when its request arrives: a context that
//! HBM no longer holds but a slower tier does is read back first, and the
//! request queues once the read lands. Every finished context is written
//! through. `prefix_tiers_w<id>.csv` records, per session request, where its
//! declared prefix was found; `prefix_tiers_w<id>.json` the tiers' totals.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::common::{RequestId, Time, WorkerId};
use crate::worker::kv::{PrefixTierHit, PrefixTierSpec, PrefixTiers};

pub(crate) struct HeadPrefixTiers {
    tiers: PrefixTiers,
    /// HBM resumes a session only at multiples of this (a hybrid model's
    /// state block; 1 for full attention), so a tier hit counts in it too.
    hit_quantum: u32,
    rows: Option<BufWriter<File>>,
    summary_path: Option<PathBuf>,
    /// Session requests seen, and declared prefix tokens found in HBM, in
    /// each tier, and nowhere.
    requests: u64,
    declared_tokens: u64,
    hbm_tokens: u64,
    tier_tokens: Vec<u64>,
    tier_requests: Vec<u64>,
}

impl HeadPrefixTiers {
    pub(crate) fn new(
        specs: &[PrefixTierSpec],
        kv_bytes_per_token: u64,
        state_tokens: u64,
        hit_quantum: u32,
        log_dir: Option<&Path>,
        worker: WorkerId,
    ) -> Self {
        let rows = log_dir.map(|dir| {
            let mut file = BufWriter::new(
                File::create(dir.join(format!("prefix_tiers_w{}.csv", worker.0)))
                    .expect("create the prefix tier log"),
            );
            writeln!(
                file,
                "request_id,session_id,arrival_ms,declared_tokens,hbm_tokens,tier,tier_tokens,ready_ms"
            )
            .expect("write the prefix tier log");
            file
        });
        Self {
            tiers: PrefixTiers::new(specs, kv_bytes_per_token, state_tokens),
            hit_quantum: hit_quantum.max(1),
            rows,
            summary_path: log_dir.map(|dir| dir.join(format!("prefix_tiers_w{}.json", worker.0))),
            requests: 0,
            declared_tokens: 0,
            hbm_tokens: 0,
            tier_tokens: vec![0; specs.len()],
            tier_requests: vec![0; specs.len()],
        }
    }

    /// Decide where an arriving session request's prefix comes from. A slower
    /// tier serves it only when, floored to the HBM hit quantum, it holds more
    /// than HBM does, and then reads just the difference. Returns the hit and
    /// when its read lands, or `None` to queue it now.
    pub(crate) fn on_arrival(
        &mut self,
        request: RequestId,
        session_id: u32,
        declared_tokens: u32,
        hbm_tokens: u32,
        arrival: Time,
    ) -> Option<(PrefixTierHit, Time)> {
        let quantum = self.hit_quantum;
        let hit = self
            .tiers
            .lookup(session_id, declared_tokens)
            .map(|hit| PrefixTierHit {
                tokens: hit.tokens / quantum * quantum,
                ..hit
            })
            .filter(|hit| hit.tokens > hbm_tokens);
        let ready = hit.map(|hit| self.tiers.load(session_id, hit, hbm_tokens, arrival));
        self.requests += 1;
        self.declared_tokens += u64::from(declared_tokens);
        match hit {
            Some(hit) => {
                self.tier_tokens[hit.tier] += u64::from(hit.tokens);
                self.tier_requests[hit.tier] += 1;
            }
            None => self.hbm_tokens += u64::from(hbm_tokens),
        }
        if let Some(rows) = &mut self.rows {
            let (tier, tokens) = match hit {
                Some(hit) => (self.tiers.tier_name(hit.tier), hit.tokens),
                None if hbm_tokens > 0 => ("hbm", hbm_tokens),
                None => ("none", 0),
            };
            writeln!(
                rows,
                "{},{},{:.3},{},{},{},{},{:.3}",
                request.0,
                session_id,
                arrival.as_ms(),
                declared_tokens,
                hbm_tokens,
                tier,
                tokens,
                ready.unwrap_or(arrival).as_ms()
            )
            .expect("write the prefix tier log");
        }
        hit.zip(ready)
    }

    pub(crate) fn store(&mut self, session_id: u32, tokens: u64) {
        self.tiers.store(session_id, tokens);
    }
}

impl Drop for HeadPrefixTiers {
    fn drop(&mut self) {
        if let Some(rows) = &mut self.rows {
            let _ = rows.flush();
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
                })
            })
            .collect();
        let summary = serde_json::json!({
            "session_requests": self.requests,
            "declared_tokens": self.declared_tokens,
            "hbm_hit_tokens_at_arrival": self.hbm_tokens,
            "tiers": tiers,
        });
        if let Ok(file) = File::create(path) {
            let _ = serde_json::to_writer_pretty(file, &summary);
        }
    }
}
