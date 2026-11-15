//! Per-worker exact replay log for retained-prefix cache operations.
//!
//! `PrefixCache` remains the sole cache ledger. The KV owner records only the
//! mutation receipts returned by that ledger, so this logger never infers cache
//! state from throttled `kv_snapshot` samples or maintains shadow ownership.

use std::path::Path;
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::thread::JoinHandle;

use anyhow::{anyhow, Result};

use crate::common::{RequestId, Time, WorkerId};
use crate::log::cost_logger::cost_artifact_stem;
use crate::log::parquet_writer::StreamingParquetWriter;
use crate::log::rows::{prefix_cache_event_to_record_batch, PrefixCacheEventEntry};
use crate::log::schemas::prefix_cache_event_schema;

const STREAM_FLUSH_ROWS: usize = 8_192;
const CHANNEL_CAP: usize = 64;

/// Why an active request removed a retained prefix-cache entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixCacheActivation {
    Hit,
    Miss,
    /// A host tier's entry read toward HBM.
    TierRead,
    /// A staged SSD read moving on from DRAM's loading area to HBM's.
    TierHop,
}

/// Why a retained prefix-cache entry was evicted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixCacheEvictionReason {
    ActiveKvPressure,
    ReplacementPolicy,
    RetentionCapacity,
    SameSessionReplacement,
    /// A host tier's entry cut to what the session's next request shares.
    TierCut,
}

/// Why an active request returned its context to the retained cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixCacheRetentionReason {
    RequestComplete,
    HandoffComplete,
    NoCacheCapacity,
    /// A context read back from a host tier, put into HBM at admission.
    TierRead,
    /// A finished context written through to a host tier.
    TierWriteThrough,
    /// What the tier above evicted or could not keep, in its victim tier.
    TierVictim,
    /// A read from a slower tier copied into a faster one on its way.
    TierReadCopy,
    /// A context older than the run put into the slowest tier.
    TierWarmStart,
}

/// One valid prefix-cache operation/reason pair.
///
/// Keeping the pair typed prevents impossible rows such as `activate` with an
/// eviction reason. Stable strings are produced only at the parquet boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixCacheEventKind {
    Activate(PrefixCacheActivation),
    Evict(PrefixCacheEvictionReason),
    Retain(PrefixCacheRetentionReason),
}

impl PrefixCacheEventKind {
    fn operation(self) -> &'static str {
        match self {
            Self::Activate(_) => "activate",
            Self::Evict(_) => "evict",
            Self::Retain(_) => "retain",
        }
    }

    fn reason(self) -> &'static str {
        match self {
            Self::Activate(PrefixCacheActivation::Hit) => "hit",
            Self::Activate(PrefixCacheActivation::Miss) => "miss",
            Self::Activate(PrefixCacheActivation::TierRead) => "read",
            Self::Activate(PrefixCacheActivation::TierHop) => "hop",
            Self::Evict(PrefixCacheEvictionReason::ActiveKvPressure) => "active-kv-pressure",
            Self::Evict(PrefixCacheEvictionReason::ReplacementPolicy) => "replacement-policy",
            Self::Evict(PrefixCacheEvictionReason::RetentionCapacity) => "retention-capacity",
            Self::Evict(PrefixCacheEvictionReason::SameSessionReplacement) => {
                "same-session-replacement"
            }
            Self::Evict(PrefixCacheEvictionReason::TierCut) => "cut",
            Self::Retain(PrefixCacheRetentionReason::RequestComplete) => "request-complete",
            Self::Retain(PrefixCacheRetentionReason::HandoffComplete) => "handoff-complete",
            Self::Retain(PrefixCacheRetentionReason::NoCacheCapacity) => "no-cache-capacity",
            Self::Retain(PrefixCacheRetentionReason::TierRead) => "tier-read",
            Self::Retain(PrefixCacheRetentionReason::TierWriteThrough) => "write-through",
            Self::Retain(PrefixCacheRetentionReason::TierVictim) => "victim",
            Self::Retain(PrefixCacheRetentionReason::TierReadCopy) => "read-copy",
            Self::Retain(PrefixCacheRetentionReason::TierWarmStart) => "warm-start",
        }
    }
}

/// The `tier` of an HBM prefix-cache row.
pub const HBM_TIER: &str = "hbm";

/// Cache transition facts supplied by the KV owner.
#[derive(Clone, Copy, Debug)]
pub struct PrefixCacheEvent {
    /// [`HBM_TIER`], or the host tier's name.
    pub tier: &'static str,
    pub partition_id: u16,
    pub time: Time,
    pub request_id: RequestId,
    pub session_id: u32,
    pub kind: PrefixCacheEventKind,
    pub entry_tokens_before: u64,
    pub entry_tokens_after: u64,
    pub cache_used_before: u64,
    pub cache_used_after: u64,
    pub requested_tokens: u64,
    /// Tokens the activation resumes from: the entry's last resume position at
    /// or below the request's shared prefix.
    pub hit_tokens: u64,
    /// The shared prefix the entry covers, wherever it could resume. The gap to
    /// `hit_tokens` is what resume positions cost a hit.
    pub full_hit_tokens: u64,
    /// A host tier's used capacity in its own units (one GPU's per-token
    /// bytes, states included), before and after; `None` on HBM rows.
    pub cache_used_units: Option<(u64, u64)>,
    /// A tier read's landing time.
    pub ready: Option<Time>,
}

/// Sim-thread handle for one worker's sparse cache-operation stream. Sequence
/// assignment happens before buffering, giving equal-time events a total order.
pub struct PrefixCacheLogger {
    tx: Option<SyncSender<Vec<PrefixCacheEventEntry>>>,
    handle: Option<JoinHandle<Result<()>>>,
    buffer: Vec<PrefixCacheEventEntry>,
    worker_id: WorkerId,
    next_sequence: u64,
    closed: bool,
}

impl PrefixCacheLogger {
    pub fn open_opt(
        log_dir: Option<&Path>,
        pool_tag: &'static str,
        worker_id: WorkerId,
    ) -> Option<Self> {
        let directory = log_dir?;
        match Self::open(directory, pool_tag, worker_id) {
            Ok(logger) => Some(logger),
            Err(error) => {
                tracing::warn!("prefix_cache_event disabled: failed to open writer: {error:#}");
                None
            }
        }
    }

    pub fn open(log_dir: &Path, pool_tag: &'static str, worker_id: WorkerId) -> Result<Self> {
        let stream_directory = log_dir.join("raw").join("prefix_cache_event");
        std::fs::create_dir_all(&stream_directory)?;
        let path = stream_directory.join(format!(
            "{}.parquet",
            cost_artifact_stem(pool_tag, worker_id)
        ));
        let mut writer = StreamingParquetWriter::new(path, prefix_cache_event_schema());
        let (tx, rx) = sync_channel::<Vec<PrefixCacheEventEntry>>(CHANNEL_CAP);
        let handle = std::thread::Builder::new()
            .name("vibesim-prefix-cache-logger".to_string())
            .spawn(move || -> Result<()> {
                for chunk in rx {
                    writer.write(&prefix_cache_event_to_record_batch(pool_tag, &chunk)?)?;
                }
                writer.close()?;
                Ok(())
            })?;

        Ok(Self {
            tx: Some(tx),
            handle: Some(handle),
            buffer: Vec::with_capacity(STREAM_FLUSH_ROWS),
            worker_id,
            next_sequence: 0,
            closed: false,
        })
    }

    /// Record one authoritative mutation receipt. Logging failures disable this
    /// stream and warn; they never abort or alter the simulation FSM.
    pub fn record(&mut self, event: PrefixCacheEvent) {
        if self.closed {
            return;
        }
        self.buffer.push(PrefixCacheEventEntry {
            tier: event.tier,
            worker_id: self.worker_id.0,
            partition_id: event.partition_id,
            sequence: self.next_sequence,
            time_ms: event.time.as_ms(),
            request_id: event.request_id.0,
            session_id: event.session_id,
            operation: event.kind.operation(),
            reason: event.kind.reason(),
            entry_tokens_before: event.entry_tokens_before,
            entry_tokens_after: event.entry_tokens_after,
            cache_used_before: event.cache_used_before,
            cache_used_after: event.cache_used_after,
            requested_tokens: event.requested_tokens,
            hit_tokens: event.hit_tokens,
            full_hit_tokens: event.full_hit_tokens,
            cache_used_units_before: event.cache_used_units.map(|(before, _)| before),
            cache_used_units_after: event.cache_used_units.map(|(_, after)| after),
            ready_ms: event.ready.map(Time::as_ms),
        });
        self.next_sequence = self.next_sequence.saturating_add(1);
        if self.buffer.len() >= STREAM_FLUSH_ROWS {
            if let Err(error) = self.send() {
                tracing::warn!("prefix_cache_event record failed: {error:#}");
                self.closed = true;
            }
        }
    }

    fn send(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let chunk = std::mem::replace(&mut self.buffer, Vec::with_capacity(STREAM_FLUSH_ROWS));
        let tx = self.tx.as_ref().expect("tx present until flush");
        let chunk = match tx.try_send(chunk) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(chunk)) => {
                tracing::warn!(
                    "prefix-cache-log channel full ({CHANNEL_CAP} chunks in flight): sim thread \
                     blocking on prefix-cache logger backpressure"
                );
                chunk
            }
            Err(TrySendError::Disconnected(_)) => {
                return Err(self
                    .join_writer()
                    .err()
                    .unwrap_or_else(|| anyhow!("prefix-cache writer thread disconnected")))
            }
        };
        match self.tx.as_ref().expect("tx present").send(chunk) {
            Ok(()) => Ok(()),
            Err(_) => Err(self
                .join_writer()
                .err()
                .unwrap_or_else(|| anyhow!("prefix-cache writer thread disconnected"))),
        }
    }

    fn join_writer(&mut self) -> Result<()> {
        drop(self.tx.take());
        match self.handle.take() {
            Some(handle) => handle
                .join()
                .map_err(|_| anyhow!("prefix-cache writer thread panicked"))?,
            None => Ok(()),
        }
    }

    pub fn flush_all(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        self.send()?;
        self.join_writer()
    }
}

impl Drop for PrefixCacheLogger {
    fn drop(&mut self) {
        let _ = self.flush_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::File;

    use arrow_array::{Array, Float64Array, StringArray, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::tempdir;

    #[test]
    fn writes_strictly_ordered_replay_rows() {
        let directory = tempdir().unwrap();
        let mut logger = PrefixCacheLogger::open(directory.path(), "main", WorkerId(3)).unwrap();
        logger.record(PrefixCacheEvent {
            tier: HBM_TIER,
            partition_id: 1,
            time: Time::from_ms(5.0),
            request_id: RequestId(42),
            session_id: 7,
            kind: PrefixCacheEventKind::Activate(PrefixCacheActivation::Hit),
            entry_tokens_before: 120,
            entry_tokens_after: 0,
            cache_used_before: 300,
            cache_used_after: 180,
            requested_tokens: 100,
            hit_tokens: 100,
            full_hit_tokens: 100,
            cache_used_units: None,
            ready: None,
        });
        logger.record(PrefixCacheEvent {
            tier: HBM_TIER,
            partition_id: 1,
            time: Time::from_ms(5.0),
            request_id: RequestId(42),
            session_id: 7,
            kind: PrefixCacheEventKind::Retain(PrefixCacheRetentionReason::RequestComplete),
            entry_tokens_before: 0,
            entry_tokens_after: 140,
            cache_used_before: 180,
            cache_used_after: 320,
            requested_tokens: 140,
            hit_tokens: 0,
            full_hit_tokens: 0,
            cache_used_units: None,
            ready: None,
        });
        logger.record(PrefixCacheEvent {
            tier: "dram",
            partition_id: 1,
            time: Time::from_ms(5.0),
            request_id: RequestId(43),
            session_id: 7,
            kind: PrefixCacheEventKind::Activate(PrefixCacheActivation::TierRead),
            entry_tokens_before: 140,
            entry_tokens_after: 140,
            cache_used_before: 140,
            cache_used_after: 140,
            requested_tokens: 140,
            hit_tokens: 140,
            full_hit_tokens: 140,
            cache_used_units: Some((150, 150)),
            ready: Some(Time::from_ms(7.0)),
        });
        logger.flush_all().unwrap();

        let path = directory
            .path()
            .join("raw/prefix_cache_event/worker_main_3.parquet");
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 3);

        let sequence = batch
            .column_by_name("sequence")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!((sequence.value(0), sequence.value(2)), (0, 2));
        let tier = batch
            .column_by_name("tier")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!((tier.value(0), tier.value(2)), ("hbm", "dram"));
        let units = batch
            .column_by_name("cache_used_units_after")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert!(units.is_null(0));
        assert_eq!(units.value(2), 150);
        let ready = batch
            .column_by_name("ready_ms")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!(ready.is_null(1));
        assert_eq!(ready.value(2), 7.0);
        let operation = batch
            .column_by_name("operation")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            (operation.value(0), operation.value(1)),
            ("activate", "retain")
        );

        for row in 0..batch.num_rows() {
            let entry_before = batch
                .column_by_name("entry_tokens_before")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(row);
            let entry_after = batch
                .column_by_name("entry_tokens_after")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(row);
            let cache_before = batch
                .column_by_name("cache_used_before")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(row);
            let cache_after = batch
                .column_by_name("cache_used_after")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(row);
            assert_eq!(cache_after, cache_before - entry_before + entry_after);
        }
    }
}
