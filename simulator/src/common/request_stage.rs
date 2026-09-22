//! Cross-layer request stage/location protocol.
//!
//! Workers emit [`StageEvent`]s, deployments select one vocabulary, and the L7
//! logger persists both the events and the vocabulary needed to decode them.
//! Keeping this protocol in `common` avoids assigning a cross-worker request
//! lifecycle to any one worker or making L5 depend upward on deployment code.
//!
//! As a request flows through a deployment its location is recorded as a
//! sequence of `(time, code, pool, worker)` events. `code` is a
//! deployment-defined enum discriminant; the sim core and parquet layer treat
//! it as an opaque `u16`.
//!
//! Each enum ships a `NAMES` table mapping every discriminant to a name string.
//! The cross-deployment contract is the open **`category:detail` shape**: two
//! non-empty snake-case terms joined by one `:`. The analyzer currently
//! recognizes `pending`, `active`, `transfer`, and `done` as core categories,
//! but that set is deliberately not exhaustive — future categories such as
//! `suspended` remain valid and older readers must preserve unknown categories.
//! Physical location is carried separately by `(pool, worker)`.
//!
//! `code as u16` is the stored value and `NAMES[code as usize]` is its name, so
//! enum and table order must stay in lockstep (asserted by this module's tests).
//! The full worker location is `(code, pool, worker)` because `worker` alone is
//! only unique within a pool.

use super::id::{PoolId, WorkerId};
use super::time::Time;

/// Sentinel pool/worker for an event not yet placed on a real worker (the
/// initial `current_stage` before the first transition, or a cluster-level
/// stage). `u16::MAX` is outside any real `PoolId`/`WorkerId` range.
pub const NO_POOL: PoolId = PoolId(u16::MAX);
pub const NO_WORKER: WorkerId = WorkerId(u16::MAX);

/// One entry in a request's stage/location timeline.
#[derive(Clone, Copy, Debug)]
pub struct StageEvent {
    pub time: Time,
    pub code: u16,
    pub pool: PoolId,
    pub worker: WorkerId,
}

impl StageEvent {
    /// Placeholder `current_stage` before the first real transition. Its
    /// sentinel pool/worker never equal a real `(code, pool, worker)`, so the
    /// first [`crate::common::RequestRecord::record_stage`] always registers a
    /// move.
    pub fn unset(time: Time) -> Self {
        Self {
            time,
            code: u16::MAX,
            pool: NO_POOL,
            worker: NO_WORKER,
        }
    }
}

/// A deployment's `code -> "category:detail"` name table, dumped to
/// `run_meta.json` alongside the stage events it decodes.
#[derive(Clone, Copy, Debug)]
pub struct StageVocab {
    pub deployment: &'static str,
    pub names: &'static [&'static str],
}

/// Barebone / HP unified lifecycle: one pool runs prefill then decode.
#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnifiedStage {
    /// In the worker's `pending_prefills` queue (admitted to no batch yet).
    Pending = 0,
    /// Admitted (`promised` / `prefill_admits`), prefilling this iter.
    Prefill = 1,
    /// Resolved to decode; in the worker's decode batch.
    Decode = 2,
    /// Completed.
    Done = 3,
    /// Drained off this worker by a migration: its KV has been released here
    /// and it is on its way to another worker, which stamps `Pending` when it
    /// takes over. Appended after `Done` so the existing discriminants keep
    /// their stored values.
    Suspended = 4,
}

impl UnifiedStage {
    pub const NAMES: &'static [&'static str] = &[
        "pending:prefill",
        "active:prefill",
        "active:decode",
        "done:request",
        "suspended:migration",
    ];
    pub const VOCAB: StageVocab = StageVocab {
        deployment: "unified",
        names: Self::NAMES,
    };
}

/// PD lifecycle: separate prefill and decode pools bridged by a KV handoff.
#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PdStage {
    /// Prefill pool: in `pending_prefills`.
    PendingPrefill = 0,
    /// Prefill pool: prefilling this iter.
    Prefill = 1,
    /// Prefill pool: prefill done, KV resident, awaiting the decode-side pull.
    PrefillDoneAwaitPull = 2,
    /// Decode pool: KV pull submitted/in transit (`in_transit`).
    Transfer = 3,
    /// Decode pool: KV landed, awaiting a decode slot (`pending_decodes`).
    PendingDecode = 4,
    /// Decode pool: in the decode batch.
    Decode = 5,
    /// Completed.
    Done = 6,
}

impl PdStage {
    pub const NAMES: &'static [&'static str] = &[
        "pending:prefill",
        "active:prefill",
        "pending:kv_pull",
        "transfer:kv_pull",
        "pending:decode",
        "active:decode",
        "done:request",
    ];
    pub const VOCAB: StageVocab = StageVocab {
        deployment: "pd",
        names: Self::NAMES,
    };
}

/// AFD lifecycle across the layer-lockstep attention and FFN pools.
///
/// V1 is deliberately coarse: the sticky attention worker owns request-state
/// location for the whole lifecycle. The FFN Terminal owns token emission and
/// completion bookkeeping, but its `Decode` / `Done` category transitions retain
/// the attention owner's `(pool, worker)` rather than moving the request to a
/// round-robin FFN executor. Only once-per-request points are recorded because a
/// per-iteration marker would explode. Intra-slot pipeline sub-stages remain
/// outside this lifecycle vocabulary.
#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AfdStage {
    /// Attention worker: in `worker_pending` (KV-gated admission backlog).
    Pending = 0,
    /// Attention worker: admitted into a slot (`mark_admitted`), prefilling.
    Prefill = 1,
    /// Attention owner: first output token emitted (prefill resolved -> decoding).
    Decode = 2,
    /// Attention owner: completed.
    Done = 3,
}

impl AfdStage {
    pub const NAMES: &'static [&'static str] = &[
        "pending:prefill",
        "active:prefill",
        "active:decode",
        "done:request",
    ];
    pub const VOCAB: StageVocab = StageVocab {
        deployment: "afd",
        names: Self::NAMES,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_snake_case_term(term: &str) -> bool {
        !term.is_empty()
            && term
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
            && !term.starts_with('_')
            && !term.ends_with('_')
            && !term.contains("__")
    }

    /// Categories are intentionally open; only the two-level snake-case wire
    /// shape is a cross-deployment contract.
    fn assert_category_detail(names: &[&str]) {
        for n in names {
            let parts: Vec<&str> = n.split(':').collect();
            assert_eq!(parts.len(), 2, "name {n:?} is not category:detail");
            assert!(
                is_snake_case_term(parts[0]) && is_snake_case_term(parts[1]),
                "name {n:?} must contain two non-empty snake-case terms"
            );
        }
    }

    #[test]
    fn names_are_category_detail() {
        for vocab in [UnifiedStage::VOCAB, PdStage::VOCAB, AfdStage::VOCAB] {
            assert_category_detail(vocab.names);
        }
    }

    #[test]
    fn code_indexes_its_name() {
        // The discriminant is the index into NAMES; a drift here mislabels every
        // logged event, so pin the mapping for one representative per enum.
        assert_eq!(
            UnifiedStage::NAMES[UnifiedStage::Decode as usize],
            "active:decode"
        );
        assert_eq!(
            PdStage::NAMES[PdStage::PrefillDoneAwaitPull as usize],
            "pending:kv_pull"
        );
        assert_eq!(
            PdStage::NAMES[PdStage::PendingPrefill as usize],
            "pending:prefill"
        );
        assert_eq!(AfdStage::NAMES[AfdStage::Done as usize], "done:request");
        assert_eq!(PdStage::NAMES.len(), 7);
    }
}
