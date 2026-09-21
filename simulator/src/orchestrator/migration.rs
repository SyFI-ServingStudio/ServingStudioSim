//! When a DP pool moves resident work between its own workers (L6a policy).
//!
//! The decision lives here and the execution lives in L5: a policy sees a
//! read-only load snapshot and returns orders naming two workers. It never
//! holds a worker, so it cannot reach into a request or KV lifecycle even by
//! accident — the pool turns an order into one `drain` call on the source and
//! one message per drained request to the destination.
//!
//! Migration here is **recompute, not transfer**: the source releases its KV
//! and the destination re-prefills `prompt + emitted` tokens. No policy in this
//! module knows that, which is the point — it decides *whether* to move work,
//! not what moving costs.

use crate::common::{Time, WorkerId};

/// What a migration policy is allowed to see about one worker.
///
/// Today this is exactly `WorkerStatus`'s two counters plus the retired flag.
/// It is a separate struct so widening L6's view later (resident KV, say) does
/// not change every policy's signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkerLoad {
    pub worker: WorkerId,
    /// Requests admitted to the worker but not yet in a batch.
    pub queued_requests: u32,
    /// The active batch: live decodes plus this iteration's prefill admits.
    /// This is the quantity an "active batch below N" threshold means.
    pub active_requests: u32,
    /// Already drained by an earlier migration. A retired worker is never a
    /// source or a destination again, which is what keeps the pool's redirect
    /// chain acyclic.
    pub retired: bool,
}

/// Move everything `src` holds onto `dst`.
///
/// Whole-worker, not per-request: a policy that could name individual requests
/// would need to know which requests sit on a worker, and that membership is
/// owned by the worker's selection policy and KV store, not by L6.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MigrationOrder {
    pub src: WorkerId,
    pub dst: WorkerId,
}

/// The per-tick hook. Called once per pool tick, before the pool ticks its
/// workers, so a destination can start prefilling in the same tick.
pub trait MigrationPolicy {
    /// Append the orders to execute now. `loads` is indexed by worker id and
    /// covers every worker in the pool, retired ones included.
    fn decide(&mut self, now: Time, loads: &[WorkerLoad], out: &mut Vec<MigrationOrder>);
}

/// The preset-selectable triggers.
///
/// A closed set, so an enum dispatcher rather than a registry — the same shape
/// `PendingOrder` uses for admission's selection axis. A study that wants its
/// own rule implements [`MigrationPolicy`] directly instead of extending this.
#[derive(Clone, Copy, Debug)]
pub enum MigrationTrigger {
    /// Consolidate: when a live worker's active batch falls below `threshold`,
    /// drain it onto the busiest remaining live worker.
    ActiveBatchBelow {
        threshold: u32,
        cooldown: Time,
        last_fired: Option<Time>,
    },
}

impl MigrationTrigger {
    pub const fn active_batch_below(threshold: u32, cooldown: Time) -> Self {
        Self::ActiveBatchBelow {
            threshold,
            cooldown,
            last_fired: None,
        }
    }
}

impl MigrationPolicy for MigrationTrigger {
    fn decide(&mut self, now: Time, loads: &[WorkerLoad], out: &mut Vec<MigrationOrder>) {
        match self {
            Self::ActiveBatchBelow {
                threshold,
                cooldown,
                last_fired,
            } => {
                if let Some(fired) = *last_fired {
                    if now < fired + *cooldown {
                        return;
                    }
                }
                let Some(order) = plan_consolidation(loads, *threshold) else {
                    return;
                };
                *last_fired = Some(now);
                out.push(order);
            }
        }
    }
}

/// Pick the emptiest live worker that is under `threshold` and fold it into the
/// fullest other live worker.
///
/// Every comparison breaks ties on the worker id, so the same load snapshot
/// always yields the same order — a migration must not depend on iteration
/// order for the run to stay reproducible.
///
/// Returns `None` when nothing should move: no live worker is under threshold,
/// the only candidate is already empty (draining it would move nothing and
/// still retire it), or there is no second live worker to receive the work.
fn plan_consolidation(loads: &[WorkerLoad], threshold: u32) -> Option<MigrationOrder> {
    let src = loads
        .iter()
        .filter(|load| {
            !load.retired && load.active_requests > 0 && load.active_requests < threshold
        })
        .min_by_key(|load| (load.active_requests, load.worker))?;
    let dst = loads
        .iter()
        .filter(|load| !load.retired && load.worker != src.worker)
        .max_by_key(|load| (load.active_requests, std::cmp::Reverse(load.worker)))?;
    Some(MigrationOrder {
        src: src.worker,
        dst: dst.worker,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(worker: u16, active: u32) -> WorkerLoad {
        WorkerLoad {
            worker: WorkerId(worker),
            queued_requests: 0,
            active_requests: active,
            retired: false,
        }
    }

    fn decide(
        policy: &mut MigrationTrigger,
        now: Time,
        loads: &[WorkerLoad],
    ) -> Vec<MigrationOrder> {
        let mut out = Vec::new();
        policy.decide(now, loads, &mut out);
        out
    }

    #[test]
    fn the_emptiest_under_threshold_folds_into_the_fullest() {
        let mut policy = MigrationTrigger::active_batch_below(32, Time::ZERO);

        let orders = decide(
            &mut policy,
            Time::ZERO,
            &[load(0, 40), load(1, 5), load(2, 12)],
        );

        assert_eq!(
            orders,
            vec![MigrationOrder {
                src: WorkerId(1),
                dst: WorkerId(0),
            }]
        );
    }

    #[test]
    fn a_pool_at_or_above_threshold_stays_put() {
        let mut policy = MigrationTrigger::active_batch_below(32, Time::ZERO);

        assert!(decide(&mut policy, Time::ZERO, &[load(0, 32), load(1, 40)]).is_empty());
    }

    #[test]
    fn an_idle_worker_is_not_worth_retiring() {
        // Draining a worker with nothing on it moves no request but still takes
        // it out of the pool, so the pool would shrink on a quiet moment.
        let mut policy = MigrationTrigger::active_batch_below(32, Time::ZERO);

        assert!(decide(&mut policy, Time::ZERO, &[load(0, 0), load(1, 40)]).is_empty());
    }

    #[test]
    fn the_last_live_worker_has_nowhere_to_send_its_work() {
        let mut policy = MigrationTrigger::active_batch_below(32, Time::ZERO);
        let retired = WorkerLoad {
            retired: true,
            ..load(1, 0)
        };

        assert!(decide(&mut policy, Time::ZERO, &[load(0, 5), retired]).is_empty());
    }

    #[test]
    fn a_retired_worker_is_never_a_destination() {
        let mut policy = MigrationTrigger::active_batch_below(32, Time::ZERO);
        let retired = WorkerLoad {
            retired: true,
            ..load(2, 99)
        };

        let orders = decide(&mut policy, Time::ZERO, &[load(0, 5), load(1, 10), retired]);

        assert_eq!(
            orders,
            vec![MigrationOrder {
                src: WorkerId(0),
                dst: WorkerId(1),
            }],
            "the retired worker has the biggest batch but must not receive work"
        );
    }

    #[test]
    fn the_cooldown_holds_a_second_order_back_until_it_elapses() {
        let mut policy = MigrationTrigger::active_batch_below(32, Time::from_ms(50.0));
        let loads = [load(0, 5), load(1, 40)];

        assert_eq!(decide(&mut policy, Time::from_ms(10.0), &loads).len(), 1);
        assert!(decide(&mut policy, Time::from_ms(59.0), &loads).is_empty());
        assert_eq!(decide(&mut policy, Time::from_ms(60.0), &loads).len(), 1);
    }

    #[test]
    fn equal_loads_break_ties_on_the_worker_id_in_both_directions() {
        // Determinism, not fairness: the same snapshot must always produce the
        // same order, so the lowest id drains and the lowest id receives.
        let mut policy = MigrationTrigger::active_batch_below(32, Time::ZERO);

        let orders = decide(
            &mut policy,
            Time::ZERO,
            &[load(0, 7), load(1, 7), load(2, 7)],
        );

        assert_eq!(
            orders,
            vec![MigrationOrder {
                src: WorkerId(0),
                dst: WorkerId(1),
            }]
        );
    }
}
