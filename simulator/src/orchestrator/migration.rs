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
    /// Prompt groups with at least one request still unfinished here.
    ///
    /// A *prompt group* is a block of `migration_group_size` consecutive request
    /// ids — the `n_samples_per_prompt` copies an RL rollout draws from one
    /// prompt. The group stays in flight until its **slowest** member lands, so
    /// this counts work a scheduler cannot retire yet even where most of the
    /// group is done. With a group size of 1 it degenerates to "unfinished
    /// requests placed here", which is what a policy that does not care about
    /// grouping should read it as.
    pub in_flight_groups: u32,
    /// Requests that have completed here since the run began.
    pub completed_requests: u32,
    /// Already drained by an earlier migration. A retired worker is never a
    /// source or a destination again, which is what keeps the pool's redirect
    /// chain acyclic.
    pub retired: bool,
}

/// What the pool should do with one source worker's resident work.
///
/// `Consolidate` and `Scatter` drain the source completely and retire it,
/// differing only in how many destinations the work is split across;
/// `MoveGroup` takes one prompt group and leaves the source running.
///
/// None of them names a request. Keeping the unit at "a worker" or "a group"
/// leaves request membership where it belongs — in the worker's selection
/// policy and KV store — while still letting a policy that counts prompt groups
/// say where each group should land.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MigrationOrder {
    /// Everything `src` holds goes to `dst`.
    Consolidate { src: WorkerId, dst: WorkerId },
    /// `src`'s prompt groups are dealt out one destination each: the group with
    /// the lowest id goes to `dsts[0]`, the next to `dsts[1]`, and so on. Every
    /// request of a group moves together, so a group never straddles two
    /// workers and the receiving worker's own group count stays meaningful.
    ///
    /// `dsts` must have exactly one entry per in-flight group on `src` — the
    /// count the policy read from [`WorkerLoad::in_flight_groups`]. It may be
    /// empty: a worker that finished its share still retires, because the block
    /// it belongs to is handed back as a unit.
    ///
    /// `retire_to` is where later arrivals pinned to `src` go, and is separate
    /// from `dsts` precisely so an empty source still leaves the pool. Letting
    /// it redirect to itself instead would keep a released worker eligible as a
    /// destination, and the next block to fire would hand its work straight
    /// back to a machine that is already gone.
    Scatter {
        src: WorkerId,
        dsts: Vec<WorkerId>,
        retire_to: WorkerId,
    },
    /// Move one of `src`'s in-flight prompt groups — the lowest id it still
    /// holds — to `dst`, and leave `src` running with the rest.
    ///
    /// The partial counterpart to `Scatter`, and the only order that does not
    /// retire its source. It exists because a release is not instantaneous in
    /// the system this models: the router aborts one group, waits for the
    /// acknowledgement, re-dispatches it, and moves on to the next, while the
    /// engine it is emptying keeps decoding everything still on it. Expressed
    /// as one `Scatter` at the end of that loop, the work sits on the source
    /// for the whole abort loop and then lands all at once — which is not what
    /// the destinations see.
    ///
    /// Which requests make up the group is the pool's to know, not the policy's:
    /// a policy reads group *counts* off [`WorkerLoad`] and never sees an id.
    MoveGroup { src: WorkerId, dst: WorkerId },
}

impl MigrationOrder {
    pub fn src(&self) -> WorkerId {
        match self {
            Self::Consolidate { src, .. }
            | Self::Scatter { src, .. }
            | Self::MoveGroup { src, .. } => *src,
        }
    }
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
#[derive(Clone, Debug)]
pub enum MigrationTrigger {
    /// Consolidate: when a live worker's active batch *falls* below
    /// `threshold`, drain it onto the busiest remaining live worker.
    ActiveBatchBelow {
        threshold: u32,
        cooldown: Time,
        last_fired: Option<Time>,
        /// Whether each worker has ever reached `threshold`.
        ///
        /// A pool starts empty, so "below threshold" is trivially true of
        /// every worker before any work arrives. Without this the trigger
        /// would consolidate the whole pool on the first tick and the run
        /// would never use more than one worker. A worker becomes a
        /// consolidation candidate only once its batch has been up and come
        /// back down, which is the situation the threshold is about.
        peaked: Vec<bool>,
    },
    /// Release a whole *train group* of workers at once: when the samples still
    /// in flight across its workers fall below `threshold`, scatter every
    /// prompt group they hold over the workers of the other train groups and
    /// retire the pair.
    ///
    /// Models RL post-training, where inference workers are borrowed from
    /// training in fixed blocks: a block is only useful back in training once
    /// *all* of its workers are free, so the decision is per block, and the
    /// gain is not a faster rollout but an earlier training start. Hence the
    /// differences from `ActiveBatchBelow`, all of which are load-bearing:
    ///
    /// * the counted unit is the prompt group, weighted by its full size, since
    ///   a group's slowest sample gates the whole group;
    /// * the trigger is one-shot per train group — a block is handed back once;
    /// * the work fans out across every eligible worker instead of piling onto
    ///   the busiest one, because the point is to keep the *receivers* at a full
    ///   batch, not to empty one more machine.
    TrainGroupSamplesBelow {
        /// The threshold, denominated in samples (not groups): a group of
        /// `group_size` contributes `group_size` until its last sample lands.
        threshold: u32,
        /// Requests per prompt group; ids are grouped in consecutive blocks.
        group_size: u32,
        /// Workers per train group, blocked by worker id: with 2, workers 0
        /// and 1 form train group 0, workers 2 and 3 form train group 1.
        workers_per_train_group: u16,
        /// How long handing back one prompt group takes. The system this models
        /// releases a block one group at a time — abort the group on its engine,
        /// wait for the acknowledgement, re-dispatch it elsewhere — and the
        /// engines it is leaving keep decoding throughout. A block holding 15
        /// groups therefore takes ~10 s to actually leave, which is invisible
        /// at a threshold that releases one group at a time and is most of the
        /// release at a threshold that releases fifteen.
        ///
        /// Zero collapses the whole loop into one tick, which is the historical
        /// behaviour and stays the default: with no latency there is no partial
        /// release to model, and the block goes in a single `Scatter`.
        group_latency: Time,
        /// Per train group: when it crossed the threshold, and how many of its
        /// groups have gone since. Together they say when the next one is due —
        /// `crossed_at + group_latency * moved` — so a release paces itself
        /// without the policy holding a timer. Cleared when the block's last
        /// group leaves and the block retires.
        ///
        /// A block that climbs back over the threshold while it is leaving
        /// still leaves: the system this models starts aborting immediately and
        /// does not reconsider.
        armed: Vec<Option<(Time, u32)>>,
        /// Train groups that have already fired. A block is handed back to
        /// training once; it never comes back to take more inference work.
        fired: Vec<bool>,
    },
}

impl MigrationTrigger {
    pub const fn active_batch_below(threshold: u32, cooldown: Time) -> Self {
        Self::ActiveBatchBelow {
            threshold,
            cooldown,
            last_fired: None,
            peaked: Vec::new(),
        }
    }

    pub const fn train_group_samples_below(
        threshold: u32,
        group_size: u32,
        workers_per_train_group: u16,
        group_latency: Time,
    ) -> Self {
        Self::TrainGroupSamplesBelow {
            threshold,
            group_size,
            workers_per_train_group,
            group_latency,
            armed: Vec::new(),
            fired: Vec::new(),
        }
    }

    /// Requests per prompt group this trigger counts in. The pool needs it to
    /// keep [`WorkerLoad::in_flight_groups`] and to bucket a `Scatter`.
    pub fn group_size(&self) -> u32 {
        match self {
            Self::ActiveBatchBelow { .. } => 1,
            Self::TrainGroupSamplesBelow { group_size, .. } => *group_size,
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
                peaked,
            } => {
                // Recorded every tick, including the ones the cooldown skips:
                // a worker that filled and emptied inside a cooldown window
                // still filled.
                peaked.resize(loads.len(), false);
                for load in loads {
                    if load.active_requests >= *threshold {
                        peaked[load.worker.0 as usize] = true;
                    }
                }
                if let Some(fired) = *last_fired {
                    if now < fired + *cooldown {
                        return;
                    }
                }
                let Some(order) = plan_consolidation(loads, *threshold, peaked) else {
                    return;
                };
                *last_fired = Some(now);
                out.push(order);
            }
            Self::TrainGroupSamplesBelow {
                threshold,
                group_size,
                workers_per_train_group,
                group_latency,
                armed,
                fired,
            } => plan_train_group_release(
                now,
                loads,
                *threshold,
                *group_size,
                *workers_per_train_group,
                *group_latency,
                armed,
                fired,
                out,
            ),
        }
    }
}

/// Hand back every train group whose remaining samples have fallen under
/// `threshold`, scattering its prompt groups over the workers that are left.
///
/// The candidate set shrinks *as this function runs*, not on the next tick: a
/// train group that fires is taken out of `live` immediately. Without that, two
/// groups crossing the threshold on the same tick would each read the other as
/// an idle, maximally attractive destination and hand their work straight to a
/// block that is on its way out — the work would land on workers nothing routes
/// to again.
#[allow(clippy::too_many_arguments)]
fn plan_train_group_release(
    now: Time,
    loads: &[WorkerLoad],
    threshold: u32,
    group_size: u32,
    workers_per_train_group: u16,
    group_latency: Time,
    armed: &mut Vec<Option<(Time, u32)>>,
    fired: &mut Vec<bool>,
    out: &mut Vec<MigrationOrder>,
) {
    assert!(workers_per_train_group > 0, "a train group needs a worker");
    assert!(group_size > 0, "a prompt group needs a request");
    let per_block = workers_per_train_group as usize;
    let num_blocks = loads.len().div_ceil(per_block);
    fired.resize(num_blocks, false);
    armed.resize(num_blocks, None);

    // Working copies: `live` starts from the pool's own retirement state and
    // then tracks this tick's decisions; `load` is the destination ranking key
    // and is bumped as groups are placed, so the fan-out stays even.
    let mut live: Vec<bool> = loads.iter().map(|load| !load.retired).collect();
    let mut load: Vec<u32> = loads.iter().map(|load| load.in_flight_groups).collect();
    // A block that is mid-release is on its way out, so it does not take work,
    // but it is still running and still holds requests until its release lands.
    let mut leaving: Vec<bool> = (0..loads.len())
        .map(|idx| armed[idx / per_block].is_some())
        .collect();

    for block in 0..num_blocks {
        if fired[block] {
            continue;
        }
        let members: Vec<usize> = (block * per_block..(block + 1) * per_block)
            .filter(|idx| *idx < loads.len())
            .collect();
        if !members.iter().all(|idx| live[*idx]) {
            continue;
        }
        // A block that has produced nothing is not behind, it has not started.
        // The two are indistinguishable by in-flight count alone, and firing on
        // the second reading would release the pool before it ever filled. The
        // system this models cannot make that mistake by construction: it
        // re-evaluates a block only when one of that block's own requests has
        // just completed.
        if members.iter().all(|idx| loads[*idx].completed_requests == 0) {
            continue;
        }
        let groups: u32 = members.iter().map(|idx| load[*idx]).sum();
        // Nothing left to move. Retiring here would be harmless but pointless,
        // and it would take workers out of the pool on the strength of a
        // threshold they met by finishing rather than by falling behind. A
        // block already mid-release is the exception: it has nothing left to
        // hand over because it handed it all over, and it still has to retire.
        if groups == 0 && armed[block].is_none() {
            continue;
        }
        // Already leaving: the threshold is not re-read, only the clock. A
        // release that starts does not stop, which is what the real router does
        // — it begins aborting groups and never reconsiders.
        if armed[block].is_none() && groups * group_size >= threshold {
            continue;
        }
        let mut candidates: Vec<usize> = (0..loads.len())
            .filter(|idx| {
                // A worker that has run out of groups is done, even though its
                // GPU is not handed back until its whole block is. Done is
                // read off the tick's opening snapshot, not the running copy,
                // so it is stable across every block that fires this tick: a
                // worker cannot be too finished to receive one block's work and
                // then eligible for the next one's.
                live[*idx]
                    && !leaving[*idx]
                    && loads[*idx].in_flight_groups > 0
                    && !members.contains(idx)
            })
            .collect();
        // Either the last block is standing, or everything else has already
        // finished its own work or is on its way out. All of them mean this
        // block finishes what it holds — and if it was already leaving it stays
        // armed and tries again, which is what the real router does when it
        // fires and finds nowhere to put the work.
        if candidates.is_empty() {
            continue;
        }
        candidates.sort_unstable();
        // Lowest load wins, lowest worker id breaks the tie, and the winner's
        // load rises before the next group is placed — so a level pool is
        // filled round-robin and an uneven one is levelled first.
        let emptiest = |load: &[u32]| {
            *candidates
                .iter()
                .min_by_key(|idx| (load[**idx], **idx))
                .expect("candidate set is non-empty")
        };

        if group_latency == Time::ZERO {
            // No abort loop to model, so there is no partial state to be in:
            // the block hands everything over and retires in one step.
            armed[block] = None;
            fired[block] = true;
            for idx in &members {
                live[*idx] = false;
                leaving[*idx] = true;
            }
            for src in members {
                let mut dsts = Vec::with_capacity(load[src] as usize);
                for _ in 0..load[src] {
                    let dst = emptiest(&load);
                    dsts.push(loads[dst].worker);
                    load[dst] += 1;
                }
                load[src] = 0;
                let retire_to = dsts
                    .first()
                    .copied()
                    .unwrap_or_else(|| loads[emptiest(&load)].worker);
                out.push(MigrationOrder::Scatter {
                    src: loads[src].worker,
                    dsts,
                    retire_to,
                });
            }
            continue;
        }

        // The abort loop. Group `k` of this release leaves at
        // `crossed_at + group_latency * k`, so a block that crosses now hands
        // over its first group now and its last one `latency * (n - 1)` later.
        // Everything it has not handed over yet keeps decoding on it, and every
        // group that has landed is already being worked on by its new host —
        // which is the whole difference from waiting out the loop and then
        // moving the lot.
        let (crossed_at, mut moved) = armed[block].unwrap_or((now, 0));
        for idx in &members {
            leaving[*idx] = true;
        }
        while now.as_ns() >= crossed_at.as_ns() + group_latency.as_ns() * u64::from(moved) {
            // Members in id order, each emptied before the next is touched.
            // Which of the two goes first is not observable — both keep running
            // until the whole block is gone — so the id order is picked for
            // being the reproducible one.
            let Some(src) = members.iter().copied().find(|idx| load[*idx] > 0) else {
                break;
            };
            let dst = emptiest(&load);
            load[src] -= 1;
            load[dst] += 1;
            moved += 1;
            out.push(MigrationOrder::MoveGroup {
                src: loads[src].worker,
                dst: loads[dst].worker,
            });
        }

        if members.iter().any(|idx| load[*idx] > 0) {
            armed[block] = Some((crossed_at, moved));
            continue;
        }
        // The last group is gone: now the block is worth something to training,
        // and only now does it stop being part of the pool. The scatters carry
        // no destinations because there is nothing left to scatter — they exist
        // to retire the workers and redirect anything the trace still sends
        // them.
        armed[block] = None;
        fired[block] = true;
        let retire_to = loads[emptiest(&load)].worker;
        for idx in &members {
            live[*idx] = false;
            out.push(MigrationOrder::Scatter {
                src: loads[*idx].worker,
                dsts: Vec::new(),
                retire_to,
            });
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
/// Returns `None` when nothing should move: no live worker has fallen under the
/// threshold, the only candidate is already empty (draining it would move
/// nothing and still retire it), or there is no second live worker to receive
/// the work.
fn plan_consolidation(
    loads: &[WorkerLoad],
    threshold: u32,
    peaked: &[bool],
) -> Option<MigrationOrder> {
    let src = loads
        .iter()
        .filter(|load| {
            !load.retired
                && peaked[load.worker.0 as usize]
                && load.active_requests > 0
                && load.active_requests < threshold
        })
        .min_by_key(|load| (load.active_requests, load.worker))?;
    let dst = loads
        .iter()
        .filter(|load| !load.retired && load.worker != src.worker)
        .max_by_key(|load| (load.active_requests, std::cmp::Reverse(load.worker)))?;
    Some(MigrationOrder::Consolidate {
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
            in_flight_groups: active,
            completed_requests: 1,
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

    /// Show the policy a full pool first, so the run under test is about a
    /// batch that *fell* below the threshold rather than one that never rose.
    fn decide_after_warmup(
        policy: &mut MigrationTrigger,
        now: Time,
        loads: &[WorkerLoad],
    ) -> Vec<MigrationOrder> {
        let full: Vec<WorkerLoad> = loads
            .iter()
            .map(|load| WorkerLoad {
                active_requests: 32,
                ..*load
            })
            .collect();
        assert!(
            decide(policy, Time::ZERO, &full).is_empty(),
            "a full pool has nothing to consolidate"
        );
        decide(policy, now, loads)
    }

    #[test]
    fn a_pool_that_has_not_filled_yet_is_not_consolidated() {
        // Every worker is "below threshold" while the pool is still ramping
        // up. Firing there would retire machines before the workload arrives.
        let mut policy = MigrationTrigger::active_batch_below(32, Time::ZERO);

        assert!(decide(&mut policy, Time::ZERO, &[load(0, 1), load(1, 1)]).is_empty());
        assert!(decide(&mut policy, Time::from_ms(1.0), &[load(0, 8), load(1, 6)]).is_empty());
        // Worker 1 fills up…
        assert!(decide(&mut policy, Time::from_ms(2.0), &[load(0, 40), load(1, 32)]).is_empty());
        // …and now its decay is a real consolidation.
        assert_eq!(
            decide(&mut policy, Time::from_ms(3.0), &[load(0, 40), load(1, 5)]),
            vec![MigrationOrder::Consolidate {
                src: WorkerId(1),
                dst: WorkerId(0),
            }]
        );
    }

    #[test]
    fn the_emptiest_under_threshold_folds_into_the_fullest() {
        let mut policy = MigrationTrigger::active_batch_below(32, Time::ZERO);

        let orders = decide_after_warmup(
            &mut policy,
            Time::ZERO,
            &[load(0, 40), load(1, 5), load(2, 12)],
        );

        assert_eq!(
            orders,
            vec![MigrationOrder::Consolidate {
                src: WorkerId(1),
                dst: WorkerId(0),
            }]
        );
    }

    #[test]
    fn a_pool_at_or_above_threshold_stays_put() {
        let mut policy = MigrationTrigger::active_batch_below(32, Time::ZERO);

        assert!(decide_after_warmup(&mut policy, Time::ZERO, &[load(0, 32), load(1, 40)]).is_empty());
    }

    #[test]
    fn an_idle_worker_is_not_worth_retiring() {
        // Draining a worker with nothing on it moves no request but still takes
        // it out of the pool, so the pool would shrink on a quiet moment.
        let mut policy = MigrationTrigger::active_batch_below(32, Time::ZERO);

        assert!(decide_after_warmup(&mut policy, Time::ZERO, &[load(0, 0), load(1, 40)]).is_empty());
    }

    #[test]
    fn the_last_live_worker_has_nowhere_to_send_its_work() {
        let mut policy = MigrationTrigger::active_batch_below(32, Time::ZERO);
        let retired = WorkerLoad {
            retired: true,
            ..load(1, 0)
        };

        assert!(decide_after_warmup(&mut policy, Time::ZERO, &[load(0, 5), retired]).is_empty());
    }

    #[test]
    fn a_retired_worker_is_never_a_destination() {
        let mut policy = MigrationTrigger::active_batch_below(32, Time::ZERO);
        let retired = WorkerLoad {
            retired: true,
            ..load(2, 99)
        };

        let orders = decide_after_warmup(&mut policy, Time::ZERO, &[load(0, 5), load(1, 10), retired]);

        assert_eq!(
            orders,
            vec![MigrationOrder::Consolidate {
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

        assert_eq!(decide_after_warmup(&mut policy, Time::from_ms(10.0), &loads).len(), 1);
        assert!(decide(&mut policy, Time::from_ms(59.0), &loads).is_empty());
        assert_eq!(decide(&mut policy, Time::from_ms(60.0), &loads).len(), 1);
    }

    #[test]
    fn equal_loads_break_ties_on_the_worker_id_in_both_directions() {
        // Determinism, not fairness: the same snapshot must always produce the
        // same order, so the lowest id drains and the lowest id receives.
        let mut policy = MigrationTrigger::active_batch_below(32, Time::ZERO);

        let orders = decide_after_warmup(
            &mut policy,
            Time::ZERO,
            &[load(0, 7), load(1, 7), load(2, 7)],
        );

        assert_eq!(
            orders,
            vec![MigrationOrder::Consolidate {
                src: WorkerId(0),
                dst: WorkerId(1),
            }]
        );
    }

    // ── TrainGroupSamplesBelow ────────────────────────────────────────────────

    /// Workers 2k and 2k+1 form train group k; `groups` is prompt groups held.
    fn block_load(worker: u16, groups: u32) -> WorkerLoad {
        WorkerLoad {
            worker: WorkerId(worker),
            queued_requests: 0,
            active_requests: groups * 8,
            in_flight_groups: groups,
            completed_requests: 1,
            retired: false,
        }
    }

    fn release(policy: &mut MigrationTrigger, loads: &[WorkerLoad]) -> Vec<MigrationOrder> {
        decide(policy, Time::ZERO, loads)
    }

    /// Every worker an order sends work to, in order. A retiring `Scatter` with
    /// no groups left contributes nothing — `retire_to` is a redirect, not a
    /// delivery.
    fn destinations(orders: &[MigrationOrder]) -> Vec<WorkerId> {
        orders
            .iter()
            .flat_map(|order| match order {
                MigrationOrder::Scatter { dsts, .. } => dsts.clone(),
                MigrationOrder::Consolidate { dst, .. } | MigrationOrder::MoveGroup { dst, .. } => {
                    vec![*dst]
                }
            })
            .collect()
    }

    #[test]
    fn a_train_group_under_the_threshold_scatters_onto_the_emptiest_workers() {
        // Block 0 holds 2 groups = 16 samples, under B=32. Its two groups go to
        // the two emptiest eligible workers, and the first placement raises
        // that worker's load so the second group lands elsewhere.
        let mut policy = MigrationTrigger::train_group_samples_below(32, 8, 2, Time::ZERO);
        let loads = [
            block_load(0, 1),
            block_load(1, 1),
            block_load(2, 5),
            block_load(3, 5),
            block_load(4, 9),
            block_load(5, 9),
            block_load(6, 9),
            block_load(7, 9),
        ];

        assert_eq!(
            release(&mut policy, &loads),
            vec![
                MigrationOrder::Scatter {
                    src: WorkerId(0),
                    dsts: vec![WorkerId(2)],
                    retire_to: WorkerId(2),
                },
                MigrationOrder::Scatter {
                    src: WorkerId(1),
                    dsts: vec![WorkerId(3)],
                    retire_to: WorkerId(3),
                },
            ]
        );
    }

    #[test]
    fn the_threshold_counts_samples_not_groups() {
        // 3 groups is under 32 only because a group is worth 8 samples; read as
        // groups it would be 3 and every block would fire immediately.
        let mut policy = MigrationTrigger::train_group_samples_below(32, 8, 2, Time::ZERO);
        let under = [
            block_load(0, 2),
            block_load(1, 1),
            block_load(2, 9),
            block_load(3, 9),
        ];
        assert_eq!(release(&mut policy, &under).len(), 2, "24 samples < 32");

        let mut policy = MigrationTrigger::train_group_samples_below(32, 8, 2, Time::ZERO);
        let at = [
            block_load(0, 2),
            block_load(1, 2),
            block_load(2, 9),
            block_load(3, 9),
        ];
        assert!(release(&mut policy, &at).is_empty(), "32 samples is not below 32");
    }

    #[test]
    fn a_block_already_released_this_tick_is_not_the_next_one_s_destination() {
        // Block 0 goes first and its group lands on worker 2, which lifts block
        // 1 to 24 samples — still under 32, so block 1 goes too. Its work must
        // reach workers 4 and 5: ranking on the snapshot alone would rate the
        // just-emptied workers 0 and 1 as the best destinations in the pool and
        // strand the work on a pair nothing routes to again.
        let mut policy = MigrationTrigger::train_group_samples_below(32, 8, 2, Time::ZERO);
        let loads = [
            block_load(0, 1),
            block_load(1, 0),
            block_load(2, 1),
            block_load(3, 1),
            block_load(4, 40),
            block_load(5, 40),
        ];

        let orders = release(&mut policy, &loads);

        let dsts = destinations(&orders);
        assert!(
            dsts.iter().all(|w| w.0 != 0 && w.0 != 1),
            "workers 0 and 1 were released this same tick: {dsts:?}"
        );
        let released: Vec<WorkerId> = orders.iter().map(MigrationOrder::src).collect();
        assert_eq!(
            released,
            vec![WorkerId(0), WorkerId(1), WorkerId(2), WorkerId(3)],
            "both blocks hand back both of their workers"
        );
    }

    #[test]
    fn a_worker_that_has_finished_its_own_work_takes_none_of_anyone_else_s() {
        // Worker 2 is empty and its block partner is not, so its GPU is still
        // held — but it is done, and done workers are not sent more work. The
        // load ranking would otherwise rate it the single most attractive
        // destination in the pool.
        let mut policy = MigrationTrigger::train_group_samples_below(32, 8, 2, Time::ZERO);
        let loads = [
            block_load(0, 1),
            block_load(1, 1),
            block_load(2, 0),
            block_load(3, 9),
            block_load(4, 9),
            block_load(5, 9),
        ];

        let orders = release(&mut policy, &loads);

        let dsts = destinations(&orders);
        assert_eq!(dsts.len(), 2, "block 0 holds two groups");
        assert!(
            dsts.iter().all(|w| w.0 != 2),
            "worker 2 has no groups left to finish: {dsts:?}"
        );
    }

    #[test]
    fn a_block_with_only_finished_workers_left_keeps_its_own_work() {
        // Every other worker is done. The trigger fires and finds nowhere to
        // put the work, so the block stays live and finishes it — the same
        // shape as the last block standing.
        let mut policy = MigrationTrigger::train_group_samples_below(32, 8, 2, Time::ZERO);
        let loads = [
            block_load(0, 1),
            block_load(1, 1),
            block_load(2, 0),
            block_load(3, 0),
        ];

        assert!(release(&mut policy, &loads).is_empty());
    }

    #[test]
    fn taking_in_work_can_lift_a_block_back_over_the_threshold() {
        // Block 1 is under the threshold on the raw snapshot, but block 0's two
        // groups land on it first and put it back over. It keeps its workers —
        // the same self-correction the real system gets from re-reading load
        // after each migration rather than planning the whole tick up front.
        let mut policy = MigrationTrigger::train_group_samples_below(32, 8, 2, Time::ZERO);
        let loads = [
            block_load(0, 1),
            block_load(1, 1),
            block_load(2, 1),
            block_load(3, 1),
            block_load(4, 40),
            block_load(5, 40),
        ];

        let orders = release(&mut policy, &loads);

        assert_eq!(
            orders.iter().map(MigrationOrder::src).collect::<Vec<_>>(),
            vec![WorkerId(0), WorkerId(1)],
            "only block 0 is released; block 1 is at 32 samples once it takes the work"
        );
    }

    #[test]
    fn a_train_group_is_handed_back_only_once() {
        let mut policy = MigrationTrigger::train_group_samples_below(32, 8, 2, Time::ZERO);
        let loads = [
            block_load(0, 1),
            block_load(1, 1),
            block_load(2, 9),
            block_load(3, 9),
        ];

        assert_eq!(release(&mut policy, &loads).len(), 2);
        // Same snapshot again — the pool has not marked the pair retired yet,
        // but the trigger must not re-issue the release.
        assert!(release(&mut policy, &loads).is_empty());
    }

    #[test]
    fn a_release_hands_its_groups_over_one_at_a_time() {
        // Two groups at 700 ms each: one leaves at the crossing, the other
        // 700 ms later, and only then does the block retire. In between it is
        // still a running engine holding the group it has not handed over yet,
        // which is the whole difference from waiting out the loop and then
        // moving both — the destinations get the first group 700 ms earlier and
        // the source is busy either way.
        let mut policy =
            MigrationTrigger::train_group_samples_below(32, 8, 2, Time::from_ms(700.0));
        let crossing = [
            block_load(0, 1),
            block_load(1, 1),
            block_load(2, 9),
            block_load(3, 9),
        ];

        assert_eq!(
            decide(&mut policy, Time::ZERO, &crossing),
            vec![MigrationOrder::MoveGroup {
                src: WorkerId(0),
                dst: WorkerId(2),
            }],
        );

        // The pool ticks again with the first group gone. The second is not due.
        let after_first = [
            block_load(0, 0),
            block_load(1, 1),
            block_load(2, 10),
            block_load(3, 9),
        ];
        assert!(decide(&mut policy, Time::from_ms(100.0), &after_first).is_empty());

        assert_eq!(
            decide(&mut policy, Time::from_ms(700.0), &after_first),
            vec![
                MigrationOrder::MoveGroup {
                    src: WorkerId(1),
                    dst: WorkerId(3),
                },
                MigrationOrder::Scatter {
                    src: WorkerId(0),
                    dsts: vec![],
                    retire_to: WorkerId(2),
                },
                MigrationOrder::Scatter {
                    src: WorkerId(1),
                    dsts: vec![],
                    retire_to: WorkerId(2),
                },
            ],
            "the last group out is what retires the block",
        );

        // One-shot still holds once it has actually gone.
        let gone = [
            block_load(0, 0),
            block_load(1, 0),
            block_load(2, 10),
            block_load(3, 10),
        ];
        assert!(decide(&mut policy, Time::from_ms(1400.0), &gone).is_empty());
    }

    #[test]
    fn a_block_on_its_way_out_takes_no_new_work() {
        // Block 0 is mid-release and, having handed most of its work over, is
        // the emptiest pair in the pool. When block 1 crosses it must skip them
        // anyway: work landing on a block that is leaving would only have to
        // move a second time, and its whole point is to stop taking work.
        let mut policy =
            MigrationTrigger::train_group_samples_below(32, 8, 2, Time::from_ms(700.0));
        let crossing = [
            block_load(0, 2),
            block_load(1, 1),
            block_load(2, 3),
            block_load(3, 2),
            block_load(4, 9),
            block_load(5, 9),
        ];
        assert_eq!(
            decide(&mut policy, Time::ZERO, &crossing),
            vec![MigrationOrder::MoveGroup {
                src: WorkerId(0),
                dst: WorkerId(3),
            }],
            "only block 0 is under the threshold",
        );

        // Block 1 has drained under the threshold too, while block 0 is still
        // waiting out its own loop.
        let later = [
            block_load(0, 1),
            block_load(1, 1),
            block_load(2, 1),
            block_load(3, 0),
            block_load(4, 9),
            block_load(5, 9),
        ];
        let orders = decide(&mut policy, Time::from_ms(100.0), &later);

        assert_eq!(
            orders.iter().map(MigrationOrder::src).collect::<Vec<_>>(),
            vec![WorkerId(2), WorkerId(2), WorkerId(3)],
            "block 1 moves its one group and then retires; block 0's next is not due",
        );
        let dsts = destinations(&orders);
        assert!(
            dsts.iter().all(|w| w.0 >= 4),
            "block 0 is on its way out, so only block 2 may receive: {dsts:?}"
        );
    }

    #[test]
    fn the_last_train_group_keeps_its_work() {
        let mut policy = MigrationTrigger::train_group_samples_below(32, 8, 2, Time::ZERO);
        let gone = |worker: u16| WorkerLoad {
            retired: true,
            ..block_load(worker, 0)
        };

        assert!(release(
            &mut policy,
            &[block_load(0, 1), block_load(1, 1), gone(2), gone(3)]
        )
        .is_empty());
    }

    #[test]
    fn a_finished_train_group_is_not_released() {
        // Zero in flight also satisfies "below the threshold", but it means the
        // block finished its share rather than fell behind, and retiring on
        // that reading would shrink the pool for a reason the knob is not about.
        let mut policy = MigrationTrigger::train_group_samples_below(32, 8, 2, Time::ZERO);

        assert!(release(
            &mut policy,
            &[block_load(0, 0), block_load(1, 0), block_load(2, 9), block_load(3, 9)]
        )
        .is_empty());
    }

    #[test]
    fn an_uneven_pair_still_releases_both_of_its_workers() {
        // Only one worker of the block has anything left. Both must go: a block
        // is useful to training only when all of its workers are free.
        let mut policy = MigrationTrigger::train_group_samples_below(32, 8, 2, Time::ZERO);
        let loads = [
            block_load(0, 3),
            block_load(1, 0),
            block_load(2, 9),
            block_load(3, 9),
        ];

        assert_eq!(
            release(&mut policy, &loads),
            vec![
                MigrationOrder::Scatter {
                    src: WorkerId(0),
                    dsts: vec![WorkerId(2), WorkerId(3), WorkerId(2)],
                    retire_to: WorkerId(2),
                },
                MigrationOrder::Scatter {
                    src: WorkerId(1),
                    dsts: vec![],
                    retire_to: WorkerId(3),
                },
            ]
        );
    }
}
