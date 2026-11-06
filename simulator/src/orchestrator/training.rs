//! `TrainingPool` — the L6 half of streaming RL: which finished prompt groups a
//! freed block picks up, and when it may pick them up at all.
//!
//! The generation side of this deployment hands blocks of engines back to the
//! trainer as their work drains (`migration.rs`). This is what the trainer does
//! with them. Three rules, all read off the measured run rather than assumed:
//!
//! 1. **A group enters the queue when its slowest sample lands.** A training
//!    step consumes a whole prompt group — all of its samples — so a group that
//!    is seven-eighths done is worth nothing to the trainer.
//! 2. **A block starts training the instant both of its engines are empty.**
//!    Verified per pair on the measured r0/B=96 rollout: the four blocks freed
//!    at 225.4 / 303.3 / 359.7 / 468.1 s and took their first chunk at
//!    225.4 / 303.3 / 358.4 / 468.3 s.
//! 3. **The queue is shared, so the block that frees first trains more.** That
//!    is work stealing, and it is the reason a staggered release is worth
//!    anything: on the same rollout one block ran 30 chunks and another 2, and
//!    all four still finished within ~12 s of each other.
//!
//! The grab policy is slime's `graduated_tail_split`, reproduced from
//! `slime/ray/grab_policy.py` and confirmed against that run's 1,283 logged
//! grabs — which come out to exactly 48 grabs of 2, then 4 of 4, then 4 of 2,
//! then 8 of 1, per rollout of 128 groups. See [`TrainingConfig::bulk_grab`].
//!
//! What a chunk *costs* is not here: that is
//! [`TrainChunkWorker`](crate::worker::TrainChunkWorker), one per block.

use std::collections::VecDeque;
use std::path::PathBuf;

use crate::common::{Time, WorkerId};
use crate::worker::{
    IterWorker, TrainChunkCost, TrainChunkWorker, TrainWorkerEvent, TrainWorkerMsg,
};

use super::migration::WorkerLoad;

/// `cost_log` stream tag for the training blocks. Distinct from the inference
/// pool's `main` because the row key is `(pool_tag, worker_id)` and both sides
/// number their workers from zero.
pub const TRAIN_POOL_TAG: &str = "main.train";

/// Everything the training side needs, all of it measured.
#[derive(Debug, Clone, Copy)]
pub struct TrainingConfig {
    /// Engines per training block — the same topology the migration trigger
    /// releases in, because it is the same blocks.
    pub workers_per_block: u16,
    /// Samples per prompt group. The trainer's queue item is a whole group, so
    /// this is the same topology number the migration trigger counts in; the
    /// flow reads it to install the group ledger when training runs without a
    /// migration policy.
    pub group_size: u32,
    /// Per-chunk cost model (calibrated; see [`TrainChunkCost`]).
    pub cost: TrainChunkCost,
    /// Groups a free block takes per grab away from the tail — slime's
    /// `max_items_per_grab`. The measured run used 2.
    pub bulk_grab: u16,
    /// Tail ladder base, slime's `TAIL_SINGLE_ITEM_THRESHOLD` (8). With `T`
    /// groups left to hand out over the whole rollout, the cap steps down
    /// `bulk → 4 → 2 → 1` as the remainder falls through `4T → 2T → T`, so the
    /// heavy tail fans out over all blocks instead of one block swallowing it.
    /// Zero disables the ladder (every grab is a bulk grab).
    pub tail_threshold: u32,
    /// Groups the rollout will produce — slime's `expected_items_per_rollout`,
    /// which is what its remainder is counted against. A configured number
    /// there and here, because the queue cannot know the future. Zero disables
    /// the ladder along with `tail_threshold`.
    pub expected_groups: u32,
}

/// A finished prompt group waiting to be trained on: the token count of each of
/// its samples, which is everything a training step needs from it.
#[derive(Debug, Clone)]
struct PendingGroup {
    samples: Vec<u32>,
}

/// The training side of one deployment: a block per training group, a shared
/// queue of finished prompt groups, and the grab policy between them.
pub struct TrainingPool {
    blocks: Vec<TrainChunkWorker>,
    /// The shared queue. One `VecDeque` for the whole pool *is* the work
    /// stealing — a block takes from the front whenever it is free, so a block
    /// that frees early keeps taking while the others are still generating.
    pending: VecDeque<PendingGroup>,
    /// Per block: it has held generation work at some point. Without this every
    /// block would look free at `t = 0` — nothing has arrived yet — and the
    /// whole pool would "start training" before the run began.
    worked: Vec<bool>,
    /// Per block: every engine in it is empty, so the trainer owns it now.
    free: Vec<bool>,
    workers_per_block: usize,
    bulk_grab: usize,
    tail_threshold: u32,
    expected_groups: u32,
    /// Groups handed out so far — slime's `items_grabbed_so_far`, the counter
    /// the tail ladder reads. Deliberately not `pending.len()`: the ladder is
    /// about how much of the rollout is left, not how much happens to be
    /// queued at this instant.
    grabbed: u32,
    next_chunk: u32,
    /// Reused per-tick event sink.
    events: Vec<TrainWorkerEvent>,
    chunks_completed: u64,
    groups_completed: u32,
    /// Groups each block ended up training. The shape of the work-stealing
    /// split, which is the thing a release policy is actually judged on.
    groups_per_block: Vec<u32>,
    first_start: Option<Time>,
    last_end: Option<Time>,
}

impl TrainingPool {
    /// One block per `workers_per_block` engines of the inference pool.
    pub fn new(cfg: TrainingConfig, num_workers: usize, log_dir: Option<PathBuf>) -> Self {
        let per_block = usize::from(cfg.workers_per_block);
        assert!(per_block > 0, "a training block needs at least one engine");
        assert_eq!(
            num_workers % per_block,
            0,
            "a pool of {num_workers} engines does not divide into blocks of {per_block}",
        );
        let num_blocks = num_workers / per_block;
        let blocks = (0..num_blocks)
            .map(|idx| {
                TrainChunkWorker::new(
                    WorkerId(idx as u16),
                    cfg.cost,
                    log_dir.clone(),
                    TRAIN_POOL_TAG,
                )
            })
            .collect();
        Self {
            blocks,
            pending: VecDeque::new(),
            worked: vec![false; num_blocks],
            free: vec![false; num_blocks],
            workers_per_block: per_block,
            bulk_grab: usize::from(cfg.bulk_grab).max(1),
            tail_threshold: cfg.tail_threshold,
            expected_groups: cfg.expected_groups,
            grabbed: 0,
            next_chunk: 0,
            events: Vec::new(),
            chunks_completed: 0,
            groups_completed: 0,
            groups_per_block: vec![0; num_blocks],
            first_start: None,
            last_end: None,
        }
    }

    /// A prompt group's last sample has landed: queue it, carrying the token
    /// counts the trainer will run over.
    pub fn admit(&mut self, samples: Vec<u32>) {
        self.pending.push_back(PendingGroup { samples });
    }

    /// Re-read which blocks the trainer owns. A block is the trainer's once
    /// every engine in it holds no in-flight group — which covers both ways an
    /// engine empties, the router releasing it mid-rollout and it simply
    /// finishing what it had. The two are indistinguishable from here, and to
    /// the trainer they are the same event.
    pub fn observe(&mut self, loads: &[WorkerLoad]) {
        for (block, free) in self.free.iter_mut().enumerate() {
            let engines = &loads[block * self.workers_per_block..][..self.workers_per_block];
            if engines
                .iter()
                .any(|load| load.in_flight_groups > 0 || load.completed_requests > 0)
            {
                self.worked[block] = true;
            }
            *free = self.worked[block] && engines.iter().all(|load| load.in_flight_groups == 0);
        }
    }

    /// Advance the training side one tick: land whatever finished, hand the
    /// freed blocks their next chunk, and start it.
    ///
    /// The loop is what makes a chunk boundary free of slack — a block that
    /// lands a chunk at `now` takes the next one at `now`, not a tick later.
    /// It terminates because every pass either dispatches to a block (of which
    /// there are finitely many, each holding one chunk) or stops.
    pub fn tick(&mut self, now: Time) {
        for _ in 0..=self.blocks.len() {
            self.collect(now);
            if !self.dispatch() {
                break;
            }
        }
    }

    fn collect(&mut self, now: Time) {
        let mut events = std::mem::take(&mut self.events);
        events.clear();
        for block in &mut self.blocks {
            block.tick(now, &mut events);
        }
        for event in events.drain(..) {
            let TrainWorkerEvent::ChunkComplete {
                worker,
                groups,
                started,
                ended,
                ..
            } = event;
            self.chunks_completed += 1;
            self.groups_completed += u32::from(groups);
            self.groups_per_block[worker.0 as usize] += u32::from(groups);
            self.first_start = Some(self.first_start.map_or(started, |at| at.min(started)));
            self.last_end = Some(self.last_end.map_or(ended, |at| at.max(ended)));
        }
        self.events = events;
    }

    /// Hand one chunk to every block that is free and idle. Returns whether
    /// anything was handed over.
    fn dispatch(&mut self) -> bool {
        let mut dispatched = false;
        for block in 0..self.blocks.len() {
            let status = self.blocks[block].status();
            // Idle means holding nothing at all, started or not. `collect` runs
            // first every pass, so a block handed a chunk has already started it
            // by the time this is asked again — but a block is allowed to hold
            // exactly one chunk, and the worker panics rather than queue a
            // second, so the check names both halves.
            if !self.free[block] || status.active_requests + status.queued_requests > 0 {
                continue;
            }
            let take = self.grab_cap().min(self.pending.len());
            if take == 0 {
                continue;
            }
            let mut samples = Vec::new();
            for _ in 0..take {
                let group = self.pending.pop_front().expect("take <= pending.len()");
                samples.extend_from_slice(&group.samples);
            }
            self.grabbed += take as u32;
            let chunk_id = self.next_chunk;
            self.next_chunk += 1;
            self.blocks[block].enqueue(TrainWorkerMsg::Chunk {
                chunk_id,
                groups: take as u16,
                samples,
            });
            dispatched = true;
        }
        dispatched
    }

    /// Groups this grab may take: the bulk cap until the rollout's remainder
    /// falls into the tail ladder, then `4 → 2 → 1`.
    fn grab_cap(&self) -> usize {
        if self.tail_threshold == 0 || self.expected_groups == 0 {
            return self.bulk_grab;
        }
        let remaining = self.expected_groups.saturating_sub(self.grabbed);
        if remaining == 0 {
            return self.bulk_grab;
        }
        for step in 0..3u32 {
            if remaining <= self.tail_threshold << step {
                return 1usize << step;
            }
        }
        self.bulk_grab
    }

    /// Whether the trainer still owes work — queued groups or a chunk in
    /// flight. The run loop keeps ticking while this holds, even with every
    /// request complete.
    pub fn outstanding(&self) -> bool {
        !self.pending.is_empty()
            || self
                .blocks
                .iter()
                .any(|block| block.status().active_requests + block.status().queued_requests > 0)
    }

    /// Monotone progress signal for the run loop's stuck watchdog.
    pub fn chunks_completed(&self) -> u64 {
        self.chunks_completed
    }

    /// Groups each block trained — the work-stealing split.
    pub fn groups_per_block(&self) -> &[u32] {
        &self.groups_per_block
    }

    /// Groups finished but not yet grabbed, and the tokens they carry — the
    /// queue depth a streaming trainer is trying to keep above zero.
    pub fn queued(&self) -> (usize, u64) {
        let tokens = self
            .pending
            .iter()
            .flat_map(|group| group.samples.iter())
            .map(|len| u64::from(*len))
            .sum();
        (self.pending.len(), tokens)
    }

    /// First chunk start → last chunk end: the run's `training_time`, in the
    /// same sense `report.json` uses it.
    pub fn window(&self) -> Option<(Time, Time)> {
        Some((self.first_start?, self.last_end?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost() -> TrainChunkCost {
        TrainChunkCost {
            tokens_per_s: 10_000.0,
            overhead: Time::from_ms(0.0),
        }
    }

    fn config(bulk_grab: u16, expected_groups: u32) -> TrainingConfig {
        TrainingConfig {
            workers_per_block: 2,
            group_size: 8,
            cost: cost(),
            bulk_grab,
            tail_threshold: 8,
            expected_groups,
        }
    }

    /// `n` engines, all idle and all having worked — the state after generation
    /// has drained.
    fn loads(in_flight: &[u32]) -> Vec<WorkerLoad> {
        in_flight
            .iter()
            .enumerate()
            .map(|(idx, groups)| WorkerLoad {
                worker: WorkerId(idx as u16),
                queued_requests: 0,
                active_requests: 0,
                in_flight_groups: *groups,
                completed_requests: 1,
                retired: false,
            })
            .collect()
    }

    fn pool(bulk_grab: u16, expected_groups: u32, engines: usize) -> TrainingPool {
        TrainingPool::new(config(bulk_grab, expected_groups), engines, None)
    }

    fn fill(pool: &mut TrainingPool, groups: usize, tokens_each: u32) {
        for _ in 0..groups {
            pool.admit(vec![tokens_each]);
        }
    }

    /// A block trains the moment both of its engines are empty, and not before.
    #[test]
    fn a_pair_starts_the_moment_both_of_its_engines_are_empty() {
        let mut pool = pool(2, 0, 4);
        fill(&mut pool, 4, 10_000);

        // Block 0 still holds a group on its second engine.
        pool.observe(&loads(&[0, 1, 1, 1]));
        pool.tick(Time::ZERO);
        assert_eq!(pool.pending.len(), 4, "no block is free yet");

        pool.observe(&loads(&[0, 0, 1, 1]));
        pool.tick(Time::ZERO);
        assert_eq!(pool.pending.len(), 2, "block 0 took its bulk grab");
        assert_eq!(pool.blocks[0].status().active_requests, 1);
        assert_eq!(pool.blocks[1].status().active_requests, 0);
    }

    /// At `t = 0` nothing has arrived, so every engine reads as empty. A pool
    /// that trusted that would train the whole rollout before it started.
    #[test]
    fn an_empty_pool_does_not_start_training_before_it_has_had_work() {
        let mut pool = pool(2, 0, 4);
        fill(&mut pool, 2, 10_000);
        let cold: Vec<WorkerLoad> = loads(&[0, 0, 0, 0])
            .into_iter()
            .map(|load| WorkerLoad {
                completed_requests: 0,
                ..load
            })
            .collect();
        pool.observe(&cold);
        pool.tick(Time::ZERO);
        assert_eq!(pool.pending.len(), 2, "nothing has generated anything yet");
        assert_eq!(pool.groups_completed, 0);
        assert!(pool.blocks.iter().all(|b| b.status().active_requests == 0));
    }

    /// One shared queue: the block that frees first keeps taking while the
    /// other is still generating, so it ends up with strictly more of the work.
    #[test]
    fn the_queue_is_shared_so_the_pair_that_frees_first_takes_more() {
        let mut pool = pool(2, 0, 4);
        fill(&mut pool, 12, 10_000); // 1 s per group at 10k tok/s

        // Block 0 free from the start; block 1 stays busy for 4 s.
        let mut at = Time::ZERO;
        for _ in 0..4_000 {
            pool.observe(&loads(&[0, 0, 1, 1]));
            pool.tick(at);
            at = at + Time::from_ms(1.0);
        }
        assert_eq!(
            pool.groups_per_block(),
            [2, 0],
            "only block 0 has landed anything while block 1 still generates"
        );

        for _ in 0..8_000 {
            pool.observe(&loads(&[0, 0, 0, 0]));
            pool.tick(at);
            at = at + Time::from_ms(1.0);
        }
        assert!(
            !pool.outstanding(),
            "the queue drains once both blocks work"
        );
        assert_eq!(pool.groups_completed, 12);
        assert_eq!(
            pool.groups_per_block(),
            [8, 4],
            "the block that freed first trained twice as much"
        );
    }

    /// The measured grab sequence, reproduced exactly: 128 groups against a
    /// bulk cap of 2 come out as 48 grabs of 2, then 4 of 4, then 4 of 2, then
    /// 8 of 1 — which is what that run's 1,283 logged grabs average to over its
    /// 20 rollouts.
    #[test]
    fn the_tail_ladder_matches_the_measured_grab_sequence() {
        let mut pool = pool(2, 128, 2);
        let mut sizes = Vec::new();
        for _ in 0..128 {
            let take = pool.grab_cap();
            pool.grabbed += take as u32;
            sizes.push(take);
            if pool.grabbed >= 128 {
                break;
            }
        }
        let count = |n: usize| sizes.iter().filter(|size| **size == n).count();
        assert_eq!(sizes.iter().sum::<usize>(), 128);
        assert_eq!((count(2), count(4), count(1)), (52, 4, 8));
        assert_eq!(sizes.len(), 64, "64 grabs, as logged");
        // The ladder steps down, never back up.
        assert_eq!(
            &sizes[46..],
            &[2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
        );
    }

    /// With the ladder off, every grab is a bulk grab.
    #[test]
    fn without_an_expected_count_every_grab_is_a_bulk_grab() {
        let pool = pool(2, 0, 2);
        assert_eq!(pool.grab_cap(), 2);
    }
}
