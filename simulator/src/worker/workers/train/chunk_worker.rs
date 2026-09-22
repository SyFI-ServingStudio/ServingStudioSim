//! `TrainChunkWorker` — the L5 shell that runs one RL training chunk.
//!
//! Streaming RL borrows its engines from the trainer: a block of GPUs generates
//! until the router hands it back, then trains on whatever prompt groups have
//! finished. This worker is the second half of that — it holds a block for the
//! duration of one *chunk* (the batch of prompt groups a free block grabbed) and
//! reports when the block is free again. Which groups go into a chunk is an L6
//! scheduling decision ([`TrainingPool`](crate::orchestrator::TrainingPool));
//! what the chunk costs is here.
//!
//! **The cost is calibrated end to end, not composed from L1 kernels.** There are
//! no training rows in `profile.db` — no backward pass, no optimizer step, no
//! gradient all-reduce — so a CostTree built out of the forward kernels we do
//! have would be a forward pass wearing a training costume. The honest form is a
//! measured rate plus a measured per-chunk constant:
//!
//! ```text
//! chunk_total_s = tokens / tokens_per_s + chunk_overhead
//! ```
//!
//! fitted over the five fixed-B slime arms' 100 rollouts
//! (`tools/slime-b-sweep/`), against `busy_gpu_s` halved — a block is a TP=2
//! pair — which is the trainer's own accounting of how long its pairs were
//! held, per rollout:
//!
//! ```text
//! busy_pair_s = trace_tokens / 9,296 + 1.227 per chunk
//! ```
//!
//! mean error +0.1%, |mean| 1.9% over the 100 cells. **Not** the obvious fit
//! over that rollout's `chunk_total_s` records, for two reasons:
//!
//! * A pair is held for more than the fwd/bwd chunks it logs. Interior idle
//!   inside the training span is 0.5%, so "held continuously" is the right
//!   shape, and being held is what this models.
//! * `train_metrics` does not log every chunk: about 60 of the 64 grabs a
//!   rollout makes get written, taking ~6% of the tokens with them. That is
//!   why a rollout's logged `total_tokens` reads as 94% of the trace's
//!   `input_len + output_len` — the trace is right and the log is short, which
//!   three B=128 cells settle by matching to the exact token.
//!
//! Fitting `chunk_total_s` instead leaves the simulated training span 10–20%
//! short, which is what the end-to-end sweep showed. The per-chunk fit on the
//! logged numbers — 10,176 tok/s + 915 ms, r = 0.9884 over 6,053 chunks — is a
//! good description of a chunk and is NOT what a preset should carry.
//!
//! The constants are preset knobs, not literals, precisely because they do not
//! survive a change of model, parallelism, or hardware.
//!
//! The `cost_log` row it writes therefore carries `slot_flops` / `slot_bytes` of
//! zero — the established "not measured" sentinel — under a hand-written one-leaf
//! `train` section. See [`train_cost_manifest`].

use std::path::PathBuf;

use crate::common::{Time, WorkerId};
use crate::log::GroupInputLog;
use crate::timing::{
    CostManifest, CostManifestDoc, CoverageFlags, FlatCostNode, LeafDesc, LeafMetrics, Metrics4,
};
use crate::worker::cost_buffers::GroupLogSource;
use crate::worker::{CostBuffers, IterWorker, TrainWorkerEvent, TrainWorkerMsg, WorkerStatus};

/// `cost_log` section name for a training chunk. The first section in the
/// repository that is not compiled from a CostTree, which the open-vocabulary
/// `section` column was built to allow.
const TRAIN_SECTION: &str = "train";

/// What one training chunk costs: a token rate plus a fixed per-chunk overhead.
///
/// Both are **measured**, in the units the simulation clock runs in. If that
/// clock is itself uncalibrated (a deployment whose generation side carries an
/// end-to-end correction factor applied downstream), the preset is where the
/// conversion belongs — this struct takes the numbers at face value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainChunkCost {
    /// Tokens per simulated second of the fused forward+backward.
    pub tokens_per_s: f64,
    /// Per-chunk constant: the framework work a grab pays regardless of size.
    pub overhead: Time,
}

impl TrainChunkCost {
    /// How long a block is held by a chunk of `tokens` tokens.
    pub fn of(&self, tokens: u64) -> Time {
        assert!(
            self.tokens_per_s > 0.0,
            "a training chunk needs a positive token rate, got {}",
            self.tokens_per_s
        );
        Time::from_ms(tokens as f64 / self.tokens_per_s * 1000.0) + self.overhead
    }
}

/// The one-leaf `train` section a training worker's `cost_log` rows are read
/// through. Hand-written rather than compiled: the leaf stands for one
/// calibrated forward+backward, so its `kernel_config` records the calibration
/// itself — the only thing about this "kernel" that is true.
pub fn train_cost_manifest(cost: TrainChunkCost) -> CostManifestDoc {
    CostManifestDoc::single(
        TRAIN_SECTION,
        CostManifest {
            slots: vec![LeafDesc {
                name: "train.chunk".to_owned(),
                kind: "calibrated".to_owned(),
                kernel_config: serde_json::json!({
                    "tokens_per_s": cost.tokens_per_s,
                    "overhead_ms": cost.overhead.as_ms(),
                    "source": "slime fixed-B sweep, fitted against trace tokens",
                }),
            }],
            nodes: vec![FlatCostNode::Leaf(0)],
            node_labels: vec![None],
        },
    )
}

/// A chunk the pool has handed over, before or during execution.
#[derive(Clone, Debug)]
struct Chunk {
    chunk_id: u32,
    groups: u16,
    /// Per-sample token counts (`prompt + emitted`), which sum to the chunk's
    /// token total. Kept as the list rather than the sum so the `cost_log` row
    /// records the shape the trainer actually saw.
    samples: Vec<u32>,
}

impl Chunk {
    fn tokens(&self) -> u64 {
        self.samples.iter().map(|len| u64::from(*len)).sum()
    }
}

/// A chunk in flight, with the block held until `end`.
#[derive(Clone, Debug)]
struct Running {
    chunk: Chunk,
    started: Time,
    end: Time,
}

/// One training block (`workers_per_train_group` engines, TP-joined) modeled as
/// a single worker: it runs one chunk at a time and is free the moment that
/// chunk lands.
///
/// Implements [`IterWorker`] — the trait asks only for an id, an enqueue, a tick
/// and a load reading, none of which mention KV, admission or a model, so a
/// worker with no requests at all fits it without widening anything.
pub struct TrainChunkWorker {
    id: WorkerId,
    cost: TrainChunkCost,
    /// Handed over but not yet started. At most one: L6 dispatches to a block
    /// only while it is idle.
    pending: Option<Chunk>,
    running: Option<Running>,
    /// The `cost_log` writer + its scratch. `gpu_time_multiplier` is 1.0: the
    /// calibration is already wall time, so there is no inter-kernel gap left to
    /// model on top of it.
    buffers: CostBuffers,
}

impl TrainChunkWorker {
    pub fn new(
        id: WorkerId,
        cost: TrainChunkCost,
        log_dir: Option<PathBuf>,
        pool_tag: &'static str,
    ) -> Self {
        Self {
            id,
            cost,
            pending: None,
            running: None,
            buffers: CostBuffers::new(log_dir, pool_tag, id, &train_cost_manifest(cost), 1.0),
        }
    }

    /// Start `chunk` at `now`, writing its `cost_log` row.
    fn start(&mut self, now: Time, chunk: Chunk) -> Time {
        let tokens = chunk.tokens();
        let held = self.cost.of(tokens);
        let metrics = LeafMetrics {
            m: Metrics4 {
                time_ms: held.as_ms() as f32,
                // Not measured — the same sentinel a profile row without a
                // tflops rate writes. A training FLOP count is derivable from
                // the model, but nothing here measured one.
                flops: 0.0,
                bytes: 0.0,
                energy_j: 0.0,
            },
            coverage: CoverageFlags::EMPTY,
            backend_index: LeafMetrics::NO_BACKEND,
        };
        let log = ChunkLog {
            samples: &chunk.samples,
        };
        self.buffers.run_section(
            TRAIN_SECTION,
            -1,
            u64::from(chunk.chunk_id),
            0,
            &log,
            // One calibrated leaf per chunk: there is no eval to memoize.
            None,
            now,
            |slots, _scratch, _inputs| {
                slots.clear();
                slots.push(metrics);
                metrics
            },
        );
        let end = now + held;
        self.running = Some(Running {
            chunk,
            started: now,
            end,
        });
        end
    }
}

impl IterWorker for TrainChunkWorker {
    type Msg = TrainWorkerMsg;
    type Event = TrainWorkerEvent;

    fn id(&self) -> WorkerId {
        self.id
    }

    fn enqueue(&mut self, msg: Self::Msg) {
        let TrainWorkerMsg::Chunk {
            chunk_id,
            groups,
            samples,
        } = msg;
        if let Some(queued) = self.pending.as_ref() {
            panic!(
                "train block {:?} was handed chunk {chunk_id} while chunk {} was still queued; \
                 L6 dispatches to a block only while it is idle",
                self.id, queued.chunk_id,
            );
        }
        self.pending = Some(Chunk {
            chunk_id,
            groups,
            samples,
        });
    }

    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time> {
        if let Some(running) = self.running.as_ref() {
            if running.end <= now {
                let Running {
                    chunk,
                    started,
                    end,
                } = self.running.take().expect("checked");
                events.push(TrainWorkerEvent::ChunkComplete {
                    worker: self.id,
                    chunk_id: chunk.chunk_id,
                    groups: chunk.groups,
                    tokens: chunk.tokens(),
                    started,
                    ended: end,
                });
            }
        }
        // A chunk that lands this tick frees the block for the next one in the
        // same tick — the trainer does not wait a tick to pick work back up.
        if self.running.is_none() && self.pending.is_some() {
            let chunk = self.pending.take().expect("checked");
            return Some(self.start(now, chunk));
        }
        self.running.as_ref().map(|running| running.end)
    }

    /// Chunks, not requests: `queued` is the handed-over chunk not yet started,
    /// `active` the one holding the block. L6 reads this to tell a busy block
    /// from a free one, which is the only question it asks.
    fn status(&self) -> WorkerStatus {
        WorkerStatus {
            queued_requests: u32::from(self.pending.is_some()),
            active_requests: u32::from(self.running.is_some()),
        }
    }
}

/// One training chunk lowered to the `cost_log` row's group section.
///
/// A training step runs the whole sequence through one fused forward+backward,
/// so its shape is a prefill and not a decode: every sample contributes an
/// append of its full length over an empty prefix, and the decode columns stay
/// zero because nothing here is generating.
struct ChunkLog<'a> {
    samples: &'a [u32],
}

impl GroupLogSource for ChunkLog<'_> {
    fn fill_group_log(&self, dst: &mut Vec<GroupInputLog>) {
        let tokens: u32 = self.samples.iter().sum();
        dst.clear();
        dst.push(GroupInputLog {
            batch_tokens: tokens,
            prefill_tokens: tokens,
            decode_request_count: 0,
            decode_kv_total: 0,
            prefill_chunk_pairs: self.samples.iter().map(|len| (0, *len)).collect(),
            decode_query_rows: 0,
            speculative_geometry: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::File;

    use arrow_array::{Float64Array, StringArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use tempfile::tempdir;

    fn cost() -> TrainChunkCost {
        TrainChunkCost {
            tokens_per_s: 10_000.0,
            overhead: Time::from_ms(850.0),
        }
    }

    fn chunk(chunk_id: u32, groups: u16, samples: &[u32]) -> TrainWorkerMsg {
        TrainWorkerMsg::Chunk {
            chunk_id,
            groups,
            samples: samples.to_vec(),
        }
    }

    /// Drive `worker` from `now` on a 1 ms grid until it reports a completion.
    fn run_to_completion(worker: &mut TrainChunkWorker, now: Time) -> (Time, TrainWorkerEvent) {
        let mut at = now;
        for _ in 0..100_000 {
            let mut events = Vec::new();
            worker.tick(at, &mut events);
            if let Some(event) = events.pop() {
                return (at, event);
            }
            at = at + Time::from_ms(1.0);
        }
        panic!("training chunk never completed");
    }

    /// The calibrated cost: 100k tokens at 10k tok/s is 10 s, plus the 850 ms
    /// per-chunk constant. The block is held for the whole of it.
    #[test]
    fn a_chunk_takes_its_tokens_divided_by_the_rate() {
        let mut worker = TrainChunkWorker::new(WorkerId(0), cost(), None, "train");
        worker.enqueue(chunk(0, 2, &[50_000, 50_000]));
        let mut events = Vec::new();
        let wakeup = worker.tick(Time::ZERO, &mut events);
        assert!(events.is_empty(), "the chunk has only just started");
        assert_eq!(wakeup, Some(Time::from_ms(10_850.0)));
        assert_eq!(worker.status().active_requests, 1);

        let (at, event) = run_to_completion(&mut worker, Time::from_ms(1.0));
        assert_eq!(at, Time::from_ms(10_850.0));
        let TrainWorkerEvent::ChunkComplete {
            worker: id,
            chunk_id,
            groups,
            tokens,
            started,
            ended,
        } = event;
        assert_eq!((id, chunk_id, groups, tokens), (WorkerId(0), 0, 2, 100_000));
        assert_eq!((started, ended), (Time::ZERO, Time::from_ms(10_850.0)));
        assert_eq!(worker.status().active_requests, 0);
    }

    /// An idle block asks for no wakeup, so a pool of them costs nothing per
    /// tick while generation still owns every engine.
    #[test]
    fn an_idle_train_worker_asks_for_no_wakeup() {
        let mut worker = TrainChunkWorker::new(WorkerId(1), cost(), None, "train");
        let mut events = Vec::new();
        assert_eq!(worker.tick(Time::from_ms(5.0), &mut events), None);
        assert!(events.is_empty());
        assert_eq!(worker.status().queued_requests, 0);
    }

    /// A chunk landing frees the block in the same tick the next one starts, so
    /// back-to-back chunks leave no gap the trainer would not have.
    #[test]
    fn a_landed_chunk_frees_the_block_for_the_next_one_in_the_same_tick() {
        let mut worker = TrainChunkWorker::new(WorkerId(0), cost(), None, "train");
        worker.enqueue(chunk(0, 1, &[10_000]));
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        let end = Time::from_ms(1_850.0);

        worker.enqueue(chunk(1, 1, &[10_000]));
        events.clear();
        let wakeup = worker.tick(end, &mut events);
        assert_eq!(events.len(), 1, "the first chunk lands");
        assert_eq!(
            wakeup,
            Some(end + Time::from_ms(1_850.0)),
            "and the second starts at the same instant"
        );
    }

    /// The `cost_log` row round-trips: one row per chunk under the `train`
    /// section, carrying the held time and the pool tag that keeps a training
    /// block from colliding with the inference worker of the same id.
    #[test]
    fn a_chunk_writes_one_train_section_cost_row() {
        let dir = tempdir().unwrap();
        let mut worker = TrainChunkWorker::new(
            WorkerId(2),
            cost(),
            Some(dir.path().to_path_buf()),
            "main.train",
        );
        worker.enqueue(chunk(7, 2, &[20_000, 30_000]));
        let mut events = Vec::new();
        worker.tick(Time::from_ms(100.0), &mut events);
        drop(worker); // flush + join the writer thread

        let manifest_path = dir
            .path()
            .join("raw/cost_manifest/worker_main.train_2.json");
        let manifest: CostManifestDoc =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        assert_eq!(manifest.sections.len(), 1);
        assert_eq!(manifest.sections[0].section, "train");
        assert_eq!(manifest.sections[0].manifest.slots.len(), 1);

        let path = dir.path().join("raw/cost_log/worker_main.train_2.parquet");
        let mut reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1);
        let pool = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(pool.value(0), "main.train");
        let section = batch
            .column_by_name("section")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(section.value(0), "train");
        let wall_start = batch
            .column_by_name("wall_start_ms")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(wall_start.value(0), 100.0);
        let total = batch
            .column_by_name("total_time_ms")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        // 50k tokens / 10k tok/s + 850 ms, through the f32 the slot columns hold.
        assert!((total.value(0) - 5_850.0).abs() < 1.0);
    }
}
