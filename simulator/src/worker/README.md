# L5 Worker — the per-worker FSM

A **worker** models one serving worker (one GPU's worth of execution). It owns
the request lifecycle: admit queued requests under a KV-memory budget, form a
batch, ask the L4 model how long the batch's iteration costs, advance the clock,
then book-keep tokens and KV until requests finish. It is the first layer that
holds **mutable sim state**; everything below it (L4→L1) is a pure cost query.

This is the practical, code-matching reference; the code is the ground truth.
For deeper design intent see `docs/detailed_design/L5/design.md`.

## What's in the tree

Iter-wise workers currently include `BareboneWorker` (single-group unified),
`HpUnifiedWorker` (one batch per attention-DP shard), and the PD pair
`PdPrefillWorker` / `PdDecodeWorker`. `WorkerConfig` is their construction tier
(KV budget plus hot-path logging controls; `Default` = 80 GB). Shared admission
primitives live beside them. `chunked_prefill` and the AFD attn/ffn workers parse
and are advertised in the schema, but their deployments still bail for now
(`config.rs`).

## The L6-facing surface (event-driven)

The L6 pool drives a worker through three methods. Events are pushed into a
caller-owned sink; `tick` returns the worker's next wakeup so L6 can skip sleeping
workers on the fixed global clock:

| Method | Role |
|---|---|
| `enqueue(WorkerMsg::Request(rid))` | hand the worker an admitted request |
| `tick(now, events) -> Option<Time>` | advance the FSM as far as it can at time `now`, pushing self-tagged events and returning the next wakeup (`None` = quiescent) |
| `status() -> WorkerStatus` | queued + active request counts (for the pool's load view) |

`complete_iter` pushes `WorkerEvent::RequestComplete { worker, req }` into the
caller-provided event sink, rather than returning a `Vec`. The request slab is the shared
`RequestStore` (`SharedRequests`), injected at construction and borrowed
transiently inside each method. `release_request(rid, current_kv)` is the
external cancellation entry point — it cleans the request out of wherever it sits
(queue, promise, or live batch) and frees its KV.

## The tick FSM

State = `WorkerFsmState {Idle, Active}` × the batch cursor `BatchFsmState`
(`cursor: IterCursor {NotStarted, Computing, Done}` + `compute_end`). `tick` is a
state-forwarding loop over three stages:

1. **`form_batch`** (Idle) — Phase A admits one fresh prefill from the queue if
   `KvAdmission::try_admit` passes the KV-memory check; Phase B drains ready
   promises into the iteration's `prefill_admits`. Returns false (stay Idle) when
   nothing is admitted and no decode is in flight.
2. **`start_iter`** (Active/NotStarted) — build the `UnifiedArchInput` for the
   batch, run **one CostTree eval pass** on the L4 model
   (`model.eval_iter(&input, &mut cost_slots, &mut cost_scratch)`), and arm `compute_end = now +
   cost_time` (`agg.m.time_ms` is the clock). The cursor goes NotStarted →
   Computing; the worker waits until `now >= compute_end`, then Computing → Done.
3. **`complete_iter`** (Active/Done) — advance decode tokens, transition admitted
   prefills to decoding, release finished requests' KV, and emit
   `RequestComplete { worker, req }`. Back to Idle to form the next batch.

## Admission & batching primitives (`admission_helpers.rs`)

- **`KvPool`** — the worker's KV-cache memory budget. The worker sizes its own
  pool at construction: `attn_kv_bytes / model.kv_bytes_per_token` (L5 owns the
  division; the arch owns the per-token footprint). `projected_peak` estimates
  peak KV until all live decodes drain: the peak always lands *at* a decode-exit
  step, so it evaluates `KV(t)` only at those steps — all of them for `n < 8`, a
  sampled (front-dense + evenly-spaced) subset otherwise, making it a heuristic
  admission guard rather than an exact bound.
- **`Batch`** — the in-flight request group: this iter's `prefill_admits` plus the
  set of decoding requests (`DecodeReqState`), with projected-peak-KV accounting.
- **`KvAdmission`** (`Strict`) — `try_admit(batch, promised, prompt, decode)`: the
  go/no-go that keeps a batch within its projected peak KV.
- **`LoadBalance`** — which group a request lands in (`Single` in barebone).

## cost_log (optional, per-iteration)

When a `log_dir` is supplied, the worker opens a `CostLogger` against the model's
`cost_log_manifest()`. `start_iter` always fills the reused `cost_slots` buffer
(the eval pass materializes it anyway); when logging is active it also captures
per-leaf typed inputs (`cost_slot_inputs`) and writes one `CostLogEntry` per
iteration (per-slot `slot_time_ms` / `slot_coverage`, the group input section, the
aggregate time/energy). A failed open disables logging with a warning, never
aborts the sim. The worker no longer allocates per-row `Vec`s for those variable
sections: it hands `cost_slots`, `cost_groups`, and `cost_slot_inputs` to
`CostLogger::record`, which appends them into reused chunk-level flat buffers.

## Up / down

- **Below (required):** the L4 model via the `IterwiseUnifiedModel` contract —
  `eval_iter` / `eval_iter_with_inputs` (the per-iter CostTree eval),
  `kv_bytes_per_token`, `cost_log_manifest`. The worker holds it as `Arc<M>`.
- **Above (consumer):** the L6 orchestrator pool (`orchestrator::simple_dp`)
  enqueues requests, ticks due workers, and receives self-tagged events through
  the shared event sink.

## Config selectors (`config.rs`)

Worker selectors are serde tagged enums (`#[serde(tag = "type")]`), the symmetric
sibling of the arch selector, co-located with the workers they pick:
`IterWorkerSel` (`barebone` wired; `chunked_prefill` parses, `build` bails),
`AttnWorkerSel` / `FfnWorkerSel` (the AFD layer-wise contract; config-only today).
`#[derive(ProviderSchema)]` emits each selector's `(tag, params)` rows for the
launcher's `list-params` schema.
