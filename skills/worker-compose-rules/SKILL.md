---
name: worker-compose-rules
description: Use when composing, reviewing, or tidying an MLSim iter-wise worker (the `IterWorker` impls under `simulator/src/worker/` — barebone, hp_unified, pd_prefill, pd_decode, disagg_attn, disagg_ffn) for its house conventions — module-doc + reading order, `on_msg_*` message-handler naming, and the naming-quality checks (a name must earn its keep; no misleading quantifiers; name the actual transition; qualify thin wrappers; reserve `on_*` for handlers; keep `ref` anchors). NOT for cost-model/kernel work (that is top-add-kernel / impl-validate-kernel-cache) or a general file review (that is dev-file-design-review).
---

# Worker Compose Rules

Conventions for an MLSim **iter-wise worker** — a struct that implements
`IterWorker` (`simulator/src/worker/iter_worker.rs`: `id` / `enqueue` / `tick` /
`status`) and is driven by `SimpleDpPoolController` one tick at a time. The
canonical exemplar is `simulator/src/worker/disagg_attn.rs`; read it before
composing a new worker and mirror its shape.

This skill is the *worker-specific* layer on top of `dev-file-design-review`. When a
rule here and a rule there conflict, this one wins for worker files. Apply these
when writing a new worker, porting one from `ref/moesim-rs`, or tidying an
existing one.

## 1. Module docstring carries the design narrative

The `//!` block is the worker's contract, not a summary. It states, in order:
what the worker owns and what it explicitly does NOT own (cross-reference the
design decision tags, e.g. `D4`/`D11`/`D16`); the state machine / pipeline it
runs; the accounting invariants (e.g. the two-phase `promised` vs resident-KV
split, and which membership set is the prefill/decode discriminator); and it
**ends with an explicit reading order**. Keep the literal phrasing style of
`disagg_attn.rs`:

> Reading order: types → construction → message handling (the `IterWorker` entry
> points) → the tick / pipeline loop → its helpers, each following the function
> that calls it → tests.

Every non-obvious field gets a one-line doc naming the invariant it participates
in (see the `Slot` / worker-struct fields), not a restatement of its type.

## 2. Reading order is fixed; helpers follow their caller

Lay the file out top-to-bottom so a first read flows along the main path:

1. module docstring
2. types — consts, state enums, the per-unit struct, the worker struct
3. construction — `new(...)` (kept OFF the trait; same signature across workers)
4. message handling — the `IterWorker` impl: `enqueue` dispatch + `tick` + `status`
5. the tick / pipeline loop — `tick_inner` (the fixpoint loop / FSM driver)
6. helpers — **each placed immediately after the first function that calls it**
   (depth-first by call order), main path above supporting detail
7. `#[cfg(test)] mod tests`

Do NOT group helpers by kind (all getters together, etc.). Order by call so the
reader descends through the call tree. If helper `b` is first called by `a`,
`b` comes right after `a`.

## 3. Message handlers are `on_msg_<variant>`, 1:1 with the Msg enum

`enqueue` is a pure dispatch: each `Self::Msg` variant routes to exactly one
handler named `on_msg_<variant_snake_case>`, and nothing else calls those
handlers.

```rust
fn enqueue(&mut self, msg: Self::Msg) {
    match msg {
        AttnWorkerMsg::Admit { req }            => self.on_msg_admit(req),
        AttnWorkerMsg::Release { req }          => self.on_msg_release(req),
        AttnWorkerMsg::ReadyNotification { .. } => self.on_msg_ready_notification(..),
    }
}
```

This reserves the `on_*` prefix for message entry points. **Do not name an
internal lifecycle method `on_*`** — that is why a prefill→decode transition is
`begin_decode`, not the ported ref name `on_decode_start` (which would read as a
handler). A handler with no shared body stays a one-liner; one that shares logic
(e.g. a `Batch` method of the same concept) delegates — but keep the worker
handler and the delegate distinct (see rule 4d).

## 4. Naming-quality checks

A name's job is to save the reader an inference. Run each new or touched
function/field through these:

**a. A name must earn its keep — not "how many callers", but "what concept does
it document".** A single-use helper is justified when its name + doc encode a
concept or invariant a reader would otherwise reconstruct (e.g. `reserved_kv`
carries the *promised-membership = still-prefilling* discriminator). A single-use
helper that is only a mechanical expression with no documented concept (a bare
`chain().map().sum()`) is a candidate to inline into its one caller. Judge the
name, not the call count.

**b. No misleading quantifiers.** Do not use `all_` / `any_` / `count`-shaped
names for what is a single boolean. `all_notified` implied a per-request
aggregation that never existed (the flag is one slot-level bool plus an
empty-slot short-circuit) → `input_ready`. Name the condition the predicate
actually expresses.

**c. Name the actual transition, not a generic verb.** `finalize` ("finalize
*what*?") → `begin_decode`. For a state-machine worker, a lifecycle method names
the phase it moves *into* or the concrete domain event, never a vague
`finalize` / `process` / `handle` / `update`.

**d. Qualify a thin wrapper over another object's same-named method.** The worker
method `is_prefill` just forwarded to `Request::is_prefill`; both read identical
at the call site → rename the wrapper `req_is_prefill` so it says *whose* state
it queries and does not look like the inner call. Same for `slot_*` wrappers over
slot state.

**e. Keep `ref` cross-references as doc anchors when the local name diverges
from the ported name.** Workers are ported faithfully from `ref/moesim-rs`; when
you rename away from the ref symbol, keep a `(ref old_name)` note in the docstring
so the mapping survives — e.g. `input_ready` (ref `all_notified`), `on_msg_admit`
(ref `enqueue_to_pending`), `begin_decode` (ref `on_decode_start` /
`finalize_to_decode`). Rename the **code**, preserve the **trace to ref**.

## 5. Rename hygiene

When you rename, update together: the definition, every call site, and prose
comments that name *that method*. Do NOT touch:
- prose describing the underlying concept, or a differently-named inner method
  the worker delegates to (e.g. `batch.finalize_to_decode` stays when the worker
  method becomes `begin_decode`);
- a same-spelled method on another type (e.g. `Batch::release` is not the
  worker's `on_msg_release`).
Confirm the split with `grep -n` before and after.

## Workflow

1. Read `disagg_attn.rs` (the exemplar) and the target worker fully, plus
   `iter_worker.rs` for the trait surface.
2. Check the module docstring against rule 1; fix the reading-order footer if
   missing or stale.
3. Walk the file top-to-bottom and confirm the rule-2 order; move any helper that
   sits before its caller or is grouped by kind.
4. Confirm `enqueue` is pure dispatch and every handler is `on_msg_*` (rule 3);
   flag any `on_*` internal method.
5. Run every touched name through the rule-4 checks; for each rename keep the ref
   anchor and apply rule-5 hygiene.
6. Validate: `uv run cargo check -p simulator --lib`. Renames/reorders do not
   change the cost model, so no golden update is needed; if you changed behavior,
   run `just test-cpu` and treat a ±1% throughput warning as an unintended
   cost-model change.

## Output

Lead with the highest-value changes. Include:

- Which rules were applied and where (file:line).
- Renames as an old → new table, each with its `ref` anchor preserved.
- `grep` confirmation that no call site or sibling-type method was missed.
- Validation command + result.
