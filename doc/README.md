# VibeSim

VibeSim is a discrete-event simulator that predicts the performance of
machine-learning **serving** workloads.
Its defining choice is that timing is **measured, not modelled from first
principles**: the cost of every GPU kernel is profiled once on real hardware into
a cache, and the simulator predicts a whole deployment's throughput and latency by
**composing those measured kernel timings upward** through a seven-layer cost
model.

Two ideas make that fast and honest:

- **Measured timing (L1 is the only source of numbers).** The Rust core never
  guesses how long a kernel takes — it looks the time up in `profile.db`, which was
  filled by launching the real kernel on a real GPU. Every layer above L1 only
  *composes* those numbers; it never invents a new one.
- **Structure compiled once, numbers streamed per iteration.** A model's cost
  *structure* (which kernels run, how they compose, where a homogeneous layer
  repeats) is stable across iterations; only the leaf numbers change with the batch
  shape. So the structure is compiled once into a **CostTree**, and each simulated
  iteration just evaluates leaves and folds them to one total.

A **deployment** (unified, prefill/decode-disaggregated, attention/FFN-
disaggregated, …) is a typed composition of model arch, worker recipe, and
orchestrator protocol. Its serde tag selects that code path; it is not a mode
flag interpreted throughout the stack.

## The seven layers

Timing flows up; state lives only at the top. Each layer answers exactly one
question and hides the layer below it.

| Layer | Name | One-line role |
|---|---|---|
| **L1** | Kernel / Profiling | Measure one kernel on one GPU; serve the time from a cache. The only layer that produces a number. |
| **L2** | Op | Name one or more kernels into an operation (`attn`, `o_proj`) with a fixed internal composition. |
| **L3** | Worklet | Compose ops into one model-module sync section, deriving the per-rank parallel partition. |
| **L4** | Model arch | Wire worklets into a full model or disaggregated section; compile stable CostTree sections once. |
| **L5** | Worker | Compose KV, admission/selection, and execution under a concrete cadence shell; own mutable lifecycle and clock state. |
| **L6** | Orchestrator | Route a deployment's requests across pools of workers. |
| **L7** | Sim infrastructure | The deployment-agnostic tick loop, trace frontend, logging, and the launcher. |

L1–L4 are **pure cost queries** (no simulation state, parallelism-agnostic below
L4); L5 is where mutable state first appears; L6–L7 are routing and run
infrastructure.

## How this documentation is organised

- **[architecture.md](architecture.md)** — the layer stack in depth (each layer's
  boundary: what it sees and what it must not) and the **module → layer map** that
  tells you which directory implements which layer.
- **[invariants.md](invariants.md)** — the cross-cutting rules every layer upholds,
  in one place.
- **[detailed_design/](detailed_design/)** — one concise document per layer
  (`L1.md` … `L7.md`): its role, boundary, the directory and key types that
  implement it, and its own invariants.
- **[analyzer.md](analyzer.md)** — the post-run analyzer's design contract (the
  Rust-computes / Python-renders split, the report/payload envelope, the flat
  subject registry, applicability/scope, and the speed budget).

Two companion sources sit alongside this folder:

- **Per-module `README.md` files** live next to the code (`simulator/src/*/README.md`,
  `profiling/README.md`, `launcher/README.md`, `analyzer/README.md`). They are the
  finest-grained, code-matching reference; when a README disagrees with these docs,
  the code — and the README next to it — wins.
- **`old-doc/`** is the archived design record (deep cost-math derivations, design
  rationale, discussions, and forward-looking plans). This `doc/` folder is the
  clean, current picture; `old-doc/` is where the history and the math live.
